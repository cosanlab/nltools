'''
NeuroLearn pbs_job
==================

Tools to run distributed jobs on Dartmouth Discovery PBS scheduler

'''

__all__ = ['PBS_Job']
__author__ = ["Sam Greydanus","Luke Chang"]
__license__ = "MIT"

import os

import time
import sys
import warnings
from distutils.version import LooseVersion
import random

import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import plotting

import nibabel as nib

import sklearn
from sklearn import neighbors
from sklearn.externals.joblib import Parallel, delayed, cpu_count
from sklearn import svm
# from sklearn.cross_validation import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.svm import SVR

from nilearn import masking
from nilearn.input_data import NiftiMasker

from nltools.utils import get_resource_path
import glob

class PBS_Job:
    def __init__(self, data, parallel_out = None, process_mask=None, radius=4, kwargs=None): #no scoring param
        '''The __init__ function gives the PBS_Job object access to brain data, brain mask,
            and all the other fundamental parameters it needs to run the task. Note: the PBS_Job runs on every
            core in a distributed task and each of those cores gets an identical copy of these class variables'''
        self.data = data

        #set up parallel_out
        if parallel_out is None:
            os.system("mkdir parallel_out")
            self.parallel_out = os.path.join(os.getcwd(),'parallel_out')
        elif type(parallel_out) is str:
            os.system("mkdir " + parallel_out) #make directory if it does not exist
            self.parallel_out = parallel_out
        else:
            print(type(parallel_out))
            raise ValueError("parallel_out should be a string")
        os.system('mkdir ' + os.path.join(self.parallel_out,'core_out') )
        self.core_out = os.path.join(self.parallel_out,'core_out')
        
        #set up process_mask
        if type(process_mask) is str:
            process_mask = nib.load(process_mask)
        elif process_mask is None:
            process_mask = nib.load(os.path.join(get_resource_path(),"FSL_RIns_thr0.nii.gz"))
        elif type(process_mask) is not nib.nifti1.Nifti1Image:
            print(process_mask)
            print(type(process_mask))
            raise ValueError("process_mask is not a nibabel instance")
        self.process_mask = process_mask
        
        #set up other parameters
        self.radius = radius
        self.kwargs = kwargs

    def make_startup_script(self, fn):
        '''When we run the PBS script which starts each of the cores, these cores need to start
            up by running a startup script. This function writes that startup script to a .py file'''
        #clear data in r_all and weights files, if any
        pf = os.path.join(self.core_out, "progress.txt") # progress file
        with open(pf, 'w') as p_file:
            p_file.seek(0)
            p_file.truncate()
            p_file.write("0") #0 cores have finished

        with open(os.path.join(self.parallel_out, fn), "w") as f:
            f.write("from nltools.pbs_job import PBS_Job \n\
import cPickle \n\
import os \n\
import sys \n\
pdir = \"" + os.path.join(self.parallel_out,'pbs_searchlight.pkl') + "\" \n\
parallel_job = cPickle.load( open(pdir) ) \n\
core_i = int(sys.argv[1]) \n\
ncores = int(sys.argv[2]) \n\
parallel_job.run_core(core_i, ncores) ")
      
    def make_pbs_email_alert(self, email):
        ''' Simply sends an email alert when it's run'''
        title  = "email_alert.pbs"
        with open(os.path.join(self.parallel_out, title), "w") as f:
            f.write("#PBS -m ea \n\
#PBS -N email_alert \n\
#PBS -M " + email + " \n\
exit 0")

    def make_pbs_scripts(self, script_name, core_i, ncores, walltime):
        '''Essentially a hacky interface between python and Discovery's job submission
            process. It writes a PBS job submission file with the given parameters.'''
        with open(os.path.join(self.parallel_out, script_name), "w") as f:
            f.write("#!/bin/bash -l \n\
# declare a job name \n\
#PBS -N sl_core_" + str(core_i) + " \n\
# request a queue \n\
#PBS -q default \n\
# request 1 core \n\
#PBS -l nodes=1:ppn=1 \n\
# request wall time (default is 1 hr if not set)\n\
#PBS -l walltime=" + walltime + " \n\
# execute core-level code in same directory as head core \n\
cd " + self.parallel_out + " \n\
python core_startup.py " + str(core_i) + " " + str(ncores) + " \n\
exit 0" )

    def run_core(self, core_i, ncores):
        '''This is the main processing loop. A copy of this runs on each
            of the cores we've requested'''
        tic = time.time()
        self.errf("Started run_core", core_i = core_i, dt = (time.time() - tic))

        core_groups = [] #a list of lists of indices
        for i in range(0,ncores):
            start_i = i*self.A.shape[0] / ncores
            stop_i = (i+1)*self.A.shape[0] / ncores
            core_groups.append( range(start_i, stop_i) )

        runs_per_core = len(core_groups[core_i])
        runs_total = self.A.shape[0]
        self.errf("Started run_core", core_i = core_i, dt=(time.time() - tic))
        self.errf("This core will be doing " + str(runs_per_core) + " runs out of " \
             + str(runs_total) + " total.", core_i=core_i, dt=(time.time() - tic))

        #clear data in r_all and weights files, if any
        rf = os.path.join(self.core_out, "r_all" + str(core_i) + '.txt') #correlation file
        wf = os.path.join(self.core_out, "weights" + str(core_i) + '.txt') # weight file
        with open(rf, 'w') as r_file, open(wf, 'w') as w_file:
            r_file.seek(0), w_file.seek(0)
            r_file.truncate(), w_file.truncate()

        self.errf("Begin main loop", core_i = core_i, dt=(time.time() - tic))
        t0 = time.time()
        for i in range( runs_per_core ):
            tic = time.time()
            searchlight_sphere = self.A[core_groups[core_i][i]][:].toarray() #1D vector
            searchlight_mask = self.data.nifti_masker.inverse_transform( searchlight_sphere )

            #select some data
            data_sphere = self.data.apply_mask(searchlight_mask)
            data_sphere.file_name = os.path.join( self.core_out,'data_core_' + str(core_i) + '_run_' + str(i) )

            #apply the Predict method
            output = data_sphere.predict(algorithm=self.kwargs['algorithm'], \
                cv_dict=self.kwargs['cv_dict'], \
                plot=False, \
                **self.kwargs['predict_kwargs'])
            
            #save r correlation values
            with open(os.path.join(self.core_out, "r_all" + str(core_i) + ".txt"), "a") as f:
                r = output['r_xval']
                if r != r: r=0.0
                if i + 1 == runs_per_core:
                    f.write(str(r)) #if it's the last entry, don't add a comma at the end
                else:
                    f.write(str(r) + ",")

            #save weights
            with open(os.path.join(self.core_out, "weights" + str(core_i) + ".txt"), "a") as f:
                if i + 1 < runs_per_core:
                    l = output['weight_map_xval'].data
                    for j in range(len(l) - 1):
                        f.write(str(l[j]) + ',')
                    f.write( str(l[j]) +  "\n") #if it's the last entry, don't add a comma at the end
                else:
                    l = output['weight_map_xval'].data
                    for j in range(len(l) - 1):
                        f.write(str(l[j]) + ',')
                    f.write( str(l[j])) #if it's the last entry, don't add a comma or a \n at the end

            #periodically update estimate of processing rate
            if i%7 == 0:
                self.estimate_rate(core_i, (time.time() - t0), i + 1, runs_per_core)
                #every so often, clear the file
                if i%21 == 0 and core_i == 0:
                    ratef = os.path.join(self.parallel_out,"rate.txt")
                    with open(ratef, 'w') as f:
                        f.write("") #clear the file
            
        # read the count of the number of cores that have finished
        with open(os.path.join(self.core_out,"progress.txt"), 'r') as f:
            cores_finished = int(f.readline())

        # if all cores are finished, run a clean up method
        # otherwise, increment number of finished cores and terminate process
        with open(os.path.join(self.core_out,"progress.txt"), 'w') as f:
            cores_finished += 1
            f.write( str(cores_finished) + "\n" + str(cores_finished == ncores) )
            if (cores_finished == ncores):
                self.errf("Last core is finished", dt=(time.time() - tic))
                self.clean_up( email_flag = True)

    def errf(self, text, core_i = None, dt = None):
        '''There are many cores simultaneously running the run_core code.
            Call this function if we only want one core to print debug information.
            Appends a time stamp'''
        if core_i is None or core_i == 0:
            with open(os.path.join(self.parallel_out,'errf.txt'), 'a') as f:
                f.write(text + "\n")
                if dt is not None:
                    f.write("       ->Time: " + str(dt) + " seconds\n")

    def get_t_remaining(self, rate, jobs, runs_per_core):
        '''Based on the rates that cores have finished jobs in the past (see estimate_rate() function),
            estimate how much time is left. This is a really rough estimater,
            but it works well in practice'''
        t = int(rate*(runs_per_core-jobs))
        t_day = t / (60*60*24)
        t -= t_day*60*60*24
        t_hr = t / (60*60)
        t -= t_hr*60*60
        t_min = t / (60)
        t -= t_min*60
        t_sec = t
        return str(t_day) + "d" + str(t_hr) + "h" + str(t_min) + "m" + str(t_sec) + "s"

    def estimate_rate(self, core, tdif, jobs, runs_per_core):
        '''Write progress of a given core to a text file,
            then use this file to estimate the rate that cores are finishing'''
        ratef = os.path.join(self.parallel_out,"rate.txt")
        if not os.path.isfile(ratef):
            with open(ratef, 'w') as f:
                f.write("")

        maxrate = ''
        prevtime = ''
        with open(ratef, 'r') as f:
            maxrate = f.readline().strip('\n')
            prevtime = f.readline().strip('\n')
            coreid = f.readline().strip('\n')
            est = f.readline().strip('\n')

        with open(ratef, 'w') as f:
            if (len(maxrate) > 0):
                if (float(maxrate) < tdif/jobs):
                    est = self.get_t_remaining(tdif/jobs, jobs, runs_per_core)
                    f.write(str(tdif/jobs) + "\n" + str(time.time()) + "\nCore " + \
                        str(core) + " is slowest: " + str(tdif/jobs) + " seconds/job\n" + \
                        "This run will finish in " + est + "\n")
                else:
                    f.write(maxrate + "\n" + prevtime + "\n" + coreid + "\n" + est + "\n")
            elif (len(prevtime) == 0):
                est = self.get_t_remaining(tdif/jobs, jobs, runs_per_core)
                f.write(str(tdif/jobs) + "\n" + str(time.time()) + "\nCore " + str(core) + \
                    " is slowest: " + str(tdif/jobs) + " seconds/job\n" + "This run will finish in " \
                    + est + "\n")
        
    # helper function which finds the indices of each searchlight and returns a lil file
    def make_searchlight_masks(self):
        ''' Compute world coordinates of all in-mask voxels.
            Returns a list of masks, one mask for each searchlight. For a whole brain,
            this will generate thousands of masks. Returns masks in lil format
            (efficient for storing sparse matrices)'''
        # Compute world coordinates of all in-mask voxels.
        # Return indices as sparse matrix of 0's and 1's
        print("start get coords")
        world_process_mask = self.data.nifti_masker.fit_transform(self.process_mask)
        world_brain_mask = self.data.nifti_masker.fit_transform(self.data.mask)

        process_mask_1D = world_brain_mask.copy()
        process_mask_1D[:,:] = 0
        no_overlap = np.where( world_process_mask * world_brain_mask > 0 ) #get the indices where at least one entry is 0
        process_mask_1D[no_overlap] = 1 #delete entries for which there is no overlap

        mask, mask_affine = masking._load_mask_img(self.data.mask)
        mask_coords = np.where(mask != 0)
        mc1 = np.reshape(mask_coords[0], (1, -1))
        mc2 = np.reshape(mask_coords[1], (1, -1))
        mc3 = np.reshape(mask_coords[2], (1, -1))
        mask_coords = np.concatenate((mc1.T,mc2.T, mc3.T), axis = 1)

        selected_3D = self.data.nifti_masker.inverse_transform( process_mask_1D )
        process_mask_coords = np.where(selected_3D.get_data()[:,:,:,0] != 0)
        pmc1 = np.reshape(process_mask_coords[0], (1, -1))
        pmc2 = np.reshape(process_mask_coords[1], (1, -1))
        pmc3 = np.reshape(process_mask_coords[2], (1, -1))
        process_mask_coords = np.concatenate((pmc1.T,pmc2.T, pmc3.T), axis = 1)

        clf = neighbors.NearestNeighbors(radius = 3)
        A = clf.fit(mask_coords).radius_neighbors_graph(process_mask_coords)
        del mask_coords, process_mask_coords, selected_3D, no_overlap

        print("Built searchlight masks.")
        print("Each searchlight has on the order of " + str( sum(sum(A[0].toarray())) ) + " voxels")
        self.A = A.tolil()
        self.process_mask_1D = process_mask_1D
            
    def clean_up(self, email_flag = True):
        '''Once all the cores have finished running, the last core will call this function.
            It deletes temporary text files and merges the output files from all the cores into 
            one single file. Finally, it parses this file and transforms it into a 3D nifti file
            which can be opened with MRIcroGL or any other .nii viewing software'''
        #clear data in reassembled and weights files, if any

                #clear data in r_all and weights files, if any
        rf = os.path.join(self.parallel_out, "correlations.txt") # correlation file (for combined nodes)
        wf = os.path.join(self.parallel_out, "weights.txt") # weight file (for combined nodes)
        with open(rf, 'w') as r_combo, open(wf, 'w') as w_combo:
            r_combo.seek(0), w_combo.seek(0)
            r_combo.truncate(), w_combo.truncate()

        #get name and location of each core's correlation, weights file
        core_i = 0
        r_prefix, w_prefix, = "r_all", "weights"
        r_core_data = os.path.join(self.core_out, r_prefix + str(core_i) + ".txt")
        w_core_data = os.path.join(self.core_out, w_prefix + str(core_i) + ".txt")

        data_was_merged = False
        print( "Merging data to combined files:" )
        print( "   -->" + r_core_data )
        print( "   -->" + w_core_data )
        #write results from all cores to one text file in a csv format
        while (os.path.isfile(r_core_data) and os.path.isfile(w_core_data)):
            with open (r_core_data, "r") as r_core, open (w_core_data, "r") as w_core:
                rdata = r_core.read() ; weights = w_core.read()

                with open(rf, "a") as r_combo, open(wf, "a") as w_combo:
                    r_combo.write(rdata + ','), w_combo.write(weights + '\n')

            core_i += 1
            r_core_data = os.path.join(self.core_out, r_prefix + str(core_i) + ".txt")
            w_core_data = os.path.join(self.core_out, w_prefix + str(core_i) + ".txt")

            data_was_merged = True

        #remove the last comma in the csv file we just generated
        if (data_was_merged):
            with open(rf, 'rb+') as r_combo, open(wf, 'rb+') as w_combo:
                r_combo.seek(-1, os.SEEK_END), w_combo.seek(-1, os.SEEK_END)
                r_combo.truncate(), w_combo.truncate()

            #get data from combo file and convert it to a .nii file
            self.reconstruct(rf)
        else:
            print("ERROR: data not merged correctly (in directory: " + self.parallel_out + ")")

        print( "Finished reassembly (reassembled " + str(core_i) + " items)" )

        #send user an alert email  alert by executing a blank script with an email alert tag
        if email_flag:
            os.system("qsub email_alert.pbs")

        print("Cleaning up...")
        os.system("rm " + os.path.join(self.parallel_out, "sl_core_*"))
        os.system("rm " + os.path.join(self.parallel_out, "rate*"))
        os.system("rm " + os.path.join(self.parallel_out, "*core_pbs_script_*"))
        os.system("rm " + os.path.join(self.parallel_out, "core_startup.py*"))

    def reconstruct(self, rf):
        '''This is a helper for the clean_up function. It reads a single file with all the 
                correlation coefficients, ocnverts them to a numpy array and then to a 3D .nii file'''
            #open the reassembled correlation data and build a python list of float type numbers
        with open(rf, 'r') as r_combo:
            rdata = np.fromstring(r_combo.read(), dtype=float, sep=',')

            #find coords of all values in process mask that are equal to 1
            coords = np.where(self.process_mask_1D == 1)[1]

            #in all the locations that are equal to 1, store the correlation data
            #   (coords and data should have same length)
            self.process_mask_1D[0][coords] = rdata

            #transform rdata to 3D "correlation heat map" (nifti format)
            rdata_3D = self.data.nifti_masker.inverse_transform( self.process_mask_1D )
            rdata_3D.to_filename(os.path.join(self.parallel_out,'rdata_3D.nii.gz')) #save nifti image