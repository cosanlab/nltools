# Sam Greydanus and Luke Chang 2015
# Some code taken from nilearn searchlight implementation: https://github.com/nilearn/nilearn/blob/master/nilearn/decoding/searchlight.py

import os

import time
import sys
import warnings
from distutils.version import LooseVersion
import random

import cPickle 
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import plotting

import nibabel as nib

import sklearn
from sklearn import neighbors
from sklearn.externals.joblib import Parallel, delayed, cpu_count
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn.base import BaseEstimator
from sklearn import neighbors
from sklearn.svm import SVR

from nilearn import masking
from nilearn.input_data import NiftiMasker

from nltools.analysis import Predict
import glob

########## HELPER FUNCTIONS ############

# # parallelizable searchlight function (called within different cores)
# def parallel_searchlight(X, y, estimator, A, n_jobs=-1,verbose=0):
#     group_iter = GroupIterator(A.shape[0], n_jobs)
#     scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
#         delayed(_group_iter_search_light)(
#             A.rows[list_i],
#             estimator, X, y,
#             thread_id + 1, A.shape[0], verbose)
#         for thread_id, list_i in enumerate(group_iter))
#     return np.concatenate(scores)


# #code that is executed on each of the cores (core searchlight functionality)
# def _group_iter_search_light(list_rows, estimator, X, y,
#                              scoring, thread_id, total, verbose=0):
#     par_scores = np.zeros(len(list_rows))
#     t0 = time.time()
#     for i, row in enumerate(list_rows):
#         kwargs = dict()
#         if not LooseVersion(sklearn.__version__) < LooseVersion('0.15'):
#             kwargs['scoring'] = scoring
#         elif scoring is not None:
#             warnings.warn('Scikit-learn version is too old. '
#                           'scoring argument ignored', stacklevel=2)
#         #import random
#         #par_scores[i] = random.random() #np.mean(cross_val_score(estimator, X[:, row],y, n_jobs=1,**kwargs))
        
        
#         ######## REGRESSION HAPPENS HERE ########
#         #svr = Predict(X[:, row], y, algorithm='svr', cv_dict = {'kfolds':5}, **{'kernel':"linear"})
#         #svr.predict(save_images=False, save_output=False).rmse
        
#         #svr_lin = SVR(kernel='linear', C=1e3)
#         #y_lin = svr_lin.fit(X, y).predict(X)
        
#         import random
#         par_scores[i] = random.random()
        
#         ########
        
        
#         if verbose > 0:
#             # One can't print less than each 10 iterations
#             step = 11 - min(verbose, 10)
#             if (i % step == 0):
#                 # If there is only one job, progress information is fixed
#                 if total == len(list_rows):
#                     crlf = "\r"
#                 else:
#                     crlf = "\n"
#                 percent = float(i) / len(list_rows)
#                 percent = round(percent * 100, 2)
#                 dt = time.time() - t0
#                 # We use a max to avoid a division by zero
#                 remaining = (100. - percent) / max(0.01, percent) * dt
#                 sys.stderr.write(
#                     "Job #%d, processed %d/%d voxels "
#                     "(%0.2f%%, %i seconds remaining)%s"
#                     % (thread_id, i, len(list_rows), percent, remaining, crlf))
#     return par_scores

# # iterator that helps with parallelization process
# class GroupIterator(object):
#     """Group iterator
#     Provides group of features for search_light loop
#     that may be used with Parallel.
#     Parameters
#     ----------
#     n_features : int
#         Total number of features
#     n_jobs : int, optional
#         The number of CPUs to use to do the computation. -1 means
#         'all CPUs'. Defaut is 1
#     """
#     def __init__(self, n_features, n_jobs=1):
#         self.n_features = n_features
#         if n_jobs == -1:
#             n_jobs = cpu_count()
#         self.n_jobs = n_jobs

#     def __iter__(self):
#         split = np.array_split(np.arange(self.n_features), self.n_jobs)
#         for list_i in split:
#             yield list_i
    
########## CLASS DEFINITIONS ############

class Searchlight:
    def __init__(self, brain_mask=None, process_mask=None, radius=4): #no scoring param
        self.resource_folder = os.path.join(os.getcwd(),'resources')
        self.outfolder = os.path.join(os.getcwd(),'outfolder')
        
        if type(brain_mask) is str:
            brain_mask = nib.load(brain_mask)
        elif brain_mask is None:
            brain_mask = nib.load(os.path.join(self.resource_folder,'MNI152_T1_2mm_brain_mask_dil.nii.gz'))
        elif type(brain_mask) is not nib.nifti1.Nifti1Image:
            print(brain_mask)
            print(type(brain_mask))
            raise ValueError("brain_mask is not a nibabel instance")
        self.brain_mask = brain_mask
        
        if type(process_mask) is str:
            process_mask = nib.load(process_mask)
        elif process_mask is None:
            process_mask = nib.load(os.path.join(self.resource_folder,"FSL_RIns_thr0.nii.gz"))
        elif type(brain_mask) is not nib.nifti1.Nifti1Image:
            print(process_mask)
            print(type(process_mask))
            raise ValueError("process_mask is not a nibabel instance")
        self.process_mask = process_mask
            
        self.radius = radius
        self.nifti_masker = NiftiMasker(mask_img=self.brain_mask)
        
    def predict(self, core_i, n_cores, params): #CHANGE NAME
        
        (bdata, A, self.nifti_masker, process_mask_1D, algorithm, cv_dict, output_dir, kwargs) = params
        
        print("getting data")
        if isinstance(bdata, str):
            file_list = glob.glob(bdata + '*.nii.gz')
            bdata = nib.funcs.concat_images(file_list[0:9])
            y = np.array([3, 1, 2, 3, 1, 2, 3, 1, 2]).T
        
        print("making core divs")
        core_divs = [] #a list of lists of indices
        for i in range(0,n_cores):
            a = i*A.shape[0] / n_cores
            b = (i+1)*A.shape[0] / n_cores
            core_divs.append( range(a,b) )
        
        divs = A[core_divs[core_i]].shape[0]
        tot = A.shape[0]
        print("This core will be doing " + str(divs) + " searchlights out of " + str(tot) + " total.")
        
        # clear the text file's contents if there are any
        title  = "out" + str(core_i)
        text_file = open(os.path.join(self.outfolder, title + ".txt"), "w")
        text_file.close()
        
        text_file = open(os.path.join(self.outfolder, "progress.txt"), "w")
        text_file.close()

        print("starting process loop")
        results = []
        for i in range( A[core_divs[core_i]].shape[0] ):

            searchlight = A[core_divs[core_i]][i].toarray() #1D vector
            
            searchlight_mask = self.nifti_masker.inverse_transform( searchlight )

            #apply the Predict method
            svr = Predict(bdata, y, mask = searchlight_mask, algorithm=algorithm, output_dir=output_dir, cv_dict = cv_dict, **kwargs)
            svr.predict(save_plot=False)
            
            print(svr.rmse_all)
            results.append(svr.rmse_all)
            
            title  = "out" + str(core_i)
            text_file = open(os.path.join(self.outfolder,title + ".txt"), "a")
            if i + 1 == A[core_divs[core_i]].shape[0]:
                text_file.write(str(svr.rmse_all))
            else:
                text_file.write(str(svr.rmse_all) + ",")
            text_file.close()
            
        #check progress of all cores. If all cores are finished, run the reassemble helper function
        progress_fn = os.path.join(self.outfolder,"progress.txt")
        cores_finished = ""
        with open(progress_fn, 'r') as f:
            cores_finished = f.readline()
        with open(progress_fn, 'w') as f:
            if (len(cores_finished) > 0):
                f.write( str(int(cores_finished) + 1) )
                if (int(cores_finished) + 2 >= n_cores):
                    f.seek(0)
                    f.truncate()
                    self.reassemble_()
            else:
                f.write( "0" )
            
    
    # helper function which finds the indices of each searchlight and returns a lil file
    def get_coords(self):
        # Compute world coordinates of all in-mask voxels.
        # Return indices as sparse matrix of 0's and 1's
        print("start get coords")
        world_process_mask = self.nifti_masker.fit_transform(self.process_mask)
        world_brain_mask = self.nifti_masker.fit_transform(self.brain_mask)
        
        process_mask_1D = world_brain_mask.copy()
        process_mask_1D[:,:] = 0
        no_overlap = np.where( world_process_mask * world_brain_mask > 0 ) #get the indices where at least one entry is 0
        process_mask_1D[no_overlap] = 1 #delete entries for which there is no overlap
        
        mask, mask_affine = masking._load_mask_img(self.brain_mask)
        mask_coords = np.where(mask != 0)
        mc1 = np.reshape(mask_coords[0], (1, -1))
        mc2 = np.reshape(mask_coords[1], (1, -1))
        mc3 = np.reshape(mask_coords[2], (1, -1))
        mask_coords = np.concatenate((mc1.T,mc2.T, mc3.T), axis = 1)
        
#         print("FULL BRAIN COORDS")
#         print(mask_coords)
#         print(mask_coords.shape)
        
        selected_3D = self.nifti_masker.inverse_transform( process_mask_1D )
        process_mask_coords = np.where(selected_3D.get_data()[:,:,:,0] != 0)
        pmc1 = np.reshape(process_mask_coords[0], (1, -1))
        pmc2 = np.reshape(process_mask_coords[1], (1, -1))
        pmc3 = np.reshape(process_mask_coords[2], (1, -1))
        process_mask_coords = np.concatenate((pmc1.T,pmc2.T, pmc3.T), axis = 1)
        
#         print("PROCESS REGION COORDS")
#         print(process_mask_coords)
#         print(process_mask_coords.shape)
        
        clf = neighbors.NearestNeighbors(radius = self.radius)
        A = clf.fit(mask_coords).radius_neighbors_graph(process_mask_coords)
        del mask_coords, process_mask_coords, selected_3D, no_overlap
        
#         print("~~~~~~~~~~~~~~~~")
#         print(A.shape)
#         print(A[1000].toarray())
        print("There are " + str( sum(sum(A[0].toarray())) ) + " voxels in each searchlight")
        print("finish searchlight")
        return (A.tolil(), self.nifti_masker, process_mask_1D)
    
    @staticmethod
    def run_searchlight_(bdata, brain_mask = None, process_mask = None, radius=4, n_cores = 0):
        
        print("start run searchlight")
        
        os.system("mkdir outfolder")
        
        #n_cores start at 0, so if the input param is 10, there are 11 cores
        sl = Searchlight(brain_mask=brain_mask, process_mask=process_mask, radius=radius)
        
        # parameters for Predict function
        (A, nifti_masker, process_mask_1D) = sl.get_coords()
        print("getting A, nifti_masker, and process_mask_coords")
        algorithm = 'svr'
        cv_dict = None #{'kfolds':5}
        output_dir = os.path.join(os.getcwd(),'outfolder')
        kwargs = {'kernel':"linear"}
        
        print("finished making data")
        
        # save all parameters in a file in the same directory that the code is being executed
        cPickle.dump([bdata, A.tolil(), nifti_masker, process_mask_1D, algorithm, cv_dict, output_dir, kwargs], open("searchlight.pickle", "w"))
        
        print("finished storing data")

        Searchlight.make_inner_python_script_()
        print("wrote inner script)")
        
        #generate BA$H scripts
        for ith_core in range(n_cores):
            Searchlight.make_pbs_scripts_(ith_core, n_cores) # create a script
            os.system("qsub div_script" + str(ith_core) + ".pbs") # run it on a core

        print("wrote div scripts")
        print("finished...")

    @staticmethod        
    def make_pbs_scripts_(ith_core = 0, n_cores = 0):
        title  = "div_script" + str(ith_core)
        text_file = open(os.path.join(os.getcwd(), title + ".pbs"), "w")
        
        text_file.write("#!/bin/bash -l \n\
# declare a name for this job to be my_serial_job \n\
# it is recommended that this name be kept to 16 characters or less \n\
#PBS -N searchlight_pbs \n\
# request the queue (enter the possible names, if omitted, default is the default) \n\
# this job is going to use the default \n\
#PBS -q default \n\
# request 1 node \n\
#PBS -l nodes=1:ppn=1 \n\
# request 0 hours and 15 minutes of wall time \n\
# (Default is 1 hour without this directive) \n\
#PBS -l walltime=00:15:00 \n\
# mail is sent to you when the job starts and when it terminates or aborts \n\
# specify your email address \n\
#PBS -M samuel.j.greydanus.17@dartmouth.edu \n\
# By default, PBS scripts execute in your home directory, not the \n\
# directory from which they were submitted. The following line \n\
# places the job in the directory from which the job was submitted. \n\
cd /ihome/sgreydan/searchlight_simulation \n\
# run the program using the relative path \n\
python inner_searchlight_script.py " + str(ith_core) + " " + str(n_cores) + " \n\
exit 0" )
        text_file.close()

    @staticmethod
    def make_inner_python_script_():
        title  = "inner_searchlight_script.py"
        f = open(os.path.join(os.getcwd(), title), "w")
        f.write("from nltools.searchlight import Searchlight \n\
import cPickle \n\
import os \n\
import sys \n\
pdir = \"" + os.path.join(os.getcwd(),'searchlight.pickle') + "\" \n\
params = cPickle.load( open(pdir) ) \n\
sl = Searchlight() \n\
ith_core = int(sys.argv[1]) \n\
n_cores = int(sys.argv[2]) \n\
sl.predict(ith_core, n_cores, params) ")
        f.close()


    @staticmethod
    def make_email_alert_pbs_():
        title  = "email_alert_pbs.pbs"
        f = open(os.path.join(os.getcwd(), title), "w")
        f.write("#PBS -m ea \n\
#PBS -N email_alert_pbs \n\
#PBS -M samuel.j.greydanus.17@dartmouth.edu \n\
exit 0")
        f.close()

    
    @staticmethod
    def reassemble_(reconstruct_flag = True):
        # if there is already data in the reassembled.txt file, delete it
        outfolder = os.path.join(os.getcwd(),'outfolder')

        #clear data in reassembled.txt
        rs_fn = "reassembled"
        rs_dir = os.path.join(os.getcwd(), rs_fn + '.txt')
        rs = open(rs_dir, 'w')
        rs.seek(0)
        rs.truncate()
        rs.close()

        #get name and location of div file
        div_fn_prefix = "out"
        ith_core = 0
        div_dir = os.path.join(outfolder, div_fn_prefix + str(ith_core) + ".txt")

        #write results from all cores to one text file in a csv format
        while (os.path.isfile(div_dir)):
            with open (div_dir, "r") as div_file:
                data=div_file.read()

                rs = open(rs_dir, "a")
                rs.write(data + ',')
                rs.close()

            command = "rm div_script" + str(ith_core) + ".pbs"
            os.system(command) # delete all the scripts we generated

            command = "rm " + str(div_dir)
            os.system(command) # delete all the smaller text files we generated

            ith_core = ith_core + 1

            div_dir = "outfolder/" + div_fn_prefix + str(ith_core) + ".txt"

        #delete the last comma in the csv file we just generated
        with open(rs_dir, 'rb+') as f:
            f.seek(-1, os.SEEK_END)
            f.truncate()

        print( "Finished reassembly (reassembled " + str(ith_core) + " items)" )

        print("Cleaning up...")
        os.system("rm searchlight_pbs*") # delete all the automatically generated error and output files in the home directory
        os.system("rm *~*") # delete all files auto generated by gedit
        os.system("rm outfolder/progress.text")

        #send user an alert email by executing a blank script with an email alert tag
        Searchlight.make_email_alert_pbs_()
        os.system("qsub email_alert.pbs")
        os.system("rm email_alert_pbs*")
        os.system("rm inner_searchlight_script*")
        os.system("rm email_alert_pbs*")

        #get data from reassembled.txt and convert it to a .nii file
        # if reconstruct_flag:
        #     with open(rs_dir, 'r') as rs:
        #         data = rs.read()
        os.system("rm searchlight.pickle")


