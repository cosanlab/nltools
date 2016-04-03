# '''
#     NeuroLearn Searchlight Tools
#     ============================
    
#     Tools to perform searchlight analyses. Some code taken from nilearn searchlight 
#     implementation: https://github.com/nilearn/nilearn/blob/master/nilearn/decoding/searchlight.py

# '''

# __all__ = ['Searchlight']
# __author__ = ["Sam Greydanus","Luke Chang"]
# __license__ = "MIT"

# import os

# import time
# import sys
# import warnings
# from distutils.version import LooseVersion
# import random

# import cPickle 
# import numpy as np
# import matplotlib.pyplot as plt
# from nilearn import datasets
# from nilearn import plotting

# import nibabel as nib

# import sklearn
# from sklearn import neighbors
# from sklearn.externals.joblib import Parallel, delayed, cpu_count
# from sklearn import svm
# from sklearn.cross_validation import cross_val_score
# from sklearn.base import BaseEstimator
# from sklearn import neighbors
# from sklearn.svm import SVR

# from nilearn import masking
# from nilearn.input_data import NiftiMasker

# from nltools.analysis import Predict
# from nltools.utils import get_resource_path
# import glob
    
# ########## CLASS DEFINITIONS ############

# class Searchlight:
#     def __init__(self, brain_mask=None, process_mask=None, radius=4, searchlight_out = None): #no scoring param
#         if searchlight_out is None:
#             self.searchlight_out = os.path.join(self.searchlight_out,'searchlight')
#         else:
#             self.searchlight_out = searchlight_out
        
#         if type(brain_mask) is str:
#             brain_mask = nib.load(brain_mask)
#         elif brain_mask is None:
#             brain_mask = nib.load(os.path.join(get_resource_path(),'MNI152_T1_2mm_brain_mask_dil.nii.gz'))
#         elif type(brain_mask) is not nib.nifti1.Nifti1Image:
#             print(brain_mask)
#             print(type(brain_mask))
#             raise ValueError("brain_mask is not a nibabel instance")
#         self.brain_mask = brain_mask
        
#         if type(process_mask) is str:
#             process_mask = nib.load(process_mask)
#         elif process_mask is None:
#             process_mask = nib.load(os.path.join(get_resource_path(),'FSL_RIns_thr0.nii.gz'))
#         elif type(brain_mask) is not nib.nifti1.Nifti1Image:
#             print(process_mask)
#             print(type(process_mask))
#             raise ValueError("process_mask is not a nibabel instance")
#         self.process_mask = process_mask
            
#         self.radius = radius
#         self.nifti_masker = NiftiMasker(mask_img=self.brain_mask)

#     @staticmethod
#     def errf(text, ith_core):
#         if ith_core == 0:
#             f = open(os.path.join(self.searchlight_out,'errf.txt'), 'a')
#             f.write(text + "\n")
#             f.close()

#     @staticmethod
#     def t_est(rate, jobs, divs):
#         t = int(rate*(divs-jobs))
#         t_day = t / (60*60*24)
#         t -= t_day*60*60*24
#         t_hr = t / (60*60)
#         t -= t_hr*60*60
#         t_min = t / (60)
#         t -= t_min*60
#         t_sec = t
#         return str(t_day) + "d" + str(t_hr) + "h" + str(t_min) + "m" + str(t_sec) + "s"

#     @staticmethod
#     def write_predict_rate_(core, tdif, jobs, divs):
#         ratef = os.path.join(self.searchlight_out,"rate.txt")

#         if not os.path.isfile(ratef):
#             with open(ratef, 'w') as f:
#                 f.write("")

#         maxrate = ''
#         prevtime = ''
#         with open(ratef, 'r') as f:
#             maxrate = f.readline().strip('\n')
#             prevtime = f.readline().strip('\n')
#             coreid = f.readline().strip('\n')
#             est = f.readline().strip('\n')

#         with open(ratef, 'w') as f:
#             if (len(maxrate) > 0):
#                 if (float(maxrate) < tdif/jobs):
#                     est = Searchlight.t_est(tdif/jobs, jobs, divs)
#                     f.write(str(tdif/jobs) + "\n" + str(time.time()) + "\nCore " + str(core) + " is slowest: " + str(tdif/jobs) + " seconds/job\n" \
#                         + "This run will finish in " + est + "\n")
#                 else:
#                     f.write(maxrate + "\n" + prevtime + "\n" + coreid + "\n" + est + "\n")
#             elif (len(prevtime) == 0):
#                 est = Searchlight.t_est(tdif/jobs, jobs, divs)
#                 f.write(str(tdif/jobs) + "\n" + str(time.time()) + "\nCore " + str(core) + " is slowest: " + str(tdif/jobs) + " seconds/job\n" \
#                         + "This run will finish in " + est + "\n")
        
#     def predict(self, core_i, n_cores, params): #CHANGE NAME
#         tic = time.time()

#         (predict_params, A, self.nifti_masker, self.process_mask, process_mask_1D) = params
#         (bdata, y, algorithm, cv_dict, searchlight_out, kwargs) = predict_params
        
#         if isinstance(bdata, str):
#             file_list = glob.glob(bdata + "*.nii.gz")
#             bdata = nib.funcs.concat_images(file_list)
        
#         Searchlight.errf("Finished reading data. Start making core divs: " + str((time.time() - tic)) + " seconds", core_i)

#         core_divs = [] #a list of lists of indices
#         for i in range(0,n_cores):
#             a = i*A.shape[0] / n_cores
#             b = (i+1)*A.shape[0] / n_cores
#             core_divs.append( range(a,b) )

#         divs = len(core_divs[core_i])
#         tot = A.shape[0]
#         Searchlight.errf("This core will be doing " + str(divs) + " searchlights out of " + str(tot) + " total.", core_i)
#         Searchlight.errf("Time: " + str((time.time() - tic)) + " seconds", core_i)

#         #clear data in r_all and weights files, if any
#         rs_dir = os.path.join(self.searchlight_out, "r_all" + str(core_i) + '.txt')
#         w_dir = os.path.join(self.searchlight_out, "weights" + str(core_i) + '.txt')
#         with open(rs_dir, 'w') as rs, open(w_dir, 'w') as w:
#             rs.seek(0), w.seek(0)
#             rs.truncate(), w.truncate()
        
#         text_file = open(os.path.join(self.searchlight_out, "progress.txt"), "w")
#         text_file.close()

#         t0 = time.time()
#         for i in xrange( divs ):

#             tic = time.time()
#             searchlight = A[core_divs[core_i][i]][:].toarray() #1D vector
            
#             searchlight_mask = self.nifti_masker.inverse_transform( searchlight )

#             #apply the Predict method
#             svr = Predict(bdata, y, mask = searchlight_mask, algorithm=algorithm, searchlight_out=searchlight_out, cv_dict = cv_dict, **kwargs)
#             # Searchlight.errf("      After initializing Predict: " + str((time.time() - tic)) + " seconds", core_i)
#             svr.predict(save_plot=False)
#             # Searchlight.errf("      After running Predict: " + str((time.time() - tic)) + " seconds\n", core_i)
            
#             #save r correlation values
#             title  = "r_all" + str(core_i)
#             text_file = open(os.path.join(self.searchlight_out,title + ".txt"), "a")
#             r = svr.r_xval
#             if r != r: r=0.0
#             if i + 1 == divs:
#                 text_file.write(str(r)) #if it's the last entry, don't add a comma at the end
#             else:
#                 text_file.write(str(r) + ",")
#             text_file.close()

#             #save weights
#             title  = "weights" + str(core_i)
#             text_file = open(os.path.join(self.searchlight_out,title + ".txt"), "a")
#             if i + 1 != divs:
#                 l = svr.predictor.coef_.squeeze()
#                 for j in xrange(len(l) - 1):
#                     text_file.write(str(l[j]) + ',')
#                 text_file.write( str(l[j]) +  "\n") #if it's the last entry, don't add a comma at the end
#             else:
#                 l = svr.predictor.coef_.squeeze()
#                 for j in xrange(len(l) - 1):
#                     text_file.write(str(l[j]) + ',')
#                 text_file.write( str(l[j])) #if it's the last entry, don't add a comma or a \n at the end
#             text_file.close()

#             #periodically update estimate of processing rate
#             if i%7 == 0:
#                 Searchlight.write_predict_rate_(core_i, (time.time() - t0), i + 1, divs)
#                 if i%21 == 0 and core_i == 0:
#                     ratef = os.path.join(self.searchlight_out,"rate.txt")
#                     with open(ratef, 'w') as f:
#                         f.write("")
            
#         #check progress of all cores. If all cores are finished, run the reassemble helper function
#         progress_fn = os.path.join(self.searchlight_out,"progress.txt")
#         cores_finished = ""
#         with open(progress_fn, 'r') as f:
#             cores_finished = f.readline()
#         with open(progress_fn, 'w') as f:
#             if (len(cores_finished) > 0):
#                 f.write( str(int(cores_finished) + 1) )
#                 if (int(cores_finished) + 2 >= n_cores):
#                     f.seek(0)
#                     f.truncate()
#                     self.reassemble_(reconstruct_flag = True, email_flag = True)
#             else:
#                 f.write( "0" )
            
    
#     # helper function which finds the indices of each searchlight and returns a lil file
#     def get_coords(self):
#         # Compute world coordinates of all in-mask voxels.
#         # Return indices as sparse matrix of 0's and 1's
#         print("start get coords")
#         world_process_mask = self.nifti_masker.fit_transform(self.process_mask)
#         world_brain_mask = self.nifti_masker.fit_transform(self.brain_mask)
        
#         process_mask_1D = world_brain_mask.copy()
#         process_mask_1D[:,:] = 0
#         no_overlap = np.where( world_process_mask * world_brain_mask > 0 ) #get the indices where at least one entry is 0
#         process_mask_1D[no_overlap] = 1 #delete entries for which there is no overlap
        
#         mask, mask_affine = masking._load_mask_img(self.brain_mask)
#         mask_coords = np.where(mask != 0)
#         mc1 = np.reshape(mask_coords[0], (1, -1))
#         mc2 = np.reshape(mask_coords[1], (1, -1))
#         mc3 = np.reshape(mask_coords[2], (1, -1))
#         mask_coords = np.concatenate((mc1.T,mc2.T, mc3.T), axis = 1)
        
#         selected_3D = self.nifti_masker.inverse_transform( process_mask_1D )
#         process_mask_coords = np.where(selected_3D.get_data()[:,:,:,0] != 0)
#         pmc1 = np.reshape(process_mask_coords[0], (1, -1))
#         pmc2 = np.reshape(process_mask_coords[1], (1, -1))
#         pmc3 = np.reshape(process_mask_coords[2], (1, -1))
#         process_mask_coords = np.concatenate((pmc1.T,pmc2.T, pmc3.T), axis = 1)
        
#         clf = neighbors.NearestNeighbors(radius = self.radius)
#         A = clf.fit(mask_coords).radius_neighbors_graph(process_mask_coords)
#         del mask_coords, process_mask_coords, selected_3D, no_overlap
        
#         print("There are " + str( sum(sum(A[0].toarray())) ) + " voxels in each searchlight")
#         print("finish searchlight")
#         return (A.tolil(), self.nifti_masker, process_mask_1D)
    
#     @staticmethod
#     def run_searchlight_(bdata, y, algorithm='svr', cv_dict=None, searchlight_out=None, kwargs=None, n_cores=1, radius=3, brain_mask=None, process_mask=None, walltime='07:30:00'):
        
#         print("start run searchlight")
        
#         os.system("mkdir outfolder")
        
#         #n_cores start at 0, so if the input param is 10, there are 11 cores
#         searchlight_out = os.path.join(self.searchlight_out,'outfolder')

#         sl = Searchlight(brain_mask=brain_mask, process_mask=process_mask, radius=radius, searchlight_out = searchlight_out)
        
#         # parameters for Predict function
#         (A, nifti_masker, process_mask_1D) = sl.get_coords()
#         print("got A, nifti_masker, and process_mask_1D")

#         y = np.array(y)
#         predict_params = [bdata, y, algorithm, cv_dict, searchlight_out, kwargs]
        
#         # save all parameters in a file in the same directory that the code is being executed
#         cPickle.dump([predict_params, A, nifti_masker, sl.process_mask, process_mask_1D], open("searchlight.pickle", "w"))
#         del predict_params, A, nifti_masker, process_mask_1D, sl
        
#         print("finished storing data")

#         Searchlight.make_inner_python_script_()
#         print("wrote inner script)")
        
#         #generate shell scripts
#         for ith_core in range(n_cores):
#             Searchlight.make_pbs_scripts_(ith_core, n_cores, walltime) # create a script
#             os.system("qsub div_script" + str(ith_core) + ".pbs") # run it on a core

#         print("wrote div scripts")
#         print("finished...")

#     @staticmethod        
#     def make_pbs_scripts_(ith_core, n_cores, walltime):
#         title  = "div_script" + str(ith_core)
#         text_file = open(os.path.join(self.searchlight_out, title + ".pbs"), "w")
        
#         text_file.write("#!/bin/bash -l \n\
# # declare a name for this job to be my_serial_job \n\
# # it is recommended that this name be kept to 16 characters or less \n\
# #PBS -N searchlight_pbs \n\
# # request the queue (enter the possible names, if omitted, default is the default) \n\
# # this job is going to use the default \n\
# #PBS -q default \n\
# # request 1 node \n\
# #PBS -l nodes=1:ppn=1 \n\
# # request 0 hours and 15 minutes of wall time \n\
# # (Default is 1 hour without this directive) \n\
# #PBS -l walltime="+walltime+" \n\
# # mail is sent to you when the job starts and when it terminates or aborts \n\
# # specify your email address \n\
# #PBS -M samuel.j.greydanus.17@dartmouth.edu \n\
# # By default, PBS scripts execute in your home directory, not the \n\
# # directory from which they were submitted. The following line \n\
# # places the job in the directory from which the job was submitted. \n\
# cd " + self.searchlight_out + " \n\
# # run the program using the relative path \n\
# python inner_searchlight_script.py " + str(ith_core) + " " + str(n_cores) + " \n\
# exit 0" )
#         text_file.close()

#     @staticmethod
#     def make_inner_python_script_():
#         title  = "inner_searchlight_script.py"
#         f = open(os.path.join(self.searchlight_out, title), "w")
#         f.write("from nltools.searchlight import Searchlight \n\
# import cPickle \n\
# import os \n\
# import sys \n\
# pdir = \"" + os.path.join(self.searchlight_out,'searchlight.pickle') + "\" \n\
# params = cPickle.load( open(pdir) ) \n\
# sl = Searchlight() \n\
# ith_core = int(sys.argv[1]) \n\
# n_cores = int(sys.argv[2]) \n\
# sl.predict(ith_core, n_cores, params) ")
#         f.close()


#     @staticmethod
#     def make_email_alert_pbs_():
#         title  = "email_alert_pbs.pbs"
#         f = open(os.path.join(self.searchlight_out, title), "w")
#         f.write("#PBS -m ea \n\
# #PBS -N email_alert_pbs \n\
# #PBS -M samuel.j.greydanus.17@dartmouth.edu \n\
# exit 0")
#         f.close()

    
#     @staticmethod
#     def reassemble_(reconstruct_flag = True, email_flag = True):
#         # if there is already data in the reassembled.txt file, delete it
#         searchlight_out = os.path.join(self.searchlight_out,'outfolder')

#         #clear data in reassembled and weights files, if any
#         rs_fn = "correlations"
#         rs_all = os.path.join(self.searchlight_out, rs_fn + '.txt')
#         w_fn = "weights"
#         w_all = os.path.join(self.searchlight_out, w_fn + '.txt')
#         with open(rs_all, 'w') as rs, open(w_all, 'w') as w:
#             rs.seek(0), w.seek(0)
#             rs.truncate(), w.truncate()

#         #get name and location of each core's correlation, weights file
#         ith_core = 0
#         rdiv_name, wdiv_name, = "r_all", "weights"
#         rdiv_dir = os.path.join(searchlight_out, rdiv_name + str(ith_core) + ".txt")
#         wdiv_dir = os.path.join(searchlight_out, wdiv_name + str(ith_core) + ".txt")

#         success = False
#         #write results from all cores to one text file in a csv format
#         while (os.path.isfile(rdiv_dir) and os.path.isfile(wdiv_dir)):
#             with open (rdiv_dir, "r") as rdiv, open (wdiv_dir, "r") as wdiv:
#                 data = rdiv.read()
#                 weights = wdiv.read()

#                 with open(rs_all, "a") as rs, open(w_all, "a") as w:
#                     rs.write(data + ','), w.write(weights + '\n')


#             command = "rm div_script" + str(ith_core) + ".pbs"
#             os.system(command) # delete all the scripts we generated

#             ith_core = ith_core + 1
#             rdiv_dir = os.path.join(searchlight_out, rdiv_name + str(ith_core) + ".txt")
#             wdiv_dir = os.path.join(searchlight_out, wdiv_name + str(ith_core) + ".txt")

#             success = True

#         #delete the last comma in the csv file we just generated
#         if (success):
#             with open(rs_all, 'rb+') as f1, open(w_all, 'rb+') as f2:
#                 f1.seek(-1, os.SEEK_END), f2.seek(-1, os.SEEK_END)
#                 f1.truncate(), f2.truncate()

#         print( "Finished reassembly (reassembled " + str(ith_core) + " items)" )

#         #send user an alert email by executing a blank script with an email alert tag
#         if email_flag:
#             Searchlight.make_email_alert_pbs_()
#             os.system("qsub email_alert_pbs.pbs")

#         #get data from reassembled.txt and convert it to a .nii file
#         pdir = os.path.join(self.searchlight_out,'searchlight.pickle')
#         if (reconstruct_flag and os.path.isfile(pdir) and success):
#             #get location of searchlight pickle and retrieve its contents
#             (predict_params, A, nifti_masker, _, process_mask_1D) = cPickle.load( open(pdir) )

#             #open the reassembled correlation data and build a python list of float type numbers
#             with open(rs_all, 'r') as rs:
#                 data = np.fromstring(rs.read(), dtype=float, sep=',')

#             coords = np.where(process_mask_1D == 1)[1] #find coords of all values in process mask that are equal to 1

#             #in all the locations that are equal to 1, store the correlation data (coords and data should have same length)
#             process_mask_1D[0][coords] = data

#             data_3D = nifti_masker.inverse_transform( process_mask_1D ) #transform scores to 3D nifti image
#             data_3D.to_filename(os.path.join(self.searchlight_out,'data_3D.nii.gz')) #save nifti image
#             # os.system("rm correlations.txt")
#         elif reconstruct_flag:
#             print("ERROR: File 'searchlight.pickle' does not exist or 'reassemble.txt' is empty (in directory: " + self.searchlight_out + ")")

#         # os.system("rm searchlight.pickle")
#         print("Cleaning up...")
#         os.system("rm email_alert_pbs.e*")
#         os.system("rm email_alert_pbs.o*")
#         os.system("rm *searchlight_* *div* *errf* rate.txt errf.txt")
#         os.system("rm inner_searchlight_script*")


