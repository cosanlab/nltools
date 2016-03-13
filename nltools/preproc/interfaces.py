'''
    nltools Nipype Interfaces
    =====================================
    Classes for various nipype interfaces

    Author: Luke Chang
    License: MIT

'''

# import matplotlib
# matplotlib.use('Agg')
# from nipype.interfaces import spm
# import nipype.interfaces.io as nio           # Data i/o
# import nipype.interfaces.utility as util     # utility
# from nipype.pipeline.engine import Node, Workflow
# from nipype.interfaces.base import BaseInterface, TraitedSpec, File, traits
# import nipype.algorithms.rapidart as ra      # artifact detection
# from nipype.interfaces.nipy.preprocess import ComputeMask
# import nipype.interfaces.matlab as mlab
# from IPython.display import Image
# import glob

import pandas as pd
import numpy as np
import numpy as np
import pylab as plt
import os
import nibabel as nib
from nipype.interfaces.base import isdefined, BaseInterface, TraitedSpec, File, traits
from nilearn import plotting, datasets, image
import nibabel as nib

# # Specify various inputs files for pipeline
# spm_path = '/Users/lukechang/Documents/MATLAB/spm8/'
# data_dir='/Users/lukechang/Dropbox/PTSD/Data/Imaging'
# canonical_file = spm_path + 'canonical/single_subj_T1.nii'
# t1_template_file = spm_path + 'templates/T1.nii'

# # Set the way matlab should be called
# mlab.MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")
# mlab.MatlabCommand.set_default_paths(spm_path)



class Plot_Coregistration_Montage_InputSpec(TraitedSpec):	
	wra_img = File(exists=True, mandatory=True) 
	canonical_img = File(exists=True, mandatory=True)
	title = traits.Str("Normalized Functional Check", usedefault=True)

class Plot_Coregistration_Montage_OutputSpec(TraitedSpec):
	plot = File(exists=True)

class Plot_Coregistration_Montage(BaseInterface):
	# This function creates an axial montage of the average normalized functional data
	# and overlays outline of the normalized single subject overlay.  
	# Could probably pick a better overlay later.

	input_spec = Plot_Coregistration_Montage_InputSpec
	output_spec = Plot_Coregistration_Montage_OutputSpec

	def _run_interface(self, runtime):
		import matplotlib
		matplotlib.use('Agg')
		import pylab as plt

		wra_img = nib.load(self.inputs.wra_img)
		canonical_img = nib.load(self.inputs.canonical_img)
		title = self.inputs.title
		mean_wraimg = image.mean_img(wra_img)

		if title != "":
			filename = title.replace(" ", "_")+".pdf"
		else:
			filename = "plot.pdf"

		fig = plotting.plot_anat(mean_wraimg, title="wrafunc & canonical single subject", cut_coords=range(-40, 40, 10), display_mode='z')
		fig.add_edges(canonical_img)     
		fig.savefig(filename)
		fig.close()

		self._plot = filename

		runtime.returncode=0
		return runtime

	def _list_outputs(self):
		outputs = self._outputs().get()
		outputs["plot"] = os.path.abspath(self._plot)
		return outputs
    
class PlotRealignmentParametersInputSpec(TraitedSpec):
	realignment_parameters = File(exists=True, mandatory=True)
	outlier_files = File(exists=True)
	title = traits.Str("Realignment parameters", usedefault=True)
	dpi = traits.Int(300, usedefault = True)
    
class PlotRealignmentParametersOutputSpec(TraitedSpec):
	plot = File(exists=True)

class PlotRealignmentParameters(BaseInterface):
	#This function is adapted from Chris Gorgolewski and creates a figure of the realignment parameters

	input_spec = PlotRealignmentParametersInputSpec
	output_spec = PlotRealignmentParametersOutputSpec
    
	def _run_interface(self, runtime):
		import matplotlib
		matplotlib.use('Agg')
		import pylab as plt
		realignment_parameters = np.loadtxt(self.inputs.realignment_parameters)
		title = self.inputs.title
        
		F = plt.figure(figsize=(8.3,11.7))	
		F.text(0.5, 0.96, self.inputs.title, horizontalalignment='center')
		ax1 = plt.subplot2grid((2,2),(0,0), colspan=2)
		handles =ax1.plot(realignment_parameters[:,0:3])
		ax1.legend(handles, ["x translation", "y translation", "z translation"], loc=0)
		ax1.set_xlabel("image #")
		ax1.set_ylabel("mm")
		ax1.set_xlim((0,realignment_parameters.shape[0]-1))
		ax1.set_ylim(bottom = realignment_parameters[:,0:3].min(), top = realignment_parameters[:,0:3].max())
        
		ax2 = plt.subplot2grid((2,2),(1,0), colspan=2)
		handles= ax2.plot(realignment_parameters[:,3:6]*180.0/np.pi)
		ax2.legend(handles, ["pitch", "roll", "yaw"], loc=0)
		ax2.set_xlabel("image #")
		ax2.set_ylabel("degrees")
		ax2.set_xlim((0,realignment_parameters.shape[0]-1))
		ax2.set_ylim(bottom=(realignment_parameters[:,3:6]*180.0/np.pi).min(), top= (realignment_parameters[:,3:6]*180.0/np.pi).max())
        
		if isdefined(self.inputs.outlier_files):
			try:
				outliers = np.loadtxt(self.inputs.outlier_files)
			except IOError as e:
				if e.args[0] == "End-of-file reached before encountering data.":
					pass
				else:
					raise
			else:
				if outliers.size > 0:
					ax1.vlines(outliers, ax1.get_ylim()[0], ax1.get_ylim()[1])
					ax2.vlines(outliers, ax2.get_ylim()[0], ax2.get_ylim()[1])
        
		if title != "":
			filename = title.replace(" ", "_")+".pdf"
		else:
			filename = "plot.pdf"

		F.savefig(filename,papertype="a4",dpi=self.inputs.dpi)
		plt.clf()
		plt.close()
		del F

		self._plot = filename
        
		runtime.returncode=0
		return runtime

	def _list_outputs(self):
		outputs = self._outputs().get()
		outputs["plot"] = os.path.abspath(self._plot)
		return outputs

class Create_Covariates_InputSpec(TraitedSpec):	
	realignment_parameters = File(exists=True, mandatory=True) 
	spike_id = File(exists=True, mandatory=True)

class Create_Covariates_OutputSpec(TraitedSpec):
	covariates = File(exists=True)

class Create_Covariates(BaseInterface):
	input_spec = Create_Covariates_InputSpec
	output_spec = Create_Covariates_OutputSpec

	def _run_interface(self, runtime):
		ra = pd.read_table(self.inputs.realignment_parameters, header=None, sep=r"\s*", names=['ra' + str(x) for x in range(1,7)])
		spike = pd.read_table(self.inputs.spike_id, header=None,names=['Spikes'])

		ra = ra-ra.mean() #mean center
		ra[['rasq' + str(x) for x in range(1,7)]] = ra**2 #add squared
		ra[['radiff' + str(x) for x in range(1,7)]] = pd.DataFrame(ra[ra.columns[0:6]].diff()) #derivative
		ra[['radiffsq' + str(x) for x in range(1,7)]] = pd.DataFrame(ra[ra.columns[0:6]].diff())**2 #derivatives squared

		#build spike regressors
		for i,loc in enumerate(spike['Spikes']):
			ra['spike' + str(i+1)] = 0
			ra['spike' + str(i+1)].iloc[loc] = 1

		filename = 'covariates.csv'
		ra.to_csv(filename, index=False) #write out to file
		self._covariates = filename

		runtime.returncode=0
		return runtime

	def _list_outputs(self):
		outputs = self._outputs().get()
		outputs["covariates"] = os.path.abspath(self._covariates)
		return outputs

class MB_Distortion_Correction_InputSpec(TraitedSpec):	
	ap_file = File(exists=True, mandatory=True) 
	pa_file = File(exists=True, mandatory=True) 
	encodings_file = File(exists=True, mandatory=True)
	direction = traits.Str("AP", usedefault=True)

class MB_Distortion_Correction_OutputSpec(TraitedSpec):
	distortion_correction = File(exists=True)

class MB_Distortion_Correction(BaseInterface):
	input_spec = MB_Distortion_Correction_InputSpec
	output_spec = MB_Distortion_Correction_OutputSpec

	from nipype.interfaces.fsl import Merge
	merger = Merge()
	merger.inputs.dimension = 't'
	merger.inputs.output_type = 'NIFTI_GZ'
	merger.inputs.merged_file = 'dist_corr_merged.nii.gz'
	if direction is "AP":
		merger.inputs.in_files = [ap_file, pa_file]
	elif direction is "PA":
		merger.inputs.in_files = [pa_file, ap_file]
	else:
		raise ValueError("direction variable should be 'AP' or 'PA'.")

	from nipype.interfaces.fsl import topup
	topup = TOPUP()
	topup.inputs.in_file = 'dist_corr_merged.nii.gz'
	topup.inputs.encoding_file = encodings_file
	topup.inputs.output_type = "NIFTI_GZ"
	topup.inputs.config = 'b02b0.cnf'
	topup.inputs.out_base = 'topup_distortion_correction_' + direction


