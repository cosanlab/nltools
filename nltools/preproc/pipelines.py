
'''
    nltools Nipype Pipelines
    =====================================
    Various nipype pipelines

    Author: Luke Chang
    License: MIT

'''

import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.base import BaseInterface, TraitedSpec, File, traits
import nipype.algorithms.rapidart as ra      # artifact detection
from nipype.interfaces import spm
from nipype.interfaces.nipy.preprocess import ComputeMask
import nipype.interfaces.matlab as mlab
import os
import nibabel as nib
from IPython.display import Image
import glob

from nltools.utils import get_n_slices, get_ta, get_slice_order, get_vox_dims

def create_spm_preproc_func_pipeline(data_dir=None, subject_id=None, task_list=None):

	###############################
	## Set up Nodes
	###############################

	ds = Node(nio.DataGrabber(infields=['subject_id', 'task_id'], outfields=['func', 'struc']),name='datasource')
	ds.inputs.base_directory = os.path.abspath(data_dir + '/' + subject_id)
	ds.inputs.template = '*'
	ds.inputs.sort_filelist = True
	ds.inputs.template_args = {'func': [['task_id']], 'struc':[]}
	ds.inputs.field_template = {'func': 'Functional/Raw/%s/func.nii','struc': 'Structural/SPGR/spgr.nii'}
	ds.inputs.subject_id = subject_id
	ds.inputs.task_id = task_list
	ds.iterables = ('task_id',task_list)
	# ds.run().outputs #show datafiles

	# #Setup Data Sinker for writing output files
	# datasink = Node(nio.DataSink(), name='sinker')
	# datasink.inputs.base_directory = '/path/to/output'
	# workflow.connect(realigner, 'realignment_parameters', datasink, 'motion.@par')
	# datasink.inputs.substitutions = [('_variable', 'variable'),('file_subject_', '')]

	#Get Timing Acquisition for slice timing
	tr = 2
	ta = Node(interface=util.Function(input_names=['tr', 'n_slices'], output_names=['ta'],  function = get_ta), name="ta")
	ta.inputs.tr=tr

	#Slice Timing: sequential ascending 
	slice_timing = Node(interface=spm.SliceTiming(), name="slice_timing")
	slice_timing.inputs.time_repetition = tr
	slice_timing.inputs.ref_slice = 1

	#Realignment - 6 parameters - realign to first image of very first series.
	realign = Node(interface=spm.Realign(), name="realign")
	realign.inputs.register_to_mean = True

	#Plot Realignment
	plot_realign = Node(interface=PlotRealignmentParameters(), name="plot_realign")

	#Artifact Detection
	art = Node(interface=ra.ArtifactDetect(), name="art")
	art.inputs.use_differences      = [True,False]
	art.inputs.use_norm             = True
	art.inputs.norm_threshold       = 1
	art.inputs.zintensity_threshold = 3
	art.inputs.mask_type            = 'file'
	art.inputs.parameter_source     = 'SPM'

	#Coregister - 12 parameters, cost function = 'nmi', fwhm 7, interpolate, don't mask
	#anatomical to functional mean across all available data.
	coregister = Node(interface=spm.Coregister(), name="coregister")
	coregister.inputs.jobtype = 'estimate'

	# Segment structural, gray/white/csf,mni, 
	segment = Node(interface=spm.Segment(), name="segment")
	segment.inputs.save_bias_corrected = True

	#Normalize - structural to MNI - then apply this to the coregistered functionals
	normalize = Node(interface=spm.Normalize(), name = "normalize")
	normalize.inputs.template = os.path.abspath(t1_template_file)

	#Plot normalization Check
	plot_normalization_check = Node(interface=Plot_Coregistration_Montage(), name="plot_normalization_check")
	plot_normalization_check.inputs.canonical_img = canonical_file

	#Create Mask
	compute_mask = Node(interface=ComputeMask(), name="compute_mask")
	#remove lower 5% of histogram of mean image
	compute_mask.inputs.m = .05

	#Smooth
	#implicit masking (.im) = 0, dtype = 0
	smooth = Node(interface=spm.Smooth(), name = "smooth")
	fwhmlist = [0,5,8]
	smooth.iterables = ('fwhm',fwhmlist)

	#Create Covariate matrix
	make_covariates = Node(interface=Create_Covariates(), name="make_covariates")

	###############################
	## Create Pipeline
	###############################

	Preprocessed = Workflow(name="Preprocessed")
	Preprocessed.base_dir = os.path.abspath(data_dir + '/' + subject_id + '/Functional')

	Preprocessed.connect([(ds, ta, [(('func', get_n_slices), "n_slices")]),
						(ta, slice_timing, [("ta", "time_acquisition")]),
						(ds, slice_timing, [('func', 'in_files'),
											(('func', get_n_slices), "num_slices"),
											(('func', get_slice_order), "slice_order"),
											]),
						(slice_timing, realign, [('timecorrected_files', 'in_files')]),
						(realign, compute_mask, [('mean_image','mean_volume')]),
						(realign,coregister, [('mean_image', 'target')]),
						(ds,coregister, [('struc', 'source')]),
						(coregister,segment, [('coregistered_source', 'data')]),
						(segment, normalize, [('transformation_mat','parameter_file'),
											('bias_corrected_image', 'source'),]),
						(realign,normalize, [('realigned_files', 'apply_to_files'),
											(('realigned_files', get_vox_dims), 'write_voxel_sizes')]),
						(normalize, smooth, [('normalized_files', 'in_files')]),
						(compute_mask,art,[('brain_mask','mask_file')]),
						(realign,art,[('realignment_parameters','realignment_parameters')]),
						(realign,art,[('realigned_files','realigned_files')]),
						(realign,plot_realign, [('realignment_parameters', 'realignment_parameters')]),
						(normalize, plot_normalization_check, [('normalized_files', 'wra_img')]),
						(realign, make_covariates, [('realignment_parameters', 'realignment_parameters')]),
						(art, make_covariates, [('outlier_files', 'spike_id')]),
						])
	return Preprocessed

