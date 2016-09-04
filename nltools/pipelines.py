
'''
    nltools Nipype Pipelines
    ========================
    Various nipype pipelines

'''

__all__ = ['create_spm_preproc_func_pipeline','Couple_Preproc_Pipeline']
__author__ = ["Luke Chang"]
__license__ = "MIT"

import nipype.interfaces.io as nio 
import nipype.interfaces.utility as util
from nipype.pipeline.engine import Node, Workflow
from nipype.interfaces.base import BaseInterface, TraitedSpec, File, traits
import nipype.algorithms.rapidart as ra
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

def Couple_Preproc_Pipeline(base_dir=None, output_dir=None, subject_id=None, spm_path=None):
    """ Create a preprocessing workflow for the Couples Conflict Study using nipype

    Args:
        base_dir: path to data folder where raw subject folder is located
        output_dir: path to where key output files should be saved
        subject_id: subject_id (str)
        spm_path: path to spm folder

    Returns:
        workflow: a nipype workflow that can be run
        
    """
    
    from nipype.interfaces.dcm2nii import Dcm2nii
    from nipype.interfaces.fsl import Merge, TOPUP, ApplyTOPUP
    import nipype.interfaces.io as nio
    import nipype.interfaces.utility as util
    from nipype.interfaces.utility import Merge as Merge_List
    from nipype.pipeline.engine import Node, Workflow
    from nipype.interfaces.fsl.maths import UnaryMaths
    from nipype.interfaces.nipy.preprocess import Trim
    from nipype.algorithms.rapidart import ArtifactDetect 
    from nipype.interfaces import spm
    from nipype.interfaces.spm import Normalize12
    from nipype.algorithms.misc import Gunzip
    from nipype.interfaces.nipy.preprocess import ComputeMask
    import nipype.interfaces.matlab as mlab
    from nltools.utils import get_resource_path, get_vox_dims, get_n_volumes
    from nltools.interfaces import Plot_Coregistration_Montage, PlotRealignmentParameters, Create_Covariates
    import os
    import glob

    ########################################
    ## Setup Paths and Nodes
    ########################################

    # Specify Paths
    canonical_file = os.path.join(spm_path,'canonical','single_subj_T1.nii')
    template_file = os.path.join(spm_path,'tpm','TPM.nii')

    # Set the way matlab should be called
    mlab.MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")
    mlab.MatlabCommand.set_default_paths(spm_path)
    
    # Get File Names for different types of scans.  Parse into separate processing streams
    datasource = Node(interface=nio.DataGrabber(infields=['subject_id'], outfields=['struct', 'ap', 'pa']),name='datasource')
    datasource.inputs.base_directory = base_dir
    datasource.inputs.template = '*'
    datasource.inputs.field_template = {'struct':'%s/Study*/t1w_32ch_mpr_08mm*',
                                        'ap':'%s/Study*/distortion_corr_32ch_ap*',
                                        'pa':'%s/Study*/distortion_corr_32ch_pa*'}
    datasource.inputs.template_args = {'struct':[['subject_id']],'ap':[['subject_id']],'pa':[['subject_id']]}
    datasource.inputs.subject_id = subject_id
    datasource.inputs.sort_filelist=True

    # iterate over functional scans to define paths
    scan_file_list = glob.glob(os.path.join(base_dir,subject_id,'Study*','*'))
    func_list = [s for s in scan_file_list if "romcon_ap_32ch_mb8" in s]
    func_list = [s for s in func_list if "SBRef" not in s] # Exclude sbref for now.
    func_source = Node(interface=util.IdentityInterface(fields=['scan']),name="func_source")
    func_source.iterables = ('scan', func_list)

    # Create Separate Converter Nodes for each different type of file. (dist corr scans need to be done before functional)
    ap_dcm2nii = Node(interface = Dcm2nii(),name='ap_dcm2nii')
    ap_dcm2nii.inputs.gzip_output = True
    ap_dcm2nii.inputs.output_dir = '.'
    ap_dcm2nii.inputs.date_in_filename = False
    
    pa_dcm2nii = Node(interface = Dcm2nii(),name='pa_dcm2nii')
    pa_dcm2nii.inputs.gzip_output = True
    pa_dcm2nii.inputs.output_dir = '.'
    pa_dcm2nii.inputs.date_in_filename = False

    f_dcm2nii = Node(interface = Dcm2nii(),name='f_dcm2nii')
    f_dcm2nii.inputs.gzip_output = True
    f_dcm2nii.inputs.output_dir = '.'
    f_dcm2nii.inputs.date_in_filename = False

    s_dcm2nii = Node(interface = Dcm2nii(),name='s_dcm2nii')
    s_dcm2nii.inputs.gzip_output = True
    s_dcm2nii.inputs.output_dir = '.'
    s_dcm2nii.inputs.date_in_filename = False
    
    ########################################
    ## Setup Nodes for distortion correction
    ########################################
    
    # merge output files into list
    merge_to_file_list = Node(interface=Merge_List(2), infields=['in1','in2'], name='merge_to_file_list')

    # fsl merge AP + PA files (depends on direction)
    merger = Node(interface=Merge(dimension = 't'),name='merger')
    merger.inputs.output_type = 'NIFTI_GZ'

    # use topup to create distortion correction map
    topup = Node(interface=TOPUP(), name='topup')
    topup.inputs.encoding_file = os.path.join(get_resource_path(),'epi_params_APPA_MB8.txt')
    topup.inputs.output_type = "NIFTI_GZ"
    topup.inputs.config = 'b02b0.cnf'

    # apply topup to all functional images
    apply_topup = Node(interface = ApplyTOPUP(), name='apply_topup')
    apply_topup.inputs.in_index = [1]
    apply_topup.inputs.encoding_file = os.path.join(get_resource_path(),'epi_params_APPA_MB8.txt')
    apply_topup.inputs.output_type = "NIFTI_GZ"
    apply_topup.inputs.method = 'jac'
    apply_topup.inputs.interp = 'spline'

    # Clear out Zeros from spline interpolation using absolute value.
    abs_maths = Node(interface=UnaryMaths(), name='abs_maths')
    abs_maths.inputs.operation = 'abs'

    ########################################
    ## Preprocessing
    ########################################

    # Trim - remove first 10 TRs
    n_vols = 10
    trim = Node(interface = Trim(), name='trim')
    trim.inputs.begin_index=n_vols

    #Realignment - 6 parameters - realign to first image of very first series.
    realign = Node(interface=spm.Realign(), name="realign")
    realign.inputs.register_to_mean = True

    #Coregister - 12 parameters
    coregister = Node(interface=spm.Coregister(), name="coregister")
    coregister.inputs.jobtype = 'estwrite'

    #Plot Realignment
    plot_realign = Node(interface=PlotRealignmentParameters(), name="plot_realign")

    #Artifact Detection
    art = Node(interface=ArtifactDetect(), name="art")
    art.inputs.use_differences      = [True,False]
    art.inputs.use_norm             = True
    art.inputs.norm_threshold       = 1
    art.inputs.zintensity_threshold = 3
    art.inputs.mask_type            = 'file'
    art.inputs.parameter_source     = 'SPM'

    # Gunzip - unzip the functional and structural images
    gunzip_struc = Node(Gunzip(), name="gunzip_struc")
    gunzip_func = Node(Gunzip(), name="gunzip_func")

    # Normalize - normalizes functional and structural images to the MNI template
    normalize = Node(interface=Normalize12(jobtype='estwrite',tpm=template_file),
                     name="normalize")

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
    smooth.inputs.fwhm=6

    #Create Covariate matrix
    make_cov = Node(interface=Create_Covariates(), name="make_cov")

    # Create a datasink to clean up output files
    datasink = Node(interface=nio.DataSink(), name='datasink')
    datasink.inputs.base_directory = output_dir
    datasink.inputs.container = subject_id
                                       
    ########################################
    # Create Workflow
    ########################################

    workflow = Workflow(name = 'Preprocessed')
    workflow.base_dir = os.path.join(base_dir,subject_id)
    workflow.connect([(datasource, ap_dcm2nii,[('ap','source_dir')]),
                        (datasource, pa_dcm2nii,[('pa','source_dir')]),
                        (datasource, s_dcm2nii,[('struct','source_dir')]),
                        (func_source, f_dcm2nii,[('scan','source_dir')]),
                        (ap_dcm2nii, merge_to_file_list,[('converted_files','in1')]),
                        (pa_dcm2nii, merge_to_file_list,[('converted_files','in2')]),
                        (merge_to_file_list, merger,[('out','in_files')]),
                        (merger, topup,[('merged_file','in_file')]),
                        (topup, apply_topup,[('out_fieldcoef','in_topup_fieldcoef'),
                                            ('out_movpar','in_topup_movpar')]),                  
                        (f_dcm2nii, trim,[('converted_files','in_file')]),
                        (trim, apply_topup,[('out_file','in_files')]),
                        (apply_topup, abs_maths,[('out_corrected','in_file')]),
                        (abs_maths, gunzip_func, [('out_file', 'in_file')]),
                        (gunzip_func, realign, [('out_file', 'in_files')]),
                        (s_dcm2nii, gunzip_struc,[('converted_files','in_file')]),
                        (gunzip_struc,coregister, [('out_file', 'source')]),
                        (coregister, normalize,[('coregistered_source','image_to_align')]),
                        (realign,coregister, [('mean_image', 'target'),
                                              ('realigned_files', 'apply_to_files')]),
                        (realign,normalize, [(('mean_image', get_vox_dims), 'write_voxel_sizes')]),              
                        (coregister,normalize, [('coregistered_files', 'apply_to_files')]),
                        (normalize, smooth, [('normalized_files', 'in_files')]),
                        (realign, compute_mask, [('mean_image','mean_volume')]),
                        (compute_mask,art,[('brain_mask','mask_file')]),
                        (realign,art,[('realignment_parameters','realignment_parameters'),
                                      ('realigned_files','realigned_files')]),
                        (realign,plot_realign, [('realignment_parameters', 'realignment_parameters')]),
                        (normalize, plot_normalization_check, [('normalized_files', 'wra_img')]),
                        (realign, make_cov, [('realignment_parameters', 'realignment_parameters')]),
                        (art, make_cov, [('outlier_files', 'spike_id')]),
                        (normalize, datasink, [('normalized_files', 'structural.@normalize')]),
                        (coregister, datasink, [('coregistered_source', 'structural.@struct')]),
                        (topup, datasink, [('out_fieldcoef', 'distortion.@fieldcoef')]),
                        (topup, datasink, [('out_movpar', 'distortion.@movpar')]),
                        (smooth, datasink, [('smoothed_files', 'functional.@smooth')]),
                        (plot_realign, datasink, [('plot', 'functional.@plot_realign')]),
                        (plot_normalization_check, datasink, [('plot', 'functional.@plot_normalization')]),
                        (make_cov, datasink, [('covariates', 'functional.@covariates')])])
    return workflow

def TV_Preproc_Pipeline(base_dir=None, output_dir=None, subject_id=None, spm_path=None):
    """ Create a preprocessing workflow for the Couples Conflict Study using nipype

    Args:
        base_dir: path to data folder where raw subject folder is located
        output_dir: path to where key output files should be saved
        subject_id: subject_id (str)
        spm_path: path to spm folder

    Returns:
        workflow: a nipype workflow that can be run
        
    """
    
    import nipype.interfaces.io as nio
    import nipype.interfaces.utility as util
    from nipype.interfaces.utility import Merge as Merge_List
    from nipype.pipeline.engine import Node, Workflow
    from nipype.interfaces.fsl.maths import UnaryMaths
    from nipype.interfaces.nipy.preprocess import Trim
    from nipype.algorithms.rapidart import ArtifactDetect 
    from nipype.interfaces import spm
    from nipype.interfaces.spm import Normalize12
    from nipype.algorithms.misc import Gunzip
    from nipype.interfaces.nipy.preprocess import ComputeMask
    import nipype.interfaces.matlab as mlab
    from nltools.utils import get_resource_path, get_vox_dims, get_n_volumes
    from nltools.interfaces import Plot_Coregistration_Montage, PlotRealignmentParameters, Create_Covariates, Plot_Quality_Control
    import os
    import glob

    ########################################
    ## Setup Paths and Nodes
    ########################################

    # Specify Paths
    canonical_file = os.path.join(spm_path,'canonical','single_subj_T1.nii')
    template_file = os.path.join(spm_path,'tpm','TPM.nii')

    # Set the way matlab should be called
    mlab.MatlabCommand.set_default_matlab_cmd("matlab -nodesktop -nosplash")
    mlab.MatlabCommand.set_default_paths(spm_path)
    
    # Get File Names for different types of scans.  Parse into separate processing streams
    datasource = Node(interface=nio.DataGrabber(infields=['subject_id'], outfields=[
                'struct', 'func']),name='datasource')
    datasource.inputs.base_directory = base_dir
    datasource.inputs.template = '*'
    datasource.inputs.field_template = {'struct':'%s/T1.nii.gz',
                                        'func':'%s/*ep*.nii.gz'}
    datasource.inputs.template_args = {'struct':[['subject_id']],
                                       'func':[['subject_id']]}
    datasource.inputs.subject_id = subject_id
    datasource.inputs.sort_filelist=True
   
    # iterate over functional scans to define paths
    func_source = Node(interface=util.IdentityInterface(fields=['scan']),name="func_source")
    func_source.iterables = ('scan', glob.glob(os.path.join(base_dir,subject_id,'*ep*nii.gz')))
    

    ########################################
    ## Preprocessing
    ########################################

    # Trim - remove first 5 TRs
    n_vols = 5
    trim = Node(interface = Trim(), name='trim')
    trim.inputs.begin_index=n_vols

    #Realignment - 6 parameters - realign to first image of very first series.
    realign = Node(interface=spm.Realign(), name="realign")
    realign.inputs.register_to_mean = True

    #Coregister - 12 parameters
    coregister = Node(interface=spm.Coregister(), name="coregister")
    coregister.inputs.jobtype = 'estwrite'

    #Plot Realignment
    plot_realign = Node(interface=PlotRealignmentParameters(), name="plot_realign")

    #Artifact Detection
    art = Node(interface=ArtifactDetect(), name="art")
    art.inputs.use_differences      = [True,False]
    art.inputs.use_norm             = True
    art.inputs.norm_threshold       = 1
    art.inputs.zintensity_threshold = 3
    art.inputs.mask_type            = 'file'
    art.inputs.parameter_source     = 'SPM'

    # Gunzip - unzip the functional and structural images
    gunzip_struc = Node(Gunzip(), name="gunzip_struc")
    gunzip_func = Node(Gunzip(), name="gunzip_func")

    # Normalize - normalizes functional and structural images to the MNI template
    normalize = Node(interface=Normalize12(jobtype='estwrite',tpm=template_file),
                     name="normalize")

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
    smooth.inputs.fwhm=6

    #Create Covariate matrix
    make_cov = Node(interface=Create_Covariates(), name="make_cov")

    #Plot Quality Control Check
    quality_control = Node(interface=Plot_Quality_Control(), name='quality_control')

    # Create a datasink to clean up output files
    datasink = Node(interface=nio.DataSink(), name='datasink')
    datasink.inputs.base_directory = output_dir
    datasink.inputs.container = subject_id
                                       
    ########################################
    # Create Workflow
    ########################################

    workflow = Workflow(name = 'Preprocessed')
    workflow.base_dir = os.path.join(base_dir,subject_id)
    workflow.connect([(datasource, gunzip_struc,[('struct','in_file')]),
                        (func_source, trim,[('scan','in_file')]),                
                        (trim, gunzip_func,[('out_file','in_file')]),
                        (gunzip_func, realign, [('out_file', 'in_files')]),
                        (realign, quality_control, [('realigned_files', 'dat_img')]),
                        (gunzip_struc,coregister, [('out_file', 'source')]),
                        (coregister, normalize,[('coregistered_source','image_to_align')]),
                        (realign,coregister, [('mean_image', 'target'),
                                              ('realigned_files', 'apply_to_files')]),
                        (realign,normalize, [(('mean_image', get_vox_dims), 'write_voxel_sizes')]),              
                        (coregister,normalize, [('coregistered_files', 'apply_to_files')]),
                        (normalize, smooth, [('normalized_files', 'in_files')]),
                        (realign, compute_mask, [('mean_image','mean_volume')]),
                        (compute_mask,art,[('brain_mask','mask_file')]),
                        (realign,art,[('realignment_parameters','realignment_parameters'),
                                      ('realigned_files','realigned_files')]),
                        (realign,plot_realign, [('realignment_parameters', 'realignment_parameters')]),
                        (normalize, plot_normalization_check, [('normalized_files', 'wra_img')]),
                        (realign, make_cov, [('realignment_parameters', 'realignment_parameters')]),
                        (art, make_cov, [('outlier_files', 'spike_id')]),
                        (normalize, datasink, [('normalized_files', 'structural.@normalize')]),
                        (coregister, datasink, [('coregistered_source', 'structural.@struct')]),
                        (smooth, datasink, [('smoothed_files', 'functional.@smooth')]),
                        (plot_realign, datasink, [('plot', 'functional.@plot_realign')]),
                        (plot_normalization_check, datasink, [('plot', 'functional.@plot_normalization')]),
                        (make_cov, datasink, [('covariates', 'functional.@covariates')]),
                        (quality_control, datasink, [('plot', 'functional.@quality_control')])
                     ])
    return workflow

   