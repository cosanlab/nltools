[![Build Status](https://api.travis-ci.org/ljchang/neurolearn.png)](https://travis-ci.org/ljchang/neurolearn/)

neurolearn
==========
Python toolbox for analyzing neuroimaging data.  It is based off of Tor Wager's object oriented matlab <a href=http://wagerlab.colorado.edu/tools>canlab core tools</a> and relies heavily on <a href = http://nilearn.github.io>nilearn</a> and <a href=http://scikit-learn.org/stable/index.html>scikit learn</a>

<h3>Current Tools</h3>
<ul>
<li>Predict: apply various classification and prediction algorithms to 4D dataset</li>
<li>apply_mask: apply 3D weight map to 4D dataset</li>
<li>Roc: perform ROC analysis</li>
</ul>

<h3>Installation</h3>
<ol>
<li>Clone github repository</li>
<li>python setup.py install</li>
</ol>

<h3>Documentation</h3>
<p>
Current Documentation can be found at <a href=http://neurolearn.readthedocs.org/en/latest/>readthedocs</a>.  Please see the ipython notebook examples for walkthroughs of how to use most of the toolbox.
<br><br>
Here is a <a href=https://github.com/ljchang/neurolearn/blob/master/scripts/NLTools_Brain_Data_Class_Tutorial.ipynb>notebook</a> with a detailed overview of how to use the main Brain_Data class.  We also have a <a href=https://github.com/ljchang/neurolearn/blob/master/scripts/Chang_ML_fMRI_Tutorial.ipynb>notebook</a> containing other analysis methods such as prediction and ROI curves (note it is now recommended to use the prediction Brain_Data method).
</p>

### Preprocessing

Here is an example preprocessing pipeline for multiband data.  It uses [nipype](http://nipy.org/nipype/) and tools from [SPM12](http://www.fil.ion.ucl.ac.uk/spm/software/spm12/) and [FSL](http://fsl.fmrib.ox.ac.uk/).  Make sure that fsl, matlab, dcm2nii are on your unix environment path.  It might be helpful to create a symbolic link somewhere common like /usr/local/bin.  This pipeline can be run on a cluster see [nipype workflow documentaiton](http://nipy.org/nipype/users/plugins.html).  The nipype folder is quite large due to matlab's need for unzipped .nii files.  It can be deleted if space is an issue.

 - Uses Chris Rorden's [dcm2nii](http://www.mccauslandcenter.sc.edu/mricro/mricron/dcm2nii.html) to convert dcm to nii
 - Uses Nipy's Trim to remove the first 10 volumes (i.e., disdaqs)
 - Uses FSL's [topup](http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/TOPUP) to perform distortion correction.  Default is AP (need to switch order of concatentation if PA is needed)
 - Uses SPM12 realignment to mean
 - Uses SPM12 to coregister functional to structural
 - Uses SPM12 new nonlinear normalization routine
 - Uses SPM12 smoothing with 6mm fwhm as default
 - Uses [Artifact Detection Toolbox](http://www.nitrc.org/projects/artifact_detect/) to detect scanner spikes.
 - Uses Nipype Datasink to write out key files to new output directory under subject name
 - Will create a quick montage to check normalization
 - Will output a plot of realignment parameters
 - Will output a covariate csv file with 24 parameter centered motion parameters, their squares, and the 12 derivatives (6 motion + 6 squared motion).

Here is an example script.

```
from nltools.pipelines import Couple_Preproc_Pipeline
import os

base_dir = '/Users/lukechang/Dropbox/Couple_Conflict/Data/Scanner'
spm_path = '/Users/lukechang/Resources/spm12/'
output_dir = '/Users/lukechang/Dropbox/Couple_Conflict/Data/Imaging'

# Get Subject ID
subject_list = os.listdir(os.path.join(base_dir))
subject_id = subject_list[1]

#Run Pipeline
wf = Couple_Preproc_Pipeline(base_dir=base_dir, output_dir=output_dir, subject_id=subject_id, spm_path=spm_path)
# wf.run('MultiProc', plugin_args={'n_procs': 8}) # This command runs the pipeline in parallel (using 8 cores)
wf.run()
```

