[![Build Status](https://api.travis-ci.org/ljchang/neurolearn.png)](https://travis-ci.org/ljchang/neurolearn/)

# neurolearn
Python toolbox for analyzing neuroimaging data.  It is based off of Tor Wager's object oriented matlab [canlab core tools](http://wagerlab.colorado.edu/tools) and relies heavily on [nilearn](http://nilearn.github.io) and [scikit learn](http://scikit-learn.org/stable/index.html)

### Current Tools
- *data.Brain_Data*: Class to work with 4D imaging data in Python
- *data.Brain_Data.predict*: Multivariate Prediction 
- *data.Brain_Data.similarity*: Calculate spatial similarity with another image
- *data.Brain_Data.distance*: Calculate spatial distance of a group of images
- *data.Brain_Data.regress*: Univariate Regression 
- *data.Brain_Data.ttest*: Univariate One Sample t-test 
- *analysis.Roc*: perform ROC analysis
- *pipelines.Couple_Preproc_Pipeline*: preprocessing pipeline for multiband data
- *simulator.Simulator*: Class for simulating multivariate data
- *mask.create_sphere*: Create spherical masks

### Installation
1. Method 1
  
   ```
   git clone git+https://github.com/ljchang/neurolearn
   ```

2. Method 2

   ```
   git clone https://github.com/ljchang/neurolearn
   python setup.py install
   ```

### Documentation
Current Documentation can be found at [readthedocs](http://neurolearn.readthedocs.org/en/latest).  Please see the ipython notebook examples for walkthroughs of how to use most of the toolbox.

Here is a [jupyter notebook](https://github.com/ljchang/neurolearn/blob/master/scripts/NLTools_Brain_Data_Class_Tutorial.ipynb) with a detailed overview of how to use the main *Brain_Data* class.  We also have a [notebook](https://github.com/ljchang/neurolearn/blob/master/scripts/Chang_ML_fMRI_Tutorial.ipynb) containing other analysis methods such as prediction and ROI curves (note it is now recommended to use the prediction Brain_Data method).

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

``` python
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
wf.write_graph(dotfilename=os.path.join(output_dir,'Workflow_Pipeline.dot'),format='png')
wf.run()
```
![pipeline](https://github.com/ljchang/neurolearn/blob/master/docs/img/Workflow_Pipeline.dot.png)



