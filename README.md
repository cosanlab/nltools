[![Build Status](https://api.travis-ci.org/ljchang/nltools.png)](https://travis-ci.org/ljchang/nltools/)
[![Coverage Status](https://coveralls.io/repos/github/ljchang/neurolearn/badge.svg?branch=master)](https://coveralls.io/github/ljchang/neurolearn?branch=master)
# neurolearn
Python toolbox for analyzing neuroimaging data.  It is based off of Tor Wager's object oriented matlab [canlab core tools](http://wagerlab.colorado.edu/tools) and relies heavily on [nilearn](http://nilearn.github.io) and [scikit learn](http://scikit-learn.org/stable/index.html)

### Installation
1. Method 1
  
   ```
   pip install nltools
   ```

2. Method 2
  
   ```
   pip install git+https://github.com/ljchang/neurolearn
   ```

3. Method 3

   ```
   git clone https://github.com/ljchang/neurolearn
   python setup.py install
   ```

### Dependencies
nltools requires several dependencies.  All are available in pypi.  Can use `pip install 'package'`
 - importlib
 - nibabel>=2.0.1
 - scikit-learn>=0.17
 - nilearn>=0.2
 - pandas>=0.16
 - numpy>=1.9
 - seaborn>=0.7.0
 - matplotlib
 - scipy
 - six
 
### Optional Dependencies
 - mne
 - pyneurovault_upload (`pip install git+https://github.com/neurolearn/pyneurovault_upload`)
 
### Documentation
Current Documentation can be found at [readthedocs](http://neurolearn.readthedocs.org/en/latest).  

Please see our [jupyter notebook](https://github.com/ljchang/neurolearn/blob/master/scripts/NLTools_Brain_Data_Class_Tutorial.ipynb), which provides a detailed overview of how to use the main *Brain_Data* class.  

### Brain_Data()

The nltools toolbox is built around the Brain_Data class which provides an intuitive 2-D representation of imaging data.  It is possible to do simple data manipulations, merging, plotting, and masking.  The main advanatage of the toolbox is that it provides an intuitive method to perform flexible data-analysis.  Here are a couple of quick examples to show you how easy it is to perform manipulations and analyses on brain imaging data.

``` python
from nltools.data import Brain_Data
import seaborn as sns

# Create a Brain_Data instance from a list of files
dat=Brain_Data(['file1','file2','file3'])

# Plot an axial montage of the mean of images [2,4,6]
dat[[2,4,6]].mean().plot()

# Create a distance matrix using cosine similarity between each image in dat and plot using seaborn
sns.heatmap(dat.distance(method='cosine'))

# Run a SVM on the data classifying dat.Y and plot results with k=2 cross-validation
dat.Y = 'pandas object with class labels
results = dat.predict(algorithm='svm', cv_dict={'type': 'kfolds','n_folds': 2, 'n':len(dat.Y)}, plot=False,**{'kernel':"linear"})
results['weight_map'].plot()

# Calculate spatial similarity between each image in dat and the SVM weightmap
r = dat.similarity(stats['weight_map'])

# Run a univariate regression on each voxel using model dat.X
dat.X = pd.DataFrame('Your Model')
results = dat.regress()

# Extract average intensity in spherical ROI across each image
s = create_sphere([41, 64, 55], radius=10)
roi_avg = dat.extract_roi(s)
```

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

