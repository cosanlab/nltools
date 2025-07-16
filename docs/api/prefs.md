# `nltools.prefs`

**Preferences**

This module can be used to adjust the default MNI template settings that are used internally by all `Brain_Data` operations. For historic reasons, the default MNI template is in the same **[MNI152 2mm space as FSL](https://nist.mni.mcgill.ca/mni-icbm152-non-linear-6th-generation-symmetric-average-brain-stereotaxic-registration-model/)**. Different software use [different versions](https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/) `nltools` supports the following additional verisons:

- [`nilearn` MNI152 2009a](https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_icbm152_2009.html#nilearn.datasets.fetch_icbm152_2009)
- [`fmriprep` MNI52 2009c](https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_icbm152_2009.html#nilearn.datasets.fetch_icbm152_2009)

Switching the MNI template uses will affect **all** subsequent operations by resampling data to the chosen space You can control this on a per-file loading basis using the `mask` argument of `Brain_Data`, e.g.

```python
from nltools.data import Brain_Data

# my_brain will be resampled to 2mm
brain = Brain_Data('my_brain.nii.gz') 

# my_brain will now be resampled to the same space as my_mask
brain = Brain_Data('my_brain.nii.gz', mask='my_mask.nii.gz') # will be resampled 
```

Alternatively this module can be used to switch between 2mm or 3mm MNI spaces with and without ventricles:

```python
from nltools.prefs import MNI_Template, resolve_mni_path
from nltools.data import Brain_Data

# Update the resolution globally
MNI_Template['resolution'] = '3mm'

# This works too:
MNI_Template.resolution = 3

# my_brain will be resampled to 3mm and future operation will be in 3mm space
brain = Brain_Data('my_brain.nii.gz') 

# get the template nifti files
resolve_mni_path(MNI_Template)

# will print like:
{
    'resolution': '3mm',
    'mask_type': 'with_ventricles',
    'mask': '/Users/Esh/Documents/pypackages/nltools/nltools/resources/MNI152_T1_3mm_brain_mask.nii.gz',
    'plot': '/Users/Esh/Documents/pypackages/nltools/nltools/resources/MNI152_T1_3mm.nii.gz',
    'brain':
    '/Users/Esh/Documents/pypackages/nltools/nltools/resources/MNI152_T1_3mm_brain.nii.gz'
}
```

```{eval-rst}
.. automodule:: nltools.prefs
   :members:
   :show-inheritance: