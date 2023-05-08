""" 
Brain resolution and MNI Template Preferences
=============================================

By default nltools uses a 2mm MNI template which means all `Brain_Data` operations will automatically be resampled to that space if they aren't already at that resolution. If you know you want to work in another space you can set that for all operations using the prefs module:
"""

#########################################################################
# Setting GLOBAL MNI template preferences
# ---------------------
#
from nltools.prefs import MNI_Template, resolve_mni_path
from nltools.data import Brain_Data
from nltools.simulator import Simulator  # just for dummy data

#########################################################################
# Here we create some dummy data. Notice that it defaults to 2mm resolution. You can verify this by seeing that the voxel count is approximately 240k:
dummy_brain = Simulator().create_data([0, 1], 1, reps=3)
dummy_brain.write("dummy_2mm_brain.nii.gz")  # save it for later
dummy_brain  # default 2mm resolution

#########################################################################
# You can also get the exact file locations of the currently loaded default template and masks:
resolve_mni_path(MNI_Template)

#########################################################################
# To update this simply change the resolution attribute of the MNI_Template. NOTE: that this will change **all** subsequent Brain_Data operations to utilize this new space. Therefore we **highly recommend** doing this at the top of any analysis notebook or script you use to prevent unexpected results
MNI_Template.resolution = 3  # passing the string '3mm' also works
dummy_brain_3mm = Simulator().create_data([0, 1], 1, reps=3)
dummy_brain_3mm  # should be 3mm

#########################################################################
# The voxel count is now ~70k and you can see the file paths of the global template:

resolve_mni_path(MNI_Template)

#########################################################################
# Notice that when we load we load the previous 2mm brain, it's **automatically** resampled to the currently set default MNI template (3mm):
loaded_brain = Brain_Data("dummy_2mm_brain.nii.gz")
loaded_brain  # now in 3mm space!

#########################################################################
# Setting local resolution preferences
# ------------------------------------
#
# If you want to override the global setting on a case-by-case basis, simply use the `mask` argument in `Brain_Data`. This will resample data to the resolution of the `mask` ignoring whatever `MNI_Template` is set to:

# Here we save the 3mm path as a variable, but in your own data you can provide
# the location of any nifti file
mask_file_3mm = resolve_mni_path(MNI_Template)["mask"]

MNI_Template.resolution = 2  # reset the global MNI template to 2mm
load_using_default = Brain_Data("dummy_2mm_brain.nii.gz")
load_using_default  # 2mm space

#########################################################################
load_using_3mm_mask = Brain_Data("dummy_2mm_brain.nii.gz", mask=mask_file_3mm)
load_using_3mm_mask  # resampled to 3mm space because a mask was provided

#########################################################################
# Notice that the global setting is still 2mm, but by providing a `mask` we were able to override it

resolve_mni_path(MNI_Template)
