""" 
Masking Example
===============
This tutorial illustrates methods to help with masking data.

"""

#########################################################################
# Load Data
# ---------
# 
# First, let's load the pain data for this example.

from nltools.datasets import fetch_pain

data = fetch_pain()

#########################################################################
# Apply_Mask
# ----------
#
# Spherical masks can be created using the create_sphere function.
# It requires specifying a center voxel and the radius of the sphere.

from nltools.mask import create_sphere

mask = create_sphere([0, 0, 0], radius=30)
masked_data = data.apply_mask(mask)
masked_data.mean().plot()

#########################################################################
# Extract Mean Within ROI
# -----------------------
#
# We can easily calculate the mean within an ROI for each image within a 
# Brain_Data() instance using the extract_roi() method. 

import matplotlib.pyplot as plt

mean = data.extract_roi(mask)
plt.plot(mean)

#########################################################################
# Expand and Contract ROIs
# ------------------------
#
# Some masks have many ROIs indicated by a unique ID.  It is possible to
# expand these masks into separate ROIs and also collapse them into a single
# image again.  Here we will demonstrate on a k=50 parcellation hosted on
# http://neurovault.org.

from nltools.mask import expand_mask, collapse_mask
from nltools.data import Brain_Data

mask = Brain_Data('http://neurovault.org/media/images/2099/Neurosynth%20Parcellation_0.nii.gz')

mask.plot()

mask_x = expand_mask(mask)
mask_x.plot()

mask_c = collapse_mask(mask_x)
mask_c.plot()

#########################################################################
# Threshold and Regions
# ---------------------
#
# Images can be thresholded using an arbitrary cutoff or a percentile using the 
# threshold method.  Here we calculate the mean of the high pain images and
# threshold using the 95 percentile.

import numpy as np

high = data[np.where(data.X['PainLevel']==3)[0]]
high.mean().threshold(threshold='95%').plot()

#########################################################################
# We might be interested in creating a binary mask from this threshold.

mask = high.mean().threshold(threshold='95%',binarize=True)

#########################################################################
# We might also want to create separate images from each contiguous ROI.

region = high.mean().threshold(threshold='95%').regions()
region.plot()

