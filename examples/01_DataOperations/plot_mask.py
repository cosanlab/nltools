""" 
Masking Example
===========
This tutorial illustrates methods to help with masking data.

"""

#########################################################################
# Load Data
# ---------------------------------------------------
# 
# First, let's load the pain data for this example.

from nltools.datasets import fetch_pain

data = fetch_pain()

#########################################################################
# Apply_Mask
# ---------------------------------------------------------
#
# We can calculate the pairwise spatial distance between all images in a Brain_Data()
# instance using any method from sklearn or scipy.  This outputs an Adjacency() class
# object.

s1 = create_sphere([41, 64, 55], radius=10)
masked_dat = dat.apply_mask(s1)
masked_dat[1].plot()

#########################################################################
# Extract Mean Within ROI
# ---------------------------------------------------------
#
# We can calculate the pairwise spatial distance between all images in a Brain_Data()
# instance using any method from sklearn or scipy.  This outputs an Adjacency() class
# object.

mask = create_sphere([41, 64, 55], radius=10)
out = dat.extract_roi(mask)
plt.plot(out)

#########################################################################
# Expand and Contract ROIs
# ---------------------------------------------------------
#
# We can calculate the pairwise spatial distance between all images in a Brain_Data()
# instance using any method from sklearn or scipy.  This outputs an Adjacency() class
# object.

#########################################################################
# Threshold and Regions
# ---------------------------------------------------------
#
# We can calculate the pairwise spatial distance between all images in a Brain_Data()
# instance using any method from sklearn or scipy.  This outputs an Adjacency() class
# object.
