"""
Basic Data Operations
=====================

A simple example showing how to download a dataset from neurovault and perform
basic data operations.  The bulk of the nltools toolbox is built around the
Brain_Data() class.  This class represents imaging data as a vectorized
features by observations matrix.  Each image is an observation and each voxel
is a feature.  The concept behind the class is to have a similar feel to a pandas
dataframe, which means that it should feel intuitive to manipulate the data.

"""

#########################################################################
# Download pain dataset from neurovault
# ---------------------------------------------------
#
# Here we fetch the pain dataset used in `Chang et al., 2015 <http://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002180>`_
# from `neurovault <http://neurovault.org/collections/504/>`_. In this dataset
# there are 28 subjects with 3 separate beta images reflecting varying intensities
# of thermal pain (i.e., high, medium, low).  The data will be downloaded to ~/nilearn_data,
# and automatically loaded as a Brain_Data() instance.  The image metadata will be stored in data.X.

from nltools.datasets import fetch_pain

data = fetch_pain()

#########################################################################
# Load files
# ---------------------------------------------------
#
# Nifti images can be easily loaded simply by passing a string to a nifti file.
# Many images can be loaded together by passing a list of nifti files.
# For example, on linux or OSX systmes, the downloads from fetch_pain() will be
# stored in ~/nilearn_data.  We will load subject 1's data.

# NOTES: Need to figure out how to get path to data working on rtd server
# from nltools.data import Brain_Data
# import glob
#
# sub1 = Brain_Data(glob.glob('~/nilearn_data/chang2015_pain/Pain_Subject_1*.nii.gz'))

#########################################################################
# Basic Brain_Data() Operations
# ---------------------------------------------------------
#
# Here are a few quick basic data operations.
# Find number of images in Brain_Data() instance

print(len(data))

#########################################################################
# Find the dimensions of the data.  images x voxels

print(data.shape())

#########################################################################
# We can use any type of indexing to slice the data such as integers, lists
# of integers, or boolean.

print(data[[1,6,2]])

#########################################################################
# Calculate the mean for every voxel over images

data.mean()

#########################################################################
# Calculate the standard deviation for every voxel over images

data.std()

#########################################################################
# Methods can be chained.  Here we get the shape of the mean.

print(data.mean().shape())

#########################################################################
# Brain_Data instances can be added and subtracted

new = data[1]+data[2]

#########################################################################
# Brain_Data instances can be manipulated with basic arithmetic operations
# Here we add 10 to every voxel and scale by 2

data2 = (data+10)*2

#########################################################################
# Brain_Data instances can be copied

new = data.copy()

#########################################################################
# Brain_Data instances can be easily converted to nibabel instances, which
# store the data in a 3D/4D matrix.  This is useful for interfacing with other
# python toolboxes such as `nilearn <http://nilearn.github.io/>`_

data.to_nifti()

#########################################################################
# Brain_Data instances can be concatenated using the append method

new = new.append(data[4])

#########################################################################
# Any Brain_Data object can be written out to a nifti file

data.write('Tmp_Data.nii.gz')

#########################################################################
# Images within a Brain_Data() instance are iterable.  Here we use a list
# comprehension to calculate the overall mean across all voxels within an
# image.

[x.mean() for x in data]

#########################################################################
# Basic Brain_Data() Plotting
# ---------------------------------------------------------
#
# There are multiple ways to plot data.  First, Brain_Data() instances can be
# converted to a nibabel instance and plotted using any plot method such as
# nilearn.

from nilearn.plotting import plot_glass_brain

plot_glass_brain(data.mean().to_nifti())

#########################################################################
# There is also a fast montage plotting method.  Here we plot the average image
# it will render a separate plot for each image.  There is a 'limit' flag
# which allows you to specify the maximum number of images to display.

data.mean().plot()
