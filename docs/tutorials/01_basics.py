# %% [markdown]
# # Introduction to Neuroimaging Data
#
# In this tutorial we will learn the basics of the organization of data folders, and how to load, plot, and manipulate neuroimaging data in Python.
#
# To introduce the basics of fMRI data structures, watch this short video by Martin Lindquist.

# %% [markdown]
# ## Loading Neuroimaging Data with nltools
#
# In this tutorial, we'll learn how to load, manipulate, and visualize neuroimaging data using the `nltools` package. The `nltools` toolbox makes working with neuroimaging data intuitive by providing a pandas-like interface for brain images.
#
# We'll be working with the **Haxby dataset**, a classic fMRI dataset that contains data from participants viewing images of faces, houses, and other objects. This dataset is publicly available and perfect for learning neuroimaging analysis.

# %% [markdown]
# ## Manipulating Data with Nltools
#
# The bulk of the nltools toolbox is built around the `BrainData` class. The concept behind the class is to have a similar feel to a pandas DataFrame, which means that it should feel intuitive to manipulate the data.
#
# The `BrainData` class has several attributes that may be helpful to know about. First, it stores imaging data in `.data` as a vectorized features by observations matrix. Each image is an observation and each voxel is a feature. Space is flattened using `nifti_masker` from nilearn. This object is also stored as an attribute in `.nifti_masker` to allow transformations from 2D to 3D/4D matrices. In addition, a brain_mask is stored in `.mask`. Finally, there are attributes to store either class labels for prediction/classification analyses in `.Y` and design matrices in `.design_matrix`.
#
# We will give a quick overview of basic BrainData operations, but we encourage you to see our [documentation](https://nltools.org/) for more details.
#
# ### BrainData basics
# To get a feel for `BrainData`, let's load an example anatomical overlay image that comes packaged with the toolbox.

# %%
from nltools.datasets import fetch_haxby
import numpy as np

subject_fmri_runs, subject_design_mats = fetch_haxby(n_subjects=1)

# Use the first run for this tutorial
data = subject_fmri_runs[0]
print(f"Loaded {len(subject_fmri_runs)} runs")
print(f"First run has {len(data)} timepoints")

# %% [markdown]
# `BrainData` has many methods to help manipulate, plot, and analyze imaging data. We can use the `dir()` function to get a quick list of all of the available methods that can be used on this class.
#
# To learn more about how to use these tools either use the `?` function, or look up the function in the [api documentation](https://nltools.org/api.html).

# %%
print([m for m in dir(data) if not m.startswith("_")][:20])

# %% [markdown]
# Here are a few quick basic data operations.

# %% [markdown]
# Find the dimensions of the data (images x voxels)

# %%
print(f"Data shape: {data.shape} (timepoints x voxels)")

# %% [markdown]
# We can use any type of indexing to slice the data such as integers, lists of integers, slices, or boolean vectors.

# %%
# Slice first 10 timepoints
data[0:10]

# %%
# Get a single time-point
data[5]

# %%
# boolean/mask indexing like numpy
mask = np.zeros(len(data), dtype=bool)

# Get only these 6 time-points
mask[[1, 5, 9, 16, 20, 22]] = True

data[mask]

# %% [markdown]
# ### Simple Arithmetic Operations

# %% [markdown]
# Calculate the mean for every voxel over images

# %%
data.mean()

# %% [markdown]
# Or over voxels using `axis=1`

# %%
data.mean(axis=1)

# %% [markdown]
# Calculate the standard deviation for every voxel over images

# %%
data.std()

# %% [markdown]
# BrainData instances can be added and subtracted

# %%
new = data[1] + data[2]
new

# %% [markdown]
# And manipulated with basic arithmetic operations.
#
# Here we add 10 to every voxel and scale by 2

# %%
data2 = (data + 10) * 2
data2


# %% [markdown]
# BrainData instances can be concatenated using the append method

# %%
new = new.append(data[4])
print(f"After appending one timepoint: {len(new)} timepoints")

# %% [markdown]
# Lists of `BrainData` instances can also be concatenated using the `concatenate` function.

# %%
from nltools.utils import concatenate

# Concatenate multiple BrainData objects
data_subset = concatenate([data[i] for i in range(4)])
print(f"Concatenated data has {len(data_subset)} timepoints")

# %% [markdown]
# `BrainData` is also compatible with all tools that support the standard NiFTI image format.
# So you can use additional tools like [`nilearn`](http://nilearn.github.io) for plotting and `FSL` for additional image statistics.

# %%
nifti_img = data.to_nifti()

# 4d volume
print(f"Nifti image shape: {nifti_img.shape}")

# %% [markdown]
# Any BrainData object can be written out to a nifti file:
#
# ```python
#
# data.write('myfile.nii.gz')
#
# ```

# %% [markdown]
# ### Plotting
# There are multiple ways to plot your data.
#
# For a very quick plot, you can return a montage of axial slices with the `.plot()` method. As an example, we will plot the mean of each voxel over time.

# %%
# Plot mean activation across all timepoints
mean_brain = data.mean()
mean_brain.plot()

# %% [markdown]
# Using the `kind` argument you can also generate a few other quick plot types such as a timeseries (collapsing over voxel)

# %%
# Equivalent to data.mean(axis=1) and then using matplotlib
data.plot(kind="timeseries")

# %% [markdown]
# Or a histogram to quickly see values across the entire brain

# %%
data.mean().plot(kind="histogram", title="Histogram of mean signal")

# %% [markdown]
# Using `.to_nifti()` you can always utlize all of the additional more sophisticated plotting methods available in `nilearn`

# %%
from nilearn.plotting import plot_glass_brain

# Plot mean brain on glass brain
plot_glass_brain(mean_brain.to_nifti())
