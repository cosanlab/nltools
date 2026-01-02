# %% [markdown]
# # BrainData Basics
#
# This tutorial covers the core `BrainData` class - the foundation of nltools.
#
# ## Learning Objectives
#
# By the end of this tutorial, you will be able to:
# - Load neuroimaging data into `BrainData` objects
# - Perform basic operations (indexing, slicing, arithmetic)
# - Compute summary statistics across images and voxels
# - Apply common preprocessing steps (smoothing, standardization)
# - Visualize brain images and timeseries
# - Save and load data in different formats
# - Work with masks and metadata

# %% [markdown]
# ## Introduction
#
# The `BrainData` class is the core data structure in nltools for working with
# neuroimaging data. It's designed to make common operations intuitive while
# providing access to powerful preprocessing and analysis tools.
#
# **Key features**:
# - Stores data as 2D arrays (images × voxels) for efficient computation
# - Automatically handles resampling to standard MNI space
# - Supports standard Python operations (indexing, arithmetic, etc.)
# - Integrates seamlessly with nilearn and nibabel
# - Includes metadata storage for experimental designs

# %% [markdown]
# ## Loading Data

# %%
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for script execution

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltools.data import BrainData
from nltools.datasets import fetch_haxby

# Load example dataset
data, design = fetch_haxby(n_subjects=1)
brain = data[0]  # First run

print(brain)
print(f"Shape: {brain.shape}")

# %% [markdown]
# The `BrainData` object shows:
# - **Shape**: images × voxels
# - **Y**: Outcome variables (if any)
# - **X**: Design matrix/metadata
# - **mask**: Brain mask being used

# %% [markdown]
# ## Indexing and Slicing
#
# `BrainData` supports standard Python indexing:

# %%
# Get a single image
first_image = brain[0]
print(f"Single image shape: {first_image.shape}")

# Get a subset using slicing
first_five = brain[:5]
print(f"Five images shape: {first_five.shape}")

# Index with a list
selected = brain[[0, 10, 20, 30]]
print(f"Selected images shape: {selected.shape}")

# %%
# Index with boolean array
high_intensity = brain[brain.mean(axis=1) > brain.mean(axis=1).mean()]
print(f"High intensity images: {len(high_intensity)}")

# %% [markdown]
# **Note**: Indexing preserves metadata (X and Y dataframes)

# %% [markdown]
# ## Basic Statistics
#
# Compute statistics across images (axis=0) or within images across voxels (axis=1):

# %%
# Mean across all images (returns single brain image)
mean_brain = brain.mean()
print(f"Mean brain shape: {mean_brain.shape}")

# Standard deviation across images
std_brain = brain.std()

# %%
# Statistics within each image
# Mean intensity per image (global signal)
mean_timeseries = brain.mean(axis=1)

# Plot global signal across images
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(mean_timeseries)
ax.set_xlabel("Image Number")
ax.set_ylabel("Mean Intensity")
ax.set_title("Global Signal Timeseries")
plt.tight_layout()
plt.close()

# %% [markdown]
# ## Arithmetic Operations
#
# `BrainData` supports element-wise operations:

# %%
# Operations with scalars
scaled = brain * 2  # Multiply all voxels by 2
shifted = brain + 100  # Add 100 to all voxels

# Mean-center the data
centered = brain - brain.mean()
print(f"Mean of centered data: {centered.mean().data.mean():.10f}")  # Should be ~0

# %%
# Operations between BrainData objects
difference = brain[1] - brain[0]
print(f"Difference shape: {difference.shape}")

# %% [markdown]
# ## Data Processing Methods
#
# ### Standardization (Z-scoring)

# %%
# Standardize each voxel across images
z_scored = brain.standardize()

print(f"Original mean: {brain.mean().data.mean():.4f}")
print(f"Z-scored mean: {z_scored.mean().data.mean():.4f}")
print(f"Z-scored std: {z_scored.std().data.mean():.4f}")

# %% [markdown]
# ### Spatial Smoothing

# %%
# Smooth with 6mm FWHM Gaussian kernel
smoothed = brain[0].smooth(fwhm=6)

print(f"Original range: [{brain[0].data.min():.2f}, {brain[0].data.max():.2f}]")
print(f"Smoothed range: [{smoothed.data.min():.2f}, {smoothed.data.max():.2f}]")

# %% [markdown]
# ### Thresholding

# %%
# Threshold by percentile
top_10_percent = mean_brain.threshold(upper="90%")
print(f"Voxels in top 10%: {(top_10_percent.data > 0).sum()}")

# %% [markdown]
# ## Visualization
#
# ### Basic Plotting

# %%
# Plot a single brain image
mean_brain.plot(title="Mean Brain")

# %%
# Plot as timeseries (mean per timepoint)
brain.plot(kind="timeseries")

# %%
# Histogram of voxel values
mean_brain.plot(kind="histogram", title="Voxel Value Distribution")

# %% [markdown]
# ### Using nilearn for advanced plots

# %%
from nilearn.plotting import plot_glass_brain  # noqa: E402

# Convert to NIfTI and use nilearn
plot_glass_brain(mean_brain.to_nifti(), title="Glass Brain View")

# %% [markdown]
# ## Working with Masks
#
# ### Creating and Applying Masks

# %%
# Create mask from thresholded image
mask = mean_brain.threshold(lower=mean_brain.data.mean())
print(f"Mask contains {(mask.data > 0).sum():.0f} voxels")

# %% [markdown]
# ## File I/O
#
# ### Saving Data

# %%
# Save as NIfTI file
mean_brain.write("/tmp/mean_activation.nii.gz")
print("Saved to /tmp/mean_activation.nii.gz")

# %% [markdown]
# ### Loading Data

# %%
# Load from NIfTI (using same mask to preserve shape)
loaded = BrainData("/tmp/mean_activation.nii.gz", mask=brain.mask)
print(f"Loaded shape: {loaded.shape}")

# Verify data matches
assert np.allclose(mean_brain.data, loaded.data, rtol=1e-5), "Data should match"
print("✓ Data verified")

# %% [markdown]
# ## Converting Between Formats
#
# ### To/From Nibabel

# %%
# Convert to nibabel NIfTI object
nifti_img = mean_brain.to_nifti()
print(f"Type: {type(nifti_img).__name__}")
print(f"Shape: {nifti_img.shape}")

# Create BrainData from nibabel object
from_nifti = BrainData(nifti_img)
print(f"From NIfTI shape: {from_nifti.shape}")

# %% [markdown]
# ### Working with NumPy

# %%
# Access raw data array
raw_data = brain.data
print(f"Type: {type(raw_data).__name__}")
print(f"Shape: {raw_data.shape}")  # (images, voxels)
print(f"Dtype: {raw_data.dtype}")

# %% [markdown]
# ## Working with Metadata
#
# ### Understanding X and Y

# %%
# Check if design matrix is attached
print(
    f"Design matrix (X) shape: {brain.X.shape if hasattr(brain.X, 'shape') else 'None'}"
)

# Y is for outcome variables (used in prediction/regression)
print(f"Y: {brain.Y}")

# %% [markdown]
# ### Adding Metadata

# %%
# Create outcome labels
n_images = len(brain)
labels = pd.DataFrame(
    {
        "condition": ["task"] * (n_images // 2) + ["rest"] * (n_images - n_images // 2),
        "intensity": np.random.randn(n_images),
    }
)

# Create BrainData with metadata
labeled = brain.copy()
labeled.Y = labels
print(f"Y shape: {labeled.Y.shape}")
print(labeled.Y.head())

# %% [markdown]
# ### Filtering by Metadata

# %%
# Use boolean indexing with metadata
task_images = labeled[labeled.Y["condition"] == "task"]
print(f"Task images: {len(task_images)}")

rest_images = labeled[labeled.Y["condition"] == "rest"]
print(f"Rest images: {len(rest_images)}")

# Compare conditions
task_mean = task_images.mean()
rest_mean = rest_images.mean()
difference = task_mean - rest_mean
print(f"Difference computed: {difference.shape}")

# %% [markdown]
# ## Concatenation

# %%
from nltools.utils import concatenate  # noqa: E402

# Concatenate multiple BrainData objects
data_subset = concatenate([brain[i] for i in range(4)])
print(f"Concatenated: {len(data_subset)} images")

# Append single image
new = brain[0].append(brain[1])
print(f"After append: {len(new)} images")

# %% [markdown]
# ## Summary
#
# In this tutorial, you learned:
# - ✓ How to load data into `BrainData` objects
# - ✓ Indexing and slicing operations
# - ✓ Computing statistics across images and voxels
# - ✓ Arithmetic operations on brain data
# - ✓ Preprocessing (smoothing, standardization, thresholding)
# - ✓ Creating and applying masks
# - ✓ Visualization methods
# - ✓ File I/O and format conversion
# - ✓ Working with metadata (X and Y)
#
# ## Next Steps
#
# - **[DesignMatrix Basics](02_design_matrix)**: Creating experimental designs
# - **[Adjacency Basics](03_adjacency)**: Connectivity and similarity matrices
# - **[GLM Workflow](../workflows/01_glm)**: Complete first-level analysis

# %%
# Clean up
import os  # noqa: E402

if os.path.exists("/tmp/mean_activation.nii.gz"):
    os.remove("/tmp/mean_activation.nii.gz")
