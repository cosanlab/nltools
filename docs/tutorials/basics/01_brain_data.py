# %% [markdown]
# # BrainData Basics
#
# The `BrainData` class is the core data structure in nltools.

# %%
import matplotlib

matplotlib.use("Agg")

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn.plotting import plot_glass_brain

from nltools.data import BrainData
from nltools.datasets import fetch_haxby
from nltools.utils import concatenate

# %% [markdown]
# ## Load Data

# %%
data, design = fetch_haxby(n_subjects=1)
brain = data[0]  # First run

print(brain)
print(f"Shape: {brain.shape}")

# %% [markdown]
# ## Indexing and Slicing

# %%
# Single image
first_image = brain[0]
print(f"Single image: {first_image.shape}")

# Slicing
first_five = brain[:5]
print(f"Sliced: {first_five.shape}")

# List indexing
selected = brain[[0, 10, 20, 30]]
print(f"Selected: {selected.shape}")

# Boolean indexing
high_intensity = brain[brain.mean(axis=1) > brain.mean(axis=1).mean()]
print(f"Boolean filtered: {len(high_intensity)}")

# %% [markdown]
# ## Statistics

# %%
# Mean across images (returns single brain)
mean_brain = brain.mean()
print(f"Mean brain: {mean_brain.shape}")

# Std across images
std_brain = brain.std()

# Mean within each image (global signal)
mean_timeseries = brain.mean(axis=1)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(mean_timeseries)
ax.set_xlabel("Image Number")
ax.set_ylabel("Mean Intensity")
ax.set_title("Global Signal")
plt.close()

# %% [markdown]
# ## Arithmetic

# %%
# Scalar operations
scaled = brain * 2
shifted = brain + 100

# Mean-center
centered = brain - brain.mean()
print(f"Centered mean: {centered.mean().data.mean():.10f}")

# Between objects
difference = brain[1] - brain[0]
print(f"Difference: {difference.shape}")

# %% [markdown]
# ## Processing

# %%
# Standardize (z-score each voxel across images)
z_scored = brain.standardize()
print(f"Z-scored mean: {z_scored.mean().data.mean():.4f}")
print(f"Z-scored std: {z_scored.std().data.mean():.4f}")

# %%
# Smooth with 6mm FWHM
smoothed = brain[0].smooth(fwhm=6)
print(f"Original range: [{brain[0].data.min():.2f}, {brain[0].data.max():.2f}]")
print(f"Smoothed range: [{smoothed.data.min():.2f}, {smoothed.data.max():.2f}]")

# %%
# Threshold by percentile
top_10_percent = mean_brain.threshold(upper="90%")
print(f"Voxels in top 10%: {(top_10_percent.data > 0).sum()}")

# %% [markdown]
# ## Visualization

# %%
mean_brain.plot(title="Mean Brain")

# %%
brain.plot(kind="timeseries")

# %%
mean_brain.plot(kind="histogram", title="Voxel Distribution")

# %%
# Use nilearn for glass brain
plot_glass_brain(mean_brain.to_nifti(), title="Glass Brain")

# %% [markdown]
# ## File I/O

# %%
# Save as NIfTI
mean_brain.write("/tmp/mean_activation.nii.gz")

# Load (with mask to preserve shape)
loaded = BrainData("/tmp/mean_activation.nii.gz", mask=brain.mask)
print(f"Loaded: {loaded.shape}")

assert np.allclose(mean_brain.data, loaded.data, rtol=1e-5)
print("✓ Data verified")

# %% [markdown]
# ## Format Conversion

# %%
# To nibabel NIfTI
nifti_img = mean_brain.to_nifti()
print(f"NIfTI shape: {nifti_img.shape}")

# From nibabel
from_nifti = BrainData(nifti_img)
print(f"From NIfTI: {from_nifti.shape}")

# %%
# Raw numpy array
raw_data = brain.data
print(f"Array shape: {raw_data.shape}")  # (images, voxels)

# %% [markdown]
# ## Metadata (X and Y)

# %%
print(f"X shape: {brain.X.shape if hasattr(brain.X, 'shape') else 'None'}")
print(f"Y: {brain.Y}")

# %%
# Add outcome labels
n_images = len(brain)
labels = pd.DataFrame(
    {
        "condition": ["task"] * (n_images // 2) + ["rest"] * (n_images - n_images // 2),
        "intensity": np.random.randn(n_images),
    }
)

labeled = brain.copy()
labeled.Y = labels
print(labeled.Y.head())

# %%
# Filter by metadata
task_images = labeled[labeled.Y["condition"] == "task"]
rest_images = labeled[labeled.Y["condition"] == "rest"]
print(f"Task: {len(task_images)}, Rest: {len(rest_images)}")

# Compare conditions
difference = task_images.mean() - rest_images.mean()
print(f"Difference: {difference.shape}")

# %% [markdown]
# ## Concatenation

# %%
# Concatenate multiple objects
data_subset = concatenate([brain[i] for i in range(4)])
print(f"Concatenated: {len(data_subset)} images")

# Append single image
new = brain[0].append(brain[1])
print(f"After append: {len(new)} images")

# %%
# Clean up
if os.path.exists("/tmp/mean_activation.nii.gz"):
    os.remove("/tmp/mean_activation.nii.gz")
