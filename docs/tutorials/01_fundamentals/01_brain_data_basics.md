# Brain_Data Basics

## Learning Objectives

By the end of this tutorial, you will be able to:
- Load neuroimaging data into `Brain_Data` objects
- Perform basic operations (indexing, slicing, arithmetic)
- Compute summary statistics across images and voxels
- Apply common preprocessing steps (smoothing, standardization)
- Visualize brain images and timeseries
- Save and load data in different formats
- Work with masks and metadata

## Introduction

The `Brain_Data` class is the core data structure in nltools for working with neuroimaging data. It's designed to make common operations intuitive while providing access to powerful preprocessing and analysis tools.

**Key features**:
- Stores data as 2D arrays (images × voxels) for efficient computation
- Automatically handles resampling to standard MNI space
- Supports standard Python operations (indexing, arithmetic, etc.)
- Integrates seamlessly with nilearn and nibabel
- Includes metadata storage for experimental designs

## Loading Data

```python
# Import required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltools.data import Brain_Data
from nltools.datasets import fetch_pain

# Load example dataset (pain perception study)
data = fetch_pain()
print(data)
```

The `Brain_Data` object shows:
- **Shape**: 84 images × 238,955 voxels (MNI 2mm space)
- **Y**: Outcome variables (if any)
- **X**: Design matrix/metadata
- **mask**: Brain mask being used

## Indexing and Slicing

`Brain_Data` supports standard Python indexing:

```python
# Get a single image
first_image = data[0]
print(f"Single image shape: {first_image.shape()}")

# Get a subset using slicing
first_five = data[:5]
print(f"Five images shape: {first_five.shape()}")

# Index with a list
selected = data[[0, 10, 20, 30]]
print(f"Selected images shape: {selected.shape()}")

# Index with boolean array
high_intensity = data[data.mean(axis=1) > 0.1]
print(f"High intensity images: {len(high_intensity)}")
```

**Note**: Indexing preserves metadata (X and Y dataframes)

## Basic Statistics

Compute statistics across images (axis=0) or within images across voxels (axis=1):

```python
# Mean across all images (returns single brain image)
mean_brain = data.mean()
print(f"Mean brain shape: {mean_brain.shape()}")

# Standard deviation across images
std_brain = data.std()

# Create temporal SNR (tSNR) map
tsnr_brain = mean_brain / std_brain
tsnr_brain.plot()
```

Statistics within each image:

```python
# Mean intensity per image (global signal)
mean_timeseries = data.mean(axis=1)

# Plot global signal across images
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(mean_timeseries)
ax.set_xlabel("Image Number")
ax.set_ylabel("Mean Intensity")
ax.set_title("Global Signal Timeseries")
plt.show()
```

## Arithmetic Operations

`Brain_Data` supports element-wise operations:

```python
# Operations with scalars
scaled = data * 2       # Multiply all voxels by 2
shifted = data + 100    # Add 100 to all voxels

# Mean-center the data
centered = data - data.mean()
print(f"Mean of centered data: {centered.mean().mean():.10f}")  # Should be ~0

# Operations between Brain_Data objects
difference = data[1] - data[0]
difference.plot()
```

## Data Processing Methods

### Standardization (Z-scoring)

```python
# Standardize each voxel across images
z_scored = data.standardize()

print(f"Original mean: {data.mean().mean():.4f}")
print(f"Z-scored mean: {z_scored.mean().mean():.4f}")
print(f"Z-scored std: {z_scored.std().mean():.4f}")
```

### Spatial Smoothing

```python
# Smooth with 6mm FWHM Gaussian kernel
smoothed = data[0].smooth(fwhm=6)

print(f"Original range: [{data[0].data.min():.2f}, {data[0].data.max():.2f}]")
print(f"Smoothed range: [{smoothed.data.min():.2f}, {smoothed.data.max():.2f}]")

# Compare original and smoothed
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
data[0].plot(axes=axes[0])
axes[0].set_title("Original")
smoothed.plot(axes=axes[1])
axes[1].set_title("Smoothed (6mm FWHM)")
plt.show()
```

### Thresholding

```python
# Threshold by value
thresholded = mean_brain.threshold(lower=2, upper=10)
n_voxels = (thresholded.data > 0).sum()
print(f"Voxels within threshold: {n_voxels}")

# Threshold by percentile
top_5_percent = mean_brain.threshold(upper='95%')
top_5_percent.plot()

# Binarize for masking
binary_mask = mean_brain.threshold(upper='95%', binarize=True)
binary_mask.plot()
```

## Visualization

### Basic Plotting

```python
# Plot a single brain image
mean_brain.plot()

# Plot multiple images in a grid
data[:4].plot()

# Glass brain view
mean_brain.plot(view='glass')
```

### Interactive Plotting

```python
# Interactive viewer (requires additional dependencies)
mean_brain.iplot()
```

### Customizing Plots

```python
# Control colormap and thresholds
mean_brain.plot(
    cmap='RdBu_r',
    vmin=-2,
    vmax=2,
    title='Mean Activation'
)
```

## Working with Masks

### Creating Masks

```python
# Create mask from thresholded image
mask = mean_brain.threshold(lower=0.1, binarize=True)
print(f"Mask contains {mask.data.sum():.0f} voxels")

# Visualize the mask
mask.plot()
```

### Applying Masks

```python
# Apply mask to data
masked_data = data.apply_mask(mask)

# Visualize original vs masked
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
data.mean().plot(axes=axes[0])
axes[0].set_title("Original")
masked_data.mean().plot(axes=axes[1])
axes[1].set_title("Masked")
plt.show()
```

### Loading Standard Masks

```python
# Load atlas or ROI mask from file
# roi_mask = Brain_Data('path/to/roi.nii.gz')

# Apply to your data
# roi_data = data.apply_mask(roi_mask)
```

## File I/O

### Saving Data

```python
# Save as NIfTI file
mean_brain.write('mean_activation.nii.gz')

# Save as HDF5 (preserves metadata)
data.write('brain_data_with_metadata.h5')
```

### Loading Data

```python
# Load from NIfTI
loaded_nifti = Brain_Data('mean_activation.nii.gz')

# Load from HDF5
loaded_h5 = Brain_Data('brain_data_with_metadata.h5')

# Load from list of files
# file_list = ['subj1.nii.gz', 'subj2.nii.gz', 'subj3.nii.gz']
# multi_subject = Brain_Data(file_list)
```

## Converting Between Formats

### To/From Nibabel

```python
# Convert to nibabel NIfTI object
nifti_img = mean_brain.to_nifti()
print(f"Type: {type(nifti_img)}")
print(f"Shape: {nifti_img.shape}")

# Create Brain_Data from nibabel object
from_nifti = Brain_Data(nifti_img)
print(from_nifti)
```

### Working with NumPy

```python
# Access raw data array
raw_data = data.data
print(f"Type: {type(raw_data)}")
print(f"Shape: {raw_data.shape}")  # (images, voxels)
print(f"Dtype: {raw_data.dtype}")

# Perform custom operations on raw data
custom_result = np.mean(raw_data, axis=0)

# Create Brain_Data from numpy array
# Note: Must specify the mask
# from_array = Brain_Data(data=custom_result, mask=data.mask)
```

## Working with Metadata

### Understanding X and Y

```python
# Check design matrix (X)
if not data.X.empty:
    print(f"Design matrix shape: {data.X.shape}")
    print(f"Columns: {list(data.X.columns)[:5]}...")
    print(data.X.head())

# Y is for outcome variables (used in prediction/regression)
print(f"Y shape: {data.Y.shape}")
```

### Adding Metadata

```python
# Create outcome labels
labels = pd.DataFrame({
    'condition': ['pain'] * 42 + ['no_pain'] * 42,
    'intensity': np.random.randn(84),
    'subject_id': np.repeat(range(1, 29), 3)
})

# Create Brain_Data with metadata
labeled_data = Brain_Data(data.to_nifti(), Y=labels)
print(f"Y shape: {labeled_data.Y.shape}")
print(labeled_data.Y.head())
```

### Filtering by Metadata

```python
# Use boolean indexing with metadata
pain_images = labeled_data[labeled_data.Y['condition'] == 'pain']
print(f"Pain images: {len(pain_images)}")

no_pain_images = labeled_data[labeled_data.Y['condition'] == 'no_pain']
print(f"No-pain images: {len(no_pain_images)}")

# Compare conditions
pain_mean = pain_images.mean()
no_pain_mean = no_pain_images.mean()
difference = pain_mean - no_pain_mean
difference.plot()
```

## Common Patterns and Workflows

### Quality Control

```python
# Check for motion artifacts (framewise displacement proxy)
motion_proxy = data.std(axis=1)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(motion_proxy)
ax.axhline(y=motion_proxy.mean() + 2*motion_proxy.std(),
           color='r', linestyle='--', label='2 SD threshold')
ax.set_xlabel("Image Number")
ax.set_ylabel("Spatial SD")
ax.set_title("Motion Proxy (Spatial Variability)")
ax.legend()
plt.show()
```

### Chaining Operations

```python
# Chain multiple operations together
result = (data
          .smooth(fwhm=6)
          .standardize()
          .mean()
          .threshold(upper='95%'))
result.plot()
```

## Sanity Checks and Validation

```python
# Verify data properties
assert data.shape()[0] == 84, "Should have 84 images"
assert not np.any(np.isnan(data.data)), "Should have no NaN values"

# Verify standardization worked
z_data = data.standardize()
assert abs(z_data.mean().mean()) < 1e-6, "Z-scored data should have mean ~0"
assert abs(z_data.std().mean() - 1) < 0.1, "Z-scored data should have std ~1"
print("✓ All sanity checks passed!")
```

## Summary

In this tutorial, you learned:
- ✓ How to load data into `Brain_Data` objects
- ✓ Indexing and slicing operations
- ✓ Computing statistics across images and voxels
- ✓ Arithmetic operations on brain data
- ✓ Preprocessing (smoothing, standardization, thresholding)
- ✓ Creating and applying masks
- ✓ Visualization methods
- ✓ File I/O and format conversion
- ✓ Working with metadata (X and Y)

## Next Steps

- **Tutorial 02**: Learn about `DesignMatrix` for creating experimental designs
- **Tutorial 03**: Explore `Adjacency` for connectivity analyses
- **Workflow Tutorials**: Apply these concepts to complete analyses (GLM, MVPA, etc.)
- **API Reference**: Deep dive into all `Brain_Data` methods

## Clean Up

```python
# Remove files created during tutorial
import os
if os.path.exists('mean_activation.nii.gz'):
    os.remove('mean_activation.nii.gz')
if os.path.exists('brain_data_with_metadata.h5'):
    os.remove('brain_data_with_metadata.h5')
```
