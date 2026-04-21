---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# BrainData Basics

## Learning Objectives

By the end of this tutorial, you will be able to:
- Load neuroimaging data into `BrainData` objects
- Perform basic operations (indexing, slicing, arithmetic)
- Compute summary statistics across images and voxels
- Apply common preprocessing steps (smoothing, standardization)
- Visualize brain images and timeseries
- Save and load data in different formats
- Work with masks and metadata

## Introduction

The `BrainData` class is the core data structure in nltools for working with neuroimaging data. It stores data as 2D arrays (images x voxels) for efficient computation, automatically handles resampling to standard MNI space, and supports standard Python operations like indexing, arithmetic, and iteration.

```{code-cell} python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltools.data import BrainData
from nltools.datasets import fetch_pain
from nltools.utils import concatenate
```

## Loading Data

The simplest way to get started is with a built-in dataset. `fetch_pain` downloads a pain perception study (Chang et al. 2015) with 28 subjects x 3 conditions = 84 images.

```{code-cell} python3
data = fetch_pain()
print(data)
```

The `BrainData` repr shows the shape (images x voxels), and whether metadata (X, Y) is attached.

You can also create `BrainData` from NIfTI files, nibabel objects, numpy arrays, or lists of file paths:

```{code-cell} python3
# From a nibabel Nifti1Image
nifti_img = data[0].to_nifti()
from_nifti = BrainData(nifti_img)
print(f"From nibabel: {from_nifti.shape}")

# Access the underlying numpy array
print(f"Raw data shape: {data.data.shape}")  # (images, voxels)
print(f"Data type: {data.data.dtype}")
```

## Indexing and Slicing

`BrainData` supports standard Python indexing. All indexing preserves metadata (X and Y dataframes).

```{code-cell} python3
# Single image
first_image = data[0]
print(f"Single image: {first_image.shape}")

# Slicing
first_five = data[:5]
print(f"Sliced: {first_five.shape}")

# List indexing
selected = data[[0, 10, 20, 30]]
print(f"Selected: {selected.shape}")
```

Boolean indexing lets you filter images based on computed properties:

```{code-cell} python3
global_mean = data.mean(axis=1)
threshold = np.median(global_mean)
high_intensity = data[global_mean > threshold]
print(f"Images above median intensity: {len(high_intensity)}")
```

## Basic Statistics

Compute statistics across images (`axis=0`, the default) or within images across voxels (`axis=1`):

```{code-cell} python3
# Mean across all images → single brain map
mean_brain = data.mean()
print(f"Mean brain: {mean_brain.shape}")

# Standard deviation across images
std_brain = data.std()

# Temporal signal-to-noise ratio
tsnr = mean_brain.data / std_brain.data
print(f"Median tSNR: {np.nanmedian(tsnr):.2f}")
```

Statistics within each image give you a timeseries:

```{code-cell} python3
# Global signal: mean intensity per image
global_signal = data.mean(axis=1)

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(global_signal)
ax.set_xlabel("Image Number")
ax.set_ylabel("Mean Intensity")
ax.set_title("Global Signal Across Images")
plt.tight_layout()
plt.show()
```

## Arithmetic Operations

`BrainData` supports element-wise arithmetic with scalars and other `BrainData` objects:

```{code-cell} python3
# Scalar operations
scaled = data * 2
shifted = data + 100

# Mean-centering
centered = data - data.mean()
print(f"Mean of centered data: {centered.mean().data.mean():.10f}")

# Difference between images
difference = data[1] - data[0]
print(f"Difference map: {difference.shape}")
```

## Preprocessing

### Standardization

`standardize()` operates per-voxel across images by default. Use `method='zscore'` for full z-scoring (subtract mean and divide by std), or `method='center'` (default) for mean-centering only.

```{code-cell} python3
z_scored = data.standardize(method="zscore", verbose=False)
print(f"Z-scored mean: {z_scored.mean().data.mean():.6f}")
print(f"Z-scored std: {z_scored.std().data.mean():.4f}")
```

### Spatial Smoothing

Apply a Gaussian spatial filter with a specified FWHM (in mm):

```{code-cell} python3
smoothed = data[0].smooth(fwhm=6)
print(f"Original range: [{data[0].data.min():.2f}, {data[0].data.max():.2f}]")
print(f"Smoothed range: [{smoothed.data.min():.2f}, {smoothed.data.max():.2f}]")
```

### Thresholding

Threshold by absolute value or percentile. Optionally binarize for mask creation:

```{code-cell} python3
# Keep only voxels in the top 5%
top_5 = mean_brain.threshold(upper="95%")
print(f"Voxels in top 5%: {(top_5.data != 0).sum()}")

# Binarize for use as a mask
binary_mask = mean_brain.threshold(upper="95%", binarize=True)
print(f"Mask voxels: {binary_mask.data.sum():.0f}")
```

### Chaining Operations

Preprocessing methods return new `BrainData` objects, so you can chain them:

```{code-cell} python3
result = data.smooth(fwhm=6).standardize(method="zscore", verbose=False).mean().threshold(upper="95%")
print(f"Chained result: {result.shape}")
```

## Visualization

`BrainData.plot()` supports several visualization types via the `method` parameter.

### Glass Brain (default)

```{code-cell} python3
mean_brain.plot(title="Mean Activation")
```

### Timeseries

For multi-image `BrainData`, plot the mean signal over time:

```{code-cell} python3
data.plot(method="timeseries")
```

### Voxel Distribution

```{code-cell} python3
mean_brain.plot(method="histogram", title="Voxel Intensity Distribution")
```

## Working with Masks

### Applying Masks

Use `apply_mask` to restrict your data to a region of interest:

```{code-cell} python3
# Create a mask from the top 10% of mean activation
roi_mask = mean_brain.threshold(upper="90%", binarize=True)
masked_data = data.apply_mask(roi_mask)
print(f"Original: {data.shape}")
print(f"After masking: {masked_data.shape}")
```

## File I/O

`BrainData` can be saved as NIfTI (`.nii.gz`) or HDF5 (`.h5`). HDF5 preserves metadata (X, Y).

```{code-cell} python3
import os
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    # Save and reload NIfTI
    nifti_path = os.path.join(tmpdir, "mean_brain.nii.gz")
    mean_brain.write(nifti_path)
    loaded = BrainData(nifti_path, mask=data.mask)
    print(f"Loaded NIfTI: {loaded.shape}")
    assert np.allclose(mean_brain.data, loaded.data, rtol=1e-5)
    print("NIfTI round-trip verified")
```

HDF5 (`.h5`) format additionally preserves metadata (X, Y DataFrames). It requires the optional `pytables` dependency:

```python
# Save with metadata (requires pytables)
data.write("data.h5")
loaded = BrainData("data.h5")
```

## Metadata (X and Y)

`BrainData` carries two metadata DataFrames:
- **X**: Design matrix / covariates for modeling
- **Y**: Outcome variables or labels

```{code-cell} python3
# The pain dataset comes with metadata in X
# Show just the study-specific columns
study_cols = ["SubjectID", "PainLevel", "Age", "Sex"]
print(data.X[study_cols].head(10))
```

## Concatenation

Combine `BrainData` objects with `append`:

```{code-cell} python3
# Append one image to another
combined = data[0].append(data[1])
```

## Summary

In this tutorial you learned the core `BrainData` operations:
- **Loading**: from files, nibabel objects, or built-in datasets
- **Indexing**: integer, slice, list, and boolean indexing
- **Statistics**: `mean()`, `std()`, `sum()` with axis control
- **Arithmetic**: element-wise operations between objects and scalars
- **Preprocessing**: `standardize()`, `smooth()`, `threshold()`
- **Visualization**: `plot()` with glass brain, timeseries, and histogram views
- **Masks**: `apply_mask()` and `threshold(binarize=True)` for ROI analysis
- **I/O**: NIfTI and HDF5 read/write
- **Metadata**: X and Y DataFrames for experimental designs

Next, explore [DesignMatrix](02_design_matrix.md) for building experimental design matrices, or [Adjacency](03_adjacency.md) for connectivity analysis.
