---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# BrainData Basics

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

The simplest way to get started is with a built-in dataset. `fetch_pain()` downloads a pain perception study (Chang et al. 2015) with 28 subjects x 3 conditions = 84 images.

```{code-cell} python3
data = fetch_pain()
```

The `BrainData` repr shows the shape (images x voxels), and whether metadata `polars` Dataframes (X, Y) are attached.

```{code-cell} python3
data
```

You can also create `BrainData` from NIfTI files, nibabel objects, numpy arrays, or lists of file paths. You can access the underlying numpy data with the `.data` attribute:

```{code-cell} python3
# Access the underlying numpy array
print(f"Raw data shape: {data.data.shape}")  # (images, voxels)
```

`BrainData` carries two DataFrame _attributes_:

- **X**: Design matrix / covariates for modeling
- **Y**: Outcome variables or labels

```{code-cell} python3
# The pain dataset comes with metadata in X; show just the study-specific columns
study_cols = ["SubjectID", "PainLevel", "Age", "Sex"]

data.X[study_cols].head(10)
```

## Indexing and Slicing

`BrainData` supports standard Python-style indexing. All indexing preserves metadata (X and Y dataframes).

```{code-cell} python3
# Single image
data[0]
```

```{code-cell} python3
# Slicing
first_five = data[:5]
print(f"Sliced: {first_five.shape}")
```

```{code-cell} python3
# List indexing
selected = data[[0, 10, 20, 30]]
print(f"Selected: {selected.shape}")
```

Boolean indexing lets you filter images based on computed properties:

```{code-cell} python3
global_mean = data.mean(axis=1)
threshold = global_mean * 2

high_intensity = data[global_mean > threshold]

print(f"Images above threshold: {len(high_intensity)}")
```

Combine `BrainData` objects with `append`:

```{code-cell} python3
# Append one image to another
combined = data[0].append(data[1])

combined
```

## Arithmetic Operations

`BrainData` supports element-wise arithmetic with scalars and other `BrainData` objects:

```{code-cell} python3
# Addition
data + 100
```

```{code-cell} python3
# Multiplication
data * 2
```

```{code-cell} python3
# Subtraction of first 2 time-points (images) → single brain map
data[1] - data[0]
```

```{code-cell} python3
# Addition of all images and time-points
data_doubled = data + data
data_doubled
```

## Statistical Operations

`BrainData` supports a wide variety of [statistical methods](/api/data/brain-data) that work across images (`axis=0`) or across voxels (`axis=1`):

```{code-cell} python3
# Mean across all images → single brain map
data.mean()
```

```{code-cell} python3
# Standard deviation across images → single brain map
data.std()
```

```{code-cell} python3
# Temporal signal-to-noise ratio → simple division
tsnr = data.mean() / data.std()

# Visualize it
tsnr.plot()
```

```{code-cell} python3
# Standardization/z-scoring
z_scored = data.standardize(method="zscore", verbose=False)

print(f"Z-scored mean: {z_scored.mean().data.mean():.6f}")

print(f"Z-scored std: {z_scored.std().data.mean():.4f}")
```

```{code-cell} python3
# Apply a Gaussian spatial filter with a specified FWHM (in mm):
smoothed = data[0].smooth(fwhm=6)
print(f"Original range: [{data[0].data.min():.2f}, {data[0].data.max():.2f}]")
print(f"Smoothed range: [{smoothed.data.min():.2f}, {smoothed.data.max():.2f}]")
```

Threshold by absolute value or percentile. Optionally binarize for mask creation:

```{code-cell} python3
# Keep only voxels in the top 5%
data.mean().threshold(upper="95%").plot()
```

```{code-cell} python3
# Binarize for use as a mask
binary_mask = data.mean().threshold(upper="95%", binarize=True)
print(f"Mask voxels: {binary_mask.data.sum():.0f}")
```

## Visualization

`BrainData.plot()` supports several visualization types via the `method` parameter. Most are just convenience wrappers around [`nilearn.plotting`](https://nilearn.github.io/dev/modules/plotting.html#module-nilearn.plotting), which means you can always use `BrainData.to_nifti()` to work directly with `nilearn`'s plotting functions:

### Glass Brain (default)

```{code-cell} python3
mean_brain = data.mean()

mean_brain.plot(title="Mean Activation")
```

### Timeseries

For multi-image `BrainData`, plot the mean signal over time:

```{code-cell} python3
data.plot(method="timeseries", figsize=(6,4));
```

### Voxel Distribution

```{code-cell} python3
mean_brain.plot(method="histogram", title="Voxel Intensity Distribution", figsize=(6,4));
```

<!--TODO: Update this section with included ROI masks-->

## Masking

Use `apply_mask` to restrict your data to a region of interest:

```{code-cell} python3
# Original data
# Get original data bounds to keep color bars consistent
vmin, vmax = mean_brain.data.min(), mean_brain.data.max()

# Plot it
mean_brain.plot(vmin=vmin, vmax=vmax)
```

```{code-cell} python3
# Create a mask from the top 10% of mean activation
roi_mask = mean_brain.threshold(upper="90%", binarize=True)

# Plot the mask
roi_mask.plot(vmin=0, vmax=1, cmap='gray_r')
```

```{code-cell} python3
# Apply it
masked_data = mean_brain.apply_mask(roi_mask)

# Plot it
masked_data.plot(vmin=vmin, vmax=vmax, cmap="RdBu_r", threshold=0)
```

## File I/O

`BrainData` can be saved as NIfTI (`.nii.gz`) or HDF5 (`.h5`). HDF5 preserves metadata (X, Y), masks, and produces smaller file sizes:

```{code-cell} python3
# Save nifti
data.write("data.nii.gz")
```

```{code-cell} python3
# Save hdf5, with X, Y, mask, etc
data.write("data.h5")
```
