---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# BrainData Basics

## Introduction

The `BrainData` class is the core data structure in `nltools` for working with neuroimaging data. It stores data as 2D arrays (images x voxels) for efficient computation, automatically handles resampling to standard MNI space (default), and supports standard Python operations like indexing, arithmetic, and iteration.

```{code-cell} python3
from nltools import BrainData

# Empty brain
BrainData()
```

## Loading Data

You pass a file path, `nilearn/nibabel` nifti image, file URL, or lists of any those to `BrainData()` to load and automatically resample to MNI space if needed: `BrainData('myfile.nii.gz')`. 

To keep things simple, we can use one of the included datasets. `fetch_pain()` downloads a pain perception study (Chang et al. 2015) with 28 subjects x 3 conditions = 84 images.

```{code-cell} python3
from nltools.datasets import fetch_pain
brains = fetch_pain()
```

The `BrainData` repr shows the shape (images x voxels), and whether metadata `polars` Dataframes (X, Y) are attached.

```{code-cell} python3
brains
```

And we can always access the underlying *data* as a numpy array using the `.data` attribute:

```{code-cell} python3
# Access the underlying numpy array
brains.data.shape # (images, voxels)
```

In addition, `BrainData` stores additional meta-data as `polars` DataFrames using `.X` and `.Y`:

- **X**: Design matrix / covariates for modeling
- **Y**: Outcome variables or labels

```{code-cell} python3
# The pain dataset comes with metadata in X
brains.X.head()
```

## Saving Data

`BrainData` can be saved as NIfTI (`.nii.gz`) or HDF5 (`.h5`). HDF5 preserves metadata (X, Y), masks, and produces smaller file sizes:

```{code-cell} python3
# Save nifti
# brains.write("data.nii.gz")
```

```{code-cell} python3
# Save hdf5, with X, Y, mask, etc
# brains.write("data.h5")
```

## Indexing and Slicing

`BrainData` supports standard Python-style indexing. All indexing preserves metadata (X and Y dataframes).

```{code-cell} python3
# Single image
brains[0]
```

```{code-cell} python3
# Slicing
first_five = brains[:5]
print(f"Sliced: {first_five.shape}")
```

```{code-cell} python3
# List indexing
selected = brains[[0, 10, 20, 30]]
print(f"Selected: {selected.shape}")
```

Boolean indexing lets you filter images based on computed properties:

```{code-cell} python3
# Create a boolean mask of extreme values
global_mean = brains.mean(axis=1)
threshold = global_mean * 2
mask = global_mean > threshold

# Apply it
high_intensity = brains[mask]
print(f"Images above threshold: {len(high_intensity)}")
```

You can use `.append()` to concatenate `BrainData` objects:

```{code-cell} python3
# Append one image to another
img1 = brains[0]
img2 = brains[1]

# Combined
img1.append(img2).shape
```

## Arithmetic Operations

`BrainData` supports element-wise arithmetic with scalars and other `BrainData` objects:

```{code-cell} python3
# Addition
brains + 100
```

```{code-cell} python3
# Multiplication
brains * 2
```

```{code-cell} python3
# Subtraction of first 2 time-points (images) → single brain map
brains[1] - brains[0]
```

```{code-cell} python3
# Addition of all images and time-points
data_doubled = brains + brains
data_doubled
```

## Statistical Operations

`BrainData` supports a wide variety of [statistical methods](/api/data/brain-data) that work across images (`axis=0`) or across voxels (`axis=1`):

```{code-cell} python3
# Mean across all images → single brain map
brains.mean()
```

```{code-cell} python3
# Standard deviation across images → single brain map
brains.std()
```

```{code-cell} python3
# Temporal signal-to-noise ratio → simple division
tsnr = brains.mean() / brains.std()

# Visualize it
tsnr.plot()
```

```{code-cell} python3
# Standardization/z-scoring
z_scored = brains.standardize(method="zscore", verbose=False)

print(f"Z-scored mean: {z_scored.mean().data.mean():.6f}")

print(f"Z-scored std: {z_scored.std().data.mean():.4f}")
```

```{code-cell} python3
# Apply a Gaussian spatial filter with a specified FWHM (in mm):
smoothed = brains[0].smooth(fwhm=6)
print(f"Original range: [{brains[0].data.min():.2f}, {brains[0].data.max():.2f}]")
print(f"Smoothed range: [{smoothed.data.min():.2f}, {smoothed.data.max():.2f}]")
```

Threshold by absolute value or percentile. Optionally binarize for mask creation:

```{code-cell} python3
# Keep only voxels in the top 5%
brains.mean().threshold(upper="95%").plot()
```

```{code-cell} python3
# Binarize for use as a mask
binary_mask = brains.mean().threshold(upper="95%", binarize=True)
print(f"Mask voxels: {binary_mask.data.sum():.0f}")
```

<!--TODO: Update this section with included ROI masks-->

## Masking

Use `apply_mask` to restrict your data to a region of interest:

```{code-cell} python3
# Original data
mean_brain = brains.mean()

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

# Plot it — voxels outside the mask render transparent
masked_data.plot(vmin=vmin, vmax=vmax, cmap="RdBu_r")
```

## Visualization

`BrainData.plot()` supports several visualization types via the `method` parameter. Most are just convenience wrappers around [`nilearn.plotting`](https://nilearn.github.io/dev/modules/plotting.html#module-nilearn.plotting), which means you can always use `BrainData.to_nifti()` to work directly with `nilearn`'s plotting functions:

### Glass Brain (default)

```{code-cell} python3
masked_data.plot(title="Mean Activation")
```

### Slices

```{code-cell} python3
# Default all views
masked_data.plot(method="slices")
```

```{code-cell} python3
# Only Z
masked_data.plot(method="slices", view="z")
```

```{code-cell} python3
# Only X & Y
masked_data.plot(method="slices", view="xy")
```

### Surface

```{code-cell} python3
masked_data.plot_surf(zoom=1.3);
```

### Flat-map

```{code-cell} python3
masked_data.plot_flatmap();
```

### Timeseries

For multi-image `BrainData`, plot the mean signal over time:

```{code-cell} python3
brains.plot(method="timeseries", figsize=(6,4));
```

### Voxel Distribution

```{code-cell} python3
mean_brain.plot(method="histogram", title="Voxel Intensity Distribution", figsize=(6,4));
```

### Interactive Viewer

`BrainData.iplot()` returns an interactive [niivue](https://niivue.com) viewer — a WebGL [`ipyniivue.NiiVue`](https://github.com/niivue/ipyniivue) widget. Right-drag to window the stat map live, scroll to move through slices, scrub 4D frames natively, render in 3D, and overlay nltools atlases with hover-to-label. It renders in a **live kernel** (Jupyter, marimo); for static figures use `plot()` (above). The snippets below are not executed when this page is built.

```python
# 3D ortho viewer — right-drag to window the stat map live
masked_data.iplot()
```

Set the initial threshold window via kwargs. `threshold=` is a symmetric magnitude floor (sub-threshold voxels become transparent); `lower=`/`upper=` set explicit divergent window endpoints (negative limb uses a cool colormap, positive limb a warm one):

```python
import numpy as np
# Floor the window at the 95th percentile of |x| (the "top 5%")
p95 = float(np.percentile(np.abs(masked_data.data), 95))
masked_data.iplot(threshold=p95)

# Explicit divergent window: tighter on the negative side, looser on positives
masked_data.iplot(lower=-0.4, upper=0.2)
```

```python
# 4D: same call — scrub frames with niivue's native 4D controls
brains[:10].iplot()
```

```python
# 3D volume render (replaces the old view="surface"; for a cortical mesh use plot_surf)
masked_data.iplot(view="render")
```

```python
# Overlay a deterministic atlas: colored regions with hover labels.
# outline=2 draws region boundaries so the stat map stays visible.
masked_data.iplot(atlas="aal", outline=2)
```
