---
# AUTO-GENERATED from 01_brain_data.py by scripts/marimo_to_myst.py — DO NOT EDIT.
# Edit the marimo notebook, then run `uv run poe docs-generate`.
kernelspec:
  name: python3
  display_name: Python 3
---

# BrainData Basics

:::{tip} Run this tutorial locally
The outputs below were baked in at build time. This page is rendered from a [marimo](https://marimo.io) notebook — [`docs/tutorials/basics/01_brain_data.py`](https://github.com/cosanlab/nltools/blob/master/docs/tutorials/basics/01_brain_data.py) — that you can open and edit locally with `uvx marimo edit --sandbox 01_brain_data.py`.
:::

The `BrainData` class is the core data structure in `nltools` for working with
neuroimaging data. It stores data as 2D arrays (images x voxels) for efficient
computation, automatically handles resampling to standard MNI space (default),
and supports standard Python operations like indexing, arithmetic, and iteration.

```{code-cell} python3
from nltools import BrainData

# Empty brain
BrainData()
```

## Loading data

You pass a file path, a `nilearn`/`nibabel` image, a file URL, or lists of any of
those to `BrainData()` — it loads and resamples to MNI space if needed, e.g.
`BrainData('myfile.nii.gz')`.

To keep things simple we use one of the included datasets. `fetch_pain()`
downloads a pain-perception study (Chang et al., 2015): 28 subjects x 3
conditions = 84 images.

```{code-cell} python3
from nltools.datasets import fetch_pain

brains = fetch_pain()
```

The `BrainData` repr shows the shape (images x voxels) and whether metadata `polars` DataFrames (X, Y) are attached.

```{code-cell} python3
brains
```

Access the underlying data as a numpy array with the `.data` attribute:

```{code-cell} python3
brains.data.shape  # (images, voxels)
```

`BrainData` also stores metadata as `polars` DataFrames on `.X` and `.Y`:

- **X**: design matrix / covariates for modeling
- **Y**: outcome variables or labels

```{code-cell} python3
# The pain dataset ships metadata in X
brains.X.head()
```

## Saving data

`BrainData` saves as NIfTI (`.nii.gz`) or HDF5 (`.h5`). HDF5 preserves metadata
(X, Y) and masks and produces smaller files:

```python
brains.write("data.nii.gz")   # NIfTI
brains.write("data.h5")       # HDF5, with X/Y/mask/etc.
```
<!---->
## Indexing and slicing

`BrainData` supports standard Python-style indexing, and all indexing preserves
the X/Y metadata.

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

Boolean indexing filters images by computed properties:

```{code-cell} python3
# Keep the images whose global mean is above the average across images
# (illustrative boolean-mask indexing)
_global_mean = brains.mean(axis=1)
_keep = _global_mean > _global_mean.mean()
high_intensity = brains[_keep]
print(f"Images kept: {len(high_intensity)}")
```

Use `.append()` to concatenate `BrainData` objects:

```{code-cell} python3
# Append one image to another
brains[0].append(brains[1]).shape
```

## Arithmetic operations

`BrainData` supports element-wise arithmetic with scalars and other `BrainData`
objects.

```{code-cell} python3
# Addition (scalar, broadcast over every voxel)
brains + 100
```

```{code-cell} python3
# Subtraction of two images → single brain map
brains[1] - brains[0]
```

```{code-cell} python3
# Adding two BrainData objects element-wise
brains + brains
```

## Statistical operations

`BrainData` exposes many statistical methods that reduce across images
(`axis=0`) or across voxels (`axis=1`).

```{code-cell} python3
# Mean across all images → single brain map
brains.mean()
```

```{code-cell} python3
# Standard deviation across images → single brain map
brains.std()
```

```{code-cell} python3
# Temporal signal-to-noise ratio, then plot it
tsnr = brains.mean() / brains.std()
tsnr.plot()
```

```{code-cell} python3
# Standardization / z-scoring across images
z_scored = brains.standardize(method="zscore")
print(f"Z-scored mean: {z_scored.mean().data.mean():.6f}")
print(f"Z-scored std:  {z_scored.std().data.mean():.4f}")
```

```{code-cell} python3
# Gaussian spatial smoothing at a given FWHM (mm)
_smoothed = brains[0].smooth(fwhm=6)
print(f"Original range: [{brains[0].data.min():.2f}, {brains[0].data.max():.2f}]")
print(f"Smoothed range: [{_smoothed.data.min():.2f}, {_smoothed.data.max():.2f}]")
```

Threshold by absolute value or percentile, optionally binarizing for a mask:

```{code-cell} python3
# Keep only voxels in the top 5%
brains.mean().threshold(upper="95%").plot()
```

```{code-cell} python3
# Binarize for use as a mask
_binary_mask = brains.mean().threshold(upper="95%", binarize=True)
print(f"Mask voxels: {_binary_mask.data.sum():.0f}")
```

## Masking

Use `apply_mask` to restrict data to a region of interest.

```{code-cell} python3
# Mean map, with color bounds captured so later plots stay comparable
mean_brain = brains.mean()
vmin, vmax = mean_brain.data.min(), mean_brain.data.max()
mean_brain.plot(vmin=vmin, vmax=vmax)
```

```{code-cell} python3
# An ROI mask from the top 10% of mean activation
roi_mask = mean_brain.threshold(upper="90%", binarize=True)
roi_mask.plot(vmin=0, vmax=1, cmap="gray_r")
```

```{code-cell} python3
# Apply it — voxels outside the mask render transparent
masked_data = mean_brain.apply_mask(roi_mask)
masked_data.plot(vmin=vmin, vmax=vmax, cmap="RdBu_r")
```

## Visualization

`BrainData.plot()` supports several visualization types via the `method`
argument. Most wrap [`nilearn.plotting`](https://nilearn.github.io/dev/modules/plotting.html),
so you can always drop down to `BrainData.to_nifti()` and call nilearn directly.

### Glass brain (default)

```{code-cell} python3
masked_data.plot(title="Mean Activation")
```

### Slices

```{code-cell} python3
# Default: axial (z) slices only
masked_data.plot(method="slices")
```

```{code-cell} python3
# Any combination of x/y/z views, one row per axis
masked_data.plot(method="slices", view="xyz")
```

### Surface & flat-map

```{code-cell} python3
masked_data.plot_surf(zoom=1.3)
```

```{code-cell} python3
masked_data.plot_flatmap()
```

### Timeseries & voxel distribution

For multi-image `BrainData`, plot the mean signal over images; `histogram` shows
the voxel-intensity distribution.

```{code-cell} python3
brains.plot(method="timeseries", figsize=(6, 4))
```

```{code-cell} python3
mean_brain.plot(
    method="histogram", title="Voxel Intensity Distribution", figsize=(6, 4)
)
```

### Interactive viewer

`BrainData.iplot()` returns an interactive [niivue](https://niivue.com) viewer —
a WebGL `anywidget` that drives `@niivue/niivue` directly: a threshold slider
stacked above the viewer, with the stat-map colorbar shown. Drag the slider (or
right-drag on the image) to window the map live; scroll through slices, scrub 4D
frames, render in 3D, and overlay nltools atlases with hover-to-label. It speaks
anywidget's standard model API, so it renders in any live kernel (marimo, Jupyter).

Pass `controls=False` to hide the slider (right-drag windowing still works), and
`colorbar=False` to hide the colorbar. No `ipywidgets` dependency needed.

```{code-cell} python3
# Interactive niivue viewer with a threshold slider (an anywidget driving
# @niivue/niivue directly). It needs a live kernel, so on this static page
# only a placeholder appears; run the notebook to explore the volume.
masked_data.iplot()
```
