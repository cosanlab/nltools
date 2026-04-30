# nilearn.image — Image Processing & Resampling

Mathematical operations on Niimg-like objects (a 3D or 4D block of data plus an affine). Covers loading/creation, validation, spatial resampling and smoothing, multi-image manipulation, math, thresholding, and temporal cleaning.

**Source:** https://nilearn.github.io/dev/modules/image.html

## Function inventory

| Function | Purpose |
|---|---|
| `binarize_img(img, threshold, mask_img, ...)` | Binarize an image so values are either 0 or 1 |
| `check_niimg(niimg, ensure_ndim, ...)` | Check that `niimg` is a proper 3D/4D Niimg-like and load it |
| `check_niimg_3d(niimg, dtype)` | Check / load as a 3D Niimg-like |
| `check_niimg_4d(niimg, return_iterator, dtype)` | Check / load as a 4D Niimg-like |
| `clean_img(imgs, runs, detrend, ...)` | Improve SNR on (masked) 4D fMRI signals (detrend, filter, confounds) |
| `concat_imgs(niimgs, dtype, ensure_ndim, ...)` | Concatenate a list of images of varying lengths into a single 4D image |
| `coord_transform(x, y, z, affine)` | Convert `(x, y, z)` coordinates from one image space to another via an affine |
| `copy_img(img)` | Deep-copy an image into a new `nibabel.Nifti1Image` |
| `crop_img(img, rtol, copy, pad, ...)` | Crop an image as much as possible to its non-zero bounding box |
| `get_data(img)` | Get the image data as a `numpy.ndarray` (returns a copy) |
| `high_variance_confounds(imgs, n_confounds, ...)` | CompCor-like confounds: extract regressors from voxels with highest variance |
| `index_img(imgs, index)` | Index into an image along its last dimension |
| `iter_img(imgs)` | Iterate over the volumes of a 4D image |
| `largest_connected_component_img(imgs)` | Return only the largest connected component of an image (or each image in a list) |
| `load_img(img, wildcards, dtype)` | Load a Niimg-like from a filename or list of filenames |
| `math_img(formula, copy_header_from, **imgs)` | Evaluate a numpy-based string formula using Niimg arguments |
| `mean_img(imgs, target_affine, ...)` | Compute the mean over images (temporal mean for 4D) |
| `new_img_like(ref_niimg, data, affine, ...)` | Create a new image of the same class/affine/header as a reference |
| `resample_img(img, target_affine, ...)` | Resample a Niimg-like to a target affine and/or shape |
| `resample_to_img(source_img, target_img, ...)` | Resample a source image to match a target image's space |
| `reorder_img(img, resample, copy_header)` | Reorder axes so the affine has a positive diagonal (canonical orientation) |
| `smooth_img(imgs, fwhm)` | Apply Gaussian smoothing (FWHM in mm) |
| `swap_img_hemispheres(img)` | Swap left and right hemispheres of a NIfTI image |
| `threshold_img(img, threshold, ...)` | Threshold an image (statistical or atlas) and optionally cluster-filter |

## Loading & creation

```python
from nilearn import image

img = image.load_img('path.nii.gz')             # load any Niimg-like (str/Path/Nifti1Image/4D list)
img = image.load_img('sub-*/run-*/bold.nii.gz', wildcards=True)  # glob pattern
img = image.new_img_like(ref_img, data)         # new image with ref's affine + header
img = image.copy_img(ref_img)                   # deep copy as Nifti1Image

data = image.get_data(img)                      # numpy.ndarray (copy)

image.check_niimg(img, ensure_ndim=3)           # validate generic
image.check_niimg_3d(img)                       # validate as 3D
image.check_niimg_4d(img)                       # validate as 4D
```

## Spatial operations

```python
# Resample to a given affine and/or shape
img = image.resample_img(
    img,
    target_affine=aff,
    target_shape=shape,
    interpolation='continuous',     # 'continuous'|'linear'|'nearest'
    force_resample=True,            # default in v0.13+
    copy_header=True,
)

# Resample to match another image's space
img = image.resample_to_img(source, target, interpolation='nearest')

# Gaussian smoothing
img = image.smooth_img(img, fwhm=6)             # mm; scalar or per-axis tuple

# Crop to non-zero bounding box
img = image.crop_img(img, pad=True)

# Reorder axes so affine diagonal is positive (canonical)
img = image.reorder_img(img)

# Hemisphere swap
img = image.swap_img_hemispheres(img)
```

## Multi-image operations

```python
img_4d = image.concat_imgs(list_of_3d)          # 3D list -> 4D
vol = image.index_img(img_4d, 5)                # extract volume 5 (3D)
sub = image.index_img(img_4d, slice(10, 30))    # subvolume (4D)
mean = image.mean_img(img_4d)                   # temporal mean
for vol in image.iter_img(img_4d):              # generator of 3D volumes
    ...
```

## Math & thresholding

```python
# Element-wise math; formula is a string evaluated with numpy as `np`
result = image.math_img("img1 + img2", img1=a, img2=b)
result = image.math_img("np.mean(img, axis=-1)", img=four_d)
result = image.math_img("np.where(img > 3, img, 0)", img=stat)

# Threshold (optionally with cluster size filter)
thresh = image.threshold_img(
    img,
    threshold=3.0,                  # scalar or '95%' string for percentile
    cluster_threshold=10,           # min cluster size in voxels
    two_sided=True,
    mask_img=None,
    copy=True,
)

# Binarize at threshold
binary = image.binarize_img(img, threshold=0.5, mask_img=mask)

# Keep only the largest connected component
largest = image.largest_connected_component_img(img)
```

## Signal processing on images

```python
# Temporal cleaning of 4D images (detrend / filter / confound regression)
cleaned = image.clean_img(
    img_4d,
    runs=None,
    detrend=True,
    standardize='zscore_sample',
    confounds=conf,                 # ndarray, DataFrame, or path
    low_pass=0.1,
    high_pass=0.01,
    t_r=2.0,
    mask_img=mask,
)

# CompCor-like nuisance regressors (high-variance voxels)
confounds = image.high_variance_confounds(
    img_4d,
    n_confounds=5,
    percentile=2.0,
    detrend=True,
)
```

## Coordinates

```python
import numpy as np

# Voxel -> world (e.g. MNI) using the image's affine
mni_x, mni_y, mni_z = image.coord_transform(i, j, k, img.affine)

# World -> voxel via the inverse affine
inv = np.linalg.inv(img.affine)
i, j, k = image.coord_transform(mni_x, mni_y, mni_z, inv)
```

## Gotchas
- `get_data(img)` returns a copy — for read-only access prefer `img.get_fdata()` (nibabel) directly.
- Use interpolation `'nearest'` for discrete labels / atlases, `'continuous'` (or `'linear'`) for continuous maps.
- `clean_img` takes 4D NIfTI; for 2D arrays use `signal.clean` (see `references/signal.md`).
- Processing order in `clean_img` / `signal.clean`: detrend -> filter -> confound removal -> standardize. Filters are also applied to confounds.
- In v0.13+, `resample_img` defaults to `force_resample=True`; pass `force_resample=False` to skip a no-op resample.
- `threshold_img` and `binarize_img` accept percentile strings like `'95%'` for `threshold=`.
- `math_img` formulas evaluate `np` and any kwargs you pass as Niimg-like — broadcasting follows numpy rules; all input images must share shape/affine.
- `concat_imgs` enforces a common shape and affine across the list; mixed grids must be resampled first.
- `index_img` accepts an int, slice, list, or boolean mask along the last axis; returns a 3D image only when given a single int.
- `mean_img` accepts `target_affine`/`target_shape` so it can resample on the way out.
- `coord_transform` operates in homogeneous coordinates; pass the *inverse* affine to go world->voxel.
- `crop_img` crops to the non-zero bounding box of the data — supply `pad=True` to keep a 1-voxel margin.

## See also
- `references/signal.md` — `signal.clean` for 2D arrays, `butterworth`
- `references/masking.md` — `apply_mask` / `unmask`, `compute_*_mask`
- `references/glm.md` — `threshold_stats_img` for statistical thresholding with multiple-comparison control
- https://nilearn.github.io/dev/modules/image.html
