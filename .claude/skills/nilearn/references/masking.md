# nilearn.masking — Data Masking Utilities

Low-level utilities to compute brain masks from fMRI/anatomical data, intersect masks across subjects, and apply or invert a mask to convert between 3D/4D images and 2D `(timepoints, voxels)` arrays. The maskers in `nilearn.maskers` build on these.

**Source:** https://nilearn.github.io/dev/modules/masking.html

## Inventory

### Functions
| Function | Purpose |
|---|---|
| `compute_epi_mask(epi_img, ...)` | Compute brain mask from fMRI data (intensity histogram). |
| `compute_multi_epi_mask(epi_imgs, ...)` | Compute a common EPI mask across subjects/runs. |
| `compute_brain_mask(target_img, ...)` | Compute whole-brain, GM, or WM mask via MNI template. |
| `compute_multi_brain_mask(target_imgs, ...)` | Multi-subject template mask. |
| `compute_background_mask(data_imgs, ...)` | Brain mask by guessing background from image borders. |
| `compute_multi_background_mask(data_imgs, ...)` | Multi-subject background mask. |
| `intersect_masks(mask_imgs, ...)` | Compute intersection of several masks. |
| `apply_mask(imgs, mask_img, ...)` | Extract signals from images using mask -> 2D `(t, voxels)`. |
| `unmask(X, mask_img, ...)` | Take masked data and bring it back into 3D/4D image. |

## Compute masks

```python
from nilearn import masking

# From EPI intensity histogram (good for 4D fMRI)
mask = masking.compute_epi_mask(epi_img, lower_cutoff=0.2, upper_cutoff=0.85,
                                 connected=True, opening=2, exclude_zeros=False)

# From MNI template (works for any image with reasonable affine)
mask = masking.compute_brain_mask(target_img, threshold=0.5,
                                   mask_type='whole-brain')   # 'whole-brain'|'gm'|'wm'

# Background-based (assumes border is background)
mask = masking.compute_background_mask(data_img, border_size=2,
                                        connected=True, opening=False)
```

`mask_type` for `compute_brain_mask`:
- `'whole-brain'` — full brain mask (default).
- `'gm'` — grey matter only.
- `'wm'` — white matter only.

## Multi-subject masks

```python
mask = masking.compute_multi_epi_mask(epi_imgs, threshold=0.5,
                                       lower_cutoff=0.2, upper_cutoff=0.85,
                                       connected=True, n_jobs=1)
mask = masking.compute_multi_brain_mask(target_imgs, threshold=0.5,
                                         mask_type='whole-brain', n_jobs=1)
mask = masking.compute_multi_background_mask(data_imgs, threshold=0.5,
                                              border_size=2, n_jobs=1)
```

Each computes a per-subject mask, then intersects with `threshold` (fraction of subjects required to retain a voxel).

## intersect_masks

```python
intersected = masking.intersect_masks(mask_imgs, threshold=0.5,
                                       connected=True)
```

`threshold=0.5` keeps voxels present in at least 50% of input masks; `threshold=1.0` is full intersection; `threshold=0.0` is union.

## apply_mask / unmask

```python
data_2d = masking.apply_mask(imgs, mask_img, dtype='f',
                              smoothing_fwhm=None, ensure_finite=True)
# imgs: 3D -> (n_voxels,), 4D -> (n_timepoints, n_voxels)

img = masking.unmask(data_2d, mask_img, order='F')
# data_2d 1D -> 3D Nifti1Image, 2D -> 4D Nifti1Image
```

## Common patterns

Compute and apply a mask:

```python
from nilearn import masking

mask = masking.compute_brain_mask(fmri_img, mask_type='whole-brain')
ts = masking.apply_mask(fmri_img, mask)            # (n_timepoints, n_voxels)
recon = masking.unmask(ts, mask)                    # 4D Nifti1Image
```

Group mask via intersection:

```python
group_mask = masking.intersect_masks(per_subject_masks, threshold=0.8,
                                      connected=True)
```

GM-restricted analysis:

```python
gm_mask = masking.compute_brain_mask(template_img, mask_type='gm', threshold=0.5)
```

## Gotchas

- `compute_brain_mask` resamples the MNI template to your image — quality depends on the affine being correct. For raw subject-space data without good registration, prefer `compute_epi_mask`.
- `apply_mask` returns voxels in C-order traversal of the mask; the order is preserved by `unmask` so the round-trip is lossless **only if you use the same mask**.
- `unmask` accepts `order='F'` (default) or `'C'` — must match how `apply_mask` produced the data; defaults are compatible.
- `intersect_masks(threshold=0)` is the union; `threshold=1` is strict intersection. The default `0.5` is "majority vote".
- For a smoothed extraction, prefer the masker class (`NiftiMasker(smoothing_fwhm=...)`) — passing `smoothing_fwhm` to `apply_mask` smooths after masking, which is rarely what you want.
- `compute_background_mask` assumes image borders are non-brain; fails on tightly-cropped images.

## See also

- `nilearn.maskers.NiftiMasker` — full pipeline (mask + clean + transform).
- `nilearn.datasets.load_mni152_brain_mask`, `load_mni152_gm_mask`, `load_mni152_wm_mask` — pre-computed templates.
- `nilearn.image.binarize_img`, `largest_connected_component_img`.
- https://nilearn.github.io/dev/modules/masking.html
