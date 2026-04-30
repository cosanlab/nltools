# nilearn.maskers — Extracting Signals from Brain Images

The `nilearn.maskers` module contains masker objects that follow the scikit-learn transformer API. Maskers convert 3D/4D NIfTI or surface images into 2D `(n_samples, n_features)` arrays suitable for machine learning, and back. They also wrap signal cleaning (detrend, filter, standardize, confound removal) so a single object handles extraction and preprocessing.

**Source:** https://nilearn.github.io/dev/modules/maskers.html

## When to use

- Voxel-level extraction with a brain mask: `NiftiMasker`
- ROI averaging from a discrete (integer) parcellation: `NiftiLabelsMasker`
- Soft/probabilistic component extraction (ICA, DiFuMo): `NiftiMapsMasker`
- Seed-based extraction at MNI coordinates: `NiftiSpheresMasker`
- Multiple subjects in parallel (same masker, list of images): `Multi*` variants
- Surface-based vertex/region extraction: `Surface*` variants

## Inventory

### Classes

| Class | Purpose |
|---|---|
| `BaseMasker` | Base class for NiftiMaskers. |
| `NiftiMasker` | Applying a mask to extract time-series from Niimg-like objects. |
| `MultiNiftiMasker` | Applying a mask to extract time-series from multiple Niimg-like objects. |
| `NiftiLabelsMasker` | Class for extracting data from Niimg-like objects using labels of non-overlapping brain regions. |
| `MultiNiftiLabelsMasker` | Class for extracting data from multiple Niimg-like objects using labels of non-overlapping brain regions. |
| `NiftiMapsMasker` | Class for extracting data from Niimg-like objects using maps of potentially overlapping brain regions. |
| `MultiNiftiMapsMasker` | Class for extracting data from multiple Niimg-like objects using maps of potentially overlapping brain regions. |
| `NiftiSpheresMasker` | Class for masking of Niimg-like objects using seeds. |
| `SurfaceMasker` | Extract data from a SurfaceImage. |
| `MultiSurfaceMasker` | Extract time-series from multiple SurfaceImage objects. |
| `SurfaceLabelsMasker` | Extract data from a SurfaceImage, averaging over atlas regions. |
| `MultiSurfaceLabelsMasker` | Extract time-series from multiple SurfaceImage objects. |
| `SurfaceMapsMasker` | Extract data from a SurfaceImage, using maps of potentially overlapping brain regions. |
| `MultiSurfaceMapsMasker` | Extract time-series from multiple SurfaceImage objects. |

## Choosing the Right Masker

| Masker | Input | Use Case |
|--------|-------|----------|
| `NiftiMasker` | 3D/4D NIfTI | Whole-brain voxel-level (decoding, searchlight, RSA) |
| `MultiNiftiMasker` | List of 4D NIfTI | Voxel-level across many subjects, parallel |
| `NiftiLabelsMasker` | 3D integer atlas | ROI-based analysis with discrete parcellation |
| `MultiNiftiLabelsMasker` | List + atlas | ROI extraction across many subjects |
| `NiftiMapsMasker` | 4D probabilistic maps | ICA components, soft/probabilistic atlases |
| `MultiNiftiMapsMasker` | List + maps | Probabilistic extraction across subjects |
| `NiftiSpheresMasker` | MNI coordinates | Seed-based connectivity, literature ROIs |
| `SurfaceMasker` | `SurfaceImage` | Surface-based vertex-level analysis |
| `MultiSurfaceMasker` | List of `SurfaceImage` | Surface vertex-level across subjects |
| `SurfaceLabelsMasker` | Surface parcellation | Surface ROI extraction |
| `MultiSurfaceLabelsMasker` | List + surface labels | Surface ROI across subjects |
| `SurfaceMapsMasker` | Surface probability maps | Surface probabilistic atlas extraction |
| `MultiSurfaceMapsMasker` | List + surface maps | Surface probabilistic across subjects |

## Shared API (all maskers)

```python
masker.fit(imgs=None, y=None)
arr = masker.transform(imgs, confounds=None, sample_mask=None)
arr = masker.fit_transform(imgs, y=None, confounds=None, sample_mask=None)
img = masker.inverse_transform(arr)            # 2D -> Nifti1Image / SurfaceImage
report = masker.generate_report()              # HTMLReport; .save_as_html('out.html')
```

`transform` returns shape `(n_timepoints, n_features)` for a single 4D image, or `(n_features,)` for a 3D image. `Multi*` maskers return a list of arrays.

### Cross-cutting signal-cleaning parameters

These appear on every masker and are passed to `nilearn.signal.clean` internally:

```
smoothing_fwhm    # float, mm Gaussian smoothing (Nifti only)
standardize       # 'zscore_sample' | 'zscore' | 'psc' | True | False
standardize_confounds  # bool, default True
detrend           # bool, linear detrend before filtering
high_pass         # Hz
low_pass          # Hz
t_r               # seconds (required if filtering)
clean_args        # dict of extra kwargs passed to signal.clean
```

Processing order: detrend -> filter -> confound removal -> standardize. Filters are also applied to confounds.

## NiftiMasker

```python
NiftiMasker(
    mask_img=None,              # pre-computed mask, or None to auto-compute
    runs=None,                  # per-volume run labels (cleaning per run)
    smoothing_fwhm=None,
    standardize=False,          # 'zscore_sample'|'zscore'|'psc'|True|False
    standardize_confounds=True,
    detrend=False,
    high_variance_confounds=False,
    low_pass=None,
    high_pass=None,
    t_r=None,
    target_affine=None,         # 3x3 or 4x4 resampling target
    target_shape=None,
    mask_strategy='background', # 'background'|'epi'|'whole-brain-template'|'gm-template'|'wm-template'
    mask_args=None,             # dict of kwargs for the chosen mask strategy
    dtype=None,
    memory=None,                # joblib.Memory or path string for caching
    memory_level=1,
    verbose=0,
    reports=True,
    cmap='CMRmap_r',
    clean_args=None,
)
```

Post-fit attributes: `mask_img_`, `affine_`, `n_elements_`.

Key methods: `fit`, `transform`, `fit_transform`, `inverse_transform`, `generate_report`.

## MultiNiftiMasker

Same parameters as `NiftiMasker` plus `n_jobs=1` for parallel processing across subjects. Accepts a list of 4D NIfTI images; returns a list of 2D arrays.

## NiftiLabelsMasker

```python
NiftiLabelsMasker(
    labels_img=None,            # 3D integer label atlas
    labels=None,                # list of region names (excluding background)
    background_label=0,
    lut=None,                   # DataFrame lookup table mapping integers to names
    mask_img=None,
    smoothing_fwhm=None,
    standardize=False,
    standardize_confounds=True,
    detrend=False,
    low_pass=None,
    high_pass=None,
    t_r=None,
    dtype=None,
    resampling_target='data',   # 'data'|'labels'|None
    memory=None,
    memory_level=1,
    verbose=0,
    strategy='mean',            # 'mean'|'median'|'sum'|'minimum'|'maximum'|'variance'|'standard_deviation'
    reports=True,
    cmap='CMRmap_r',
    clean_args=None,
)
```

Post-fit attributes: `labels_img_`, `labels_`, `region_ids_`, `region_names_`, `n_elements_`.

## MultiNiftiLabelsMasker

`NiftiLabelsMasker` parameters plus `n_jobs=1`. List input -> list output.

## NiftiMapsMasker

```python
NiftiMapsMasker(
    maps_img=None,              # 4D continuous/probabilistic maps
    mask_img=None,
    allow_overlap=True,
    smoothing_fwhm=None,
    standardize=False,
    standardize_confounds=True,
    detrend=False,
    low_pass=None,
    high_pass=None,
    t_r=None,
    dtype=None,
    resampling_target='data',   # 'data'|'mask'|'maps'|None
    memory=None,
    memory_level=1,
    verbose=0,
    reports=True,
    cmap='CMRmap_r',
    clean_args=None,
)
```

Post-fit attributes: `maps_img_`, `n_elements_`.

## MultiNiftiMapsMasker

`NiftiMapsMasker` parameters plus `n_jobs=1`. List input -> list output.

## NiftiSpheresMasker

```python
NiftiSpheresMasker(
    seeds=None,                 # list of (x, y, z) MNI coordinates
    radius=None,                # mm; None = single-voxel
    mask_img=None,
    allow_overlap=False,
    smoothing_fwhm=None,
    standardize=False,
    standardize_confounds=True,
    detrend=False,
    low_pass=None,
    high_pass=None,
    t_r=None,
    dtype=None,
    memory=None,
    memory_level=1,
    verbose=0,
    reports=True,
    clean_args=None,
)
```

Post-fit attributes: `seeds_`, `n_elements_`.

## SurfaceMasker

```python
SurfaceMasker(
    mask_img=None,              # SurfaceImage mask, or None
    smoothing_fwhm=None,
    standardize=False,
    standardize_confounds=True,
    detrend=False,
    low_pass=None,
    high_pass=None,
    t_r=None,
    memory=None,
    memory_level=1,
    verbose=0,
    reports=True,
    cmap='inferno',
    clean_args=None,
)
```

Post-fit attributes: `mask_img_`, `output_dimension_`, `n_elements_`.

## MultiSurfaceMasker

`SurfaceMasker` parameters plus `n_jobs=1`. Accepts a list of `SurfaceImage`s.

## SurfaceLabelsMasker

```python
SurfaceLabelsMasker(
    labels_img=None,            # SurfaceImage with integer labels per vertex
    labels=None,                # list of region names
    background_label=0,
    lut=None,
    mask_img=None,
    smoothing_fwhm=None,
    standardize=False,
    standardize_confounds=True,
    detrend=False,
    low_pass=None,
    high_pass=None,
    t_r=None,
    memory=None,
    memory_level=1,
    verbose=0,
    strategy='mean',
    reports=True,
    cmap='inferno',
    clean_args=None,
)
```

Post-fit attributes: `labels_`, `label_names_`, `n_elements_`.

## MultiSurfaceLabelsMasker

`SurfaceLabelsMasker` parameters plus `n_jobs=1`.

## SurfaceMapsMasker

```python
SurfaceMapsMasker(
    maps_img=None,              # SurfaceImage with continuous maps per vertex
    mask_img=None,
    allow_overlap=True,
    smoothing_fwhm=None,
    standardize=False,
    standardize_confounds=True,
    detrend=False,
    low_pass=None,
    high_pass=None,
    t_r=None,
    memory=None,
    memory_level=1,
    verbose=0,
    reports=True,
    cmap='inferno',
    clean_args=None,
)
```

Post-fit attributes: `n_elements_`.

## MultiSurfaceMapsMasker

`SurfaceMapsMasker` parameters plus `n_jobs=1`.

## Common patterns

### Voxel-level extraction with auto-computed mask

```python
from nilearn.maskers import NiftiMasker

masker = NiftiMasker(
    mask_strategy='epi',
    standardize='zscore_sample',
    detrend=True,
    high_pass=0.01,
    t_r=2.0,
    memory='nilearn_cache',
    memory_level=1,
)
X = masker.fit_transform(fmri_img)         # (n_timepoints, n_voxels)
back = masker.inverse_transform(X)         # back to 4D Nifti1Image
masker.generate_report().save_as_html('mask_report.html')
```

### ROI atlas extraction with confounds

```python
from nilearn.maskers import NiftiLabelsMasker
from nilearn.datasets import fetch_atlas_schaefer_2018

atlas = fetch_atlas_schaefer_2018(n_rois=200)
masker = NiftiLabelsMasker(
    labels_img=atlas.maps,
    labels=atlas.labels,
    standardize='zscore_sample',
    memory='nilearn_cache',
)
ts = masker.fit_transform(fmri_img, confounds=conf_df)
```

### Seed-based connectivity

```python
from nilearn.maskers import NiftiSpheresMasker

pcc = NiftiSpheresMasker(
    seeds=[(0, -52, 26)], radius=8,
    standardize='zscore_sample', t_r=2.0,
    high_pass=0.01, low_pass=0.1,
)
seed_ts = pcc.fit_transform(fmri_img, confounds=conf_df)
```

### Multi-subject parallel extraction

```python
from nilearn.maskers import MultiNiftiMasker

masker = MultiNiftiMasker(
    mask_img=group_mask,
    standardize='zscore_sample',
    n_jobs=4,
    memory='nilearn_cache',
)
list_of_X = masker.fit_transform(list_of_fmri_imgs)
```

### Surface vertex extraction (modern API)

```python
from nilearn.surface import SurfaceImage
from nilearn.datasets import load_fsaverage
from nilearn.maskers import SurfaceMasker

mesh = load_fsaverage('fsaverage5')
surf_img = SurfaceImage.from_volume(mesh=mesh, volume_img=fmri_4d)
surf_signals = SurfaceMasker().fit_transform(surf_img)
```

### Passing extra kwargs to signal.clean

```python
masker = NiftiMasker(
    standardize='zscore_sample',
    clean_args={'filter': 'cosine', 'extrapolate': False},
)
```

## Gotchas

- `standardize=True` maps to `'zscore'` (divides by N). Prefer `'zscore_sample'` (divides by N-1) for most analyses.
- `transform` requires `t_r` if any of `low_pass`, `high_pass`, or cosine drift filtering is used.
- `resampling_target='data'` (default for label/maps maskers) resamples the atlas onto the input image grid; `'labels'` or `'maps'` resamples the data instead. Mismatched grids without a strategy raise.
- `NiftiSpheresMasker` with `radius=None` extracts a single voxel — usually a mistake; pass an explicit radius in mm.
- Multi-subject maskers expect a list as input; passing a single 4D image works but you get a one-element list back.
- `mask_args` (dict) configures the auto-mask strategy; `clean_args` (dict) configures `signal.clean`. Don't confuse them.
- `generate_report()` requires `reports=True` at construction (default). For surface maskers, the report visualizes per-hemisphere coverage.

## See also

- `references/decoding.md` — `Decoder` accepts a masker via `mask=`
- `references/decomposition.md` — `CanICA`/`DictLearning` use a masker internally
- `references/surface.md` — `SurfaceImage`, `PolyMesh`, `vol_to_surf` for the inputs to surface maskers
- Source: https://nilearn.github.io/dev/modules/maskers.html
