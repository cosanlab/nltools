---
name: nilearn
description: Comprehensive nilearn (neuroimaging) API reference and best practices. Use when writing, reviewing, or debugging code that uses nilearn for fMRI analysis, GLM modeling, brain plotting, masking, decoding, connectivity, or surface analysis. Per-submodule references live under `references/` — load the relevant file (e.g. `references/glm.md`) for full inventories, signatures, and patterns.
---

# Nilearn

Statistical and machine-learning tools for neuroimaging. Every masker and model class follows scikit-learn's `fit` / `transform` / `predict` / `fit_transform` API.

## When to use this skill

- Writing, reviewing, or debugging any code that imports `nilearn`.
- Picking the right masker, atlas, GLM noise model, or decoder.
- Building a first-level or second-level GLM, connectivity pipeline, MVPA decoder, or surface workflow.
- Visualizing brain data (slices, glass brain, surface, interactive HTML).

## How this skill is organized

`SKILL.md` (this file) is a lean overview, picker tables, common workflows, and cross-cutting gotchas. **For full per-submodule API surface — every public class, every function, full constructor signatures, post-fit attributes — load the relevant file from `references/`.**

| Submodule | Reference | Covers |
|---|---|---|
| `nilearn.connectome` | [`references/connectome.md`](references/connectome.md) | `ConnectivityMeasure`, sparse covariance, vec/matrix utils |
| `nilearn.datasets` | [`references/datasets.md`](references/datasets.md) | All `fetch_atlas_*`, `fetch_*` example data, `load_mni152_*`, `load_fsaverage` |
| `nilearn.decoding` | [`references/decoding.md`](references/decoding.md) | `Decoder`, `FREM*`, `SpaceNet*`, `SearchLight` |
| `nilearn.decomposition` | [`references/decomposition.md`](references/decomposition.md) | `CanICA`, `DictLearning` |
| `nilearn.exceptions` | [`references/exceptions.md`](references/exceptions.md) | `MaskWarning`, `DimensionError`, `MeshDimensionError`, etc. |
| `nilearn.glm` | [`references/glm.md`](references/glm.md) | `FirstLevelModel`, `SecondLevelModel`, design matrices, contrasts, thresholding |
| `nilearn.image` | [`references/image.md`](references/image.md) | `load_img`, `resample_*`, `smooth_img`, `math_img`, `clean_img`, etc. |
| `nilearn.interfaces` | [`references/interfaces.md`](references/interfaces.md) | fmriprep `load_confounds`, BIDS helpers, FSL helpers |
| `nilearn.maskers` | [`references/maskers.md`](references/maskers.md) | `NiftiMasker`, `NiftiLabelsMasker`, `NiftiMapsMasker`, `NiftiSpheresMasker`, all `Multi*` and `Surface*` variants |
| `nilearn.masking` | [`references/masking.md`](references/masking.md) | `compute_*_mask`, `apply_mask`, `unmask`, `intersect_masks` |
| `nilearn.mass_univariate` | [`references/mass_univariate.md`](references/mass_univariate.md) | `permuted_ols` (TFCE / cluster-mass / cluster-size) |
| `nilearn.plotting` | [`references/plotting.md`](references/plotting.md) | All `plot_*`, `view_*`, `find_*`, slicer/projector classes |
| `nilearn.regions` | [`references/regions.md`](references/regions.md) | `RegionExtractor`, `Parcellations`, `connected_regions`, signal/img helpers |
| `nilearn.reporting` | [`references/reporting.md`](references/reporting.md) | `get_clusters_table`, `HTMLReport`, `model.generate_report()` |
| `nilearn.signal` | [`references/signal.md`](references/signal.md) | `clean`, `butterworth`, `high_variance_confounds` |
| `nilearn.surface` | [`references/surface.md`](references/surface.md) | `SurfaceImage`, `PolyMesh`, `vol_to_surf` |
| `nilearn.utils` | [`references/utils.md`](references/utils.md) | Estimator/function introspection helpers |

## Mental model

```
raw NIfTI ──┐
            │  masker (extract signals)        masker.transform(img) → (T × features)
atlas ──────┘
                 │
                 ▼
       sklearn-style model (fit / predict / score)
                 │
                 ▼
       img back out via masker.inverse_transform(weights) → Niimg
```

Three things to choose every time:
1. **Masker** — what region (whole brain, atlas, spheres, surface)? See `references/maskers.md`.
2. **Cleaning params** — `standardize`, `detrend`, `high_pass`, `low_pass`, `t_r`, `confounds`, `sample_mask`. Order is always: detrend → filter → confound removal → standardize.
3. **Model** — GLM (`glm.FirstLevelModel`), MVPA (`decoding.Decoder`), connectivity (`connectome.ConnectivityMeasure`), parcellation (`regions.Parcellations`), or decomposition (`decomposition.CanICA` / `DictLearning`).

## Picker: which submodule for which task?

| Task | Submodule(s) |
|---|---|
| Run a task-fMRI GLM | `glm` + `interfaces.fmriprep` (confounds) + `maskers` (optional) |
| Group-level analysis | `glm.second_level` + `mass_univariate` (permutation) |
| Multiple-comparison correction | `glm.threshold_stats_img`, `mass_univariate.permuted_ols` (TFCE) |
| Functional connectivity | `maskers` (extract ROI signals) → `connectome.ConnectivityMeasure` |
| MVPA / decoding | `decoding.Decoder` (or `FREM*`, `SpaceNet*`, `SearchLight`) |
| ICA / dictionary learning | `decomposition.CanICA` / `DictLearning` |
| Data-driven parcellation | `regions.Parcellations` |
| Brain visualization (volume) | `plotting.plot_stat_map`, `plot_glass_brain`, `view_img` |
| Brain visualization (surface) | `plotting.plot_img_on_surf`, `view_img_on_surf` |
| Project volume → surface | `surface.vol_to_surf` or `SurfaceImage.from_volume` |
| Get an atlas / template | `datasets.fetch_atlas_*` / `datasets.load_mni152_*` |
| Image arithmetic / resample / smooth | `image.math_img`, `resample_img`, `smooth_img` |
| Build a mask | `masking.compute_brain_mask`, `compute_epi_mask` |
| Load fMRIPrep confounds | `interfaces.fmriprep.load_confounds` |

## Picker: which masker?

| Masker | Input | Use case |
|---|---|---|
| `NiftiMasker` | 3D/4D NIfTI | Whole-brain voxel-level (decoding, searchlight, RSA) |
| `NiftiLabelsMasker` | 3D integer atlas | ROI-based with discrete parcellation |
| `NiftiMapsMasker` | 4D probabilistic maps | ICA components, soft/probabilistic atlases |
| `NiftiSpheresMasker` | MNI coordinates | Seed-based connectivity, literature ROIs |
| `MultiNiftiMasker` (and `MultiLabels`/`MultiMaps`/`MultiSpheres`) | Multiple subjects | Same as above, parallel across subjects |
| `SurfaceMasker` | `SurfaceImage` | Surface-based vertex-level |
| `SurfaceLabelsMasker` | Surface parcellation | Surface ROI extraction |
| `SurfaceMapsMasker` | Surface probability maps | Surface probabilistic atlas extraction |

Full constructor signatures, post-fit attributes, and `clean_args`/`mask_args` semantics: [`references/maskers.md`](references/maskers.md).

## Common workflows

### 1. fMRIPrep → first-level GLM

```python
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm import threshold_stats_img

confounds, sample_mask = load_confounds(
    fmri_img, strategy=('motion', 'high_pass', 'wm_csf', 'scrub'),
    motion='full', scrub=5,
)

flm = FirstLevelModel(t_r=2.0, hrf_model='glover', noise_model='ar1',
                      smoothing_fwhm=6, high_pass=0.01, mask_img=mask)
flm.fit(fmri_img, events=events_df, confounds=confounds, sample_masks=sample_mask)

z_map = flm.compute_contrast('active - rest', output_type='z_score')
thresholded, thresh = threshold_stats_img(
    z_map, alpha=0.05, height_control='fdr', cluster_threshold=10,
)
```

Details: [`references/glm.md`](references/glm.md), [`references/interfaces.md`](references/interfaces.md).

### 2. Functional connectivity (multi-subject)

```python
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_atlas_schaefer_2018

atlas = fetch_atlas_schaefer_2018(n_rois=200)
masker = NiftiLabelsMasker(labels_img=atlas.maps, labels=atlas.labels,
                           standardize='zscore_sample', memory='nilearn_cache')

all_ts = [masker.fit_transform(img, confounds=conf) for img, conf in zip(imgs, confs)]

conn = ConnectivityMeasure(kind='tangent', vectorize=True)
features = conn.fit_transform(all_ts)  # (n_subjects, n_features) — ready for sklearn
```

Details: [`references/connectome.md`](references/connectome.md), [`references/maskers.md`](references/maskers.md).

### 3. Decoding / MVPA

```python
from nilearn.decoding import Decoder

decoder = Decoder(
    estimator='svc', mask=mask_img, cv=5,
    screening_percentile=5, scoring='accuracy',
    smoothing_fwhm=4, standardize=True,
)
decoder.fit(fmri_imgs, y=labels)
print(decoder.cv_scores_)
weight_img = decoder.coef_img_['face']
```

Details: [`references/decoding.md`](references/decoding.md).

### 4. Surface analysis (modern API)

```python
from nilearn.surface import SurfaceImage
from nilearn.datasets import load_fsaverage
from nilearn.maskers import SurfaceMasker, SurfaceLabelsMasker

fsaverage5 = load_fsaverage('fsaverage5')                          # PolyMesh
surf_img   = SurfaceImage.from_volume(mesh=fsaverage5, volume_img=fmri_4d)

masker  = SurfaceMasker()
signals = masker.fit_transform(surf_img)
```

Details: [`references/surface.md`](references/surface.md), [`references/maskers.md`](references/maskers.md).

## Cross-cutting gotchas

These apply across submodules. Per-submodule gotchas live in each `references/<sub>.md`.

1. **`standardize=True` ≠ `'zscore_sample'`.** `True` maps to `'zscore'` (divides by N). Use `'zscore_sample'` (divides by N−1) for most analyses.

2. **Events DataFrame column names are strict:** `onset` (sec), `duration` (sec), `trial_type` (str). Optional: `modulation`. Other column names are silently ignored.

3. **`threshold_stats_img` returns a tuple** `(thresholded_img, threshold_value)` — always unpack.

4. **`load_confounds` returns a tuple** `(confounds_df, sample_mask)`. Pass `sample_mask` to `flm.fit(sample_masks=...)`.

5. **`fetch_*` vs `load_*`.** `fetch_*` downloads from the internet on first call (cached at `~/nilearn_data`, override with `data_dir=`). `load_*` loads bundled data instantly — no network.

6. **Surface API: old vs new.** `fetch_surf_fsaverage()` returns the old dict format. `load_fsaverage()` returns the new `PolyMesh`. Use the new API; only the new API works with `SurfaceImage`/`SurfaceMasker`.

7. **`make_glm_report` is deprecated.** Use `model.generate_report(contrasts=...)` instead.

8. **`get_data(img)` returns a copy.** For read-only access, use `img.get_fdata()` (nibabel) directly.

9. **`n_jobs=-1` caution.** Can hang or OOM on large images. Start with `n_jobs=1`, increase carefully.

10. **Default cmaps changed in v0.13:** `'RdBu_r'` for diverging stat maps, `'gray'` for anat, `'inferno'` for sequential.

11. **Resampling interpolation depends on image type.** Use `'nearest'` for discrete label/ROI images, `'continuous'` (default) for stat maps and continuous data.

12. **Processing order in `signal.clean` and maskers:** detrend → filter → remove confounds → standardize. Filters are applied to confounds too.

13. **`clean_img` vs `signal.clean`.** `clean_img` takes 4D NIfTI; `signal.clean` takes 2D `(timepoints, features)` arrays. Maskers use `signal.clean` internally.

14. **Memory/caching.** All maskers and models accept `memory=` for joblib caching. Use it for any expensive op:
    ```python
    masker = NiftiMasker(memory='nilearn_cache', memory_level=1)
    ```

15. **`force_resample` default changed.** In v0.13+, `image.resample_img` defaults to `force_resample=True`.

## Quick imports cheat-sheet

```python
# Maskers
from nilearn.maskers import (
    NiftiMasker, NiftiLabelsMasker, NiftiMapsMasker, NiftiSpheresMasker,
    MultiNiftiMasker, SurfaceMasker, SurfaceLabelsMasker, SurfaceMapsMasker,
)

# GLM
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix, first_level_from_bids
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.glm import threshold_stats_img, fdr_threshold, expression_to_contrast_vector

# Image / signal / masking
from nilearn import image
from nilearn.signal import clean, butterworth, high_variance_confounds
from nilearn import masking

# Connectivity / decoding / decomposition
from nilearn.connectome import ConnectivityMeasure, sym_matrix_to_vec, vec_to_sym_matrix
from nilearn.decoding import Decoder, DecoderRegressor, SearchLight, FREMClassifier, SpaceNetClassifier
from nilearn.decomposition import CanICA, DictLearning

# Regions / surface
from nilearn.regions import RegionExtractor, Parcellations, connected_regions
from nilearn.surface import SurfaceImage, PolyMesh, vol_to_surf

# Plotting
from nilearn import plotting
from nilearn.plotting import plot_stat_map, plot_glass_brain, plot_img_on_surf, view_img

# Datasets / interfaces / mass-univariate / reporting
from nilearn.datasets import (
    load_mni152_template, load_mni152_brain_mask, load_fsaverage,
    fetch_atlas_schaefer_2018, fetch_atlas_harvard_oxford, fetch_atlas_difumo,
    fetch_haxby, fetch_adhd, fetch_development_fmri,
)
from nilearn.interfaces.fmriprep import load_confounds, load_confounds_strategy
from nilearn.interfaces.bids import get_bids_files, parse_bids_filename
from nilearn.mass_univariate import permuted_ols
from nilearn.reporting import get_clusters_table
```
