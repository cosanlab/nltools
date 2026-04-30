# nilearn.decomposition — Multivariate Decompositions

The `nilearn.decomposition` module includes a subject-level variant of ICA called Canonical ICA, plus a sparse dictionary-learning alternative. Both fit on a list of 4D fMRI images and produce a 4D `components_img_` of spatial maps.

**Source:** https://nilearn.github.io/dev/modules/decomposition.html

## When to use

- Group-level resting-state networks: `CanICA`
- Sparse, more localized component maps (recommended over ICA when interpretability matters): `DictLearning`
- Both produce 4D component maps suitable for `NiftiMapsMasker` extraction downstream.

## Inventory

### Classes

| Class | Purpose |
|---|---|
| `CanICA` | Perform Canonical Independent Component Analysis. |
| `DictLearning` | Perform a map learning algorithm based on spatial component sparsity, over a CanICA initialization. |

## Shared parameters

Both classes inherit a common set from the internal `_BaseDecomposition` and accept fMRI images as a list (one 4D per subject).

```
n_components       # int, default 20
random_state       # int|None
mask               # Niimg-like, masker, or None (auto-compute)
smoothing_fwhm     # mm
standardize        # 'zscore_sample'|True|False
detrend            # bool
low_pass, high_pass, t_r
target_affine, target_shape
mask_strategy      # 'epi'|'background'|...
mask_args          # dict
memory             # joblib.Memory or path
memory_level       # int
n_jobs             # int
verbose            # int
```

Shared post-fit attributes:
- `components_img_` — 4D `Nifti1Image` of spatial components
- `components_` — 2D array `(n_components, n_voxels)`
- `mask_img_`
- `masker_` — fitted `MultiNiftiMasker` used internally

Shared methods: `fit(imgs, y=None, confounds=None)`, `transform(imgs, confounds=None)`, `inverse_transform(loadings)`, `score(imgs, per_component=False)`.

## CanICA

```python
CanICA(
    mask=None,
    n_components=20,
    smoothing_fwhm=6.0,
    do_cca=True,                # Canonical correlation step
    threshold='auto',           # float|'auto'|None — sparsifies components
    n_init=10,                  # ICA restarts; best run kept
    random_state=None,
    standardize='zscore_sample',
    standardize_confounds=True,
    detrend=True,
    low_pass=None,
    high_pass=None,
    t_r=None,
    target_affine=None,
    target_shape=None,
    mask_strategy='epi',
    mask_args=None,
    memory=None,
    memory_level=0,
    n_jobs=1,
    verbose=0,
)
```

CanICA-specific post-fit attributes:
- `components_img_`, `components_`
- `variance_` — per-component explained variance
- `mask_img_`, `masker_`

## DictLearning

```python
DictLearning(
    n_components=20,
    n_epochs=1,                 # passes over the data
    alpha=10,                   # sparsity penalty (higher = sparser)
    dict_init=None,             # initial atoms (default: CanICA)
    random_state=None,
    batch_size=20,
    method='cd',                # 'cd' (coordinate descent) | 'lars'
    mask=None,
    smoothing_fwhm=4.0,
    standardize='zscore_sample',
    standardize_confounds=True,
    detrend=True,
    low_pass=None,
    high_pass=None,
    t_r=None,
    target_affine=None,
    target_shape=None,
    mask_strategy='epi',
    mask_args=None,
    memory=None,
    memory_level=0,
    n_jobs=1,
    verbose=0,
)
```

DictLearning-specific post-fit attributes:
- `components_img_`, `components_`
- `mask_img_`, `masker_`

## Common patterns

### Group ICA on resting-state

```python
from nilearn.decomposition import CanICA

canica = CanICA(
    n_components=20,
    smoothing_fwhm=6,
    threshold='auto',
    n_init=10,
    standardize='zscore_sample',
    memory='nilearn_cache',
    memory_level=2,
    n_jobs=1,
)
canica.fit(list_of_fmri_imgs)
components = canica.components_img_      # 4D Nifti1Image
```

### Dictionary Learning (sparser, often more interpretable)

```python
from nilearn.decomposition import DictLearning

dl = DictLearning(
    n_components=20,
    n_epochs=1,
    alpha=10,
    smoothing_fwhm=4,
    standardize='zscore_sample',
    memory='nilearn_cache',
)
dl.fit(list_of_fmri_imgs)
atoms = dl.components_img_
```

### Use components downstream as a probabilistic atlas

```python
from nilearn.maskers import NiftiMapsMasker

masker = NiftiMapsMasker(maps_img=canica.components_img_,
                         standardize='zscore_sample')
ts = masker.fit_transform(fmri_img, confounds=conf_df)
```

### Visualize components

```python
from nilearn.image import iter_img
from nilearn import plotting

for i, comp in enumerate(iter_img(canica.components_img_)):
    plotting.plot_stat_map(comp, title=f'comp {i}', threshold=1e-3)
```

## Gotchas

- Both classes accept a **list** of 4D images; passing a single image works but treats it as a one-subject group.
- CanICA's `threshold='auto'` chooses a sparsity level via random matrix theory — explicit floats are also valid.
- CanICA `n_init=10` runs ICA ten times and keeps the best; this dominates runtime. Lower for prototyping.
- DictLearning is initialized from a CanICA fit; expect ~2x the CanICA cost. Increase `n_epochs` for cleaner atoms.
- DictLearning's `alpha` controls sparsity: higher alpha -> sparser, more localized maps; too high zeros things out.
- Use `memory='nilearn_cache'` — these decompositions are expensive and benefit greatly from caching.
- `components_img_` is unsigned; sign of ICA components is arbitrary. Compare absolute values.
- For surface decomposition, project to surface first via `vol_to_surf` then use surface-aware downstream tools (see `references/surface.md`).

## See also

- `references/maskers.md` — `NiftiMapsMasker` consumes `components_img_`
- `references/decoding.md` — supervised alternatives when labels exist
- Source: https://nilearn.github.io/dev/modules/decomposition.html
