# nilearn.regions — Operating on Regions

Region extraction on 4D statistical / atlas maps and helpers for converting between region signals and brain images. Includes data-driven parcellation algorithms (`Parcellations`, `HierarchicalKMeans`, `ReNA`).

**Source:** https://nilearn.github.io/dev/modules/regions.html

## Inventory

### Classes
| Class | Purpose |
|---|---|
| `RegionExtractor` | Extract connected regions from a 4D probabilistic atlas. |
| `Parcellations` | Learn parcellations on fMRI images (kmeans, ward, rena, ...). |
| `ReNA` | Recursive Neighbor Agglomeration clustering. |
| `HierarchicalKMeans` | Hierarchical KMeans clustering. |

### Functions
| Function | Purpose |
|---|---|
| `connected_regions(maps_img, ...)` | Extract brain connected regions into separate regions. |
| `connected_label_regions(labels_img, ...)` | Extract connected regions from a label-image atlas. |
| `img_to_signals_labels(imgs, labels_img, ...)` | Extract region signals from labeled image. |
| `signals_to_img_labels(signals, labels_img, ...)` | Reconstruct image from region signals (labels). |
| `img_to_signals_maps(imgs, maps_img, ...)` | Extract region signals using probabilistic maps. |
| `signals_to_img_maps(region_signals, maps_img)` | Reconstruct image from region signals (maps). |
| `recursive_neighbor_agglomeration(X, ...)` | Functional API for ReNA clustering. |

## RegionExtractor

```python
RegionExtractor(
    maps_img,                       # 4D probabilistic atlas
    mask_img=None,
    min_region_size=1350,           # mm^3
    threshold=1.0,
    thresholding_strategy='ratio_n_voxels',  # 'ratio_n_voxels'|'img_value'|'percentile'
    extractor='local_regions',      # 'local_regions'|'connected_components'
    smoothing_fwhm=6,
    standardize=False,
    detrend=False,
    low_pass=None, high_pass=None, t_r=None,
    memory=None, memory_level=0, n_jobs=1,
)

extractor.fit()
regions_img = extractor.regions_img_       # 4D, one volume per extracted region
extractor.index_                            # which input map each region came from
```

## Parcellations

```python
Parcellations(
    method='kmeans',                # 'kmeans'|'ward'|'complete'|'average'|'rena'|'hierarchical_kmeans'
    n_parcels=50,
    random_state=0,
    mask=None, mask_strategy='epi',
    smoothing_fwhm=4,
    standardize=False, detrend=False,
    low_pass=None, high_pass=None, t_r=None,
    memory=None, memory_level=0, n_jobs=1,
    verbose=0,
)

parcellator.fit(fmri_imgs)          # list of 4D images
labels_img = parcellator.labels_img_
parcellator.transform(imgs)         # extract per-region signals
```

Method choice:
- `'kmeans'` / `'hierarchical_kmeans'` — fast, isotropic clusters.
- `'ward'` / `'complete'` / `'average'` — agglomerative, spatially constrained.
- `'rena'` — Recursive Neighbor Agglomeration, fastest for large parcellations.

## ReNA / HierarchicalKMeans

Both are sklearn-compatible cluster estimators (`fit(X)`, `transform`, `inverse_transform`). Operate on 2D arrays already extracted via a masker.

```python
from nilearn.regions import ReNA, HierarchicalKMeans

rena = ReNA(mask_img=mask_img, n_clusters=500, scaling=False)
rena.fit(X)                          # X: (n_samples, n_voxels)
labels_img = rena.labels_img_
```

## Signal extraction helpers (functional API)

```python
from nilearn.regions import (
    img_to_signals_labels, signals_to_img_labels,
    img_to_signals_maps, signals_to_img_maps,
)

# Discrete labels
signals, labels = img_to_signals_labels(imgs, labels_img, mask_img=mask_img,
                                         background_label=0, strategy='mean')
img = signals_to_img_labels(signals, labels_img, mask_img=mask_img)

# Probabilistic maps
signals, labels = img_to_signals_maps(imgs, maps_img, mask_img=mask_img)
img = signals_to_img_maps(signals, maps_img, mask_img=mask_img)
```

## Connected component extraction

```python
from nilearn.regions import connected_regions, connected_label_regions

# Probabilistic maps -> separate regions
regions_img, indices = connected_regions(maps_img, min_region_size=1350,
                                          extract_type='local_regions',
                                          smoothing_fwhm=6, mask_img=None)

# Discrete atlas -> split disconnected pieces of each label
new_labels_img = connected_label_regions(labels_img, min_size=None,
                                          connect_diag=True, labels=None)
```

## Common patterns

Probabilistic atlas to discrete regions:

```python
from nilearn.regions import RegionExtractor

extractor = RegionExtractor(maps_img, min_region_size=1350, threshold=1.0,
                             thresholding_strategy='ratio_n_voxels')
extractor.fit()
regions = extractor.regions_img_
```

Data-driven parcellation:

```python
from nilearn.regions import Parcellations

parcellator = Parcellations(method='ward', n_parcels=100, mask=mask_img,
                             standardize='zscore_sample', smoothing_fwhm=4,
                             memory='nilearn_cache')
parcellator.fit(fmri_imgs)
labels = parcellator.labels_img_
```

Round-trip via labels:

```python
signals, _ = img_to_signals_labels(fmri_4d, labels_img)
recon = signals_to_img_labels(signals, labels_img)
```

## Gotchas

- `RegionExtractor` requires a 4D probabilistic input (e.g., DiFuMo, MSDL); pass discrete atlases through `connected_label_regions` instead.
- `Parcellations.method='rena'` is much faster than `'ward'` for large `n_parcels` (>500) and gives comparable results.
- `min_region_size` is in mm^3, not voxels — adjust by voxel volume.
- `img_to_signals_labels` returns `(signals, labels_used)`; `labels_used` may be a subset of input labels if some regions are empty.
- For round-trip fidelity prefer the masker classes (`NiftiLabelsMasker`, `NiftiMapsMasker`) — they handle resampling and confound regression in one call.

## See also

- `nilearn.maskers.NiftiLabelsMasker`, `NiftiMapsMasker` — higher-level alternatives.
- `nilearn.datasets.fetch_atlas_*` — pre-built atlases.
- `nilearn.plotting.find_parcellation_cut_coords`, `find_probabilistic_atlas_cut_coords`.
- https://nilearn.github.io/dev/modules/regions.html
