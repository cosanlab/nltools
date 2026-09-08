---
title: Atlases & cluster reports
label: page-tasks-atlases
---

Put anatomical names on a result. `list_atlases` and `load_atlas` fetch parcellations from the nltools Hugging Face dataset on first use, and `label_coords` looks MNI coordinates up in them. `BrainData.cluster_report` (`cluster_report_data` underneath) thresholds a statistical map and labels each cluster's peak. `roi_to_brain_from_atlas` paints per-parcel values back into a volume.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`AtlasKind` | <code>Literal['deterministic', 'probabilistic']</code> | Kind of atlas: `'deterministic'` (3-D integer labels) or `'probabilistic'` (4-D, last axis indexes regions).
`ATLASES` | <code>dict[str, AtlasMetadata]</code> | Registered atlases keyed by name; each value is an `AtlasMetadata` (kind + citation).
`DEFAULT_ATLASES` | <code>tuple[str, ...]</code> | Default atlas trio for `BrainData.cluster_report` and `label_coords`.

**Classes:**

Name | Description
---- | -----------
[`Atlas`](#tasks-atlases-atlas) | A loaded atlas — image, labels, and metadata.
[`AtlasMetadata`](#tasks-atlases-atlasmetadata) | Static description of a registered atlas.
[`ClusterReport`](#tasks-atlases-clusterreport) | Result of `BrainData.cluster_report`.

**Functions:**

Name | Description
---- | -----------
[`list_atlases`](#tasks-atlases-list-atlases) | Return the sorted list of registered atlas names.
[`load_atlas`](#tasks-atlases-load-atlas) | Lazy-load an atlas by registry name.
[`label_coords`](#tasks-atlases-label-coords) | Look up anatomical labels for a set of MNI mm coordinates.
[`cluster_report_data`](#tasks-atlases-cluster-report-data) | Compute cluster report DataFrames + thresholded BrainData.
[`roi_to_brain_from_atlas`](#tasks-atlases-roi-to-brain-from-atlas) | Paint per-parcel values onto voxel space using a labeled atlas.

## Classes

(tasks-atlases-atlas)=
### `Atlas`

```python
Atlas(name: str, image: nb.Nifti1Image, labels: pl.DataFrame, kind: AtlasKind, citation: str)
```

A loaded atlas — image, labels, and metadata.

Constructed by `load_atlas`; users normally don't instantiate
directly.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`name` | <code>str</code> | Registry key (e.g. `'harvard_oxford'`).
`image` | <code>Nifti1Image</code> | NIfTI volume. 3-D for deterministic atlases, 4-D for probabilistic ones (last axis indexes regions).
`labels` | <code>DataFrame</code> | Two-column `index, name` table. For deterministic atlases `index` is the integer voxel value; for probabilistic atlases `index` is the region index along the 4th dim of `image`.
`kind` | <code>AtlasKind</code> | `'deterministic'` or `'probabilistic'`.
`citation` | <code>str</code> | Short citation for the original atlas.

(tasks-atlases-atlasmetadata)=
### `AtlasMetadata`

```python
AtlasMetadata(kind: AtlasKind, citation: str)
```

Static description of a registered atlas.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`kind` | <code>AtlasKind</code> | `'deterministic'` (3-D integer-labeled) or `'probabilistic'` (4-D, last axis indexes regions).
`citation` | <code>str</code> | Short citation string for the original atlas.

(tasks-atlases-clusterreport)=
### `ClusterReport`

```python
ClusterReport(peaks: pl.DataFrame, clusters: pl.DataFrame, stat_img: BrainData)
```

Result of `BrainData.cluster_report`.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`peaks` | <code>DataFrame</code> | One row per peak (incl. sub-peaks). Columns `cluster_id`, `x`, `y`, `z` (mm), `peak_stat`, `volume_mm3`, `n_voxels`, then one Utf8 column per atlas. `cluster_id` shares the integer id space of `clusters` (they are joinable); sub-peaks carry their parent cluster's id.
`clusters` | <code>DataFrame</code> | One row per cluster. Columns `cluster_id`, `peak_x`, `peak_y`, `peak_z`, `mean_stat`, `volume_mm3`, `n_voxels`, then one Utf8 column per atlas (mass-weighted top regions).
`stat_img` | <code>[BrainData](#page-data-brain-data)</code> | The thresholded stat map (sub-threshold voxels and clusters smaller than `cluster_threshold` zeroed).

**Methods:**

Name | Description
---- | -----------
[`plot`](#tasks-atlases-plot) | Render an overview glass brain + one slice figure per cluster.
[`to_csv`](#tasks-atlases-to-csv) | Write `peaks.csv` and `clusters.csv` into `output_dir`.



#### Methods

(tasks-atlases-plot)=
##### `plot`

```python
plot(*, output_dir: str | Path | None = None) -> list[tuple[str, Figure]] | None
```

Render an overview glass brain + one slice figure per cluster.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`output_dir` | <code>str \| Path</code> | If given, save `overview.png` and `cluster_NN.png` files into the directory and return None. If omitted, return the figures without writing to disk. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>list[tuple[str, Figure]] \| None</code> | `(label, figure)`     tuples, or None when `output_dir` is set.

(tasks-atlases-to-csv)=
##### `to_csv`

```python
to_csv(output_dir: str | Path) -> None
```

Write `peaks.csv` and `clusters.csv` into `output_dir`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`output_dir` | <code>str \| Path</code> | Directory to write into (created if missing). | *required*

## Functions

(tasks-atlases-list-atlases)=
### `list_atlases`

```python
list_atlases() -> list[str]
```

Return the sorted list of registered atlas names.

**Returns:**

Type | Description
---- | -----------
<code>list[str]</code> | Sorted list of atlas names usable with `load_atlas`.

(tasks-atlases-load-atlas)=
### `load_atlas`

```python
load_atlas(name: str) -> Atlas
```

Lazy-load an atlas by registry name.

The first call fetches the NIfTI + label CSV from
`huggingface.co/datasets/nltools/niftis` (cached locally afterwards).
Subsequent calls in the same process are memoized.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`name` | <code>str</code> | Atlas key from `list_atlases`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Atlas](#tasks-atlases-atlas)</code> | The atlas with image, labels, and metadata loaded.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If `name` isn't a registered atlas.

(tasks-atlases-label-coords)=
### `label_coords`

```python
label_coords(coords: CoordsLike, *, atlas: str | Sequence[str] = 'harvard_oxford', prob_threshold: float = 5.0) -> pl.DataFrame
```

Look up anatomical labels for a set of MNI mm coordinates.

For each coordinate, returns the atlas region(s) it falls in. Works
for both deterministic atlases (single label per coord) and
probabilistic atlases (formatted `"42.0% Foo; 18.0% Bar"` strings,
sorted by descending probability).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`coords` | <code>array - like</code> | `(N, 3)` MNI mm coordinates `(x, y, z)`. A single coordinate like `(-42, -22, 56)` is also accepted. | *required*
`atlas` | <code>str \| Sequence[str]</code> | Atlas name or list of names from `list_atlases`. One column is added to the output per atlas. Default `'harvard_oxford'`. | <code>'harvard_oxford'</code>
`prob_threshold` | <code>float</code> | For probabilistic atlases only — drop regions with probability (in percent units) below this threshold. Default 5.0. | <code>5.0</code>

**Returns:**

Type | Description
---- | -----------
<code>DataFrame</code> | Frame with columns `x`, `y`, `z` plus one column per atlas.     All atlas columns are `Utf8`.

(tasks-atlases-cluster-report-data)=
### `cluster_report_data`

```python
cluster_report_data(bd: BrainData, *, stat_threshold: float | None = 3.0, cluster_threshold: int = 10, two_sided: bool = True, min_distance: float = 8.0, atlas: str | Sequence[str] = DEFAULT_ATLASES, prob_threshold: float = 5.0) -> tuple[pl.DataFrame, pl.DataFrame, BrainData]
```

Compute cluster report DataFrames + thresholded BrainData.

Pure function — the BrainData facade `BrainData.cluster_report`
wraps the result in a `ClusterReport`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | A single 3-D stat map. | *required*
`stat_threshold` | <code>float</code> | Voxel-level threshold. None means treat `bd` as already thresholded (skip voxel filtering, keep all non-zero voxels). Default 3.0. | <code>3.0</code>
`cluster_threshold` | <code>int</code> | Minimum cluster size in voxels. Default 10. | <code>10</code>
`two_sided` | <code>bool</code> | Report negative clusters as separate clusters. Default True. | <code>True</code>
`min_distance` | <code>float</code> | Minimum distance (mm) between sub-peaks. Passed to `get_clusters_table`. Default 8.0. | <code>8.0</code>
`atlas` | <code>str \| Sequence[str]</code> | Atlas name or list of names from `list_atlases`. Default `DEFAULT_ATLASES`. | <code>DEFAULT_ATLASES</code>
`prob_threshold` | <code>float</code> | Drop probabilistic-atlas regions below this percentage. Default 5.0. | <code>5.0</code>

**Returns:**

Type | Description
---- | -----------
<code>tuple[DataFrame, DataFrame, [BrainData](#page-data-brain-data)]</code> | `(peaks, clusters,     thresholded_bd)` — see `ClusterReport` for the frame layouts.

(tasks-atlases-roi-to-brain-from-atlas)=
### `roi_to_brain_from_atlas`

```python
roi_to_brain_from_atlas(values, atlas, source_mask, *, roi_labels = None, fill: float = np.nan)
```

Paint per-parcel values onto voxel space using a labeled atlas.

Sibling of `roi_to_brain`, but accepts a *labeled* atlas (one integer label
per voxel), not an expanded mask with
one binary row per ROI. Voxels whose atlas label is not in `roi_labels` (or
whose label is 0) receive `fill`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`values` | <code>ndarray</code> | Per-parcel scalars, either 1-D `(n_parcels,)` for a single image or 2-D `(n_images, n_parcels)` for a stack of images. The trailing (parcel) axis must match `len(roi_labels)` (or the number of unique non-zero atlas labels when `roi_labels` is None). | *required*
`atlas` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image \| str \| Path</code> | Labeled image. Resampled to `source_mask` (nearest-neighbor) if shapes/affines differ. | *required*
`source_mask` | <code>Nifti1Image \| str \| Path</code> | Image (or path) defining the output voxel grid. The returned `BrainData` is masked to this image. | *required*
`roi_labels` | <code>array - like</code> | Integer atlas IDs in the same order as `values`. If None, defaults to `np.unique` of the atlas with 0 stripped (sorted ascending). | <code>None</code>
`fill` | <code>float</code> | Value for voxels not in any provided ROI. Default `np.nan`. | <code>nan</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Masked to `source_mask`, with each in-atlas voxel set to its     parcel's scalar from `values`. Holds a single image when `values` is     1-D, or `n_images` images when `values` is 2-D `(n_images, n_parcels)`.

**Examples:**

```python
from nltools.mask import roi_to_brain_from_atlas

brain_map = roi_to_brain_from_atlas(
    values=accuracies,
    atlas=atlas_img,
    source_mask=brain_mask,
    roi_labels=[1, 2, 3],
)
```
