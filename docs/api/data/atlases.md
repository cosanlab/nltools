---
title: data.atlases
---

Atlas registry, lazy loading, and coordinate labeling.

Atlases are hosted at ``huggingface.co/datasets/nltools/niftis`` under
``atlases/`` and fetched on first use via
`fetch_resource`. Cached locally afterwards.

The labeling logic was adapted from
[atlasreader](https://github.com/miykael/atlasreader) (BSD-3-Clause). Cite:

> Notter et al. (2019). AtlasReader. JOSS 4(34), 1257.
> https://doi.org/10.21105/joss.01257

**Classes:**

Name | Description
---- | -----------
[`Atlas`](#data-atlases-atlas) | A loaded atlas — image, labels, and metadata.
[`AtlasMetadata`](#data-atlases-atlasmetadata) | Static description of a registered atlas.
[`ClusterReport`](#data-atlases-clusterreport) | Result of `BrainData.cluster_report`.

**Methods:**

Name | Description
---- | -----------
[`cluster_report_data`](#data-atlases-cluster-report-data) | Compute cluster report DataFrames + thresholded BrainData.
[`label_coords`](#data-atlases-label-coords) | Look up anatomical labels for a set of MNI mm coordinates.
[`list_atlases`](#data-atlases-list-atlases) | Return the sorted list of registered atlas names.
[`load_atlas`](#data-atlases-load-atlas) | Lazy-load an atlas by registry name.



## Classes

(data-atlases-atlas)=
### `Atlas`

```python
Atlas(name: str, image: nb.Nifti1Image, labels: pl.DataFrame, kind: AtlasKind, citation: str) -> None
```

A loaded atlas — image, labels, and metadata.

Constructed by `load_atlas`; users normally don't instantiate
directly.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`name` | <code>[str](#str)</code> | Registry key (e.g. ``"harvard_oxford"``).
`image` | <code>[Nifti1Image](#nibabel.Nifti1Image)</code> | NIfTI volume. 3D for deterministic atlases, 4D for probabilistic ones (last axis indexes regions).
`labels` | <code>[DataFrame](#polars.DataFrame)</code> | Two-column ``index, name`` table. For deterministic atlases ``index`` is the integer voxel value; for probabilistic atlases ``index`` is the region index along the 4th dim of ``image``.
`kind` | <code>[AtlasKind](#nltools.data.atlases.registry.AtlasKind)</code> | ``"deterministic"`` or ``"probabilistic"``.
`citation` | <code>[str](#str)</code> | Short citation for the original atlas.

(data-atlases-atlasmetadata)=
### `AtlasMetadata`

```python
AtlasMetadata(kind: AtlasKind, citation: str) -> None
```

Static description of a registered atlas.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`kind` | <code>[AtlasKind](#nltools.data.atlases.registry.AtlasKind)</code> | ``"deterministic"`` (3D integer-labeled) or ``"probabilistic"`` (4D, last axis indexes regions).
`citation` | <code>[str](#str)</code> | Short citation string for the original atlas.

(data-atlases-clusterreport)=
### `ClusterReport`

```python
ClusterReport(peaks: pl.DataFrame, clusters: pl.DataFrame, stat_img: BrainData) -> None
```

Result of `BrainData.cluster_report`.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`peaks` | <code>[DataFrame](#polars.DataFrame)</code> | Polars DataFrame, one row per peak (incl. sub-peaks). Columns ``cluster_id``, ``x``, ``y``, ``z`` (mm), ``peak_stat``, ``volume_mm3``, ``n_voxels``, then one Utf8 column per atlas. ``cluster_id`` shares the integer id space of ``clusters`` (they are joinable); sub-peaks carry their parent cluster's id.
`clusters` | <code>[DataFrame](#polars.DataFrame)</code> | Polars DataFrame, one row per cluster. Columns ``cluster_id``, ``peak_x``, ``peak_y``, ``peak_z``, ``mean_stat``, ``volume_mm3``, ``n_voxels``, then one Utf8 column per atlas (mass-weighted top regions).
`stat_img` | <code>[BrainData](#nltools.data.BrainData)</code> | BrainData with the thresholded stat map (sub-cluster voxels and clusters smaller than ``cluster_threshold`` zeroed).

**Methods:**

Name | Description
---- | -----------
[`plot`](#data-atlases-plot) | Render an overview glass brain + one slice figure per cluster.
[`to_csv`](#data-atlases-to-csv) | Write ``peaks.csv`` and ``clusters.csv`` into ``output_dir``.



#### Methods

(data-atlases-plot)=
##### `plot`

```python
plot(*, output_dir: str | Path | None = None) -> list[tuple[str, Figure]] | None
```

Render an overview glass brain + one slice figure per cluster.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`output_dir` | <code>[str](#str) \| [Path](#pathlib.Path) \| None</code> | If given, save ``overview.png`` and ``cluster_NN.png`` files into the directory and return ``None``. If omitted, return a list of ``(label, matplotlib.figure.Figure)`` tuples without writing to disk. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[tuple](#tuple)[[str](#str), [Figure](#matplotlib.figure.Figure)]] \| None</code> | ``None`` when ``output_dir`` is set, else a list of     ``(label, figure)`` tuples.

(data-atlases-to-csv)=
##### `to_csv`

```python
to_csv(output_dir: str | Path) -> None
```

Write ``peaks.csv`` and ``clusters.csv`` into ``output_dir``.



## Methods

(data-atlases-cluster-report-data)=
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
`bd` | <code>[BrainData](#nltools.data.BrainData)</code> | BrainData with a 3D stat map (single sample). | *required*
`stat_threshold` | <code>[float](#float) \| None</code> | Voxel-level threshold. ``None`` means treat ``bd`` as already thresholded (skip voxel filtering, keep all non-zero voxels). | <code>3.0</code>
`cluster_threshold` | <code>[int](#int)</code> | Minimum cluster size in voxels. | <code>10</code>
`two_sided` | <code>[bool](#bool)</code> | Report negative clusters as separate clusters. | <code>True</code>
`min_distance` | <code>[float](#float)</code> | Minimum distance (mm) between sub-peaks. Passed to `get_clusters_table`. | <code>8.0</code>
`atlas` | <code>[str](#str) \| [Sequence](#collections.abc.Sequence)[[str](#str)]</code> | Atlas name or list of names from `list_atlases`. | <code>[DEFAULT_ATLASES](#nltools.data.atlases.registry.DEFAULT_ATLASES)</code>
`prob_threshold` | <code>[float](#float)</code> | Drop probabilistic-atlas regions below this %. | <code>5.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[DataFrame](#polars.DataFrame), [DataFrame](#polars.DataFrame), [BrainData](#nltools.data.BrainData)]</code> | Tuple ``(peaks, clusters, thresholded_bd)``.

(data-atlases-label-coords)=
### `label_coords`

```python
label_coords(coords: CoordsLike, *, atlas: str | Sequence[str] = 'harvard_oxford', prob_threshold: float = 5.0) -> pl.DataFrame
```

Look up anatomical labels for a set of MNI mm coordinates.

For each coordinate, returns the atlas region(s) it falls in. Works
for both deterministic atlases (single label per coord) and
probabilistic atlases (formatted ``"42.0% Foo; 18.0% Bar"`` strings,
sorted by descending probability).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`coords` | <code>[CoordsLike](#nltools.data.atlases.labeling.CoordsLike)</code> | ``(N, 3)`` array-like of MNI mm coordinates ``(x, y, z)``. A single coord like ``(-42, -22, 56)`` is also accepted. | *required*
`atlas` | <code>[str](#str) \| [Sequence](#collections.abc.Sequence)[[str](#str)]</code> | Atlas name or list of names from `list_atlases`. One column is added to the output per atlas. | <code>'harvard_oxford'</code>
`prob_threshold` | <code>[float](#float)</code> | For probabilistic atlases only — drop regions with probability (in percent units) below this threshold. | <code>5.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#polars.DataFrame)</code> | Frame with columns `x`, `y`, `z` plus one column per atlas.     All atlas columns are `Utf8`.

(data-atlases-list-atlases)=
### `list_atlases`

```python
list_atlases() -> list[str]
```

Return the sorted list of registered atlas names.

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[str](#str)]</code> | Sorted list of atlas names usable with `load_atlas`.

(data-atlases-load-atlas)=
### `load_atlas`

```python
load_atlas(name: str) -> Atlas
```

Lazy-load an atlas by registry name.

First call fetches the NIfTI + label CSV from
``huggingface.co/datasets/nltools/niftis`` (cached locally
afterwards). Subsequent calls in the same process are memoized.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`name` | <code>[str](#str)</code> | Atlas key from `list_atlases`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Atlas](#nltools.data.atlases.loading.Atlas)</code> | An `Atlas` with image, labels, and metadata loaded.
