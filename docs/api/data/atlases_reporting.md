(data-atlases-reporting-reporting)=
## `reporting`

Cluster reports — peak/cluster geometry plus atlas labels.

The peak/sub-peak geometry comes from `get_clusters_table`;
the cluster masks and mass-weighted labels are computed locally so we can
attribute every voxel of every cluster to one or more atlases.

**Classes:**

Name | Description
---- | -----------
[`ClusterReport`](#data-atlases-reporting-clusterreport) | Result of `BrainData.cluster_report`.

**Methods:**

Name | Description
---- | -----------
[`cluster_report_data`](#data-atlases-reporting-cluster-report-data) | Compute cluster report DataFrames + thresholded BrainData.

### Classes

(data-atlases-reporting-clusterreport)=
#### `ClusterReport`

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
[`plot`](#data-atlases-reporting-plot) | Render an overview glass brain + one slice figure per cluster.
[`to_csv`](#data-atlases-reporting-to-csv) | Write ``peaks.csv`` and ``clusters.csv`` into ``output_dir``.

##### Methods

(data-atlases-reporting-plot)=
###### `plot`

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
<code>[list](#list)[[tuple](#tuple)[[str](#str), [Figure](#matplotlib.figure.Figure)]] \| None</code> | ``None`` when ``output_dir`` is set, else a list of
<code>[list](#list)[[tuple](#tuple)[[str](#str), [Figure](#matplotlib.figure.Figure)]] \| None</code> | ``(label, figure)`` tuples.

(data-atlases-reporting-to-csv)=
###### `to_csv`

```python
to_csv(output_dir: str | Path) -> None
```

Write ``peaks.csv`` and ``clusters.csv`` into ``output_dir``.



### Methods

(data-atlases-reporting-cluster-report-data)=
#### `cluster_report_data`

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

