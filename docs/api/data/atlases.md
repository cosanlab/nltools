## `atlases`

Atlas registry, lazy loading, and coordinate labeling.

Atlases are hosted at ``huggingface.co/datasets/nltools/niftis`` under
``atlases/`` and fetched on first use via
:func:`nltools.templates.fetch_resource`. Cached locally afterwards.

The labeling logic was adapted from
[atlasreader](https://github.com/miykael/atlasreader) (BSD-3-Clause). Cite:

> Notter et al. (2019). AtlasReader. JOSS 4(34), 1257.
> https://doi.org/10.21105/joss.01257

**Modules:**

Name | Description
---- | -----------
[`labeling`](#labeling) | Coordinate-level atlas labeling.
[`loading`](#loading) | Lazy loading of atlas NIfTI + label CSV files from the HF dataset.
[`registry`](#registry) | Static registry of atlases hosted at ``nltools/niftis/atlases``.
[`reporting`](#reporting) | Cluster reports — peak/cluster geometry plus atlas labels.

**Classes:**

Name | Description
---- | -----------
[`Atlas`](#Atlas) | A loaded atlas — image, labels, and metadata.
[`AtlasMetadata`](#AtlasMetadata) | Static description of a registered atlas.
[`ClusterReport`](#ClusterReport) | Result of :meth:`BrainData.cluster_report`.

**Methods:**

Name | Description
---- | -----------
[`cluster_report_data`](#cluster_report_data) | Compute cluster report DataFrames + thresholded BrainData.
[`label_coords`](#label_coords) | Look up anatomical labels for a set of MNI mm coordinates.
[`list_atlases`](#list_atlases) | Return the sorted list of registered atlas names.
[`load_atlas`](#load_atlas) | Lazy-load an atlas by registry name.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`ATLASES`](#ATLASES) | <code>[dict](#dict)[[str](#str), [AtlasMetadata](#nltools.data.atlases.registry.AtlasMetadata)]</code> | 
[`AtlasKind`](#AtlasKind) |  | 
[`DEFAULT_ATLASES`](#DEFAULT_ATLASES) | <code>[tuple](#tuple)[[str](#str), ...]</code> | 

##### Methods

###### `plot`

```python
plot(*, output_dir: str | Path | None = None) -> list[tuple[str, Any]] | None
```

Render an overview glass brain + one slice figure per cluster.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`output_dir` | <code>[str](#str) \| [Path](#pathlib.Path) \| None</code> | If given, save ``overview.png`` and ``cluster_NN.png`` files into the directory and return ``None``. If omitted, return a list of ``(label, matplotlib.figure.Figure)`` tuples without writing to disk. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[tuple](#tuple)[[str](#str), [Any](#typing.Any)]] \| None</code> | ``None`` when ``output_dir`` is set, else a list of
<code>[list](#list)[[tuple](#tuple)[[str](#str), [Any](#typing.Any)]] \| None</code> | ``(label, figure)`` tuples.

###### `to_csv`

```python
to_csv(output_dir: str | Path) -> None
```

Write ``peaks.csv`` and ``clusters.csv`` into ``output_dir``.



### Methods

#### `cluster_report_data`

```python
cluster_report_data(bd: BrainData, *, stat_threshold: float | None = 3.0, cluster_threshold: int = 10, two_sided: bool = True, min_distance: float = 8.0, atlas: str | Sequence[str] = DEFAULT_ATLASES, prob_threshold: float = 5.0) -> tuple[pl.DataFrame, pl.DataFrame, BrainData]
```

Compute cluster report DataFrames + thresholded BrainData.

Pure function — the BrainData facade :meth:`BrainData.cluster_report`
wraps the result in a :class:`ClusterReport`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#nltools.data.BrainData)</code> | BrainData with a 3D stat map (single sample). | *required*
`stat_threshold` | <code>[float](#float) \| None</code> | Voxel-level threshold. ``None`` means treat ``bd`` as already thresholded (skip voxel filtering, keep all non-zero voxels). | <code>3.0</code>
`cluster_threshold` | <code>[int](#int)</code> | Minimum cluster size in voxels. | <code>10</code>
`two_sided` | <code>[bool](#bool)</code> | Report negative clusters as separate clusters. | <code>True</code>
`min_distance` | <code>[float](#float)</code> | Minimum distance (mm) between sub-peaks. Passed to :func:`nilearn.reporting.get_clusters_table`. | <code>8.0</code>
`atlas` | <code>[str](#str) \| [Sequence](#collections.abc.Sequence)[[str](#str)]</code> | Atlas name or list of names from :func:`list_atlases`. | <code>[DEFAULT_ATLASES](#nltools.data.atlases.registry.DEFAULT_ATLASES)</code>
`prob_threshold` | <code>[float](#float)</code> | Drop probabilistic-atlas regions below this %. | <code>5.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[DataFrame](#polars.DataFrame), [DataFrame](#polars.DataFrame), [BrainData](#nltools.data.BrainData)]</code> | Tuple ``(peaks, clusters, thresholded_bd)``.

#### `label_coords`

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
`atlas` | <code>[str](#str) \| [Sequence](#collections.abc.Sequence)[[str](#str)]</code> | Atlas name or list of names from :func:`list_atlases`. One column is added to the output per atlas. | <code>'harvard_oxford'</code>
`prob_threshold` | <code>[float](#float)</code> | For probabilistic atlases only — drop regions with probability (in percent units) below this threshold. | <code>5.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#polars.DataFrame)</code> | Polars DataFrame with columns ``x``, ``y``, ``z`` plus one
<code>[DataFrame](#polars.DataFrame)</code> | column per atlas. All atlas columns are ``Utf8``.

#### `list_atlases`

```python
list_atlases() -> list[str]
```

Return the sorted list of registered atlas names.

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[str](#str)]</code> | Sorted list of atlas names usable with
<code>[list](#list)[[str](#str)]</code> | func:`nltools.data.atlases.load_atlas`.

#### `load_atlas`

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
`name` | <code>[str](#str)</code> | Atlas key from :func:`list_atlases`. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`An` | <code>[Atlas](#nltools.data.atlases.loading.Atlas)</code> | class:`Atlas` with image, labels, and metadata loaded.



### Modules

#### `labeling`

Coordinate-level atlas labeling.

Adapted from [atlasreader](https://github.com/miykael/atlasreader)
(BSD-3-Clause). Cite:

> Notter et al. (2019). AtlasReader. JOSS 4(34), 1257.

**Methods:**

Name | Description
---- | -----------
[`label_coords`](#label_coords) | Look up anatomical labels for a set of MNI mm coordinates.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`CoordsLike`](#CoordsLike) |  | 

##### Methods

###### `label_coords`

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
`atlas` | <code>[str](#str) \| [Sequence](#collections.abc.Sequence)[[str](#str)]</code> | Atlas name or list of names from :func:`list_atlases`. One column is added to the output per atlas. | <code>'harvard_oxford'</code>
`prob_threshold` | <code>[float](#float)</code> | For probabilistic atlases only — drop regions with probability (in percent units) below this threshold. | <code>5.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#polars.DataFrame)</code> | Polars DataFrame with columns ``x``, ``y``, ``z`` plus one
<code>[DataFrame](#polars.DataFrame)</code> | column per atlas. All atlas columns are ``Utf8``.

#### `loading`

Lazy loading of atlas NIfTI + label CSV files from the HF dataset.

**Classes:**

Name | Description
---- | -----------
[`Atlas`](#Atlas) | A loaded atlas — image, labels, and metadata.

**Methods:**

Name | Description
---- | -----------
[`load_atlas`](#load_atlas) | Lazy-load an atlas by registry name.

##### Methods

###### `load_atlas`

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
`name` | <code>[str](#str)</code> | Atlas key from :func:`list_atlases`. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`An` | <code>[Atlas](#nltools.data.atlases.loading.Atlas)</code> | class:`Atlas` with image, labels, and metadata loaded.

#### `registry`

Static registry of atlases hosted at ``nltools/niftis/atlases``.

Each entry describes an atlas's kind (deterministic vs probabilistic) and
the citation users should cite when they use it. The actual NIfTI + label
files are fetched lazily by :func:`nltools.data.atlases.load_atlas` via
:func:`nltools.templates.fetch_resource`.

Atlases were sourced from atlasreader (BSD-3-Clause) and are subject to
their original upstream licenses — see ``LICENSES.md`` in the HF dataset.

**Classes:**

Name | Description
---- | -----------
[`AtlasMetadata`](#AtlasMetadata) | Static description of a registered atlas.

**Methods:**

Name | Description
---- | -----------
[`list_atlases`](#list_atlases) | Return the sorted list of registered atlas names.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`ATLASES`](#ATLASES) | <code>[dict](#dict)[[str](#str), [AtlasMetadata](#nltools.data.atlases.registry.AtlasMetadata)]</code> | 
[`AtlasKind`](#AtlasKind) |  | 
[`DEFAULT_ATLASES`](#DEFAULT_ATLASES) | <code>[tuple](#tuple)[[str](#str), ...]</code> | 

##### Methods

###### `list_atlases`

```python
list_atlases() -> list[str]
```

Return the sorted list of registered atlas names.

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[str](#str)]</code> | Sorted list of atlas names usable with
<code>[list](#list)[[str](#str)]</code> | func:`nltools.data.atlases.load_atlas`.

#### `reporting`

Cluster reports — peak/cluster geometry plus atlas labels.

The peak/sub-peak geometry comes from :func:`nilearn.reporting.get_clusters_table`;
the cluster masks and mass-weighted labels are computed locally so we can
attribute every voxel of every cluster to one or more atlases.

**Classes:**

Name | Description
---- | -----------
[`ClusterReport`](#ClusterReport) | Result of :meth:`BrainData.cluster_report`.

**Methods:**

Name | Description
---- | -----------
[`cluster_report_data`](#cluster_report_data) | Compute cluster report DataFrames + thresholded BrainData.

##### Methods

###### `cluster_report_data`

```python
cluster_report_data(bd: BrainData, *, stat_threshold: float | None = 3.0, cluster_threshold: int = 10, two_sided: bool = True, min_distance: float = 8.0, atlas: str | Sequence[str] = DEFAULT_ATLASES, prob_threshold: float = 5.0) -> tuple[pl.DataFrame, pl.DataFrame, BrainData]
```

Compute cluster report DataFrames + thresholded BrainData.

Pure function — the BrainData facade :meth:`BrainData.cluster_report`
wraps the result in a :class:`ClusterReport`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#nltools.data.BrainData)</code> | BrainData with a 3D stat map (single sample). | *required*
`stat_threshold` | <code>[float](#float) \| None</code> | Voxel-level threshold. ``None`` means treat ``bd`` as already thresholded (skip voxel filtering, keep all non-zero voxels). | <code>3.0</code>
`cluster_threshold` | <code>[int](#int)</code> | Minimum cluster size in voxels. | <code>10</code>
`two_sided` | <code>[bool](#bool)</code> | Report negative clusters as separate clusters. | <code>True</code>
`min_distance` | <code>[float](#float)</code> | Minimum distance (mm) between sub-peaks. Passed to :func:`nilearn.reporting.get_clusters_table`. | <code>8.0</code>
`atlas` | <code>[str](#str) \| [Sequence](#collections.abc.Sequence)[[str](#str)]</code> | Atlas name or list of names from :func:`list_atlases`. | <code>[DEFAULT_ATLASES](#nltools.data.atlases.registry.DEFAULT_ATLASES)</code>
`prob_threshold` | <code>[float](#float)</code> | Drop probabilistic-atlas regions below this %. | <code>5.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[DataFrame](#polars.DataFrame), [DataFrame](#polars.DataFrame), [BrainData](#nltools.data.BrainData)]</code> | Tuple ``(peaks, clusters, thresholded_bd)``.

