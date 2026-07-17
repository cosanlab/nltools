(data-data)=
## `data`

nltools data types.

**Modules:**

Name | Description
---- | -----------
[`adjacency`](#data-adjacency) | Provide data structures for working with similarity and dissimilarity matrices.
[`atlases`](#data-atlases) | Atlas registry, lazy loading, and coordinate labeling.
[`braindata`](#data-braindata) | Represent brain image data with the BrainData class.
[`collection`](#data-collection) | BrainCollection — multi-subject brain-data container (v0.6.0).
[`designmatrix`](#data-designmatrix) | Provide a Polars-based design matrix for neuroimaging analysis.
[`fitresults`](#data-fitresults) | Immutable container for model fitting results.
[`roc`](#data-roc) | ROC (Receiver Operating Characteristic) analysis for single-interval classification.
[`simulator`](#data-simulator) | Tools to simulate multivariate brain and grid data for testing analysis pipelines.

**Classes:**

Name | Description
---- | -----------
[`Adjacency`](#data-adjacency) | Represent adjacency matrices in vectorized form.
[`BrainCollection`](#data-braincollection) | Parallel, lazy iterator of ``BrainData`` whose API mirrors ``BrainData``.
[`BrainData`](#data-braindata) | Represent neuroimaging data as vectors instead of three-dimensional matrices.
[`DesignMatrix`](#data-designmatrix) | Represent experimental designs for neuroimaging with Polars.
[`Fit`](#data-fit) | Immutable container for model fitting results.
[`Roc`](#data-roc) | Compute receiver operating characteristic curves for single-interval or forced-choice classification.
[`SimulateGrid`](#data-simulategrid) | Simulate 2D grid data for testing statistical methods.
[`Simulator`](#data-simulator) | Simulate fMRI data with realistic spatial and temporal characteristics.



### Classes

(data-adjacency)=
#### `Adjacency`

```python
Adjacency(data = None, *, Y = None, matrix_type = None, labels = None, spatial_scale: SpatialScale | None = None)
```

Represent adjacency matrices in vectorized form.

Adjacency is a class to represent Adjacency matrices as a vector rather
than a 2-dimensional matrix. This makes it easier to perform data
manipulation and analyses.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | pandas data instance or list of files | <code>None</code>
`matrix_type` |  | (str) type of matrix.  Possible values include:         ['distance','similarity','directed','distance_flat',         'similarity_flat','directed_flat'] | <code>None</code>
`Y` |  | Pandas DataFrame of training labels | <code>None</code>
`labels` |  | (list) optional node labels | <code>None</code>

**Methods:**

Name | Description
---- | -----------
[`append`](#data-append) | Append data to an Adjacency instance.
[`bootstrap`](#data-bootstrap) | Bootstrap statistics using efficient online algorithms.
[`cluster_summary`](#data-cluster-summary) | Provide summaries of clusters within Adjacency matrices.
[`copy`](#data-copy) | Create a copy of Adjacency object.
[`distance`](#data-distance) | Calculate distance between images within an Adjacency() instance.
[`distance_to_similarity`](#data-distance-to-similarity) | Convert distance matrix to similarity matrix.
[`generate_permutations`](#data-generate-permutations) | Generate permuted versions of an Adjacency instance lazily.
[`mean`](#data-mean) | Calculate mean of Adjacency.
[`median`](#data-median) | Calculate median of Adjacency.
[`plot`](#data-plot) | Create a heatmap of an Adjacency matrix.
[`plot_label_distance`](#data-plot-label-distance) | Create a violin plot of within- and between-label distances.
[`plot_mds`](#data-plot-mds) | Plot multidimensional scaling.
[`plot_silhouette`](#data-plot-silhouette) | Create a silhouette plot.
[`r_to_z`](#data-r-to-z) | Apply Fisher's r-to-z transformation to each data element.
[`regress`](#data-regress) | Run a regression on an adjacency instance.
[`similarity`](#data-similarity) | Calculate similarity between two Adjacency matrices.
[`social_relations_model`](#data-social-relations-model) | Estimate the social relations model from a matrix for a round-robin design.
[`squareform`](#data-squareform) | Convert adjacency data back to square form.
[`stats_label_distance`](#data-stats-label-distance) | Calculate permutation tests on within and between label distance.
[`std`](#data-std) | Calculate standard deviation of Adjacency.
[`sum`](#data-sum) | Calculate sum of Adjacency.
[`threshold`](#data-threshold) | Threshold an Adjacency instance.
[`to_brain`](#data-to-brain) | Project per-matrix scalars back to voxel-space `BrainData`.
[`to_graph`](#data-to-graph) | Convert a single Adjacency matrix into a NetworkX graph.
[`to_square`](#data-to-square) | Convert adjacency back to square matrix format.
[`ttest`](#data-ttest) | Calculate ttest across samples.
[`write`](#data-write) | Write out Adjacency object to csv file.
[`z_to_r`](#data-z-to-r) | Convert each z score back into an r value.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`Y`](#data-y) | <code>[DataFrame](#polars.DataFrame)</code> | Training labels as a polars DataFrame (possibly empty).
[`data`](#data-data) |  | 
`is_empty` | <code>[bool](#bool)</code> | Check if Adjacency object is empty.
`is_single_matrix` |  | 
`issymmetric` |  | 
`labels` |  | 
`matrix_type` |  | 
`n_nodes` |  | Return the number of nodes in the adjacency matrix.
`shape` |  | Return the logical shape of the adjacency matrix.
`spatial_scale` | <code>[SpatialScale](#nltools.data.adjacency.spatial.SpatialScale) \| None</code> | 
`vector_shape` |  | Return shape of internal vectorized representation.

##### Methods

(data-append)=
###### `append`

```python
append(data)
```

Append data to an Adjacency instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (Adjacency) Adjacency instance to append | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (Adjacency) new appended Adjacency instance

(data-bootstrap)=
###### `bootstrap`

```python
bootstrap(stat, *, n_samples = 5000, save_boots = False, percentiles = (2.5, 97.5), n_jobs = -1, random_state = None)
```

Bootstrap statistics using efficient online algorithms.

Uses memory-efficient bootstrap infrastructure with CPU parallelization.
Supports simple aggregation statistics (mean, std, median, sum, min, max).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stat` |  | (str) Statistic to bootstrap. Options: - Simple stats: 'mean', 'median', 'std', 'sum', 'min', 'max' | *required*
`n_samples` |  | (int) Number of bootstrap iterations. Default: 5000 | <code>5000</code>
`save_boots` |  | (bool) If True, store all bootstrap samples (memory intensive).        Default: False | <code>False</code>
`percentiles` |  | (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5) | <code>(2.5, 97.5)</code>
`n_jobs` |  | (int) Number of CPU cores for parallelization. -1 means all CPUs. | <code>-1</code>
`random_state` |  | (int, optional) Random seed for reproducibility | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | Dictionary with keys: 'Z', 'p', 'mean', 'std', 'ci_lower', 'ci_upper'   (all Adjacency objects). If save_boots=True, also includes 'samples'.

**Examples:**

```pycon
>>> # Simple aggregation
>>> boot = adj.bootstrap(stat='mean', n_samples=1000)
>>> assert 'mean' in boot
>>> assert isinstance(boot['mean'], Adjacency)
```

(data-cluster-summary)=
###### `cluster_summary`

```python
cluster_summary(*, clusters = None, method = 'mean', summary = 'within')
```

Provide summaries of clusters within Adjacency matrices.

Computes mean/median of within and between cluster values. Requires a
list of cluster ids indicating the row/column of each cluster.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`clusters` |  | (list) list of cluster labels | <code>None</code>
`method` |  | (str) how to summarize, 'mean' or 'median'. If `None` then return all r values | <code>'mean'</code>
`summary` |  | (str) summarize within cluster or between clusters | <code>'within'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | within cluster means

(data-copy)=
###### `copy`

```python
copy()
```

Create a copy of Adjacency object.

(data-distance)=
###### `distance`

```python
distance(metric = 'correlation', include_diag = False, **kwargs)
```

Calculate distance between images within an Adjacency() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` |  | (str) type of distance metric (can use any scikit learn or     scipy metric) | <code>'correlation'</code>
`include_diag` |  | (bool) whether to include the main diagonal when     computing distances between adjacency matrices. Only applies     to symmetric matrices. Default False (consistent with how     symmetric matrices are stored without diagonal). | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dist` |  | (Adjacency) Outputs a 2D distance matrix.

(data-distance-to-similarity)=
###### `distance_to_similarity`

```python
distance_to_similarity(metric = 'correlation', beta = 1)
```

Convert distance matrix to similarity matrix.

Note: currently only implemented for correlation and euclidean.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` |  | (str) Can only be correlation or euclidean | <code>'correlation'</code>
`beta` |  | (float) parameter to scale exponential function (default: 1) for euclidean | <code>1</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (Adjacency) Adjacency object

(data-generate-permutations)=
###### `generate_permutations`

```python
generate_permutations(n_permute, random_state = None)
```

Generate permuted versions of an Adjacency instance lazily.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_permute` | <code>[int](#int)</code> | number of permutations | *required*
`random_state` | <code>([int](#int), [seed](#numpy.random.seed))</code> | random seed for reproducibility. | <code>None</code>

**Examples:**

```pycon
>>> for perm in adj.generate_permutations(1000):
>>>     out = neural_distance_mat.similarity(perm)
>>>     ...
```

**Yields:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | permuted version of self

(data-mean)=
###### `mean`

```python
mean(axis = 0)
```

Calculate mean of Adjacency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | Calculate mean over matrices (0) or upper triangle (1). | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float if single matrix, Adjacency if axis=0, np.array if axis=1.

(data-median)=
###### `median`

```python
median(axis = 0)
```

Calculate median of Adjacency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | Calculate median over matrices (0) or upper triangle (1). | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float if single matrix, Adjacency if axis=0, np.array if axis=1.

(data-plot)=
###### `plot`

```python
plot(limit = 3, axes = None, *args, **kwargs)
```

Create a heatmap of an Adjacency matrix.

Can pass in any sns.heatmap argument

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`limit` |  | (int) number of heatmaps to plot if object contains multiple adjacencies (default: 3) | <code>3</code>
`axes` |  | matplotlib axis handle | <code>None</code>

(data-plot-label-distance)=
###### `plot_label_distance`

```python
plot_label_distance(labels = None, ax = None)
```

Create a violin plot of within- and between-label distances.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`labels` | <code>[array](#numpy.array)</code> | numpy array of labels to plot | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`f` |  | violin plot handles

(data-plot-mds)=
###### `plot_mds`

```python
plot_mds(*, n_components = 2, metric_mds = True, labels = None, labels_color = None, cmap = None, view = (30, 20), figsize = None, ax = None, n_jobs = -1, **kwargs)
```

Plot multidimensional scaling.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_components` |  | (int) Number of dimensions to project (can be 2 or 3) | <code>2</code>
`metric_mds` |  | (bool) Perform metric (True) or non-metric (False) dimensional scaling; default True | <code>True</code>
`labels` |  | (list) Can override labels stored in Adjacency Class | <code>None</code>
`labels_color` |  | (str) list of colors for labels, if len(1) then make all same color | <code>None</code>
`cmap` |  | colormap instance (default: plt.cm.hot_r) | <code>None</code>
`view` |  | (tuple) view for 3-Dimensional plot; default (30,20) | <code>(30, 20)</code>
`figsize` |  | (list) figure size; default [12, 8] | <code>None</code>
`ax` |  | matplotlib axis handle | <code>None</code>
`n_jobs` |  | (int) Number of parallel jobs | <code>-1</code>

(data-plot-silhouette)=
###### `plot_silhouette`

```python
plot_silhouette(*, labels = None, ax = None, permutation_test = True, n_permute = 5000, colors = None, figsize = (6, 4))
```

Create a silhouette plot.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`labels` |  | Numpy array of cluster/group labels (overrides stored labels). | <code>None</code>
`ax` |  | Matplotlib axis handle. | <code>None</code>
`permutation_test` |  | (bool) Whether to run a permutation test. Default True. | <code>True</code>
`n_permute` |  | (int) Number of permutations for the test. Default 5000. | <code>5000</code>
`colors` |  | Optional list of RGB triplets, one per cluster (default: seaborn 'hls' palette). | <code>None</code>
`figsize` |  | Figure size tuple. Default (6, 4). | <code>(6, 4)</code>

(data-r-to-z)=
###### `r_to_z`

```python
r_to_z()
```

Apply Fisher's r-to-z transformation to each data element.

(data-regress)=
###### `regress`

```python
regress(X, method = 'ols')
```

Run a regression on an adjacency instance.
You can decompose an adjacency instance with another adjacency instance.
You can also decompose each pixel by passing a design_matrix instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` |  | Design matrix can be an Adjacency or DesignMatrix instance | *required*
`method` |  | type of regression (default: ols) - only 'ols' is currently supported | <code>'ols'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`stats` |  | (dict) dictionary of stats outputs.

(data-similarity)=
###### `similarity`

```python
similarity(data, *, plot = False, method = '2d', n_permute = 5000, metric = 'spearman', include_diag = False, nan_policy = 'omit', tail = 2, return_null = False, n_jobs = -1, random_state = None, project: bool = False)
```

Calculate similarity between two Adjacency matrices.

The default uses Spearman correlation and a permutation test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Adjacency](#nltools.data.adjacency.Adjacency) or [array](#array)</code> | Adjacency data, or 1-d array same size as self.data | *required*
`method` |  | (str) permutation scheme '1d', '2d', or None | <code>'2d'</code>
`metric` |  | (str) 'spearman','pearson','kendall' | <code>'spearman'</code>
`include_diag` |  | (bool) only applies to 'directed' Adjacency types using method=None or method='1d'. Default False (self-similarity is uninformative). Symmetric matrices never store the diagonal, so this flag is a no-op for them. | <code>False</code>
`nan_policy` |  | (str) How to handle NaN values. Options: - 'omit': Remove NaN values pairwise before computing correlation (default) - 'propagate': Allow NaN to propagate through calculations - 'raise': Raise an error if NaN values are present | <code>'omit'</code>
`tail` |  | (int) Tail of the test (1 or 2). Default 2. | <code>2</code>
`return_null` |  | (bool) If True, also return the null distribution. Default False. | <code>False</code>
`n_jobs` |  | (int) Number of parallel jobs. Default -1 (all cores). | <code>-1</code>
`random_state` |  | (int, optional) Random seed for reproducibility. | <code>None</code>
`project` | <code>[bool](#bool)</code> | (bool) If True and this Adjacency has a spatial_scale, project the per-matrix correlations back into brain space. Default False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
 | dict or list or BrainData: A correlation result dict with keys 'r' and 'p' for a single matrix, a list of such dicts when this Adjacency holds multiple matrices, or a `BrainData` when `project=True` (per-matrix correlations projected via spatial_scale).

(data-social-relations-model)=
###### `social_relations_model`

```python
social_relations_model(summarize_results = True, nan_replace = True)
```

Estimate the social relations model from a matrix for a round-robin design.

X_{ij} = m + \alpha_i + \beta_j + g_{ij} + \epsilon_{ijl}

where X_{ij} is the score for person i rating person j, m is the group mean,
\alpha_i  is person i's actor effect, \beta_j is person j's partner effect, g_{ij}
is the relationship  effect and \epsilon_{ijl} is the error in measure l  for actor i and partner j.

This model is primarily concerned with partioning the variance of the various effects.

Code is based on implementation presented in Chapter 8 of Kenny, Kashy, & Cook (2006).
Tests replicate examples  presented in the book. Note, that this method assumes that
actor scores are rows (lower triangle), while partner scores are columnns (upper triangle).
The minimal sample size to estimate these effects is 4.

<details class="model-assumptions" open markdown="1">
<summary>Model Assumptions</summary>

- Social interactions are exclusively dyadic
- People are randomly sampled from population
- No order effects
- The effects combine additively and relationships are linear

</details>

In the future we might update the formulas and standard errors based on
Bond and Lashley, 1996

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`self` |  | (adjacency) can be a single matrix or many matrices for each group | *required*
`summarize_results` |  | (bool) will provide a formatted summary of model results | <code>True</code>
`nan_replace` |  | (bool) will replace nan values with row and column means | <code>True</code>

**Returns:**

Type | Description
---- | -----------
 | estimated effects: (pd.Series/pd.DataFrame) All of the effects estimated using SRM

(data-squareform)=
###### `squareform`

```python
squareform()
```

Convert adjacency data back to square form.

(data-stats-label-distance)=
###### `stats_label_distance`

```python
stats_label_distance(*, labels = None, n_permute = 5000, n_jobs = -1)
```

Calculate permutation tests on within and between label distance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`labels` | <code>[array](#numpy.array)</code> | numpy array of labels to plot | <code>None</code>
`n_permute` | <code>[int](#int)</code> | number of permutations to run (default=5000) | <code>5000</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | dictionary of within and between group differences     and p-values

(data-std)=
###### `std`

```python
std(axis = 0)
```

Calculate standard deviation of Adjacency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | Calculate std over matrices (0) or upper triangle (1). | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float if single matrix, Adjacency if axis=0, np.array if axis=1.

(data-sum)=
###### `sum`

```python
sum(axis = 0)
```

Calculate sum of Adjacency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | Calculate sum over matrices (0) or upper triangle (1). | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float if single matrix, Adjacency if axis=0, np.array if axis=1.

(data-threshold)=
###### `threshold`

```python
threshold(*, upper = None, lower = None, binarize = False)
```

Threshold an Adjacency instance.

Provide upper and lower values or percentages to perform two-sided
thresholding. Binarize will return a mask image respecting thresholds
if provided, otherwise respecting every non-zero value.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`upper` |  | (float or str) Upper cutoff for thresholding. If string     will interpret as percentile; can be None for one-sided     thresholding. | <code>None</code>
`lower` |  | (float or str) Lower cutoff for thresholding. If string     will interpret as percentile; can be None for one-sided     thresholding. | <code>None</code>
`binarize` | <code>[bool](#bool)</code> | return binarized image respecting thresholds if     provided, otherwise binarize on every non-zero value;     default False | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | thresholded Adjacency instance

(data-to-brain)=
###### `to_brain`

```python
to_brain(values, *, fill: float = np.nan)
```

Project per-matrix scalars back to voxel-space `BrainData`.

Requires `spatial_scale` to be set (i.e. this stack came from
`BrainData.distance` or another spatial-scale-aware producer).
Each entry of ``values`` is painted onto the voxels assigned to its
corresponding parcel by ``spatial_scale.atlas`` /
``spatial_scale.roi_labels``. Voxels outside the atlas receive
``fill``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`values` |  | 1-D array of length ``len(self)`` — one scalar per matrix in the stack. | *required*
`fill` | <code>[float](#float)</code> | Value for voxels not covered by any provided ROI label. Default ``np.nan``. | <code>[nan](#numpy.nan)</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Single image masked to ``spatial_scale.source_mask``.

**Examples:**

```pycon
>>> rdms = brain.distance(metric='correlation',
...                       spatial_scale='roi', roi_mask=atlas)
>>> sims = rdms.similarity(model_rdm)
>>> brain_map = rdms.to_brain(sims)
```

(data-to-graph)=
###### `to_graph`

```python
to_graph()
```

Convert a single Adjacency matrix into a NetworkX graph.

This currently works only for ``single_matrix``.

(data-to-square)=
###### `to_square`

```python
to_square()
```

Convert adjacency back to square matrix format.

This is an alias for `squareform`.

**Returns:**

Type | Description
---- | -----------
 | np.ndarray or list: Square matrix representation. Returns a list
 | of matrices if this object contains multiple adjacency matrices.

(data-ttest)=
###### `ttest`

```python
ttest(*, permutation = False, n_permute = 5000, tail = 2, return_null = False, n_jobs = -1, random_state = None)
```

Calculate ttest across samples.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`permutation` |  | (bool) Run ttest as permutation. Note this can be very slow. | <code>False</code>
`n_permute` |  | Number of permutations (used only when ``permutation=True``). Default 5000. | <code>5000</code>
`tail` |  | Tail of the test (1 or 2). Default 2. | <code>2</code>
`return_null` |  | If True, also return the null distribution. Default False. | <code>False</code>
`n_jobs` |  | Number of parallel jobs. Default -1 (all cores). | <code>-1</code>
`random_state` |  | Random seed for reproducibility. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (dict) contains Adjacency instances of t values (or mean if  running permutation) and Adjacency instance of p values.

(data-write)=
###### `write`

```python
write(file_name, method = 'long')
```

Write out Adjacency object to csv file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>[str](#str)</code> | name of file name to write | *required*
`method` | <code>[str](#str)</code> | method to write out data ['long','square'] | <code>'long'</code>

(data-z-to-r)=
###### `z_to_r`

```python
z_to_r()
```

Convert each z score back into an r value.

(data-braincollection)=
#### `BrainCollection`

```python
BrainCollection(brains: list, *, mask: nib.Nifti1Image | Path | str, designs: list | None = None, metadata: pl.DataFrame | pd.DataFrame | dict | None = None, lazy: bool = True, cache_dir: Path | str | None = './.nltools_cache') -> None
```

Parallel, lazy iterator of ``BrainData`` whose API mirrors ``BrainData``.

Constructed via ``__init__`` (explicit lists) or one of the classmethod
factories (``from_bids``, ``from_glob``, ``from_paths``, ``read``).

See ``SPEC.md`` §"Public API" for the full contract; key invariants:
  - Per-subject ops route through ``execution._apply`` and return a
    lightweight clone via ``self._clone(...)`` over the same cache root.
  - Path-backed by default after parallel ops; ``cache='auto'`` follows
    source state. ``cache=`` is only accepted on collection-returning ops.
  - ``load`` / ``unload`` are the only methods that mutate ``self``.

Internal state (mutable list at top level; per-item slots are parallel):

  _items          list[BrainData | Path]        per-item brain data
  _mask           nib.Nifti1Image               shared mask (by reference)
  _designs        list[DesignMatrix | Path | None]
  _confounds      list[pd.DataFrame | None]
  _sample_masks   list[np.ndarray | None]
  _metadata       pl.DataFrame                  simple-typed columns only
  _cache_root     Path | None                   shared by clones
  _step_id        str | None                    this collection's step id
  _parent_step_id str | None                    upstream step id (lineage)
  _step_dirs      list[Path]                    lineage of step subdirs
                                                that produced these items
  _source_paths   list[Path | None]             per-item backing path
                                                (None for in-memory only)

**Methods:**

Name | Description
---- | -----------
[`align`](#data-align) | Functionally align subjects into a common space via `LocalAlignment`.
[`anova`](#data-anova) | One-way ANOVA across subjects grouped by ``groups``.
[`apply`](#data-apply) | Call ``BrainData.<op>(*args, **kwargs)`` on every item in parallel.
[`cleanup`](#data-cleanup) | Remove ``cache_root`` and invalidate every clone derived from ``self``.
[`cleanup_all`](#data-cleanup-all) | Remove every ``.nltools_cache/{run_id}/`` under ``directory``.
[`compute_contrasts`](#data-compute-contrasts) | Compute per-subject contrast maps from fit-bundle items.
[`concat`](#data-concat) | Stack all subject maps into a single `BrainData` (subjects as rows).
[`cv`](#data-cv) | Build a CV pipeline for cross-subject prediction.
[`detrend`](#data-detrend) | Detrend every subject's image in parallel (delegates to `BrainData.detrend`).
[`filter`](#data-filter) | Filter to a subset by predicate, polars expression, or boolean array.
[`fit`](#data-fit) | Per-subject fit; returns a path-backed collection of HDF5 fit bundles.
[`from_bids`](#data-from-bids) | Auto-pair BOLD with events.tsv (→ ``DesignMatrix``) and confounds.tsv.
[`from_glob`](#data-from-glob) | Build a collection by glob-matching brain images (and optional designs).
[`from_paths`](#data-from-paths) | Build a collection from explicit lists of brain (and design) paths.
[`isc`](#data-isc) | Inter-subject correlation (ISC) across the time dimension.
[`isc_test`](#data-isc-test) | Bootstrap inference on ISC (per-voxel p-values).
[`iter_pairs`](#data-iter-pairs) | Yield ``(BrainData, DesignMatrix | None)`` pairs.
[`load`](#data-load) | Materialize path-backed items in place. Returns ``self`` for chaining.
[`map`](#data-map) | Apply an arbitrary ``fn(BrainData) -> BrainData`` to each item in parallel.
[`max`](#data-max) | Voxelwise maximum across subjects as a single `BrainData`.
[`mean`](#data-mean) | Voxelwise mean across subjects as a single `BrainData`.
[`median`](#data-median) | Voxelwise median across subjects as a single `BrainData`.
[`memory_estimate`](#data-memory-estimate) | Human-readable RAM estimate if every item were loaded into memory.
[`min`](#data-min) | Voxelwise minimum across subjects as a single `BrainData`.
[`permutation_test`](#data-permutation-test) | One-sample sign-flipping permutation test across subjects.
[`permutation_test2`](#data-permutation-test2) | Two-sample permutation test between this collection and ``other``.
[`predict`](#data-predict) | Two distinct paths, dispatched by argument:
[`read`](#data-read) | Inverse of ``write()``. Does not recover from cache subdirs in v0.6.0.
[`resample`](#data-resample) | Resample every subject's image to a target space in parallel.
[`smooth`](#data-smooth) | Spatially smooth every subject's image in parallel (delegates to `BrainData.smooth`).
[`standardize`](#data-standardize) | Standardize every subject's image in parallel (delegates to `BrainData.standardize`).
[`std`](#data-std) | Voxelwise standard deviation across subjects as a single `BrainData`.
[`steps`](#data-steps) | Step subdirs that produced this collection's items, oldest to newest.
[`sum`](#data-sum) | Voxelwise sum across subjects as a single `BrainData`.
[`threshold`](#data-threshold) | Threshold every subject's image in parallel (delegates to `BrainData.threshold`).
[`transform_designs`](#data-transform-designs) | Map ``fn(dm) -> DesignMatrix`` over each paired design.
[`ttest`](#data-ttest) | One-sample t-test across subjects (delegates to `inference.ttest`).
[`ttest2`](#data-ttest2) | Two-sample t-test between this collection and ``other`` (subject-level).
[`unload`](#data-unload) | Drop in-memory data for items with backing paths. Returns ``self``.
[`var`](#data-var) | Voxelwise variance across subjects as a single `BrainData`.
[`write`](#data-write) | Write a clean, portable copy of the collection outside the cache root.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`cache_root`](#data-cache-root) | <code>[Path](#pathlib.Path)</code> | Run-scoped cache directory shared by clones. Raises if unset.
`designs` | <code>[list](#list)</code> | Per-subject paired designs (a copy of the list; ``None`` where unpaired).
`is_loaded` | <code>[list](#list)[[bool](#bool)]</code> | Per-item flag — True iff the slot holds a ``BrainData`` (not a path).
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image)</code> | Shared mask image for the collection. Raises if the mask is unset.
`metadata` | <code>[DataFrame](#polars.DataFrame)</code> | Per-subject metadata as a polars DataFrame (one row per item).
`n_subjects` | <code>[int](#int)</code> | Number of subjects (items) in the collection.
`n_voxels` | <code>[int](#int)</code> | Voxel count from the mask. Raises if mask is unset.
`shape` | <code>[tuple](#tuple)[[int](#int), [int](#int) \| None, [int](#int)]</code> | ``(n_subjects, n_obs_or_None_if_ragged, n_voxels)``.

``cache_dir`` precedence: explicit arg → ``NLTOOLS_CACHE_DIR`` env →
``./.nltools_cache``. Pass ``None`` for an auto-cleaned tempdir.
Resolved at construction and frozen on the instance.

##### Methods

(data-align)=
###### `align`

```python
align(*, method: str = 'procrustes', spatial_scale: str = 'searchlight', radius_mm: float = 10.0, roi_mask: nib.Nifti1Image | None = None, n_features: int | None = None, n_iter: int = 3, device: str = 'cpu', return_model: bool = False, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto')
```

Functionally align subjects into a common space via `LocalAlignment`.

Materializes all subjects (algorithm constraint in v0.6.0).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Alignment solver (e.g. ``'procrustes'``). | <code>'procrustes'</code>
`spatial_scale` | <code>[str](#str)</code> | Alignment scope (``'searchlight'``, ``'roi'``, ``'whole_brain'``). | <code>'searchlight'</code>
`radius_mm` | <code>[float](#float)</code> | Searchlight sphere radius in mm. | <code>10.0</code>
`roi_mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| None</code> | Parcellation/ROI mask (used when ``spatial_scale='roi'``). | <code>None</code>
`n_features` | <code>[int](#int) \| None</code> | Optional target feature count for the common space. | <code>None</code>
`n_iter` | <code>[int](#int)</code> | LocalAlignment solver iteration count (not a permutation count). | <code>3</code>
`device` | <code>[str](#str)</code> | Backend selector (``'cpu'``/``'gpu'``). | <code>'cpu'</code>
`return_model` | <code>[bool](#bool)</code> | If True, also return the fitted `LocalAlignment`. | <code>False</code>
`n_jobs` | <code>[int](#int)</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>[bool](#bool)</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>[Literal](#typing.Literal)['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
 | A new `BrainCollection` of aligned data, or a
 | ``(BrainCollection, LocalAlignment)`` tuple when
 | ``return_model=True``.

(data-anova)=
###### `anova`

```python
anova(groups: str | list | np.ndarray) -> dict
```

One-way ANOVA across subjects grouped by ``groups``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`groups` | <code>[str](#str) \| [list](#list) \| [ndarray](#numpy.ndarray)</code> | A metadata column name, or a list/ndarray of length ``n_subjects`` giving each subject's group label. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict with ``{'F', 'p'}`` `BrainData` maps plus ``df_between`` and
<code>[dict](#dict)</code> | ``df_within`` degrees of freedom.

(data-apply)=
###### `apply`

```python
apply(op: str, *args: str, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto', **kwargs: Literal['auto', True, False]) -> BrainCollection
```

Call ``BrainData.<op>(*args, **kwargs)`` on every item in parallel.

All per-subject methods (``smooth``, ``standardize``, ...) reduce to
this. Centralizes the ``_apply`` plumbing and the cache-knob handling.
``op`` is named ``op`` (not ``method``) to avoid colliding with
``BrainData`` methods that themselves take a ``method=`` kwarg
(``standardize``, ``detrend``, ...).

(data-cleanup)=
###### `cleanup`

```python
cleanup() -> None
```

Remove ``cache_root`` and invalidate every clone derived from ``self``.

Idempotent — calling twice is a no-op. Path-backed items in any
clone become unloadable after this; use ``bc.write(...)`` first to
materialize a portable copy if needed.

(data-cleanup-all)=
###### `cleanup_all`

```python
cleanup_all(directory: Path | str = '.') -> None
```

Remove every ``.nltools_cache/{run_id}/`` under ``directory``.

Wide brush — can kill sibling sessions in the same cwd. Prefer
``bc.cleanup()`` for surgical removal.

(data-compute-contrasts)=
###### `compute_contrasts`

```python
compute_contrasts(contrasts: str | list[str] | dict[str, np.ndarray], *, statistic: str = 'beta', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection | dict[str, BrainCollection] | dict[str, dict[str, BrainCollection]]
```

Compute per-subject contrast maps from fit-bundle items.

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)] \| [dict](#dict)[[str](#str), [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]]</code> | single contrast + single ``statistic`` → ``BrainCollection``
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)] \| [dict](#dict)[[str](#str), [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]]</code> | multiple contrasts (single type)            → ``dict[str, BrainCollection]``
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)] \| [dict](#dict)[[str](#str), [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]]</code> | ``statistic='all'`` (single contrast)   → ``dict['beta'|'t'|'z'|'p'|'se', BrainCollection]``
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)] \| [dict](#dict)[[str](#str), [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]]</code> | multiple contrasts + ``statistic='all'`` → nested                                              ``dict[name, dict[stat, BrainCollection]]``

Each per-subject NIfTI gets a JSON sidecar with lineage attrs
(``step_id``, ``parent_step_id``, ``op``, ``kwargs``,
``nltools_version``).

(data-concat)=
###### `concat`

```python
concat() -> BrainData
```

Stack all subject maps into a single `BrainData` (subjects as rows).

(data-cv)=
###### `cv`

```python
cv(*, k: int | None = None, method: str = 'kfold', split_by: str | None = None, groups: np.ndarray | None = None, n: int = 1000, random_state: int | None = None) -> BrainCollectionPipeline
```

Build a CV pipeline for cross-subject prediction.

See ``pipeline.py`` for the builder API. The pipeline's ``predict``
terminal returns a ``BrainData`` with CV attrs attached.

(data-detrend)=
###### `detrend`

```python
detrend(*, method: str = 'linear', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Detrend every subject's image in parallel (delegates to `BrainData.detrend`).

(data-filter)=
###### `filter`

```python
filter(predicate: Callable[[Any], Any] | list | np.ndarray | pl.Series | pd.Series) -> BrainCollection
```

Filter to a subset by predicate, polars expression, or boolean array.

(data-fit)=
###### `fit`

```python
fit(model: str = 'glm', X: DesignMatrix | list | Callable | None = None, *, scale: bool | str = 'auto', standardize: str | None = 'auto', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto', **model_kwargs: Literal['auto', True, False]) -> BrainCollection
```

Per-subject fit; returns a path-backed collection of HDF5 fit bundles.

``X`` resolution priority:
  - ``None``         → use ``self.designs`` (must be set per subject)
  - ``DesignMatrix`` → shared across all subjects
  - ``list``         → per-subject (len == n_subjects)
  - ``callable``     → ``fn(ctx: _DesignContext) -> DesignMatrix``

(data-from-bids)=
###### `from_bids`

```python
from_bids(root: Path | str | Any, *, mask: nib.Nifti1Image | Path | str, task: str | None = None, space: str | None = None, sub_labels: list[str] | None = None, img_filters: list[tuple[str, str]] | None = None, derivatives_folder: str = 'derivatives', pair_events: bool = True, confounds_strategy: str | tuple[str, ...] | None = None, confounds_kwargs: dict | None = None, TR: float | str = 'infer', cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Auto-pair BOLD with events.tsv (→ ``DesignMatrix``) and confounds.tsv.

Full design and edge cases: SPEC §"``from_bids`` — concrete design".

(data-from-glob)=
###### `from_glob`

```python
from_glob(pattern: str, *, mask: nib.Nifti1Image | Path | str, design_pattern: str | None = None, pattern_groups: dict[str, int] | str | None = None, sort: bool = True, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Build a collection by glob-matching brain images (and optional designs).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`pattern` | <code>[str](#str)</code> | Glob pattern matching the per-subject brain image files. | *required*
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str)</code> | Shared mask image, path, or nltools template name. | *required*
`design_pattern` | <code>[str](#str) \| None</code> | Optional glob matching per-subject design files, paired positionally with the brain images. | <code>None</code>
`pattern_groups` | <code>[dict](#dict)[[str](#str), [int](#int)] \| [str](#str) \| None</code> | Regex capture-group spec used to extract metadata (e.g. subject/run) from each matched path. | <code>None</code>
`sort` | <code>[bool](#bool)</code> | If True, sort matched paths before pairing (stable ordering). | <code>True</code>
`cache_dir` | <code>[Path](#pathlib.Path) \| [str](#str) \| None</code> | Cache-directory precedence: explicit arg → ``NLTOOLS_CACHE_DIR`` env → ``./.nltools_cache``; ``None`` for a temp dir. | <code>'./.nltools_cache'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | A lazy, path-backed `BrainCollection`.

(data-from-paths)=
###### `from_paths`

```python
from_paths(brain_paths: list, *, mask: nib.Nifti1Image | Path | str, design_paths: list | None = None, metadata: pl.DataFrame | pd.DataFrame | dict | None = None, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Build a collection from explicit lists of brain (and design) paths.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_paths` | <code>[list](#list)</code> | Per-subject brain image paths. | *required*
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str)</code> | Shared mask image, path, or nltools template name. | *required*
`design_paths` | <code>[list](#list) \| None</code> | Optional per-subject design paths, aligned positionally with ``brain_paths`` (length must match, ``None`` entries allowed). | <code>None</code>
`metadata` | <code>[DataFrame](#polars.DataFrame) \| [DataFrame](#pandas.DataFrame) \| [dict](#dict) \| None</code> | Optional per-subject metadata (polars/pandas DataFrame or dict-of-columns), one row per path. | <code>None</code>
`cache_dir` | <code>[Path](#pathlib.Path) \| [str](#str) \| None</code> | Cache-directory precedence: explicit arg → ``NLTOOLS_CACHE_DIR`` env → ``./.nltools_cache``; ``None`` for a temp dir. | <code>'./.nltools_cache'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | A lazy, path-backed `BrainCollection`.

(data-isc)=
###### `isc`

```python
isc(*, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, metric: str = 'median') -> dict
```

Inter-subject correlation (ISC) across the time dimension.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ``'loo'`` (leave-one-out template) or ``'pairwise'`` (all subject pairs). | <code>'loo'</code>
`roi_mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str) \| None</code> | Optional ROI/atlas mask restricting the computation to those voxels. The returned maps carry the ROI mask. If None, ISC is computed across the collection's whole-brain mask. | <code>None</code>
`metric` | <code>[str](#str)</code> | Aggregation across subjects/pairs (e.g. ``'median'``). | <code>'median'</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict ``{'isc', 'per_subject'}`` for ``method='loo'`` or
<code>[dict](#dict)</code> | ``{'isc', 'pairs'}`` for ``method='pairwise'`` (``'isc'`` is a
<code>[dict](#dict)</code> | `BrainData` map).

(data-isc-test)=
###### `isc_test`

```python
isc_test(*, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, n_samples: int = 5000, metric: str = 'median', random_state: int | None = None) -> dict
```

Bootstrap inference on ISC (per-voxel p-values).

Resamples subjects with replacement, recomputes ISC each draw, and
derives a per-voxel two-tailed p-value from the null centered at 0.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ``'loo'`` or ``'pairwise'`` (matches `isc`). | <code>'loo'</code>
`roi_mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str) \| None</code> | Optional ROI/atlas mask restricting the computation to those voxels. The returned maps carry the ROI mask. If None, ISC is computed across the collection's whole-brain mask. | <code>None</code>
`n_samples` | <code>[int](#int)</code> | Number of bootstrap resamples. | <code>5000</code>
`metric` | <code>[str](#str)</code> | Aggregation across subjects/pairs (e.g. ``'median'``). | <code>'median'</code>
`random_state` | <code>[int](#int) \| None</code> | Seed for the bootstrap RNG. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict ``{'isc', 'p', 'null_distribution'}`` (``'isc'`` and ``'p'`` are
<code>[dict](#dict)</code> | `BrainData` maps).

(data-iter-pairs)=
###### `iter_pairs`

```python
iter_pairs() -> Iterator[tuple]
```

Yield ``(BrainData, DesignMatrix | None)`` pairs.

(data-load)=
###### `load`

```python
load(indices: list[int] | None = None) -> BrainCollection
```

Materialize path-backed items in place. Returns ``self`` for chaining.

(data-map)=
###### `map`

```python
map(fn: Callable, *, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Apply an arbitrary ``fn(BrainData) -> BrainData`` to each item in parallel.

(data-max)=
###### `max`

```python
max() -> BrainData
```

Voxelwise maximum across subjects as a single `BrainData`.

###### `mean`

```python
mean() -> BrainData
```

Voxelwise mean across subjects as a single `BrainData`.

###### `median`

```python
median() -> BrainData
```

Voxelwise median across subjects as a single `BrainData`.

(data-memory-estimate)=
###### `memory_estimate`

```python
memory_estimate() -> str
```

Human-readable RAM estimate if every item were loaded into memory.

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | A string reporting ``n_subjects``, the per-item shape (or "unknown"
<code>[str](#str)</code> | for path-backed items not yet loaded), and an estimated float32
<code>[str](#str)</code> | total in MB/GB.

(data-min)=
###### `min`

```python
min() -> BrainData
```

Voxelwise minimum across subjects as a single `BrainData`.

(data-permutation-test)=
###### `permutation_test`

```python
permutation_test(*, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

One-sample sign-flipping permutation test across subjects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_permute` | <code>[int](#int)</code> | Number of sign-flip permutations. | <code>5000</code>
`tail` | <code>[int](#int)</code> | 1 for one-tailed, 2 for two-tailed. | <code>2</code>
`device` | <code>[str](#str)</code> | Backend selector (currently informational). | <code>'cpu'</code>
`return_null` | <code>[bool](#bool)</code> | If True, include the null distribution in the result. | <code>False</code>
`n_jobs` | <code>[int](#int)</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Seed for the sign-flip RNG. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict ``{'mean', 'p'}`` of `BrainData` maps, plus
<code>[dict](#dict)</code> | ``'null_distribution'`` when ``return_null=True``.

(data-permutation-test2)=
###### `permutation_test2`

```python
permutation_test2(other: BrainCollection, *, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

Two-sample permutation test between this collection and ``other``.

Uses random label shuffling of the pooled subjects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | The second collection to compare against. | *required*
`n_permute` | <code>[int](#int)</code> | Number of label-shuffle permutations. | <code>5000</code>
`tail` | <code>[int](#int)</code> | 1 for one-tailed, 2 for two-tailed. | <code>2</code>
`device` | <code>[str](#str)</code> | Backend selector (currently informational). | <code>'cpu'</code>
`return_null` | <code>[bool](#bool)</code> | If True, include the null distribution in the result. | <code>False</code>
`n_jobs` | <code>[int](#int)</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Seed for the shuffling RNG. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict ``{'mean', 'p'}`` of `BrainData` maps (``mean`` is the group
<code>[dict](#dict)</code> | difference), plus ``'null_distribution'`` when ``return_null=True``.

(data-predict)=
###### `predict`

```python
predict(y: str | list | np.ndarray | None = None, *, X_new: np.ndarray | None = None, spatial_scale: str = 'whole_brain', model: str = 'svm', cv: int | str = 'loso', groups: str | np.ndarray | None = None, roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float = 10.0, scoring: str = 'auto', standardize: bool = True, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto')
```

Two distinct paths, dispatched by argument:

  ``y=`` only    → group MVPA (subjects as samples) → ``Predict``
  ``X_new=`` only → per-subject predict-after-fit  → ``BrainCollection``
  both / neither → raise

``predict(y=...)`` requires single-map-per-subject items (run
``compute_contrasts(...)`` first if you have GLM/ridge bundles).

(data-read)=
###### `read`

```python
read(directory: Path | str, *, mask: nib.Nifti1Image | Path | str, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Inverse of ``write()``. Does not recover from cache subdirs in v0.6.0.

(data-resample)=
###### `resample`

```python
resample(target, *, interpolation: str = 'continuous', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Resample every subject's image to a target space in parallel.

Delegates to `BrainData.resample`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` |  | Resampling target (image, affine/shape spec, or template) passed through to `BrainData.resample`. | *required*
`interpolation` | <code>[str](#str)</code> | Interpolation method (``'continuous'``, ``'linear'``, ``'nearest'``). | <code>'continuous'</code>
`n_jobs` | <code>[int](#int)</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>[bool](#bool)</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>[Literal](#typing.Literal)['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | A new `BrainCollection` of resampled items.

(data-smooth)=
###### `smooth`

```python
smooth(fwhm: float, *, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Spatially smooth every subject's image in parallel (delegates to `BrainData.smooth`).

(data-standardize)=
###### `standardize`

```python
standardize(*, axis: int = 0, method: str = 'center', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Standardize every subject's image in parallel (delegates to `BrainData.standardize`).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>[int](#int)</code> | Axis along which to standardize (0 = across observations). | <code>0</code>
`method` | <code>[str](#str)</code> | Standardization variant (e.g. ``'center'``, ``'zscore'``). | <code>'center'</code>
`n_jobs` | <code>[int](#int)</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>[bool](#bool)</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>[Literal](#typing.Literal)['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | A new `BrainCollection` of standardized items.

###### `std`

```python
std() -> BrainData
```

Voxelwise standard deviation across subjects as a single `BrainData`.

(data-steps)=
###### `steps`

```python
steps() -> list[Path]
```

Step subdirs that produced this collection's items, oldest to newest.

Lineage chain accumulated through clones (one entry per upstream
cached op). Empty when the collection was constructed directly or
no ancestor wrote to disk.

###### `sum`

```python
sum() -> BrainData
```

Voxelwise sum across subjects as a single `BrainData`.

###### `threshold`

```python
threshold(*, lower: float | None = None, upper: float | None = None, binarize: bool = False, coerce_nan: bool = True, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Threshold every subject's image in parallel (delegates to `BrainData.threshold`).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`lower` | <code>[float](#float) \| None</code> | Values below this are zeroed (or set NaN); ``None`` disables. | <code>None</code>
`upper` | <code>[float](#float) \| None</code> | Values above this are zeroed (or set NaN); ``None`` disables. | <code>None</code>
`binarize` | <code>[bool](#bool)</code> | If True, set surviving voxels to 1. | <code>False</code>
`coerce_nan` | <code>[bool](#bool)</code> | If True, coerce thresholded-out voxels to NaN instead of 0. | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>[bool](#bool)</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>[Literal](#typing.Literal)['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | A new `BrainCollection` of thresholded items.

(data-transform-designs)=
###### `transform_designs`

```python
transform_designs(fn: Callable, *, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Map ``fn(dm) -> DesignMatrix`` over each paired design.

Items with no paired design are skipped (kept as ``None``). Runs in
the parent process — designs are small. ``n_jobs``/``progress_bar``/
``cache`` are accepted for surface consistency but ignored.

###### `ttest`

```python
ttest(*, popmean: float = 0.0) -> dict
```

One-sample t-test across subjects (delegates to `inference.ttest`).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`popmean` | <code>[float](#float)</code> | Null-hypothesis population mean to test against. | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict ``{'mean', 't', 'z', 'p'}`` of `BrainData` maps.

(data-ttest2)=
###### `ttest2`

```python
ttest2(other: BrainCollection, *, equal_var: bool = True) -> dict
```

Two-sample t-test between this collection and ``other`` (subject-level).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | The second collection to compare against. | *required*
`equal_var` | <code>[bool](#bool)</code> | If True, pooled-variance t-test; if False, Welch's test. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict ``{'mean', 't', 'z', 'p'}`` of `BrainData` maps (``mean`` is the
<code>[dict](#dict)</code> | group difference).

(data-unload)=
###### `unload`

```python
unload(indices: list[int] | None = None) -> BrainCollection
```

Drop in-memory data for items with backing paths. Returns ``self``.

(data-var)=
###### `var`

```python
var() -> BrainData
```

Voxelwise variance across subjects as a single `BrainData`.

###### `write`

```python
write(directory: Path | str, *, pattern: str = 'image_{i:04d}.nii.gz', metadata_file: str | None = 'metadata.csv') -> list[Path]
```

Write a clean, portable copy of the collection outside the cache root.

Inverse of `BrainCollection.read`. Writes one NIfTI per item plus an
optional metadata CSV, skipping the internal cache layout so the result
is shareable/archival.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`directory` | <code>[Path](#pathlib.Path) \| [str](#str)</code> | Output directory (created if missing). | *required*
`pattern` | <code>[str](#str)</code> | Filename template per item, formatted with ``i`` (item index). | <code>'image_{i:04d}.nii.gz'</code>
`metadata_file` | <code>[str](#str) \| None</code> | CSV filename for the metadata table, or ``None`` to skip. | <code>'metadata.csv'</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[Path](#pathlib.Path)]</code> | List of written NIfTI paths, in item order.

(data-braindata)=
#### `BrainData`

```python
BrainData(data = None, *, Y = None, X = None, mask = None, masker = None, h5_compression = 'gzip', verbose = False, resample = True, interpolation = 'auto')
```

Represent neuroimaging data as vectors instead of three-dimensional matrices.

This representation makes it easier to perform data manipulation and analyses.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | Neuroimaging data. Can be: - None (empty BrainData) - BrainData object - List of BrainData objects or file paths - File path (str/Path) to .nii/.nii.gz/.h5/.hdf5 - nibabel Nifti1Image object - URL to download data from - numpy array (1D ``(n_voxels,)`` for a single image or 2D   ``(n_images, n_voxels)`` for a stack). The ``mask`` argument   is required and must define the same number of in-mask voxels. | <code>None</code>
`mask` |  | Brain mask. Can be None (uses MNI template), a nibabel Nifti1Image, a file path (str/Path) to a mask file, or a template name string like ``'2mm-MNI152-2009c'`` (version: 'fsl' for default/, 'a' for nilearn/, 'c' for fmriprep/). | <code>None</code>
`masker` |  | nilearn masker object (e.g. ROI or searchlight extractor). Default will load data as voxels. | <code>None</code>
`resample` | <code>bool, default=True</code> | Whether to automatically resample data to mask space. If True, data is resampled to match mask spatial characteristics. If False, data must already be in mask space. Default True preserves backward compatibility with v0.5.1. | <code>True</code>
`interpolation` | <code>str, default='auto'</code> | Interpolation method for resampling. Options: 'auto' (detect based on data type; uses 'nearest' for discrete data like atlases/masks and 'continuous' for stat maps), 'nearest' (nearest-neighbor, preserves discrete values), 'linear' (linear interpolation), 'continuous' (higher-order spline, use for stat maps). | <code>'auto'</code>

**Methods:**

Name | Description
---- | -----------
[`align`](#data-align) | Align BrainData instance to target object using functional alignment.
[`append`](#data-append) | Append data to BrainData instance.
[`apply_mask`](#data-apply-mask) | Mask BrainData instance using nilearn functionality.
[`astype`](#data-astype) | Cast BrainData.data as type.
[`bootstrap`](#data-bootstrap) | Bootstrap statistics using efficient online algorithms.
[`cluster_report`](#data-cluster-report) | Generate a cluster report with anatomical labels.
[`compute_contrasts`](#data-compute-contrasts) | Compute contrasts from fitted GLM results.
[`copy`](#data-copy) | Create a deep copy of a BrainData instance.
[`create_empty`](#data-create-empty) | Create a copy of BrainData with empty data array.
[`decompose`](#data-decompose) | Decompose BrainData object.
[`detrend`](#data-detrend) | Remove linear trend from each voxel.
[`distance`](#data-distance) | Calculate distance between images within a BrainData() instance.
[`extract_roi`](#data-extract-roi) | Extract activity from mask or ROI atlas using NiftiLabelsMasker.
[`filter`](#data-filter) | Apply butterworth filter to data. Wraps nilearn.signal.clean.
[`find_spikes`](#data-find-spikes) | Identify spikes from Time Series Data.
[`fit`](#data-fit) | Fit a model to brain imaging data.
[`iplot`](#data-iplot) | Interactive WebGL brain viewer powered by niivue (`ipyniivue`).
[`mean`](#data-mean) | Get mean of each voxel or image.
[`median`](#data-median) | Get median of each voxel or image.
[`multivariate_similarity`](#data-multivariate-similarity) | Predict a BrainData spatial distribution from a linear combination.
[`plot`](#data-plot) | Plot BrainData instance using nilearn visualization or matplotlib.
[`plot_flatmap`](#data-plot-flatmap) | Plot brain data on cortical flatmap.
[`plot_surf`](#data-plot-surf) | Render this BrainData on fsaverage surfaces as a tight 2×2 montage.
[`predict`](#data-predict) | Predict voxel timeseries (encoding) or decode labels (MVPA).
[`r_to_z`](#data-r-to-z) | Apply Fisher's r-to-z transformation to each data element.
[`regions`](#data-regions) | Extract brain connected regions into separate regions.
[`report`](#data-report) | Generate a nilearn HTML report for a fitted GLM.
[`resample_to`](#data-resample-to) | Resample BrainData to match target image or resolution.
[`scale`](#data-scale) | Scale data via mean scaling.
[`similarity`](#data-similarity) | Calculate similarity to a single BrainData or nibabel image.
[`smooth`](#data-smooth) | Apply spatial smoothing using nilearn smooth_img().
[`standardize`](#data-standardize) | Standardize BrainData() instance.
[`std`](#data-std) | Get standard deviation of each voxel or image.
[`sum`](#data-sum) | Get sum of each voxel or image.
[`temporal_resample`](#data-temporal-resample) | Resample BrainData timeseries to a new target frequency or number of samples.
[`threshold`](#data-threshold) | Threshold BrainData instance with optional cluster filtering.
[`to_nifti`](#data-to-nifti) | Convert BrainData Instance into Nifti Object.
[`transform_pairwise`](#data-transform-pairwise) | Transform data into pairwise comparisons.
[`ttest`](#data-ttest) | One-sample voxelwise t-test across images (axis 0).
[`ttest2`](#data-ttest2) | Two-sample voxelwise t-test between two BrainData stacks.
[`upload_neurovault`](#data-upload-neurovault) | Upload BrainData images and metadata to NeuroVault.
[`write`](#data-write) | Write out BrainData object to Nifti or HDF5 File.
[`z_to_r`](#data-z-to-r) | Convert z score back into r value for each element of data object.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`X`](#data-x) |  | Design matrix / per-image covariates as a polars DataFrame.
[`Y`](#data-y) |  | Per-image targets as a polars DataFrame.
[`data`](#data-data) |  | 
`design_matrix` |  | 
`dtype` |  | Get data type of BrainData.data.
`is_empty` | <code>[bool](#bool)</code> | Check if BrainData.data is empty.
`masker` |  | 
`shape` |  | Get images by voxels shape.
`size` |  | Total number of elements in BrainData.data (numpy convention).
`verbose` |  | 

##### Methods

###### `align`

```python
align(target, method = 'procrustes', axis = 0, *, spatial_scale: str = 'whole_brain', roi_mask: str = None, radius_mm: float = 10.0)
```

Align BrainData instance to target object using functional alignment.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` |  | (BrainData) object to align to. | *required*
`method` |  | (str) alignment method to use ['probabilistic_srm','deterministic_srm','procrustes'] | <code>'procrustes'</code>
`axis` |  | (int) axis to align on | <code>0</code>
`spatial_scale` | <code>[str](#str)</code> | ``'whole_brain'`` (default), ``'roi'``, or ``'searchlight'``. ``'roi'`` is supported (per-parcel transforms + reassembly, requires `roi_mask`). ``'searchlight'`` is not yet implemented (overlapping spheres have no canonical per-voxel transform). | <code>'whole_brain'</code>
`roi_mask` |  | Atlas image used when `spatial_scale='roi'`. | <code>None</code>
`radius_mm` | <code>[float](#float)</code> | Reserved for ``spatial_scale='searchlight'``. | <code>10.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (dict) a dictionary containing transformed object, transformation matrix, and the shared response matrix

**Examples:**

```pycon
>>> out = data.align(target, method='procrustes')
>>> out = data.align(target, method='probabilistic_srm')
```

###### `append`

```python
append(data, ignore_attrs = False, **kwargs)
```

Append data to BrainData instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | BrainData instance to append. | *required*
`ignore_attrs` |  | (bool) If True, skip concatenation of X and Y     attributes. Useful when appending images where .X or .Y     have different column counts. Default False. | <code>False</code>
`kwargs` |  | Optional arguments passed to pandas concat for X/Y. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | New appended BrainData instance.

(data-apply-mask)=
###### `apply_mask`

```python
apply_mask(mask, resample_mask_to_brain = False)
```

Mask BrainData instance using nilearn functionality.

Note target data will be resampled into the same space as the mask. If you would like the mask
resampled into the BrainData space, then set resample_mask_to_brain=True.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask` |  | (BrainData or nifti object) mask to apply to BrainData object. | *required*
`resample_mask_to_brain` |  | (bool) Will resample mask to brain space before applying mask (default=False). | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`masked` |  | (BrainData) masked BrainData object

(data-astype)=
###### `astype`

```python
astype(dtype)
```

Cast BrainData.data as type.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dtype` |  | datatype to convert | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with new datatype

###### `bootstrap`

```python
bootstrap(stat, *, n_samples = 5000, save_boots = False, percentiles = (2.5, 97.5), X_test = None, backend = None, max_gpu_memory_gb = 4.0, n_jobs = -1, random_state = None)
```

Bootstrap statistics using efficient online algorithms.

Uses memory-efficient bootstrap infrastructure with CPU parallelization or GPU acceleration.
Supports simple aggregation statistics and fitted model statistics (Ridge).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stat` |  | (str) Statistic to bootstrap. Options: Simple stats ('mean', 'median', 'std', 'sum', 'min', 'max') or Model stats ('weights' requires fitted Ridge model, 'predict' requires fitted Ridge model + X_test). | *required*
`n_samples` |  | (int) Number of bootstrap iterations. Default: 5000 | <code>5000</code>
`save_boots` |  | (bool) If True, store all bootstrap samples. Default: False | <code>False</code>
`percentiles` |  | (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5) | <code>(2.5, 97.5)</code>
`X_test` |  | (np.ndarray, optional) Test features for 'predict' bootstrap. | <code>None</code>
`backend` |  | (str, optional) Backend for Ridge bootstrap: None (CPU), 'torch' (GPU if available), or 'auto' (auto-select). Ignored for simple stats. | <code>None</code>
`max_gpu_memory_gb` |  | (float) Maximum GPU memory to use when backend is 'torch' or 'auto'. Default: 4.0 | <code>4.0</code>
`n_jobs` |  | (int) Number of CPU cores for parallelization. -1 means all CPUs. | <code>-1</code>
`random_state` |  | (int, optional) Random seed for reproducibility | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or dict: - For simple stats: Returns BrainData with bootstrap mean - For model stats: Returns dict with keys: 'mean', 'std', 'Z', 'p',   'ci_lower', 'ci_upper' (all BrainData objects) - If ``save_boots=True``: Returns dict with 'samples' key containing all samples

**Examples:**

```pycon
>>> boot = brain.bootstrap(stat='mean', n_samples=1000)
>>> brain.fit(X=dm, model='ridge', alpha=1.0)
>>> boot = brain.bootstrap(stat='weights', n_samples=1000)
```

(data-cluster-report)=
###### `cluster_report`

```python
cluster_report(*, stat_threshold: float | None = 3.0, cluster_threshold: int = 10, two_sided: bool = True, min_distance: float = 8.0, atlas: str | Sequence[str] | None = None, prob_threshold: float = 5.0) -> ClusterReport
```

Generate a cluster report with anatomical labels.

Identifies surviving clusters in the stat map (after voxel + extent
thresholding), reports peak coordinates and sub-peaks, and labels
each peak/cluster against one or more atlases.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stat_threshold` | <code>[float](#float) \| None</code> | Voxel-level threshold (e.g. z- or t-cutoff). ``None`` treats ``self`` as already thresholded. | <code>3.0</code>
`cluster_threshold` | <code>[int](#int)</code> | Minimum cluster size in voxels. | <code>10</code>
`two_sided` | <code>[bool](#bool)</code> | Report negative clusters separately. | <code>True</code>
`min_distance` | <code>[float](#float)</code> | Minimum mm between sub-peaks within a cluster. | <code>8.0</code>
`atlas` | <code>[str](#str) \| [Sequence](#collections.abc.Sequence)[[str](#str)] \| None</code> | Atlas name or list of names (see `list_atlases`). Defaults to ``("harvard_oxford", "aal", "schaefer_200")``. | <code>None</code>
`prob_threshold` | <code>[float](#float)</code> | Drop probabilistic-atlas regions below this %. | <code>5.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[ClusterReport](#nltools.data.atlases.ClusterReport)</code> | `ClusterReport` with ``peaks``,
<code>[ClusterReport](#nltools.data.atlases.ClusterReport)</code> | ``clusters`` (polars DataFrames), and ``stat_img`` (BrainData).

###### `compute_contrasts`

```python
compute_contrasts(contrasts, statistic = 't')
```

Compute contrasts from fitted GLM results.

This method computes contrasts as linear combinations of the GLM beta coefficients.
Must be called after .fit(model='glm', X=design_matrix) has been run.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`contrasts` |  | Can be:<br>- str: A string specifying the contrast using column names   e.g., "conditionA - conditionB" or "2*conditionA - conditionB - conditionC" - dict: Dictionary with contrast names as keys and contrast strings/vectors as values   e.g., {"main_effect": "conditionA - conditionB", "interaction": [1, -1, -1, 1]} - array: Numeric contrast vector matching the number of regressors   e.g., [1, -1, 0, 0] for a 4-regressor model | *required*
`statistic` | <code>[str](#str)</code> | Which statistic to return per contrast. One of `"t"` (default, t-statistic map), `"z"` (z-score), `"p"` (p-value), `"beta"` / `"effect_size"` (effect-size β map — use this when feeding a second-level group analysis), or `"all"` (a bundle dict `{"beta", "t", "z", "p", "se"}` of maps for one contrast). Default: `"t"`. | <code>'t'</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or dict: A single contrast with a scalar `statistic` returns a `BrainData` map; with `statistic="all"` it returns a flat dict keyed by `"beta"`/`"t"`/`"z"`/`"p"`/`"se"`. A dict of contrasts returns a dict keyed by contrast name (nested under the five keys when `statistic="all"`).

**Examples:**

```pycon
>>> brain.fit(model='glm', X=design_matrix)
>>> contrast1 = brain.compute_contrasts([0, 1, -1])
>>> contrast2 = brain.compute_contrasts("conditionA - conditionB")
>>> results = brain.compute_contrasts({
...     "A_vs_B": "conditionA - conditionB",
...     "avg_effect": [0, 0.5, 0.5],
... })
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- String contrasts support coefficients: "2*A - B" or "0.5*A + 0.5*B"
- Column names must match design matrix columns exactly (case-sensitive)
- Contrast weights should sum to zero for proper inference in most cases

</details>

###### `copy`

```python
copy()
```

Create a deep copy of a BrainData instance.

All attributes including data, fitted models, and results are deep copied.
Use this when you need a complete independent copy.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Deep copied instance

(data-create-empty)=
###### `create_empty`

```python
create_empty()
```

Create a copy of BrainData with empty data array.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | A copy of this object with an empty data array.

(data-decompose)=
###### `decompose`

```python
decompose(*, method = 'pca', axis = 'voxels', n_components = None, **kwargs)
```

Decompose BrainData object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` |  | (str) Algorithm to perform decomposition         types=['pca','ica','nnmf','fa','dictionary','kernelpca'] | <code>'pca'</code>
`axis` |  | dimension to decompose ['voxels','images'] | <code>'voxels'</code>
`n_components` |  | (int) number of components. If None then retain         as many as possible. | <code>None</code>
`**kwargs` |  | forwarded to the underlying sklearn decomposition estimator. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`output` |  | a dictionary of decomposition parameters

###### `detrend`

```python
detrend(method = 'linear')
```

Remove linear trend from each voxel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` |  | ('linear','constant', optional) type of detrending | <code>'linear'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (BrainData) detrended BrainData instance

###### `distance`

```python
distance(metric = 'euclidean', *, spatial_scale: str = 'whole_brain', roi_mask: str = None, radius_mm: float = 10.0, **kwargs: float)
```

Calculate distance between images within a BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` |  | (str) type of distance metric (can use any scipy.spatial.distance     metric supported by cdist) | <code>'euclidean'</code>
`**kwargs` |  | Additional metric options forwarded to ``scipy.spatial.distance.cdist`` (e.g. ``p`` for minkowski). | <code>{}</code>
`spatial_scale` | <code>[str](#str)</code> | One of ``'whole_brain'`` (default), ``'roi'``, or ``'searchlight'``. ``'whole_brain'`` returns a single pairwise distance ``Adjacency`` between images. ``'roi'`` requires ``roi_mask`` and returns a stacked ``Adjacency`` with one RDM per parcel and ``spatial_scale`` provenance attached for back-projection via ``Adjacency.to_brain()``. ``'searchlight'`` requires ``radius_mm`` (and is not yet implemented in this slice). | <code>'whole_brain'</code>
`roi_mask` |  | Atlas image (BrainData / Nifti1Image / path) for ``spatial_scale='roi'``. | <code>None</code>
`radius_mm` | <code>[float](#float)</code> | Searchlight radius in mm. Default 10.0. | <code>10.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | Single pairwise distance matrix for ``'whole_brain'``; stacked Adjacency (one matrix per parcel/searchlight) with ``spatial_scale`` set for ``'roi'`` / ``'searchlight'``.

(data-extract-roi)=
###### `extract_roi`

```python
extract_roi(mask, metric = 'mean', n_components = None)
```

Extract activity from mask or ROI atlas using NiftiLabelsMasker.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask` |  | BrainData, nibabel image, or file path. Can be:<br>  - Binary mask (extracts from single ROI)   - Labeled atlas (extracts from multiple ROIs) | *required*
`metric` |  | Extraction method ('mean', 'median', 'pca'). Default: 'mean' | <code>'mean'</code>
`n_components` |  | If metric='pca', number of components to return | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | For binary mask: scalar or 1D array.
 | For labeled atlas: 1D or 2D array, or PCA components.

**Examples:**

```pycon
>>> roi_values = brain.extract_roi(binary_mask)
>>> atlas_values = brain.extract_roi(atlas_mask)
>>> components = brain.extract_roi(mask, metric='pca', n_components=5)
```

###### `filter`

```python
filter(*, sampling_freq = None, high_pass = None, low_pass = None, **kwargs)
```

Apply butterworth filter to data. Wraps nilearn.signal.clean.

<details class="note" open markdown="1">
<summary>Note</summary>

Unlike nilearn's default, does not detrend or standardize. Pass
detrend=True or standardize=True via kwargs to enable.

</details>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sampling_freq` |  | Sampling freq in hertz (i.e. 1 / TR) | <code>None</code>
`high_pass` |  | High pass cutoff frequency | <code>None</code>
`low_pass` |  | Low pass cutoff frequency | <code>None</code>
`**kwargs` |  | Additional arguments passed to nilearn.signal.clean | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Filtered BrainData instance

(data-find-spikes)=
###### `find_spikes`

```python
find_spikes(global_spike_cutoff = 3, diff_spike_cutoff = 3, *, TR: float | None = None, sampling_freq: float | None = None)
```

Identify spikes from Time Series Data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`global_spike_cutoff` | <code>[int](#int) or None</code> | cutoff to identify spikes in global signal in standard deviations, or None to skip. | <code>3</code>
`diff_spike_cutoff` | <code>[int](#int) or None</code> | cutoff to identify spikes in average frame difference in standard deviations, or None to skip. | <code>3</code>
`TR` | <code>[float](#float) \| None</code> | Repetition time in seconds. Sets the returned DesignMatrix's sampling_freq for downstream `.append(...)` / `.convolve()`. Pass exactly one of `TR` or `sampling_freq`. | <code>None</code>
`sampling_freq` | <code>[float](#float) \| None</code> | Sampling frequency in Hz (= 1/TR). See `TR`. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | DesignMatrix with one indicator column per detected spike, with
 | all spike columns pre-marked as confounds.

###### `fit`

```python
fit(model = 'glm', *, X = None, cv = None, local_alpha = True, fit_intercept = False, inplace = True, scale = 'auto', standardize = 'auto', progress_bar = None, design_clean = True, design_clean_thresh = 0.95, design_clean_exclude_confounds = False, design_clean_fill_na = 0, **kwargs)
```

Fit a model to brain imaging data.

Creates and fits a model from string specification. The brain data
(self.data) is always used as the target variable. Model and results
are stored for later use with predict().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model` | <code>[str](#str)</code> | Model type: 'ridge', 'glm', or future model names | <code>'glm'</code>
`X` | <code>[array](#array) - [like](#like) or [DataFrame](#DataFrame)</code> | Design matrix or feature matrix | <code>None</code>
`cv` | <code>int or sklearn CV splitter</code> | Cross-validation specification (Ridge only). int → ``KFold(cv)``; pass a splitter object (e.g. ``KFold(5, shuffle=True)``, ``GroupKFold(8)``) for non-contiguous folds. Generators (``splitter.split(X)``) are rejected. | <code>None</code>
`local_alpha` | <code>bool, default=True</code> | Ridge only. If True, select α independently per voxel via ``solve_ridge_cv``. If False, pick a single α shared across all voxels. | <code>True</code>
`fit_intercept` | <code>bool, default=False</code> | Ridge only. Forwarded to the Ridge model — center X and y on the training fold mean per fold and recover the intercept after. | <code>False</code>
`inplace` | <code>bool, default=True</code> | If True, mutate self and return self. If False, return a Fit dataclass with the results. ``self.data`` and the result attributes (``ridge_*`` / ``glm_*`` / ``cv_results_``) are left unchanged, but ``self.model_`` and ``self.X_`` (plus ``self.design_matrix`` for GLM) ARE updated on self so ``predict()`` / ``compute_contrasts()`` still work. | <code>True</code>
`scale` | <code>bool or 'auto', default='auto'</code> | Apply percent-signal-change scaling before fitting via nilearn's per-voxel ``mean_scaling``. ``'auto'`` → False for both models (PSC is opt-in). Redundant with ``standardize='zscore'`` (warns). Applied before ``standardize``. | <code>'auto'</code>
`standardize` | <code>str or None or 'auto', default='auto'</code> | Standardize each voxel across observations after scaling. ``'center'``, ``'zscore'``, or ``None``. ``'auto'`` → ``'zscore'`` for ridge, ``None`` for glm. | <code>'auto'</code>
`progress_bar` | <code>[bool](#bool)</code> | Display progress bar during fitting. | <code>None</code>
`design_clean` | <code>bool, default=True</code> | GLM only. Run ``DesignMatrix.clean()`` on ``X`` before fitting to drop highly correlated regressors. Coerces ``X`` to ``DesignMatrix`` if needed. Ignored when ``model='ridge'``. | <code>True</code>
`design_clean_thresh` | <code>float, default=0.95</code> | GLM only. Correlation threshold passed to ``DesignMatrix.clean()`` (drops if ``abs(r) >= thresh``). Ignored when ``model='ridge'``. | <code>0.95</code>
`design_clean_exclude_confounds` | <code>bool, default=False</code> | GLM only. If True, ``DesignMatrix.clean()`` skips confound columns when checking correlations. Ignored when ``model='ridge'``. | <code>False</code>
`design_clean_fill_na` | <code>int, float, or None, default=0</code> | GLM only. Fill value for NaNs before correlation check in ``DesignMatrix.clean()``. Ignored when ``model='ridge'``. | <code>0</code>
`**kwargs` | <code>[dict](#dict)</code> | Additional arguments passed to model constructor | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or Fit: If ``inplace=True``, returns self (fitted BrainData). If ``inplace=False``, returns Fit dataclass with results.

<details class="notes" open markdown="1">
<summary>Notes</summary>

After ``model="glm"``, the following per-regressor BrainData
attributes are populated — one map per design-matrix column:

- ``glm_betas``: effect-size (β) maps.
- ``glm_t``: marginal t-statistic for each regressor.
- ``glm_p``: marginal p-value.
- ``glm_se``: standard error of β.
- ``glm_r2``: voxel-wise R².

``glm_t[i]`` is a valid t-map for the trivial one-hot contrast on
regressor ``i`` only. For contrasts across regressors
(``"A - B"``, ``[1, -1, 0, ...]``) use `compute_contrasts` —
you cannot correctly combine these per-regressor maps by hand
because t-statistic arithmetic requires the off-diagonal elements
of the parameter covariance matrix, which are not stored. Pass
``statistic="all"`` to get ``β``/``t``/``z``/``p``/``se`` for
one contrast in a single call.

</details>

**Examples:**

```pycon
>>> brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
>>> fit = brain_data.fit(model='ridge', alpha=1.0, X=features, inplace=False)
```

(data-iplot)=
###### `iplot`

```python
iplot(*, view: str = 'ortho', threshold: float | None = None, lower: float | None = None, upper: float | None = None, cmap: str = 'warm', bg_img: str | bool | None = None, atlas: str | Atlas | None = None, opacity: float = 1.0, outline: float = 0.0, colorbar: bool = True, controls: bool = True, **kwargs: bool)
```

Interactive WebGL brain viewer powered by niivue (`ipyniivue`).

Renders inline in a live kernel (Jupyter, marimo) with live windowing
(right-drag to set the threshold/contrast), slice scrolling, native 4D
frame scrubbing, true 3D rendering, a stat-map colorbar, and optional
nltools-atlas overlays. Static-built docs are not supported; use
`plot` there.

By default (``controls=True``) the return value is an
`ipywidgets.VBox` stacking a threshold slider above the viewer; access
the underlying `NiiVue` via its ``.viewer`` attribute and the slider
via ``.threshold_slider``. Pass ``controls=False`` to get the bare
`NiiVue` widget instead.

Thresholding is a divergent magnitude window: ``cal_min`` is the
display floor (sub-floor voxels render transparent), ``cal_max`` the
saturation point, with the positive limb using ``cmap`` and the
negative limb its mirrored partner. Precedence: ``lower``/``upper``
win; otherwise ``threshold`` sets the floor (ceiling auto);
otherwise the window is fully auto.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`view` | <code>[str](#str)</code> | ``"ortho"`` (default), ``"axial"``, ``"coronal"``, ``"sagittal"``, or ``"render"`` (3D volume render). ``"surface"`` is no longer supported — use ``"render"`` or `plot_flatmap` / `plot_surf`. | <code>'ortho'</code>
`threshold` | <code>[float](#float) \| None</code> | Convenience symmetric magnitude floor (→ ``cal_min``). | <code>None</code>
`lower` | <code>[float](#float) \| None</code> | Window floor (→ ``cal_min``). Overrides ``threshold``. | <code>None</code>
`upper` | <code>[float](#float) \| None</code> | Window ceiling (→ ``cal_max``). Overrides ``threshold``. | <code>None</code>
`cmap` | <code>[str](#str)</code> | niivue colormap for the positive limb (default ``"warm"``). Common matplotlib names are auto-mapped with a warning. | <code>'warm'</code>
`bg_img` | <code>[str](#str) \| [bool](#bool) \| None</code> | ``None``/``True`` auto-loads the matching MNI template when the data is in standard space (else none); ``False`` disables the background; a path string uses that image. | <code>None</code>
`atlas` | <code>[str](#str) \| [Atlas](#nltools.data.atlases.Atlas) \| None</code> | Atlas overlay — a registry name (e.g. ``"aal"``), a loaded `Atlas`, or ``None``. Deterministic atlases only; probabilistic atlases raise. | <code>None</code>
`opacity` | <code>[float](#float)</code> | Stat-map (and filled-atlas) opacity in ``0..1``. | <code>1.0</code>
`outline` | <code>[float](#float)</code> | ``> 0`` draws atlas region boundaries of that width (stat map stays visible); ``0`` draws filled regions. | <code>0.0</code>
`colorbar` | <code>[bool](#bool)</code> | Show the stat-map colorbar (default ``True``). An explicit ``is_colorbar`` kwarg overrides this. | <code>True</code>
`controls` | <code>[bool](#bool)</code> | Wrap the viewer in a `VBox` with an interactive threshold slider (default ``True``). ``False`` returns the bare `NiiVue`. Requires the ``ipywidgets`` optional dependency when ``True``. | <code>True</code>
`**kwargs` |  | Forwarded verbatim to ``ipyniivue.NiiVue(**kwargs)`` (e.g. ``height``, ConfigOptions like ``is_colorbar``). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | ipywidgets.VBox with ``.viewer`` (the `NiiVue`) and
 | ``.threshold_slider`` when ``controls=True``; otherwise the bare
 | ``ipyniivue.NiiVue`` widget.

###### `mean`

```python
mean(axis = 0, *, spatial_scale: str = 'whole_brain', roi_mask: str = None)
```

Get mean of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array). Ignored when ``spatial_scale='roi'``. | <code>0</code>
`spatial_scale` | <code>[str](#str)</code> | ``'whole_brain'`` (default) preserves existing behavior. ``'roi'`` requires ``roi_mask`` and returns a BrainData of the same shape with each voxel painted with its parcel's mean per image (parcellation smoothing). | <code>'whole_brain'</code>
`roi_mask` |  | Atlas image for ``spatial_scale='roi'``. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Mean values.

###### `median`

```python
median(axis = 0, *, spatial_scale: str = 'whole_brain', roi_mask: str = None)
```

Get median of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array). Ignored when ``spatial_scale='roi'``. | <code>0</code>
`spatial_scale` | <code>[str](#str)</code> | ``'whole_brain'`` (default) or ``'roi'`` (paints each voxel with its parcel's median per image). | <code>'whole_brain'</code>
`roi_mask` |  | Atlas image for ``spatial_scale='roi'``. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Median values.

(data-multivariate-similarity)=
###### `multivariate_similarity`

```python
multivariate_similarity(images, method = 'ols')
```

Predict a BrainData spatial distribution from a linear combination.

The predictors may be other BrainData instances or nibabel images.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`images` |  | BrainData instance of weight map | *required*
`method` | <code>[str](#str)</code> | Regression method. Default: 'ols'. | <code>'ols'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | dictionary of regression statistics in BrainData instances {'beta','t','p','df','residual'}

###### `plot`

```python
plot(*, method = 'glass', upper = None, lower = None, threshold = None, view = 'z', cut_coords = None, cmap = None, bg_img = None, ax = None, figsize = (8, 6), title = None, colorbar = True, save = None, stat = 'mean', limit = 3, **kwargs)
```

Plot BrainData instance using nilearn visualization or matplotlib.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Visualization type: 'glass', 'slices', 'timeseries', 'histogram' | <code>'glass'</code>
`upper` | <code>[str](#str) / [float](#float)</code> | Upper threshold. | <code>None</code>
`lower` | <code>[str](#str) / [float](#float)</code> | Lower threshold. | <code>None</code>
`threshold` | <code>[float](#float)</code> | Convenience parameter for thresholding. | <code>None</code>
`view` | <code>[str](#str)</code> | For ``method="slices"``, any non-empty combination of ``"x"``, ``"y"``, ``"z"`` (e.g. ``"xyz"``, ``"xz"``, ``"y"``). Default: ``"z"``. | <code>'z'</code>
`cut_coords` | <code>[list](#list) or [dict](#dict)</code> | Cut coordinates for multi-slice views. Takes precedence over ``view``-based defaults. Either a list matching ``len(view)`` or a dict keyed by axis letter. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Colormap name. | <code>None</code>
`bg_img` | <code>str/nibabel image</code> | Background image. | <code>None</code>
`ax` | <code>[Axes](#matplotlib.axes.Axes)</code> | Matplotlib axis. | <code>None</code>
`figsize` | <code>[tuple](#tuple)</code> | default figure size if no axis (8, 6) | <code>(8, 6)</code>
`title` | <code>[str](#str)</code> | Plot title. | <code>None</code>
`colorbar` | <code>[bool](#bool)</code> | Whether to show colorbar. Default: True. | <code>True</code>
`save` | <code>[str](#str)</code> | Path to save figure(s). | <code>None</code>
`stat` | <code>[str](#str)</code> | Statistic for timeseries plots. Default: 'mean'. | <code>'mean'</code>
`limit` | <code>[int](#int)</code> | Maximum number of images to render when this BrainData contains multiple maps and ``method`` is ``"glass"`` or ``"slices"``. Default: 3. Warns when more images exist than ``limit``. | <code>3</code>
`**kwargs` |  | Additional arguments passed to nilearn plot functions. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure or list[matplotlib.figure.Figure]: A
 | single figure for single-image data; a list of figures for
 | multi-image data with ``method`` in ``{"glass", "slices"}``
 | (one per image for glass; one per image-and-view pair for
 | slices).

(data-plot-flatmap)=
###### `plot_flatmap`

```python
plot_flatmap(*, threshold = None, cmap = 'RdBu_r', vmax = None, vmin = None, template = 'fsaverage5', with_curvature = True, curvature_contrast = 0.5, curvature_brightness = 0.5, transparency = 'auto', colorbar = True, colorbar_orientation = 'horizontal', figsize = (12, 6), title = None, radius_mm = 3.0, interpolation = 'linear', axes = None, save = None)
```

Plot brain data on cortical flatmap.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`threshold` | <code>[float](#float)</code> | Values below this absolute threshold are masked. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Matplotlib colormap. Default: 'RdBu_r'. | <code>'RdBu_r'</code>
`vmax` | <code>[float](#float)</code> | Maximum value for colormap. | <code>None</code>
`vmin` | <code>[float](#float)</code> | Minimum value for colormap. | <code>None</code>
`template` | <code>[str](#str)</code> | Freesurfer surface resolution. Default: 'fsaverage5'. | <code>'fsaverage5'</code>
`with_curvature` | <code>[bool](#bool)</code> | Show sulcal/gyral pattern. Default: True. | <code>True</code>
`curvature_contrast` | <code>[float](#float)</code> | Contrast of curvature overlay. Default: 0.5. | <code>0.5</code>
`curvature_brightness` | <code>[float](#float)</code> | Mean brightness of curvature overlay. Default: 0.5. | <code>0.5</code>
`transparency` | <code>BrainData, Nifti1Image, str, or "auto"</code> | Binary mask used to render vertices outside the mask as transparent. ``"auto"`` (default) uses the instance's ``.mask``; pass ``None`` to disable masking. | <code>'auto'</code>
`colorbar` | <code>[bool](#bool)</code> | Show colorbar. Default: True. | <code>True</code>
`colorbar_orientation` | <code>[str](#str)</code> | 'horizontal' or 'vertical'. Default: 'horizontal'. | <code>'horizontal'</code>
`figsize` | <code>[tuple](#tuple)</code> | Figure size as (width, height). Default: (12, 6). | <code>(12, 6)</code>
`title` | <code>[str](#str)</code> | Figure title. | <code>None</code>
`radius_mm` | <code>[float](#float)</code> | Sampling radius in mm. Default: 3.0. | <code>3.0</code>
`interpolation` | <code>[str](#str)</code> | Interpolation method. Default: 'linear'. | <code>'linear'</code>
`axes` | <code>[Axes](#matplotlib.axes.Axes)</code> | Existing axes to plot on. | <code>None</code>
`save` | <code>[str](#str)</code> | File path to save figure. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

(data-plot-surf)=
###### `plot_surf`

```python
plot_surf(*, hemi = 'both', view = 'montage', surface = 'pial', template = 'fsaverage5', threshold = None, cmap = 'RdBu_r', vmin = None, vmax = None, transparency = 'auto', bg_on_data = False, colorbar = True, colorbar_orientation = 'horizontal', figsize = (10, 8), title = None, radius_mm = 3.0, interpolation = 'linear', zoom = 1.2, axes = None, save = None)
```

Render this BrainData on fsaverage surfaces as a tight 2×2 montage.

Facade over `plot_surf`. See that function's
docstring for the full argument reference. Notable defaults:
``surface="pial"``, ``zoom=1.2``, ``transparency="auto"`` (uses
this instance's ``.mask``).

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

###### `predict`

```python
predict(*, y: np.ndarray | None = None, X: np.ndarray | None = None, spatial_scale: str = 'whole_brain', model: str = 'svm', cv: int = 5, standardize: bool = True, reduce: str | None = None, n_components: int | None = None, scoring: str = 'auto', groups: np.ndarray | None = None, roi_mask: np.ndarray | None = None, radius_mm: float = 10.0, inplace: bool = False, n_jobs: int = 1, random_state: int | None = None, progress_bar: bool = False)
```

Predict voxel timeseries (encoding) or decode labels (MVPA).

Dispatched by which of ``X`` or ``y`` is provided:

1. **Timeseries prediction** (``X`` provided): use a fitted ridge /
   GLM encoding model on ``self`` to predict voxel responses.
   Returns a fresh ``BrainData`` whose ``.data`` holds the predicted
   timeseries (composes directly with ``.plot()``, ``.standardize()``
   etc.). ``inplace`` has no effect in this mode.
2. **MVPA decoding** (``y`` provided): train a classifier or
   regressor with cross-validation. Returns a `Predict`
   dataclass. Spatial fields (``weight_map``, ``fold_weight_maps``,
   ``final_weight_map``, ``accuracy_map``) are `BrainData`
   objects so ``result.weight_map.plot()`` works directly. Drop down
   to numpy via ``result.weight_map.data``.

Field shapes by ``spatial_scale=``:

- **whole_brain**: ``predictions`` (n_samples,) OOF predictions,
  ``scores`` (n_folds,), ``mean_score`` float, ``std_score`` float,
  ``weight_map`` BrainData (``coef_`` from one fit on the **full**
  ``(X, y)`` — the publishable map), ``fold_weight_maps`` BrainData
  (n_folds, n_voxels) for stability analysis, ``estimator`` the
  fitted all-data sklearn estimator (use for ``.predict()`` on new
  data).
- **roi**: ``scores`` (n_folds, n_rois), ``mean_score`` (n_rois,),
  ``std_score`` (n_rois,), ``roi_labels`` (n_rois,) atlas IDs in
  matching order, ``accuracy_map`` / ``weight_map`` /
  ``fold_weight_maps`` BrainData (per-parcel coefs reassembled to
  voxel space; voxels outside the atlas = NaN), ``estimator`` dict
  keyed by atlas label.
- **searchlight**: ``accuracy_map`` BrainData.

With ``inplace=True``, fields are attached to ``self`` with a
``predict_`` prefix (e.g. ``self.predict_weight_map``,
``self.predict_accuracy_map``), mirroring ``bd.fit()``'s
``glm_*`` / ``ridge_*`` naming.

Why ``weight_map`` is the all-data refit, not the CV mean:
the mean of K per-fold ``coef_`` vectors doesn't correspond to
any actual fitted estimator (each fold saw a different subset).
The all-data refit is a single legitimate model with all the
information used. CV gives the honest *score*; the refit gives
the publishable *map*. The CV-mean is one line away if you want
it: ``result.fold_weight_maps.data.mean(axis=0)``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` | <code>[array](#array) - [like](#like)</code> | Labels (classification) or continuous targets (regression), shape ``(n_samples,)``. Triggers MVPA mode. | <code>None</code>
`X` | <code>[array](#array) - [like](#like)</code> | Features for timeseries prediction, shape ``(n_samples, n_features)``. Triggers encoding mode. | <code>None</code>
`spatial_scale` | <code>[str](#str)</code> | MVPA dispatch — ``'whole_brain'``, ``'searchlight'``, or ``'roi'``. | <code>'whole_brain'</code>
`model` | <code>str or sklearn estimator</code> | Algorithm. String shortcuts:<br>- Classification: ``'svm'`` (LinearSVC), ``'logistic'``,   ``'lda'``, ``'ridge_classifier'``. - Regression: ``'ridge'``, ``'lasso'``, ``'svr'``.<br>Or pass any sklearn estimator / Pipeline (e.g., ``make_pipeline(StandardScaler(), SelectKBest(k=500), LinearSVC())``). When ``model`` is a sklearn ``Pipeline``, ``standardize`` is auto-defaulted to ``False`` (with a warning) so we don't wrap another StandardScaler around your pipeline. Pass ``standardize=True`` explicitly to override. | <code>'svm'</code>
`cv` | <code>int or sklearn CV splitter</code> | ``int`` → KFold (regression) or StratifiedKFold (classification); pass a splitter for custom schemes (e.g., ``GroupKFold``). | <code>5</code>
`standardize` | <code>[bool](#bool)</code> | Z-score features per fold before fitting. Default ``True``. Auto-flipped to ``False`` when ``model`` is a sklearn ``Pipeline`` (see ``model`` above). | <code>True</code>
`reduce` | <code>[str](#str)</code> | Per-fold dimensionality reduction. Currently only ``'pca'`` supported. Default ``None``. Weight maps are back-projected through PCA to voxel space. | <code>None</code>
`n_components` | <code>[int](#int)</code> | PCA components when ``reduce='pca'``. | <code>None</code>
`scoring` | <code>[str](#str)</code> | Sklearn scoring string. Default ``'auto'`` → ``'accuracy'`` if classifier, ``'r2'`` if regressor. | <code>'auto'</code>
`groups` | <code>[array](#array) - [like](#like)</code> | Group labels for CV splitters that need them (e.g., leave-one-run-out). | <code>None</code>
`roi_mask` | <code>[Nifti1Image](#Nifti1Image) or [path](#path) - [like](#like)</code> | Atlas image for ``spatial_scale='roi'``. | <code>None</code>
`radius_mm` | <code>[float](#float)</code> | Searchlight radius in mm. Default ``10.0``. | <code>10.0</code>
`inplace` | <code>[bool](#bool)</code> | If ``True``, populate result fields as ``predict_*`` attributes on ``self`` and return ``self``. Default ``False`` returns a fresh `Predict`. | <code>False</code>
`n_jobs` | <code>[int](#int)</code> | Parallel jobs for searchlight / ROI. Default ``1``; searchlight on a real brain at higher ``n_jobs`` can be memory-heavy. | <code>1</code>
`random_state` | <code>[int](#int)</code> | Seed for the shuffled fold splitter when ``cv`` is an int (MVPA mode). Default ``None`` (unseeded shuffle each call). Ignored when ``cv`` is a splitter object — set its own ``random_state`` instead. | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar for searchlight / ROI. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
 | Predict | BrainData: ``Predict`` dataclass when ``inplace=False``; ``self`` (mutated, with ``predict_*`` attrs) when ``inplace=True``.

**Examples:**

```pycon
>>> result = brain.predict(y=labels, spatial_scale='whole_brain', cv=5)
>>> result.weight_map.plot()       # publishable map (all-data fit)
>>> result.mean_score              # honest CV-derived accuracy
>>> new_pred = result.estimator.predict(new_X)  # apply to new data
```

```pycon
>>> result = brain.predict(y=labels, spatial_scale='searchlight',
...                        radius_mm=8.0, n_jobs=4)
>>> result.accuracy_map.plot()
```

```pycon
>>> result = brain.predict(y=labels, spatial_scale='roi', roi_mask=atlas)
>>> top = result.roi_labels[result.mean_score.argsort()[::-1][:10]]
>>> result.accuracy_map.plot()  # brain-space view of the same map
```

Custom sklearn pipeline as model — standardize auto-defaults to
False because we detect the Pipeline:

```python
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
pipe = make_pipeline(StandardScaler(), SelectKBest(k=500),
                     LinearSVC())
result = brain.predict(y=labels, model=pipe)
```

###### `r_to_z`

```python
r_to_z()
```

Apply Fisher's r-to-z transformation to each data element.

(data-regions)=
###### `regions`

```python
regions(*, min_region_size = 1350, method = 'local_regions', smoothing_fwhm = 6, is_mask = False)
```

Extract brain connected regions into separate regions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`min_region_size` | <code>[int](#int)</code> | Minimum volume in mm3 for a region to be kept. | <code>1350</code>
`method` | <code>[str](#str)</code> | Type of extraction method                 ['connected_components', 'local_regions']. | <code>'local_regions'</code>
`smoothing_fwhm` | <code>[scalar](#scalar)</code> | Smooth an image to extract more sparser regions. | <code>6</code>
`is_mask` | <code>[bool](#bool)</code> | Whether to treat as boolean mask. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with extracted ROIs as data.

(data-report)=
###### `report`

```python
report(contrasts = None, **kwargs)
```

Generate a nilearn HTML report for a fitted GLM.

Must be called after ``fit(model='glm', ...)``. Renders the design
matrix, requested contrast maps, and model parameters as a
self-contained HTML report.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`contrasts` | <code>str, list, or dict</code> | Contrast(s) to render, same forms as `compute_contrasts`. | <code>None</code>
`**kwargs` |  | Forwarded to nilearn's ``generate_report`` (e.g. ``title``, ``threshold``, ``alpha``). | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`HTMLReport` |  | nilearn report; call ``.save_as_html(path)`` or display it in a notebook.

**Examples:**

```pycon
>>> brain.fit(model='glm', X=design_matrix)
>>> brain.report(contrasts='conditionA - conditionB').save_as_html('report.html')
```

(data-resample-to)=
###### `resample_to`

```python
resample_to(*, img = None, resolution = None, interpolation = None)
```

Resample BrainData to match target image or resolution.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`img` |  | Target image for resampling (nibabel Nifti1Image, str/Path, or None). | <code>None</code>
`resolution` |  | Target voxel size in mm (float/int for isotropic, or None). | <code>None</code>
`interpolation` |  | Interpolation method ('nearest', 'linear', 'continuous', or None). | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | New BrainData instance with resampled data

(data-scale)=
###### `scale`

```python
scale(scale_val = 100.0, axis = None)
```

Scale data via mean scaling.

Two scaling modes are available:

- **Grand-mean scaling** (axis=None, default): Divides all values by the
  global mean across all voxels and timepoints.

- **Voxel-wise scaling** (axis=0): Divides each voxel's time-series by
  its own temporal mean.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`scale_val` |  | (int/float) Target value for the mean after scaling. Default 100. | <code>100.0</code>
`axis` |  | (int or None) None for grand-mean scaling (default), 0 for voxel-wise scaling. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | New BrainData instance with scaled data.

###### `similarity`

```python
similarity(image, metric = 'correlation')
```

Calculate similarity to a single BrainData or nibabel image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`image` |  | (BrainData, nifti) image to evaluate similarity | *required*
`metric` |  | (str) Type of similarity     ['correlation','pearson','rank_correlation','spearman','dot_product','cosine'] | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
 | float or np.ndarray: Similarity value(s).

###### `smooth`

```python
smooth(fwhm)
```

Apply spatial smoothing using nilearn smooth_img().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fwhm` |  | (float) full width half maximum of gaussian spatial filter | *required*

**Returns:**

Type | Description
---- | -----------
 | BrainData instance (copy with smoothed data)

###### `standardize`

```python
standardize(*, axis = 0, method = 'center', suppress_warnings = False)
```

Standardize BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>[int](#int)</code> | 0 standardizes each voxel across observations (default). 1 standardizes each observation across voxels. | <code>0</code>
`method` | <code>[str](#str)</code> | 'center' subtracts the mean (default). 'zscore' subtracts the mean and divides by standard deviation. | <code>'center'</code>
`suppress_warnings` | <code>[bool](#bool)</code> | If True, suppress sklearn numerical warnings that occur when voxels have near-zero variance. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Standardized BrainData instance.

###### `std`

```python
std(axis = 0, *, spatial_scale: str = 'whole_brain', roi_mask: str = None)
```

Get standard deviation of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array). Ignored when ``spatial_scale='roi'``. | <code>0</code>
`spatial_scale` | <code>[str](#str)</code> | ``'whole_brain'`` (default) or ``'roi'`` (paints each voxel with its parcel's std per image). | <code>'whole_brain'</code>
`roi_mask` |  | Atlas image for ``spatial_scale='roi'``. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Standard deviation values.

###### `sum`

```python
sum(axis = 0)
```

Get sum of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array) | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Sum values.

(data-temporal-resample)=
###### `temporal_resample`

```python
temporal_resample(*, sampling_freq = None, target = None, target_type = 'hz')
```

Resample BrainData timeseries to a new target frequency or number of samples.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sampling_freq` |  | (float) sampling frequency of data in hertz | <code>None</code>
`target` |  | (float) upsampling target | <code>None</code>
`target_type` |  | (str) type of target can be [samples,seconds,hz] | <code>'hz'</code>

**Returns:**

Type | Description
---- | -----------
 | upsampled BrainData instance

###### `threshold`

```python
threshold(*, upper = None, lower = None, binarize = False, coerce_nan = True, cluster_threshold = 0)
```

Threshold BrainData instance with optional cluster filtering.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`upper` |  | (float or str) Upper cutoff for thresholding. | <code>None</code>
`lower` |  | (float or str) Lower cutoff for thresholding. | <code>None</code>
`binarize` | <code>[bool](#bool)</code> | return binarized image. Default False. | <code>False</code>
`coerce_nan` | <code>[bool](#bool)</code> | coerce nan values to 0s. Default True. | <code>True</code>
`cluster_threshold` | <code>[int](#int)</code> | Minimum cluster size in voxels. Default 0. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | Thresholded BrainData object.

(data-to-nifti)=
###### `to_nifti`

```python
to_nifti()
```

Convert BrainData Instance into Nifti Object.

**Returns:**

Type | Description
---- | -----------
 | nibabel.Nifti1Image: Brain data as a NIfTI image.

(data-transform-pairwise)=
###### `transform_pairwise`

```python
transform_pairwise()
```

Transform data into pairwise comparisons.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance transformed into pairwise comparisons

###### `ttest`

```python
ttest(*, popmean = 0.0, permutation = False, n_permute = 5000, tail = 2, return_null = False, n_jobs = -1, random_state = None)
```

One-sample voxelwise t-test across images (axis 0).

Tests whether the per-voxel mean across images differs from
``popmean``. Operates on a stack of images (e.g. subject-level
contrast maps) with shape ``(n_samples, n_voxels)``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`popmean` |  | Population mean to test against. Default 0.0. | <code>0.0</code>
`permutation` |  | If True, use sign-flip permutation test via `one_sample_permutation_test`. | <code>False</code>
`n_permute` |  | Number of permutations (used only when ``permutation=True``). Default 5000. | <code>5000</code>
`tail` |  | Tail of the test (1 or 2). Default 2. | <code>2</code>
`return_null` |  | If True, also return the null distribution. Default False. | <code>False</code>
`n_jobs` |  | Number of parallel jobs. Default -1 (all cores). | <code>-1</code>
`random_state` |  | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | dict with four BrainData keys:<br>- ``"mean"``: voxelwise mean across images (effect size). - ``"t"``: parametric one-sample t-statistic. - ``"z"``: signed z-score, ``sign(t) * norm.isf(p/2)`` —   matches nilearn's ``output_type='z_score'``. - ``"p"``: parametric p-value, or empirical p when   ``permutation=True``.
 | The effect size is always returned alongside the inferential maps
 | so group-level code never has to recompute the mean.

**Examples:**

```pycon
>>> # Stack of subject-level contrast maps
>>> result = contrast_maps.ttest()
>>> sig = result["p"].data < 0.05
>>> effect = result["mean"]       # for reporting magnitude
>>> z_map = result["z"]           # for nilearn-style thresholding
```

```pycon
>>> # Permutation-based p-values; still reports t/z/mean
>>> result = contrast_maps.ttest(permutation=True, n_permute=5000)
```

###### `ttest2`

```python
ttest2(other, equal_var = True)
```

Two-sample voxelwise t-test between two BrainData stacks.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` |  | BrainData to compare against. Must have the same number of voxels. | *required*
`equal_var` |  | If True (default), standard two-sample t-test. If False, Welch's t-test. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | ``{"t": BrainData, "p": BrainData}``.

(data-upload-neurovault)=
###### `upload_neurovault`

```python
upload_neurovault(*, access_token = None, collection_name = None, collection_id = None, img_type = None, img_modality = None, **kwargs)
```

Upload BrainData images and metadata to NeuroVault.

Adds any columns in ``self.X`` to image metadata. The index is used as
the image name.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`access_token` |  | (str, Required) Neurovault api access token | <code>None</code>
`collection_name` |  | (str, Optional) name of new collection to create | <code>None</code>
`collection_id` |  | (int, Optional) neurovault collection_id if adding images             to existing collection | <code>None</code>
`img_type` |  | (str, Required) Neurovault map_type | <code>None</code>
`img_modality` |  | (str, Required) Neurovault image modality | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`collection` |  | (pd.DataFrame) neurovault collection information

###### `write`

```python
write(file_name)
```

Write out BrainData object to Nifti or HDF5 File.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>[str](#str) or [Path](#Path)</code> | Output file path (.nii/.nii.gz for NIfTI, .h5/.hdf5 for HDF5). | *required*

###### `z_to_r`

```python
z_to_r()
```

Convert z score back into r value for each element of data object.

(data-designmatrix)=
#### `DesignMatrix`

```python
DesignMatrix(data: DesignMatrix | pl.DataFrame | pd.DataFrame | np.ndarray | dict | str | Path | None = None, *, sampling_freq: float | None = None, TR: float | None = None, run_length: int | str | None = None, columns: list[str] | None = None, convolved: list[str] | None = None, confounds: list[str] | None = None, hrf_model: str | None = 'glover')
```

Represent experimental designs for neuroimaging with Polars.

This is a Polars-based design matrix for experimental designs in
neuroimaging.

Wraps a Polars DataFrame with neuroimaging-specific metadata and methods.
Uses composition pattern (not subclassing) for clean metadata preservation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>DataFrame, ndarray, dict, str/Path, or None</code> | Input data. Accepts: - Polars DataFrame (zero-copy) - pandas DataFrame (converted to Polars) - numpy ndarray - dict (keys=columns, values=data) - str or Path to a `.tsv`/`.csv` file. BIDS events files   (containing `onset` and `duration` columns) are converted to   boxcar regressors — call ``convolve()`` afterwards if you want   HRF convolution. Any other tabular file is read as-is and is   typically used for confounds. - None (empty initialization) | <code>None</code>
`sampling_freq` | <code>[float](#float)</code> | Sampling frequency in Hz (1/TR for fMRI data). Mutually exclusive with ``TR``. | <code>None</code>
`TR` | <code>[float](#float)</code> | Repetition time in seconds. Convenience for ``sampling_freq = 1/TR``. Mutually exclusive with ``sampling_freq``. | <code>None</code>
`run_length` | <code>[int](#int) or 'infer'</code> | Required when ``data`` is a file path. Number of TRs in the run. Pass ``'infer'`` for tabular/confounds files to accept whatever row count the file has (not valid for events files). | <code>None</code>
`columns` | <code>list of str</code> | Column names (used with ndarray input) | <code>None</code>
`convolved` | <code>list of str</code> | Names of convolved columns (tracked internally) | <code>None</code>
`confounds` | <code>list of str</code> | Names of nuisance/confound columns (intercept, polynomial drift, DCT cosines, motion, …) tracked internally | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`sampling_freq` | <code>[float](#float) or None</code> | Sampling frequency in Hz
`convolved` | <code>list of str</code> | Columns that have been convolved
`confounds` | <code>list of str</code> | Nuisance/confound columns (intercept, polynomial trends, DCT bases, motion, physio, …) — these are skipped by ``.convolve()`` and kept separate per run on multi-run vertical append.
`multi` | <code>[bool](#bool)</code> | True if created from multi-run concatenation

**Examples:**

```pycon
>>> # Create from numpy array
>>> dm = DesignMatrix(np.zeros((100, 2)), sampling_freq=0.5, columns=['a', 'b'])
```

```pycon
>>> # Add columns
>>> dm['stim'] = [0, 1, 1, 0] * 25
```

```pycon
>>> # Convolve with HRF — convolved columns get a `_c0` suffix
>>> dm_conv = dm.convolve()  # 'stim' → 'stim_c0'
```

```pycon
>>> # Add polynomial drift terms
>>> dm_conv = dm_conv.add_poly(order=2)
```

```pycon
>>> # Multi-run concatenation (auto-separates polynomials)
>>> dm_run1 = DesignMatrix(...).add_poly(0)
>>> dm_run2 = DesignMatrix(...).add_poly(0)
>>> dm_multi = dm_run1.append(dm_run2, axis=0)  # Creates 0_poly_0, 1_poly_0
```

**Methods:**

Name | Description
---- | -----------
[`add_dct_basis`](#data-add-dct-basis) | Add discrete cosine transform basis functions for high-pass filtering.
[`add_poly`](#data-add-poly) | Add Legendre polynomial drift terms.
[`append`](#data-append) | Concatenate design matrices.
[`clean`](#data-clean) | Remove highly correlated columns.
[`convolve`](#data-convolve) | Convolve columns with an HRF or custom kernel.
[`copy`](#data-copy) | Create a deep copy of the DesignMatrix.
[`corr`](#data-corr) | Calculate column correlations as a similarity ``Adjacency``.
[`downsample`](#data-downsample) | Reduce temporal resolution using Polars-native operations.
[`drop`](#data-drop) | Drop specified columns.
[`fillna`](#data-fillna) | Fill NaN/null values with specified value.
[`plot`](#data-plot) | Visualize the design matrix.
[`replace_data`](#data-replace-data) | Replace data columns while preserving confounds and metadata.
[`standardize`](#data-standardize) | Standardize columns using the specified method.
[`sum`](#data-sum) | Compute the sum along an axis.
[`to_numpy`](#data-to-numpy) | Convert a DesignMatrix to a NumPy array.
[`to_pandas`](#data-to-pandas) | Convert DesignMatrix to pandas DataFrame.
[`upsample`](#data-upsample) | Increase temporal resolution to a target frequency.
[`vif`](#data-vif) | Compute the variance inflation factor for each column.
[`with_columns`](#data-with-columns) | Add or replace columns via Polars expressions.
[`write`](#data-write) | Write DesignMatrix to file.
[`zscore`](#data-zscore) | Z-score standardize columns to mean zero and unit variance.

Passing another ``DesignMatrix`` returns a copy: ``data``,
``sampling_freq``, ``convolved``, ``confounds``, and ``multi`` are
carried over. Any explicit kwarg overrides the inherited value.

When ``data`` is a path to a BIDS events file, the constructor
HRF-convolves the regressors by default (``hrf_model='glover'``,
matching nilearn's ``make_first_level_design_matrix``). The output
columns are suffixed ``_c0`` and ``.convolved`` is populated. Pass
``hrf_model=None`` to load raw boxcar regressors instead — useful
for FIR designs, PPI flows that build interaction terms before
convolution, or pedagogical material that introduces convolution
as a separate step. ``hrf_model`` is silently ignored when ``data``
is anything other than an events file.

##### Methods

(data-add-dct-basis)=
###### `add_dct_basis`

```python
add_dct_basis(duration: float = 180, drop: int = 0, *, include_constant: bool = True) -> DesignMatrix
```

Add discrete cosine transform basis functions for high-pass filtering.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`duration` | <code>[float](#float)</code> | Filter duration in seconds. Default: 180. | <code>180</code>
`drop` | <code>[int](#int)</code> | Number of low-frequency bases to drop. Default: 0. | <code>0</code>
`include_constant` | <code>[bool](#bool)</code> | If True, also add a constant/intercept column named ``cosine_0`` (analogous to ``poly_0`` in `add_poly`). The underlying DCT basis drops the constant per SPM convention; set False to match SPM behavior. Default: True. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with DCT basis columns appended.

(data-add-poly)=
###### `add_poly`

```python
add_poly(order: int = 0, include_lower: bool = True) -> DesignMatrix
```

Add Legendre polynomial drift terms.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`order` | <code>[int](#int)</code> | Polynomial order (0=intercept, 1=linear, 2=quadratic, ...). Default: 0. | <code>0</code>
`include_lower` | <code>[bool](#bool)</code> | If True, include all orders from 0 to order. Default: True. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with polynomial columns appended.

###### `append`

```python
append(dm: DesignMatrix | list[DesignMatrix], *, axis: int = 0, keep_separate: bool = True, unique_cols: list[str] | None = None, fill_na: int | float = 0, as_confounds: bool = False, progress_bar: bool = False) -> DesignMatrix
```

Concatenate design matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>DesignMatrix or list of DesignMatrix</code> | Design matrix/matrices to append. | *required*
`axis` | <code>[int](#int)</code> | 0 for row-wise (vertical), 1 for column-wise (horizontal). Default: 0. | <code>0</code>
`keep_separate` | <code>[bool](#bool)</code> | Whether to separate confound columns across runs (only applies when axis=0). Default: True. | <code>True</code>
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | <code>None</code>
`fill_na` | <code>[int](#int) or [float](#float)</code> | Value to fill NaN values during vertical concatenation. Default: 0. | <code>0</code>
`as_confounds` | <code>[bool](#bool)</code> | Only applies when ``axis=1``. If True, mark all columns from ``dm`` as nuisance/confounds in the result — they get skipped by ``.convolve()`` and separated across runs on later vertical appends. Default: False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Print messages about confound separation. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Concatenated design matrix.

(data-clean)=
###### `clean`

```python
clean(*, fill_na: int | float | None = 0, exclude_confounds: bool = False, thresh: float = 0.95, progress_bar: bool = False) -> DesignMatrix
```

Remove highly correlated columns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fill_na` | <code>int, float, or None</code> | Fill NaN values before checking correlations (default 0) | <code>0</code>
`exclude_confounds` | <code>[bool](#bool)</code> | Skip confound/nuisance columns from correlation check | <code>False</code>
`thresh` | <code>[float](#float)</code> | Correlation threshold (drop if abs(r) >= thresh, default 0.95) | <code>0.95</code>
`progress_bar` | <code>[bool](#bool)</code> | Print dropped column names. Default: False | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Cleaned matrix with highly correlated columns removed

(data-convolve)=
###### `convolve`

```python
convolve(conv_func: str | np.ndarray = 'hrf', columns: list[str] | None = None) -> DesignMatrix
```

Convolve columns with an HRF or custom kernel.

Convolved columns are always renamed to ``<col>_c{i}`` (where ``i`` is
the kernel index, ``0`` for a single 1-D kernel). The source columns
are dropped, and ``self.convolved`` lists the post-suffix names so
downstream metadata stays in sync with the dataframe.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`conv_func` | <code>[str](#str) or [ndarray](#ndarray)</code> | 'hrf' for canonical Glover HRF, or custom kernel(s). Can be 1D array (single kernel) or 2D (samples x kernels). | <code>'hrf'</code>
`columns` | <code>list of str</code> | Columns to convolve (default: all non-confound columns). | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with convolved columns renamed.

###### `copy`

```python
copy() -> DesignMatrix
```

Create a deep copy of the DesignMatrix.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Copy of the current DesignMatrix

(data-corr)=
###### `corr`

```python
corr(*, metric: str = 'pearson', columns: list[str] | None = None)
```

Calculate column correlations as a similarity ``Adjacency``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` | <code>[str](#str)</code> | ``'pearson'`` (default) or ``'spearman'``. | <code>'pearson'</code>
`columns` | <code>list of str</code> | Subset of columns to correlate. Defaults to all columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | Similarity matrix whose ``labels`` are the column names. The unit diagonal is dropped (self-correlation isn't an edge); use ``.plot(method='corr')`` for a heatmap with the diagonal restored.

(data-downsample)=
###### `downsample`

```python
downsample(target: float, method: str = 'mean') -> DesignMatrix
```

Reduce temporal resolution using Polars-native operations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` | <code>[float](#float)</code> | Target sampling frequency in Hz (must be < current sampling_freq) | *required*
`method` | <code>[str](#str)</code> | Aggregation method - 'mean' or 'median' (default: 'mean') | <code>'mean'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Downsampled DesignMatrix with updated sampling_freq

(data-drop)=
###### `drop`

```python
drop(columns: list[str]) -> DesignMatrix
```

Drop specified columns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`columns` | <code>list of str</code> | Column names to remove. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix without the specified columns.

(data-fillna)=
###### `fillna`

```python
fillna(value: int | float) -> DesignMatrix
```

Fill NaN/null values with specified value.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`value` | <code>[int](#int) or [float](#float)</code> | Value to replace NaN/null entries with. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with NaN/null values replaced.

###### `plot`

```python
plot(method: str = 'matrix', *, columns: list[str] | None = None, rescale: bool = True, metric: str = 'pearson', ax: str = None, figsize: tuple | None = None, title: str | None = None, cmap: str | None = None, save: str | None = None, **kwargs: str | None)
```

Visualize the design matrix.

Dispatches over ``method`` (mirroring ``BrainData.plot``):

- ``'matrix'`` (default): SPM-style heatmap (rows=TRs, cols=regressors).
- ``'timeseries'``: overlaid line plot of regressor time courses. Pass
  the same ``ax`` across calls to overlay multiple DesignMatrices
  (e.g. original vs. convolved).
- ``'corr'``: labeled correlation heatmap of the columns (reuses
  `corr`; diagonal restored to 1.0 for display).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ``'matrix'`` | ``'timeseries'`` | ``'corr'``. Default: ``'matrix'``. | <code>'matrix'</code>
`columns` | <code>list of str</code> | Subset of columns to plot. Defaults to all columns. | <code>None</code>
`rescale` | <code>[bool](#bool)</code> | ``'matrix'`` only. Rescale each column by its L2 norm so columns with different native magnitudes are visually comparable (SPM/nilearn convention). Default: True. | <code>True</code>
`metric` | <code>[str](#str)</code> | ``'corr'`` only. ``'pearson'`` (default) or ``'spearman'``. | <code>'pearson'</code>
`ax` | <code>[Axes](#matplotlib.axes.Axes)</code> | Existing axis to draw on; a new figure is created if omitted. | <code>None</code>
`figsize` | <code>[tuple](#tuple)</code> | Figure size; sensible per-method default when omitted. | <code>None</code>
`title` | <code>[str](#str)</code> | Axis title. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Colormap (``'matrix'`` / ``'corr'``). | <code>None</code>
`save` | <code>[str](#str)</code> | Path to save the figure. | <code>None</code>
`**kwargs` |  | Forwarded to the underlying plotter (``seaborn.heatmap`` for ``'matrix'`` / ``'corr'``; ``Axes.plot`` for ``'timeseries'``). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure: The figure containing the plot.

(data-replace-data)=
###### `replace_data`

```python
replace_data(data: np.ndarray, column_names: list[str] | None = None) -> DesignMatrix
```

Replace data columns while preserving confounds and metadata.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#ndarray)</code> | New data array (must match number of rows in current DesignMatrix) | *required*
`column_names` | <code>list of str</code> | Names for new data columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with replaced data columns, preserved confounds

###### `standardize`

```python
standardize(method: str = 'zscore', columns: list[str] | None = None) -> DesignMatrix
```

Standardize columns using the specified method.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Standardization method ('zscore' or 'center'). Default: 'zscore'. | <code>'zscore'</code>
`columns` | <code>[list](#list)[[str](#str)] \| None</code> | Columns to standardize. If None, standardize all non-confound columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with standardized columns.

###### `sum`

```python
sum(axis: int = 0) -> pl.Series
```

Compute the sum along an axis.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>int, default=0</code> | 0: sum down columns, 1: sum across rows. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>[Series](#polars.Series)</code> | pl.Series: Sums along specified axis.

(data-to-numpy)=
###### `to_numpy`

```python
to_numpy() -> np.ndarray
```

Convert a DesignMatrix to a NumPy array.

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | np.ndarray: 2D array with shape (n_samples, n_columns)

(data-to-pandas)=
###### `to_pandas`

```python
to_pandas() -> pd.DataFrame
```

Convert DesignMatrix to pandas DataFrame.

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#pandas.DataFrame)</code> | pd.DataFrame: Pandas DataFrame with same data and column names.

(data-upsample)=
###### `upsample`

```python
upsample(target: float, method: str = 'linear') -> DesignMatrix
```

Increase temporal resolution to a target frequency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` | <code>[float](#float)</code> | Target sampling frequency in Hz (must be > current sampling_freq) | *required*
`method` | <code>[str](#str)</code> | Interpolation method - 'linear' or 'nearest' (default: 'linear') | <code>'linear'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Upsampled DesignMatrix with updated sampling_freq

(data-vif)=
###### `vif`

```python
vif(exclude_confounds: bool = True) -> np.ndarray | None
```

Compute the variance inflation factor for each column.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`exclude_confounds` | <code>[bool](#bool)</code> | Skip confound/nuisance columns. Default: True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray) \| None</code> | np.ndarray: VIF values for each included column. Returns None if the correlation matrix is singular.

(data-with-columns)=
###### `with_columns`

```python
with_columns(*exprs, **named_exprs) -> DesignMatrix
```

Add or replace columns via Polars expressions.

Mirrors `DataFrame.with_columns`. Named kwargs become
named columns; positional ``pl.Expr`` arguments are accepted as-is
(including ``pl.Expr.alias("name")``). Returns a new ``DesignMatrix``
with metadata preserved; new columns are *not* auto-tagged as
convolved or confounds.

For convenience, named-kwarg values that aren't ``pl.Expr`` /
``pl.Series`` are coerced:

- ``int``/``float`` → broadcast scalar via ``pl.lit``
- ``list`` / ``np.ndarray`` → wrapped as ``pl.Series``

**Examples:**

```pycon
>>> dm = dm.with_columns(motor=pl.sum_horizontal(motor_cols)).drop(motor_cols)
>>> dm = dm.with_columns(
...     vmpfc=seed_signal,
...     vmpfc_motor=pl.col("vmpfc") * pl.col("motor_c0"),
... )
```

###### `write`

```python
write(file_name: str, sep: str = '\t') -> None
```

Write DesignMatrix to file.

Supports TSV (default), CSV, and HDF5 formats. Format is
auto-detected from file extension.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>[str](#str)</code> | Output file path. Use .tsv, .csv, or .h5/.hdf5 extension. | *required*
`sep` | <code>[str](#str)</code> | Column separator for text files (default: tab). | <code>'\t'</code>

(data-zscore)=
###### `zscore`

```python
zscore(columns: list[str] | None = None) -> DesignMatrix
```

Z-score standardize columns to mean zero and unit variance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`columns` | <code>list of str</code> | Columns to standardize. If None, standardize all non-confound columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with standardized columns

#### `Fit`

```python
Fit(fitted_values: np.ndarray, weights: np.ndarray | None = None, scores: np.ndarray | None = None, betas: np.ndarray | None = None, t_stats: np.ndarray | None = None, p_values: np.ndarray | None = None, se: np.ndarray | None = None, residuals: np.ndarray | None = None, r2: np.ndarray | None = None, cv_scores: np.ndarray | None = None, cv_mean_score: np.ndarray | None = None, cv_predictions: np.ndarray | None = None, cv_folds: np.ndarray | None = None, cv_best_alpha: float | None = None, cv_alpha_scores: np.ndarray | None = None) -> None
```

Immutable container for model fitting results.

Pure numpy arrays with minimal introspection methods. This allows
users to work directly with nltools inference algorithms without
requiring BrainData objects.

Attributes depend on model type and CV usage:

**Ridge (no CV):**
    weights (ndarray): Coefficients, shape (n_features, n_voxels)
    scores (ndarray): R² scores, shape (n_voxels,)
    fitted_values (ndarray): Training predictions, shape (n_samples, n_voxels)

**Ridge (with CV):**
    All above plus:
    cv_scores (ndarray): Per-fold R², shape (n_folds, n_voxels)
    cv_mean_score (ndarray): Mean R² across folds, shape (n_voxels,)
    cv_predictions (ndarray): Out-of-fold predictions, shape (n_samples, n_voxels)
    cv_folds (ndarray): Fold indices, shape (n_samples,)
    cv_best_alpha (float): Selected alpha (if alpha='auto')
    cv_alpha_scores (ndarray): Alpha selection scores (if alpha='auto')

**GLM:**
    betas (ndarray): Beta coefficients, shape (n_regressors, n_voxels)
    t_stats (ndarray): T-statistics, shape (n_regressors, n_voxels)
    p_values (ndarray): P-values, shape (n_regressors, n_voxels)
    se (ndarray): Standard errors, shape (n_regressors, n_voxels)
    residuals (ndarray): Residuals, shape (n_samples, n_voxels)
    fitted_values (ndarray): Fitted values, shape (n_samples, n_voxels)
    r2 (ndarray): R² values, shape (n_voxels,)

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`fitted_values` | <code>[ndarray](#ndarray)</code> | Fitted values or predictions, always present.
`weights` | <code>[ndarray](#ndarray) \| None</code> | Model coefficients (Ridge).
`scores` | <code>[ndarray](#ndarray) \| None</code> | R² scores (Ridge).
[`betas`](#data-betas) | <code>[ndarray](#ndarray) \| None</code> | Beta coefficients (GLM).
`t_stats` | <code>[ndarray](#ndarray) \| None</code> | T-statistics (GLM).
`p_values` | <code>[ndarray](#ndarray) \| None</code> | P-values (GLM).
`se` | <code>[ndarray](#ndarray) \| None</code> | Standard errors (GLM).
`residuals` | <code>[ndarray](#ndarray) \| None</code> | Residuals (GLM).
`r2` | <code>[ndarray](#ndarray) \| None</code> | R² values (GLM).
`cv_scores` | <code>[ndarray](#ndarray) \| None</code> | Per-fold cross-validation scores.
`cv_mean_score` | <code>[ndarray](#ndarray) \| None</code> | Mean cross-validation score across folds.
`cv_predictions` | <code>[ndarray](#ndarray) \| None</code> | Out-of-fold predictions.
`cv_folds` | <code>[ndarray](#ndarray) \| None</code> | Fold indices for each sample.
`cv_best_alpha` | <code>[float](#float) \| None</code> | Best alpha selected via cross-validation.
`cv_alpha_scores` | <code>[ndarray](#ndarray) \| None</code> | Cross-validation scores for each alpha tested.

<details class="note" open markdown="1">
<summary>Note</summary>

Methods: `available` returns the list of non-None attribute names
(excludes private fields); `asdict` converts to a dictionary,
optionally excluding None values.

</details>

**Examples:**

Creating a Fit object (Ridge without CV):

```pycon
>>> import numpy as np
>>> from nltools.data.fitresults import Fit
>>> fit = Fit(
...     fitted_values=np.random.randn(100, 1000),
...     weights=np.random.randn(5, 1000),
...     scores=np.random.randn(1000)
... )
>>> fit.available()
['fitted_values', 'weights', 'scores']
```

Creating a Fit object (Ridge with CV):

```pycon
>>> fit_cv = Fit(
...     fitted_values=np.random.randn(100, 1000),
...     weights=np.random.randn(5, 1000),
...     scores=np.random.randn(1000),
...     cv_scores=np.random.randn(5, 1000),
...     cv_mean_score=np.random.randn(1000),
...     cv_predictions=np.random.randn(100, 1000),
...     cv_folds=np.arange(100) % 5
... )
>>> 'cv_scores' in fit_cv.available()
True
```

Immutability:

```pycon
>>> try:
...     fit.scores = np.zeros(1000)  # Will raise FrozenInstanceError
... except AttributeError:
...     print("Cannot modify frozen dataclass")
Cannot modify frozen dataclass
```

Export/serialization:

```pycon
>>> # Save to .npz
>>> np.savez("results.npz", **fit.asdict())
>>>
>>> # Load and reconstruct
>>> loaded = np.load("results.npz")
>>> fit_reloaded = Fit(**{k: loaded[k] for k in loaded.files})
```

<details class="note" open markdown="1">
<summary>Note</summary>

- Frozen dataclass ensures results cannot be accidentally modified.
- All attributes are numpy arrays (except cv_best_alpha which is float).
- None values indicate that field was not computed for this model/method.
- Private fields (starting with _) are excluded from available() and asdict().

</details>

**Methods:**

Name | Description
---- | -----------
[`asdict`](#data-asdict) | Convert to dictionary.
[`available`](#data-available) | Return list of non-None attribute names.

##### Methods

(data-asdict)=
###### `asdict`

```python
asdict(include_none: bool = False) -> dict
```

Convert to dictionary.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`include_none` | <code>[bool](#bool)</code> | If True, include attributes with None values. Private fields (starting with _) are always excluded. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary of attribute names to values.

**Examples:**

```pycon
>>> import numpy as np
>>> from nltools.data.fitresults import Fit
>>> fit = Fit(
...     fitted_values=np.random.randn(100, 1000),
...     weights=np.random.randn(5, 1000),
...     scores=None
... )
>>> d = fit.asdict(include_none=False)
>>> 'scores' in d
False
>>> d = fit.asdict(include_none=True)
>>> 'scores' in d
True
>>> d['scores'] is None
True
```

(data-available)=
###### `available`

```python
available() -> list
```

Return list of non-None attribute names.

Excludes private fields (starting with _).

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)</code> | Names of attributes that are not None.

**Examples:**

```pycon
>>> import numpy as np
>>> from nltools.data.fitresults import Fit
>>> fit = Fit(
...     fitted_values=np.random.randn(100, 1000),
...     weights=np.random.randn(5, 1000)
... )
>>> fit.available()
['fitted_values', 'weights']
>>> 'scores' in fit.available()
False
```

(data-roc)=
#### `Roc`

```python
Roc(*, input_values = None, binary_outcome = None, method = 'optimal_overall', forced_choice = None)
```

Compute receiver operating characteristic curves for single-interval or forced-choice classification.

The Roc class is based on Tor Wager's Matlab roc_plot.m function and
allows a user to easily run different types of receiver operator
characteristic curves.  For example, one might be interested in single
interval or forced choice.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`input_values` |  | 1-D array/vector of continuous decision values (one per observation) | <code>None</code>
`binary_outcome` |  | vector of training labels | <code>None</code>
`method` |  | threshold-selection variant, one of `'optimal_overall'`, `'optimal_balanced'`, `'minimum_sdt_bias'` | <code>'optimal_overall'</code>
`forced_choice` |  | index indicating position for each unique subject (default=None) | <code>None</code>

**Methods:**

Name | Description
---- | -----------
[`calculate`](#data-calculate) | Calculate ROC metrics for single-interval classification.
[`plot`](#data-plot) | Create a ROC plot.
[`summary`](#data-summary) | Display a formatted summary of ROC analysis.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`binary_outcome`](#data-binary-outcome) |  | 
`forced_choice` |  | 
`input_values` |  | 
`method` |  | 

##### Methods

(data-calculate)=
###### `calculate`

```python
calculate(*, input_values = None, binary_outcome = None, criterion_values = None, method = 'optimal_overall', forced_choice = None, balanced_acc = False)
```

Calculate ROC metrics for single-interval classification.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`input_values` |  | 1-D array/vector of continuous decision values (one per observation) | <code>None</code>
`binary_outcome` |  | vector of training labels | <code>None</code>
`criterion_values` |  | (optional) criterion values for calculating fpr             & tpr | <code>None</code>
`method` |  | threshold-selection variant, one of `'optimal_overall'`,             `'optimal_balanced'`, `'minimum_sdt_bias'` | <code>'optimal_overall'</code>
`forced_choice` |  | index indicating position for each unique subject             (default=None) | <code>None</code>
`balanced_acc` |  | balanced accuracy for single-interval classification             (bool). THIS IS NOT COMPLETELY IMPLEMENTED BECAUSE             IT AFFECTS ACCURACY ESTIMATES, BUT NOT P-VALUES OR             THRESHOLD AT WHICH TO EVALUATE SENS/SPEC | <code>False</code>

###### `plot`

```python
plot(*, method = 'gaussian', balanced_acc = False)
```

Create a ROC plot.

Create a specific kind of ROC curve plot, based on input values
along a continuous distribution and a binary outcome variable (logical)

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` |  | type of plot, one of `'gaussian'`, `'observed'` | <code>'gaussian'</code>
`balanced_acc` |  | balanced accuracy for single-interval classification | <code>False</code>

**Returns:**

Type | Description
---- | -----------
 | fig

(data-summary)=
###### `summary`

```python
summary()
```

Display a formatted summary of ROC analysis.

(data-simulategrid)=
#### `SimulateGrid`

```python
SimulateGrid(*, grid_width = 100, signal_width = 20, n_subjects = 20, sigma = 1, signal_amplitude = None, random_state = None)
```

Simulate 2D grid data for testing statistical methods.

Creates a 2D grid (e.g., 100x100 pixels) with optional embedded signal
regions and Gaussian noise. Useful for testing multiple comparison
correction methods, threshold selection, and visualization of
statistical maps.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`grid_width` |  | Width/height of the square grid (default: 100). | <code>100</code>
`signal_width` |  | Width of the embedded signal region (default: 20). | <code>20</code>
`n_subjects` |  | Number of simulated subjects (default: 20). | <code>20</code>
`sigma` |  | Standard deviation of the Gaussian noise (default: 1). | <code>1</code>
`signal_amplitude` |  | Amplitude of the embedded signal. If None, no signal is added. | <code>None</code>
`random_state` |  | Random seed or numpy RandomState for reproducibility. | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`data`](#data-data) |  | The simulated data array of shape (n_subjects, grid_width, grid_width).
`t_values` |  | T-statistic values after fitting.
`p_values` |  | P-values after fitting.
`thresholded` |  | Thresholded statistical map.
`isfit` |  | Whether fit() has been called.

**Examples:**

```pycon
>>> from nltools.data.simulator import SimulateGrid
>>> sim = SimulateGrid(signal_amplitude=0.5, random_state=42)
>>> sim.fit()
>>> sim.plot()
```

**Methods:**

Name | Description
---- | -----------
[`add_signal`](#data-add-signal) | Add a rectangular signal to self.data.
[`create_mask`](#data-create-mask) | Create a mask for where the signal is located in grid.
[`fit`](#data-fit) | Run a one-sample t-test on self.data.
[`plot_grid_simulation`](#data-plot-grid-simulation) | Create a plot of the simulations.
[`run_multiple_simulations`](#data-run-multiple-simulations) | Run multiple simulations to calculate the overall false positive rate.
[`threshold_simulation`](#data-threshold-simulation) | Threshold the fitted simulation.

##### Methods

(data-add-signal)=
###### `add_signal`

```python
add_signal(signal_width = 20, signal_amplitude = 1)
```

Add a rectangular signal to self.data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`signal_width` | <code>[int](#int)</code> | width of signal box | <code>20</code>
`signal_amplitude` | <code>[int](#int)</code> | intensity of signal | <code>1</code>

(data-create-mask)=
###### `create_mask`

```python
create_mask(signal_width)
```

Create a mask for where the signal is located in grid.

###### `fit`

```python
fit()
```

Run a one-sample t-test on self.data.

(data-plot-grid-simulation)=
###### `plot_grid_simulation`

```python
plot_grid_simulation(threshold, threshold_type, n_simulations = 100, correction = None)
```

Create a plot of the simulations.

(data-run-multiple-simulations)=
###### `run_multiple_simulations`

```python
run_multiple_simulations(threshold, threshold_type, n_simulations = 100, correction = None)
```

Run multiple simulations to calculate the overall false positive rate.

(data-threshold-simulation)=
###### `threshold_simulation`

```python
threshold_simulation(threshold, threshold_type, correction = None)
```

Threshold the fitted simulation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`threshold` | <code>[float](#float)</code> | threshold to apply to simulation | *required*
`threshhold_type` | <code>[str](#str)</code> | type of threshold to use can be a specific t-value or p-value ['t', 'p', 'q'] | *required*

(data-simulator)=
#### `Simulator`

```python
Simulator(*, brain_mask = None, output_dir = None, random_state = None)
```

Simulate fMRI data with realistic spatial and temporal characteristics.

This class provides methods for generating synthetic fMRI data with
controlled signal patterns, including Gaussian blobs, multi-subject
datasets, and various noise structures. Useful for testing analysis
pipelines and power analyses.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_mask` |  | Path to a NIfTI brain mask file, a nibabel image object, or None to use the default MNI template mask. | <code>None</code>
`output_dir` |  | Directory for saving generated data. Defaults to the current working directory. | <code>None</code>
`random_state` |  | Random seed or numpy RandomState for reproducibility. | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`brain_mask`](#data-brain-mask) |  | The brain mask image used for simulation.
`output_dir` |  | Output directory path.
`random_state` |  | Random state for reproducible simulations.

**Examples:**

```pycon
>>> from nltools.data.simulator import Simulator
>>> sim = Simulator(random_state=42)
>>> # Create a dataset with signal in specific regions
>>> data = sim.create_data(levels=[1, -1, 1, -1], sigma=1, reps=10)
```

**Methods:**

Name | Description
---- | -----------
[`create_cov_data`](#data-create-cov-data) | Create continuous simulated data with covariance within a single region.
[`create_data`](#data-create-data) | Create simulated data with discrete intensity levels.
[`create_ncov_data`](#data-create-ncov-data) | Create continuous simulated data with covariance across multiple regions.
[`gaussian`](#data-gaussian) | Create a 3D gaussian signal normalized to a given intensity.
[`n_spheres`](#data-n-spheres) | Generate a set of spheres in the brain mask space.
[`normal_noise`](#data-normal-noise) | Produce a normal noise distribution for all points in the brain mask.
[`sphere`](#data-sphere) | Create a sphere of given radius at some point p in the brain mask.
[`to_nifti`](#data-to-nifti) | Convert a numpy matrix to the nifti format and assign it the brain_mask's affine matrix.

##### Methods

(data-create-cov-data)=
###### `create_cov_data`

```python
create_cov_data(cor, cov, sigma, *, mask = None, reps = 1, n_sub = 1, output_dir = None)
```

Create continuous simulated data with covariance within a single region.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cor` |  | amount of covariance between each voxel and Y variable | *required*
`cov` |  | amount of covariance between voxels | *required*
`sigma` |  | amount of noise to add | *required*
`mask` |  | region where activations are placed (a single mask image); defaults to a sphere if None | <code>None</code>
`reps` |  | number of data repetitions | <code>1</code>
`n_sub` |  | number of subjects to simulate | <code>1</code>
`output_dir` |  | string path of directory to output data.  If None, no data will be written | <code>None</code>

(data-create-data)=
###### `create_data`

```python
create_data(levels, sigma, *, radius = 5, center = None, reps = 1, output_dir = None)
```

Create simulated data with discrete intensity levels.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`levels` |  | vector of intensities or class labels | *required*
`sigma` |  | amount of noise to add | *required*
`radius` |  | vector of radius.  Will create multiple spheres if len(radius) > 1 | <code>5</code>
`center` |  | center(s) of sphere(s) of the form [px, py, pz] or [[px1, py1, pz1], ..., [pxn, pyn, pzn]] | <code>None</code>
`reps` |  | number of data repetitions useful for trials or subjects | <code>1</code>
`output_dir` |  | string path of directory to output data.  If None, no data will be written | <code>None</code>

(data-create-ncov-data)=
###### `create_ncov_data`

```python
create_ncov_data(cor, cov, sigma, *, masks = None, reps = 1, n_sub = 1, output_dir = None)
```

Create continuous simulated data with covariance across multiple regions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cor` |  | amount of covariance between each voxel and Y variable (an int or a vector) | *required*
`cov` |  | amount of covariance between voxels (an int or a matrix) | *required*
`sigma` |  | amount of noise to add | *required*
`masks` |  | region(s) where we will have activations (list if more than one) | <code>None</code>
`reps` |  | number of data repetitions | <code>1</code>
`n_sub` |  | number of subjects to simulate | <code>1</code>
`output_dir` |  | string path of directory to output data.  If None, no data will be written | <code>None</code>

(data-gaussian)=
###### `gaussian`

```python
gaussian(mu, sigma, i_tot)
```

Create a 3D gaussian signal normalized to a given intensity.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mu` |  | average value of the gaussian signal (usually set to 0) | *required*
`sigma` |  | standard deviation | *required*
`i_tot` |  | sum total of activation (numerical integral over the gaussian returns this value) | *required*

(data-n-spheres)=
###### `n_spheres`

```python
n_spheres(radius, center)
```

Generate a set of spheres in the brain mask space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`radius` |  | vector of radius.  Will create multiple spheres if len(radius) > 1 | *required*
`centers` |  | a vector of sphere centers of the form [px, py, pz] or [[px1, py1, pz1], ..., [pxn, pyn, pzn]] | *required*

(data-normal-noise)=
###### `normal_noise`

```python
normal_noise(mu, sigma)
```

Produce a normal noise distribution for all points in the brain mask.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mu` |  | average value of the gaussian signal (usually set to 0) | *required*
`sigma` |  | standard deviation | *required*

(data-sphere)=
###### `sphere`

```python
sphere(r, p)
```

Create a sphere of given radius at some point p in the brain mask.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`r` |  | radius of the sphere | *required*
`p` |  | point (in coordinates of the brain mask) of the center of the sphere | *required*

###### `to_nifti`

```python
to_nifti(m)
```

Convert a numpy matrix to the nifti format and assign it the brain_mask's affine matrix.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`m` |  | the 3D numpy matrix we wish to convert to .nii | *required*



### Modules

#### `adjacency`

Provide data structures for working with similarity and dissimilarity matrices.

**Modules:**

Name | Description
---- | -----------
[`io`](#data-io) | I/O functions for Adjacency objects.
[`modeling`](#data-modeling) | Provide standalone modeling and inference functions for Adjacency matrices.
[`plotting`](#data-plotting) | Plotting functions for Adjacency matrices.
[`spatial`](#data-spatial) | Spatial-scale provenance for stacked Adjacency matrices.
[`stats`](#data-stats) | Provide standalone statistical functions for Adjacency matrices.
[`utils`](#data-utils) | Shared helpers for Adjacency submodules.

**Classes:**

Name | Description
---- | -----------
[`Adjacency`](#data-adjacency) | Represent adjacency matrices in vectorized form.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`MAX_INT` |  | 
`nx` |  | 

##### Classes

###### `Adjacency`

```python
Adjacency(data = None, *, Y = None, matrix_type = None, labels = None, spatial_scale: SpatialScale | None = None)
```

Represent adjacency matrices in vectorized form.

Adjacency is a class to represent Adjacency matrices as a vector rather
than a 2-dimensional matrix. This makes it easier to perform data
manipulation and analyses.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | pandas data instance or list of files | <code>None</code>
`matrix_type` |  | (str) type of matrix.  Possible values include:         ['distance','similarity','directed','distance_flat',         'similarity_flat','directed_flat'] | <code>None</code>
`Y` |  | Pandas DataFrame of training labels | <code>None</code>
`labels` |  | (list) optional node labels | <code>None</code>

**Methods:**

Name | Description
---- | -----------
[`append`](#data-append) | Append data to an Adjacency instance.
[`bootstrap`](#data-bootstrap) | Bootstrap statistics using efficient online algorithms.
[`cluster_summary`](#data-cluster-summary) | Provide summaries of clusters within Adjacency matrices.
[`copy`](#data-copy) | Create a copy of Adjacency object.
[`distance`](#data-distance) | Calculate distance between images within an Adjacency() instance.
[`distance_to_similarity`](#data-distance-to-similarity) | Convert distance matrix to similarity matrix.
[`generate_permutations`](#data-generate-permutations) | Generate permuted versions of an Adjacency instance lazily.
[`mean`](#data-mean) | Calculate mean of Adjacency.
[`median`](#data-median) | Calculate median of Adjacency.
[`plot`](#data-plot) | Create a heatmap of an Adjacency matrix.
[`plot_label_distance`](#data-plot-label-distance) | Create a violin plot of within- and between-label distances.
[`plot_mds`](#data-plot-mds) | Plot multidimensional scaling.
[`plot_silhouette`](#data-plot-silhouette) | Create a silhouette plot.
[`r_to_z`](#data-r-to-z) | Apply Fisher's r-to-z transformation to each data element.
[`regress`](#data-regress) | Run a regression on an adjacency instance.
[`similarity`](#data-similarity) | Calculate similarity between two Adjacency matrices.
[`social_relations_model`](#data-social-relations-model) | Estimate the social relations model from a matrix for a round-robin design.
[`squareform`](#data-squareform) | Convert adjacency data back to square form.
[`stats_label_distance`](#data-stats-label-distance) | Calculate permutation tests on within and between label distance.
[`std`](#data-std) | Calculate standard deviation of Adjacency.
[`sum`](#data-sum) | Calculate sum of Adjacency.
[`threshold`](#data-threshold) | Threshold an Adjacency instance.
[`to_brain`](#data-to-brain) | Project per-matrix scalars back to voxel-space `BrainData`.
[`to_graph`](#data-to-graph) | Convert a single Adjacency matrix into a NetworkX graph.
[`to_square`](#data-to-square) | Convert adjacency back to square matrix format.
[`ttest`](#data-ttest) | Calculate ttest across samples.
[`write`](#data-write) | Write out Adjacency object to csv file.
[`z_to_r`](#data-z-to-r) | Convert each z score back into an r value.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`Y`](#data-y) | <code>[DataFrame](#polars.DataFrame)</code> | Training labels as a polars DataFrame (possibly empty).
[`data`](#data-data) |  | 
`is_empty` | <code>[bool](#bool)</code> | Check if Adjacency object is empty.
`is_single_matrix` |  | 
`issymmetric` |  | 
`labels` |  | 
`matrix_type` |  | 
`n_nodes` |  | Return the number of nodes in the adjacency matrix.
`shape` |  | Return the logical shape of the adjacency matrix.
`spatial_scale` | <code>[SpatialScale](#nltools.data.adjacency.spatial.SpatialScale) \| None</code> | 
`vector_shape` |  | Return shape of internal vectorized representation.



####### Attributes##

(data-y)=
###### `Y`

```python
Y: pl.DataFrame
```

Training labels as a polars DataFrame (possibly empty).

######## `data`

```python
data = np.array(f['data'])
```

######## `is_empty`

```python
is_empty: bool
```

Check if Adjacency object is empty.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`bool` | <code>[bool](#bool)</code> | True if the adjacency matrix is empty, False otherwise.

######## `is_single_matrix`

```python
is_single_matrix = f['is_single_matrix'][]
```

######## `issymmetric`

```python
issymmetric = f['issymmetric'][]
```

######## `labels`

```python
labels = deepcopy(tmp.labels)
```

######## `matrix_type`

```python
matrix_type = f['matrix_type'][].decode()
```

######## `n_nodes`

```python
n_nodes
```

Return the number of nodes in the adjacency matrix.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`int` |  | Number of nodes (n) for an (n, n) matrix.

######## `shape`

```python
shape
```

Return the logical shape of the adjacency matrix.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` |  | For single matrix: (n_nodes, n_nodes)    For stacked matrices: (n_matrices, n_nodes, n_nodes)    For empty: (0, 0)

<details class="note" open markdown="1">
<summary>Note</summary>

Use `.vector_shape` to get the internal vectorized representation shape.

</details>

######## `spatial_scale`

```python
spatial_scale: SpatialScale | None = spatial_scale
```

######## `vector_shape`

```python
vector_shape
```

Return shape of internal vectorized representation.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` |  | For single matrix: (vector_length,)    For stacked matrices: (n_matrices, vector_length)

<details class="note" open markdown="1">
<summary>Note</summary>

This is the raw shape of the internal data storage.
Use `.shape` for the logical (n_nodes, n_nodes) shape.

</details>



####### Functions##

###### `append`

```python
append(data)
```

Append data to an Adjacency instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (Adjacency) Adjacency instance to append | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (Adjacency) new appended Adjacency instance

######## `bootstrap`

```python
bootstrap(stat, *, n_samples = 5000, save_boots = False, percentiles = (2.5, 97.5), n_jobs = -1, random_state = None)
```

Bootstrap statistics using efficient online algorithms.

Uses memory-efficient bootstrap infrastructure with CPU parallelization.
Supports simple aggregation statistics (mean, std, median, sum, min, max).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stat` |  | (str) Statistic to bootstrap. Options: - Simple stats: 'mean', 'median', 'std', 'sum', 'min', 'max' | *required*
`n_samples` |  | (int) Number of bootstrap iterations. Default: 5000 | <code>5000</code>
`save_boots` |  | (bool) If True, store all bootstrap samples (memory intensive).        Default: False | <code>False</code>
`percentiles` |  | (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5) | <code>(2.5, 97.5)</code>
`n_jobs` |  | (int) Number of CPU cores for parallelization. -1 means all CPUs. | <code>-1</code>
`random_state` |  | (int, optional) Random seed for reproducibility | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | Dictionary with keys: 'Z', 'p', 'mean', 'std', 'ci_lower', 'ci_upper'   (all Adjacency objects). If save_boots=True, also includes 'samples'.

**Examples:**

```pycon
>>> # Simple aggregation
>>> boot = adj.bootstrap(stat='mean', n_samples=1000)
>>> assert 'mean' in boot
>>> assert isinstance(boot['mean'], Adjacency)
```

######## `cluster_summary`

```python
cluster_summary(*, clusters = None, method = 'mean', summary = 'within')
```

Provide summaries of clusters within Adjacency matrices.

Computes mean/median of within and between cluster values. Requires a
list of cluster ids indicating the row/column of each cluster.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`clusters` |  | (list) list of cluster labels | <code>None</code>
`method` |  | (str) how to summarize, 'mean' or 'median'. If `None` then return all r values | <code>'mean'</code>
`summary` |  | (str) summarize within cluster or between clusters | <code>'within'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | within cluster means

######## `copy`

```python
copy()
```

Create a copy of Adjacency object.

######## `distance`

```python
distance(metric = 'correlation', include_diag = False, **kwargs)
```

Calculate distance between images within an Adjacency() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` |  | (str) type of distance metric (can use any scikit learn or     scipy metric) | <code>'correlation'</code>
`include_diag` |  | (bool) whether to include the main diagonal when     computing distances between adjacency matrices. Only applies     to symmetric matrices. Default False (consistent with how     symmetric matrices are stored without diagonal). | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dist` |  | (Adjacency) Outputs a 2D distance matrix.

######## `distance_to_similarity`

```python
distance_to_similarity(metric = 'correlation', beta = 1)
```

Convert distance matrix to similarity matrix.

Note: currently only implemented for correlation and euclidean.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` |  | (str) Can only be correlation or euclidean | <code>'correlation'</code>
`beta` |  | (float) parameter to scale exponential function (default: 1) for euclidean | <code>1</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (Adjacency) Adjacency object

######## `generate_permutations`

```python
generate_permutations(n_permute, random_state = None)
```

Generate permuted versions of an Adjacency instance lazily.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_permute` | <code>[int](#int)</code> | number of permutations | *required*
`random_state` | <code>([int](#int), [seed](#numpy.random.seed))</code> | random seed for reproducibility. | <code>None</code>

**Examples:**

```pycon
>>> for perm in adj.generate_permutations(1000):
>>>     out = neural_distance_mat.similarity(perm)
>>>     ...
```

**Yields:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | permuted version of self

######## `mean`

```python
mean(axis = 0)
```

Calculate mean of Adjacency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | Calculate mean over matrices (0) or upper triangle (1). | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float if single matrix, Adjacency if axis=0, np.array if axis=1.

######## `median`

```python
median(axis = 0)
```

Calculate median of Adjacency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | Calculate median over matrices (0) or upper triangle (1). | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float if single matrix, Adjacency if axis=0, np.array if axis=1.

######## `plot`

```python
plot(limit = 3, axes = None, *args, **kwargs)
```

Create a heatmap of an Adjacency matrix.

Can pass in any sns.heatmap argument

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`limit` |  | (int) number of heatmaps to plot if object contains multiple adjacencies (default: 3) | <code>3</code>
`axes` |  | matplotlib axis handle | <code>None</code>

######## `plot_label_distance`

```python
plot_label_distance(labels = None, ax = None)
```

Create a violin plot of within- and between-label distances.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`labels` | <code>[array](#numpy.array)</code> | numpy array of labels to plot | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`f` |  | violin plot handles

######## `plot_mds`

```python
plot_mds(*, n_components = 2, metric_mds = True, labels = None, labels_color = None, cmap = None, view = (30, 20), figsize = None, ax = None, n_jobs = -1, **kwargs)
```

Plot multidimensional scaling.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_components` |  | (int) Number of dimensions to project (can be 2 or 3) | <code>2</code>
`metric_mds` |  | (bool) Perform metric (True) or non-metric (False) dimensional scaling; default True | <code>True</code>
`labels` |  | (list) Can override labels stored in Adjacency Class | <code>None</code>
`labels_color` |  | (str) list of colors for labels, if len(1) then make all same color | <code>None</code>
`cmap` |  | colormap instance (default: plt.cm.hot_r) | <code>None</code>
`view` |  | (tuple) view for 3-Dimensional plot; default (30,20) | <code>(30, 20)</code>
`figsize` |  | (list) figure size; default [12, 8] | <code>None</code>
`ax` |  | matplotlib axis handle | <code>None</code>
`n_jobs` |  | (int) Number of parallel jobs | <code>-1</code>

######## `plot_silhouette`

```python
plot_silhouette(*, labels = None, ax = None, permutation_test = True, n_permute = 5000, colors = None, figsize = (6, 4))
```

Create a silhouette plot.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`labels` |  | Numpy array of cluster/group labels (overrides stored labels). | <code>None</code>
`ax` |  | Matplotlib axis handle. | <code>None</code>
`permutation_test` |  | (bool) Whether to run a permutation test. Default True. | <code>True</code>
`n_permute` |  | (int) Number of permutations for the test. Default 5000. | <code>5000</code>
`colors` |  | Optional list of RGB triplets, one per cluster (default: seaborn 'hls' palette). | <code>None</code>
`figsize` |  | Figure size tuple. Default (6, 4). | <code>(6, 4)</code>

######## `r_to_z`

```python
r_to_z()
```

Apply Fisher's r-to-z transformation to each data element.

######## `regress`

```python
regress(X, method = 'ols')
```

Run a regression on an adjacency instance.
You can decompose an adjacency instance with another adjacency instance.
You can also decompose each pixel by passing a design_matrix instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` |  | Design matrix can be an Adjacency or DesignMatrix instance | *required*
`method` |  | type of regression (default: ols) - only 'ols' is currently supported | <code>'ols'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`stats` |  | (dict) dictionary of stats outputs.

######## `similarity`

```python
similarity(data, *, plot = False, method = '2d', n_permute = 5000, metric = 'spearman', include_diag = False, nan_policy = 'omit', tail = 2, return_null = False, n_jobs = -1, random_state = None, project: bool = False)
```

Calculate similarity between two Adjacency matrices.

The default uses Spearman correlation and a permutation test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Adjacency](#nltools.data.adjacency.Adjacency) or [array](#array)</code> | Adjacency data, or 1-d array same size as self.data | *required*
`method` |  | (str) permutation scheme '1d', '2d', or None | <code>'2d'</code>
`metric` |  | (str) 'spearman','pearson','kendall' | <code>'spearman'</code>
`include_diag` |  | (bool) only applies to 'directed' Adjacency types using method=None or method='1d'. Default False (self-similarity is uninformative). Symmetric matrices never store the diagonal, so this flag is a no-op for them. | <code>False</code>
`nan_policy` |  | (str) How to handle NaN values. Options: - 'omit': Remove NaN values pairwise before computing correlation (default) - 'propagate': Allow NaN to propagate through calculations - 'raise': Raise an error if NaN values are present | <code>'omit'</code>
`tail` |  | (int) Tail of the test (1 or 2). Default 2. | <code>2</code>
`return_null` |  | (bool) If True, also return the null distribution. Default False. | <code>False</code>
`n_jobs` |  | (int) Number of parallel jobs. Default -1 (all cores). | <code>-1</code>
`random_state` |  | (int, optional) Random seed for reproducibility. | <code>None</code>
`project` | <code>[bool](#bool)</code> | (bool) If True and this Adjacency has a spatial_scale, project the per-matrix correlations back into brain space. Default False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
 | dict or list or BrainData: A correlation result dict with keys 'r' and 'p' for a single matrix, a list of such dicts when this Adjacency holds multiple matrices, or a `BrainData` when `project=True` (per-matrix correlations projected via spatial_scale).

######## `social_relations_model`

```python
social_relations_model(summarize_results = True, nan_replace = True)
```

Estimate the social relations model from a matrix for a round-robin design.

X_{ij} = m + \alpha_i + \beta_j + g_{ij} + \epsilon_{ijl}

where X_{ij} is the score for person i rating person j, m is the group mean,
\alpha_i  is person i's actor effect, \beta_j is person j's partner effect, g_{ij}
is the relationship  effect and \epsilon_{ijl} is the error in measure l  for actor i and partner j.

This model is primarily concerned with partioning the variance of the various effects.

Code is based on implementation presented in Chapter 8 of Kenny, Kashy, & Cook (2006).
Tests replicate examples  presented in the book. Note, that this method assumes that
actor scores are rows (lower triangle), while partner scores are columnns (upper triangle).
The minimal sample size to estimate these effects is 4.

<details class="model-assumptions" open markdown="1">
<summary>Model Assumptions</summary>

- Social interactions are exclusively dyadic
- People are randomly sampled from population
- No order effects
- The effects combine additively and relationships are linear

</details>

In the future we might update the formulas and standard errors based on
Bond and Lashley, 1996

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`self` |  | (adjacency) can be a single matrix or many matrices for each group | *required*
`summarize_results` |  | (bool) will provide a formatted summary of model results | <code>True</code>
`nan_replace` |  | (bool) will replace nan values with row and column means | <code>True</code>

**Returns:**

Type | Description
---- | -----------
 | estimated effects: (pd.Series/pd.DataFrame) All of the effects estimated using SRM

######## `squareform`

```python
squareform()
```

Convert adjacency data back to square form.

######## `stats_label_distance`

```python
stats_label_distance(*, labels = None, n_permute = 5000, n_jobs = -1)
```

Calculate permutation tests on within and between label distance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`labels` | <code>[array](#numpy.array)</code> | numpy array of labels to plot | <code>None</code>
`n_permute` | <code>[int](#int)</code> | number of permutations to run (default=5000) | <code>5000</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | dictionary of within and between group differences     and p-values

######## `std`

```python
std(axis = 0)
```

Calculate standard deviation of Adjacency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | Calculate std over matrices (0) or upper triangle (1). | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float if single matrix, Adjacency if axis=0, np.array if axis=1.

######## `sum`

```python
sum(axis = 0)
```

Calculate sum of Adjacency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | Calculate sum over matrices (0) or upper triangle (1). | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float if single matrix, Adjacency if axis=0, np.array if axis=1.

######## `threshold`

```python
threshold(*, upper = None, lower = None, binarize = False)
```

Threshold an Adjacency instance.

Provide upper and lower values or percentages to perform two-sided
thresholding. Binarize will return a mask image respecting thresholds
if provided, otherwise respecting every non-zero value.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`upper` |  | (float or str) Upper cutoff for thresholding. If string     will interpret as percentile; can be None for one-sided     thresholding. | <code>None</code>
`lower` |  | (float or str) Lower cutoff for thresholding. If string     will interpret as percentile; can be None for one-sided     thresholding. | <code>None</code>
`binarize` | <code>[bool](#bool)</code> | return binarized image respecting thresholds if     provided, otherwise binarize on every non-zero value;     default False | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | thresholded Adjacency instance

######## `to_brain`

```python
to_brain(values, *, fill: float = np.nan)
```

Project per-matrix scalars back to voxel-space `BrainData`.

Requires `spatial_scale` to be set (i.e. this stack came from
`BrainData.distance` or another spatial-scale-aware producer).
Each entry of ``values`` is painted onto the voxels assigned to its
corresponding parcel by ``spatial_scale.atlas`` /
``spatial_scale.roi_labels``. Voxels outside the atlas receive
``fill``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`values` |  | 1-D array of length ``len(self)`` — one scalar per matrix in the stack. | *required*
`fill` | <code>[float](#float)</code> | Value for voxels not covered by any provided ROI label. Default ``np.nan``. | <code>[nan](#numpy.nan)</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Single image masked to ``spatial_scale.source_mask``.

**Examples:**

```pycon
>>> rdms = brain.distance(metric='correlation',
...                       spatial_scale='roi', roi_mask=atlas)
>>> sims = rdms.similarity(model_rdm)
>>> brain_map = rdms.to_brain(sims)
```

######## `to_graph`

```python
to_graph()
```

Convert a single Adjacency matrix into a NetworkX graph.

This currently works only for ``single_matrix``.

######## `to_square`

```python
to_square()
```

Convert adjacency back to square matrix format.

This is an alias for `squareform`.

**Returns:**

Type | Description
---- | -----------
 | np.ndarray or list: Square matrix representation. Returns a list
 | of matrices if this object contains multiple adjacency matrices.

######## `ttest`

```python
ttest(*, permutation = False, n_permute = 5000, tail = 2, return_null = False, n_jobs = -1, random_state = None)
```

Calculate ttest across samples.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`permutation` |  | (bool) Run ttest as permutation. Note this can be very slow. | <code>False</code>
`n_permute` |  | Number of permutations (used only when ``permutation=True``). Default 5000. | <code>5000</code>
`tail` |  | Tail of the test (1 or 2). Default 2. | <code>2</code>
`return_null` |  | If True, also return the null distribution. Default False. | <code>False</code>
`n_jobs` |  | Number of parallel jobs. Default -1 (all cores). | <code>-1</code>
`random_state` |  | Random seed for reproducibility. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (dict) contains Adjacency instances of t values (or mean if  running permutation) and Adjacency instance of p values.

######## `write`

```python
write(file_name, method = 'long')
```

Write out Adjacency object to csv file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>[str](#str)</code> | name of file name to write | *required*
`method` | <code>[str](#str)</code> | method to write out data ['long','square'] | <code>'long'</code>

######## `z_to_r`

```python
z_to_r()
```

Convert each z score back into an r value.



##### Methods

##### Modules

(data-io)=
###### `io`

I/O functions for Adjacency objects.

**Methods:**

Name | Description
---- | -----------
[`to_graph`](#data-to-graph) | Convert Adjacency into networkx graph.
[`write`](#data-write) | Write out Adjacency object to csv file.



####### Functions##

###### `to_graph`

```python
to_graph(adj)
```

Convert Adjacency into networkx graph.

Only works on single matrices for now.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency instance (must be a single matrix). | *required*

**Returns:**

Type | Description
---- | -----------
 | networkx.Graph or networkx.DiGraph: Graph representation of the adjacency matrix. Uses DiGraph for directed matrices.

######## `write`

```python
write(adj, file_name, method = 'long')
```

Write out Adjacency object to csv file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` |  | Adjacency object to write | *required*
`file_name` | <code>[str](#str)</code> | name of file name to write | *required*
`method` | <code>[str](#str)</code> | method to write out data ['long','square'] | <code>'long'</code>

(data-modeling)=
###### `modeling`

Provide standalone modeling and inference functions for Adjacency matrices.

Each function takes an Adjacency instance as its first argument (`adj`).

**Methods:**

Name | Description
---- | -----------
[`bootstrap`](#data-bootstrap) | Bootstrap statistics using efficient online algorithms.
`convert_bootstrap_results_to_adjacency` | Convert bootstrap results dictionary to Adjacency format.
[`generate_permutations`](#data-generate-permutations) | Generate permuted versions of an Adjacency instance lazily.
[`regress`](#data-regress) | Run a regression on an adjacency instance.
[`social_relations_model`](#data-social-relations-model) | Estimate the social relations model from a matrix for a round-robin design.



####### Functions##

###### `bootstrap`

```python
bootstrap(adj, stat, *, n_samples = 5000, save_boots = False, percentiles = (2.5, 97.5), n_jobs = -1, random_state = None)
```

Bootstrap statistics using efficient online algorithms.

Uses memory-efficient bootstrap infrastructure with CPU parallelization.
Supports simple aggregation statistics (mean, std, median, sum, min, max).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` |  | (Adjacency) Adjacency instance containing multiple matrices | *required*
`stat` |  | (str) Statistic to bootstrap. Options: - Simple stats: 'mean', 'median', 'std', 'sum', 'min', 'max' | *required*
`n_samples` |  | (int) Number of bootstrap iterations. Default: 5000 | <code>5000</code>
`save_boots` |  | (bool) If True, store all bootstrap samples (memory intensive).        Default: False | <code>False</code>
`percentiles` |  | (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5) | <code>(2.5, 97.5)</code>
`n_jobs` |  | (int) Number of CPU cores for parallelization. -1 means all CPUs. | <code>-1</code>
`random_state` |  | (int, optional) Random seed for reproducibility | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | Dictionary with keys: 'Z', 'p', 'mean', 'std', 'ci_lower', 'ci_upper'   (all Adjacency objects). If save_boots=True, also includes 'samples'.

**Examples:**

```pycon
>>> # Simple aggregation
>>> boot = bootstrap(adj, stat='mean', n_samples=1000)
>>> assert 'mean' in boot
>>> assert isinstance(boot['mean'], Adjacency)
```

######## `convert_bootstrap_results_to_adjacency`

```python
convert_bootstrap_results_to_adjacency(adj, result, save_boots = False)
```

Convert bootstrap results dictionary to Adjacency format.

Helper function to convert numpy arrays from bootstrap functions into
Adjacency objects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` |  | (Adjacency) Adjacency instance (used for matrix_type metadata) | *required*
`result` |  | (dict) Result dictionary from bootstrap function with keys:     'mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper', and optionally 'samples' | *required*
`save_boots` |  | (bool) If True, include 'samples' key in output | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | Dictionary with Adjacency objects for each statistic

######## `generate_permutations`

```python
generate_permutations(adj, n_permute, random_state = None)
```

Generate permuted versions of an Adjacency instance lazily.

This is useful for iterative comparisons.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` |  | (Adjacency) Adjacency instance | *required*
`n_permute` | <code>[int](#int)</code> | number of permutations | *required*
`random_state` | <code>[int](#int) or [RandomState](#numpy.random.RandomState)</code> | random seed for reproducibility. Defaults to None. | <code>None</code>

**Examples:**

```pycon
>>> for perm in generate_permutations(adj, 1000):
>>>     out = neural_distance_mat.similarity(perm)
>>>     ...
```

**Yields:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | permuted version of adj

######## `regress`

```python
regress(adj, X, method = 'ols')
```

Run a regression on an adjacency instance.
You can decompose an adjacency instance with another adjacency instance.
You can also decompose each pixel by passing a design_matrix instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` |  | (Adjacency) Adjacency instance | *required*
`X` |  | Design matrix can be an Adjacency or DesignMatrix instance | *required*
`method` |  | type of regression (default: ols) - only 'ols' is currently supported | <code>'ols'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`stats` |  | (dict) dictionary of stats outputs.

######## `social_relations_model`

```python
social_relations_model(adj, summarize_results = True, nan_replace = True)
```

Estimate the social relations model from a matrix for a round-robin design.

X_{ij} = m + \alpha_i + \beta_j + g_{ij} + \epsilon_{ijl}

where X_{ij} is the score for person i rating person j, m is the group mean,
\alpha_i  is person i's actor effect, \beta_j is person j's partner effect, g_{ij}
is the relationship  effect and \epsilon_{ijl} is the error in measure l  for actor i and partner j.

This model is primarily concerned with partioning the variance of the various effects.

Code is based on implementation presented in Chapter 8 of Kenny, Kashy, & Cook (2006).
Tests replicate examples  presented in the book. Note, that this method assumes that
actor scores are rows (lower triangle), while partner scores are columnns (upper triangle).
The minimal sample size to estimate these effects is 4.

<details class="model-assumptions" open markdown="1">
<summary>Model Assumptions</summary>

- Social interactions are exclusively dyadic
- People are randomly sampled from population
- No order effects
- The effects combine additively and relationships are linear

</details>

In the future we might update the formulas and standard errors based on
Bond and Lashley, 1996

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` |  | (Adjacency) can be a single matrix or many matrices for each group | *required*
`summarize_results` |  | (bool) will provide a formatted summary of model results | <code>True</code>
`nan_replace` |  | (bool) will replace nan values with row and column means | <code>True</code>

**Returns:**

Type | Description
---- | -----------
 | estimated effects: (pd.Series/pd.DataFrame) All of the effects estimated using SRM

(data-plotting)=
###### `plotting`

Plotting functions for Adjacency matrices.

**Methods:**

Name | Description
---- | -----------
[`plot_adjacency`](#data-plot-adjacency) | Create Heatmap of Adjacency Matrix.
[`plot_mds`](#data-plot-mds) | Plot Multidimensional Scaling.



####### Functions##

(data-plot-adjacency)=
###### `plot_adjacency`

```python
plot_adjacency(adj, limit = 3, axes = None, *args, **kwargs)
```

Create Heatmap of Adjacency Matrix.

Can pass in any ``sns.heatmap`` argument.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency object to plot. | *required*
`limit` | <code>[int](#int)</code> | Number of heatmaps to plot if object contains multiple adjacencies (default: 3). | <code>3</code>
`axes` |  | Matplotlib axis handle. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | None

######## `plot_mds`

```python
plot_mds(adj, *, n_components = 2, metric_mds = True, labels = None, labels_color = None, cmap = None, view = (30, 20), figsize = None, ax = None, n_jobs = -1, **kwargs)
```

Plot Multidimensional Scaling.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency object to plot (must be a distance matrix). | *required*
`n_components` | <code>[int](#int)</code> | Number of dimensions to project (can be 2 or 3). | <code>2</code>
`metric_mds` | <code>[bool](#bool)</code> | Perform metric (True) or non-metric (False) dimensional scaling. Default True. | <code>True</code>
`labels` | <code>[list](#list)</code> | Can override labels stored in Adjacency Class. | <code>None</code>
`labels_color` | <code>[list](#list)</code> | List of colors for labels. | <code>None</code>
`cmap` |  | Colormap instance (default: ``plt.cm.hot_r``). | <code>None</code>
`view` | <code>[tuple](#tuple)</code> | View for 3-Dimensional plot. Default (30, 20). | <code>(30, 20)</code>
`figsize` | <code>[list](#list)</code> | Figure size. Default [12, 8]. | <code>None</code>
`ax` |  | Matplotlib axis handle. | <code>None</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
 | None

(data-spatial)=
###### `spatial`

Spatial-scale provenance for stacked Adjacency matrices.

When a stack of Adjacency matrices comes from a per-parcel or per-searchlight
operation on a BrainData, attaching a `SpatialScale` records the atlas,
the parcel labels in stack order, and the source mask — enough to project
per-matrix reductions back to a voxel-space `BrainData` via
``Adjacency.to_brain()``.

See `Adjacency` for the optional ``spatial_scale`` attribute, and
`BrainData.distance` (with ``spatial_scale='roi'|'searchlight'``) for
the canonical producer.

**Classes:**

Name | Description
---- | -----------
[`SpatialScale`](#data-spatialscale) | Record provenance for a per-parcel or per-searchlight Adjacency stack.



####### Classes##

(data-spatialscale)=
###### `SpatialScale`

```python
SpatialScale(atlas: BrainData, roi_labels: np.ndarray, source_mask: Nifti1Image, kind: Literal['roi', 'searchlight'] = 'roi') -> None
```

Record provenance for a per-parcel or per-searchlight Adjacency stack.

The stack comes from a per-parcel or per-searchlight operation on a
`BrainData`.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`atlas`](#data-atlas) | <code>[BrainData](#nltools.data.BrainData)</code> | Labeled volume indicating parcel membership (or searchlight centers). One matrix in the stack per unique label.
`roi_labels` | <code>[ndarray](#numpy.ndarray)</code> | Integer atlas IDs in stack order. ``len(roi_labels)`` must equal the number of matrices in the stack.
`source_mask` | <code>[Nifti1Image](#nibabel.Nifti1Image)</code> | The brain mask the atlas/values live in. Used as the target space for back-projection in ``Adjacency.to_brain()``.
`kind` | <code>[Literal](#typing.Literal)['roi', 'searchlight']</code> | Which spatial scale produced this stack — ``'roi'`` or ``'searchlight'``.



######### Attributes####

(data-atlas)=
###### `atlas`

```python
atlas: BrainData
```

########## `kind`

```python
kind: Literal['roi', 'searchlight'] = field(default='roi')
```

########## `roi_labels`

```python
roi_labels: np.ndarray
```

########## `source_mask`

```python
source_mask: Nifti1Image
```

(data-stats)=
###### `stats`

Provide standalone statistical functions for Adjacency matrices.

Each function takes an Adjacency instance as its first argument (`adj`).

**Methods:**

Name | Description
---- | -----------
[`cluster_summary`](#data-cluster-summary) | This function provides summaries of clusters within Adjacency matrices.
[`plot_label_distance`](#data-plot-label-distance) | Create a violin plot of within- and between-label distances.
[`plot_silhouette`](#data-plot-silhouette) | Create a silhouette plot.
[`r_to_z`](#data-r-to-z) | Apply Fisher's r to z transformation to each element of the data object.
[`similarity`](#data-similarity) | Calculate similarity between two Adjacency matrices.
[`stats_label_distance`](#data-stats-label-distance) | Calculate permutation tests on within and between label distance.
[`threshold`](#data-threshold) | Threshold an Adjacency instance.
[`ttest`](#data-ttest) | Calculate ttest across samples.
[`z_to_r`](#data-z-to-r) | Convert z score back into r value for each element of data object.



####### Functions##

###### `cluster_summary`

```python
cluster_summary(adj, *, clusters = None, method = 'mean', summary = 'within')
```

This function provides summaries of clusters within Adjacency matrices.

It can compute mean/median of within and between cluster values. Requires a
list of cluster ids indicating the row/column of each cluster.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency instance | *required*
`clusters` |  | (list) list of cluster labels | <code>None</code>
`method` |  | (str) how to summarize, 'mean' or 'median'. If `None` then return all r values | <code>'mean'</code>
`summary` |  | (str) summarize within cluster or between clusters | <code>'within'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | (dict) within cluster means

######## `plot_label_distance`

```python
plot_label_distance(adj, labels = None, ax = None)
```

Create a violin plot of within- and between-label distances.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency instance (must be a single matrix) | *required*
`labels` | <code>[array](#numpy.array)</code> | numpy array of labels to plot | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | None

######## `plot_silhouette`

```python
plot_silhouette(adj, *, labels = None, ax = None, permutation_test = True, n_permute = 5000, colors = None, figsize = (6, 4))
```

Create a silhouette plot.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency instance (must be a single matrix). | *required*
`labels` | <code>[array](#numpy.array)</code> | Numpy array of cluster/group labels. | <code>None</code>
`ax` |  | Matplotlib axis handle. | <code>None</code>
`permutation_test` | <code>[bool](#bool)</code> | Whether to run a permutation test. Default True. | <code>True</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations for the test. Default 5000. | <code>5000</code>
`colors` |  | Optional list of RGB triplets, one per cluster (default: seaborn 'hls' palette). | <code>None</code>
`figsize` |  | Figure size tuple. Default (6, 4). | <code>(6, 4)</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | Silhouette plot results including scores and optional permutation p-value.

######## `r_to_z`

```python
r_to_z(adj)
```

Apply Fisher's r to z transformation to each element of the data object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency instance. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | New Adjacency with z-transformed values.

######## `similarity`

```python
similarity(adj, data, plot = False, method = '2d', n_permute = 5000, metric = 'spearman', include_diag = False, nan_policy = 'omit', tail = 2, return_null = False, n_jobs = -1, random_state = None, *, project: bool = False)
```

Calculate similarity between two Adjacency matrices.

The default uses Spearman correlation and a permutation test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency instance. | *required*
`data` | <code>[Adjacency](#nltools.data.adjacency.Adjacency) or [array](#array)</code> | Adjacency data, or 1-d array same size as adj.data. | *required*
`plot` | <code>[bool](#bool)</code> | If True, plot stacked adjacency matrices. Default False. | <code>False</code>
`method` | <code>[str](#str)</code> | permutation scheme '1d', '2d', or None. | <code>'2d'</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations. Default 5000. | <code>5000</code>
`metric` | <code>[str](#str)</code> | 'spearman', 'pearson', or 'kendall'. | <code>'spearman'</code>
`include_diag` | <code>[bool](#bool)</code> | Only applies to 'directed' Adjacency types using method=None or method='1d'. Default False (self-similarity is uninformative). Symmetric matrices never store the diagonal, so this flag is a no-op for them. | <code>False</code>
`nan_policy` | <code>[str](#str)</code> | How to handle NaN values. Options: - 'omit': Remove NaN values pairwise before computing correlation (default) - 'propagate': Allow NaN to propagate through calculations - 'raise': Raise an error if NaN values are present | <code>'omit'</code>
`tail` | <code>[int](#int)</code> | Tail of the test (1 or 2). Default 2. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, also return the null distribution. Default False. | <code>False</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. -1 means all cores. Default -1. | <code>-1</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | dict or list: Correlation result dict with keys 'r' and 'p', or a list of such dicts when adj contains multiple matrices.
 | BrainData when `project=True` (per-matrix correlations projected via spatial_scale).

######## `stats_label_distance`

```python
stats_label_distance(adj, *, labels = None, n_permute = 5000, n_jobs = -1)
```

Calculate permutation tests on within and between label distance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency instance (must be a single matrix) | *required*
`labels` | <code>[array](#numpy.array)</code> | numpy array of labels to plot | <code>None</code>
`n_permute` | <code>[int](#int)</code> | number of permutations to run (default=5000) | <code>5000</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | dictionary of within and between group differences     and p-values

######## `threshold`

```python
threshold(adj, *, upper = None, lower = None, binarize = False)
```

Threshold an Adjacency instance.

Provide upper and lower values or percentages to perform two-sided
thresholding. Binarize will return a mask image respecting thresholds if
provided, otherwise respecting every non-zero value.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency instance | *required*
`upper` |  | (float or str) Upper cutoff for thresholding. If string     will interpret as percentile; can be None for one-sided     thresholding. | <code>None</code>
`lower` |  | (float or str) Lower cutoff for thresholding. If string     will interpret as percentile; can be None for one-sided     thresholding. | <code>None</code>
`binarize` | <code>[bool](#bool)</code> | return binarized image respecting thresholds if     provided, otherwise binarize on every non-zero value;     default False | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | thresholded Adjacency instance

######## `ttest`

```python
ttest(adj, *, permutation = False, n_permute = 5000, tail = 2, return_null = False, n_jobs = -1, random_state = None)
```

Calculate ttest across samples.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency instance (must contain multiple matrices) | *required*
`permutation` |  | (bool) Run ttest as permutation. Note this can be very slow. | <code>False</code>
`n_permute` |  | Number of permutations (used only when ``permutation=True``). Default 5000. | <code>5000</code>
`tail` |  | Tail of the test (1 or 2). Default 2. | <code>2</code>
`return_null` |  | If True, also return the null distribution. Default False. | <code>False</code>
`n_jobs` |  | Number of parallel jobs. Default -1 (all cores). | <code>-1</code>
`random_state` |  | Random seed for reproducibility. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (dict) contains Adjacency instances of t values (or mean if  running permutation) and Adjacency instance of p values.

######## `z_to_r`

```python
z_to_r(adj)
```

Convert z score back into r value for each element of data object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency instance. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | New Adjacency with r values.

(data-utils)=
###### `utils`

Shared helpers for Adjacency submodules.

These are internal utilities used by the facade and submodules — not part of the
public API.

**Methods:**

Name | Description
---- | -----------
[`apply_stat`](#data-apply-stat) | Apply a statistical function along an axis.
`import_single_data` | Import and validate a single adjacency data matrix.
`perform_arithmetic` | Perform arithmetic operation with validation.
`test_is_single_matrix` | Check whether data represents a single matrix (1-D vector).



####### Functions##

(data-apply-stat)=
###### `apply_stat`

```python
apply_stat(adj, func, axis = 0)
```

Apply a statistical function along an axis.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` |  | Adjacency instance. | *required*
`func` |  | Numpy function to apply (e.g., np.nanmean). | *required*
`axis` |  | Axis along which to apply function. 0 for across matrices,   1 for across upper triangle elements. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float if single matrix, Adjacency if axis=0 with multiple matrices,
 | np.array if axis=1 with multiple matrices.

######## `import_single_data`

```python
import_single_data(data, matrix_type = None)
```

Import and validate a single adjacency data matrix.

Handles file paths (CSV), DataFrames, and numpy arrays. Determines
symmetry and matrix type when not provided.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | File path (str/Path), DataFrame, or numpy array. | *required*
`matrix_type` |  | Optional explicit matrix type ('distance', 'similarity', 'directed', or their '_flat' variants). | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` |  | (data, issymmetric, matrix_type, is_single_matrix)

######## `perform_arithmetic`

```python
perform_arithmetic(adj, y, op, op_name, reverse = False)
```

Perform arithmetic operation with validation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` |  | Adjacency instance (left operand unless *reverse* is True). | *required*
`y` |  | Operand (scalar or Adjacency). | *required*
`op` |  | Callable that performs the operation on arrays. | *required*
`op_name` |  | Name of operation for error messages. | *required*
`reverse` |  | If True, reverse operand order (y op adj). | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | New instance with result.

######## `test_is_single_matrix`

```python
test_is_single_matrix(data)
```

Check whether data represents a single matrix (1-D vector).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | numpy array of adjacency data. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`bool` |  | True if data is 1-D (single matrix in vector form).

(data-atlases)=
#### `atlases`

Atlas registry, lazy loading, and coordinate labeling.

Atlases are hosted at ``huggingface.co/datasets/nltools/niftis`` under
``atlases/`` and fetched on first use via
`fetch_resource`. Cached locally afterwards.

The labeling logic was adapted from
[atlasreader](https://github.com/miykael/atlasreader) (BSD-3-Clause). Cite:

> Notter et al. (2019). AtlasReader. JOSS 4(34), 1257.
> https://doi.org/10.21105/joss.01257

**Modules:**

Name | Description
---- | -----------
[`labeling`](#data-labeling) | Coordinate-level atlas labeling.
[`loading`](#data-loading) | Lazy loading of atlas NIfTI + label CSV files from the HF dataset.
[`registry`](#data-registry) | Static registry of atlases hosted at ``nltools/niftis/atlases``.
[`reporting`](#data-reporting) | Cluster reports — peak/cluster geometry plus atlas labels.

**Classes:**

Name | Description
---- | -----------
[`Atlas`](#data-atlas) | A loaded atlas — image, labels, and metadata.
[`AtlasMetadata`](#data-atlasmetadata) | Static description of a registered atlas.
[`ClusterReport`](#data-clusterreport) | Result of `BrainData.cluster_report`.

**Methods:**

Name | Description
---- | -----------
[`cluster_report_data`](#data-cluster-report-data) | Compute cluster report DataFrames + thresholded BrainData.
[`label_coords`](#data-label-coords) | Look up anatomical labels for a set of MNI mm coordinates.
[`list_atlases`](#data-list-atlases) | Return the sorted list of registered atlas names.
[`load_atlas`](#data-load-atlas) | Lazy-load an atlas by registry name.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`ATLASES`](#data-atlases) | <code>[dict](#dict)[[str](#str), [AtlasMetadata](#nltools.data.atlases.registry.AtlasMetadata)]</code> | 
`AtlasKind` |  | 
`DEFAULT_ATLASES` | <code>[tuple](#tuple)[[str](#str), ...]</code> | 

##### Classes

###### `Atlas`

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
[`citation`](#data-citation) | <code>[str](#str)</code> | Short citation for the original atlas.



####### Attributes##

(data-citation)=
###### `citation`

```python
citation: str
```

######## `image`

```python
image: nb.Nifti1Image
```

######## `kind`

```python
kind: AtlasKind
```

######## `labels`

```python
labels: pl.DataFrame
```

######## `name`

```python
name: str
```

(data-atlasmetadata)=
###### `AtlasMetadata`

```python
AtlasMetadata(kind: AtlasKind, citation: str) -> None
```

Static description of a registered atlas.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`kind` | <code>[AtlasKind](#nltools.data.atlases.registry.AtlasKind)</code> | ``"deterministic"`` (3D integer-labeled) or ``"probabilistic"`` (4D, last axis indexes regions).
[`citation`](#data-citation) | <code>[str](#str)</code> | Short citation string for the original atlas.



####### Attributes##

###### `citation`

```python
citation: str
```

######## `kind`

```python
kind: AtlasKind
```

(data-clusterreport)=
###### `ClusterReport`

```python
ClusterReport(peaks: pl.DataFrame, clusters: pl.DataFrame, stat_img: BrainData) -> None
```

Result of `BrainData.cluster_report`.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`peaks` | <code>[DataFrame](#polars.DataFrame)</code> | Polars DataFrame, one row per peak (incl. sub-peaks). Columns ``cluster_id``, ``x``, ``y``, ``z`` (mm), ``peak_stat``, ``volume_mm3``, ``n_voxels``, then one Utf8 column per atlas. ``cluster_id`` shares the integer id space of ``clusters`` (they are joinable); sub-peaks carry their parent cluster's id.
[`clusters`](#data-clusters) | <code>[DataFrame](#polars.DataFrame)</code> | Polars DataFrame, one row per cluster. Columns ``cluster_id``, ``peak_x``, ``peak_y``, ``peak_z``, ``mean_stat``, ``volume_mm3``, ``n_voxels``, then one Utf8 column per atlas (mass-weighted top regions).
`stat_img` | <code>[BrainData](#nltools.data.BrainData)</code> | BrainData with the thresholded stat map (sub-cluster voxels and clusters smaller than ``cluster_threshold`` zeroed).

**Methods:**

Name | Description
---- | -----------
[`plot`](#data-plot) | Render an overview glass brain + one slice figure per cluster.
`to_csv` | Write ``peaks.csv`` and ``clusters.csv`` into ``output_dir``.



####### Attributes##

(data-clusters)=
###### `clusters`

```python
clusters: pl.DataFrame
```

######## `peaks`

```python
peaks: pl.DataFrame
```

######## `stat_img`

```python
stat_img: BrainData
```



####### Functions##

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

######## `to_csv`

```python
to_csv(output_dir: str | Path) -> None
```

Write ``peaks.csv`` and ``clusters.csv`` into ``output_dir``.



##### Methods

(data-cluster-report-data)=
###### `cluster_report_data`

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

(data-label-coords)=
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
`atlas` | <code>[str](#str) \| [Sequence](#collections.abc.Sequence)[[str](#str)]</code> | Atlas name or list of names from `list_atlases`. One column is added to the output per atlas. | <code>'harvard_oxford'</code>
`prob_threshold` | <code>[float](#float)</code> | For probabilistic atlases only — drop regions with probability (in percent units) below this threshold. | <code>5.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#polars.DataFrame)</code> | Polars DataFrame with columns ``x``, ``y``, ``z`` plus one
<code>[DataFrame](#polars.DataFrame)</code> | column per atlas. All atlas columns are ``Utf8``.

(data-list-atlases)=
###### `list_atlases`

```python
list_atlases() -> list[str]
```

Return the sorted list of registered atlas names.

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[str](#str)]</code> | Sorted list of atlas names usable with
<code>[list](#list)[[str](#str)]</code> | `load_atlas`.

(data-load-atlas)=
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
`name` | <code>[str](#str)</code> | Atlas key from `list_atlases`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Atlas](#nltools.data.atlases.loading.Atlas)</code> | An `Atlas` with image, labels, and metadata loaded.



##### Modules

(data-labeling)=
###### `labeling`

Coordinate-level atlas labeling.

Adapted from [atlasreader](https://github.com/miykael/atlasreader)
(BSD-3-Clause). Cite:

> Notter et al. (2019). AtlasReader. JOSS 4(34), 1257.

**Methods:**

Name | Description
---- | -----------
[`label_coords`](#data-label-coords) | Look up anatomical labels for a set of MNI mm coordinates.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`CoordsLike`](#data-coordslike) |  | 



####### Attributes##

(data-coordslike)=
###### `CoordsLike`

```python
CoordsLike = ArrayLike | Sequence[Sequence[float]]
```



####### Classes

####### Functions##

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
`atlas` | <code>[str](#str) \| [Sequence](#collections.abc.Sequence)[[str](#str)]</code> | Atlas name or list of names from `list_atlases`. One column is added to the output per atlas. | <code>'harvard_oxford'</code>
`prob_threshold` | <code>[float](#float)</code> | For probabilistic atlases only — drop regions with probability (in percent units) below this threshold. | <code>5.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#polars.DataFrame)</code> | Polars DataFrame with columns ``x``, ``y``, ``z`` plus one
<code>[DataFrame](#polars.DataFrame)</code> | column per atlas. All atlas columns are ``Utf8``.

(data-loading)=
###### `loading`

Lazy loading of atlas NIfTI + label CSV files from the HF dataset.

**Classes:**

Name | Description
---- | -----------
[`Atlas`](#data-atlas) | A loaded atlas — image, labels, and metadata.

**Methods:**

Name | Description
---- | -----------
[`load_atlas`](#data-load-atlas) | Lazy-load an atlas by registry name.



####### Attributes

####### Classes##

###### `Atlas`

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
[`citation`](#data-citation) | <code>[str](#str)</code> | Short citation for the original atlas.



######### Attributes####

###### `citation`

```python
citation: str
```

########## `image`

```python
image: nb.Nifti1Image
```

########## `kind`

```python
kind: AtlasKind
```

########## `labels`

```python
labels: pl.DataFrame
```

########## `name`

```python
name: str
```



####### Functions##

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
`name` | <code>[str](#str)</code> | Atlas key from `list_atlases`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Atlas](#nltools.data.atlases.loading.Atlas)</code> | An `Atlas` with image, labels, and metadata loaded.

(data-registry)=
###### `registry`

Static registry of atlases hosted at ``nltools/niftis/atlases``.

Each entry describes an atlas's kind (deterministic vs probabilistic) and
the citation users should cite when they use it. The actual NIfTI + label
files are fetched lazily by `load_atlas` via
`fetch_resource`.

Atlases were sourced from atlasreader (BSD-3-Clause) and are subject to
their original upstream licenses — see ``LICENSES.md`` in the HF dataset.

**Classes:**

Name | Description
---- | -----------
[`AtlasMetadata`](#data-atlasmetadata) | Static description of a registered atlas.

**Methods:**

Name | Description
---- | -----------
[`list_atlases`](#data-list-atlases) | Return the sorted list of registered atlas names.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`ATLASES`](#data-atlases) | <code>[dict](#dict)[[str](#str), [AtlasMetadata](#nltools.data.atlases.registry.AtlasMetadata)]</code> | 
`AtlasKind` |  | 
`DEFAULT_ATLASES` | <code>[tuple](#tuple)[[str](#str), ...]</code> | 



####### Attributes##

###### `ATLASES`

```python
ATLASES: dict[str, AtlasMetadata] = {'aal': AtlasMetadata(kind='deterministic', citation='Tzourio-Mazoyer et al. 2002, NeuroImage'), 'aicha': AtlasMetadata(kind='deterministic', citation='Joliot et al. 2015, J Neurosci Methods'), 'desikan_killiany': AtlasMetadata(kind='deterministic', citation='Desikan et al. 2006, NeuroImage (FreeSurfer license)'), 'destrieux': AtlasMetadata(kind='deterministic', citation='Destrieux et al. 2010, NeuroImage (FreeSurfer license)'), 'harvard_oxford': AtlasMetadata(kind='probabilistic', citation='Desikan et al. 2006, NeuroImage / FSL Harvard-Oxford'), 'juelich': AtlasMetadata(kind='probabilistic', citation='Eickhoff et al. 2005, NeuroImage'), 'marsatlas': AtlasMetadata(kind='deterministic', citation='Auzias et al. 2016, Hum Brain Mapp'), 'neuromorphometrics': AtlasMetadata(kind='deterministic', citation='MICCAI 2012 Multi-Atlas Labeling Challenge'), 'schaefer_200': AtlasMetadata(kind='deterministic', citation='Schaefer et al. 2018, Cereb Cortex (200-parcel, 7-network)'), 'talairach_ba': AtlasMetadata(kind='deterministic', citation='Talairach & Tournoux 1988 (Brodmann areas)'), 'talairach_gyrus': AtlasMetadata(kind='deterministic', citation='Talairach & Tournoux 1988 (gyri)')}
```

######## `AtlasKind`

```python
AtlasKind = Literal['deterministic', 'probabilistic']
```

######## `DEFAULT_ATLASES`

```python
DEFAULT_ATLASES: tuple[str, ...] = ('harvard_oxford', 'aal', 'schaefer_200')
```



####### Classes##

###### `AtlasMetadata`

```python
AtlasMetadata(kind: AtlasKind, citation: str) -> None
```

Static description of a registered atlas.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`kind` | <code>[AtlasKind](#nltools.data.atlases.registry.AtlasKind)</code> | ``"deterministic"`` (3D integer-labeled) or ``"probabilistic"`` (4D, last axis indexes regions).
[`citation`](#data-citation) | <code>[str](#str)</code> | Short citation string for the original atlas.



######### Attributes####

###### `citation`

```python
citation: str
```

########## `kind`

```python
kind: AtlasKind
```



####### Functions##

###### `list_atlases`

```python
list_atlases() -> list[str]
```

Return the sorted list of registered atlas names.

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[str](#str)]</code> | Sorted list of atlas names usable with
<code>[list](#list)[[str](#str)]</code> | `load_atlas`.

(data-reporting)=
###### `reporting`

Cluster reports — peak/cluster geometry plus atlas labels.

The peak/sub-peak geometry comes from `get_clusters_table`;
the cluster masks and mass-weighted labels are computed locally so we can
attribute every voxel of every cluster to one or more atlases.

**Classes:**

Name | Description
---- | -----------
[`ClusterReport`](#data-clusterreport) | Result of `BrainData.cluster_report`.

**Methods:**

Name | Description
---- | -----------
[`cluster_report_data`](#data-cluster-report-data) | Compute cluster report DataFrames + thresholded BrainData.



####### Attributes

####### Classes##

###### `ClusterReport`

```python
ClusterReport(peaks: pl.DataFrame, clusters: pl.DataFrame, stat_img: BrainData) -> None
```

Result of `BrainData.cluster_report`.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`peaks` | <code>[DataFrame](#polars.DataFrame)</code> | Polars DataFrame, one row per peak (incl. sub-peaks). Columns ``cluster_id``, ``x``, ``y``, ``z`` (mm), ``peak_stat``, ``volume_mm3``, ``n_voxels``, then one Utf8 column per atlas. ``cluster_id`` shares the integer id space of ``clusters`` (they are joinable); sub-peaks carry their parent cluster's id.
[`clusters`](#data-clusters) | <code>[DataFrame](#polars.DataFrame)</code> | Polars DataFrame, one row per cluster. Columns ``cluster_id``, ``peak_x``, ``peak_y``, ``peak_z``, ``mean_stat``, ``volume_mm3``, ``n_voxels``, then one Utf8 column per atlas (mass-weighted top regions).
`stat_img` | <code>[BrainData](#nltools.data.BrainData)</code> | BrainData with the thresholded stat map (sub-cluster voxels and clusters smaller than ``cluster_threshold`` zeroed).

**Methods:**

Name | Description
---- | -----------
[`plot`](#data-plot) | Render an overview glass brain + one slice figure per cluster.
`to_csv` | Write ``peaks.csv`` and ``clusters.csv`` into ``output_dir``.



######### Attributes####

###### `clusters`

```python
clusters: pl.DataFrame
```

########## `peaks`

```python
peaks: pl.DataFrame
```

########## `stat_img`

```python
stat_img: BrainData
```



######### Functions####

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

########## `to_csv`

```python
to_csv(output_dir: str | Path) -> None
```

Write ``peaks.csv`` and ``clusters.csv`` into ``output_dir``.



####### Functions##

###### `cluster_report_data`

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

#### `braindata`

Represent brain image data with the BrainData class.

# NeuroLearn Brain Data

Classes to represent brain image data.

**Modules:**

Name | Description
---- | -----------
[`analysis`](#data-analysis) | BrainData analysis functions.
[`bootstrap`](#data-bootstrap) | Bootstrap functions extracted from BrainData methods.
[`cache`](#data-cache) | Disk-based caching infrastructure for expensive computations.
[`io`](#data-io) | BrainData I/O and loading functions.
[`modeling`](#data-modeling) | BrainData modeling functions.
[`neighborhoods`](#data-neighborhoods) | Spatial neighborhood computation for neuroimaging analyses.
[`plotting`](#data-plotting) | BrainData plotting functions.
[`prediction`](#data-prediction) | BrainData prediction — timeseries (encoding) and MVPA (decoding).
[`utils`](#data-utils) | Shared helpers for BrainData submodules.
[`validation`](#data-validation) | Validation utilities for BrainData class.
[`viewer`](#data-viewer) | ipyniivue (niivue) interactive viewer for BrainData.

**Classes:**

Name | Description
---- | -----------
[`BrainData`](#data-braindata) | Represent neuroimaging data as vectors instead of three-dimensional matrices.



##### Classes

###### `BrainData`

```python
BrainData(data = None, *, Y = None, X = None, mask = None, masker = None, h5_compression = 'gzip', verbose = False, resample = True, interpolation = 'auto')
```

Represent neuroimaging data as vectors instead of three-dimensional matrices.

This representation makes it easier to perform data manipulation and analyses.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | Neuroimaging data. Can be: - None (empty BrainData) - BrainData object - List of BrainData objects or file paths - File path (str/Path) to .nii/.nii.gz/.h5/.hdf5 - nibabel Nifti1Image object - URL to download data from - numpy array (1D ``(n_voxels,)`` for a single image or 2D   ``(n_images, n_voxels)`` for a stack). The ``mask`` argument   is required and must define the same number of in-mask voxels. | <code>None</code>
`mask` |  | Brain mask. Can be None (uses MNI template), a nibabel Nifti1Image, a file path (str/Path) to a mask file, or a template name string like ``'2mm-MNI152-2009c'`` (version: 'fsl' for default/, 'a' for nilearn/, 'c' for fmriprep/). | <code>None</code>
`masker` |  | nilearn masker object (e.g. ROI or searchlight extractor). Default will load data as voxels. | <code>None</code>
`resample` | <code>bool, default=True</code> | Whether to automatically resample data to mask space. If True, data is resampled to match mask spatial characteristics. If False, data must already be in mask space. Default True preserves backward compatibility with v0.5.1. | <code>True</code>
`interpolation` | <code>str, default='auto'</code> | Interpolation method for resampling. Options: 'auto' (detect based on data type; uses 'nearest' for discrete data like atlases/masks and 'continuous' for stat maps), 'nearest' (nearest-neighbor, preserves discrete values), 'linear' (linear interpolation), 'continuous' (higher-order spline, use for stat maps). | <code>'auto'</code>

**Methods:**

Name | Description
---- | -----------
[`align`](#data-align) | Align BrainData instance to target object using functional alignment.
[`append`](#data-append) | Append data to BrainData instance.
[`apply_mask`](#data-apply-mask) | Mask BrainData instance using nilearn functionality.
[`astype`](#data-astype) | Cast BrainData.data as type.
[`bootstrap`](#data-bootstrap) | Bootstrap statistics using efficient online algorithms.
[`cluster_report`](#data-cluster-report) | Generate a cluster report with anatomical labels.
[`compute_contrasts`](#data-compute-contrasts) | Compute contrasts from fitted GLM results.
[`copy`](#data-copy) | Create a deep copy of a BrainData instance.
[`create_empty`](#data-create-empty) | Create a copy of BrainData with empty data array.
[`decompose`](#data-decompose) | Decompose BrainData object.
[`detrend`](#data-detrend) | Remove linear trend from each voxel.
[`distance`](#data-distance) | Calculate distance between images within a BrainData() instance.
[`extract_roi`](#data-extract-roi) | Extract activity from mask or ROI atlas using NiftiLabelsMasker.
[`filter`](#data-filter) | Apply butterworth filter to data. Wraps nilearn.signal.clean.
[`find_spikes`](#data-find-spikes) | Identify spikes from Time Series Data.
[`fit`](#data-fit) | Fit a model to brain imaging data.
[`iplot`](#data-iplot) | Interactive WebGL brain viewer powered by niivue (`ipyniivue`).
[`mean`](#data-mean) | Get mean of each voxel or image.
[`median`](#data-median) | Get median of each voxel or image.
[`multivariate_similarity`](#data-multivariate-similarity) | Predict a BrainData spatial distribution from a linear combination.
[`plot`](#data-plot) | Plot BrainData instance using nilearn visualization or matplotlib.
[`plot_flatmap`](#data-plot-flatmap) | Plot brain data on cortical flatmap.
[`plot_surf`](#data-plot-surf) | Render this BrainData on fsaverage surfaces as a tight 2×2 montage.
[`predict`](#data-predict) | Predict voxel timeseries (encoding) or decode labels (MVPA).
[`r_to_z`](#data-r-to-z) | Apply Fisher's r-to-z transformation to each data element.
[`regions`](#data-regions) | Extract brain connected regions into separate regions.
[`report`](#data-report) | Generate a nilearn HTML report for a fitted GLM.
[`resample_to`](#data-resample-to) | Resample BrainData to match target image or resolution.
[`scale`](#data-scale) | Scale data via mean scaling.
[`similarity`](#data-similarity) | Calculate similarity to a single BrainData or nibabel image.
[`smooth`](#data-smooth) | Apply spatial smoothing using nilearn smooth_img().
[`standardize`](#data-standardize) | Standardize BrainData() instance.
[`std`](#data-std) | Get standard deviation of each voxel or image.
[`sum`](#data-sum) | Get sum of each voxel or image.
[`temporal_resample`](#data-temporal-resample) | Resample BrainData timeseries to a new target frequency or number of samples.
[`threshold`](#data-threshold) | Threshold BrainData instance with optional cluster filtering.
[`to_nifti`](#data-to-nifti) | Convert BrainData Instance into Nifti Object.
[`transform_pairwise`](#data-transform-pairwise) | Transform data into pairwise comparisons.
[`ttest`](#data-ttest) | One-sample voxelwise t-test across images (axis 0).
[`ttest2`](#data-ttest2) | Two-sample voxelwise t-test between two BrainData stacks.
[`upload_neurovault`](#data-upload-neurovault) | Upload BrainData images and metadata to NeuroVault.
[`write`](#data-write) | Write out BrainData object to Nifti or HDF5 File.
[`z_to_r`](#data-z-to-r) | Convert z score back into r value for each element of data object.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`X`](#data-x) |  | Design matrix / per-image covariates as a polars DataFrame.
[`Y`](#data-y) |  | Per-image targets as a polars DataFrame.
[`data`](#data-data) |  | 
`design_matrix` |  | 
`dtype` |  | Get data type of BrainData.data.
`is_empty` | <code>[bool](#bool)</code> | Check if BrainData.data is empty.
`masker` |  | 
`shape` |  | Get images by voxels shape.
`size` |  | Total number of elements in BrainData.data (numpy convention).
`verbose` |  | 



####### Attributes##

(data-x)=
###### `X`

```python
X
```

Design matrix / per-image covariates as a polars DataFrame.

######## `Y`

```python
Y
```

Per-image targets as a polars DataFrame.

######## `data`

```python
data = np.array([])
```

######## `design_matrix`

```python
design_matrix = None
```

######## `dtype`

```python
dtype
```

Get data type of BrainData.data.

######## `is_empty`

```python
is_empty: bool
```

Check if BrainData.data is empty.

######## `masker`

```python
masker = masker
```

######## `shape`

```python
shape
```

Get images by voxels shape.

######## `size`

```python
size
```

Total number of elements in BrainData.data (numpy convention).

######## `verbose`

```python
verbose = verbose
```



####### Functions##

###### `align`

```python
align(target, method = 'procrustes', axis = 0, *, spatial_scale: str = 'whole_brain', roi_mask: str = None, radius_mm: float = 10.0)
```

Align BrainData instance to target object using functional alignment.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` |  | (BrainData) object to align to. | *required*
`method` |  | (str) alignment method to use ['probabilistic_srm','deterministic_srm','procrustes'] | <code>'procrustes'</code>
`axis` |  | (int) axis to align on | <code>0</code>
`spatial_scale` | <code>[str](#str)</code> | ``'whole_brain'`` (default), ``'roi'``, or ``'searchlight'``. ``'roi'`` is supported (per-parcel transforms + reassembly, requires `roi_mask`). ``'searchlight'`` is not yet implemented (overlapping spheres have no canonical per-voxel transform). | <code>'whole_brain'</code>
`roi_mask` |  | Atlas image used when `spatial_scale='roi'`. | <code>None</code>
`radius_mm` | <code>[float](#float)</code> | Reserved for ``spatial_scale='searchlight'``. | <code>10.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (dict) a dictionary containing transformed object, transformation matrix, and the shared response matrix

**Examples:**

```pycon
>>> out = data.align(target, method='procrustes')
>>> out = data.align(target, method='probabilistic_srm')
```

######## `append`

```python
append(data, ignore_attrs = False, **kwargs)
```

Append data to BrainData instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | BrainData instance to append. | *required*
`ignore_attrs` |  | (bool) If True, skip concatenation of X and Y     attributes. Useful when appending images where .X or .Y     have different column counts. Default False. | <code>False</code>
`kwargs` |  | Optional arguments passed to pandas concat for X/Y. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | New appended BrainData instance.

######## `apply_mask`

```python
apply_mask(mask, resample_mask_to_brain = False)
```

Mask BrainData instance using nilearn functionality.

Note target data will be resampled into the same space as the mask. If you would like the mask
resampled into the BrainData space, then set resample_mask_to_brain=True.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask` |  | (BrainData or nifti object) mask to apply to BrainData object. | *required*
`resample_mask_to_brain` |  | (bool) Will resample mask to brain space before applying mask (default=False). | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`masked` |  | (BrainData) masked BrainData object

######## `astype`

```python
astype(dtype)
```

Cast BrainData.data as type.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dtype` |  | datatype to convert | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with new datatype

######## `bootstrap`

```python
bootstrap(stat, *, n_samples = 5000, save_boots = False, percentiles = (2.5, 97.5), X_test = None, backend = None, max_gpu_memory_gb = 4.0, n_jobs = -1, random_state = None)
```

Bootstrap statistics using efficient online algorithms.

Uses memory-efficient bootstrap infrastructure with CPU parallelization or GPU acceleration.
Supports simple aggregation statistics and fitted model statistics (Ridge).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stat` |  | (str) Statistic to bootstrap. Options: Simple stats ('mean', 'median', 'std', 'sum', 'min', 'max') or Model stats ('weights' requires fitted Ridge model, 'predict' requires fitted Ridge model + X_test). | *required*
`n_samples` |  | (int) Number of bootstrap iterations. Default: 5000 | <code>5000</code>
`save_boots` |  | (bool) If True, store all bootstrap samples. Default: False | <code>False</code>
`percentiles` |  | (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5) | <code>(2.5, 97.5)</code>
`X_test` |  | (np.ndarray, optional) Test features for 'predict' bootstrap. | <code>None</code>
`backend` |  | (str, optional) Backend for Ridge bootstrap: None (CPU), 'torch' (GPU if available), or 'auto' (auto-select). Ignored for simple stats. | <code>None</code>
`max_gpu_memory_gb` |  | (float) Maximum GPU memory to use when backend is 'torch' or 'auto'. Default: 4.0 | <code>4.0</code>
`n_jobs` |  | (int) Number of CPU cores for parallelization. -1 means all CPUs. | <code>-1</code>
`random_state` |  | (int, optional) Random seed for reproducibility | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or dict: - For simple stats: Returns BrainData with bootstrap mean - For model stats: Returns dict with keys: 'mean', 'std', 'Z', 'p',   'ci_lower', 'ci_upper' (all BrainData objects) - If ``save_boots=True``: Returns dict with 'samples' key containing all samples

**Examples:**

```pycon
>>> boot = brain.bootstrap(stat='mean', n_samples=1000)
>>> brain.fit(X=dm, model='ridge', alpha=1.0)
>>> boot = brain.bootstrap(stat='weights', n_samples=1000)
```

######## `cluster_report`

```python
cluster_report(*, stat_threshold: float | None = 3.0, cluster_threshold: int = 10, two_sided: bool = True, min_distance: float = 8.0, atlas: str | Sequence[str] | None = None, prob_threshold: float = 5.0) -> ClusterReport
```

Generate a cluster report with anatomical labels.

Identifies surviving clusters in the stat map (after voxel + extent
thresholding), reports peak coordinates and sub-peaks, and labels
each peak/cluster against one or more atlases.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stat_threshold` | <code>[float](#float) \| None</code> | Voxel-level threshold (e.g. z- or t-cutoff). ``None`` treats ``self`` as already thresholded. | <code>3.0</code>
`cluster_threshold` | <code>[int](#int)</code> | Minimum cluster size in voxels. | <code>10</code>
`two_sided` | <code>[bool](#bool)</code> | Report negative clusters separately. | <code>True</code>
`min_distance` | <code>[float](#float)</code> | Minimum mm between sub-peaks within a cluster. | <code>8.0</code>
`atlas` | <code>[str](#str) \| [Sequence](#collections.abc.Sequence)[[str](#str)] \| None</code> | Atlas name or list of names (see `list_atlases`). Defaults to ``("harvard_oxford", "aal", "schaefer_200")``. | <code>None</code>
`prob_threshold` | <code>[float](#float)</code> | Drop probabilistic-atlas regions below this %. | <code>5.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[ClusterReport](#nltools.data.atlases.ClusterReport)</code> | `ClusterReport` with ``peaks``,
<code>[ClusterReport](#nltools.data.atlases.ClusterReport)</code> | ``clusters`` (polars DataFrames), and ``stat_img`` (BrainData).

######## `compute_contrasts`

```python
compute_contrasts(contrasts, statistic = 't')
```

Compute contrasts from fitted GLM results.

This method computes contrasts as linear combinations of the GLM beta coefficients.
Must be called after .fit(model='glm', X=design_matrix) has been run.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`contrasts` |  | Can be:<br>- str: A string specifying the contrast using column names   e.g., "conditionA - conditionB" or "2*conditionA - conditionB - conditionC" - dict: Dictionary with contrast names as keys and contrast strings/vectors as values   e.g., {"main_effect": "conditionA - conditionB", "interaction": [1, -1, -1, 1]} - array: Numeric contrast vector matching the number of regressors   e.g., [1, -1, 0, 0] for a 4-regressor model | *required*
`statistic` | <code>[str](#str)</code> | Which statistic to return per contrast. One of `"t"` (default, t-statistic map), `"z"` (z-score), `"p"` (p-value), `"beta"` / `"effect_size"` (effect-size β map — use this when feeding a second-level group analysis), or `"all"` (a bundle dict `{"beta", "t", "z", "p", "se"}` of maps for one contrast). Default: `"t"`. | <code>'t'</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or dict: A single contrast with a scalar `statistic` returns a `BrainData` map; with `statistic="all"` it returns a flat dict keyed by `"beta"`/`"t"`/`"z"`/`"p"`/`"se"`. A dict of contrasts returns a dict keyed by contrast name (nested under the five keys when `statistic="all"`).

**Examples:**

```pycon
>>> brain.fit(model='glm', X=design_matrix)
>>> contrast1 = brain.compute_contrasts([0, 1, -1])
>>> contrast2 = brain.compute_contrasts("conditionA - conditionB")
>>> results = brain.compute_contrasts({
...     "A_vs_B": "conditionA - conditionB",
...     "avg_effect": [0, 0.5, 0.5],
... })
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- String contrasts support coefficients: "2*A - B" or "0.5*A + 0.5*B"
- Column names must match design matrix columns exactly (case-sensitive)
- Contrast weights should sum to zero for proper inference in most cases

</details>

######## `copy`

```python
copy()
```

Create a deep copy of a BrainData instance.

All attributes including data, fitted models, and results are deep copied.
Use this when you need a complete independent copy.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Deep copied instance

######## `create_empty`

```python
create_empty()
```

Create a copy of BrainData with empty data array.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | A copy of this object with an empty data array.

######## `decompose`

```python
decompose(*, method = 'pca', axis = 'voxels', n_components = None, **kwargs)
```

Decompose BrainData object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` |  | (str) Algorithm to perform decomposition         types=['pca','ica','nnmf','fa','dictionary','kernelpca'] | <code>'pca'</code>
`axis` |  | dimension to decompose ['voxels','images'] | <code>'voxels'</code>
`n_components` |  | (int) number of components. If None then retain         as many as possible. | <code>None</code>
`**kwargs` |  | forwarded to the underlying sklearn decomposition estimator. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`output` |  | a dictionary of decomposition parameters

######## `detrend`

```python
detrend(method = 'linear')
```

Remove linear trend from each voxel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` |  | ('linear','constant', optional) type of detrending | <code>'linear'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (BrainData) detrended BrainData instance

######## `distance`

```python
distance(metric = 'euclidean', *, spatial_scale: str = 'whole_brain', roi_mask: str = None, radius_mm: float = 10.0, **kwargs: float)
```

Calculate distance between images within a BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` |  | (str) type of distance metric (can use any scipy.spatial.distance     metric supported by cdist) | <code>'euclidean'</code>
`**kwargs` |  | Additional metric options forwarded to ``scipy.spatial.distance.cdist`` (e.g. ``p`` for minkowski). | <code>{}</code>
`spatial_scale` | <code>[str](#str)</code> | One of ``'whole_brain'`` (default), ``'roi'``, or ``'searchlight'``. ``'whole_brain'`` returns a single pairwise distance ``Adjacency`` between images. ``'roi'`` requires ``roi_mask`` and returns a stacked ``Adjacency`` with one RDM per parcel and ``spatial_scale`` provenance attached for back-projection via ``Adjacency.to_brain()``. ``'searchlight'`` requires ``radius_mm`` (and is not yet implemented in this slice). | <code>'whole_brain'</code>
`roi_mask` |  | Atlas image (BrainData / Nifti1Image / path) for ``spatial_scale='roi'``. | <code>None</code>
`radius_mm` | <code>[float](#float)</code> | Searchlight radius in mm. Default 10.0. | <code>10.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | Single pairwise distance matrix for ``'whole_brain'``; stacked Adjacency (one matrix per parcel/searchlight) with ``spatial_scale`` set for ``'roi'`` / ``'searchlight'``.

######## `extract_roi`

```python
extract_roi(mask, metric = 'mean', n_components = None)
```

Extract activity from mask or ROI atlas using NiftiLabelsMasker.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask` |  | BrainData, nibabel image, or file path. Can be:<br>  - Binary mask (extracts from single ROI)   - Labeled atlas (extracts from multiple ROIs) | *required*
`metric` |  | Extraction method ('mean', 'median', 'pca'). Default: 'mean' | <code>'mean'</code>
`n_components` |  | If metric='pca', number of components to return | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | For binary mask: scalar or 1D array.
 | For labeled atlas: 1D or 2D array, or PCA components.

**Examples:**

```pycon
>>> roi_values = brain.extract_roi(binary_mask)
>>> atlas_values = brain.extract_roi(atlas_mask)
>>> components = brain.extract_roi(mask, metric='pca', n_components=5)
```

######## `filter`

```python
filter(*, sampling_freq = None, high_pass = None, low_pass = None, **kwargs)
```

Apply butterworth filter to data. Wraps nilearn.signal.clean.

<details class="note" open markdown="1">
<summary>Note</summary>

Unlike nilearn's default, does not detrend or standardize. Pass
detrend=True or standardize=True via kwargs to enable.

</details>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sampling_freq` |  | Sampling freq in hertz (i.e. 1 / TR) | <code>None</code>
`high_pass` |  | High pass cutoff frequency | <code>None</code>
`low_pass` |  | Low pass cutoff frequency | <code>None</code>
`**kwargs` |  | Additional arguments passed to nilearn.signal.clean | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Filtered BrainData instance

######## `find_spikes`

```python
find_spikes(global_spike_cutoff = 3, diff_spike_cutoff = 3, *, TR: float | None = None, sampling_freq: float | None = None)
```

Identify spikes from Time Series Data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`global_spike_cutoff` | <code>[int](#int) or None</code> | cutoff to identify spikes in global signal in standard deviations, or None to skip. | <code>3</code>
`diff_spike_cutoff` | <code>[int](#int) or None</code> | cutoff to identify spikes in average frame difference in standard deviations, or None to skip. | <code>3</code>
`TR` | <code>[float](#float) \| None</code> | Repetition time in seconds. Sets the returned DesignMatrix's sampling_freq for downstream `.append(...)` / `.convolve()`. Pass exactly one of `TR` or `sampling_freq`. | <code>None</code>
`sampling_freq` | <code>[float](#float) \| None</code> | Sampling frequency in Hz (= 1/TR). See `TR`. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | DesignMatrix with one indicator column per detected spike, with
 | all spike columns pre-marked as confounds.

######## `fit`

```python
fit(model = 'glm', *, X = None, cv = None, local_alpha = True, fit_intercept = False, inplace = True, scale = 'auto', standardize = 'auto', progress_bar = None, design_clean = True, design_clean_thresh = 0.95, design_clean_exclude_confounds = False, design_clean_fill_na = 0, **kwargs)
```

Fit a model to brain imaging data.

Creates and fits a model from string specification. The brain data
(self.data) is always used as the target variable. Model and results
are stored for later use with predict().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model` | <code>[str](#str)</code> | Model type: 'ridge', 'glm', or future model names | <code>'glm'</code>
`X` | <code>[array](#array) - [like](#like) or [DataFrame](#DataFrame)</code> | Design matrix or feature matrix | <code>None</code>
`cv` | <code>int or sklearn CV splitter</code> | Cross-validation specification (Ridge only). int → ``KFold(cv)``; pass a splitter object (e.g. ``KFold(5, shuffle=True)``, ``GroupKFold(8)``) for non-contiguous folds. Generators (``splitter.split(X)``) are rejected. | <code>None</code>
`local_alpha` | <code>bool, default=True</code> | Ridge only. If True, select α independently per voxel via ``solve_ridge_cv``. If False, pick a single α shared across all voxels. | <code>True</code>
`fit_intercept` | <code>bool, default=False</code> | Ridge only. Forwarded to the Ridge model — center X and y on the training fold mean per fold and recover the intercept after. | <code>False</code>
`inplace` | <code>bool, default=True</code> | If True, mutate self and return self. If False, return a Fit dataclass with the results. ``self.data`` and the result attributes (``ridge_*`` / ``glm_*`` / ``cv_results_``) are left unchanged, but ``self.model_`` and ``self.X_`` (plus ``self.design_matrix`` for GLM) ARE updated on self so ``predict()`` / ``compute_contrasts()`` still work. | <code>True</code>
`scale` | <code>bool or 'auto', default='auto'</code> | Apply percent-signal-change scaling before fitting via nilearn's per-voxel ``mean_scaling``. ``'auto'`` → False for both models (PSC is opt-in). Redundant with ``standardize='zscore'`` (warns). Applied before ``standardize``. | <code>'auto'</code>
`standardize` | <code>str or None or 'auto', default='auto'</code> | Standardize each voxel across observations after scaling. ``'center'``, ``'zscore'``, or ``None``. ``'auto'`` → ``'zscore'`` for ridge, ``None`` for glm. | <code>'auto'</code>
`progress_bar` | <code>[bool](#bool)</code> | Display progress bar during fitting. | <code>None</code>
`design_clean` | <code>bool, default=True</code> | GLM only. Run ``DesignMatrix.clean()`` on ``X`` before fitting to drop highly correlated regressors. Coerces ``X`` to ``DesignMatrix`` if needed. Ignored when ``model='ridge'``. | <code>True</code>
`design_clean_thresh` | <code>float, default=0.95</code> | GLM only. Correlation threshold passed to ``DesignMatrix.clean()`` (drops if ``abs(r) >= thresh``). Ignored when ``model='ridge'``. | <code>0.95</code>
`design_clean_exclude_confounds` | <code>bool, default=False</code> | GLM only. If True, ``DesignMatrix.clean()`` skips confound columns when checking correlations. Ignored when ``model='ridge'``. | <code>False</code>
`design_clean_fill_na` | <code>int, float, or None, default=0</code> | GLM only. Fill value for NaNs before correlation check in ``DesignMatrix.clean()``. Ignored when ``model='ridge'``. | <code>0</code>
`**kwargs` | <code>[dict](#dict)</code> | Additional arguments passed to model constructor | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or Fit: If ``inplace=True``, returns self (fitted BrainData). If ``inplace=False``, returns Fit dataclass with results.

<details class="notes" open markdown="1">
<summary>Notes</summary>

After ``model="glm"``, the following per-regressor BrainData
attributes are populated — one map per design-matrix column:

- ``glm_betas``: effect-size (β) maps.
- ``glm_t``: marginal t-statistic for each regressor.
- ``glm_p``: marginal p-value.
- ``glm_se``: standard error of β.
- ``glm_r2``: voxel-wise R².

``glm_t[i]`` is a valid t-map for the trivial one-hot contrast on
regressor ``i`` only. For contrasts across regressors
(``"A - B"``, ``[1, -1, 0, ...]``) use `compute_contrasts` —
you cannot correctly combine these per-regressor maps by hand
because t-statistic arithmetic requires the off-diagonal elements
of the parameter covariance matrix, which are not stored. Pass
``statistic="all"`` to get ``β``/``t``/``z``/``p``/``se`` for
one contrast in a single call.

</details>

**Examples:**

```pycon
>>> brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
>>> fit = brain_data.fit(model='ridge', alpha=1.0, X=features, inplace=False)
```

######## `iplot`

```python
iplot(*, view: str = 'ortho', threshold: float | None = None, lower: float | None = None, upper: float | None = None, cmap: str = 'warm', bg_img: str | bool | None = None, atlas: str | Atlas | None = None, opacity: float = 1.0, outline: float = 0.0, colorbar: bool = True, controls: bool = True, **kwargs: bool)
```

Interactive WebGL brain viewer powered by niivue (`ipyniivue`).

Renders inline in a live kernel (Jupyter, marimo) with live windowing
(right-drag to set the threshold/contrast), slice scrolling, native 4D
frame scrubbing, true 3D rendering, a stat-map colorbar, and optional
nltools-atlas overlays. Static-built docs are not supported; use
`plot` there.

By default (``controls=True``) the return value is an
`ipywidgets.VBox` stacking a threshold slider above the viewer; access
the underlying `NiiVue` via its ``.viewer`` attribute and the slider
via ``.threshold_slider``. Pass ``controls=False`` to get the bare
`NiiVue` widget instead.

Thresholding is a divergent magnitude window: ``cal_min`` is the
display floor (sub-floor voxels render transparent), ``cal_max`` the
saturation point, with the positive limb using ``cmap`` and the
negative limb its mirrored partner. Precedence: ``lower``/``upper``
win; otherwise ``threshold`` sets the floor (ceiling auto);
otherwise the window is fully auto.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`view` | <code>[str](#str)</code> | ``"ortho"`` (default), ``"axial"``, ``"coronal"``, ``"sagittal"``, or ``"render"`` (3D volume render). ``"surface"`` is no longer supported — use ``"render"`` or `plot_flatmap` / `plot_surf`. | <code>'ortho'</code>
`threshold` | <code>[float](#float) \| None</code> | Convenience symmetric magnitude floor (→ ``cal_min``). | <code>None</code>
`lower` | <code>[float](#float) \| None</code> | Window floor (→ ``cal_min``). Overrides ``threshold``. | <code>None</code>
`upper` | <code>[float](#float) \| None</code> | Window ceiling (→ ``cal_max``). Overrides ``threshold``. | <code>None</code>
`cmap` | <code>[str](#str)</code> | niivue colormap for the positive limb (default ``"warm"``). Common matplotlib names are auto-mapped with a warning. | <code>'warm'</code>
`bg_img` | <code>[str](#str) \| [bool](#bool) \| None</code> | ``None``/``True`` auto-loads the matching MNI template when the data is in standard space (else none); ``False`` disables the background; a path string uses that image. | <code>None</code>
`atlas` | <code>[str](#str) \| [Atlas](#nltools.data.atlases.Atlas) \| None</code> | Atlas overlay — a registry name (e.g. ``"aal"``), a loaded `Atlas`, or ``None``. Deterministic atlases only; probabilistic atlases raise. | <code>None</code>
`opacity` | <code>[float](#float)</code> | Stat-map (and filled-atlas) opacity in ``0..1``. | <code>1.0</code>
`outline` | <code>[float](#float)</code> | ``> 0`` draws atlas region boundaries of that width (stat map stays visible); ``0`` draws filled regions. | <code>0.0</code>
`colorbar` | <code>[bool](#bool)</code> | Show the stat-map colorbar (default ``True``). An explicit ``is_colorbar`` kwarg overrides this. | <code>True</code>
`controls` | <code>[bool](#bool)</code> | Wrap the viewer in a `VBox` with an interactive threshold slider (default ``True``). ``False`` returns the bare `NiiVue`. Requires the ``ipywidgets`` optional dependency when ``True``. | <code>True</code>
`**kwargs` |  | Forwarded verbatim to ``ipyniivue.NiiVue(**kwargs)`` (e.g. ``height``, ConfigOptions like ``is_colorbar``). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | ipywidgets.VBox with ``.viewer`` (the `NiiVue`) and
 | ``.threshold_slider`` when ``controls=True``; otherwise the bare
 | ``ipyniivue.NiiVue`` widget.

######## `mean`

```python
mean(axis = 0, *, spatial_scale: str = 'whole_brain', roi_mask: str = None)
```

Get mean of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array). Ignored when ``spatial_scale='roi'``. | <code>0</code>
`spatial_scale` | <code>[str](#str)</code> | ``'whole_brain'`` (default) preserves existing behavior. ``'roi'`` requires ``roi_mask`` and returns a BrainData of the same shape with each voxel painted with its parcel's mean per image (parcellation smoothing). | <code>'whole_brain'</code>
`roi_mask` |  | Atlas image for ``spatial_scale='roi'``. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Mean values.

######## `median`

```python
median(axis = 0, *, spatial_scale: str = 'whole_brain', roi_mask: str = None)
```

Get median of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array). Ignored when ``spatial_scale='roi'``. | <code>0</code>
`spatial_scale` | <code>[str](#str)</code> | ``'whole_brain'`` (default) or ``'roi'`` (paints each voxel with its parcel's median per image). | <code>'whole_brain'</code>
`roi_mask` |  | Atlas image for ``spatial_scale='roi'``. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Median values.

######## `multivariate_similarity`

```python
multivariate_similarity(images, method = 'ols')
```

Predict a BrainData spatial distribution from a linear combination.

The predictors may be other BrainData instances or nibabel images.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`images` |  | BrainData instance of weight map | *required*
`method` | <code>[str](#str)</code> | Regression method. Default: 'ols'. | <code>'ols'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | dictionary of regression statistics in BrainData instances {'beta','t','p','df','residual'}

######## `plot`

```python
plot(*, method = 'glass', upper = None, lower = None, threshold = None, view = 'z', cut_coords = None, cmap = None, bg_img = None, ax = None, figsize = (8, 6), title = None, colorbar = True, save = None, stat = 'mean', limit = 3, **kwargs)
```

Plot BrainData instance using nilearn visualization or matplotlib.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Visualization type: 'glass', 'slices', 'timeseries', 'histogram' | <code>'glass'</code>
`upper` | <code>[str](#str) / [float](#float)</code> | Upper threshold. | <code>None</code>
`lower` | <code>[str](#str) / [float](#float)</code> | Lower threshold. | <code>None</code>
`threshold` | <code>[float](#float)</code> | Convenience parameter for thresholding. | <code>None</code>
`view` | <code>[str](#str)</code> | For ``method="slices"``, any non-empty combination of ``"x"``, ``"y"``, ``"z"`` (e.g. ``"xyz"``, ``"xz"``, ``"y"``). Default: ``"z"``. | <code>'z'</code>
`cut_coords` | <code>[list](#list) or [dict](#dict)</code> | Cut coordinates for multi-slice views. Takes precedence over ``view``-based defaults. Either a list matching ``len(view)`` or a dict keyed by axis letter. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Colormap name. | <code>None</code>
`bg_img` | <code>str/nibabel image</code> | Background image. | <code>None</code>
`ax` | <code>[Axes](#matplotlib.axes.Axes)</code> | Matplotlib axis. | <code>None</code>
`figsize` | <code>[tuple](#tuple)</code> | default figure size if no axis (8, 6) | <code>(8, 6)</code>
`title` | <code>[str](#str)</code> | Plot title. | <code>None</code>
`colorbar` | <code>[bool](#bool)</code> | Whether to show colorbar. Default: True. | <code>True</code>
`save` | <code>[str](#str)</code> | Path to save figure(s). | <code>None</code>
`stat` | <code>[str](#str)</code> | Statistic for timeseries plots. Default: 'mean'. | <code>'mean'</code>
`limit` | <code>[int](#int)</code> | Maximum number of images to render when this BrainData contains multiple maps and ``method`` is ``"glass"`` or ``"slices"``. Default: 3. Warns when more images exist than ``limit``. | <code>3</code>
`**kwargs` |  | Additional arguments passed to nilearn plot functions. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure or list[matplotlib.figure.Figure]: A
 | single figure for single-image data; a list of figures for
 | multi-image data with ``method`` in ``{"glass", "slices"}``
 | (one per image for glass; one per image-and-view pair for
 | slices).

######## `plot_flatmap`

```python
plot_flatmap(*, threshold = None, cmap = 'RdBu_r', vmax = None, vmin = None, template = 'fsaverage5', with_curvature = True, curvature_contrast = 0.5, curvature_brightness = 0.5, transparency = 'auto', colorbar = True, colorbar_orientation = 'horizontal', figsize = (12, 6), title = None, radius_mm = 3.0, interpolation = 'linear', axes = None, save = None)
```

Plot brain data on cortical flatmap.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`threshold` | <code>[float](#float)</code> | Values below this absolute threshold are masked. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Matplotlib colormap. Default: 'RdBu_r'. | <code>'RdBu_r'</code>
`vmax` | <code>[float](#float)</code> | Maximum value for colormap. | <code>None</code>
`vmin` | <code>[float](#float)</code> | Minimum value for colormap. | <code>None</code>
`template` | <code>[str](#str)</code> | Freesurfer surface resolution. Default: 'fsaverage5'. | <code>'fsaverage5'</code>
`with_curvature` | <code>[bool](#bool)</code> | Show sulcal/gyral pattern. Default: True. | <code>True</code>
`curvature_contrast` | <code>[float](#float)</code> | Contrast of curvature overlay. Default: 0.5. | <code>0.5</code>
`curvature_brightness` | <code>[float](#float)</code> | Mean brightness of curvature overlay. Default: 0.5. | <code>0.5</code>
`transparency` | <code>BrainData, Nifti1Image, str, or "auto"</code> | Binary mask used to render vertices outside the mask as transparent. ``"auto"`` (default) uses the instance's ``.mask``; pass ``None`` to disable masking. | <code>'auto'</code>
`colorbar` | <code>[bool](#bool)</code> | Show colorbar. Default: True. | <code>True</code>
`colorbar_orientation` | <code>[str](#str)</code> | 'horizontal' or 'vertical'. Default: 'horizontal'. | <code>'horizontal'</code>
`figsize` | <code>[tuple](#tuple)</code> | Figure size as (width, height). Default: (12, 6). | <code>(12, 6)</code>
`title` | <code>[str](#str)</code> | Figure title. | <code>None</code>
`radius_mm` | <code>[float](#float)</code> | Sampling radius in mm. Default: 3.0. | <code>3.0</code>
`interpolation` | <code>[str](#str)</code> | Interpolation method. Default: 'linear'. | <code>'linear'</code>
`axes` | <code>[Axes](#matplotlib.axes.Axes)</code> | Existing axes to plot on. | <code>None</code>
`save` | <code>[str](#str)</code> | File path to save figure. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

######## `plot_surf`

```python
plot_surf(*, hemi = 'both', view = 'montage', surface = 'pial', template = 'fsaverage5', threshold = None, cmap = 'RdBu_r', vmin = None, vmax = None, transparency = 'auto', bg_on_data = False, colorbar = True, colorbar_orientation = 'horizontal', figsize = (10, 8), title = None, radius_mm = 3.0, interpolation = 'linear', zoom = 1.2, axes = None, save = None)
```

Render this BrainData on fsaverage surfaces as a tight 2×2 montage.

Facade over `plot_surf`. See that function's
docstring for the full argument reference. Notable defaults:
``surface="pial"``, ``zoom=1.2``, ``transparency="auto"`` (uses
this instance's ``.mask``).

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

######## `predict`

```python
predict(*, y: np.ndarray | None = None, X: np.ndarray | None = None, spatial_scale: str = 'whole_brain', model: str = 'svm', cv: int = 5, standardize: bool = True, reduce: str | None = None, n_components: int | None = None, scoring: str = 'auto', groups: np.ndarray | None = None, roi_mask: np.ndarray | None = None, radius_mm: float = 10.0, inplace: bool = False, n_jobs: int = 1, random_state: int | None = None, progress_bar: bool = False)
```

Predict voxel timeseries (encoding) or decode labels (MVPA).

Dispatched by which of ``X`` or ``y`` is provided:

1. **Timeseries prediction** (``X`` provided): use a fitted ridge /
   GLM encoding model on ``self`` to predict voxel responses.
   Returns a fresh ``BrainData`` whose ``.data`` holds the predicted
   timeseries (composes directly with ``.plot()``, ``.standardize()``
   etc.). ``inplace`` has no effect in this mode.
2. **MVPA decoding** (``y`` provided): train a classifier or
   regressor with cross-validation. Returns a `Predict`
   dataclass. Spatial fields (``weight_map``, ``fold_weight_maps``,
   ``final_weight_map``, ``accuracy_map``) are `BrainData`
   objects so ``result.weight_map.plot()`` works directly. Drop down
   to numpy via ``result.weight_map.data``.

Field shapes by ``spatial_scale=``:

- **whole_brain**: ``predictions`` (n_samples,) OOF predictions,
  ``scores`` (n_folds,), ``mean_score`` float, ``std_score`` float,
  ``weight_map`` BrainData (``coef_`` from one fit on the **full**
  ``(X, y)`` — the publishable map), ``fold_weight_maps`` BrainData
  (n_folds, n_voxels) for stability analysis, ``estimator`` the
  fitted all-data sklearn estimator (use for ``.predict()`` on new
  data).
- **roi**: ``scores`` (n_folds, n_rois), ``mean_score`` (n_rois,),
  ``std_score`` (n_rois,), ``roi_labels`` (n_rois,) atlas IDs in
  matching order, ``accuracy_map`` / ``weight_map`` /
  ``fold_weight_maps`` BrainData (per-parcel coefs reassembled to
  voxel space; voxels outside the atlas = NaN), ``estimator`` dict
  keyed by atlas label.
- **searchlight**: ``accuracy_map`` BrainData.

With ``inplace=True``, fields are attached to ``self`` with a
``predict_`` prefix (e.g. ``self.predict_weight_map``,
``self.predict_accuracy_map``), mirroring ``bd.fit()``'s
``glm_*`` / ``ridge_*`` naming.

Why ``weight_map`` is the all-data refit, not the CV mean:
the mean of K per-fold ``coef_`` vectors doesn't correspond to
any actual fitted estimator (each fold saw a different subset).
The all-data refit is a single legitimate model with all the
information used. CV gives the honest *score*; the refit gives
the publishable *map*. The CV-mean is one line away if you want
it: ``result.fold_weight_maps.data.mean(axis=0)``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` | <code>[array](#array) - [like](#like)</code> | Labels (classification) or continuous targets (regression), shape ``(n_samples,)``. Triggers MVPA mode. | <code>None</code>
`X` | <code>[array](#array) - [like](#like)</code> | Features for timeseries prediction, shape ``(n_samples, n_features)``. Triggers encoding mode. | <code>None</code>
`spatial_scale` | <code>[str](#str)</code> | MVPA dispatch — ``'whole_brain'``, ``'searchlight'``, or ``'roi'``. | <code>'whole_brain'</code>
`model` | <code>str or sklearn estimator</code> | Algorithm. String shortcuts:<br>- Classification: ``'svm'`` (LinearSVC), ``'logistic'``,   ``'lda'``, ``'ridge_classifier'``. - Regression: ``'ridge'``, ``'lasso'``, ``'svr'``.<br>Or pass any sklearn estimator / Pipeline (e.g., ``make_pipeline(StandardScaler(), SelectKBest(k=500), LinearSVC())``). When ``model`` is a sklearn ``Pipeline``, ``standardize`` is auto-defaulted to ``False`` (with a warning) so we don't wrap another StandardScaler around your pipeline. Pass ``standardize=True`` explicitly to override. | <code>'svm'</code>
`cv` | <code>int or sklearn CV splitter</code> | ``int`` → KFold (regression) or StratifiedKFold (classification); pass a splitter for custom schemes (e.g., ``GroupKFold``). | <code>5</code>
`standardize` | <code>[bool](#bool)</code> | Z-score features per fold before fitting. Default ``True``. Auto-flipped to ``False`` when ``model`` is a sklearn ``Pipeline`` (see ``model`` above). | <code>True</code>
`reduce` | <code>[str](#str)</code> | Per-fold dimensionality reduction. Currently only ``'pca'`` supported. Default ``None``. Weight maps are back-projected through PCA to voxel space. | <code>None</code>
`n_components` | <code>[int](#int)</code> | PCA components when ``reduce='pca'``. | <code>None</code>
`scoring` | <code>[str](#str)</code> | Sklearn scoring string. Default ``'auto'`` → ``'accuracy'`` if classifier, ``'r2'`` if regressor. | <code>'auto'</code>
`groups` | <code>[array](#array) - [like](#like)</code> | Group labels for CV splitters that need them (e.g., leave-one-run-out). | <code>None</code>
`roi_mask` | <code>[Nifti1Image](#Nifti1Image) or [path](#path) - [like](#like)</code> | Atlas image for ``spatial_scale='roi'``. | <code>None</code>
`radius_mm` | <code>[float](#float)</code> | Searchlight radius in mm. Default ``10.0``. | <code>10.0</code>
`inplace` | <code>[bool](#bool)</code> | If ``True``, populate result fields as ``predict_*`` attributes on ``self`` and return ``self``. Default ``False`` returns a fresh `Predict`. | <code>False</code>
`n_jobs` | <code>[int](#int)</code> | Parallel jobs for searchlight / ROI. Default ``1``; searchlight on a real brain at higher ``n_jobs`` can be memory-heavy. | <code>1</code>
`random_state` | <code>[int](#int)</code> | Seed for the shuffled fold splitter when ``cv`` is an int (MVPA mode). Default ``None`` (unseeded shuffle each call). Ignored when ``cv`` is a splitter object — set its own ``random_state`` instead. | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar for searchlight / ROI. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
 | Predict | BrainData: ``Predict`` dataclass when ``inplace=False``; ``self`` (mutated, with ``predict_*`` attrs) when ``inplace=True``.

**Examples:**

```pycon
>>> result = brain.predict(y=labels, spatial_scale='whole_brain', cv=5)
>>> result.weight_map.plot()       # publishable map (all-data fit)
>>> result.mean_score              # honest CV-derived accuracy
>>> new_pred = result.estimator.predict(new_X)  # apply to new data
```

```pycon
>>> result = brain.predict(y=labels, spatial_scale='searchlight',
...                        radius_mm=8.0, n_jobs=4)
>>> result.accuracy_map.plot()
```

```pycon
>>> result = brain.predict(y=labels, spatial_scale='roi', roi_mask=atlas)
>>> top = result.roi_labels[result.mean_score.argsort()[::-1][:10]]
>>> result.accuracy_map.plot()  # brain-space view of the same map
```

Custom sklearn pipeline as model — standardize auto-defaults to
False because we detect the Pipeline:

```python
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
pipe = make_pipeline(StandardScaler(), SelectKBest(k=500),
                     LinearSVC())
result = brain.predict(y=labels, model=pipe)
```

######## `r_to_z`

```python
r_to_z()
```

Apply Fisher's r-to-z transformation to each data element.

######## `regions`

```python
regions(*, min_region_size = 1350, method = 'local_regions', smoothing_fwhm = 6, is_mask = False)
```

Extract brain connected regions into separate regions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`min_region_size` | <code>[int](#int)</code> | Minimum volume in mm3 for a region to be kept. | <code>1350</code>
`method` | <code>[str](#str)</code> | Type of extraction method                 ['connected_components', 'local_regions']. | <code>'local_regions'</code>
`smoothing_fwhm` | <code>[scalar](#scalar)</code> | Smooth an image to extract more sparser regions. | <code>6</code>
`is_mask` | <code>[bool](#bool)</code> | Whether to treat as boolean mask. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with extracted ROIs as data.

######## `report`

```python
report(contrasts = None, **kwargs)
```

Generate a nilearn HTML report for a fitted GLM.

Must be called after ``fit(model='glm', ...)``. Renders the design
matrix, requested contrast maps, and model parameters as a
self-contained HTML report.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`contrasts` | <code>str, list, or dict</code> | Contrast(s) to render, same forms as `compute_contrasts`. | <code>None</code>
`**kwargs` |  | Forwarded to nilearn's ``generate_report`` (e.g. ``title``, ``threshold``, ``alpha``). | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`HTMLReport` |  | nilearn report; call ``.save_as_html(path)`` or display it in a notebook.

**Examples:**

```pycon
>>> brain.fit(model='glm', X=design_matrix)
>>> brain.report(contrasts='conditionA - conditionB').save_as_html('report.html')
```

######## `resample_to`

```python
resample_to(*, img = None, resolution = None, interpolation = None)
```

Resample BrainData to match target image or resolution.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`img` |  | Target image for resampling (nibabel Nifti1Image, str/Path, or None). | <code>None</code>
`resolution` |  | Target voxel size in mm (float/int for isotropic, or None). | <code>None</code>
`interpolation` |  | Interpolation method ('nearest', 'linear', 'continuous', or None). | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | New BrainData instance with resampled data

######## `scale`

```python
scale(scale_val = 100.0, axis = None)
```

Scale data via mean scaling.

Two scaling modes are available:

- **Grand-mean scaling** (axis=None, default): Divides all values by the
  global mean across all voxels and timepoints.

- **Voxel-wise scaling** (axis=0): Divides each voxel's time-series by
  its own temporal mean.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`scale_val` |  | (int/float) Target value for the mean after scaling. Default 100. | <code>100.0</code>
`axis` |  | (int or None) None for grand-mean scaling (default), 0 for voxel-wise scaling. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | New BrainData instance with scaled data.

######## `similarity`

```python
similarity(image, metric = 'correlation')
```

Calculate similarity to a single BrainData or nibabel image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`image` |  | (BrainData, nifti) image to evaluate similarity | *required*
`metric` |  | (str) Type of similarity     ['correlation','pearson','rank_correlation','spearman','dot_product','cosine'] | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
 | float or np.ndarray: Similarity value(s).

######## `smooth`

```python
smooth(fwhm)
```

Apply spatial smoothing using nilearn smooth_img().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fwhm` |  | (float) full width half maximum of gaussian spatial filter | *required*

**Returns:**

Type | Description
---- | -----------
 | BrainData instance (copy with smoothed data)

######## `standardize`

```python
standardize(*, axis = 0, method = 'center', suppress_warnings = False)
```

Standardize BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>[int](#int)</code> | 0 standardizes each voxel across observations (default). 1 standardizes each observation across voxels. | <code>0</code>
`method` | <code>[str](#str)</code> | 'center' subtracts the mean (default). 'zscore' subtracts the mean and divides by standard deviation. | <code>'center'</code>
`suppress_warnings` | <code>[bool](#bool)</code> | If True, suppress sklearn numerical warnings that occur when voxels have near-zero variance. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Standardized BrainData instance.

######## `std`

```python
std(axis = 0, *, spatial_scale: str = 'whole_brain', roi_mask: str = None)
```

Get standard deviation of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array). Ignored when ``spatial_scale='roi'``. | <code>0</code>
`spatial_scale` | <code>[str](#str)</code> | ``'whole_brain'`` (default) or ``'roi'`` (paints each voxel with its parcel's std per image). | <code>'whole_brain'</code>
`roi_mask` |  | Atlas image for ``spatial_scale='roi'``. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Standard deviation values.

######## `sum`

```python
sum(axis = 0)
```

Get sum of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array) | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Sum values.

######## `temporal_resample`

```python
temporal_resample(*, sampling_freq = None, target = None, target_type = 'hz')
```

Resample BrainData timeseries to a new target frequency or number of samples.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sampling_freq` |  | (float) sampling frequency of data in hertz | <code>None</code>
`target` |  | (float) upsampling target | <code>None</code>
`target_type` |  | (str) type of target can be [samples,seconds,hz] | <code>'hz'</code>

**Returns:**

Type | Description
---- | -----------
 | upsampled BrainData instance

######## `threshold`

```python
threshold(*, upper = None, lower = None, binarize = False, coerce_nan = True, cluster_threshold = 0)
```

Threshold BrainData instance with optional cluster filtering.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`upper` |  | (float or str) Upper cutoff for thresholding. | <code>None</code>
`lower` |  | (float or str) Lower cutoff for thresholding. | <code>None</code>
`binarize` | <code>[bool](#bool)</code> | return binarized image. Default False. | <code>False</code>
`coerce_nan` | <code>[bool](#bool)</code> | coerce nan values to 0s. Default True. | <code>True</code>
`cluster_threshold` | <code>[int](#int)</code> | Minimum cluster size in voxels. Default 0. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | Thresholded BrainData object.

######## `to_nifti`

```python
to_nifti()
```

Convert BrainData Instance into Nifti Object.

**Returns:**

Type | Description
---- | -----------
 | nibabel.Nifti1Image: Brain data as a NIfTI image.

######## `transform_pairwise`

```python
transform_pairwise()
```

Transform data into pairwise comparisons.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance transformed into pairwise comparisons

######## `ttest`

```python
ttest(*, popmean = 0.0, permutation = False, n_permute = 5000, tail = 2, return_null = False, n_jobs = -1, random_state = None)
```

One-sample voxelwise t-test across images (axis 0).

Tests whether the per-voxel mean across images differs from
``popmean``. Operates on a stack of images (e.g. subject-level
contrast maps) with shape ``(n_samples, n_voxels)``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`popmean` |  | Population mean to test against. Default 0.0. | <code>0.0</code>
`permutation` |  | If True, use sign-flip permutation test via `one_sample_permutation_test`. | <code>False</code>
`n_permute` |  | Number of permutations (used only when ``permutation=True``). Default 5000. | <code>5000</code>
`tail` |  | Tail of the test (1 or 2). Default 2. | <code>2</code>
`return_null` |  | If True, also return the null distribution. Default False. | <code>False</code>
`n_jobs` |  | Number of parallel jobs. Default -1 (all cores). | <code>-1</code>
`random_state` |  | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | dict with four BrainData keys:<br>- ``"mean"``: voxelwise mean across images (effect size). - ``"t"``: parametric one-sample t-statistic. - ``"z"``: signed z-score, ``sign(t) * norm.isf(p/2)`` —   matches nilearn's ``output_type='z_score'``. - ``"p"``: parametric p-value, or empirical p when   ``permutation=True``.
 | The effect size is always returned alongside the inferential maps
 | so group-level code never has to recompute the mean.

**Examples:**

```pycon
>>> # Stack of subject-level contrast maps
>>> result = contrast_maps.ttest()
>>> sig = result["p"].data < 0.05
>>> effect = result["mean"]       # for reporting magnitude
>>> z_map = result["z"]           # for nilearn-style thresholding
```

```pycon
>>> # Permutation-based p-values; still reports t/z/mean
>>> result = contrast_maps.ttest(permutation=True, n_permute=5000)
```

######## `ttest2`

```python
ttest2(other, equal_var = True)
```

Two-sample voxelwise t-test between two BrainData stacks.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` |  | BrainData to compare against. Must have the same number of voxels. | *required*
`equal_var` |  | If True (default), standard two-sample t-test. If False, Welch's t-test. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | ``{"t": BrainData, "p": BrainData}``.

######## `upload_neurovault`

```python
upload_neurovault(*, access_token = None, collection_name = None, collection_id = None, img_type = None, img_modality = None, **kwargs)
```

Upload BrainData images and metadata to NeuroVault.

Adds any columns in ``self.X`` to image metadata. The index is used as
the image name.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`access_token` |  | (str, Required) Neurovault api access token | <code>None</code>
`collection_name` |  | (str, Optional) name of new collection to create | <code>None</code>
`collection_id` |  | (int, Optional) neurovault collection_id if adding images             to existing collection | <code>None</code>
`img_type` |  | (str, Required) Neurovault map_type | <code>None</code>
`img_modality` |  | (str, Required) Neurovault image modality | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`collection` |  | (pd.DataFrame) neurovault collection information

######## `write`

```python
write(file_name)
```

Write out BrainData object to Nifti or HDF5 File.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>[str](#str) or [Path](#Path)</code> | Output file path (.nii/.nii.gz for NIfTI, .h5/.hdf5 for HDF5). | *required*

######## `z_to_r`

```python
z_to_r()
```

Convert z score back into r value for each element of data object.



##### Methods

##### Modules

(data-analysis)=
###### `analysis`

BrainData analysis functions.

Standalone functions extracted from BrainData class methods for similarity,
distance, masking, ROI extraction, filtering, thresholding, decomposition,
alignment, smoothing, and other analytical operations.

**Methods:**

Name | Description
---- | -----------
[`align`](#data-align) | Align a BrainData instance to a target using functional alignment.
`align_per_roi` | Per-parcel functional alignment + voxel-space reassembly.
[`apply_mask`](#data-apply-mask) | Mask BrainData instance using nilearn functionality.
`check_masks` | Ensure two datasets use compatible masks, creating a union mask if needed.
[`decompose`](#data-decompose) | Decompose a BrainData object.
`detrend_data` | Remove the linear trend from each voxel.
[`distance`](#data-distance) | Calculate distance between images within a BrainData() instance.
[`extract_roi`](#data-extract-roi) | Extract activity from mask or ROI atlas using NiftiLabelsMasker.
`filter_data` | Apply butterworth filter to data. Wraps nilearn.signal.clean.
`find_spikes_data` | Identify spikes from time-series data; see `find_spikes`.
[`multivariate_similarity`](#data-multivariate-similarity) | Predict a BrainData spatial distribution from a linear combination.
[`r_to_z`](#data-r-to-z) | Apply Fisher's r-to-z transformation to each data element.
`reduce_per_roi` | Apply a reducer within each parcel and paint results back to voxel space.
[`regions`](#data-regions) | Extract brain connected regions into separate regions.
`scale_data` | Scale data via mean scaling.
[`similarity`](#data-similarity) | Calculate similarity to a single BrainData or nibabel image.
[`smooth`](#data-smooth) | Apply spatial smoothing using nilearn's ``smooth_img``.
[`standardize`](#data-standardize) | Standardize BrainData() instance.
[`temporal_resample`](#data-temporal-resample) | Resample a BrainData time series to a target frequency or sample count.
`threshold_data` | Threshold BrainData instance with optional cluster filtering.
`transform_pairwise_data` | Transform BrainData into pairwise comparisons.
[`z_to_r`](#data-z-to-r) | Convert z score back into r value for each element of data object.



####### Functions##

###### `align`

```python
align(bd, target, method = 'procrustes', axis = 0)
```

Align a BrainData instance to a target using functional alignment.

Alignment type can be hyperalignment or Shared Response Model. When
using hyperalignment, `target` image can be another subject or an
already estimated common model. When using SRM, `target` must be a previously
estimated common model stored as a numpy array. Transformed data can be back
projected to original data using Transformation matrix.

See nltools.stats.align for aligning multiple BrainData instances

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`target` |  | (BrainData) object to align to. | *required*
`method` |  | (str) alignment method to use ['probabilistic_srm','deterministic_srm','procrustes'] | <code>'procrustes'</code>
`axis` |  | (int) axis to align on (default: 0) | <code>0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (dict) a dictionary containing transformed object, transformation matrix, and the shared response matrix

**Examples:**

- Hyperalign using procrustes transform:
    >>> out = data.align(target, method='procrustes')
- Align using shared response model:
    >>> out = data.align(target, method='probabilistic_srm', n_features=None)
- Project aligned data into original data:
    >>> original_data = np.dot(out['transformed'].data,out['transformation_matrix'].T)

######## `align_per_roi`

```python
align_per_roi(bd, target, *, method, axis, roi_mask)
```

Per-parcel functional alignment + voxel-space reassembly.

For each atlas parcel, runs ``align()`` on the slice of ``bd`` and
``target`` restricted to that parcel's voxels and collects results.
The ``transformed`` field is reassembled into a single
`BrainData` of the same shape as the input (each voxel filled
with its parcel's transformed value per image; voxels outside any
parcel = NaN). Per-parcel transform matrices and common-model
objects are kept as dicts keyed by atlas label, since matrices over
different voxel subsets can't be painted into one image.

######## `apply_mask`

```python
apply_mask(bd, mask, resample_mask_to_brain = False)
```

Mask BrainData instance using nilearn functionality.

Note target data will be resampled into the same space as the mask. If you would like the mask
resampled into the BrainData space, then set resample_mask_to_brain=True.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`mask` |  | (BrainData or nifti object) mask to apply to BrainData object. | *required*
`resample_mask_to_brain` |  | (bool) Will resample mask to brain space before applying mask (default=False). | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`masked` |  | (BrainData) masked BrainData object

<details class="note" open markdown="1">
<summary>Note</summary>

Uses nilearn.masking.apply_mask for efficient, validated masking.
Simplified from 47-line manual implementation to leverage nilearn's
Cython-optimized code with better validation and memory management.

</details>

######## `check_masks`

```python
check_masks(bd, image)
```

Ensure two datasets use compatible masks, creating a union mask if needed.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance | *required*
`image` |  | BrainData instance to compare masks with | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` |  | (data2, image2) arrays with compatible masks

######## `decompose`

```python
decompose(bd, *, method = 'pca', axis = 'voxels', n_components = None, **kwargs)
```

Decompose a BrainData object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`method` |  | (str) Algorithm to perform decomposition         types=['pca','ica','nnmf','fa','dictionary','kernelpca'] | <code>'pca'</code>
`axis` |  | dimension to decompose ['voxels','images'] | <code>'voxels'</code>
`n_components` |  | (int) number of components. If None then retain         as many as possible (default: None). | <code>None</code>
`**kwargs` |  | Additional keyword arguments passed to the decomposition algorithm. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`output` |  | a dictionary of decomposition parameters

######## `detrend_data`

```python
detrend_data(bd, method = 'linear')
```

Remove the linear trend from each voxel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`method` |  | ('linear','constant', optional) type of detrending | <code>'linear'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (BrainData) detrended BrainData instance

######## `distance`

```python
distance(bd, metric = 'euclidean', *, spatial_scale: str = 'whole_brain', roi_mask: str = None, radius_mm: float = 10.0, **kwargs: float)
```

Calculate distance between images within a BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`metric` |  | (str) type of distance metric (can use any scipy.spatial.distance     metric supported by cdist, e.g., 'euclidean', 'cityblock', 'cosine',     'correlation', 'hamming', 'jaccard', etc.) | <code>'euclidean'</code>
`spatial_scale` | <code>[str](#str)</code> | ``'whole_brain'`` (default), ``'roi'``, or ``'searchlight'``. See `BrainData.distance`. | <code>'whole_brain'</code>
`roi_mask` |  | Atlas for ``spatial_scale='roi'``. | <code>None</code>
`radius_mm` | <code>[float](#float)</code> | Searchlight radius for ``spatial_scale='searchlight'``. | <code>10.0</code>
`**kwargs` |  | Additional arguments passed to scipy.spatial.distance.cdist. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dist` |  | (Adjacency) Whole-brain pairwise distance matrix, or a stacked Adjacency (one per parcel/searchlight) with ``spatial_scale`` provenance set.

######## `extract_roi`

```python
extract_roi(bd, mask, metric = 'mean', n_components = None)
```

Extract activity from mask or ROI atlas using NiftiLabelsMasker.

This method now uses nilearn's NiftiLabelsMasker for efficient ROI extraction
when dealing with labeled atlases (multiple ROIs).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`mask` |  | BrainData, nibabel image, or file path. Can be:<br>  - Binary mask (extracts from single ROI)   - Labeled atlas (extracts from multiple ROIs) | *required*
`metric` |  | Extraction method ('mean', 'median', 'pca'). Default: 'mean'     Note: 'median' and 'pca' require additional computation after extraction | <code>'mean'</code>
`n_components` |  | If metric='pca', number of components to return | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | For binary mask:<br>- Single image: scalar value - Multiple images: 1D array of values
 | For labeled atlas:<br>- Single image: 1D array (one value per ROI) - Multiple images: 2D array (images x ROIs) - If metric='pca': returns components array

**Examples:**

```pycon
>>> # Extract mean from binary mask
>>> roi_values = brain.extract_roi(binary_mask)
>>> # Extract from atlas
>>> atlas_values = brain.extract_roi(atlas_mask)
>>> # PCA extraction
>>> components = brain.extract_roi(mask, metric='pca', n_components=5)
```

######## `filter_data`

```python
filter_data(bd, *, sampling_freq = None, high_pass = None, low_pass = None, **kwargs)
```

Apply butterworth filter to data. Wraps nilearn.signal.clean.

Does not default to detrending and standardizing like nilearn
implementation, but this can be overridden using kwargs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`sampling_freq` |  | Sampling freq in hertz (i.e. 1 / TR). Default: None. | <code>None</code>
`high_pass` |  | High pass cutoff frequency. Default: None. | <code>None</code>
`low_pass` |  | Low pass cutoff frequency. Default: None. | <code>None</code>
`**kwargs` |  | Additional arguments passed to nilearn.signal.clean       Common options:       - confounds: Confound timeseries to remove       - sample_mask: Volumes to exclude (scrubbing)       - detrend: Enable detrending (default False)       - standardize: Enable standardization (default False)       - ensure_finite: Replace NaN/inf (default False) | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Filtered BrainData instance

<details class="see-also" open markdown="1">
<summary>See Also</summary>

nilearn.signal.clean documentation for all available options

</details>

######## `find_spikes_data`

```python
find_spikes_data(bd, global_spike_cutoff = 3, diff_spike_cutoff = 3, *, TR = None, sampling_freq = None)
```

Identify spikes from time-series data; see `find_spikes`.

######## `multivariate_similarity`

```python
multivariate_similarity(bd, images, method = 'ols')
```

Predict a BrainData spatial distribution from a linear combination.

The predictors may be other BrainData instances or nibabel images.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance of data to be applied | *required*
`images` |  | BrainData instance of weight map | *required*
`method` | <code>[str](#str)</code> | Regression method. Default: 'ols'. | <code>'ols'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | dictionary of regression statistics in BrainData instances {'beta','t','p','df','residual'}

######## `r_to_z`

```python
r_to_z(bd)
```

Apply Fisher's r-to-z transformation to each data element.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Transformed BrainData instance.

######## `reduce_per_roi`

```python
reduce_per_roi(bd, reducer, *, roi_mask)
```

Apply a reducer within each parcel and paint results back to voxel space.

This performs spatial smoothing via parcellation using a reducer such as
``np.mean``.

For each image ``i`` and each parcel ``p``, computes
``reducer(bd.data[i, voxels-in-p])`` and assigns that scalar to every
voxel in parcel ``p`` for image ``i``. Voxels outside any parcel get
NaN. Output is a `BrainData` of the same shape as the input.

Used by ``BrainData.{mean,std,median}(spatial_scale='roi')``.

######## `regions`

```python
regions(bd, *, min_region_size = 1350, method = 'local_regions', smoothing_fwhm = 6, is_mask = False)
```

Extract brain connected regions into separate regions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`min_region_size` | <code>[int](#int)</code> | Minimum volume in mm3 for a region to be                 kept. | <code>1350</code>
`method` | <code>[str](#str)</code> | Type of extraction method                 ['connected_components', 'local_regions'].                 If 'connected_components', each component/region                 in the image is extracted automatically by                 labelling each region based upon the presence of                 unique features in their respective regions.                 If 'local_regions', each component/region is                 extracted based on their maximum peak value to                 define a seed marker and then using random                 walker segementation algorithm on these                 markers for region separation. | <code>'local_regions'</code>
`smoothing_fwhm` | <code>[scalar](#scalar)</code> | Smooth an image to extract more sparser                 regions. Only works for method='local_regions'. | <code>6</code>
`is_mask` | <code>[bool](#bool)</code> | Whether the BrainData instance should be treated             as a boolean mask and if so, calls             connected_label_regions instead. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with extracted ROIs as data.

######## `scale_data`

```python
scale_data(bd, scale_val = 100.0, axis = None)
```

Scale data via mean scaling.

Two scaling modes are available:

- **Grand-mean scaling** (axis=None, default): Divides all values by the
  global mean across all voxels and timepoints. This is consistent with
  FSL and SPM behavior. Use scale_val=10000 for FSL-style scaling.

- **Voxel-wise scaling** (axis=0): Divides each voxel's time-series by
  its own temporal mean. This is AFNI-style scaling and can be useful
  when voxels have very different baseline intensities. Voxels with
  zero or near-zero mean are set to zero to avoid NaN/Inf.

When scale_val=100 (default), the result can be interpreted as something
akin to (but not exactly) "percent signal change."

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`scale_val` |  | (int/float) Target value for the mean after scaling. Default 100. | <code>100.0</code>
`axis` |  | (int or None) Axis along which to compute the mean. None for grand-mean scaling (default, FSL/SPM style). 0 for voxel-wise scaling (AFNI style, each voxel scaled by its own temporal mean). | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | New BrainData instance with scaled data.

**Examples:**

```pycon
>>> # Grand-mean scaling (default)
>>> scaled = brain.scale(100.0)
>>>
>>> # Voxel-wise scaling (AFNI style)
>>> scaled = brain.scale(100.0, axis=0)
```

######## `similarity`

```python
similarity(bd, image, metric = 'correlation')
```

Calculate similarity to a single BrainData or nibabel image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`image` |  | (BrainData, nifti)  image to evaluate similarity | *required*
`metric` |  | (str) Type of similarity     ['correlation', 'pearson', 'rank_correlation', 'spearman', 'dot_product', 'cosine'] | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
 | np.ndarray: Similarity values.

######## `smooth`

```python
smooth(bd, fwhm)
```

Apply spatial smoothing using nilearn's ``smooth_img``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`fwhm` |  | (float) full width half maximum of gaussian spatial filter | *required*

**Returns:**

Type | Description
---- | -----------
 | BrainData instance (copy with smoothed data)

######## `standardize`

```python
standardize(bd, *, axis = 0, method = 'center', suppress_warnings = False)
```

Standardize BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`axis` |  | 0 for observations 1 for voxels (default: 0) | <code>0</code>
`method` |  | ['center','zscore'] (default: 'center') | <code>'center'</code>
`suppress_warnings` |  | If True, suppress sklearn numerical warnings that occur when voxels have near-zero variance. (default: False) | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Standardized BrainData instance.

######## `temporal_resample`

```python
temporal_resample(bd, *, sampling_freq = None, target = None, target_type = 'hz')
```

Resample a BrainData time series to a target frequency or sample count.

Resample BrainData timeseries to a new target frequency or number of samples
using Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) interpolation.
This function can up- or down-sample data.

Note: this function can use quite a bit of RAM.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`sampling_freq` |  | (float) sampling frequency of data in hertz (default: None) | <code>None</code>
`target` |  | (float) upsampling target (default: None) | <code>None</code>
`target_type` |  | (str) type of target can be [samples,seconds,hz] (default: 'hz') | <code>'hz'</code>

**Returns:**

Type | Description
---- | -----------
 | upsampled BrainData instance

######## `threshold_data`

```python
threshold_data(bd, *, upper = None, lower = None, binarize = False, coerce_nan = True, cluster_threshold = 0)
```

Threshold BrainData instance with optional cluster filtering.

Provide upper and lower values or percentages to perform two-sided
thresholding. Binarize will return a mask image respecting thresholds
if provided, otherwise respecting every non-zero value.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`upper` |  | (float or str) Upper cutoff for thresholding. If string     will interpret as percentile; can be None for one-sided     thresholding. | <code>None</code>
`lower` |  | (float or str) Lower cutoff for thresholding. If string     will interpret as percentile; can be None for one-sided     thresholding. | <code>None</code>
`bd` |  | BrainData instance. | *required*
`binarize` | <code>[bool](#bool)</code> | return binarized image respecting thresholds if     provided, otherwise binarize on every non-zero value;     default False | <code>False</code>
`coerce_nan` | <code>[bool](#bool)</code> | coerce nan values to 0s; default True | <code>True</code>
`cluster_threshold` | <code>[int](#int)</code> | Minimum cluster size in voxels. If > 0, uses     nilearn.image.threshold_img with cluster filtering.     Band-pass filtering (both upper AND lower) not supported     with cluster thresholding. Default 0 (disabled). | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | Thresholded BrainData object.

<details class="note" open markdown="1">
<summary>Note</summary>

When cluster_threshold=0 (default), uses fast path for basic thresholding.
When cluster_threshold>0, uses nilearn for cluster filtering.
Band-pass filtering (unique nltools feature) preserved when cluster_threshold=0.

</details>

######## `transform_pairwise_data`

```python
transform_pairwise_data(bd)
```

Transform BrainData into pairwise comparisons.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance transformed into pairwise comparisons.

######## `z_to_r`

```python
z_to_r(bd)
```

Convert z score back into r value for each element of data object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Transformed BrainData instance.

###### `bootstrap`

Bootstrap functions extracted from BrainData methods.

**Methods:**

Name | Description
---- | -----------
[`bootstrap`](#data-bootstrap) | Bootstrap statistics using efficient online algorithms.
`convert_bootstrap_results_to_brain_data` | Convert bootstrap results dictionary to BrainData format.



####### Functions##

###### `bootstrap`

```python
bootstrap(bd, stat, *, n_samples = 5000, save_boots = False, percentiles = (2.5, 97.5), X_test = None, backend = None, max_gpu_memory_gb = 4.0, n_jobs = -1, random_state = None)
```

Bootstrap statistics using efficient online algorithms.

Uses memory-efficient bootstrap infrastructure with CPU parallelization or GPU acceleration.
Supports simple aggregation statistics and fitted model statistics (Ridge).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`stat` |  | (str) Statistic to bootstrap. Options: Simple stats ('mean', 'median', 'std', 'sum', 'min', 'max') or Model stats ('weights' requires fitted Ridge model, 'predict' requires fitted Ridge model + X_test). | *required*
`n_samples` |  | (int) Number of bootstrap iterations. Default: 5000 | <code>5000</code>
`save_boots` |  | (bool) If True, store all bootstrap samples (memory intensive).        Default: False | <code>False</code>
`percentiles` |  | (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5) | <code>(2.5, 97.5)</code>
`X_test` |  | (np.ndarray, optional) Test features for 'predict' bootstrap.    Required if stat='predict' | <code>None</code>
`backend` |  | (str, optional) Backend for Ridge bootstrap: None (CPU), 'torch' (GPU if available), or 'auto' (auto-select). Ignored for simple stats. Default: None | <code>None</code>
`max_gpu_memory_gb` |  | (float) Maximum GPU memory to use when backend is 'torch' or 'auto'. Default: 4.0 | <code>4.0</code>
`n_jobs` |  | (int) Number of CPU cores for parallelization. Default: -1 (all CPUs). | <code>-1</code>
`random_state` |  | (int, optional) Random seed for reproducibility | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or dict: - For simple stats: Returns BrainData with bootstrap mean - For model stats: Returns dict with keys: 'mean', 'std', 'Z', 'p',   'ci_lower', 'ci_upper' (all BrainData objects) - If ``save_boots=True``: Returns dict with 'samples' key containing all samples

**Examples:**

```pycon
>>> # Simple aggregation
>>> boot = brain.bootstrap(stat='mean', n_samples=1000)
>>> assert isinstance(boot, BrainData)
```

```pycon
>>> # Ridge weights bootstrap (CPU)
>>> brain.fit(X=dm, model='ridge', alpha=1.0)
>>> boot = brain.bootstrap(stat='weights', n_samples=1000)
>>> assert 'mean' in boot
>>> assert isinstance(boot['mean'], BrainData)
```

```pycon
>>> # Ridge weights bootstrap (GPU accelerated)
>>> brain.fit(X=dm, model='ridge', alpha=1.0)
>>> boot = brain.bootstrap(stat='weights', n_samples=1000, backend='torch')
>>> assert 'mean' in boot
>>> assert isinstance(boot['mean'], BrainData)
```

```pycon
>>> # Ridge predict bootstrap
>>> brain.fit(X=dm, model='ridge', alpha=1.0)
>>> boot = brain.bootstrap(stat='predict', X_test=X_new, n_samples=1000)
>>> assert 'mean' in boot
>>> assert isinstance(boot['mean'], BrainData)
```

<details class="note" open markdown="1">
<summary>Note</summary>

This method replaces the removed `summarize_bootstrap()` function.

**New API:**
>>> # Option 1: Use BrainData.bootstrap() for generating bootstrap samples
>>> boot = brain.bootstrap(stat='mean', n_samples=1000, save_boots=False)
>>> # Returns BrainData with bootstrap mean
>>> # To get Z and p, use stat='weights' or 'predict' which returns dict

>>> # Option 2: For existing bootstrap samples (BrainData with multiple images),
>>> # use OnlineBootstrapStats directly:
>>> from nltools.algorithms.inference.bootstrap import OnlineBootstrapStats
>>> stats = OnlineBootstrapStats(shape=(brain.shape[1],), save_samples=False)
>>> for sample in bootstrap_samples:  # Iterate over samples
...     stats.update(sample.data)
>>> result = stats.get_results()
>>> # Returns: {'mean': array, 'std': array, 'Z': array, 'p': array,
>>> #           'ci_lower': array, 'ci_upper': array}
>>> # Convert to BrainData if needed:
>>> mean_brain = shallow_copy(brain)
>>> mean_brain.data = result['mean']

</details>

######## `convert_bootstrap_results_to_brain_data`

```python
convert_bootstrap_results_to_brain_data(bd, result, save_boots = False, return_dict = False)
```

Convert bootstrap results dictionary to BrainData format.

Helper method to convert numpy arrays from bootstrap functions into
BrainData objects or dicts of BrainData objects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`result` |  | (dict) Result dictionary from bootstrap function with keys:     'mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper', and optionally 'samples' | *required*
`save_boots` |  | (bool) If True, include 'samples' key in output | <code>False</code>
`return_dict` |  | (bool) If True, always return dict even for simple stats.         If False, return BrainData for simple stats (when save_boots=False) | <code>False</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or dict: - If return_dict=False and save_boots=False: Returns BrainData with mean - Otherwise: Returns dict with BrainData objects for each statistic

(data-cache)=
###### `cache`

Disk-based caching infrastructure for expensive computations.

This module provides a general-purpose caching system for nltools, designed to
be reused across various computationally expensive operations like searchlight
neighborhoods, ISC, and SRM.

<details class="example" open markdown="1">
<summary>Example</summary>

>>> from nltools.data.braindata.cache import CacheManager, hash_mask
>>> import nibabel as nib
>>>
>>> # Hash a mask for cache key generation
>>> mask = nib.load("mask.nii.gz")
>>> mask_hash = hash_mask(mask)
>>>
>>> # Use cache manager for searchlight neighborhoods
>>> cache = CacheManager("searchlight")
>>> if not cache.exists(f"{mask_hash}_10mm"):
...     # Compute expensive operation
...     result = compute_something()
...     cache.save(f"{mask_hash}_10mm", data=result)
>>> else:
...     result = cache.load(f"{mask_hash}_10mm")["data"]

</details>

**Classes:**

Name | Description
---- | -----------
[`CacheManager`](#data-cachemanager) | Manages disk-based caching for expensive computations.

**Methods:**

Name | Description
---- | -----------
[`clear_cache`](#data-clear-cache) | Clear the nltools cache.
`get_cache_dir` | Get the nltools cache directory.
`hash_mask` | Compute a stable hash for a NIfTI mask image.



####### Classes##

(data-cachemanager)=
###### `CacheManager`

```python
CacheManager(category: str = 'general')
```

Manages disk-based caching for expensive computations.

CacheManager provides a simple key-value interface for caching numpy arrays
to disk. It organizes cached files by category (e.g., "searchlight", "isc")
in separate subdirectories.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`category` | <code>[str](#str)</code> | Category name for organizing cached files (e.g., "searchlight") | <code>'general'</code>

<details class="example" open markdown="1">
<summary>Example</summary>

>>> cache = CacheManager("searchlight")
>>>
>>> # Check if something is cached
>>> if cache.exists("mykey"):
...     data = cache.load("mykey")
... else:
...     result = expensive_computation()
...     cache.save("mykey", adjacency=result, metadata=metadata)
...     data = {"adjacency": result, "metadata": metadata}

</details>

**Methods:**

Name | Description
---- | -----------
[`clear`](#data-clear) | Clear all cached files in this category.
`delete` | Delete a cached file.
`exists` | Check if a cache key exists.
`get_path` | Get the file path for a cache key.
`list_keys` | List all cached keys in this category.
[`load`](#data-load) | Load cached data.
`save` | Save arrays to cache.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`cache_dir`](#data-cache-dir) |  | 
`category` |  | 



######### Attributes####

(data-cache-dir)=
###### `cache_dir`

```python
cache_dir = get_cache_dir() / category
```

########## `category`

```python
category = category
```



######### Functions####

(data-clear)=
###### `clear`

```python
clear() -> int
```

Clear all cached files in this category.

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of files deleted

########## `delete`

```python
delete(key: str, ext: str = '.npz') -> bool
```

Delete a cached file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>[str](#str)</code> | Cache key | *required*
`ext` | <code>[str](#str)</code> | File extension | <code>'.npz'</code>

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if file was deleted, False if it didn't exist

########## `exists`

```python
exists(key: str, ext: str = '.npz') -> bool
```

Check if a cache key exists.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>[str](#str)</code> | Cache key | *required*
`ext` | <code>[str](#str)</code> | File extension (default: ".npz") | <code>'.npz'</code>

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if cached file exists

########## `get_path`

```python
get_path(key: str, ext: str = '.npz') -> Path
```

Get the file path for a cache key.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>[str](#str)</code> | Cache key | *required*
`ext` | <code>[str](#str)</code> | File extension (default: ".npz") | <code>'.npz'</code>

**Returns:**

Type | Description
---- | -----------
<code>[Path](#pathlib.Path)</code> | Path to the cache file

########## `list_keys`

```python
list_keys(ext: str = '.npz') -> list[str]
```

List all cached keys in this category.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`ext` | <code>[str](#str)</code> | File extension to match | <code>'.npz'</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[str](#str)]</code> | List of cache keys (without extension)

########## `load`

```python
load(key: str) -> dict | None
```

Load cached data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>[str](#str)</code> | Cache key | *required*

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict) \| None</code> | Dictionary of cached arrays, or None if not cached

########## `save`

```python
save(key: str, compressed: bool = True, **arrays: bool) -> Path
```

Save arrays to cache.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>[str](#str)</code> | Cache key | *required*
`compressed` | <code>[bool](#bool)</code> | If True, use compressed npz format (smaller but slower) | <code>True</code>
`**arrays` |  | Named arrays to cache | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[Path](#pathlib.Path)</code> | Path to saved cache file



####### Functions##

(data-clear-cache)=
###### `clear_cache`

```python
clear_cache(category: str | None = None) -> int
```

Clear the nltools cache.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`category` | <code>[str](#str) \| None</code> | If provided, only clear this category. Otherwise clear all. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of files deleted

######## `get_cache_dir`

```python
get_cache_dir() -> Path
```

Get the nltools cache directory.

Returns ~/.nltools/cache, creating it if necessary.

**Returns:**

Type | Description
---- | -----------
<code>[Path](#pathlib.Path)</code> | Path to cache directory

######## `hash_mask`

```python
hash_mask(mask_img: Nifti1Image) -> str
```

Compute a stable hash for a NIfTI mask image.

The hash is based on the mask's shape, affine transformation, and the
actual voxel positions. This ensures that masks with the same shape but
different voxel locations (or different affines) produce different hashes.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask_img` | <code>[Nifti1Image](#nibabel.Nifti1Image)</code> | NIfTI image to hash (typically a binary mask) | *required*

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | 16-character hexadecimal hash string

<details class="example" open markdown="1">
<summary>Example</summary>

>>> import nibabel as nib
>>> mask = nib.load("mask.nii.gz")
>>> hash_mask(mask)
'a1b2c3d4e5f60789'

</details>

###### `io`

BrainData I/O and loading functions.

Standalone functions extracted from BrainData class methods for mask initialization,
data loading (from files, lists, URLs, HDF5, other BrainData objects), resampling,
writing, and uploading.

**Methods:**

Name | Description
---- | -----------
[`check_space_match`](#data-check-space-match) | Check if data and mask are in same space.
`detect_and_update_mask` | Detect best matching template from data and update mask if mask was None.
`detect_space` | Detect if mask is in MNI space or native space.
`get_interpolation` | Get the interpolation method to use for a given image.
`initialize_mask` | Initialize the mask image.
`load_from_brain_data` | Load data from another BrainData object.
`load_from_file` | Load data from file path or nibabel object.
`load_from_h5` | Load data from HDF5 file.
`load_from_list` | Load data from a list of BrainData objects or file paths.
`load_from_url` | Load data from URL.
`mask_images` | Mask a list of space-aligned images with a single fitted masker.
[`resample_to`](#data-resample-to) | Resample BrainData to match target image or resolution.
[`to_nifti`](#data-to-nifti) | Convert BrainData instance to a nibabel NIfTI image.
[`upload_neurovault`](#data-upload-neurovault) | Upload data to NeuroVault.
`warn_if_resampling` | Warn about resampling if verbose=True and resample=True.
`write_brain_data` | Write out BrainData object to Nifti or HDF5 File.



####### Functions##

(data-check-space-match)=
###### `check_space_match`

```python
check_space_match(data_img, mask_img)
```

Check if data and mask are in same space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data_img` |  | nibabel Nifti1Image object | *required*
`mask_img` |  | nibabel Nifti1Image object (mask) | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`bool` |  | True if spaces match (no resampling needed), False otherwise

######## `detect_and_update_mask`

```python
detect_and_update_mask(bd, data_img)
```

Detect best matching template from data and update mask if mask was None.

Also handles resampling if needed based on the resample kwarg.

This function is called during data loading to auto-detect template when mask=None.
After detecting or falling back to a template, it checks if resampling is needed
and resamples the data_img accordingly.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`data_img` |  | nibabel Nifti1Image object from which to detect template | *required*

**Returns:**

Type | Description
---- | -----------
 | nibabel.Nifti1Image: The data_img, possibly resampled to match the mask

######## `detect_space`

```python
detect_space(mask)
```

Detect if mask is in MNI space or native space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask` |  | nibabel Nifti1Image object | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` |  | 'mni' if mask is MNI template, 'native' otherwise

######## `get_interpolation`

```python
get_interpolation(bd, img)
```

Get the interpolation method to use for a given image.

Resolves 'auto' to either 'nearest' or 'continuous' based on data type.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`img` |  | nibabel image to check (used when interpolation='auto') | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` |  | Interpolation method. When 'auto', resolves to 'nearest' or 'continuous' based on data type. Otherwise returns the instance's configured interpolation setting.

######## `initialize_mask`

```python
initialize_mask(bd, mask)
```

Initialize the mask image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`mask` |  | Brain mask as nibabel object, file path, template name string, or None. Template name strings supported: '{res}mm-MNI152-2009{version}' (e.g., '2mm-MNI152-2009c', '3mm-MNI152-2009a', '2mm-MNI152-2009fsl') | *required*

######## `load_from_brain_data`

```python
load_from_brain_data(bd, brain_data, mask = None)
```

Load data from another BrainData object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`brain_data` |  | BrainData object to copy from. | *required*
`mask` |  | Optional mask to use. If None, uses mask from brain_data. | <code>None</code>

######## `load_from_file`

```python
load_from_file(bd, data)
```

Load data from file path or nibabel object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`data` |  | File path or nibabel object. | *required*

######## `load_from_h5`

```python
load_from_h5(bd, file_path, mask)
```

Load data from HDF5 file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`file_path` |  | Path to HDF5 file. | *required*
`mask` |  | User-specified mask (to determine if we should load mask from file). | *required*

######## `load_from_list`

```python
load_from_list(bd, data_list)
```

Load data from a list of BrainData objects or file paths.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`data_list` |  | List of BrainData objects or file paths. | *required*

######## `load_from_url`

```python
load_from_url(bd, url)
```

Load data from URL.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`url` |  | URL to download data from. | *required*

######## `mask_images`

```python
mask_images(mask, imgs)
```

Mask a list of space-aligned images with a single fitted masker.

Validates ``mask`` exactly ONCE — one ``load_mask_img`` — and reuses the
binarized mask across every image in ``imgs``, instead of re-running
nilearn's costly ``load_mask_img`` (binarization checks + ``safe_get_data``,
which each trigger nilearn's forced ``gc.collect``) per image.

``nilearn.masking.apply_mask`` is exactly ``load_mask_img`` (validate) ->
``new_img_like`` (build binary mask) -> ``apply_mask_fmri`` (extract), with
``dtype='f'``, ``smoothing_fwhm=None``, ``ensure_finite=True``. This hoists
the first two out of the per-image loop and calls the lower-level
``apply_mask_fmri`` (which "assumes mask_img contains only two different
values") per image, so the result is byte-equivalent to
``np.vstack([apply_mask(im, mask) for im in imgs])`` for space-aligned data.

Images must already share ``mask``'s space (callers resample first); no
resampling is done here. Falls back to the per-image functional
``apply_mask`` if the fast path raises for any reason.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask` |  | A ``nibabel.Nifti1Image`` boolean/binary mask. | *required*
`imgs` |  | List of space-aligned ``nibabel`` images to mask. | *required*

**Returns:**

Type | Description
---- | -----------
 | ``np.ndarray`` of shape ``(len(imgs), n_voxels)``.

######## `resample_to`

```python
resample_to(bd, *, img = None, resolution = None, interpolation = None)
```

Resample BrainData to match target image or resolution.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`img` |  | Target image for resampling. Can be: - nibabel Nifti1Image object - str/Path to .nii/.nii.gz file - None (if using resolution parameter) | <code>None</code>
`resolution` |  | Target voxel size in mm. Can be: - float/int: Isotropic resolution (e.g., 2.0 = 2mm^3) - None (if using img parameter) | <code>None</code>
`interpolation` |  | Interpolation method for resampling. Can be: - None (default): Uses instance's interpolation setting - 'nearest': Nearest-neighbor (for atlases, masks, labels) - 'linear': Linear interpolation - 'continuous': Higher-order spline (for stat maps) | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | New BrainData instance with resampled data

######## `to_nifti`

```python
to_nifti(bd)
```

Convert BrainData instance to a nibabel NIfTI image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*

**Returns:**

Type | Description
---- | -----------
 | nibabel.Nifti1Image: Brain data in volumetric NIfTI format.

######## `upload_neurovault`

```python
upload_neurovault(bd, *, access_token = None, collection_name = None, collection_id = None, img_type = None, img_modality = None, **kwargs)
```

Upload data to NeuroVault.

Adds any columns in bd.X to image metadata. Index will be used as image name.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`access_token` | <code>[str](#str)</code> | NeuroVault API access token. Required. | <code>None</code>
`collection_name` | <code>[str](#str)</code> | Name of new collection to create. | <code>None</code>
`collection_id` | <code>[int](#int)</code> | NeuroVault collection ID if adding images to an existing collection. | <code>None</code>
`img_type` | <code>[str](#str)</code> | NeuroVault map type. Required. | <code>None</code>
`img_modality` | <code>[str](#str)</code> | NeuroVault image modality. Required. | <code>None</code>
`**kwargs` |  | Additional keyword arguments passed to the NeuroVault API. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | NeuroVault collection information.

######## `warn_if_resampling`

```python
warn_if_resampling(bd, context = '')
```

Warn about resampling if verbose=True and resample=True.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`context` | <code>[str](#str)</code> | Context string to include in warning. Default: empty string. | <code>''</code>

######## `write_brain_data`

```python
write_brain_data(bd, file_name)
```

Write out BrainData object to Nifti or HDF5 File.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`file_name` | <code>[str](#str) or [Path](#pathlib.Path)</code> | Output file path. Supports .nii/.nii.gz (NIfTI) and .h5/.hdf5 (HDF5) formats. | *required*

###### `modeling`

BrainData modeling functions.

Standalone functions extracted from BrainData class methods for model
fitting, GLM estimation, Ridge regression, and contrast computation.

**Methods:**

Name | Description
---- | -----------
[`compute_contrasts`](#data-compute-contrasts) | Compute contrasts from a fitted GLM.
`compute_ridge_cv` | Held-out CV scores under a fixed Ridge α.
[`fit`](#data-fit) | Fit a model to brain imaging data.
`fit_glm` | Fit GLM model and extract results (same logic as current regress()).
`fit_ridge` | Fit Ridge model and extract results.
`parse_contrast_string` | Parse a contrast string into a numeric contrast vector.
`resolve_preprocessing_defaults` | Resolve the ``'auto'`` scale/standardize sentinels to concrete values.
`to_fit_dataclass` | Convert BrainData fit results to Fit dataclass.
[`ttest`](#data-ttest) | One-sample voxelwise t-test across images (axis 0).
[`ttest2`](#data-ttest2) | Two-sample voxelwise t-test between two BrainData stacks.



####### Functions##

###### `compute_contrasts`

```python
compute_contrasts(bd, contrasts, statistic = 't')
```

Compute contrasts from a fitted GLM.

Delegates to the underlying ``nilearn.FirstLevelModel.compute_contrast`` so
t-statistics are computed with the full parameter covariance matrix —
linear-combination-of-stored-betas cannot do this correctly for multi-
regressor contrasts (it would ignore off-diagonal covariance and produce
an effect-size map, not a t-map).

Must be called after ``.fit(model='glm', X=design_matrix)`` has been run.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`contrasts` |  | Can be:<br>- str: a contrast expressed in terms of column names, e.g.   ``"conditionA - conditionB"`` or ``"2*conditionA - conditionB - conditionC"`` - array-like: a numeric contrast vector, one weight per regressor   (e.g. ``[1, -1, 0, 0]``) - dict: ``{name: contrast}`` for multiple contrasts at once | *required*
`statistic` | <code>[str](#str)</code> | Which statistic to return per contrast. One of:<br>- ``"t"`` (default): t-statistic map (for thresholding /   single-subject inference) - ``"z"``: z-score map - ``"p"``: p-value map - ``"beta"`` / ``"effect_size"``: effect-size (β) map — use this   when feeding into a second-level (group) analysis - ``"all"``: a bundle dict ``{"beta", "t", "z", "p", "se"}``   of BrainData maps for this one contrast. One fit, one call,   every view — effect size *and* inferential maps together so   group-level code never has to recompute beta separately. | <code>'t'</code>

**Returns:**

Type | Description
---- | -----------
 | Depends on inputs:<br>- single contrast (str or array) + scalar ``statistic``:   a single BrainData. - single contrast + ``statistic="all"``: a flat dict of five   BrainData keyed by ``"beta"``/``"t"``/``"z"``/``"p"``/``"se"``. - dict of contrasts + scalar ``statistic``: a dict   ``{name: BrainData}``. - dict of contrasts + ``statistic="all"``: a nested dict   ``{name: {"beta", "t", "z", "p", "se"}}``.

**Examples:**

```pycon
>>> data.fit(model="glm", X=dm)
>>> # Single-subject t-map, ready to threshold
>>> tmap = data.compute_contrasts("conditionA - conditionB")
>>> # Effect-size map for use as input to a group-level analysis
>>> beta = data.compute_contrasts(
...     "conditionA - conditionB", statistic="beta"
... )
>>> # Everything at once: threshold on res["t"], feed group on res["beta"]
>>> res = data.compute_contrasts(
...     "conditionA - conditionB", statistic="all"
... )
>>> res["t"].plot(threshold=3.09)
>>> group_effects.append(res["beta"])
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- String contrasts support coefficients: ``"2*A - B"`` or ``"0.5*A + 0.5*B"``.
- Column names must match design matrix columns exactly (case-sensitive).
- For group analysis, stack per-subject effect-size maps
  (``statistic="beta"`` or ``res["beta"]`` from ``statistic="all"``)
  and run a second-level test (e.g. ``BrainData.ttest``). Mixing first-level
  t-maps into a group one-sample test conflates effect magnitude with precision.

</details>

######## `compute_ridge_cv`

```python
compute_ridge_cv(bd, X, cv, alpha = None, backend = 'auto')
```

Held-out CV scores under a fixed Ridge α.

Used only for the *fixed-α* + CV branch — alpha selection is now
handled by ``Ridge.fit`` (which delegates to ``solve_ridge_cv``) and
assembled into ``cv_results_`` by ``_assemble_ridge_cv_results``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`X` | <code>[ndarray](#ndarray)</code> | Training features, shape (n_samples, n_features). | *required*
`cv` | <code>int or sklearn CV splitter</code> | Cross-validation specification. | *required*
`alpha` | <code>[float](#float)</code> | Fixed regularization strength. If None, extracted from ``bd.model_.alpha``. | <code>None</code>
`backend` | <code>[str](#str)</code> | Computational backend ('numpy', 'torch', 'auto'). Default: 'auto' | <code>'auto'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | ``{"scores", "mean_score", "predictions", "folds"}``.

######## `fit`

```python
fit(bd, model = 'glm', *, X = None, cv = None, local_alpha = True, fit_intercept = False, inplace = True, progress_bar = None, scale = 'auto', standardize = 'auto', design_clean = True, design_clean_thresh = 0.95, design_clean_exclude_confounds = False, design_clean_fill_na = 0, **kwargs)
```

Fit a model to brain imaging data.

Creates and fits a model from string specification. The brain data
(bd.data) is always used as the target variable. Model and results
are stored for later use with predict().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`model` | <code>[str](#str)</code> | Model type: 'ridge', 'glm', or future model names | <code>'glm'</code>
`X` | <code>[array](#array) - [like](#like) or [DataFrame](#DataFrame)</code> | Design matrix or feature matrix, shape (n_samples, n_features) - For GLM: Design matrix with regressors (n_samples must match bd.data) - For Ridge: Feature matrix for prediction (n_samples must match bd.data) | <code>None</code>
`cv` | <code>int, 'auto', or sklearn CV splitter</code> | Cross-validation specification (Ridge only): - int: Number of folds for k-fold CV (returns CV scores) - 'auto': Triggers alpha selection via CV (implies alpha='auto') - sklearn CV object: Custom CV splitter (e.g., KFold(3, shuffle=True)) - None: No CV (default, backward compatible) | <code>None</code>
`inplace` | <code>bool, default=True</code> | If True, mutate bd and return bd (backward compatible). If False, return a Fit dataclass with the results. In this case bd's ``.data`` and the result attributes (``ridge_*`` / ``glm_*`` / ``cv_results_``) are left unchanged, but ``bd.model_`` and ``bd.X_`` (plus ``bd.design_matrix`` for GLM) ARE updated on bd so that ``predict()`` / ``compute_contrasts()`` still work off bd. Successive ``inplace=False`` fits therefore overwrite the model used by a later ``bd.predict()``. | <code>True</code>
`progress_bar` | <code>[bool](#bool)</code> | Display progress bar during fitting. - If None: Uses bd.verbose (default) - If True: Shows progress bar for long-running operations - If False: No progress bar | <code>None</code>
`scale` | <code>bool or 'auto', default='auto'</code> | Apply percent-signal-change scaling to the data before fitting, via nilearn's per-voxel ``mean_scaling`` (each voxel's time-series is divided by its own temporal mean, de-meaned, and multiplied by 100). ``'auto'`` resolves to False for both models — PSC is opt-in. Useful for GLM (interpretable % betas); for ridge it is redundant with ``standardize='zscore'`` (a warning is raised for that combination). Applied before ``standardize``. | <code>'auto'</code>
`standardize` | <code>str or None or 'auto', default='auto'</code> | Standardize each voxel across observations after scaling. One of ``'center'`` (subtract the mean), ``'zscore'`` (subtract mean, divide by std), or ``None`` (off). ``'auto'`` resolves to ``'zscore'`` for ``model='ridge'`` (so a shared alpha regularizes voxels fairly) and ``None`` for ``model='glm'``. | <code>'auto'</code>
`design_clean` | <code>bool, default=True</code> | GLM only. If True, run ``DesignMatrix.clean()`` on ``X`` before fitting to drop highly correlated regressors. Coerces ``X`` to ``DesignMatrix`` if needed. Ignored when ``model='ridge'``. | <code>True</code>
`design_clean_thresh` | <code>float, default=0.95</code> | GLM only. Correlation threshold passed to ``DesignMatrix.clean()`` (drops if ``abs(r) >= thresh``). Ignored when ``model='ridge'``. | <code>0.95</code>
`design_clean_exclude_confounds` | <code>bool, default=False</code> | GLM only. If True, ``DesignMatrix.clean()`` skips confound columns when checking correlations. Ignored when ``model='ridge'``. | <code>False</code>
`design_clean_fill_na` | <code>int, float, or None, default=0</code> | GLM only. Fill value for NaNs before correlation check in ``DesignMatrix.clean()``. Ignored when ``model='ridge'``. | <code>0</code>
`**kwargs` | <code>[dict](#dict)</code> | Additional arguments passed to model constructor - Ridge: alpha, alphas, backend, random_state - Glm: noise_model, minimize_memory, etc. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or Fit: If ``inplace=True``, returns bd (fitted BrainData). If ``inplace=False``, returns Fit dataclass with results.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`model_` | <code>[BaseModel](#BaseModel)</code> | Fitted model instance (Ridge, Glm, etc.). Set on bd when ``inplace=True``.
`X_` | <code>[ndarray](#ndarray)</code> | Training data X, stored for predict() default.
`cv_results_` | <code>[dict](#dict)</code> | Cross-validation results dict with keys 'scores', 'mean_score', 'predictions', 'folds', 'best_alpha', 'alpha_scores' (if cv is not None).
`glm_betas` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Beta coefficients (for model='glm')
`glm_t` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | T-statistics (for model='glm')
`glm_p` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | P-values (for model='glm')
`glm_se` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Standard errors (for model='glm')
`glm_residual` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Residuals (for model='glm')
`glm_predicted` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Fitted values (for model='glm')
`glm_r2` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | R-squared values (for model='glm')
`ridge_weights` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Model coefficients (for model='ridge')
`ridge_fitted_values` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Fitted values (for model='ridge')
`ridge_scores` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | R-squared scores (for model='ridge')

**Examples:**

```pycon
>>> # Old behavior (backward compatible): mutate self
>>> brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
>>> print(f"CV R2: {brain_data.cv_results_['mean_score'].mean():.3f}")
>>> weights = brain_data.ridge_weights  # Access as attribute
>>>
>>> # New behavior: return Fit dataclass (result attrs / data unchanged)
>>> fit = brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features, inplace=False)
>>> assert isinstance(fit, Fit)
>>> assert 'weights' in fit.available()
>>> assert not hasattr(brain_data, 'ridge_weights')  # result attrs not set
>>> # (model_/X_ ARE updated on brain_data so predict() works)
>>> print(f"CV R2: {fit.cv_mean_score.mean():.3f}")
>>>
>>> # GLM with Fit dataclass
>>> fit_glm = brain_data.fit(model='glm', X=design_matrix, inplace=False)
>>> assert 'betas' in fit_glm.available()
>>> assert 't_stats' in fit_glm.available()
```

######## `fit_glm`

```python
fit_glm(bd, X)
```

Fit GLM model and extract results (same logic as current regress()).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`X` |  | Design matrix (DataFrame or DesignMatrix). | *required*

<details class="note" open markdown="1">
<summary>Note</summary>

Sets glm_betas, glm_t, glm_p, glm_se, glm_residual, glm_predicted,
glm_r2, and design_matrix on bd.

</details>

######## `fit_ridge`

```python
fit_ridge(bd, X, cv = None, **kwargs)
```

Fit Ridge model and extract results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`X` | <code>[ndarray](#ndarray)</code> | Training features | *required*
`cv` | <code>int, 'auto', or sklearn CV splitter</code> | Cross-validation specification | <code>None</code>
`**kwargs` | <code>[dict](#dict)</code> | Additional arguments for CV (alpha, alphas, backend, etc.) | <code>{}</code>

<details class="note" open markdown="1">
<summary>Note</summary>

Sets ridge_weights, ridge_fitted_values, ridge_scores, and
cv_results_ (if cv provided) on bd.

</details>

######## `parse_contrast_string`

```python
parse_contrast_string(bd, contrast_str)
```

Parse a contrast string into a numeric contrast vector.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`contrast_str` | <code>[str](#str)</code> | Contrast string like "A - B" or "2*A - B - C" | *required*

**Returns:**

Type | Description
---- | -----------
 | np.array: Numeric contrast vector

######## `resolve_preprocessing_defaults`

```python
resolve_preprocessing_defaults(model, scale, standardize)
```

Resolve the ``'auto'`` scale/standardize sentinels to concrete values.

Single source of truth shared by ``BrainData.fit`` and ``BrainCollection.fit``
so both facades agree on per-model defaults. ``scale`` (percent-signal-change)
is opt-in for both models. Ridge standardizes its targets by default so a
shared alpha regularizes voxels fairly; GLM does neither so betas stay in
native units.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model` | <code>[str](#str)</code> | ``'ridge'`` or ``'glm'``. | *required*
`scale` | <code>[bool](#bool) or [auto](#auto)</code> | Requested scale flag. | *required*
`standardize` | <code>str, None, or 'auto'</code> | Requested standardize method. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` |  | ``(scale, standardize)`` with any ``'auto'`` resolved.

######## `to_fit_dataclass`

```python
to_fit_dataclass(bd, model)
```

Convert BrainData fit results to Fit dataclass.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`model` | <code>[str](#str)</code> | Model type ('ridge' or 'glm') | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Fit` |  | Dataclass containing fit results

######## `ttest`

```python
ttest(bd, *, popmean = 0.0, permutation = False, n_permute = 5000, tail = 2, return_null = False, n_jobs = -1, random_state = None)
```

One-sample voxelwise t-test across images (axis 0).

For a BrainData stack of images (e.g. subject-level contrast maps with
shape ``(n_samples, n_voxels)``), test whether the per-voxel mean differs
from ``popmean``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance (must contain multiple images). | *required*
`popmean` |  | Population mean to test against. Default 0.0. | <code>0.0</code>
`permutation` |  | If True, use sign-flip permutation test via ``nltools.stats.one_sample_permutation_test``; the p-values come from the empirical null and the parametric t-statistic is still reported alongside for reference. | <code>False</code>
`n_permute` |  | Number of permutations (used only when ``permutation=True``). Default 5000. | <code>5000</code>
`tail` |  | Tail of the test (1 or 2). Default 2. | <code>2</code>
`return_null` |  | If True, also return the null distribution. Default False. | <code>False</code>
`n_jobs` |  | Number of parallel jobs. Default -1 (all cores). | <code>-1</code>
`random_state` |  | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | dict with four BrainData keys:<br>- ``"mean"``: voxelwise mean across images (effect-size estimate). - ``"t"``: parametric one-sample t-statistic. - ``"z"``: signed z-score, ``sign(t) * norm.isf(p/2)``, matching   nilearn's ``output_type='z_score'``. Useful for thresholding   on z at small df where t tails are heavier than normal. - ``"p"``: p-value (parametric, or permutation-based when   ``permutation=True``).
 | The effect size is always returned alongside the inferential maps so
 | group-level code never has to compute the mean separately.

######## `ttest2`

```python
ttest2(bd, other, equal_var = True)
```

Two-sample voxelwise t-test between two BrainData stacks.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | First BrainData (shape ``(n1, n_voxels)``). | *required*
`other` |  | Second BrainData (shape ``(n2, n_voxels)``). | *required*
`equal_var` |  | If True (default), standard two-sample t-test. If False, Welch's t-test. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | ``{"t": BrainData, "p": BrainData}``.

(data-neighborhoods)=
###### `neighborhoods`

Spatial neighborhood computation for neuroimaging analyses.

This module provides efficient computation and caching of spatial neighborhoods
(spheres) around brain voxels. It is designed to support searchlight analyses,
ISC, and other operations that require iterating over local brain regions.

The key insight is that for a given mask and radius, the neighborhood structure
is deterministic and can be cached for reuse across analyses.

<details class="example" open markdown="1">
<summary>Example</summary>

>>> import nibabel as nib
>>> from nltools.data.braindata.neighborhoods import compute_searchlight_neighborhoods
>>>
>>> mask = nib.load("mask.nii.gz")
>>> neighborhoods = compute_searchlight_neighborhoods(mask, radius_mm=10.0)
>>>
>>> # Iterate over all voxels and their neighborhoods
>>> for center_idx, neighbor_indices in neighborhoods.iter_neighborhoods():
...     # Extract data for these voxels
...     local_data = data[:, neighbor_indices]
...     result[center_idx] = analyze(local_data)

</details>

**Classes:**

Name | Description
---- | -----------
[`SphereNeighborhoods`](#data-sphereneighborhoods) | Precomputed sphere neighborhoods for a brain mask.

**Methods:**

Name | Description
---- | -----------
[`compute_searchlight_neighborhoods`](#data-compute-searchlight-neighborhoods) | Compute sphere neighborhoods for all voxels in a brain mask.



####### Classes##

(data-sphereneighborhoods)=
###### `SphereNeighborhoods`

```python
SphereNeighborhoods(adjacency: sparse.csr_matrix, mask_hash: str, radius_mm: float, n_voxels: int) -> None
```

Precomputed sphere neighborhoods for a brain mask.

This dataclass stores a sparse adjacency matrix where row i contains True
for all voxels within the specified radius of voxel i. It provides efficient
iteration over neighborhoods for searchlight-style analyses.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`adjacency`](#data-adjacency) | <code>[csr_matrix](#scipy.sparse.csr_matrix)</code> | Sparse CSR matrix (n_voxels, n_voxels) where adjacency[i, j] is True if voxel j is within radius of voxel i
`mask_hash` | <code>[str](#str)</code> | Hash of the source mask for validation
`radius_mm` | <code>[float](#float)</code> | Radius in millimeters
`n_voxels` | <code>[int](#int)</code> | Number of voxels in the mask

<details class="example" open markdown="1">
<summary>Example</summary>

>>> neighborhoods = compute_searchlight_neighborhoods(mask, radius_mm=10.0)
>>> print(f"Mean neighborhood size: {neighborhoods.mean_size:.1f} voxels")
>>>
>>> # Get neighbors of a specific voxel
>>> neighbor_idx = neighborhoods.get_neighbors(100)
>>> print(f"Voxel 100 has {len(neighbor_idx)} neighbors")

</details>

**Methods:**

Name | Description
---- | -----------
[`get_neighborhood_size`](#data-get-neighborhood-size) | Get the number of voxels in a neighborhood.
`get_neighbors` | Get indices of all voxels in the neighborhood of a given voxel.
`iter_neighborhoods` | Iterate over all neighborhoods.



######### Attributes####

###### `adjacency`

```python
adjacency: sparse.csr_matrix
```

########## `mask_hash`

```python
mask_hash: str
```

########## `max_size`

```python
max_size: int
```

Maximum neighborhood size.

########## `mean_size`

```python
mean_size: float
```

Mean neighborhood size in voxels.

########## `min_size`

```python
min_size: int
```

Minimum neighborhood size.

########## `n_voxels`

```python
n_voxels: int
```

########## `radius_mm`

```python
radius_mm: float
```



######### Functions####

(data-get-neighborhood-size)=
###### `get_neighborhood_size`

```python
get_neighborhood_size(voxel_idx: int) -> int
```

Get the number of voxels in a neighborhood.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`voxel_idx` | <code>[int](#int)</code> | Index of the center voxel | *required*

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of voxels in the neighborhood

########## `get_neighbors`

```python
get_neighbors(voxel_idx: int) -> np.ndarray
```

Get indices of all voxels in the neighborhood of a given voxel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`voxel_idx` | <code>[int](#int)</code> | Index of the center voxel (0 to n_voxels-1) | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Array of voxel indices within radius of the center voxel

########## `iter_neighborhoods`

```python
iter_neighborhoods(progress_bar: bool = False) -> Iterator[tuple[int, np.ndarray]]
```

Iterate over all neighborhoods.

**Yields:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[int](#int), [ndarray](#numpy.ndarray)]</code> | Tuple of (center_voxel_idx, neighbor_indices) for each voxel

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`progress_bar` | <code>[bool](#bool)</code> | If True, wrap iterator with tqdm progress bar | <code>False</code>



####### Functions##

(data-compute-searchlight-neighborhoods)=
###### `compute_searchlight_neighborhoods`

```python
compute_searchlight_neighborhoods(mask_img: Nifti1Image, radius_mm: float = 10.0, use_cache: bool = True) -> SphereNeighborhoods
```

Compute sphere neighborhoods for all voxels in a brain mask.

For each voxel in the mask, this function identifies all other voxels
within the specified radius (in millimeters). The result is cached to
disk for fast reloading in subsequent analyses.

The algorithm uses sklearn's BallTree for efficient radius queries in
world coordinates (mm), ensuring accurate neighborhoods regardless of
voxel resolution.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask_img` | <code>[Nifti1Image](#nibabel.Nifti1Image)</code> | NIfTI mask image defining the brain region | *required*
`radius_mm` | <code>[float](#float)</code> | Radius of spheres in millimeters (default: 10.0) | <code>10.0</code>
`use_cache` | <code>[bool](#bool)</code> | If True, cache results to ~/.nltools/cache/searchlight/ for fast reloading (default: True) | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[SphereNeighborhoods](#nltools.data.braindata.neighborhoods.SphereNeighborhoods)</code> | SphereNeighborhoods with precomputed adjacency matrix

<details class="example" open markdown="1">
<summary>Example</summary>

>>> import nibabel as nib
>>> mask = nib.load("brain_mask.nii.gz")
>>>
>>> # First call computes and caches (may take a few seconds)
>>> neighborhoods = compute_searchlight_neighborhoods(mask, radius_mm=8.0)
>>>
>>> # Subsequent calls load from cache (~50ms)
>>> neighborhoods = compute_searchlight_neighborhoods(mask, radius_mm=8.0)
>>>
>>> print(neighborhoods)
SphereNeighborhoods(n_voxels=50000, radius=8.0mm, mean_size=33.2)

</details>

<details class="notes" open markdown="1">
<summary>Notes</summary>

Cache location: ~/.nltools/cache/searchlight/{mask_hash}_{radius}mm.npz

For a typical 2mm MNI mask (~50k voxels) with 10mm radius:
- First run: ~1-2 seconds
- Cached load: ~50ms

</details>

###### `plotting`

BrainData plotting functions.

**Methods:**

Name | Description
---- | -----------
[`auto_select_colormap`](#data-auto-select-colormap) | Auto-select colormap based on data characteristics.
`plot_brain` | Plot BrainData instance using nilearn visualization or matplotlib.
`plot_flatmap_brain` | Plot brain data on cortical flatmap.
`prepare_save_paths` | Prepare save paths for multiple plot outputs.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`DEFAULT_SLICE_CUT_COORDS`](#data-default-slice-cut-coords) |  | 



####### Attributes##

(data-default-slice-cut-coords)=
###### `DEFAULT_SLICE_CUT_COORDS`

```python
DEFAULT_SLICE_CUT_COORDS = {'x': list(range(-50, 51, 8)), 'y': list(range(-80, 50, 10)), 'z': list(range(-40, 71, 9))}
```



####### Functions##

(data-auto-select-colormap)=
###### `auto_select_colormap`

```python
auto_select_colormap(data)
```

Auto-select colormap based on data characteristics.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | numpy array of brain data | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` |  | Colormap name

######## `plot_brain`

```python
plot_brain(bd, *, method = 'glass', upper = None, lower = None, threshold = None, view = 'z', cut_coords = None, cmap = None, bg_img = None, ax = None, figsize = (8, 6), title = None, colorbar = True, save = None, stat = 'mean', limit = 3, **kwargs)
```

Plot BrainData instance using nilearn visualization or matplotlib.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`method` | <code>[str](#str)</code> | Visualization type ('glass', 'slices', 'timeseries', 'histogram'). | <code>'glass'</code>
`upper` | <code>[str](#str) / [float](#float)</code> | Upper threshold applied to the data (nltools semantics; may be a percentile string like ``"95%"``). | <code>None</code>
`lower` | <code>[str](#str) / [float](#float)</code> | Lower threshold applied to the data (nltools semantics). | <code>None</code>
`threshold` | <code>[float](#float)</code> | Absolute-value transparency cutoff forwarded to the underlying nilearn plot function. Voxels with ``|value| < threshold`` are rendered transparent. Must be >= 0. Use ``upper``/``lower`` for one-sided data thresholding. | <code>None</code>
`view` | <code>[str](#str)</code> | For ``method="slices"``, any non-empty combination of ``"x"``, ``"y"``, ``"z"`` (e.g. ``"xyz"``, ``"xz"``, ``"y"``). Default: ``"z"``. | <code>'z'</code>
`cut_coords` | <code>[list](#list) or [dict](#dict)</code> | Cut coordinates for multi-slice views. If provided, takes precedence over ``view``-based defaults. Either a list of per-axis coordinate sequences whose length matches ``view``, or a dict keyed by axis letter (``{"x": [...], "z": [...]}``) from which entries for each axis in ``view`` are looked up. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Colormap name. | <code>None</code>
`bg_img` | <code>[Nifti1Image](#Nifti1Image) or [str](#str)</code> | Background image for slice views. | <code>None</code>
`ax` | <code>[Axes](#matplotlib.axes.Axes)</code> | Matplotlib axis to plot on. | <code>None</code>
`figsize` | <code>[tuple](#tuple)</code> | default figure size if no axis (8, 6) | <code>(8, 6)</code>
`title` | <code>[str](#str)</code> | Plot title. | <code>None</code>
`colorbar` | <code>[bool](#bool)</code> | Whether to show colorbar. Default: True. | <code>True</code>
`save` | <code>[str](#str)</code> | Path to save figure(s). | <code>None</code>
`stat` | <code>[str](#str)</code> | Statistic for timeseries plots. Valid options: 'mean', 'median', 'std'. | <code>'mean'</code>
`limit` | <code>[int](#int)</code> | Maximum number of images to render when ``bd`` contains multiple maps and ``method`` is ``"glass"`` or ``"slices"``. Default: 3. A warning is emitted if the data has more images than ``limit``. Ignored for single-image data and for matplotlib-based methods (``"timeseries"``, ``"histogram"``), which already aggregate across images. | <code>3</code>
`**kwargs` |  | Additional arguments passed to nilearn plot functions. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure or list[matplotlib.figure.Figure]: For
 | single-image data, the figure object (last one created if
 | ``method="slices"`` produced multiple per-axis figures). For
 | multi-image data with ``method`` in ``{"glass", "slices"}``, a list
 | of figures (one per image for glass; one per image-and-view pair for
 | slices). All figures auto-display in notebooks.

######## `plot_flatmap_brain`

```python
plot_flatmap_brain(bd, *, threshold = None, cmap = 'RdBu_r', vmax = None, vmin = None, template = 'fsaverage5', with_curvature = True, curvature_contrast = 0.5, curvature_brightness = 0.5, transparency = 'auto', colorbar = True, colorbar_orientation = 'horizontal', figsize = (12, 6), title = None, radius_mm = 3.0, interpolation = 'linear', axes = None, save = None)
```

Plot brain data on cortical flatmap.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`threshold` | <code>[float](#float)</code> | Values below this absolute threshold are masked. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Matplotlib colormap for data. Default: 'RdBu_r'. | <code>'RdBu_r'</code>
`vmax` | <code>[float](#float)</code> | Maximum value for colormap. | <code>None</code>
`vmin` | <code>[float](#float)</code> | Minimum value for colormap. | <code>None</code>
`template` | <code>[str](#str)</code> | fsaverage resolution. Default: 'fsaverage5'. | <code>'fsaverage5'</code>
`with_curvature` | <code>[bool](#bool)</code> | Show sulcal/gyral pattern. Default: True. | <code>True</code>
`curvature_contrast` | <code>[float](#float)</code> | Contrast of curvature. Default: 0.5. | <code>0.5</code>
`curvature_brightness` | <code>[float](#float)</code> | Mean brightness of curvature. Default: 0.5. | <code>0.5</code>
`colorbar` | <code>[bool](#bool)</code> | Show colorbar. Default: True. | <code>True</code>
`colorbar_orientation` | <code>[str](#str)</code> | 'horizontal' or 'vertical'. Default: 'horizontal'. | <code>'horizontal'</code>
`figsize` | <code>[tuple](#tuple)</code> | Figure size. Default: (12, 6). | <code>(12, 6)</code>
`title` | <code>[str](#str)</code> | Figure title. | <code>None</code>
`radius_mm` | <code>[float](#float)</code> | sampling radius in mm for vol_to_surf. Default: 3.0. | <code>3.0</code>
`interpolation` | <code>[str](#str)</code> | Interpolation for vol_to_surf. Default: 'linear'. | <code>'linear'</code>
`axes` | <code>[Axes](#matplotlib.axes.Axes)</code> | Existing axes to plot on. | <code>None</code>
`save` | <code>[str](#str)</code> | File path to save figure. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

######## `prepare_save_paths`

```python
prepare_save_paths(save, idx = None)
```

Prepare save paths for multiple plot outputs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`save` |  | Base save path (str or Path) | *required*
`idx` | <code>[int](#int)</code> | Image index appended as ``_img{idx}`` to the base filename. Used to disambiguate saves across multiple images. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | Dictionary with 'glass' and 'slices' keys containing save paths

(data-prediction)=
###### `prediction`

BrainData prediction — timeseries (encoding) and MVPA (decoding).

Single entry point: `predict`. Returns `Predict`
with fields populated based on dispatch. Mirrors `BrainData.fit` /
`Fit` patterns: frozen result dataclass, ``inplace=True`` mutates
self with attributes, ``inplace=False`` returns the dataclass.

**Methods:**

Name | Description
---- | -----------
[`build_pipeline`](#data-build-pipeline) | Build a per-fold scikit-learn preprocessing and model pipeline.
[`predict`](#data-predict) | Dispatch BrainData prediction to timeseries encoding or MVPA decoding.
`predict_mvpa` | Cross-validated decoding. Returns Predict (or self if inplace=True).
`predict_timeseries` | Predict voxel timeseries from a fitted encoding model.
`resolve_model` | Resolve a string shortcut or pass through a sklearn estimator.
`resolve_scoring` | Resolve scoring='auto' to 'accuracy' (classifier) or 'r2' (regressor).

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`VALID_SPATIAL_SCALES`](#data-valid-spatial-scales) |  | 



####### Attributes##

(data-valid-spatial-scales)=
###### `VALID_SPATIAL_SCALES`

```python
VALID_SPATIAL_SCALES = {'whole_brain', 'searchlight', 'roi'}
```



####### Classes

####### Functions##

(data-build-pipeline)=
###### `build_pipeline`

```python
build_pipeline(model, standardize: bool, reduce: str | None, n_components: str | None)
```

Build a per-fold scikit-learn preprocessing and model pipeline.

The pipeline contains an optional StandardScaler, optional PCA, and the
model. If only the model is needed, returns the model itself.

######## `predict`

```python
predict(bd, *, y = None, X = None, spatial_scale: str = 'whole_brain', model: Any = 'svm', cv: int = 5, standardize: bool = True, reduce: str | None = None, n_components: int | None = None, scoring: str = 'auto', groups: str = None, roi_mask: str = None, radius_mm: float = 10.0, inplace: bool = False, n_jobs: int = 1, random_state: int | None = None, progress_bar: bool = False)
```

Dispatch BrainData prediction to timeseries encoding or MVPA decoding.

Implements `BrainData.predict`. See the class docstring for full parameter
documentation.

######## `predict_mvpa`

```python
predict_mvpa(bd, *, y, spatial_scale: str, model: Any, cv: Any, standardize: bool, reduce: str | None, n_components: int | None, scoring: str, groups: str, roi_mask: str, radius_mm: float, inplace: bool, n_jobs: int, random_state: int | None = None, progress_bar: bool = False) -> Predict | Any
```

Cross-validated decoding. Returns Predict (or self if inplace=True).

######## `predict_timeseries`

```python
predict_timeseries(bd, *, X = None)
```

Predict voxel timeseries from a fitted encoding model.

Returns a fresh ``BrainData`` whose ``.data`` is the predicted timeseries.
Encoding model prediction yields a brain image — the natural container is
``BrainData``, so it composes directly with downstream methods (`.plot()`,
`.standardize()`, etc.). MVPA decoding (``y=`` mode) returns ``Predict``.

######## `resolve_model`

```python
resolve_model(model: Any)
```

Resolve a string shortcut or pass through a sklearn estimator.

######## `resolve_scoring`

```python
resolve_scoring(scoring: str, classifier: bool) -> str
```

Resolve scoring='auto' to 'accuracy' (classifier) or 'r2' (regressor).

###### `utils`

Shared helpers for BrainData submodules.

These are internal utilities used by the facade and submodules — not part of the
public API.

**Methods:**

Name | Description
---- | -----------
[`apply_func`](#data-apply-func) | Apply a statistical function to BrainData's ``.data`` attribute.
`check_brain_data` | Return *data* as a BrainData, coercing Niimg-like inputs if needed.
`check_brain_data_is_single` | Logical test if BrainData instance is a single image.
`perform_arithmetic` | Perform an arithmetic operation with validation.
`shallow_copy` | Create a shallow copy of a BrainData for efficient method chaining.



####### Functions##

(data-apply-func)=
###### `apply_func`

```python
apply_func(bd, stat_func, axis = 0)
```

Apply a statistical function to BrainData's ``.data`` attribute.

If *axis* is 0, returns a BrainData with the statistic computed across
samples (e.g. within a voxel over time).  If *axis* is 1, returns a numpy
array with the statistic computed across features (e.g. across voxels
within a single time-point).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`stat_func` |  | Callable accepting an array and an ``axis`` kwarg. | *required*
`axis` |  | 0 = across images, 1 = within images. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float | np.ndarray | BrainData

######## `check_brain_data`

```python
check_brain_data(data, mask = None)
```

Return *data* as a BrainData, coercing Niimg-like inputs if needed.

If *data* is already a BrainData, the optional *mask* is applied via
`BrainData.apply_mask`.  Otherwise *data* is passed through
`BrainData`, which dispatches on type (file path, list of paths,
URL, h5, ``nib.Nifti1Image``).  Unsupported types raise ``TypeError`` from
`validate_data_type`.

######## `check_brain_data_is_single`

```python
check_brain_data_is_single(data)
```

Logical test if BrainData instance is a single image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | brain data | *required*

**Returns:**

Type | Description
---- | -----------
 | (bool)

######## `perform_arithmetic`

```python
perform_arithmetic(bd, other, operation, operation_name, inplace = False, reverse = False)
```

Perform an arithmetic operation with validation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance (left operand unless *reverse* is True). | *required*
`other` |  | The other operand (scalar, BrainData, or array). | *required*
`operation` |  | Numpy ufunc (e.g. ``np.add``, ``np.subtract``). | *required*
`operation_name` |  | Human-readable name for error messages. | *required*
`inplace` |  | If True, mutate *bd* in place. | <code>False</code>
`reverse` |  | If True, reverse operand order (for ``__rsub__`` etc.). | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Result of the operation.

######## `shallow_copy`

```python
shallow_copy(bd)
```

Create a shallow copy of a BrainData for efficient method chaining.

Creates a new BrainData instance that shares immutable objects (mask)
but copies mutable attributes.  The data array is NOT copied — callers
should handle data copying as needed.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance to copy. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | New instance with shared/copied attributes.

(data-validation)=
###### `validation`

Validation utilities for BrainData class.

This module contains helper functions for validating inputs, shapes, and
compatibility between BrainData objects and other data types.

**Methods:**

Name | Description
---- | -----------
[`validate_append_shapes`](#data-validate-append-shapes) | Validate shape compatibility for appending BrainData objects.
`validate_arithmetic_operand` | Validate operand type for arithmetic operations.
`validate_brain_data_shapes` | Validate shape compatibility between two BrainData objects.
`validate_data_type` | Validate input data type for BrainData initialization.
`validate_frame` | Validate and process X or Y dataframes for BrainData.
`validate_index_operations` | Validate indexing operations for BrainData.
`validate_list_data` | Validate that all items in a list are the same type.



####### Functions##

(data-validate-append-shapes)=
###### `validate_append_shapes`

```python
validate_append_shapes(data1_shape, data2_shape)
```

Validate shape compatibility for appending BrainData objects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1_shape` |  | Shape of first BrainData. | *required*
`data2_shape` |  | Shape of second BrainData to append. | *required*

######## `validate_arithmetic_operand`

```python
validate_arithmetic_operand(other, operation_name)
```

Validate operand type for arithmetic operations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` |  | The operand to validate. | *required*
`operation_name` |  | Name of operation (e.g., 'add', 'multiply'). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` |  | Type of operand ('scalar', 'brain_data', or 'array').

######## `validate_brain_data_shapes`

```python
validate_brain_data_shapes(brain1, brain2, operation = 'operation')
```

Validate shape compatibility between two BrainData objects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain1` |  | First BrainData object. | *required*
`brain2` |  | Second BrainData object. | *required*
`operation` |  | Name of operation for error messages. | <code>'operation'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` |  | (brain1_is_single, brain2_is_single) booleans.

######## `validate_data_type`

```python
validate_data_type(data)
```

Validate input data type for BrainData initialization.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | Input data to validate. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` |  | Type of data ('brain_data', 'list', 'h5', 'url', 'file', 'nibabel', 'none').

######## `validate_frame`

```python
validate_frame(frame, data_shape = None, frame_type = 'DataFrame')
```

Validate and process X or Y dataframes for BrainData.

Accepts pandas DataFrames for user convenience but always returns a
polars DataFrame. Internal BrainData state should be polars-only.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`frame` |  | Input to validate. Can be ``None``, a ``str``/``Path`` pointing to a CSV, a polars or pandas DataFrame, or a 1D/2D numpy array. | *required*
`data_shape` |  | Optional tuple of data shape to validate row count against. | <code>None</code>
`frame_type` |  | Type of frame for error messages (e.g., "X", "Y"). | <code>'DataFrame'</code>

**Returns:**

Type | Description
---- | -----------
 | pl.DataFrame: Validated frame as polars. Empty ``pl.DataFrame()`` when
 | ``frame`` is ``None``.

######## `validate_index_operations`

```python
validate_index_operations(data_shape, index)
```

Validate indexing operations for BrainData.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data_shape` |  | Shape of the data array. | *required*
`index` |  | Index to validate. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` |  | Type of indexing ('single', 'slice', 'array').

######## `validate_list_data`

```python
validate_list_data(data_list)
```

Validate that all items in a list are the same type.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data_list` |  | List to validate. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` |  | Type of items ('brain_data' or 'file').

(data-viewer)=
###### `viewer`

ipyniivue (niivue) interactive viewer for BrainData.

`build_viewer` returns a configured `NiiVue` — a
WebGL brain viewer with live windowing (right-drag), slice scrolling,
native 4D frame scrubbing, true 3D rendering, and optional nltools-atlas
overlays (colored regions / outlines / hover labels). Live-kernel only
(Jupyter, marimo); static docs keep using `BrainData.plot`.

The module is split functional-core / imperative-shell:

- Pure helpers (`resolve_cmap`, `divergent_partner`,
  `slice_type_for`, `qualitative_colors`,
  `atlas_to_label_lut`, `resolve_background`,
  `bd_to_volume`) translate BrainData / `Atlas` state into the
  vocabulary niivue understands.
- `build_viewer` is the thin assembler that constructs the widget,
  loads the volume stack, and applies the display settings.

niivue formatting deliberately lives here, not in
``nltools/data/atlases/`` — the atlas package stays niivue-agnostic and
only exposes the generic `Atlas` dataclass.

**Methods:**

Name | Description
---- | -----------
[`atlas_to_label_lut`](#data-atlas-to-label-lut) | Build a niivue integer-indexed label LUT from a deterministic atlas.
`bd_to_volume` | Build a niivue `Volume` from a BrainData stat map.
`build_controls` | Wrap a `NiiVue` in a `VBox` with a live threshold slider.
`build_viewer` | Assemble a configured `NiiVue` for a BrainData.
`divergent_partner` | Return the ``colormap_negative`` partner for a positive colormap.
`qualitative_colors` | Deterministic qualitative RGB palette of length ``n``.
`resolve_background` | Resolve the ``bg_img`` argument to a background-image path or ``None``.
`resolve_cmap` | Resolve a colormap name to a valid niivue colormap.
`slice_type_for` | Map a ``view`` string to a niivue `SliceType`.
`threshold_slider_bounds` | Compute ``(min, max, value_low, value_high, step)`` for a threshold slider.



####### Classes

####### Functions##

(data-atlas-to-label-lut)=
###### `atlas_to_label_lut`

```python
atlas_to_label_lut(atlas: Atlas) -> dict
```

Build a niivue integer-indexed label LUT from a deterministic atlas.

The LUT arrays are dense (length ``max_index + 1``) because niivue
indexes them by integer voxel value. Index 0 and any gap indices are
transparent (``A=0``, empty label); each present region gets a color
from `qualitative_colors` (assigned in table-enumeration order, so
colors stay stable under sparse / non-contiguous indices) and its name.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`atlas` | <code>[Atlas](#nltools.data.atlases.Atlas)</code> | A loaded deterministic `Atlas`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | ``{"R", "G", "B", "A", "labels"}`` dict suitable for
<code>[dict](#dict)</code> | `set_colormap_label`.

######## `bd_to_volume`

```python
bd_to_volume(bd, *, name: str, cmap: str, cmap_negative: str, cal_min: float | None, cal_max: float | None, opacity: float) -> Volume
```

Build a niivue `Volume` from a BrainData stat map.

The 3D (``bd[0]``) or 4D (a stack) image is loaded **once** as a single
volume — niivue scrubs 4D frames natively, so there is no per-frame
re-render. Thresholding is the divergent magnitude window
``[cal_min, cal_max]``: the positive side uses ``cmap``, the negative
side mirrors it via ``cmap_negative``. ``cal_min`` is the display floor;
because niivue's overlay colormaps ramp alpha to 0 at the floor,
sub-floor voxels render transparent (true thresholding).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | A BrainData (3D for a single map, 4D for a stack). | *required*
`name` | <code>[str](#str)</code> | Volume name (shown in the colorbar / legend). | *required*
`cmap` | <code>[str](#str)</code> | Resolved niivue positive colormap. | *required*
`cmap_negative` | <code>[str](#str)</code> | niivue colormap for negative values. | *required*
`cal_min` | <code>[float](#float) \| None</code> | Window floor, or ``None`` for niivue auto. | *required*
`cal_max` | <code>[float](#float) \| None</code> | Window ceiling, or ``None`` for niivue auto. | *required*
`opacity` | <code>[float](#float)</code> | Volume opacity in ``0..1``. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Volume](#ipyniivue.Volume)</code> | A configured `Volume`.

######## `build_controls`

```python
build_controls(nv: NiiVue, bd: NiiVue, *, cal_min: float | None, cal_max: float | None)
```

Wrap a `NiiVue` in a `VBox` with a live threshold slider.

The returned container stacks a `FloatRangeSlider` above the viewer; the
slider drives the stat-map volume's ``cal_min``/``cal_max`` window (the
discoverable in-notebook equivalent of niivue's right-drag windowing).
The niivue widget is exposed as ``.viewer`` and the slider as
``.threshold_slider`` for programmatic access.

When the caller passed no ``cal_min``/``cal_max``, the volume window is
left on niivue auto until the slider is first moved, so the default
display matches ``controls=False``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`nv` | <code>[NiiVue](#ipyniivue.NiiVue)</code> | The assembled viewer (from `build_viewer`). | *required*
`bd` |  | The BrainData being viewed (source of the slider range). | *required*
`cal_min` | <code>[float](#float) \| None</code> | Window floor the caller requested, or ``None`` for auto. | *required*
`cal_max` | <code>[float](#float) \| None</code> | Window ceiling the caller requested, or ``None`` for auto. | *required*

**Returns:**

Type | Description
---- | -----------
 | An `ipywidgets.VBox` with ``.viewer`` and ``.threshold_slider`` set.

<details class="note" open markdown="1">
<summary>Note</summary>

Requires ``ipywidgets``; `BrainData.iplot` guards that at the boundary
(raising a friendly install hint) before delegating here.

</details>

######## `build_viewer`

```python
build_viewer(bd, *, view: str = 'ortho', cal_min: float | None = None, cal_max: float | None = None, cmap: str = 'warm', atlas: str | Atlas | None = None, bg_img: str | bool | None = None, opacity: float = 1.0, outline: float = 0.0, colorbar: bool = True, niivue_opts: dict | None = None) -> NiiVue
```

Assemble a configured `NiiVue` for a BrainData.

Builds the volume stack ``[background?, statmap, atlas?]`` (atlas on top
so its outlines/opacity keep the stat map readable), loads it, applies
the atlas label LUT / outline, and sets the slice type.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData to view. | *required*
`view` | <code>[str](#str)</code> | See `slice_type_for`. | <code>'ortho'</code>
`cal_min` | <code>[float](#float) \| None</code> | Window floor (threshold), or ``None`` for auto. | <code>None</code>
`cal_max` | <code>[float](#float) \| None</code> | Window ceiling, or ``None`` for auto. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Positive colormap (niivue or matplotlib name). | <code>'warm'</code>
`atlas` | <code>[str](#str) \| [Atlas](#nltools.data.atlases.Atlas) \| None</code> | Atlas name, `Atlas`, or ``None``. | <code>None</code>
`bg_img` | <code>[str](#str) \| [bool](#bool) \| None</code> | See `resolve_background`. | <code>None</code>
`opacity` | <code>[float](#float)</code> | Stat-map (and filled-atlas) opacity. | <code>1.0</code>
`outline` | <code>[float](#float)</code> | ``> 0`` draws atlas region boundaries of that width; ``0`` draws filled regions. | <code>0.0</code>
`colorbar` | <code>[bool](#bool)</code> | Show the stat-map colorbar (the ``cmap`` scale). Only the stat map carries a colorbar; the background and atlas overlays are suppressed. An explicit ``is_colorbar`` in ``niivue_opts`` wins. | <code>True</code>
`niivue_opts` | <code>[dict](#dict) \| None</code> | Extra kwargs forwarded verbatim to ``NiiVue(**opts)``. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[NiiVue](#ipyniivue.NiiVue)</code> | A configured `NiiVue` ready to display.

######## `divergent_partner`

```python
divergent_partner(cmap: str) -> str
```

Return the ``colormap_negative`` partner for a positive colormap.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cmap` | <code>[str](#str)</code> | A (resolved) niivue positive colormap name. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | The niivue colormap to use for negative values.

######## `qualitative_colors`

```python
qualitative_colors(n: int, *, seed: int = 0) -> list[tuple[int, int, int]]
```

Deterministic qualitative RGB palette of length ``n``.

Hues are spaced by the golden angle for maximal separation; saturation
and value cycle through three bands so adjacent indices stay visually
distinct. Atlases carry no color data, so this assigns region colors.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n` | <code>[int](#int)</code> | Number of colors to generate. | *required*
`seed` | <code>[int](#int)</code> | Rotates the starting hue; deterministic for a given seed. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[tuple](#tuple)[[int](#int), [int](#int), [int](#int)]]</code> | ``n`` ``(r, g, b)`` tuples with components in ``0..255``.

######## `resolve_background`

```python
resolve_background(affine, bg_img: str | bool | None) -> str | None
```

Resolve the ``bg_img`` argument to a background-image path or ``None``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`affine` |  | 4x4 affine of the BrainData (``bd.mask.affine``), used to decide whether auto-MNI applies. | *required*
`bg_img` | <code>[str](#str) \| [bool](#bool) \| None</code> | ``False`` → no background; a string/path → used as-is; ``None``/``True`` (auto) → the matching MNI template when the affine is standard space, else ``None``. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[str](#str) \| None</code> | A path to a background image, or ``None`` for no background.

<details class="note" open markdown="1">
<summary>Note</summary>

``is_standard_space(np.eye(4)) == (True, None)`` (1mm is a valid
template resolution), so identity-affine fixtures count as standard
space and auto would fetch a template from HuggingFace. Offline
callers should pass ``bg_img=False``.

</details>

######## `resolve_cmap`

```python
resolve_cmap(name: str) -> str
```

Resolve a colormap name to a valid niivue colormap.

Valid niivue names pass through. Common matplotlib names are mapped to
the closest niivue equivalent (with a warning, since the mapping is
lossy). Anything else falls back to ``"warm"`` with a warning, because
niivue renders unknown colormaps as flat gray with no error.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`name` | <code>[str](#str)</code> | A niivue or matplotlib colormap name (case-insensitive). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | A valid niivue colormap name.

######## `slice_type_for`

```python
slice_type_for(view: str) -> SliceType
```

Map a ``view`` string to a niivue `SliceType`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`view` | <code>[str](#str)</code> | One of ``"ortho"``, ``"axial"``, ``"coronal"``, ``"sagittal"``, ``"render"``. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[SliceType](#ipyniivue.SliceType)</code> | The matching `SliceType`.

######## `threshold_slider_bounds`

```python
threshold_slider_bounds(bd, *, cal_min: float | None, cal_max: float | None) -> tuple[float, float, float, float, float]
```

Compute ``(min, max, value_low, value_high, step)`` for a threshold slider.

The slider spans the BrainData's finite value range, widened as needed to
include an explicit ``cal_min``/``cal_max`` so the requested window is
always representable (never silently clamped). Its initial handles sit at
``cal_min``/``cal_max`` when given, else at the data extremes.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | The BrainData being viewed. | *required*
`cal_min` | <code>[float](#float) \| None</code> | Requested window floor, or ``None``. | *required*
`cal_max` | <code>[float](#float) \| None</code> | Requested window ceiling, or ``None``. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[float](#float), [float](#float), [float](#float), [float](#float), [float](#float)]</code> | ``(lo_bound, hi_bound, value_low, value_high, step)`` — all floats.

(data-collection)=
#### `collection`

BrainCollection — multi-subject brain-data container (v0.6.0).

<details class="public-class-is-a-thin-facade-over-module-level-helpers" open markdown="1">
<summary>Public class is a thin facade over module-level helpers</summary>

- core.py       — metadata coercion, mask resolution, run/step IDs
- io.py         — constructors, write/read, load/unload
- execution.py  — parallel ``_apply``, worker dataclasses, HDF5 bundles
- inference.py  — group reductions, ISC, align, permutation tests
- pipeline.py   — ``BrainCollectionPipeline`` (CV pipeline; legacy API)

</details>

See ``SPEC.md`` for the full design contract.

**Modules:**

Name | Description
---- | -----------
[`core`](#data-core) | Module-level helpers for BrainCollection.
[`execution`](#data-execution) | Parallel execution machinery for BrainCollection.
[`inference`](#data-inference) | Group-level reductions and cross-subject ops for BrainCollection.
[`io`](#data-io) | IO and constructors for BrainCollection.
[`pipeline`](#data-pipeline) | Pipeline classes for BrainCollection.

**Classes:**

Name | Description
---- | -----------
[`BrainCollection`](#data-braincollection) | Parallel, lazy iterator of ``BrainData`` whose API mirrors ``BrainData``.
[`BrainCollectionPipeline`](#data-braincollectionpipeline) | Pipeline for BrainCollection with multi-subject CV support.
[`BrainCollectionWorkerError`](#data-braincollectionworkererror) | Raised in the parent process when a worker fails inside ``_apply``.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`BUNDLE_SCHEMA_VERSION`](#data-bundle-schema-version) |  | 

##### Classes

###### `BrainCollection`

```python
BrainCollection(brains: list, *, mask: nib.Nifti1Image | Path | str, designs: list | None = None, metadata: pl.DataFrame | pd.DataFrame | dict | None = None, lazy: bool = True, cache_dir: Path | str | None = './.nltools_cache') -> None
```

Parallel, lazy iterator of ``BrainData`` whose API mirrors ``BrainData``.

Constructed via ``__init__`` (explicit lists) or one of the classmethod
factories (``from_bids``, ``from_glob``, ``from_paths``, ``read``).

See ``SPEC.md`` §"Public API" for the full contract; key invariants:
  - Per-subject ops route through ``execution._apply`` and return a
    lightweight clone via ``self._clone(...)`` over the same cache root.
  - Path-backed by default after parallel ops; ``cache='auto'`` follows
    source state. ``cache=`` is only accepted on collection-returning ops.
  - ``load`` / ``unload`` are the only methods that mutate ``self``.

Internal state (mutable list at top level; per-item slots are parallel):

  _items          list[BrainData | Path]        per-item brain data
  _mask           nib.Nifti1Image               shared mask (by reference)
  _designs        list[DesignMatrix | Path | None]
  _confounds      list[pd.DataFrame | None]
  _sample_masks   list[np.ndarray | None]
  _metadata       pl.DataFrame                  simple-typed columns only
  _cache_root     Path | None                   shared by clones
  _step_id        str | None                    this collection's step id
  _parent_step_id str | None                    upstream step id (lineage)
  _step_dirs      list[Path]                    lineage of step subdirs
                                                that produced these items
  _source_paths   list[Path | None]             per-item backing path
                                                (None for in-memory only)

**Methods:**

Name | Description
---- | -----------
[`align`](#data-align) | Functionally align subjects into a common space via `LocalAlignment`.
[`anova`](#data-anova) | One-way ANOVA across subjects grouped by ``groups``.
[`apply`](#data-apply) | Call ``BrainData.<op>(*args, **kwargs)`` on every item in parallel.
[`cleanup`](#data-cleanup) | Remove ``cache_root`` and invalidate every clone derived from ``self``.
[`cleanup_all`](#data-cleanup-all) | Remove every ``.nltools_cache/{run_id}/`` under ``directory``.
[`compute_contrasts`](#data-compute-contrasts) | Compute per-subject contrast maps from fit-bundle items.
[`concat`](#data-concat) | Stack all subject maps into a single `BrainData` (subjects as rows).
[`cv`](#data-cv) | Build a CV pipeline for cross-subject prediction.
[`detrend`](#data-detrend) | Detrend every subject's image in parallel (delegates to `BrainData.detrend`).
[`filter`](#data-filter) | Filter to a subset by predicate, polars expression, or boolean array.
[`fit`](#data-fit) | Per-subject fit; returns a path-backed collection of HDF5 fit bundles.
[`from_bids`](#data-from-bids) | Auto-pair BOLD with events.tsv (→ ``DesignMatrix``) and confounds.tsv.
[`from_glob`](#data-from-glob) | Build a collection by glob-matching brain images (and optional designs).
[`from_paths`](#data-from-paths) | Build a collection from explicit lists of brain (and design) paths.
[`isc`](#data-isc) | Inter-subject correlation (ISC) across the time dimension.
[`isc_test`](#data-isc-test) | Bootstrap inference on ISC (per-voxel p-values).
[`iter_pairs`](#data-iter-pairs) | Yield ``(BrainData, DesignMatrix | None)`` pairs.
[`load`](#data-load) | Materialize path-backed items in place. Returns ``self`` for chaining.
[`map`](#data-map) | Apply an arbitrary ``fn(BrainData) -> BrainData`` to each item in parallel.
[`max`](#data-max) | Voxelwise maximum across subjects as a single `BrainData`.
[`mean`](#data-mean) | Voxelwise mean across subjects as a single `BrainData`.
[`median`](#data-median) | Voxelwise median across subjects as a single `BrainData`.
[`memory_estimate`](#data-memory-estimate) | Human-readable RAM estimate if every item were loaded into memory.
[`min`](#data-min) | Voxelwise minimum across subjects as a single `BrainData`.
[`permutation_test`](#data-permutation-test) | One-sample sign-flipping permutation test across subjects.
[`permutation_test2`](#data-permutation-test2) | Two-sample permutation test between this collection and ``other``.
[`predict`](#data-predict) | Two distinct paths, dispatched by argument:
[`read`](#data-read) | Inverse of ``write()``. Does not recover from cache subdirs in v0.6.0.
[`resample`](#data-resample) | Resample every subject's image to a target space in parallel.
[`smooth`](#data-smooth) | Spatially smooth every subject's image in parallel (delegates to `BrainData.smooth`).
[`standardize`](#data-standardize) | Standardize every subject's image in parallel (delegates to `BrainData.standardize`).
[`std`](#data-std) | Voxelwise standard deviation across subjects as a single `BrainData`.
[`steps`](#data-steps) | Step subdirs that produced this collection's items, oldest to newest.
[`sum`](#data-sum) | Voxelwise sum across subjects as a single `BrainData`.
[`threshold`](#data-threshold) | Threshold every subject's image in parallel (delegates to `BrainData.threshold`).
[`transform_designs`](#data-transform-designs) | Map ``fn(dm) -> DesignMatrix`` over each paired design.
[`ttest`](#data-ttest) | One-sample t-test across subjects (delegates to `inference.ttest`).
[`ttest2`](#data-ttest2) | Two-sample t-test between this collection and ``other`` (subject-level).
[`unload`](#data-unload) | Drop in-memory data for items with backing paths. Returns ``self``.
[`var`](#data-var) | Voxelwise variance across subjects as a single `BrainData`.
[`write`](#data-write) | Write a clean, portable copy of the collection outside the cache root.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`cache_root`](#data-cache-root) | <code>[Path](#pathlib.Path)</code> | Run-scoped cache directory shared by clones. Raises if unset.
`designs` | <code>[list](#list)</code> | Per-subject paired designs (a copy of the list; ``None`` where unpaired).
`is_loaded` | <code>[list](#list)[[bool](#bool)]</code> | Per-item flag — True iff the slot holds a ``BrainData`` (not a path).
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image)</code> | Shared mask image for the collection. Raises if the mask is unset.
`metadata` | <code>[DataFrame](#polars.DataFrame)</code> | Per-subject metadata as a polars DataFrame (one row per item).
`n_subjects` | <code>[int](#int)</code> | Number of subjects (items) in the collection.
`n_voxels` | <code>[int](#int)</code> | Voxel count from the mask. Raises if mask is unset.
`shape` | <code>[tuple](#tuple)[[int](#int), [int](#int) \| None, [int](#int)]</code> | ``(n_subjects, n_obs_or_None_if_ragged, n_voxels)``.

``cache_dir`` precedence: explicit arg → ``NLTOOLS_CACHE_DIR`` env →
``./.nltools_cache``. Pass ``None`` for an auto-cleaned tempdir.
Resolved at construction and frozen on the instance.



####### Attributes##

(data-cache-root)=
###### `cache_root`

```python
cache_root: Path
```

Run-scoped cache directory shared by clones. Raises if unset.

######## `designs`

```python
designs: list
```

Per-subject paired designs (a copy of the list; ``None`` where unpaired).

######## `is_loaded`

```python
is_loaded: list[bool]
```

Per-item flag — True iff the slot holds a ``BrainData`` (not a path).

######## `mask`

```python
mask: nib.Nifti1Image
```

Shared mask image for the collection. Raises if the mask is unset.

######## `metadata`

```python
metadata: pl.DataFrame
```

Per-subject metadata as a polars DataFrame (one row per item).

######## `n_subjects`

```python
n_subjects: int
```

Number of subjects (items) in the collection.

######## `n_voxels`

```python
n_voxels: int
```

Voxel count from the mask. Raises if mask is unset.

######## `shape`

```python
shape: tuple[int, int | None, int]
```

``(n_subjects, n_obs_or_None_if_ragged, n_voxels)``.

``n_obs`` is ``None`` when any item is path-backed (loading just to
report shape would defeat the purpose) or when items are ragged.



####### Functions##

###### `align`

```python
align(*, method: str = 'procrustes', spatial_scale: str = 'searchlight', radius_mm: float = 10.0, roi_mask: nib.Nifti1Image | None = None, n_features: int | None = None, n_iter: int = 3, device: str = 'cpu', return_model: bool = False, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto')
```

Functionally align subjects into a common space via `LocalAlignment`.

Materializes all subjects (algorithm constraint in v0.6.0).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Alignment solver (e.g. ``'procrustes'``). | <code>'procrustes'</code>
`spatial_scale` | <code>[str](#str)</code> | Alignment scope (``'searchlight'``, ``'roi'``, ``'whole_brain'``). | <code>'searchlight'</code>
`radius_mm` | <code>[float](#float)</code> | Searchlight sphere radius in mm. | <code>10.0</code>
`roi_mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| None</code> | Parcellation/ROI mask (used when ``spatial_scale='roi'``). | <code>None</code>
`n_features` | <code>[int](#int) \| None</code> | Optional target feature count for the common space. | <code>None</code>
`n_iter` | <code>[int](#int)</code> | LocalAlignment solver iteration count (not a permutation count). | <code>3</code>
`device` | <code>[str](#str)</code> | Backend selector (``'cpu'``/``'gpu'``). | <code>'cpu'</code>
`return_model` | <code>[bool](#bool)</code> | If True, also return the fitted `LocalAlignment`. | <code>False</code>
`n_jobs` | <code>[int](#int)</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>[bool](#bool)</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>[Literal](#typing.Literal)['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
 | A new `BrainCollection` of aligned data, or a
 | ``(BrainCollection, LocalAlignment)`` tuple when
 | ``return_model=True``.

######## `anova`

```python
anova(groups: str | list | np.ndarray) -> dict
```

One-way ANOVA across subjects grouped by ``groups``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`groups` | <code>[str](#str) \| [list](#list) \| [ndarray](#numpy.ndarray)</code> | A metadata column name, or a list/ndarray of length ``n_subjects`` giving each subject's group label. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict with ``{'F', 'p'}`` `BrainData` maps plus ``df_between`` and
<code>[dict](#dict)</code> | ``df_within`` degrees of freedom.

######## `apply`

```python
apply(op: str, *args: str, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto', **kwargs: Literal['auto', True, False]) -> BrainCollection
```

Call ``BrainData.<op>(*args, **kwargs)`` on every item in parallel.

All per-subject methods (``smooth``, ``standardize``, ...) reduce to
this. Centralizes the ``_apply`` plumbing and the cache-knob handling.
``op`` is named ``op`` (not ``method``) to avoid colliding with
``BrainData`` methods that themselves take a ``method=`` kwarg
(``standardize``, ``detrend``, ...).

######## `cleanup`

```python
cleanup() -> None
```

Remove ``cache_root`` and invalidate every clone derived from ``self``.

Idempotent — calling twice is a no-op. Path-backed items in any
clone become unloadable after this; use ``bc.write(...)`` first to
materialize a portable copy if needed.

######## `cleanup_all`

```python
cleanup_all(directory: Path | str = '.') -> None
```

Remove every ``.nltools_cache/{run_id}/`` under ``directory``.

Wide brush — can kill sibling sessions in the same cwd. Prefer
``bc.cleanup()`` for surgical removal.

######## `compute_contrasts`

```python
compute_contrasts(contrasts: str | list[str] | dict[str, np.ndarray], *, statistic: str = 'beta', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection | dict[str, BrainCollection] | dict[str, dict[str, BrainCollection]]
```

Compute per-subject contrast maps from fit-bundle items.

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)] \| [dict](#dict)[[str](#str), [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]]</code> | single contrast + single ``statistic`` → ``BrainCollection``
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)] \| [dict](#dict)[[str](#str), [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]]</code> | multiple contrasts (single type)            → ``dict[str, BrainCollection]``
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)] \| [dict](#dict)[[str](#str), [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]]</code> | ``statistic='all'`` (single contrast)   → ``dict['beta'|'t'|'z'|'p'|'se', BrainCollection]``
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)] \| [dict](#dict)[[str](#str), [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]]</code> | multiple contrasts + ``statistic='all'`` → nested                                              ``dict[name, dict[stat, BrainCollection]]``

Each per-subject NIfTI gets a JSON sidecar with lineage attrs
(``step_id``, ``parent_step_id``, ``op``, ``kwargs``,
``nltools_version``).

######## `concat`

```python
concat() -> BrainData
```

Stack all subject maps into a single `BrainData` (subjects as rows).

######## `cv`

```python
cv(*, k: int | None = None, method: str = 'kfold', split_by: str | None = None, groups: np.ndarray | None = None, n: int = 1000, random_state: int | None = None) -> BrainCollectionPipeline
```

Build a CV pipeline for cross-subject prediction.

See ``pipeline.py`` for the builder API. The pipeline's ``predict``
terminal returns a ``BrainData`` with CV attrs attached.

######## `detrend`

```python
detrend(*, method: str = 'linear', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Detrend every subject's image in parallel (delegates to `BrainData.detrend`).

######## `filter`

```python
filter(predicate: Callable[[Any], Any] | list | np.ndarray | pl.Series | pd.Series) -> BrainCollection
```

Filter to a subset by predicate, polars expression, or boolean array.

######## `fit`

```python
fit(model: str = 'glm', X: DesignMatrix | list | Callable | None = None, *, scale: bool | str = 'auto', standardize: str | None = 'auto', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto', **model_kwargs: Literal['auto', True, False]) -> BrainCollection
```

Per-subject fit; returns a path-backed collection of HDF5 fit bundles.

``X`` resolution priority:
  - ``None``         → use ``self.designs`` (must be set per subject)
  - ``DesignMatrix`` → shared across all subjects
  - ``list``         → per-subject (len == n_subjects)
  - ``callable``     → ``fn(ctx: _DesignContext) -> DesignMatrix``

######## `from_bids`

```python
from_bids(root: Path | str | Any, *, mask: nib.Nifti1Image | Path | str, task: str | None = None, space: str | None = None, sub_labels: list[str] | None = None, img_filters: list[tuple[str, str]] | None = None, derivatives_folder: str = 'derivatives', pair_events: bool = True, confounds_strategy: str | tuple[str, ...] | None = None, confounds_kwargs: dict | None = None, TR: float | str = 'infer', cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Auto-pair BOLD with events.tsv (→ ``DesignMatrix``) and confounds.tsv.

Full design and edge cases: SPEC §"``from_bids`` — concrete design".

######## `from_glob`

```python
from_glob(pattern: str, *, mask: nib.Nifti1Image | Path | str, design_pattern: str | None = None, pattern_groups: dict[str, int] | str | None = None, sort: bool = True, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Build a collection by glob-matching brain images (and optional designs).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`pattern` | <code>[str](#str)</code> | Glob pattern matching the per-subject brain image files. | *required*
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str)</code> | Shared mask image, path, or nltools template name. | *required*
`design_pattern` | <code>[str](#str) \| None</code> | Optional glob matching per-subject design files, paired positionally with the brain images. | <code>None</code>
`pattern_groups` | <code>[dict](#dict)[[str](#str), [int](#int)] \| [str](#str) \| None</code> | Regex capture-group spec used to extract metadata (e.g. subject/run) from each matched path. | <code>None</code>
`sort` | <code>[bool](#bool)</code> | If True, sort matched paths before pairing (stable ordering). | <code>True</code>
`cache_dir` | <code>[Path](#pathlib.Path) \| [str](#str) \| None</code> | Cache-directory precedence: explicit arg → ``NLTOOLS_CACHE_DIR`` env → ``./.nltools_cache``; ``None`` for a temp dir. | <code>'./.nltools_cache'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | A lazy, path-backed `BrainCollection`.

######## `from_paths`

```python
from_paths(brain_paths: list, *, mask: nib.Nifti1Image | Path | str, design_paths: list | None = None, metadata: pl.DataFrame | pd.DataFrame | dict | None = None, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Build a collection from explicit lists of brain (and design) paths.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_paths` | <code>[list](#list)</code> | Per-subject brain image paths. | *required*
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str)</code> | Shared mask image, path, or nltools template name. | *required*
`design_paths` | <code>[list](#list) \| None</code> | Optional per-subject design paths, aligned positionally with ``brain_paths`` (length must match, ``None`` entries allowed). | <code>None</code>
`metadata` | <code>[DataFrame](#polars.DataFrame) \| [DataFrame](#pandas.DataFrame) \| [dict](#dict) \| None</code> | Optional per-subject metadata (polars/pandas DataFrame or dict-of-columns), one row per path. | <code>None</code>
`cache_dir` | <code>[Path](#pathlib.Path) \| [str](#str) \| None</code> | Cache-directory precedence: explicit arg → ``NLTOOLS_CACHE_DIR`` env → ``./.nltools_cache``; ``None`` for a temp dir. | <code>'./.nltools_cache'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | A lazy, path-backed `BrainCollection`.

######## `isc`

```python
isc(*, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, metric: str = 'median') -> dict
```

Inter-subject correlation (ISC) across the time dimension.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ``'loo'`` (leave-one-out template) or ``'pairwise'`` (all subject pairs). | <code>'loo'</code>
`roi_mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str) \| None</code> | Optional ROI/atlas mask restricting the computation to those voxels. The returned maps carry the ROI mask. If None, ISC is computed across the collection's whole-brain mask. | <code>None</code>
`metric` | <code>[str](#str)</code> | Aggregation across subjects/pairs (e.g. ``'median'``). | <code>'median'</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict ``{'isc', 'per_subject'}`` for ``method='loo'`` or
<code>[dict](#dict)</code> | ``{'isc', 'pairs'}`` for ``method='pairwise'`` (``'isc'`` is a
<code>[dict](#dict)</code> | `BrainData` map).

######## `isc_test`

```python
isc_test(*, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, n_samples: int = 5000, metric: str = 'median', random_state: int | None = None) -> dict
```

Bootstrap inference on ISC (per-voxel p-values).

Resamples subjects with replacement, recomputes ISC each draw, and
derives a per-voxel two-tailed p-value from the null centered at 0.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ``'loo'`` or ``'pairwise'`` (matches `isc`). | <code>'loo'</code>
`roi_mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str) \| None</code> | Optional ROI/atlas mask restricting the computation to those voxels. The returned maps carry the ROI mask. If None, ISC is computed across the collection's whole-brain mask. | <code>None</code>
`n_samples` | <code>[int](#int)</code> | Number of bootstrap resamples. | <code>5000</code>
`metric` | <code>[str](#str)</code> | Aggregation across subjects/pairs (e.g. ``'median'``). | <code>'median'</code>
`random_state` | <code>[int](#int) \| None</code> | Seed for the bootstrap RNG. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict ``{'isc', 'p', 'null_distribution'}`` (``'isc'`` and ``'p'`` are
<code>[dict](#dict)</code> | `BrainData` maps).

######## `iter_pairs`

```python
iter_pairs() -> Iterator[tuple]
```

Yield ``(BrainData, DesignMatrix | None)`` pairs.

######## `load`

```python
load(indices: list[int] | None = None) -> BrainCollection
```

Materialize path-backed items in place. Returns ``self`` for chaining.

######## `map`

```python
map(fn: Callable, *, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Apply an arbitrary ``fn(BrainData) -> BrainData`` to each item in parallel.

######## `max`

```python
max() -> BrainData
```

Voxelwise maximum across subjects as a single `BrainData`.

######## `mean`

```python
mean() -> BrainData
```

Voxelwise mean across subjects as a single `BrainData`.

######## `median`

```python
median() -> BrainData
```

Voxelwise median across subjects as a single `BrainData`.

######## `memory_estimate`

```python
memory_estimate() -> str
```

Human-readable RAM estimate if every item were loaded into memory.

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | A string reporting ``n_subjects``, the per-item shape (or "unknown"
<code>[str](#str)</code> | for path-backed items not yet loaded), and an estimated float32
<code>[str](#str)</code> | total in MB/GB.

######## `min`

```python
min() -> BrainData
```

Voxelwise minimum across subjects as a single `BrainData`.

######## `permutation_test`

```python
permutation_test(*, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

One-sample sign-flipping permutation test across subjects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_permute` | <code>[int](#int)</code> | Number of sign-flip permutations. | <code>5000</code>
`tail` | <code>[int](#int)</code> | 1 for one-tailed, 2 for two-tailed. | <code>2</code>
`device` | <code>[str](#str)</code> | Backend selector (currently informational). | <code>'cpu'</code>
`return_null` | <code>[bool](#bool)</code> | If True, include the null distribution in the result. | <code>False</code>
`n_jobs` | <code>[int](#int)</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Seed for the sign-flip RNG. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict ``{'mean', 'p'}`` of `BrainData` maps, plus
<code>[dict](#dict)</code> | ``'null_distribution'`` when ``return_null=True``.

######## `permutation_test2`

```python
permutation_test2(other: BrainCollection, *, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

Two-sample permutation test between this collection and ``other``.

Uses random label shuffling of the pooled subjects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | The second collection to compare against. | *required*
`n_permute` | <code>[int](#int)</code> | Number of label-shuffle permutations. | <code>5000</code>
`tail` | <code>[int](#int)</code> | 1 for one-tailed, 2 for two-tailed. | <code>2</code>
`device` | <code>[str](#str)</code> | Backend selector (currently informational). | <code>'cpu'</code>
`return_null` | <code>[bool](#bool)</code> | If True, include the null distribution in the result. | <code>False</code>
`n_jobs` | <code>[int](#int)</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Seed for the shuffling RNG. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict ``{'mean', 'p'}`` of `BrainData` maps (``mean`` is the group
<code>[dict](#dict)</code> | difference), plus ``'null_distribution'`` when ``return_null=True``.

######## `predict`

```python
predict(y: str | list | np.ndarray | None = None, *, X_new: np.ndarray | None = None, spatial_scale: str = 'whole_brain', model: str = 'svm', cv: int | str = 'loso', groups: str | np.ndarray | None = None, roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float = 10.0, scoring: str = 'auto', standardize: bool = True, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto')
```

Two distinct paths, dispatched by argument:

  ``y=`` only    → group MVPA (subjects as samples) → ``Predict``
  ``X_new=`` only → per-subject predict-after-fit  → ``BrainCollection``
  both / neither → raise

``predict(y=...)`` requires single-map-per-subject items (run
``compute_contrasts(...)`` first if you have GLM/ridge bundles).

######## `read`

```python
read(directory: Path | str, *, mask: nib.Nifti1Image | Path | str, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Inverse of ``write()``. Does not recover from cache subdirs in v0.6.0.

######## `resample`

```python
resample(target, *, interpolation: str = 'continuous', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Resample every subject's image to a target space in parallel.

Delegates to `BrainData.resample`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` |  | Resampling target (image, affine/shape spec, or template) passed through to `BrainData.resample`. | *required*
`interpolation` | <code>[str](#str)</code> | Interpolation method (``'continuous'``, ``'linear'``, ``'nearest'``). | <code>'continuous'</code>
`n_jobs` | <code>[int](#int)</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>[bool](#bool)</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>[Literal](#typing.Literal)['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | A new `BrainCollection` of resampled items.

######## `smooth`

```python
smooth(fwhm: float, *, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Spatially smooth every subject's image in parallel (delegates to `BrainData.smooth`).

######## `standardize`

```python
standardize(*, axis: int = 0, method: str = 'center', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Standardize every subject's image in parallel (delegates to `BrainData.standardize`).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>[int](#int)</code> | Axis along which to standardize (0 = across observations). | <code>0</code>
`method` | <code>[str](#str)</code> | Standardization variant (e.g. ``'center'``, ``'zscore'``). | <code>'center'</code>
`n_jobs` | <code>[int](#int)</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>[bool](#bool)</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>[Literal](#typing.Literal)['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | A new `BrainCollection` of standardized items.

######## `std`

```python
std() -> BrainData
```

Voxelwise standard deviation across subjects as a single `BrainData`.

######## `steps`

```python
steps() -> list[Path]
```

Step subdirs that produced this collection's items, oldest to newest.

Lineage chain accumulated through clones (one entry per upstream
cached op). Empty when the collection was constructed directly or
no ancestor wrote to disk.

######## `sum`

```python
sum() -> BrainData
```

Voxelwise sum across subjects as a single `BrainData`.

######## `threshold`

```python
threshold(*, lower: float | None = None, upper: float | None = None, binarize: bool = False, coerce_nan: bool = True, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Threshold every subject's image in parallel (delegates to `BrainData.threshold`).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`lower` | <code>[float](#float) \| None</code> | Values below this are zeroed (or set NaN); ``None`` disables. | <code>None</code>
`upper` | <code>[float](#float) \| None</code> | Values above this are zeroed (or set NaN); ``None`` disables. | <code>None</code>
`binarize` | <code>[bool](#bool)</code> | If True, set surviving voxels to 1. | <code>False</code>
`coerce_nan` | <code>[bool](#bool)</code> | If True, coerce thresholded-out voxels to NaN instead of 0. | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>[bool](#bool)</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>[Literal](#typing.Literal)['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | A new `BrainCollection` of thresholded items.

######## `transform_designs`

```python
transform_designs(fn: Callable, *, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Map ``fn(dm) -> DesignMatrix`` over each paired design.

Items with no paired design are skipped (kept as ``None``). Runs in
the parent process — designs are small. ``n_jobs``/``progress_bar``/
``cache`` are accepted for surface consistency but ignored.

######## `ttest`

```python
ttest(*, popmean: float = 0.0) -> dict
```

One-sample t-test across subjects (delegates to `inference.ttest`).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`popmean` | <code>[float](#float)</code> | Null-hypothesis population mean to test against. | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict ``{'mean', 't', 'z', 'p'}`` of `BrainData` maps.

######## `ttest2`

```python
ttest2(other: BrainCollection, *, equal_var: bool = True) -> dict
```

Two-sample t-test between this collection and ``other`` (subject-level).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | The second collection to compare against. | *required*
`equal_var` | <code>[bool](#bool)</code> | If True, pooled-variance t-test; if False, Welch's test. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict ``{'mean', 't', 'z', 'p'}`` of `BrainData` maps (``mean`` is the
<code>[dict](#dict)</code> | group difference).

######## `unload`

```python
unload(indices: list[int] | None = None) -> BrainCollection
```

Drop in-memory data for items with backing paths. Returns ``self``.

######## `var`

```python
var() -> BrainData
```

Voxelwise variance across subjects as a single `BrainData`.

######## `write`

```python
write(directory: Path | str, *, pattern: str = 'image_{i:04d}.nii.gz', metadata_file: str | None = 'metadata.csv') -> list[Path]
```

Write a clean, portable copy of the collection outside the cache root.

Inverse of `BrainCollection.read`. Writes one NIfTI per item plus an
optional metadata CSV, skipping the internal cache layout so the result
is shareable/archival.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`directory` | <code>[Path](#pathlib.Path) \| [str](#str)</code> | Output directory (created if missing). | *required*
`pattern` | <code>[str](#str)</code> | Filename template per item, formatted with ``i`` (item index). | <code>'image_{i:04d}.nii.gz'</code>
`metadata_file` | <code>[str](#str) \| None</code> | CSV filename for the metadata table, or ``None`` to skip. | <code>'metadata.csv'</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[Path](#pathlib.Path)]</code> | List of written NIfTI paths, in item order.

(data-braincollectionpipeline)=
###### `BrainCollectionPipeline`

```python
BrainCollectionPipeline(brain_collection: BrainCollection, cv: BrainCollection = None, groups: np.ndarray | None = None)
```

Pipeline for BrainCollection with multi-subject CV support.

Wraps BrainCollection to provide fluent pipeline API with LOSO
and run-based cross-validation.

This class enables method chaining for preprocessing and prediction
with proper cross-validation semantics for multi-subject neuroimaging
analyses.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`n_subjects` | <code>[int](#int)</code> | Number of subjects/images in the collection.
[`cv`](#data-cv) |  | The cross-validation scheme configuration.
`n_steps` | <code>[int](#int)</code> | Number of transform steps in the pipeline.

**Examples:**

```pycon
>>> # Leave-one-subject-out with preprocessing
>>> result = (bc
...     .cv(method='loso')
...     .standardize()
...     .reduce(n_components=50)
...     .predict(labels, method='svm'))
>>> print(f"Mean accuracy: {result.mean_score:.2%}")
```

**Methods:**

Name | Description
---- | -----------
[`pipe`](#data-pipe) | Add custom sklearn transformer.
[`predict`](#data-predict) | Execute pipeline with CV and return prediction results.
`reduce` | Add dimensionality reduction step.
[`standardize`](#data-standardize) | Add standardization step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_collection` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | BrainCollection to wrap. | *required*
`cv` |  | CVScheme configuration. | <code>None</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Group labels for CV splits. | <code>None</code>



####### Attributes##

###### `cv`

```python
cv
```

Cross-validation scheme.

######## `n_steps`

```python
n_steps: int
```

Number of transform steps.

######## `n_subjects`

```python
n_subjects: int
```

Number of subjects/images.



####### Functions##

(data-pipe)=
###### `pipe`

```python
pipe(transformer) -> BrainCollectionPipeline
```

Add custom sklearn transformer.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`transformer` |  | sklearn-compatible transformer with fit/transform interface. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollectionPipeline](#nltools.data.collection.pipeline.BrainCollectionPipeline)</code> | New pipeline with custom step added.

######## `predict`

```python
predict(y, method: str = 'ridge', *, n_permute: int = 0, random_state: int = None, **kwargs: int)
```

Execute pipeline with CV and return prediction results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` |  | Target variable. For LOSO, shape should be (n_subjects,). | *required*
`method` | <code>[str](#str)</code> | Prediction algorithm ('ridge', 'svm', 'logistic', etc.) | <code>'ridge'</code>
`n_permute` | <code>[int](#int)</code> | If ``> 0``, also build a label-permutation null of the CV score — the classic MVPA permutation test. Each iteration shuffles ``y``, re-runs the *same* cross-validation, and records the mean score; the result gets ``permutation_scores`` (the null array) and ``permutation_pvalue`` attached. Default 0 (no null). | <code>0</code>
`random_state` |  | Seed for the label shuffling (permutation null only). | <code>None</code>
`**kwargs` |  | Passed to model constructor. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | ``BrainData`` carrying out-of-fold predictions plus CV attributes
 | (``cv_scores``, ``cv_predictions``, ``mean_score``, ``std_score``,
 | ``fold_results``, ``cv_pipeline``). When ``n_permute > 0`` it also
 | carries ``permutation_scores`` and ``permutation_pvalue``.

######## `reduce`

```python
reduce(method: str = 'pca', n_components: int | None = None, **kwargs: int | None) -> BrainCollectionPipeline
```

Add dimensionality reduction step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Reduction method ('pca', 'ica'). | <code>'pca'</code>
`n_components` | <code>[int](#int) \| None</code> | Number of components to keep. | <code>None</code>
`**kwargs` |  | Additional arguments for ReduceStep. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollectionPipeline](#nltools.data.collection.pipeline.BrainCollectionPipeline)</code> | New pipeline with reduction step added.

######## `standardize`

```python
standardize(method: str = 'zscore', **kwargs: str) -> BrainCollectionPipeline
```

Add standardization step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Standardization method ('zscore', 'minmax'). | <code>'zscore'</code>
`**kwargs` |  | Additional arguments for NormalizeStep. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollectionPipeline](#nltools.data.collection.pipeline.BrainCollectionPipeline)</code> | New pipeline with standardization step added.

(data-braincollectionworkererror)=
###### `BrainCollectionWorkerError`

Bases: <code>[RuntimeError](#RuntimeError)</code>

Raised in the parent process when a worker fails inside ``_apply``.

Wraps the original exception via ``raise ... from e`` so the full
traceback is preserved. The message embeds subject/run context from
``_ItemTask.metadata_row`` so users can locate the offending item.



##### Modules

(data-core)=
###### `core`

Module-level helpers for BrainCollection.

Pure functions: metadata coercion, mask resolution, run/step ID generation,
step-directory naming. No class state lives here.

**Methods:**

Name | Description
---- | -----------
[`coerce_metadata`](#data-coerce-metadata) | Coerce a metadata input into a polars DataFrame of length ``n_subjects``.
`make_run_id` | Build a fresh ``run_id`` of the form ``{timestamp}_{uuid8}``.
`make_step_dirname` | Name a step subdir: ``{timestamp}_{seq}_{uuid8}_{op}_{key_kwargs}/``.
`resolve_cache_dir` | Resolve ``cache_dir`` per the spec's precedence rules.
`resolve_mask` | Resolve a mask spec into a Nifti1Image.



####### Functions##

(data-coerce-metadata)=
###### `coerce_metadata`

```python
coerce_metadata(metadata: pl.DataFrame | pd.DataFrame | dict | None, n_subjects: int) -> pl.DataFrame
```

Coerce a metadata input into a polars DataFrame of length ``n_subjects``.

Accepts polars/pandas DataFrames or a dict-of-columns. ``None`` yields a
DataFrame with a default ``subject`` column (``sub-0001``, ...).

Polars ``metadata`` cannot hold DataFrames or arrays — those belong in
the parallel slots (``designs``, ``_confounds``, ``_sample_masks``).

######## `make_run_id`

```python
make_run_id(now: datetime | None = None) -> str
```

Build a fresh ``run_id`` of the form ``{timestamp}_{uuid8}``.

Timestamp is UTC ``YYYYMMDDTHHMMSS``; the uuid tail is 8 hex chars from
``secrets.token_hex(4)``. Lex-sortable, collision-free across processes.

######## `make_step_dirname`

```python
make_step_dirname(op: str, kwargs: dict[str, Any] | None = None, *, now: datetime | None = None) -> str
```

Name a step subdir: ``{timestamp}_{seq}_{uuid8}_{op}_{key_kwargs}/``.

Each call yields a unique name (UUID tail) — same op + same params
twice produces two subdirs, never overwriting. The zero-padded ``seq`` is
a process-monotonic counter placed after the second-resolution timestamp,
so lexicographic order tracks creation order even for calls that share a
second (the timestamp stays the primary, cross-process ordering key).

######## `resolve_cache_dir`

```python
resolve_cache_dir(cache_dir: Path | str | None) -> Path | None
```

Resolve ``cache_dir`` per the spec's precedence rules.

Order: explicit arg → ``NLTOOLS_CACHE_DIR`` env var → ``./.nltools_cache``.
Returns ``None`` when the caller passes ``None`` (signaling tempdir mode).
The returned path is *not* yet decorated with a ``run_id`` subdir; that
happens at construction time on the instance.

######## `resolve_mask`

```python
resolve_mask(mask: nib.Nifti1Image | Path | str) -> nib.Nifti1Image
```

Resolve a mask spec into a Nifti1Image.

Accepts a Nifti1Image, a path, or a known nltools template string
(e.g. ``"3mm-MNI152-2009c"``). String templates dispatch to the same
resolver used by ``BrainData``.

(data-execution)=
###### `execution`

Parallel execution machinery for BrainCollection.

Holds the worker-side dataclasses (``_ItemTask``, ``_DesignContext``), the
single parallel primitive (``_apply``), the worker-error type, and the
HDF5 fit-bundle IO. Every per-subject method on ``BrainCollection`` routes
through ``_apply`` here.

**Classes:**

Name | Description
---- | -----------
[`BrainCollectionWorkerError`](#data-braincollectionworkererror) | Raised in the parent process when a worker fails inside ``_apply``.
`tqdm_joblib` | Context manager that updates a tqdm bar as joblib workers complete.

**Methods:**

Name | Description
---- | -----------
[`read_glm_bundle`](#data-read-glm-bundle) | Read a GLM bundle. Validates ``bundle_schema_version``.
`read_ridge_bundle` | Read a ridge bundle. Same schema/version handling as ``read_glm_bundle``.
`write_glm_bundle` | Write a GLM fit bundle to ``out_path`` (atomic tmp+rename).
`write_ridge_bundle` | Write a ridge fit bundle to ``out_path`` (atomic tmp+rename).

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`BUNDLE_SCHEMA_VERSION`](#data-bundle-schema-version) |  | 



####### Attributes##

(data-bundle-schema-version)=
###### `BUNDLE_SCHEMA_VERSION`

```python
BUNDLE_SCHEMA_VERSION = 2
```



####### Classes##

###### `BrainCollectionWorkerError`

Bases: <code>[RuntimeError](#RuntimeError)</code>

Raised in the parent process when a worker fails inside ``_apply``.

Wraps the original exception via ``raise ... from e`` so the full
traceback is preserved. The message embeds subject/run context from
``_ItemTask.metadata_row`` so users can locate the offending item.

######## `tqdm_joblib`

```python
tqdm_joblib(total: int, desc: str = '', disable: bool = False) -> None
```

Context manager that updates a tqdm bar as joblib workers complete.

Replaces today's submit-time wrapper, which advances on dispatch rather
than completion. Monkey-patches ``BatchCompletionCallBack.__call__`` for
the duration of the ``with`` block.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`desc`](#data-desc) |  | 
`disable` |  | 
`total` |  | 



######### Attributes####

(data-desc)=
###### `desc`

```python
desc = desc
```

########## `disable`

```python
disable = disable
```

########## `total`

```python
total = total
```



####### Functions##

(data-read-glm-bundle)=
###### `read_glm_bundle`

```python
read_glm_bundle(path: Path) -> dict[str, Any]
```

Read a GLM bundle. Validates ``bundle_schema_version``.

Schema-version mismatch raises with a migration message; nltools-version
mismatch logs a warning but does not refuse — bundles are usually
forward-compatible within a minor version.

######## `read_ridge_bundle`

```python
read_ridge_bundle(path: Path) -> dict[str, Any]
```

Read a ridge bundle. Same schema/version handling as ``read_glm_bundle``.

######## `write_glm_bundle`

```python
write_glm_bundle(out_path: Path, *, betas: np.ndarray, residuals: np.ndarray, sigma2: np.ndarray, r2: np.ndarray, X: np.ndarray, mask_bytes: bytes, affine: np.ndarray, regressor_names: list[str], scale: bool, standardize: str | None, model_kwargs: dict, step_id: str, parent_step_id: str | None, op: str, op_kwargs: dict, nltools_version: str) -> Path
```

Write a GLM fit bundle to ``out_path`` (atomic tmp+rename).

Layout (see SPEC §"HDF5 fit bundle"):
    /betas, /residuals, /sigma2, /r2, /X, /mask
    attrs: affine, regressor_names, scale, standardize, model_kwargs,
           nltools_version, bundle_schema_version,
           step_id, parent_step_id, op, kwargs (JSON-encoded).

Mask is embedded as a dataset (raw NIfTI bytes) so the bundle is
portable across machines. Uses ``h5py.File(..., locking=False)``.

######## `write_ridge_bundle`

```python
write_ridge_bundle(out_path: Path, *, weights: np.ndarray, intercept: np.ndarray, cv_scores: np.ndarray, predictions: np.ndarray, scores: np.ndarray, X: np.ndarray, mask_bytes: bytes, affine: np.ndarray, regressor_names: list[str], model_kwargs: dict, step_id: str, parent_step_id: str | None, op: str, op_kwargs: dict, nltools_version: str) -> Path
```

Write a ridge fit bundle to ``out_path`` (atomic tmp+rename).

Parallel layout to ``write_glm_bundle`` with ridge-specific datasets
(``weights``, ``intercept``, ``cv_scores``, ``predictions``, ``scores``).



####### Modules

(data-inference)=
###### `inference`

Group-level reductions and cross-subject ops for BrainCollection.

Module-level functions that the ``BrainCollection`` facade delegates to.
Reductions stream from path-backed inputs (Welford-style) and produce
in-memory ``BrainData`` (or dicts of them); they never path-back their
own output.

**Methods:**

Name | Description
---- | -----------
[`align`](#data-align) | Functional alignment via ``LocalAlignment``.
[`anova`](#data-anova) | One-way ANOVA across subjects.
[`concat`](#data-concat) | Stack along axis 0 → ``BrainData`` of shape ``(n_total_obs, n_voxels)``.
[`isc`](#data-isc) | Inter-subject correlation across the time dimension.
[`isc_test`](#data-isc-test) | Bootstrap inference on ISC.
`max_` | Per-voxel max across subjects. Streams.
[`mean`](#data-mean) | Mean across subjects (leading axis). Streams from path-backed input.
[`median`](#data-median) | Median across subjects. Materializes (not streaming-friendly).
`min_` | Per-voxel min across subjects. Streams.
[`permutation_test`](#data-permutation-test) | Sign-flipping permutation test across subjects (one-sample).
[`permutation_test2`](#data-permutation-test2) | Two-sample permutation test by random label shuffling.
[`std`](#data-std) | Std across subjects. Streams via Welford; ddof=1.
`sum_` | Sum across subjects. Streams.
[`ttest`](#data-ttest) | One-sample t-test across subjects.
[`ttest2`](#data-ttest2) | Two-sample t-test between two collections (subject-level).
[`var`](#data-var) | Variance across subjects. Streams via Welford; ddof=1.



####### Classes

####### Functions##

###### `align`

```python
align(bc: BrainCollection, *, method: str = 'procrustes', spatial_scale: str = 'searchlight', radius_mm: float = 10.0, roi_mask: nib.Nifti1Image | None = None, n_features: int | None = None, n_iter: int = 3, device: str = 'cpu', return_model: bool = False, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto')
```

Functional alignment via ``LocalAlignment``.

Materializes all subjects (algorithm constraint in v0.6.0). Returns
a new ``BrainCollection`` of aligned data, or
``(BrainCollection, LocalAlignment)`` when ``return_model=True``.

######## `anova`

```python
anova(bc: BrainCollection, groups: str | list | np.ndarray) -> dict[str, BrainData | int]
```

One-way ANOVA across subjects.

``groups`` is a metadata column name, a list, or an ndarray of length
``n_subjects``. Returns ``{'F', 'p', 'df_between', 'df_within'}``.

######## `concat`

```python
concat(bc: BrainCollection) -> BrainData
```

Stack along axis 0 → ``BrainData`` of shape ``(n_total_obs, n_voxels)``.

Not streamable — the operation *is* materialization. 1D items are
promoted to ``(1, n_voxels)`` before concatenation.

######## `isc`

```python
isc(bc: BrainCollection, *, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, metric: str = 'median') -> dict
```

Inter-subject correlation across the time dimension.

method='loo' uses the leave-one-out template approach (each subject
correlated with the average of the others). method='pairwise' computes
all subject pairs. Both materialize all subjects in v0.6.0; the
streaming rewrite is deferred to a later release.

Passing ``roi_mask`` restricts the computation to that ROI; the returned
maps carry the ROI mask rather than the collection's whole-brain mask.

Returns ``{'isc', 'per_subject'}`` for ``loo`` or ``{'isc', 'pairs'}``
for ``pairwise``.

######## `isc_test`

```python
isc_test(bc: BrainCollection, *, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, n_samples: int = 5000, metric: str = 'median', random_state: int | None = None) -> dict
```

Bootstrap inference on ISC.

Resamples subjects with replacement, recomputes ISC each draw, and
derives a per-voxel p-value from the null distribution centered at 0.

Passing ``roi_mask`` restricts the computation to that ROI; the returned
maps carry the ROI mask rather than the collection's whole-brain mask.

######## `max_`

```python
max_(bc: BrainCollection) -> BrainData
```

Per-voxel max across subjects. Streams.

######## `mean`

```python
mean(bc: BrainCollection) -> BrainData
```

Mean across subjects (leading axis). Streams from path-backed input.

######## `median`

```python
median(bc: BrainCollection) -> BrainData
```

Median across subjects. Materializes (not streaming-friendly).

######## `min_`

```python
min_(bc: BrainCollection) -> BrainData
```

Per-voxel min across subjects. Streams.

######## `permutation_test`

```python
permutation_test(bc: BrainCollection, *, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

Sign-flipping permutation test across subjects (one-sample).

Per SPEC streaming-algorithms table, sign-flipping needs all subjects
in memory by design. ``device`` is currently informational; backend
selection is deferred to the parametric stats path.

######## `permutation_test2`

```python
permutation_test2(bc: BrainCollection, other: BrainCollection, *, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

Two-sample permutation test by random label shuffling.

######## `std`

```python
std(bc: BrainCollection) -> BrainData
```

Std across subjects. Streams via Welford; ddof=1.

######## `sum_`

```python
sum_(bc: BrainCollection) -> BrainData
```

Sum across subjects. Streams.

######## `ttest`

```python
ttest(bc: BrainCollection, *, popmean: float = 0.0) -> dict[str, BrainData]
```

One-sample t-test across subjects.

Returns ``{'mean', 't', 'z', 'p'}`` — same shape contract as
``BrainData.ttest``. Streams from path-backed input via Welford.

######## `ttest2`

```python
ttest2(bc: BrainCollection, other: BrainCollection, *, equal_var: bool = True) -> dict[str, BrainData]
```

Two-sample t-test between two collections (subject-level).

######## `var`

```python
var(bc: BrainCollection) -> BrainData
```

Variance across subjects. Streams via Welford; ddof=1.

###### `io`

IO and constructors for BrainCollection.

Constructors (``from_bids``, ``from_glob``, ``from_paths``, ``read``),
write, load/unload, cache plumbing, and ``memory_estimate``. Anything that
crosses the disk boundary lives here.

**Methods:**

Name | Description
---- | -----------
[`discover_bids`](#data-discover-bids) | Walk the BIDS dataset and return aligned per-item lists.
[`from_bids`](#data-from-bids) | Build a ``BrainCollection`` from a BIDS dataset.
[`from_glob`](#data-from-glob) | Build a collection by globbing for BOLD images (and optionally designs).
[`from_paths`](#data-from-paths) | Build a collection from explicit lists of brain (and design) paths.
[`load`](#data-load) | Materialize path-backed items into ``BrainData``.
[`memory_estimate`](#data-memory-estimate) | Human-readable RAM estimate if every item were loaded.
[`read`](#data-read) | Inverse of ``write()``: read images + ``metadata.csv`` from ``directory``.
[`unload`](#data-unload) | Drop in-memory data for items that have backing paths.
[`write`](#data-write) | Write a clean, portable copy of ``bc`` outside the cache root.



####### Classes

####### Functions##

(data-discover-bids)=
###### `discover_bids`

```python
discover_bids(root: Path | str | Any, *, task: str | None, space: str | None, sub_labels: list[str] | None, img_filters: list[tuple[str, str]] | None, derivatives_folder: str, confounds_strategy: str | tuple[str, ...] | None, confounds_kwargs: dict | None, TR: float | str) -> dict[str, list]
```

Walk the BIDS dataset and return aligned per-item lists.

Returns a dict with keys: ``bold_paths``, ``events_dfs``, ``confounds_dfs``,
``sample_masks``, ``metadata_rows``, ``TRs``. Each list is the same length
(one entry per BOLD file). Anything missing for an item is ``None``.

Errors per SPEC §"Edge cases / errors":
  - Missing TR with ``TR='infer'``: raise.
  - ``task=None`` + ``pair_events=True``: caller silently downgrades.
  - fmriprep absent + ``confounds_strategy`` set: raise.
  - pybids not installed: raise ``ImportError``.

######## `from_bids`

```python
from_bids(cls: type[BrainCollection], root: Path | str | Any, *, mask: nib.Nifti1Image | Path | str, task: str | None = None, space: str | None = None, sub_labels: list[str] | None = None, img_filters: list[tuple[str, str]] | None = None, derivatives_folder: str = 'derivatives', pair_events: bool = True, confounds_strategy: str | tuple[str, ...] | None = None, confounds_kwargs: dict | None = None, TR: float | str = 'infer', cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Build a ``BrainCollection`` from a BIDS dataset.

Delegates discovery to ``nilearn.glm.first_level.first_level_from_bids``
(which wraps pybids), drops the returned ``models``, and keeps paths +
events/confounds DataFrames. Per-item ``DesignMatrix`` is built from the
events DataFrame; convolution / drift / confound merging is **not** done
here — that's the user's ``transform_designs`` step.

See SPEC §"``from_bids`` — concrete design" for edge cases.

######## `from_glob`

```python
from_glob(cls: type[BrainCollection], pattern: str, *, mask: nib.Nifti1Image | Path | str, design_pattern: str | None = None, pattern_groups: dict[str, int] | str | None = None, sort: bool = True, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Build a collection by globbing for BOLD images (and optionally designs).

``pattern_groups`` extracts metadata from filename wildcards. Pass
``{column_name: wildcard_index}`` (0-based) to capture each ``*`` in
``pattern`` into a metadata column.

######## `from_paths`

```python
from_paths(cls: type[BrainCollection], brain_paths: list[Path | str], *, mask: nib.Nifti1Image | Path | str, design_paths: list[Path | str | None] | None = None, metadata: pl.DataFrame | pd.DataFrame | dict | None = None, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Build a collection from explicit lists of brain (and design) paths.

Always lazy — items are stored as ``Path`` and loaded on demand.

######## `load`

```python
load(bc: BrainCollection, indices: list[int] | None = None) -> BrainCollection
```

Materialize path-backed items into ``BrainData``.

Mutates ``bc`` in place. This is the only mutation method besides
``unload`` and does not allocate a step
subdir, does not write to disk, does not produce a new identity.

######## `memory_estimate`

```python
memory_estimate(bc: BrainCollection) -> str
```

Human-readable RAM estimate if every item were loaded.

Reports ``n_subjects``, the per-item shape (or "unknown" if path-backed
and not yet loaded), and an estimated total in MB/GB based on float32.

######## `read`

```python
read(cls: type[BrainCollection], directory: Path | str, *, mask: nib.Nifti1Image | Path | str, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Inverse of ``write()``: read images + ``metadata.csv`` from ``directory``.

Discovers items by globbing ``image_*.nii*`` (matches the ``write()``
default pattern) and pairs them with rows from ``metadata.csv`` if it
exists. Does **not** recover from cache subdirs in v0.6.0.

######## `unload`

```python
unload(bc: BrainCollection, indices: list[int] | None = None) -> BrainCollection
```

Drop in-memory data for items that have backing paths.

Mutates in place. This is a no-op for items that don't have a backing path
because dropping them would lose data.

######## `write`

```python
write(bc: BrainCollection, directory: Path | str, *, pattern: str = 'image_{i:04d}.nii.gz', metadata_file: str | None = 'metadata.csv') -> list[Path]
```

Write a clean, portable copy of ``bc`` outside the cache root.

Inverse of ``BrainCollection.read()``. Writes one NIfTI per item under
``directory`` plus a metadata CSV. Skips the cache layout entirely so
the result is shareable / archival.

(data-pipeline)=
###### `pipeline`

Pipeline classes for BrainCollection.

Provides BrainCollectionPipeline for a fluent pipeline API with
cross-validation. CV-aware ``predict()`` returns a ``BrainData`` with CV
attributes attached (``cv_scores``, ``cv_predictions``, ``mean_score``,
``std_score``, ``fold_results``, ``cv_pipeline``).

**Classes:**

Name | Description
---- | -----------
[`BrainCollectionPipeline`](#data-braincollectionpipeline) | Pipeline for BrainCollection with multi-subject CV support.



####### Classes##

###### `BrainCollectionPipeline`

```python
BrainCollectionPipeline(brain_collection: BrainCollection, cv: BrainCollection = None, groups: np.ndarray | None = None)
```

Pipeline for BrainCollection with multi-subject CV support.

Wraps BrainCollection to provide fluent pipeline API with LOSO
and run-based cross-validation.

This class enables method chaining for preprocessing and prediction
with proper cross-validation semantics for multi-subject neuroimaging
analyses.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`n_subjects` | <code>[int](#int)</code> | Number of subjects/images in the collection.
[`cv`](#data-cv) |  | The cross-validation scheme configuration.
`n_steps` | <code>[int](#int)</code> | Number of transform steps in the pipeline.

**Examples:**

```pycon
>>> # Leave-one-subject-out with preprocessing
>>> result = (bc
...     .cv(method='loso')
...     .standardize()
...     .reduce(n_components=50)
...     .predict(labels, method='svm'))
>>> print(f"Mean accuracy: {result.mean_score:.2%}")
```

**Methods:**

Name | Description
---- | -----------
[`pipe`](#data-pipe) | Add custom sklearn transformer.
[`predict`](#data-predict) | Execute pipeline with CV and return prediction results.
`reduce` | Add dimensionality reduction step.
[`standardize`](#data-standardize) | Add standardization step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_collection` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | BrainCollection to wrap. | *required*
`cv` |  | CVScheme configuration. | <code>None</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Group labels for CV splits. | <code>None</code>



######### Attributes####

###### `cv`

```python
cv
```

Cross-validation scheme.

########## `n_steps`

```python
n_steps: int
```

Number of transform steps.

########## `n_subjects`

```python
n_subjects: int
```

Number of subjects/images.



######### Functions####

###### `pipe`

```python
pipe(transformer) -> BrainCollectionPipeline
```

Add custom sklearn transformer.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`transformer` |  | sklearn-compatible transformer with fit/transform interface. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollectionPipeline](#nltools.data.collection.pipeline.BrainCollectionPipeline)</code> | New pipeline with custom step added.

########## `predict`

```python
predict(y, method: str = 'ridge', *, n_permute: int = 0, random_state: int = None, **kwargs: int)
```

Execute pipeline with CV and return prediction results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` |  | Target variable. For LOSO, shape should be (n_subjects,). | *required*
`method` | <code>[str](#str)</code> | Prediction algorithm ('ridge', 'svm', 'logistic', etc.) | <code>'ridge'</code>
`n_permute` | <code>[int](#int)</code> | If ``> 0``, also build a label-permutation null of the CV score — the classic MVPA permutation test. Each iteration shuffles ``y``, re-runs the *same* cross-validation, and records the mean score; the result gets ``permutation_scores`` (the null array) and ``permutation_pvalue`` attached. Default 0 (no null). | <code>0</code>
`random_state` |  | Seed for the label shuffling (permutation null only). | <code>None</code>
`**kwargs` |  | Passed to model constructor. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | ``BrainData`` carrying out-of-fold predictions plus CV attributes
 | (``cv_scores``, ``cv_predictions``, ``mean_score``, ``std_score``,
 | ``fold_results``, ``cv_pipeline``). When ``n_permute > 0`` it also
 | carries ``permutation_scores`` and ``permutation_pvalue``.

########## `reduce`

```python
reduce(method: str = 'pca', n_components: int | None = None, **kwargs: int | None) -> BrainCollectionPipeline
```

Add dimensionality reduction step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Reduction method ('pca', 'ica'). | <code>'pca'</code>
`n_components` | <code>[int](#int) \| None</code> | Number of components to keep. | <code>None</code>
`**kwargs` |  | Additional arguments for ReduceStep. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollectionPipeline](#nltools.data.collection.pipeline.BrainCollectionPipeline)</code> | New pipeline with reduction step added.

########## `standardize`

```python
standardize(method: str = 'zscore', **kwargs: str) -> BrainCollectionPipeline
```

Add standardization step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Standardization method ('zscore', 'minmax'). | <code>'zscore'</code>
`**kwargs` |  | Additional arguments for NormalizeStep. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollectionPipeline](#nltools.data.collection.pipeline.BrainCollectionPipeline)</code> | New pipeline with standardization step added.

#### `designmatrix`

Provide a Polars-based design matrix for neuroimaging analysis.

Efficient design matrix implementation using Polars for fast DataFrame operations.
Provides HRF convolution, resampling, polynomial regressors, and diagnostic tools.

Uses composition pattern (wrapping pl.DataFrame) for clean metadata preservation.

**Modules:**

Name | Description
---- | -----------
[`append`](#data-append) | Provide standalone DesignMatrix concatenation functions.
[`diagnostics`](#data-diagnostics) | Diagnostic and utility functions for DesignMatrix.
[`io`](#data-io) | Provide DesignMatrix I/O and visualization functions.
[`plotting`](#data-plotting) | DesignMatrix visualization functions.
[`regressors`](#data-regressors) | Provide standalone regressor functions for DesignMatrix.
[`transforms`](#data-transforms) | Standalone transform functions for DesignMatrix.
[`utils`](#data-utils) | Shared helpers for DesignMatrix submodules.

**Classes:**

Name | Description
---- | -----------
[`DesignMatrix`](#data-designmatrix) | Represent experimental designs for neuroimaging with Polars.



##### Classes

###### `DesignMatrix`

```python
DesignMatrix(data: DesignMatrix | pl.DataFrame | pd.DataFrame | np.ndarray | dict | str | Path | None = None, *, sampling_freq: float | None = None, TR: float | None = None, run_length: int | str | None = None, columns: list[str] | None = None, convolved: list[str] | None = None, confounds: list[str] | None = None, hrf_model: str | None = 'glover')
```

Represent experimental designs for neuroimaging with Polars.

This is a Polars-based design matrix for experimental designs in
neuroimaging.

Wraps a Polars DataFrame with neuroimaging-specific metadata and methods.
Uses composition pattern (not subclassing) for clean metadata preservation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>DataFrame, ndarray, dict, str/Path, or None</code> | Input data. Accepts: - Polars DataFrame (zero-copy) - pandas DataFrame (converted to Polars) - numpy ndarray - dict (keys=columns, values=data) - str or Path to a `.tsv`/`.csv` file. BIDS events files   (containing `onset` and `duration` columns) are converted to   boxcar regressors — call ``convolve()`` afterwards if you want   HRF convolution. Any other tabular file is read as-is and is   typically used for confounds. - None (empty initialization) | <code>None</code>
`sampling_freq` | <code>[float](#float)</code> | Sampling frequency in Hz (1/TR for fMRI data). Mutually exclusive with ``TR``. | <code>None</code>
`TR` | <code>[float](#float)</code> | Repetition time in seconds. Convenience for ``sampling_freq = 1/TR``. Mutually exclusive with ``sampling_freq``. | <code>None</code>
`run_length` | <code>[int](#int) or 'infer'</code> | Required when ``data`` is a file path. Number of TRs in the run. Pass ``'infer'`` for tabular/confounds files to accept whatever row count the file has (not valid for events files). | <code>None</code>
`columns` | <code>list of str</code> | Column names (used with ndarray input) | <code>None</code>
`convolved` | <code>list of str</code> | Names of convolved columns (tracked internally) | <code>None</code>
`confounds` | <code>list of str</code> | Names of nuisance/confound columns (intercept, polynomial drift, DCT cosines, motion, …) tracked internally | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`sampling_freq` | <code>[float](#float) or None</code> | Sampling frequency in Hz
`convolved` | <code>list of str</code> | Columns that have been convolved
`confounds` | <code>list of str</code> | Nuisance/confound columns (intercept, polynomial trends, DCT bases, motion, physio, …) — these are skipped by ``.convolve()`` and kept separate per run on multi-run vertical append.
`multi` | <code>[bool](#bool)</code> | True if created from multi-run concatenation

**Examples:**

```pycon
>>> # Create from numpy array
>>> dm = DesignMatrix(np.zeros((100, 2)), sampling_freq=0.5, columns=['a', 'b'])
```

```pycon
>>> # Add columns
>>> dm['stim'] = [0, 1, 1, 0] * 25
```

```pycon
>>> # Convolve with HRF — convolved columns get a `_c0` suffix
>>> dm_conv = dm.convolve()  # 'stim' → 'stim_c0'
```

```pycon
>>> # Add polynomial drift terms
>>> dm_conv = dm_conv.add_poly(order=2)
```

```pycon
>>> # Multi-run concatenation (auto-separates polynomials)
>>> dm_run1 = DesignMatrix(...).add_poly(0)
>>> dm_run2 = DesignMatrix(...).add_poly(0)
>>> dm_multi = dm_run1.append(dm_run2, axis=0)  # Creates 0_poly_0, 1_poly_0
```

**Methods:**

Name | Description
---- | -----------
[`add_dct_basis`](#data-add-dct-basis) | Add discrete cosine transform basis functions for high-pass filtering.
[`add_poly`](#data-add-poly) | Add Legendre polynomial drift terms.
[`append`](#data-append) | Concatenate design matrices.
[`clean`](#data-clean) | Remove highly correlated columns.
[`convolve`](#data-convolve) | Convolve columns with an HRF or custom kernel.
[`copy`](#data-copy) | Create a deep copy of the DesignMatrix.
[`corr`](#data-corr) | Calculate column correlations as a similarity ``Adjacency``.
[`downsample`](#data-downsample) | Reduce temporal resolution using Polars-native operations.
[`drop`](#data-drop) | Drop specified columns.
[`fillna`](#data-fillna) | Fill NaN/null values with specified value.
[`plot`](#data-plot) | Visualize the design matrix.
[`replace_data`](#data-replace-data) | Replace data columns while preserving confounds and metadata.
[`standardize`](#data-standardize) | Standardize columns using the specified method.
[`sum`](#data-sum) | Compute the sum along an axis.
[`to_numpy`](#data-to-numpy) | Convert a DesignMatrix to a NumPy array.
[`to_pandas`](#data-to-pandas) | Convert DesignMatrix to pandas DataFrame.
[`upsample`](#data-upsample) | Increase temporal resolution to a target frequency.
[`vif`](#data-vif) | Compute the variance inflation factor for each column.
[`with_columns`](#data-with-columns) | Add or replace columns via Polars expressions.
[`write`](#data-write) | Write DesignMatrix to file.
[`zscore`](#data-zscore) | Z-score standardize columns to mean zero and unit variance.

Passing another ``DesignMatrix`` returns a copy: ``data``,
``sampling_freq``, ``convolved``, ``confounds``, and ``multi`` are
carried over. Any explicit kwarg overrides the inherited value.

When ``data`` is a path to a BIDS events file, the constructor
HRF-convolves the regressors by default (``hrf_model='glover'``,
matching nilearn's ``make_first_level_design_matrix``). The output
columns are suffixed ``_c0`` and ``.convolved`` is populated. Pass
``hrf_model=None`` to load raw boxcar regressors instead — useful
for FIR designs, PPI flows that build interaction terms before
convolution, or pedagogical material that introduces convolution
as a separate step. ``hrf_model`` is silently ignored when ``data``
is anything other than an events file.



####### Attributes##

(data-columns)=
###### `columns`

```python
columns: list[str]
```

Column names of the design matrix as a list of strings.

######## `confounds`

```python
confounds: list[str]
```

Names of nuisance/confound columns (read-only).

Managed by ``.convolve()``, ``.append()``, ``.add_poly()``,
``.add_dct_basis()``, and the ``confounds=`` constructor kwarg. Direct
assignment raises ``AttributeError`` — pass via the constructor or use
``.append(other, axis=1)`` (which auto-tracks confounds when ``other``
is a raw pandas/polars DataFrame).

######## `convolved`

```python
convolved: list[str]
```

Names of HRF-convolved columns (read-only).

Managed by ``.convolve()`` and ``.append()`` (merges across inputs).
Direct assignment raises ``AttributeError`` — pass via the
``convolved=`` constructor kwarg if you need to set initial state.

######## `data`

```python
data = data.data.clone()
```

######## `is_empty`

```python
is_empty: bool
```

Check if DesignMatrix has no data.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`bool` | <code>[bool](#bool)</code> | True if the design matrix is empty, False otherwise.

######## `multi`

```python
multi = False
```

######## `sampling_freq`

```python
sampling_freq = sampling_freq
```

######## `shape`

```python
shape: tuple
```

Return (n_rows, n_cols) tuple.



####### Functions##

###### `add_dct_basis`

```python
add_dct_basis(duration: float = 180, drop: int = 0, *, include_constant: bool = True) -> DesignMatrix
```

Add discrete cosine transform basis functions for high-pass filtering.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`duration` | <code>[float](#float)</code> | Filter duration in seconds. Default: 180. | <code>180</code>
`drop` | <code>[int](#int)</code> | Number of low-frequency bases to drop. Default: 0. | <code>0</code>
`include_constant` | <code>[bool](#bool)</code> | If True, also add a constant/intercept column named ``cosine_0`` (analogous to ``poly_0`` in `add_poly`). The underlying DCT basis drops the constant per SPM convention; set False to match SPM behavior. Default: True. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with DCT basis columns appended.

######## `add_poly`

```python
add_poly(order: int = 0, include_lower: bool = True) -> DesignMatrix
```

Add Legendre polynomial drift terms.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`order` | <code>[int](#int)</code> | Polynomial order (0=intercept, 1=linear, 2=quadratic, ...). Default: 0. | <code>0</code>
`include_lower` | <code>[bool](#bool)</code> | If True, include all orders from 0 to order. Default: True. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with polynomial columns appended.

######## `append`

```python
append(dm: DesignMatrix | list[DesignMatrix], *, axis: int = 0, keep_separate: bool = True, unique_cols: list[str] | None = None, fill_na: int | float = 0, as_confounds: bool = False, progress_bar: bool = False) -> DesignMatrix
```

Concatenate design matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>DesignMatrix or list of DesignMatrix</code> | Design matrix/matrices to append. | *required*
`axis` | <code>[int](#int)</code> | 0 for row-wise (vertical), 1 for column-wise (horizontal). Default: 0. | <code>0</code>
`keep_separate` | <code>[bool](#bool)</code> | Whether to separate confound columns across runs (only applies when axis=0). Default: True. | <code>True</code>
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | <code>None</code>
`fill_na` | <code>[int](#int) or [float](#float)</code> | Value to fill NaN values during vertical concatenation. Default: 0. | <code>0</code>
`as_confounds` | <code>[bool](#bool)</code> | Only applies when ``axis=1``. If True, mark all columns from ``dm`` as nuisance/confounds in the result — they get skipped by ``.convolve()`` and separated across runs on later vertical appends. Default: False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Print messages about confound separation. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Concatenated design matrix.

######## `clean`

```python
clean(*, fill_na: int | float | None = 0, exclude_confounds: bool = False, thresh: float = 0.95, progress_bar: bool = False) -> DesignMatrix
```

Remove highly correlated columns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fill_na` | <code>int, float, or None</code> | Fill NaN values before checking correlations (default 0) | <code>0</code>
`exclude_confounds` | <code>[bool](#bool)</code> | Skip confound/nuisance columns from correlation check | <code>False</code>
`thresh` | <code>[float](#float)</code> | Correlation threshold (drop if abs(r) >= thresh, default 0.95) | <code>0.95</code>
`progress_bar` | <code>[bool](#bool)</code> | Print dropped column names. Default: False | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Cleaned matrix with highly correlated columns removed

######## `convolve`

```python
convolve(conv_func: str | np.ndarray = 'hrf', columns: list[str] | None = None) -> DesignMatrix
```

Convolve columns with an HRF or custom kernel.

Convolved columns are always renamed to ``<col>_c{i}`` (where ``i`` is
the kernel index, ``0`` for a single 1-D kernel). The source columns
are dropped, and ``self.convolved`` lists the post-suffix names so
downstream metadata stays in sync with the dataframe.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`conv_func` | <code>[str](#str) or [ndarray](#ndarray)</code> | 'hrf' for canonical Glover HRF, or custom kernel(s). Can be 1D array (single kernel) or 2D (samples x kernels). | <code>'hrf'</code>
`columns` | <code>list of str</code> | Columns to convolve (default: all non-confound columns). | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with convolved columns renamed.

######## `copy`

```python
copy() -> DesignMatrix
```

Create a deep copy of the DesignMatrix.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Copy of the current DesignMatrix

######## `corr`

```python
corr(*, metric: str = 'pearson', columns: list[str] | None = None)
```

Calculate column correlations as a similarity ``Adjacency``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` | <code>[str](#str)</code> | ``'pearson'`` (default) or ``'spearman'``. | <code>'pearson'</code>
`columns` | <code>list of str</code> | Subset of columns to correlate. Defaults to all columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | Similarity matrix whose ``labels`` are the column names. The unit diagonal is dropped (self-correlation isn't an edge); use ``.plot(method='corr')`` for a heatmap with the diagonal restored.

######## `downsample`

```python
downsample(target: float, method: str = 'mean') -> DesignMatrix
```

Reduce temporal resolution using Polars-native operations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` | <code>[float](#float)</code> | Target sampling frequency in Hz (must be < current sampling_freq) | *required*
`method` | <code>[str](#str)</code> | Aggregation method - 'mean' or 'median' (default: 'mean') | <code>'mean'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Downsampled DesignMatrix with updated sampling_freq

######## `drop`

```python
drop(columns: list[str]) -> DesignMatrix
```

Drop specified columns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`columns` | <code>list of str</code> | Column names to remove. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix without the specified columns.

######## `fillna`

```python
fillna(value: int | float) -> DesignMatrix
```

Fill NaN/null values with specified value.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`value` | <code>[int](#int) or [float](#float)</code> | Value to replace NaN/null entries with. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with NaN/null values replaced.

######## `plot`

```python
plot(method: str = 'matrix', *, columns: list[str] | None = None, rescale: bool = True, metric: str = 'pearson', ax: str = None, figsize: tuple | None = None, title: str | None = None, cmap: str | None = None, save: str | None = None, **kwargs: str | None)
```

Visualize the design matrix.

Dispatches over ``method`` (mirroring ``BrainData.plot``):

- ``'matrix'`` (default): SPM-style heatmap (rows=TRs, cols=regressors).
- ``'timeseries'``: overlaid line plot of regressor time courses. Pass
  the same ``ax`` across calls to overlay multiple DesignMatrices
  (e.g. original vs. convolved).
- ``'corr'``: labeled correlation heatmap of the columns (reuses
  `corr`; diagonal restored to 1.0 for display).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ``'matrix'`` | ``'timeseries'`` | ``'corr'``. Default: ``'matrix'``. | <code>'matrix'</code>
`columns` | <code>list of str</code> | Subset of columns to plot. Defaults to all columns. | <code>None</code>
`rescale` | <code>[bool](#bool)</code> | ``'matrix'`` only. Rescale each column by its L2 norm so columns with different native magnitudes are visually comparable (SPM/nilearn convention). Default: True. | <code>True</code>
`metric` | <code>[str](#str)</code> | ``'corr'`` only. ``'pearson'`` (default) or ``'spearman'``. | <code>'pearson'</code>
`ax` | <code>[Axes](#matplotlib.axes.Axes)</code> | Existing axis to draw on; a new figure is created if omitted. | <code>None</code>
`figsize` | <code>[tuple](#tuple)</code> | Figure size; sensible per-method default when omitted. | <code>None</code>
`title` | <code>[str](#str)</code> | Axis title. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Colormap (``'matrix'`` / ``'corr'``). | <code>None</code>
`save` | <code>[str](#str)</code> | Path to save the figure. | <code>None</code>
`**kwargs` |  | Forwarded to the underlying plotter (``seaborn.heatmap`` for ``'matrix'`` / ``'corr'``; ``Axes.plot`` for ``'timeseries'``). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure: The figure containing the plot.

######## `replace_data`

```python
replace_data(data: np.ndarray, column_names: list[str] | None = None) -> DesignMatrix
```

Replace data columns while preserving confounds and metadata.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#ndarray)</code> | New data array (must match number of rows in current DesignMatrix) | *required*
`column_names` | <code>list of str</code> | Names for new data columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with replaced data columns, preserved confounds

######## `standardize`

```python
standardize(method: str = 'zscore', columns: list[str] | None = None) -> DesignMatrix
```

Standardize columns using the specified method.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Standardization method ('zscore' or 'center'). Default: 'zscore'. | <code>'zscore'</code>
`columns` | <code>[list](#list)[[str](#str)] \| None</code> | Columns to standardize. If None, standardize all non-confound columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with standardized columns.

######## `sum`

```python
sum(axis: int = 0) -> pl.Series
```

Compute the sum along an axis.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>int, default=0</code> | 0: sum down columns, 1: sum across rows. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>[Series](#polars.Series)</code> | pl.Series: Sums along specified axis.

######## `to_numpy`

```python
to_numpy() -> np.ndarray
```

Convert a DesignMatrix to a NumPy array.

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | np.ndarray: 2D array with shape (n_samples, n_columns)

######## `to_pandas`

```python
to_pandas() -> pd.DataFrame
```

Convert DesignMatrix to pandas DataFrame.

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#pandas.DataFrame)</code> | pd.DataFrame: Pandas DataFrame with same data and column names.

######## `upsample`

```python
upsample(target: float, method: str = 'linear') -> DesignMatrix
```

Increase temporal resolution to a target frequency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` | <code>[float](#float)</code> | Target sampling frequency in Hz (must be > current sampling_freq) | *required*
`method` | <code>[str](#str)</code> | Interpolation method - 'linear' or 'nearest' (default: 'linear') | <code>'linear'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Upsampled DesignMatrix with updated sampling_freq

######## `vif`

```python
vif(exclude_confounds: bool = True) -> np.ndarray | None
```

Compute the variance inflation factor for each column.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`exclude_confounds` | <code>[bool](#bool)</code> | Skip confound/nuisance columns. Default: True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray) \| None</code> | np.ndarray: VIF values for each included column. Returns None if the correlation matrix is singular.

######## `with_columns`

```python
with_columns(*exprs, **named_exprs) -> DesignMatrix
```

Add or replace columns via Polars expressions.

Mirrors `DataFrame.with_columns`. Named kwargs become
named columns; positional ``pl.Expr`` arguments are accepted as-is
(including ``pl.Expr.alias("name")``). Returns a new ``DesignMatrix``
with metadata preserved; new columns are *not* auto-tagged as
convolved or confounds.

For convenience, named-kwarg values that aren't ``pl.Expr`` /
``pl.Series`` are coerced:

- ``int``/``float`` → broadcast scalar via ``pl.lit``
- ``list`` / ``np.ndarray`` → wrapped as ``pl.Series``

**Examples:**

```pycon
>>> dm = dm.with_columns(motor=pl.sum_horizontal(motor_cols)).drop(motor_cols)
>>> dm = dm.with_columns(
...     vmpfc=seed_signal,
...     vmpfc_motor=pl.col("vmpfc") * pl.col("motor_c0"),
... )
```

######## `write`

```python
write(file_name: str, sep: str = '\t') -> None
```

Write DesignMatrix to file.

Supports TSV (default), CSV, and HDF5 formats. Format is
auto-detected from file extension.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>[str](#str)</code> | Output file path. Use .tsv, .csv, or .h5/.hdf5 extension. | *required*
`sep` | <code>[str](#str)</code> | Column separator for text files (default: tab). | <code>'\t'</code>

######## `zscore`

```python
zscore(columns: list[str] | None = None) -> DesignMatrix
```

Z-score standardize columns to mean zero and unit variance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`columns` | <code>list of str</code> | Columns to standardize. If None, standardize all non-confound columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with standardized columns



##### Methods

##### Modules

###### `append`

Provide standalone DesignMatrix concatenation functions.

These functions implement the append/concatenation logic extracted from
DesignMatrix methods, following the "functional core" pattern.

**Methods:**

Name | Description
---- | -----------
[`append`](#data-append) | Concatenate design matrices.
`append_horizontal` | Concatenate matrices horizontally by adding columns.
`append_vertical` | Concatenate matrices vertically with optional confound separation.
`append_vertical_with_separation` | Concatenate vertically with automatic confound separation.
`get_starting_run_idx` | Determine the next run index for multi-run appending.
`identify_columns_to_separate` | Identify columns that need run-specific separation.
`match_column_pattern` | Match columns against a pattern with wildcard support.



####### Classes

####### Functions##

###### `append`

```python
append(dm: DesignMatrix, other: DesignMatrix, *, axis: int = 0, keep_separate: bool = True, unique_cols: list[str] | None = None, fill_na: int | float | None = 0, as_confounds: bool = False, progress_bar: bool = False) -> DesignMatrix
```

Concatenate design matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | The base design matrix. | *required*
`other` | <code>DesignMatrix, DataFrame, or list</code> | Matrix/matrices to append. For ``axis=1`` (horizontal), also accepts a pandas or polars DataFrame (or list thereof); the new columns are treated as nuisance regressors (tracked in ``.confounds`` on the result). For ``axis=0`` (vertical), all items must be ``DesignMatrix``. | *required*
`axis` | <code>[int](#int)</code> | 0 for row-wise (vertical), 1 for column-wise (horizontal). | <code>0</code>
`keep_separate` | <code>[bool](#bool)</code> | Whether to separate confound columns across runs (only axis=0). | <code>True</code>
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | <code>None</code>
`fill_na` | <code>int, float, or None</code> | Value to fill NaN/null entries introduced by the concatenation. Pass ``None`` to preserve nulls. Default: 0. | <code>0</code>
`as_confounds` | <code>[bool](#bool)</code> | Only applies to ``axis=1``. When True, all columns contributed by ``other`` are tracked as nuisance regressors in the result's ``.confounds`` — so they're skipped by ``.convolve()`` and kept separate across runs in later vertical appends. Useful when ``other`` is a pre-built DesignMatrix of confounds that hasn't already marked its columns. Default: False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Print messages about confound separation. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Concatenated design matrix.

######## `append_horizontal`

```python
append_horizontal(dm: DesignMatrix, to_append: list[DesignMatrix], fill_na: int | float | None, as_confounds: bool = False) -> DesignMatrix
```

Concatenate matrices horizontally by adding columns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Base DesignMatrix instance. | *required*
`to_append` | <code>list of DesignMatrix</code> | Matrices whose columns to add. | *required*
`fill_na` | <code>int, float, or None</code> | Value to fill NaN/null entries with. Pass ``None`` to preserve nulls. | *required*
`as_confounds` | <code>[bool](#bool)</code> | If True, mark all columns contributed by ``to_append`` as nuisance/confounds in the result. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with columns from all matrices.

######## `append_vertical`

```python
append_vertical(dm: DesignMatrix, to_append: list[DesignMatrix], keep_separate: bool, unique_cols: list[str] | None, fill_na: int | float | None, progress_bar: bool) -> DesignMatrix
```

Concatenate matrices vertically with optional confound separation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Base DesignMatrix instance. | *required*
`to_append` | <code>list of DesignMatrix</code> | Matrices to stack below dm. | *required*
`keep_separate` | <code>[bool](#bool)</code> | Whether to separate confound columns across runs. | *required*
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | *required*
`fill_na` | <code>int, float, or None</code> | Value to fill NaN/null entries with. Pass ``None`` to preserve nulls. | *required*
`progress_bar` | <code>[bool](#bool)</code> | Print messages about confound separation. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with rows from all matrices.

######## `append_vertical_with_separation`

```python
append_vertical_with_separation(dm: DesignMatrix, to_append: list[DesignMatrix], unique_cols: list[str] | None, fill_na: int | float | None, progress_bar: bool) -> DesignMatrix
```

Concatenate vertically with automatic confound separation.

Creates run-specific columns (e.g., 0_poly_0, 1_poly_0) that are
active only in their respective runs (sparse representation).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Base DesignMatrix instance. | *required*
`to_append` | <code>list of DesignMatrix</code> | Matrices to stack below dm. | *required*
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | *required*
`fill_na` | <code>int, float, or None</code> | Value to fill NaN/null entries with. Pass ``None`` to preserve nulls. | *required*
`progress_bar` | <code>[bool](#bool)</code> | Print messages about confound separation. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Concatenated DesignMatrix with run-separated confound columns and multi=True.

######## `get_starting_run_idx`

```python
get_starting_run_idx(dm: DesignMatrix) -> int
```

Determine the next run index for multi-run appending.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance to inspect. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`int` | <code>[int](#int)</code> | Next run index (0 if not multi-run, max_existing_idx + 1 otherwise).

######## `identify_columns_to_separate`

```python
identify_columns_to_separate(dm: DesignMatrix, all_dms: list[DesignMatrix], unique_cols: list[str] | None) -> set
```

Identify columns that need run-specific separation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | The base design matrix (used for context only). | *required*
`all_dms` | <code>list of DesignMatrix</code> | All matrices being concatenated. | *required*
`unique_cols` | <code>list of str</code> | User-specified columns to separate (supports wildcards). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`set` | <code>[set](#set)</code> | Column names that should be separated with run prefixes.

######## `match_column_pattern`

```python
match_column_pattern(columns: list[str], pattern: str) -> list[str]
```

Match columns against a pattern with wildcard support.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`columns` | <code>list of str</code> | Column names to search. | *required*
`pattern` | <code>[str](#str)</code> | Pattern to match (supports '*' as wildcard). - 'motion*' matches motion_x, motion_y - '*_motion' matches x_motion, y_motion - 'exact' matches only 'exact' | *required*

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[str](#str)]</code> | list of str: Column names matching the pattern.

(data-diagnostics)=
###### `diagnostics`

Diagnostic and utility functions for DesignMatrix.

**Methods:**

Name | Description
---- | -----------
[`clean`](#data-clean) | Remove highly correlated columns.
[`corr`](#data-corr) | Correlation between DesignMatrix columns as an Adjacency.
[`vif`](#data-vif) | Compute the variance inflation factor for each column.



####### Classes

####### Functions##

###### `clean`

```python
clean(dm: DesignMatrix, *, fill_na: int | float | None = 0, exclude_confounds: bool = False, thresh: float = 0.95, progress_bar: bool = False) -> DesignMatrix
```

Remove highly correlated columns.

Removes columns with correlation >= threshold. Keeps first instance
of correlated pair, drops duplicates.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`fill_na` | <code>int, float, or None</code> | Fill NaN values before checking correlations. Default: 0. | <code>0</code>
`exclude_confounds` | <code>[bool](#bool)</code> | Skip nuisance/confound columns from correlation check. Default: False. | <code>False</code>
`thresh` | <code>[float](#float)</code> | Correlation threshold (drop if abs(r) >= thresh). Default: 0.95. | <code>0.95</code>
`progress_bar` | <code>[bool](#bool)</code> | Print dropped column names. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Cleaned matrix with highly correlated columns removed

######## `corr`

```python
corr(dm: DesignMatrix, *, metric: str = 'pearson', columns: list[str] | None = None) -> Adjacency
```

Correlation between DesignMatrix columns as an Adjacency.

Returns the column-by-column correlation matrix wrapped in an nltools
``Adjacency`` (``matrix_type='similarity'``) so it composes with the rest
of the similarity-matrix tooling (``.plot()``, MDS, etc.). The Adjacency
stores only the off-diagonal entries — self-correlation isn't a meaningful
edge — so the unit diagonal is implicit; ``DesignMatrix.plot(method='corr')``
restores it for display.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`metric` | <code>[str](#str)</code> | ``'pearson'`` (default) or ``'spearman'``. Spearman is computed as Pearson on column ranks. | <code>'pearson'</code>
`columns` | <code>list of str</code> | Subset of columns to correlate. Defaults to all columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` | <code>[Adjacency](#nltools.data.Adjacency)</code> | Similarity matrix whose ``labels`` are the included column names.

<details class="notes" open markdown="1">
<summary>Notes</summary>

Constant columns (e.g. the ``poly_0`` intercept) have zero variance and
yield NaN correlations.

</details>

######## `vif`

```python
vif(dm: DesignMatrix, exclude_confounds: bool = True) -> np.ndarray | None
```

Compute the variance inflation factor for each column.

Uses diagonal elements of inverted correlation matrix
(same method as Matlab and R).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`exclude_confounds` | <code>[bool](#bool)</code> | Skip nuisance/confound columns. Default: True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray) \| None</code> | np.ndarray: VIF values for each included column. Returns None if the correlation matrix is singular (perfect collinearity detected).

###### `io`

Provide DesignMatrix I/O and visualization functions.

Standalone functions extracted from DesignMatrix methods.
Each takes a DesignMatrix instance (`dm`) as its first argument.

**Methods:**

Name | Description
---- | -----------
[`events_to_dm`](#data-events-to-dm) | Convert a BIDS events table to boxcar regressors aligned to TRs.
`load_from_file` | Read a TSV/CSV into the frame a DesignMatrix wraps.
[`to_numpy`](#data-to-numpy) | Convert a DesignMatrix to a NumPy array.
[`to_pandas`](#data-to-pandas) | Convert DesignMatrix to pandas DataFrame.
[`write`](#data-write) | Write DesignMatrix to file.
`write_h5` | Write DesignMatrix to HDF5 file with metadata.



####### Classes

####### Functions##

(data-events-to-dm)=
###### `events_to_dm`

```python
events_to_dm(events: pl.DataFrame | pd.DataFrame, *, run_length: int, sampling_freq: float) -> pl.DataFrame
```

Convert a BIDS events table to boxcar regressors aligned to TRs.

Uses `nilearn.glm.first_level.make_first_level_design_matrix` with
`hrf_model=None` to sample events onto the TR grid without HRF
convolution — the caller is expected to call `DesignMatrix.convolve()`
explicitly when convolution is desired. Drops nilearn's auto-added
`constant` column; users add the intercept via `add_poly(0)`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`events` | <code>[DataFrame](#polars.DataFrame) \| [DataFrame](#pandas.DataFrame)</code> | pandas or polars DataFrame with BIDS columns `onset`, `duration`, `trial_type` (required); `modulation` is passed through if present. | *required*
`run_length` | <code>[int](#int)</code> | Number of TRs the run contains. | *required*
`sampling_freq` | <code>[float](#float)</code> | Sampling frequency in Hz (= 1/TR). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#polars.DataFrame)</code> | pl.DataFrame with one column per unique `trial_type`, values in
<code>[DataFrame](#polars.DataFrame)</code> | {0, modulation} indicating where each condition is active.

######## `load_from_file`

```python
load_from_file(path: str | Path, *, run_length: int | str, sampling_freq: float) -> tuple[pl.DataFrame, bool]
```

Read a TSV/CSV into the frame a DesignMatrix wraps.

Dispatches on column inspection:

- `onset` and `duration` both present → BIDS events → boxcar DM via
  `events_to_dm` (unconvolved; caller convolves later).
- otherwise → tabular file (confounds / nuisance regressors) read as-is.

`run_length='infer'` is accepted only for the tabular path; events
files must provide an explicit integer (they have a variable row count
per run, unlike confounds which are 1 row per TR).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`path` | <code>[str](#str) \| [Path](#pathlib.Path)</code> | Path to a `.tsv` or `.csv` file. | *required*
`run_length` | <code>[int](#int) \| [str](#str)</code> | Number of TRs, or `'infer'` for tabular inputs. | *required*
`sampling_freq` | <code>[float](#float)</code> | Sampling frequency in Hz (= 1/TR). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#polars.DataFrame)</code> | Tuple of (data frame, is_events) — `is_events` signals to the
<code>[bool](#bool)</code> | caller that the columns are experimental regressors rather than
<code>[tuple](#tuple)[[DataFrame](#polars.DataFrame), [bool](#bool)]</code> | nuisance.

######## `to_numpy`

```python
to_numpy(dm: DesignMatrix) -> np.ndarray
```

Convert a DesignMatrix to a NumPy array.

Returns data columns as 2D numpy array (rows x columns).
Column order is preserved from DataFrame.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | np.ndarray: 2D array with shape (n_samples, n_columns)

**Examples:**

```pycon
>>> dm = DesignMatrix({"a": [1, 2, 3], "b": [4, 5, 6]}, sampling_freq=1)
>>> arr = to_numpy(dm)
>>> arr.shape
(3, 2)
```

######## `to_pandas`

```python
to_pandas(dm: DesignMatrix)
```

Convert DesignMatrix to pandas DataFrame.

Uses dict-based conversion to avoid pyarrow dependency. This is slightly
slower (~10-20%) than pyarrow-based conversion but removes the dependency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*

**Returns:**

Type | Description
---- | -----------
 | pd.DataFrame: Pandas DataFrame with same data and column names.

**Examples:**

```pycon
>>> dm = DesignMatrix(np.random.randn(100, 3))
>>> pd_df = to_pandas(dm)
>>> type(pd_df)
<class 'pandas.core.frame.DataFrame'>
```

######## `write`

```python
write(dm: DesignMatrix, file_name: str, sep: str = '\t') -> None
```

Write DesignMatrix to file.

Supports TSV (default), CSV, and HDF5 formats. The format is
automatically determined by file extension.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`file_name` | <code>[str](#str)</code> | Output file path. Use .tsv, .csv, or .h5/.hdf5 extension. | *required*
`sep` | <code>[str](#str)</code> | Column separator for text files (default: tab for TSV).  Ignored for HDF5 files. | <code>'\t'</code>

**Returns:**

Type | Description
---- | -----------
<code>None</code> | None

**Examples:**

```pycon
>>> dm = DesignMatrix(np.random.randn(100, 3), sampling_freq=1)
>>> write(dm, "design_matrix.tsv")  # TSV format (BIDS compatible)
>>> write(dm, "design_matrix.csv", sep=",")  # CSV format
>>> write(dm, "design_matrix.h5")  # HDF5 format
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

TSV format is recommended for BIDS compatibility.
HDF5 format preserves metadata (sampling_freq, convolved, confounds).

</details>

######## `write_h5`

```python
write_h5(dm: DesignMatrix, file_name: str) -> None
```

Write DesignMatrix to HDF5 file with metadata.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`file_name` | <code>[str](#str)</code> | Output HDF5 file path. | *required*

**Returns:**

Type | Description
---- | -----------
<code>None</code> | None

###### `plotting`

DesignMatrix visualization functions.

Standalone functions extracted from ``DesignMatrix`` methods. Each takes a
``DesignMatrix`` instance (``dm``) as its first argument. ``DesignMatrix.plot``
dispatches over ``method`` to the helpers here, mirroring ``BrainData.plot``.

**Methods:**

Name | Description
---- | -----------
[`plot_corr`](#data-plot-corr) | Render a labeled correlation heatmap of the columns.
`plot_designmatrix` | Visualize a DesignMatrix, dispatching over ``method``.
`plot_matrix` | Render the design matrix as an SPM-style heatmap (rows=TRs, cols=regressors).
`plot_timeseries` | Plot regressor time courses as overlaid lines.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`VALID_PLOT_METHODS`](#data-valid-plot-methods) |  | 



####### Attributes##

(data-valid-plot-methods)=
###### `VALID_PLOT_METHODS`

```python
VALID_PLOT_METHODS = ('matrix', 'timeseries', 'corr')
```



####### Classes

####### Functions##

(data-plot-corr)=
###### `plot_corr`

```python
plot_corr(dm: DesignMatrix, *, columns: list[str] | None = None, metric: str = 'pearson', figsize: tuple | None = None, title: str | None = None, cmap: str | None = None, ax: plt.Axes | None = None, save: str | None = None, **kwargs: str | None)
```

Render a labeled correlation heatmap of the columns.

Reuses `DesignMatrix.corr`, which returns a similarity ``Adjacency``
with the unit diagonal dropped; the diagonal is restored to ``1.0`` here so
the heatmap reads as a standard correlation matrix.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`columns` | <code>[list](#list)[[str](#str)] \| None</code> | Subset of columns to correlate. Defaults to all columns. | <code>None</code>
`metric` | <code>[str](#str)</code> | ``'pearson'`` (default) or ``'spearman'``. | <code>'pearson'</code>
`figsize` | <code>[tuple](#tuple) \| None</code> | Figure size; scales with the number of columns when omitted. | <code>None</code>
`title` | <code>[str](#str) \| None</code> | Optional axis title. | <code>None</code>
`cmap` | <code>[str](#str) \| None</code> | Colormap name. Default: ``'RdBu_r'``. | <code>None</code>
`ax` | <code>[Axes](#matplotlib.pyplot.Axes) \| None</code> | Existing axis to draw on; a new figure is created if omitted. | <code>None</code>
`save` | <code>[str](#str) \| None</code> | Optional path to save the figure. | <code>None</code>
`**kwargs` |  | Forwarded to ``seaborn.heatmap`` (e.g. ``annot=False``). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

######## `plot_designmatrix`

```python
plot_designmatrix(dm: DesignMatrix, method: str = 'matrix', *, columns: list[str] | None = None, rescale: bool = True, metric: str = 'pearson', ax: plt.Axes | None = None, figsize: tuple | None = None, title: str | None = None, cmap: str | None = None, save: str | None = None, **kwargs: str | None)
```

Visualize a DesignMatrix, dispatching over ``method``.

See `DesignMatrix.plot` for the full argument documentation.

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure: The figure containing the plot.

######## `plot_matrix`

```python
plot_matrix(dm: DesignMatrix, *, columns: list[str] | None = None, rescale: bool = True, figsize: tuple | None = None, title: str | None = None, cmap: str | None = None, ax: plt.Axes | None = None, save: str | None = None, **kwargs: str | None)
```

Render the design matrix as an SPM-style heatmap (rows=TRs, cols=regressors).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`columns` | <code>[list](#list)[[str](#str)] \| None</code> | Subset of columns to plot. Defaults to all columns. | <code>None</code>
`rescale` | <code>[bool](#bool)</code> | If True, rescale each column by its L2 norm so columns with different native magnitudes are visually comparable (SPM/nilearn convention). Default: True. | <code>True</code>
`figsize` | <code>[tuple](#tuple) \| None</code> | Figure size; defaults to ``(4, 6)`` when a new figure is made. | <code>None</code>
`title` | <code>[str](#str) \| None</code> | Optional axis title. | <code>None</code>
`cmap` | <code>[str](#str) \| None</code> | Colormap name. Default: ``'gray'``. | <code>None</code>
`ax` | <code>[Axes](#matplotlib.pyplot.Axes) \| None</code> | Existing axis to draw on; a new figure is created if omitted. | <code>None</code>
`save` | <code>[str](#str) \| None</code> | Optional path to save the figure. | <code>None</code>
`**kwargs` |  | Forwarded to ``seaborn.heatmap``. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

######## `plot_timeseries`

```python
plot_timeseries(dm: DesignMatrix, *, columns: list[str] | None = None, figsize: tuple | None = None, title: str | None = None, ax: plt.Axes | None = None, save: str | None = None, **kwargs: str | None)
```

Plot regressor time courses as overlaid lines.

One line is drawn per column. Pass the same ``ax`` across calls to overlay
multiple DesignMatrices (e.g. original vs. convolved).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`columns` | <code>[list](#list)[[str](#str)] \| None</code> | Subset of columns to plot. Defaults to all columns. | <code>None</code>
`figsize` | <code>[tuple](#tuple) \| None</code> | Figure size; defaults to ``(8, 4)`` when a new figure is made. | <code>None</code>
`title` | <code>[str](#str) \| None</code> | Optional axis title. | <code>None</code>
`ax` | <code>[Axes](#matplotlib.pyplot.Axes) \| None</code> | Existing axis to draw on; a new figure is created if omitted. | <code>None</code>
`save` | <code>[str](#str) \| None</code> | Optional path to save the figure. | <code>None</code>
`**kwargs` |  | Forwarded to ``matplotlib.axes.Axes.plot`` for each line. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

(data-regressors)=
###### `regressors`

Provide standalone regressor functions for DesignMatrix.

Each function takes a DesignMatrix as its first argument (`dm`) and returns
a new DesignMatrix with the requested transformation applied.

**Methods:**

Name | Description
---- | -----------
[`add_dct_basis`](#data-add-dct-basis) | Add discrete cosine transform basis functions for high-pass filtering.
[`add_poly`](#data-add-poly) | Add Legendre polynomial drift terms.
[`convolve`](#data-convolve) | Convolve columns with an HRF or custom kernel.



####### Classes

####### Functions##

###### `add_dct_basis`

```python
add_dct_basis(dm: DesignMatrix, *, duration: float = 180, drop: int = 0, include_constant: bool = True) -> DesignMatrix
```

Add discrete cosine transform basis functions for high-pass filtering.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix to add DCT basis to. | *required*
`duration` | <code>[float](#float)</code> | Filter duration in seconds. Default: 180. | <code>180</code>
`drop` | <code>[int](#int)</code> | Number of low-frequency bases to drop. Default: 0. | <code>0</code>
`include_constant` | <code>[bool](#bool)</code> | If True, also add a constant/intercept column named ``cosine_0`` (analogous to ``poly_0`` in `add_poly`). The underlying DCT basis drops the constant per SPM convention; set False to match SPM behavior. Default: True. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with DCT basis columns appended.

######## `add_poly`

```python
add_poly(dm: DesignMatrix, order: int = 0, include_lower: bool = True) -> DesignMatrix
```

Add Legendre polynomial drift terms.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix to add polynomials to. | *required*
`order` | <code>[int](#int)</code> | Polynomial order (0=intercept, 1=linear, 2=quadratic, ...). Default: 0. | <code>0</code>
`include_lower` | <code>[bool](#bool)</code> | If True, include all orders from 0 to order. Default: True. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with polynomial columns appended.

######## `convolve`

```python
convolve(dm: DesignMatrix, conv_func: str | np.ndarray = 'hrf', columns: list[str] | None = None) -> DesignMatrix
```

Convolve columns with an HRF or custom kernel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix to convolve. | *required*
`conv_func` | <code>[str](#str) or [ndarray](#ndarray)</code> | 'hrf' for canonical Glover HRF, or custom kernel(s). Can be 1D array (single kernel) or 2D (samples x kernels) | <code>'hrf'</code>
`columns` | <code>list of str</code> | Columns to convolve (default: all non-confound columns) | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with convolved columns

**Examples:**

```pycon
>>> # Default HRF convolution → produces 'stim_c0'
>>> dm_conv = convolve(dm)
```

```pycon
>>> # Custom 1-D kernel → produces 'stim_c0'
>>> kernel = np.array([0.5, 1.0, 0.5])
>>> dm_conv = convolve(dm, conv_func=kernel)
```

```pycon
>>> # Multiple kernels (FIR model) → produces 'stim_c0', 'stim_c1'
>>> kernels = np.array([[1.0, 0.5], [0.5, 1.0]]).T  # 2 kernels
>>> dm_conv = convolve(dm, conv_func=kernels)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

Convolved columns are always renamed to ``<col>_c{i}``; the source
column is dropped. ``dm.convolved`` records the post-suffix names
(the columns that actually exist in the returned dataframe), so
downstream metadata propagation through ``.append()`` stays in
sync with the dataframe.

</details>

(data-transforms)=
###### `transforms`

Standalone transform functions for DesignMatrix.

Each function takes a DesignMatrix instance as the first argument (`dm`)
and returns a new DesignMatrix via `copy_with(dm,...)`.

**Methods:**

Name | Description
---- | -----------
[`downsample`](#data-downsample) | Reduce temporal resolution using Polars-native operations.
[`standardize`](#data-standardize) | Standardize columns using the specified method.
[`upsample`](#data-upsample) | Increase temporal resolution using Polars-native interpolation.
[`zscore`](#data-zscore) | Z-score standardize columns to mean zero and unit variance.



####### Classes

####### Functions##

###### `downsample`

```python
downsample(dm: DesignMatrix, target: float, method: str = 'mean') -> DesignMatrix
```

Reduce temporal resolution using Polars-native operations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance to transform. | *required*
`target` | <code>[float](#float)</code> | Target sampling frequency in Hz (must be < current sampling_freq). | *required*
`method` | <code>[str](#str)</code> | Aggregation method - 'mean' or 'median'. Default: 'mean'. | <code>'mean'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Downsampled DesignMatrix with updated sampling_freq.

**Examples:**

```pycon
>>> dm = DesignMatrix({"a": list(range(100))}, sampling_freq=1.0)
>>> dm_down = downsample(dm, target=0.5)  # 1 Hz -> 0.5 Hz (100 -> 50 samples)
```

######## `standardize`

```python
standardize(dm: DesignMatrix, columns: list[str] | None = None, method: str = 'zscore') -> DesignMatrix
```

Standardize columns using the specified method.

This method provides a consistent API with BrainData and Collection
for data normalization.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance to transform. | *required*
`columns` | <code>[list](#list)[[str](#str)] \| None</code> | Columns to standardize. If None, standardize all non-confound columns. | <code>None</code>
`method` | <code>[str](#str)</code> | Standardization method. Options are: - 'zscore': Z-score standardization (mean=0, std=1) [default] - 'center': Mean centering only (mean=0) | <code>'zscore'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with standardized columns.

**Examples:**

```pycon
>>> dm = DesignMatrix(np.random.randn(100, 3))
>>> dm_z = standardize(dm, method='zscore')  # z-score all columns
>>> dm_c = standardize(dm, method='center')  # center only
```

######## `upsample`

```python
upsample(dm: DesignMatrix, target: float, method: str = 'linear') -> DesignMatrix
```

Increase temporal resolution using Polars-native interpolation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance to transform. | *required*
`target` | <code>[float](#float)</code> | Target sampling frequency in Hz (must be > current sampling_freq) | *required*
`method` | <code>[str](#str)</code> | Interpolation method - 'linear' or 'nearest' (default: 'linear') | <code>'linear'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Upsampled DesignMatrix with updated sampling_freq.

**Examples:**

```pycon
>>> dm = DesignMatrix({"a": list(range(10))}, sampling_freq=1.0)
>>> dm_up = upsample(dm, target=2.0)  # 1 Hz -> 2 Hz (10 -> 19 samples)
```

######## `zscore`

```python
zscore(dm: DesignMatrix, columns: list[str] | None = None) -> DesignMatrix
```

Z-score standardize columns to mean zero and unit variance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance to transform. | *required*
`columns` | <code>list of str</code> | Columns to standardize. If None, standardize all non-confound columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with standardized columns

###### `utils`

Shared helpers for DesignMatrix submodules.

These are internal utilities used by the facade and submodules — not part of the
public API.

**Methods:**

Name | Description
---- | -----------
[`copy_with`](#data-copy-with) | Create a new DesignMatrix with updated data and metadata.
`df_passthrough` | Resolve ``name`` on ``dm.data``; re-wrap DataFrame results for allowlisted methods.
`get_data_columns` | Get column names, optionally excluding confound regressors.
`get_metadata` | Extract metadata as dict (for copying).

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`WRAP_AS_DESIGNMATRIX`](#data-wrap-as-designmatrix) |  | 



####### Attributes##

(data-wrap-as-designmatrix)=
###### `WRAP_AS_DESIGNMATRIX`

```python
WRAP_AS_DESIGNMATRIX = frozenset({'slice', 'filter', 'select'})
```



####### Classes

####### Functions##

(data-copy-with)=
###### `copy_with`

```python
copy_with(dm: DesignMatrix, new_df: pl.DataFrame, **metadata_updates: pl.DataFrame) -> DesignMatrix
```

Create a new DesignMatrix with updated data and metadata.

This is the core pattern for immutable transformations.
All methods that transform data should use this helper.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Source DesignMatrix whose metadata to copy. | *required*
`new_df` | <code>[DataFrame](#polars.DataFrame)</code> | New underlying data. | *required*
`**metadata_updates` |  | Metadata attributes to override (e.g., convolved=['stim']). | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with updated data and metadata.

######## `df_passthrough`

```python
df_passthrough(dm: DesignMatrix, name: str)
```

Resolve ``name`` on ``dm.data``; re-wrap DataFrame results for allowlisted methods.

Used by ``DesignMatrix.__getattr__`` to expose polars' DataFrame API without
duplicating every method. Row-preserving ops in ``WRAP_AS_DESIGNMATRIX`` return
a new ``DesignMatrix`` with metadata copied via ``copy_with``; everything else
returns the raw polars object.

######## `get_data_columns`

```python
get_data_columns(dm: DesignMatrix, exclude_confounds: bool = True) -> list[str]
```

Get column names, optionally excluding confound regressors.

This helper reduces code duplication across methods that need to
distinguish between experimental regressors and nuisance/confound columns
(polynomial drift, DCT cosines, motion, etc.).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`exclude_confounds` | <code>bool, default=True</code> | If True, exclude nuisance columns tracked in ``dm.confounds`` from the result. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[str](#str)]</code> | list of str: Column names (excluding confounds if requested).

######## `get_metadata`

```python
get_metadata(dm: DesignMatrix) -> dict
```

Extract metadata as dict (for copying).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys 'sampling_freq', 'convolved', 'confounds', 'multi'.

(data-fitresults)=
#### `fitresults`

Immutable container for model fitting results.

This module provides the Fit dataclass, which stores results from model fitting
operations in nltools. It uses pure numpy arrays and has no dependencies on
BrainData or other nltools data structures, making it suitable for standalone
use with inference algorithms.

**Examples:**

Using with BrainData workflow:

```pycon
>>> from nltools.data import BrainData
>>> brain = BrainData(data="brain_data.nii.gz")
>>> fit = brain.fit(model="ridge", X=design_matrix, cv=5)
>>> print(fit.available())
['fitted_values', 'weights', 'scores', 'cv_scores', 'cv_mean_score', 'cv_predictions', 'cv_folds']
```

Using with inference algorithms directly:

```pycon
>>> from nltools.algorithms import ridge_cv
>>> import numpy as np
>>> X = np.random.randn(100, 5)
>>> y = np.random.randn(100, 1000)
>>> result = ridge_cv(X, y, cv=5)
>>> result["cv_scores"].shape
(5, 20, 1000)
```

Serialization/deserialization:

```pycon
>>> # Save all non-None results
>>> np.savez("fit_results.npz", **fit.asdict())
>>>
>>> # Load and reconstruct
>>> loaded = np.load("fit_results.npz")
>>> fit_reconstructed = Fit(**{k: loaded[k] for k in loaded.files})
```

Export to .npz:

```pycon
>>> # Export only specific fields
>>> import numpy as np
>>> np.savez("weights_and_scores.npz",
...          weights=fit.weights,
...          scores=fit.scores)
```

Introspection:

```pycon
>>> # Check what's available
>>> if 'cv_scores' in fit.available():
...     print(f"CV R² range: [{fit.cv_mean_score.min():.3f}, {fit.cv_mean_score.max():.3f}]")
>>>
>>> # Get as dict and convert to a polars DataFrame (for scalar and 1D arrays)
>>> import polars as pl
>>> results_dict = fit.asdict()
>>> df = pl.DataFrame({k: v for k, v in results_dict.items() if v.ndim <= 1})
```

**Classes:**

Name | Description
---- | -----------
[`Fit`](#data-fit) | Immutable container for model fitting results.
[`Predict`](#data-predict) | Immutable container for prediction / MVPA decoding results.



##### Classes

###### `Fit`

```python
Fit(fitted_values: np.ndarray, weights: np.ndarray | None = None, scores: np.ndarray | None = None, betas: np.ndarray | None = None, t_stats: np.ndarray | None = None, p_values: np.ndarray | None = None, se: np.ndarray | None = None, residuals: np.ndarray | None = None, r2: np.ndarray | None = None, cv_scores: np.ndarray | None = None, cv_mean_score: np.ndarray | None = None, cv_predictions: np.ndarray | None = None, cv_folds: np.ndarray | None = None, cv_best_alpha: float | None = None, cv_alpha_scores: np.ndarray | None = None) -> None
```

Immutable container for model fitting results.

Pure numpy arrays with minimal introspection methods. This allows
users to work directly with nltools inference algorithms without
requiring BrainData objects.

Attributes depend on model type and CV usage:

**Ridge (no CV):**
    weights (ndarray): Coefficients, shape (n_features, n_voxels)
    scores (ndarray): R² scores, shape (n_voxels,)
    fitted_values (ndarray): Training predictions, shape (n_samples, n_voxels)

**Ridge (with CV):**
    All above plus:
    cv_scores (ndarray): Per-fold R², shape (n_folds, n_voxels)
    cv_mean_score (ndarray): Mean R² across folds, shape (n_voxels,)
    cv_predictions (ndarray): Out-of-fold predictions, shape (n_samples, n_voxels)
    cv_folds (ndarray): Fold indices, shape (n_samples,)
    cv_best_alpha (float): Selected alpha (if alpha='auto')
    cv_alpha_scores (ndarray): Alpha selection scores (if alpha='auto')

**GLM:**
    betas (ndarray): Beta coefficients, shape (n_regressors, n_voxels)
    t_stats (ndarray): T-statistics, shape (n_regressors, n_voxels)
    p_values (ndarray): P-values, shape (n_regressors, n_voxels)
    se (ndarray): Standard errors, shape (n_regressors, n_voxels)
    residuals (ndarray): Residuals, shape (n_samples, n_voxels)
    fitted_values (ndarray): Fitted values, shape (n_samples, n_voxels)
    r2 (ndarray): R² values, shape (n_voxels,)

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`fitted_values` | <code>[ndarray](#ndarray)</code> | Fitted values or predictions, always present.
`weights` | <code>[ndarray](#ndarray) \| None</code> | Model coefficients (Ridge).
`scores` | <code>[ndarray](#ndarray) \| None</code> | R² scores (Ridge).
[`betas`](#data-betas) | <code>[ndarray](#ndarray) \| None</code> | Beta coefficients (GLM).
`t_stats` | <code>[ndarray](#ndarray) \| None</code> | T-statistics (GLM).
`p_values` | <code>[ndarray](#ndarray) \| None</code> | P-values (GLM).
`se` | <code>[ndarray](#ndarray) \| None</code> | Standard errors (GLM).
`residuals` | <code>[ndarray](#ndarray) \| None</code> | Residuals (GLM).
`r2` | <code>[ndarray](#ndarray) \| None</code> | R² values (GLM).
`cv_scores` | <code>[ndarray](#ndarray) \| None</code> | Per-fold cross-validation scores.
`cv_mean_score` | <code>[ndarray](#ndarray) \| None</code> | Mean cross-validation score across folds.
`cv_predictions` | <code>[ndarray](#ndarray) \| None</code> | Out-of-fold predictions.
`cv_folds` | <code>[ndarray](#ndarray) \| None</code> | Fold indices for each sample.
`cv_best_alpha` | <code>[float](#float) \| None</code> | Best alpha selected via cross-validation.
`cv_alpha_scores` | <code>[ndarray](#ndarray) \| None</code> | Cross-validation scores for each alpha tested.

<details class="note" open markdown="1">
<summary>Note</summary>

Methods: `available` returns the list of non-None attribute names
(excludes private fields); `asdict` converts to a dictionary,
optionally excluding None values.

</details>

**Examples:**

Creating a Fit object (Ridge without CV):

```pycon
>>> import numpy as np
>>> from nltools.data.fitresults import Fit
>>> fit = Fit(
...     fitted_values=np.random.randn(100, 1000),
...     weights=np.random.randn(5, 1000),
...     scores=np.random.randn(1000)
... )
>>> fit.available()
['fitted_values', 'weights', 'scores']
```

Creating a Fit object (Ridge with CV):

```pycon
>>> fit_cv = Fit(
...     fitted_values=np.random.randn(100, 1000),
...     weights=np.random.randn(5, 1000),
...     scores=np.random.randn(1000),
...     cv_scores=np.random.randn(5, 1000),
...     cv_mean_score=np.random.randn(1000),
...     cv_predictions=np.random.randn(100, 1000),
...     cv_folds=np.arange(100) % 5
... )
>>> 'cv_scores' in fit_cv.available()
True
```

Immutability:

```pycon
>>> try:
...     fit.scores = np.zeros(1000)  # Will raise FrozenInstanceError
... except AttributeError:
...     print("Cannot modify frozen dataclass")
Cannot modify frozen dataclass
```

Export/serialization:

```pycon
>>> # Save to .npz
>>> np.savez("results.npz", **fit.asdict())
>>>
>>> # Load and reconstruct
>>> loaded = np.load("results.npz")
>>> fit_reloaded = Fit(**{k: loaded[k] for k in loaded.files})
```

<details class="note" open markdown="1">
<summary>Note</summary>

- Frozen dataclass ensures results cannot be accidentally modified.
- All attributes are numpy arrays (except cv_best_alpha which is float).
- None values indicate that field was not computed for this model/method.
- Private fields (starting with _) are excluded from available() and asdict().

</details>

**Methods:**

Name | Description
---- | -----------
[`asdict`](#data-asdict) | Convert to dictionary.
[`available`](#data-available) | Return list of non-None attribute names.



####### Attributes##

(data-betas)=
###### `betas`

```python
betas: np.ndarray | None = None
```

######## `cv_alpha_scores`

```python
cv_alpha_scores: np.ndarray | None = None
```

######## `cv_best_alpha`

```python
cv_best_alpha: float | None = None
```

######## `cv_folds`

```python
cv_folds: np.ndarray | None = None
```

######## `cv_mean_score`

```python
cv_mean_score: np.ndarray | None = None
```

######## `cv_predictions`

```python
cv_predictions: np.ndarray | None = None
```

######## `cv_scores`

```python
cv_scores: np.ndarray | None = None
```

######## `fitted_values`

```python
fitted_values: np.ndarray
```

######## `p_values`

```python
p_values: np.ndarray | None = None
```

######## `r2`

```python
r2: np.ndarray | None = None
```

######## `residuals`

```python
residuals: np.ndarray | None = None
```

######## `scores`

```python
scores: np.ndarray | None = None
```

######## `se`

```python
se: np.ndarray | None = None
```

######## `t_stats`

```python
t_stats: np.ndarray | None = None
```

######## `weights`

```python
weights: np.ndarray | None = None
```



####### Functions##

###### `asdict`

```python
asdict(include_none: bool = False) -> dict
```

Convert to dictionary.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`include_none` | <code>[bool](#bool)</code> | If True, include attributes with None values. Private fields (starting with _) are always excluded. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary of attribute names to values.

**Examples:**

```pycon
>>> import numpy as np
>>> from nltools.data.fitresults import Fit
>>> fit = Fit(
...     fitted_values=np.random.randn(100, 1000),
...     weights=np.random.randn(5, 1000),
...     scores=None
... )
>>> d = fit.asdict(include_none=False)
>>> 'scores' in d
False
>>> d = fit.asdict(include_none=True)
>>> 'scores' in d
True
>>> d['scores'] is None
True
```

######## `available`

```python
available() -> list
```

Return list of non-None attribute names.

Excludes private fields (starting with _).

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)</code> | Names of attributes that are not None.

**Examples:**

```pycon
>>> import numpy as np
>>> from nltools.data.fitresults import Fit
>>> fit = Fit(
...     fitted_values=np.random.randn(100, 1000),
...     weights=np.random.randn(5, 1000)
... )
>>> fit.available()
['fitted_values', 'weights']
>>> 'scores' in fit.available()
False
```

###### `Predict`

```python
Predict(predictions: np.ndarray | None = None, scores: np.ndarray | None = None, mean_score: Any = None, std_score: Any = None, cv_folds: np.ndarray | None = None, roi_labels: np.ndarray | None = None, accuracy_map: Any = None, weight_map: Any = None, fold_weight_maps: Any = None, estimator: Any = None) -> None
```

Immutable container for prediction / MVPA decoding results.

Mirrors `Fit`: frozen, all fields default to `None`, populated
based on the dispatch path (`spatial_scale`, `y` vs `X`, `refit`) used
by `BrainData.predict`. Fields not applicable to the call remain
`None` and are filtered from `available` and `asdict`.

**Brain-space outputs are `BrainData` objects**, not raw arrays —
so `result.weight_map.plot()` works directly. Drop down to numpy via
`result.weight_map.data` if needed. Non-spatial fields (`predictions`,
`cv_folds`, scalar scores) stay as numpy.

Field shapes by dispatch:

**spatial_scale='whole_brain'** (with `y`):
    - `predictions`: `(n_samples,)` ndarray, OOF predictions from CV
    - `scores`: `(n_folds,)` ndarray, per-fold score
    - `mean_score`: float, mean across folds
    - `std_score`: float, std across folds
    - `cv_folds`: `(n_samples,)` ndarray, fold index per sample
    - `weight_map`: BrainData `(1, n_voxels)`, `coef_` from one
      model fit on the **full** `(X, y)`. The publishable map.
    - `fold_weight_maps`: BrainData `(n_folds, n_voxels)`, per-fold
      `coef_` stack — for stability analysis (e.g., across-fold std).
    - `estimator`: the fitted all-data sklearn estimator (use for
      `.predict()` on new data).

**spatial_scale='roi'** (with `y`):
    - `scores`: `(n_folds, n_rois)` ndarray
    - `mean_score`: `(n_rois,)` ndarray, mean across folds per parcel
    - `std_score`: `(n_rois,)` ndarray
    - `roi_labels`: `(n_rois,)` ndarray of atlas integer IDs in the
      same order as `mean_score` / `std_score` / `scores` axis 1
    - `accuracy_map`: BrainData `(1, n_voxels)`, every voxel inside
      parcel *i* set to that parcel's mean accuracy (others NaN)
    - `weight_map`: BrainData `(1, n_voxels)`, per-parcel `coef_`
      from each parcel's all-data fit, written back into voxel space
      (atlas is a label image so reassembly is disjoint). Voxels outside
      any parcel are NaN. Magnitudes across parcels are not directly
      comparable — different parcels live on different X distributions.
    - `fold_weight_maps`: BrainData `(n_folds, n_voxels)`
    - `estimator`: `dict[int, sklearn]` keyed by atlas label

    If any parcel can't expose `.coef_` (non-linear model, `SelectKBest`
    in pipeline), `weight_map` / `fold_weight_maps` / `estimator`
    all collapse to `None` for the whole call.

**spatial_scale='searchlight'** (with `y`):
    - `accuracy_map`: BrainData `(1, n_voxels)`, sphere-centered
      accuracy at each voxel

<details class="note" open markdown="1">
<summary>Note</summary>

Encoding-model timeseries prediction (`bd.predict(X=...)`) returns
a `BrainData` directly, not a `Predict` — the natural container for a
voxel timeseries.

Why the all-data fit is canonical: the CV mean of per-fold `coef_`
vectors doesn't correspond to any actual fitted estimator (each fold
saw a different subset). The all-data refit is a single, real model
with all the information used. CV gives the honest *score*; the refit
gives the publishable *map*. `fold_weight_maps` is still exposed for
stability analysis, and the CV-mean is one line away if you want it
(`fold_weight_maps.data.mean(axis=0)`).

Methods: `available` returns the names of non-None fields (excludes
private); `asdict` converts to a dict for serialization (private fields
always excluded).

</details>

**Methods:**

Name | Description
---- | -----------
[`asdict`](#data-asdict) | Convert to dictionary.
[`available`](#data-available) | Return names of non-None fields (excludes private).

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`accuracy_map`](#data-accuracy-map) | <code>[Any](#typing.Any)</code> | 
`cv_folds` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`estimator` | <code>[Any](#typing.Any)</code> | 
`fold_weight_maps` | <code>[Any](#typing.Any)</code> | 
`mean_score` | <code>[Any](#typing.Any)</code> | 
`predictions` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`roi_labels` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`scores` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`std_score` | <code>[Any](#typing.Any)</code> | 
`weight_map` | <code>[Any](#typing.Any)</code> | 



####### Attributes##

(data-accuracy-map)=
###### `accuracy_map`

```python
accuracy_map: Any = None
```

######## `cv_folds`

```python
cv_folds: np.ndarray | None = None
```

######## `estimator`

```python
estimator: Any = None
```

######## `fold_weight_maps`

```python
fold_weight_maps: Any = None
```

######## `mean_score`

```python
mean_score: Any = None
```

######## `predictions`

```python
predictions: np.ndarray | None = None
```

######## `roi_labels`

```python
roi_labels: np.ndarray | None = None
```

######## `scores`

```python
scores: np.ndarray | None = None
```

######## `std_score`

```python
std_score: Any = None
```

######## `weight_map`

```python
weight_map: Any = None
```



####### Functions##

###### `asdict`

```python
asdict(include_none: bool = False) -> dict
```

Convert to dictionary.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`include_none` | <code>[bool](#bool)</code> | If True, include fields with None values. Private fields (starting with _) are always excluded. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary of field names to values.

######## `available`

```python
available() -> list
```

Return names of non-None fields (excludes private).

#### `roc`

ROC (Receiver Operating Characteristic) analysis for single-interval classification.

These tools provide the ability to quickly run receiver operating characteristic
analyses on the output of machine-learning models applied to imaging data.

**Classes:**

Name | Description
---- | -----------
[`Roc`](#data-roc) | Compute receiver operating characteristic curves for single-interval or forced-choice classification.



##### Classes

###### `Roc`

```python
Roc(*, input_values = None, binary_outcome = None, method = 'optimal_overall', forced_choice = None)
```

Compute receiver operating characteristic curves for single-interval or forced-choice classification.

The Roc class is based on Tor Wager's Matlab roc_plot.m function and
allows a user to easily run different types of receiver operator
characteristic curves.  For example, one might be interested in single
interval or forced choice.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`input_values` |  | 1-D array/vector of continuous decision values (one per observation) | <code>None</code>
`binary_outcome` |  | vector of training labels | <code>None</code>
`method` |  | threshold-selection variant, one of `'optimal_overall'`, `'optimal_balanced'`, `'minimum_sdt_bias'` | <code>'optimal_overall'</code>
`forced_choice` |  | index indicating position for each unique subject (default=None) | <code>None</code>

**Methods:**

Name | Description
---- | -----------
[`calculate`](#data-calculate) | Calculate ROC metrics for single-interval classification.
[`plot`](#data-plot) | Create a ROC plot.
[`summary`](#data-summary) | Display a formatted summary of ROC analysis.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`binary_outcome`](#data-binary-outcome) |  | 
`forced_choice` |  | 
`input_values` |  | 
`method` |  | 



####### Attributes##

(data-binary-outcome)=
###### `binary_outcome`

```python
binary_outcome = np.asarray(binary_outcome).astype(bool).flatten()
```

######## `forced_choice`

```python
forced_choice = deepcopy(forced_choice)
```

######## `input_values`

```python
input_values = np.array(input_values)
```

######## `method`

```python
method = deepcopy(method)
```



####### Functions##

###### `calculate`

```python
calculate(*, input_values = None, binary_outcome = None, criterion_values = None, method = 'optimal_overall', forced_choice = None, balanced_acc = False)
```

Calculate ROC metrics for single-interval classification.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`input_values` |  | 1-D array/vector of continuous decision values (one per observation) | <code>None</code>
`binary_outcome` |  | vector of training labels | <code>None</code>
`criterion_values` |  | (optional) criterion values for calculating fpr             & tpr | <code>None</code>
`method` |  | threshold-selection variant, one of `'optimal_overall'`,             `'optimal_balanced'`, `'minimum_sdt_bias'` | <code>'optimal_overall'</code>
`forced_choice` |  | index indicating position for each unique subject             (default=None) | <code>None</code>
`balanced_acc` |  | balanced accuracy for single-interval classification             (bool). THIS IS NOT COMPLETELY IMPLEMENTED BECAUSE             IT AFFECTS ACCURACY ESTIMATES, BUT NOT P-VALUES OR             THRESHOLD AT WHICH TO EVALUATE SENS/SPEC | <code>False</code>

######## `plot`

```python
plot(*, method = 'gaussian', balanced_acc = False)
```

Create a ROC plot.

Create a specific kind of ROC curve plot, based on input values
along a continuous distribution and a binary outcome variable (logical)

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` |  | type of plot, one of `'gaussian'`, `'observed'` | <code>'gaussian'</code>
`balanced_acc` |  | balanced accuracy for single-interval classification | <code>False</code>

**Returns:**

Type | Description
---- | -----------
 | fig

######## `summary`

```python
summary()
```

Display a formatted summary of ROC analysis.



##### Methods

#### `simulator`

Tools to simulate multivariate brain and grid data for testing analysis pipelines.

**Classes:**

Name | Description
---- | -----------
[`SimulateGrid`](#data-simulategrid) | Simulate 2D grid data for testing statistical methods.
[`Simulator`](#data-simulator) | Simulate fMRI data with realistic spatial and temporal characteristics.



##### Classes

###### `SimulateGrid`

```python
SimulateGrid(*, grid_width = 100, signal_width = 20, n_subjects = 20, sigma = 1, signal_amplitude = None, random_state = None)
```

Simulate 2D grid data for testing statistical methods.

Creates a 2D grid (e.g., 100x100 pixels) with optional embedded signal
regions and Gaussian noise. Useful for testing multiple comparison
correction methods, threshold selection, and visualization of
statistical maps.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`grid_width` |  | Width/height of the square grid (default: 100). | <code>100</code>
`signal_width` |  | Width of the embedded signal region (default: 20). | <code>20</code>
`n_subjects` |  | Number of simulated subjects (default: 20). | <code>20</code>
`sigma` |  | Standard deviation of the Gaussian noise (default: 1). | <code>1</code>
`signal_amplitude` |  | Amplitude of the embedded signal. If None, no signal is added. | <code>None</code>
`random_state` |  | Random seed or numpy RandomState for reproducibility. | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`data`](#data-data) |  | The simulated data array of shape (n_subjects, grid_width, grid_width).
`t_values` |  | T-statistic values after fitting.
`p_values` |  | P-values after fitting.
`thresholded` |  | Thresholded statistical map.
`isfit` |  | Whether fit() has been called.

**Examples:**

```pycon
>>> from nltools.data.simulator import SimulateGrid
>>> sim = SimulateGrid(signal_amplitude=0.5, random_state=42)
>>> sim.fit()
>>> sim.plot()
```

**Methods:**

Name | Description
---- | -----------
[`add_signal`](#data-add-signal) | Add a rectangular signal to self.data.
[`create_mask`](#data-create-mask) | Create a mask for where the signal is located in grid.
[`fit`](#data-fit) | Run a one-sample t-test on self.data.
[`plot_grid_simulation`](#data-plot-grid-simulation) | Create a plot of the simulations.
[`run_multiple_simulations`](#data-run-multiple-simulations) | Run multiple simulations to calculate the overall false positive rate.
[`threshold_simulation`](#data-threshold-simulation) | Threshold the fitted simulation.



####### Attributes##

(data-correction)=
###### `correction`

```python
correction = None
```

######## `data`

```python
data = self._create_noise()
```

######## `grid_width`

```python
grid_width = grid_width
```

######## `isfit`

```python
isfit = False
```

######## `n_subjects`

```python
n_subjects = n_subjects
```

######## `p_values`

```python
p_values = None
```

######## `random_state`

```python
random_state = check_random_state(random_state)
```

######## `sigma`

```python
sigma = sigma
```

######## `signal_amplitude`

```python
signal_amplitude = None
```

######## `signal_mask`

```python
signal_mask = None
```

######## `t_values`

```python
t_values = None
```

######## `threshold`

```python
threshold = None
```

######## `threshold_type`

```python
threshold_type = None
```

######## `thresholded`

```python
thresholded = None
```



####### Functions##

###### `add_signal`

```python
add_signal(signal_width = 20, signal_amplitude = 1)
```

Add a rectangular signal to self.data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`signal_width` | <code>[int](#int)</code> | width of signal box | <code>20</code>
`signal_amplitude` | <code>[int](#int)</code> | intensity of signal | <code>1</code>

######## `create_mask`

```python
create_mask(signal_width)
```

Create a mask for where the signal is located in grid.

######## `fit`

```python
fit()
```

Run a one-sample t-test on self.data.

######## `plot_grid_simulation`

```python
plot_grid_simulation(threshold, threshold_type, n_simulations = 100, correction = None)
```

Create a plot of the simulations.

######## `run_multiple_simulations`

```python
run_multiple_simulations(threshold, threshold_type, n_simulations = 100, correction = None)
```

Run multiple simulations to calculate the overall false positive rate.

######## `threshold_simulation`

```python
threshold_simulation(threshold, threshold_type, correction = None)
```

Threshold the fitted simulation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`threshold` | <code>[float](#float)</code> | threshold to apply to simulation | *required*
`threshhold_type` | <code>[str](#str)</code> | type of threshold to use can be a specific t-value or p-value ['t', 'p', 'q'] | *required*

###### `Simulator`

```python
Simulator(*, brain_mask = None, output_dir = None, random_state = None)
```

Simulate fMRI data with realistic spatial and temporal characteristics.

This class provides methods for generating synthetic fMRI data with
controlled signal patterns, including Gaussian blobs, multi-subject
datasets, and various noise structures. Useful for testing analysis
pipelines and power analyses.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_mask` |  | Path to a NIfTI brain mask file, a nibabel image object, or None to use the default MNI template mask. | <code>None</code>
`output_dir` |  | Directory for saving generated data. Defaults to the current working directory. | <code>None</code>
`random_state` |  | Random seed or numpy RandomState for reproducibility. | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`brain_mask`](#data-brain-mask) |  | The brain mask image used for simulation.
`output_dir` |  | Output directory path.
`random_state` |  | Random state for reproducible simulations.

**Examples:**

```pycon
>>> from nltools.data.simulator import Simulator
>>> sim = Simulator(random_state=42)
>>> # Create a dataset with signal in specific regions
>>> data = sim.create_data(levels=[1, -1, 1, -1], sigma=1, reps=10)
```

**Methods:**

Name | Description
---- | -----------
[`create_cov_data`](#data-create-cov-data) | Create continuous simulated data with covariance within a single region.
[`create_data`](#data-create-data) | Create simulated data with discrete intensity levels.
[`create_ncov_data`](#data-create-ncov-data) | Create continuous simulated data with covariance across multiple regions.
[`gaussian`](#data-gaussian) | Create a 3D gaussian signal normalized to a given intensity.
[`n_spheres`](#data-n-spheres) | Generate a set of spheres in the brain mask space.
[`normal_noise`](#data-normal-noise) | Produce a normal noise distribution for all points in the brain mask.
[`sphere`](#data-sphere) | Create a sphere of given radius at some point p in the brain mask.
[`to_nifti`](#data-to-nifti) | Convert a numpy matrix to the nifti format and assign it the brain_mask's affine matrix.



####### Attributes##

(data-brain-mask)=
###### `brain_mask`

```python
brain_mask = brain_mask
```

######## `output_dir`

```python
output_dir = os.path.join(os.getcwd())
```

######## `random_state`

```python
random_state = check_random_state(random_state)
```



####### Functions##

###### `create_cov_data`

```python
create_cov_data(cor, cov, sigma, *, mask = None, reps = 1, n_sub = 1, output_dir = None)
```

Create continuous simulated data with covariance within a single region.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cor` |  | amount of covariance between each voxel and Y variable | *required*
`cov` |  | amount of covariance between voxels | *required*
`sigma` |  | amount of noise to add | *required*
`mask` |  | region where activations are placed (a single mask image); defaults to a sphere if None | <code>None</code>
`reps` |  | number of data repetitions | <code>1</code>
`n_sub` |  | number of subjects to simulate | <code>1</code>
`output_dir` |  | string path of directory to output data.  If None, no data will be written | <code>None</code>

######## `create_data`

```python
create_data(levels, sigma, *, radius = 5, center = None, reps = 1, output_dir = None)
```

Create simulated data with discrete intensity levels.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`levels` |  | vector of intensities or class labels | *required*
`sigma` |  | amount of noise to add | *required*
`radius` |  | vector of radius.  Will create multiple spheres if len(radius) > 1 | <code>5</code>
`center` |  | center(s) of sphere(s) of the form [px, py, pz] or [[px1, py1, pz1], ..., [pxn, pyn, pzn]] | <code>None</code>
`reps` |  | number of data repetitions useful for trials or subjects | <code>1</code>
`output_dir` |  | string path of directory to output data.  If None, no data will be written | <code>None</code>

######## `create_ncov_data`

```python
create_ncov_data(cor, cov, sigma, *, masks = None, reps = 1, n_sub = 1, output_dir = None)
```

Create continuous simulated data with covariance across multiple regions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cor` |  | amount of covariance between each voxel and Y variable (an int or a vector) | *required*
`cov` |  | amount of covariance between voxels (an int or a matrix) | *required*
`sigma` |  | amount of noise to add | *required*
`masks` |  | region(s) where we will have activations (list if more than one) | <code>None</code>
`reps` |  | number of data repetitions | <code>1</code>
`n_sub` |  | number of subjects to simulate | <code>1</code>
`output_dir` |  | string path of directory to output data.  If None, no data will be written | <code>None</code>

######## `gaussian`

```python
gaussian(mu, sigma, i_tot)
```

Create a 3D gaussian signal normalized to a given intensity.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mu` |  | average value of the gaussian signal (usually set to 0) | *required*
`sigma` |  | standard deviation | *required*
`i_tot` |  | sum total of activation (numerical integral over the gaussian returns this value) | *required*

######## `n_spheres`

```python
n_spheres(radius, center)
```

Generate a set of spheres in the brain mask space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`radius` |  | vector of radius.  Will create multiple spheres if len(radius) > 1 | *required*
`centers` |  | a vector of sphere centers of the form [px, py, pz] or [[px1, py1, pz1], ..., [pxn, pyn, pzn]] | *required*

######## `normal_noise`

```python
normal_noise(mu, sigma)
```

Produce a normal noise distribution for all points in the brain mask.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mu` |  | average value of the gaussian signal (usually set to 0) | *required*
`sigma` |  | standard deviation | *required*

######## `sphere`

```python
sphere(r, p)
```

Create a sphere of given radius at some point p in the brain mask.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`r` |  | radius of the sphere | *required*
`p` |  | point (in coordinates of the brain mask) of the center of the sphere | *required*

######## `to_nifti`

```python
to_nifti(m)
```

Convert a numpy matrix to the nifti format and assign it the brain_mask's affine matrix.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`m` |  | the 3D numpy matrix we wish to convert to .nii | *required*



##### Methods