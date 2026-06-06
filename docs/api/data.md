## `data`

nltools data types.

**Modules:**

Name | Description
---- | -----------
[`adjacency`](#adjacency) | This data class is for working with similarity/dissimilarity matrices
[`atlases`](#atlases) | Atlas registry, lazy loading, and coordinate labeling.
[`braindata`](#braindata) | NeuroLearn Brain Data
[`collection`](#collection) | BrainCollection — multi-subject brain-data container (v0.6.0).
[`designmatrix`](#designmatrix) | DesignMatrix - Polars-based design matrix for neuroimaging analysis
[`fitresults`](#fitresults) | Immutable container for model fitting results.
[`roc`](#roc) | NeuroLearn Analysis Tools
[`simulator`](#simulator) | NeuroLearn Simulator Tools

**Classes:**

Name | Description
---- | -----------
[`Adjacency`](#Adjacency) | Adjacency is a class to represent Adjacency matrices as a vector rather
[`BrainCollection`](#BrainCollection) | Parallel, lazy iterator of ``BrainData`` whose API mirrors ``BrainData``.
[`BrainData`](#BrainData) | BrainData is a class to represent neuroimaging data in python as a vector
[`DesignMatrix`](#DesignMatrix) | Polars-based design matrix for experimental designs in neuroimaging.
[`Fit`](#Fit) | Immutable container for model fitting results.
[`Roc`](#Roc) | Roc Class
[`SimulateGrid`](#SimulateGrid) | Simulate 2D grid data for testing statistical methods.
[`Simulator`](#Simulator) | Simulate fMRI data with realistic spatial and temporal characteristics.



### Classes

#### `Adjacency`

```python
Adjacency(data = None, *, Y = None, matrix_type = None, labels = None, spatial_scale: SpatialScale | None = None)
```

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
[`append`](#append) | Append data to Adjacency instance
[`bootstrap`](#bootstrap) | Bootstrap statistics using efficient online algorithms.
[`cluster_summary`](#cluster_summary) | Provide summaries of clusters within Adjacency matrices.
[`copy`](#copy) | Create a copy of Adjacency object.
[`distance`](#distance) | Calculate distance between images within an Adjacency() instance.
[`distance_to_similarity`](#distance_to_similarity) | Convert distance matrix to similarity matrix.
[`generate_permutations`](#generate_permutations) | Generate n_permute permutated versions of Adjacency in a lazy fashion.
[`mean`](#mean) | Calculate mean of Adjacency.
[`median`](#median) | Calculate median of Adjacency.
[`plot`](#plot) | Create Heatmap of Adjacency Matrix
[`plot_label_distance`](#plot_label_distance) | Create a violin plot indicating within and between label distance
[`plot_mds`](#plot_mds) | Plot Multidimensional Scaling
[`plot_silhouette`](#plot_silhouette) | Create a silhouette plot
[`r_to_z`](#r_to_z) | Apply Fisher's r to z transformation to each element of the data
[`regress`](#regress) | Run a regression on an adjacency instance.
[`similarity`](#similarity) | Calculate similarity between two Adjacency matrices. Default is to use spearman
[`social_relations_model`](#social_relations_model) | Estimate the social relations model from a matrix for a round-robin design.
[`squareform`](#squareform) | Convert adjacency back to squareform
[`stats_label_distance`](#stats_label_distance) | Calculate permutation tests on within and between label distance.
[`std`](#std) | Calculate standard deviation of Adjacency.
[`sum`](#sum) | Calculate sum of Adjacency.
[`threshold`](#threshold) | Threshold Adjacency instance. Provide upper and lower values or
[`to_brain`](#to_brain) | Project per-matrix scalars back to voxel-space :class:`BrainData`.
[`to_graph`](#to_graph) | Convert Adjacency into networkx graph.  only works on
[`to_square`](#to_square) | Convert adjacency back to square matrix format.
[`ttest`](#ttest) | Calculate ttest across samples.
[`write`](#write) | Write out Adjacency object to csv file.
[`z_to_r`](#z_to_r) | Convert z score back into r value for each element of data object

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`Y`](#Y) | <code>[DataFrame](#polars.DataFrame)</code> | Training labels as a polars DataFrame (possibly empty).
[`data`](#data) |  | 
[`is_empty`](#is_empty) | <code>[bool](#bool)</code> | Check if Adjacency object is empty.
[`is_single_matrix`](#is_single_matrix) |  | 
[`issymmetric`](#issymmetric) |  | 
[`labels`](#labels) |  | 
[`matrix_type`](#matrix_type) |  | 
[`n_nodes`](#n_nodes) |  | Return the number of nodes in the adjacency matrix.
[`shape`](#shape) |  | Return the logical shape of the adjacency matrix.
[`spatial_scale`](#spatial_scale) | <code>[SpatialScale](#nltools.data.adjacency.spatial.SpatialScale) \| None</code> | 
[`vector_shape`](#vector_shape) |  | Return shape of internal vectorized representation.

##### Methods

###### `append`

```python
append(data)
```

Append data to Adjacency instance

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (Adjacency) Adjacency instance to append | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (Adjacency) new appended Adjacency instance

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

###### `cluster_summary`

```python
cluster_summary(clusters = None, metric = 'mean', summary = 'within')
```

Provide summaries of clusters within Adjacency matrices.

Computes mean/median of within and between cluster values. Requires a
list of cluster ids indicating the row/column of each cluster.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`clusters` |  | (list) list of cluster labels | <code>None</code>
`metric` |  | (str) method to summarize mean or median. If 'None" then return all r values | <code>'mean'</code>
`summary` |  | (str) summarize within cluster or between clusters | <code>'within'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | within cluster means

###### `copy`

```python
copy()
```

Create a copy of Adjacency object.

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

###### `generate_permutations`

```python
generate_permutations(n_permute, random_state = None)
```

Generate n_permute permutated versions of Adjacency in a lazy fashion.

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

###### `plot`

```python
plot(limit = 3, axes = None, *args, **kwargs)
```

Create Heatmap of Adjacency Matrix

Can pass in any sns.heatmap argument

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`limit` |  | (int) number of heatmaps to plot if object contains multiple adjacencies (default: 3) | <code>3</code>
`axes` |  | matplotlib axis handle | <code>None</code>

###### `plot_label_distance`

```python
plot_label_distance(labels = None, ax = None)
```

Create a violin plot indicating within and between label distance

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`labels` | <code>[array](#numpy.array)</code> | numpy array of labels to plot | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`f` |  | violin plot handles

###### `plot_mds`

```python
plot_mds(n_components = 2, metric = True, labels = None, labels_color = None, cmap = None, view = (30, 20), figsize = None, ax = None, n_jobs = -1, *args, **kwargs)
```

Plot Multidimensional Scaling

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_components` |  | (int) Number of dimensions to project (can be 2 or 3) | <code>2</code>
`metric` |  | (bool) Perform metric or non-metric dimensional scaling; default | <code>True</code>
`labels` |  | (list) Can override labels stored in Adjacency Class | <code>None</code>
`labels_color` |  | (str) list of colors for labels, if len(1) then make all same color | <code>None</code>
`cmap` |  | colormap instance (default: plt.cm.hot_r) | <code>None</code>
`view` |  | (tuple) view for 3-Dimensional plot; default (30,20) | <code>(30, 20)</code>
`figsize` |  | (list) figure size; default [12, 8] | <code>None</code>
`ax` |  | matplotlib axis handle | <code>None</code>
`n_jobs` |  | (int) Number of parallel jobs | <code>-1</code>

###### `plot_silhouette`

```python
plot_silhouette(labels = None, ax = None, permutation_test = True, n_permute = 5000, **kwargs)
```

Create a silhouette plot

###### `r_to_z`

```python
r_to_z()
```

Apply Fisher's r to z transformation to each element of the data
object.

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

###### `similarity`

```python
similarity(data, plot = False, permutation_method = '2d', n_permute = 5000, metric = 'spearman', include_diag = False, nan_policy = 'omit', tail = 2, return_null = False, n_jobs = -1, random_state = None, *, project: bool = False)
```

Calculate similarity between two Adjacency matrices. Default is to use spearman
correlation and permutation test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Adjacency](#nltools.data.adjacency.Adjacency) or [array](#array)</code> | Adjacency data, or 1-d array same size as self.data | *required*
`permutation_method` |  | (str) '1d','2d', or None | <code>'2d'</code>
`metric` |  | (str) 'spearman','pearson','kendall' | <code>'spearman'</code>
`include_diag` |  | (bool) only applies to 'directed' Adjacency types using permutation_method=None or permutation_method='1d'. Default False (self-similarity is uninformative). Symmetric matrices never store the diagonal, so this flag is a no-op for them. | <code>False</code>
`nan_policy` |  | (str) How to handle NaN values. Options: - 'omit': Remove NaN values pairwise before computing correlation (default) - 'propagate': Allow NaN to propagate through calculations - 'raise': Raise an error if NaN values are present | <code>'omit'</code>
`tail` |  | (int) Tail of the test (1 or 2). Default 2. | <code>2</code>
`return_null` |  | (bool) If True, also return the null distribution. Default False. | <code>False</code>
`n_jobs` |  | (int) Number of parallel jobs. Default -1 (all cores). | <code>-1</code>
`random_state` |  | (int, optional) Random seed for reproducibility. | <code>None</code>

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

###### `squareform`

```python
squareform()
```

Convert adjacency back to squareform

###### `stats_label_distance`

```python
stats_label_distance(labels = None, n_permute = 5000, n_jobs = -1)
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

###### `threshold`

```python
threshold(upper = None, lower = None, binarize = False)
```

Threshold Adjacency instance. Provide upper and lower values or
   percentages to perform two-sided thresholding. Binarize will return
   a mask image respecting thresholds if provided, otherwise respecting
   every non-zero value.

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

###### `to_brain`

```python
to_brain(values, *, fill: float = np.nan)
```

Project per-matrix scalars back to voxel-space :class:`BrainData`.

Requires :attr:`spatial_scale` to be set (i.e. this stack came from
:meth:`BrainData.distance` or another spatial-scale-aware producer).
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

###### `to_graph`

```python
to_graph()
```

Convert Adjacency into networkx graph.  only works on
single_matrix for now.

###### `to_square`

```python
to_square()
```

Convert adjacency back to square matrix format.

This is an alias for :meth:`squareform`.

**Returns:**

Type | Description
---- | -----------
 | np.ndarray or list: Square matrix representation. Returns a list
 | of matrices if this object contains multiple adjacency matrices.

###### `ttest`

```python
ttest(permutation = False, n_permute = 5000, tail = 2, return_null = False, n_jobs = -1, random_state = None)
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

###### `z_to_r`

```python
z_to_r()
```

Convert z score back into r value for each element of data object

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

**Methods:**

Name | Description
---- | -----------
[`align`](#align) | 
[`anova`](#anova) | 
[`apply`](#apply) | Call ``BrainData.<method>(*args, **kwargs)`` on every item in parallel.
[`cleanup`](#cleanup) | Remove ``cache_root`` and invalidate every clone derived from ``self``.
[`cleanup_all`](#cleanup_all) | Remove every ``.nltools_cache/{run_id}/`` under ``directory``.
[`compute_contrasts`](#compute_contrasts) | Compute per-subject contrast maps from fit-bundle items.
[`concat`](#concat) | 
[`cv`](#cv) | Build a CV pipeline for cross-subject prediction. See ``pipeline.py``.
[`detrend`](#detrend) | 
[`filter`](#filter) | Filter to a subset by predicate, mask, or boolean Series.
[`fit`](#fit) | Per-subject fit; returns a path-backed collection of HDF5 fit bundles.
[`from_bids`](#from_bids) | Auto-pair BOLD with events.tsv (→ ``DesignMatrix``) and confounds.tsv.
[`from_glob`](#from_glob) | 
[`from_paths`](#from_paths) | 
[`isc`](#isc) | 
[`isc_test`](#isc_test) | 
[`iter_pairs`](#iter_pairs) | Yield ``(BrainData, DesignMatrix | None)`` pairs.
[`load`](#load) | Materialize path-backed items in place. Returns ``self`` for chaining.
[`map`](#map) | Apply an arbitrary ``fn(BrainData) -> BrainData`` to each item in parallel.
[`max`](#max) | 
[`mean`](#mean) | 
[`median`](#median) | 
[`memory_estimate`](#memory_estimate) | 
[`min`](#min) | 
[`permutation_test`](#permutation_test) | 
[`permutation_test2`](#permutation_test2) | 
[`predict`](#predict) | Two distinct paths, dispatched by argument:
[`read`](#read) | Inverse of ``write()``. Does not recover from cache subdirs in v0.6.0.
[`resample`](#resample) | 
[`smooth`](#smooth) | 
[`standardize`](#standardize) | 
[`std`](#std) | 
[`steps`](#steps) | List step subdirs under ``cache_root``, oldest to newest (lex-sorted).
[`sum`](#sum) | 
[`threshold`](#threshold) | 
[`transform_designs`](#transform_designs) | Map a function over paired ``DesignMatrix``es.
[`ttest`](#ttest) | 
[`ttest2`](#ttest2) | 
[`unload`](#unload) | Drop in-memory data for items with backing paths. Returns ``self``.
[`var`](#var) | 
[`write`](#write) | 

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`cache_root`](#cache_root) | <code>[Path](#pathlib.Path)</code> | 
[`designs`](#designs) | <code>[list](#list)</code> | 
[`is_loaded`](#is_loaded) | <code>[list](#list)[[bool](#bool)]</code> | Per-item flag — True iff the slot holds a ``BrainData`` (not a path).
[`mask`](#mask) | <code>[Nifti1Image](#nibabel.Nifti1Image)</code> | 
[`metadata`](#metadata) | <code>[DataFrame](#polars.DataFrame)</code> | 
[`n_subjects`](#n_subjects) | <code>[int](#int)</code> | 
[`n_voxels`](#n_voxels) | <code>[int](#int)</code> | Voxel count from the mask. Raises if mask is unset.
[`shape`](#shape) | <code>[tuple](#tuple)[[int](#int), [int](#int) \| None, [int](#int)]</code> | ``(n_subjects, n_obs_or_None_if_ragged, n_voxels)``.

``cache_dir`` precedence: explicit arg → ``NLTOOLS_CACHE_DIR`` env →
``./.nltools_cache``. Pass ``None`` for an auto-cleaned tempdir.
Resolved at construction and frozen on the instance.

##### Methods

###### `align`

```python
align(*, method: str = 'procrustes', scheme: str = 'searchlight', radius_mm: float = 10.0, parcellation: nib.Nifti1Image | None = None, n_features: int | None = None, n_iter: int = 3, device: str = 'cpu', return_model: bool = False, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto')
```

###### `anova`

```python
anova(groups: str | list | np.ndarray) -> dict
```

###### `apply`

```python
apply(method: str, *args: str, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto', **kwargs: Literal['auto', True, False]) -> BrainCollection
```

Call ``BrainData.<method>(*args, **kwargs)`` on every item in parallel.

All per-subject methods (``smooth``, ``standardize``, ...) reduce to
this. Centralizes the ``_apply`` plumbing and the cache-knob handling.

###### `cleanup`

```python
cleanup() -> None
```

Remove ``cache_root`` and invalidate every clone derived from ``self``.

###### `cleanup_all`

```python
cleanup_all(directory: Path | str = '.') -> None
```

Remove every ``.nltools_cache/{run_id}/`` under ``directory``.

Wide brush — can kill sibling sessions in the same cwd. Prefer
``bc.cleanup()`` for surgical removal.

###### `compute_contrasts`

```python
compute_contrasts(contrasts: str | list[str] | dict[str, np.ndarray], *, contrast_type: str = 'beta', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection | dict[str, BrainCollection]
```

Compute per-subject contrast maps from fit-bundle items.

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | single contrast + single ``contrast_type`` → ``BrainCollection``
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | multiple contrasts                          → ``dict[str, BrainCollection]``
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | ``contrast_type='all'``                     → ``dict['beta'|'t'|'z'|'p'|'se', BrainCollection]``

###### `concat`

```python
concat() -> BrainData
```

###### `cv`

```python
cv(*, k: int | None = None, method: str = 'kfold', split_by: str | None = None, groups: np.ndarray | None = None, n: int = 1000, random_state: int | None = None) -> BrainCollectionPipeline
```

Build a CV pipeline for cross-subject prediction. See ``pipeline.py``.

###### `detrend`

```python
detrend(*, method: str = 'linear', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

###### `filter`

```python
filter(predicate: Callable | list | np.ndarray | pl.Series | pd.Series) -> BrainCollection
```

Filter to a subset by predicate, mask, or boolean Series.

###### `fit`

```python
fit(model: str = 'glm', X: DesignMatrix | list | Callable | None = None, *, scale: bool = True, scale_value: float = 100.0, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto', **model_kwargs: Literal['auto', True, False]) -> BrainCollection
```

Per-subject fit; returns a path-backed collection of HDF5 fit bundles.

``X`` resolution priority:
  - ``None``         → use ``self.designs`` (must be set per subject)
  - ``DesignMatrix`` → shared across all subjects
  - ``list``         → per-subject (len == n_subjects)
  - ``callable``     → ``fn(ctx: _DesignContext) -> DesignMatrix``

###### `from_bids`

```python
from_bids(root: Path | str | Any, *, mask: nib.Nifti1Image | Path | str, task: str | None = None, space: str | None = None, sub_labels: list[str] | None = None, img_filters: list[tuple[str, str]] | None = None, derivatives_folder: str = 'derivatives', pair_events: bool = True, confounds_strategy: str | tuple[str, ...] | None = None, confounds_kwargs: dict | None = None, TR: float | str = 'infer', cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Auto-pair BOLD with events.tsv (→ ``DesignMatrix``) and confounds.tsv.

Full design and edge cases: SPEC §"``from_bids`` — concrete design".

###### `from_glob`

```python
from_glob(pattern: str, *, mask: nib.Nifti1Image | Path | str, design_pattern: str | None = None, pattern_groups: dict[str, int] | str | None = None, sort: bool = True, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

###### `from_paths`

```python
from_paths(brain_paths: list, *, mask: nib.Nifti1Image | Path | str, design_paths: list | None = None, metadata: pl.DataFrame | pd.DataFrame | dict | None = None, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

###### `isc`

```python
isc(*, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float | None = 6.0, metric: str = 'median', device: str = 'cpu', n_jobs: int = -1, progress_bar: bool = False) -> dict
```

###### `isc_test`

```python
isc_test(*, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float | None = 6.0, n_permute: int = 5000, permutation_method: str = 'bootstrap', metric: str = 'median', device: str = 'cpu', n_jobs: int = -1, progress_bar: bool = False, random_state: int | None = None) -> dict
```

###### `iter_pairs`

```python
iter_pairs() -> Iterator[tuple]
```

Yield ``(BrainData, DesignMatrix | None)`` pairs.

###### `load`

```python
load(indices: list[int] | None = None) -> BrainCollection
```

Materialize path-backed items in place. Returns ``self`` for chaining.

###### `map`

```python
map(fn: Callable, *, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Apply an arbitrary ``fn(BrainData) -> BrainData`` to each item in parallel.

###### `max`

```python
max() -> BrainData
```

###### `mean`

```python
mean() -> BrainData
```

###### `median`

```python
median() -> BrainData
```

###### `memory_estimate`

```python
memory_estimate() -> str
```

###### `min`

```python
min() -> BrainData
```

###### `permutation_test`

```python
permutation_test(*, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

###### `permutation_test2`

```python
permutation_test2(other: BrainCollection, *, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

###### `predict`

```python
predict(y: str | list | np.ndarray | None = None, *, X_new: np.ndarray | None = None, spatial_scale: str = 'whole_brain', estimator: str = 'svm', cv: int | str = 'loso', groups: str | np.ndarray | None = None, roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float = 10.0, scoring: str = 'accuracy', standardize: bool = True, return_weights: bool = True, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto', **kwargs: Literal['auto', True, False])
```

Two distinct paths, dispatched by argument:

  ``y=`` only    → group MVPA (subjects as samples) → ``BrainData``
  ``X_new=`` only → per-subject predict-after-fit  → ``BrainCollection``
  both / neither → raise

``predict(y=...)`` requires single-map-per-subject items (run
``compute_contrasts(...)`` first if you have GLM/ridge bundles).

###### `read`

```python
read(directory: Path | str, *, mask: nib.Nifti1Image | Path | str, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Inverse of ``write()``. Does not recover from cache subdirs in v0.6.0.

###### `resample`

```python
resample(target, *, interpolation: str = 'continuous', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

###### `smooth`

```python
smooth(fwhm: float, *, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

###### `standardize`

```python
standardize(*, axis: int = 0, method: str = 'center', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

###### `std`

```python
std() -> BrainData
```

###### `steps`

```python
steps() -> list[Path]
```

List step subdirs under ``cache_root``, oldest to newest (lex-sorted).

###### `sum`

```python
sum() -> BrainData
```

###### `threshold`

```python
threshold(*, lower: float | None = None, upper: float | None = None, binarize: bool = False, coerce_nan: bool = True, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

###### `transform_designs`

```python
transform_designs(fn: Callable, *, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Map a function over paired ``DesignMatrix``es.

``fn`` may take either a ``DesignMatrix`` or a ``DesignContext``;
the wrapper inspects arity and dispatches.

###### `ttest`

```python
ttest(*, popmean: float = 0.0) -> dict
```

###### `ttest2`

```python
ttest2(other: BrainCollection, *, equal_var: bool = True) -> dict
```

###### `unload`

```python
unload(indices: list[int] | None = None) -> BrainCollection
```

Drop in-memory data for items with backing paths. Returns ``self``.

###### `var`

```python
var() -> BrainData
```

###### `write`

```python
write(directory: Path | str, *, pattern: str = 'image_{i:04d}.nii.gz', metadata_file: str | None = 'metadata.csv') -> list[Path]
```

#### `BrainData`

```python
BrainData(data = None, *, Y = None, X = None, mask = None, masker = None, h5_compression = 'gzip', verbose = False, resample = True, interpolation = 'auto')
```

BrainData is a class to represent neuroimaging data in python as a vector
rather than a 3-dimensional matrix. This makes it easier to perform data
manipulation and analyses.

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
[`align`](#align) | Align BrainData instance to target object using functional alignment.
[`append`](#append) | Append data to BrainData instance.
[`apply_mask`](#apply_mask) | Mask BrainData instance using nilearn functionality.
[`astype`](#astype) | Cast BrainData.data as type.
[`bootstrap`](#bootstrap) | Bootstrap statistics using efficient online algorithms.
[`cluster_report`](#cluster_report) | Generate a cluster report with anatomical labels.
[`compute_contrasts`](#compute_contrasts) | Compute contrasts from fitted GLM results.
[`copy`](#copy) | Create a deep copy of a BrainData instance.
[`create_empty`](#create_empty) | Create a copy of BrainData with empty data array.
[`decompose`](#decompose) | Decompose BrainData object.
[`detrend`](#detrend) | Remove linear trend from each voxel.
[`distance`](#distance) | Calculate distance between images within a BrainData() instance.
[`extract_roi`](#extract_roi) | Extract activity from mask or ROI atlas using NiftiLabelsMasker.
[`filter`](#filter) | Apply butterworth filter to data. Wraps nilearn.signal.clean.
[`find_spikes`](#find_spikes) | Identify spikes from Time Series Data.
[`fit`](#fit) | Fit a model to brain imaging data.
[`icc`](#icc) | Calculate voxel-wise intraclass correlation coefficient.
[`iplot`](#iplot) | Interactive HTML viewer with threshold panel (and volume slider for 4D).
[`mean`](#mean) | Get mean of each voxel or image.
[`median`](#median) | Get median of each voxel or image.
[`multivariate_similarity`](#multivariate_similarity) | Predict spatial distribution of BrainData() instance from linear
[`plot`](#plot) | Plot BrainData instance using nilearn visualization or matplotlib.
[`plot_flatmap`](#plot_flatmap) | Plot brain data on cortical flatmap.
[`plot_surf`](#plot_surf) | Render this BrainData on fsaverage surfaces as a tight 2×2 montage.
[`predict`](#predict) | Predict voxel timeseries (encoding) or decode labels (MVPA).
[`predict_multi`](#predict_multi) | Deprecated: removed in v0.6.0; will return in a future Model class.
[`r_to_z`](#r_to_z) | Apply Fisher's r to z transformation to each element of the data
[`regions`](#regions) | Extract brain connected regions into separate regions.
[`regress`](#regress) | Deprecated: Use fit(model='glm', X=design_matrix) instead.
[`resample_to`](#resample_to) | Resample BrainData to match target image or resolution.
[`scale`](#scale) | Scale data via mean scaling.
[`similarity`](#similarity) | Calculate similarity of BrainData() instance with single
[`smooth`](#smooth) | Apply spatial smoothing using nilearn smooth_img().
[`standardize`](#standardize) | Standardize BrainData() instance.
[`std`](#std) | Get standard deviation of each voxel or image.
[`sum`](#sum) | Get sum of each voxel or image.
[`temporal_resample`](#temporal_resample) | Resample BrainData timeseries to a new target frequency or number of samples.
[`threshold`](#threshold) | Threshold BrainData instance with optional cluster filtering.
[`to_nifti`](#to_nifti) | Convert BrainData Instance into Nifti Object.
[`transform_pairwise`](#transform_pairwise) | Transform data into pairwise comparisons.
[`ttest`](#ttest) | One-sample voxelwise t-test across images (axis 0).
[`ttest2`](#ttest2) | Two-sample voxelwise t-test between two BrainData stacks.
[`upload_neurovault`](#upload_neurovault) | Upload Data to Neurovault.  Will add any columns in self.X to image
[`write`](#write) | Write out BrainData object to Nifti or HDF5 File.
[`z_to_r`](#z_to_r) | Convert z score back into r value for each element of data object.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`X`](#X) |  | Design matrix / per-image covariates as a polars DataFrame.
[`Y`](#Y) |  | Per-image targets as a polars DataFrame.
[`data`](#data) |  | 
[`design_matrix`](#design_matrix) |  | 
[`dtype`](#dtype) |  | Get data type of BrainData.data.
[`is_empty`](#is_empty) | <code>[bool](#bool)</code> | Check if BrainData.data is empty.
[`masker`](#masker) |  | 
[`shape`](#shape) |  | Get images by voxels shape.
[`size`](#size) |  | Total number of elements in BrainData.data (numpy convention).
[`verbose`](#verbose) |  | 

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
`spatial_scale` | <code>[str](#str)</code> | ``'whole_brain'`` (default), ``'roi'``, or ``'searchlight'``. ``'roi'`` / ``'searchlight'`` are not yet implemented (per-parcel transforms + reassembly is a follow-up slice). | <code>'whole_brain'</code>
`roi_mask` |  | Reserved for ``spatial_scale='roi'``. | <code>None</code>
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
bootstrap(stat, n_samples = 5000, save_boots = False, percentiles = (2.5, 97.5), X_test = None, backend = None, max_gpu_memory_gb = 4.0, n_jobs = -1, random_state = None)
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
`atlas` | <code>[str](#str) \| [Sequence](#collections.abc.Sequence)[[str](#str)] \| None</code> | Atlas name or list of names (see :func:`list_atlases`). Defaults to ``("harvard_oxford", "aal", "schaefer_200")``. | <code>None</code>
`prob_threshold` | <code>[float](#float)</code> | Drop probabilistic-atlas regions below this %. | <code>5.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[ClusterReport](#nltools.data.atlases.ClusterReport)</code> | class:`~nltools.data.atlases.ClusterReport` with ``peaks``,
<code>[ClusterReport](#nltools.data.atlases.ClusterReport)</code> | ``clusters`` (polars DataFrames), and ``stat_img`` (BrainData).

###### `compute_contrasts`

```python
compute_contrasts(contrasts, contrast_type = 't')
```

Compute contrasts from fitted GLM results.

This method computes contrasts as linear combinations of the GLM beta coefficients.
Must be called after .fit(model='glm', X=design_matrix) has been run.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`contrasts` |  | Can be:<br>- str: A string specifying the contrast using column names   e.g., "conditionA - conditionB" or "2*conditionA - conditionB - conditionC" - dict: Dictionary with contrast names as keys and contrast strings/vectors as values   e.g., {"main_effect": "conditionA - conditionB", "interaction": [1, -1, -1, 1]} - array: Numeric contrast vector matching the number of regressors   e.g., [1, -1, 0, 0] for a 4-regressor model | *required*
`contrast_type` | <code>[str](#str)</code> | Type of contrast statistic ('t' or 'F'). Default: 't' Note: Currently only 't' contrasts are supported. | <code>'t'</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or dict: If single contrast, returns BrainData object with contrast map.                If multiple contrasts (dict input), returns dict of BrainData objects.

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

###### `create_empty`

```python
create_empty()
```

Create a copy of BrainData with empty data array.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | A copy of this object with an empty data array.

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
`spatial_scale` | <code>[str](#str)</code> | One of ``'whole_brain'`` (default), ``'roi'``, or ``'searchlight'``. ``'whole_brain'`` returns a single pairwise distance ``Adjacency`` between images. ``'roi'`` requires ``roi_mask`` and returns a stacked ``Adjacency`` with one RDM per parcel and ``spatial_scale`` provenance attached for back-projection via ``Adjacency.to_brain()``. ``'searchlight'`` requires ``radius_mm`` (and is not yet implemented in this slice). | <code>'whole_brain'</code>
`roi_mask` |  | Atlas image (BrainData / Nifti1Image / path) for ``spatial_scale='roi'``. | <code>None</code>
`radius_mm` | <code>[float](#float)</code> | Searchlight radius in mm. Default 10.0. | <code>10.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | Single pairwise distance matrix for ``'whole_brain'``; stacked Adjacency (one matrix per parcel/searchlight) with ``spatial_scale`` set for ``'roi'`` / ``'searchlight'``.

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
filter(sampling_freq = None, high_pass = None, low_pass = None, **kwargs)
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
fit(model = 'glm', *, X = None, cv = None, local_alpha = True, fit_intercept = False, inplace = True, scale = True, scale_value = 100.0, progress_bar = None, design_clean = True, design_clean_thresh = 0.95, design_clean_exclude_confounds = False, design_clean_fill_na = 0, **kwargs)
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
`inplace` | <code>bool, default=True</code> | If True, mutate self and return self. If False, return Fit dataclass with results (self unchanged). | <code>True</code>
`scale` | <code>bool, default=True</code> | Apply grand-mean scaling before fitting. | <code>True</code>
`scale_value` | <code>float, default=100.0</code> | Target value for mean after scaling. | <code>100.0</code>
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
(``"A - B"``, ``[1, -1, 0, ...]``) use :meth:`compute_contrasts` —
you cannot correctly combine these per-regressor maps by hand
because t-statistic arithmetic requires the off-diagonal elements
of the parameter covariance matrix, which are not stored. Pass
``contrast_type="all"`` to get ``β``/``t``/``z``/``p``/``se`` for
one contrast in a single call.

</details>

**Examples:**

```pycon
>>> brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
>>> fit = brain_data.fit(model='ridge', alpha=1.0, X=features, inplace=False)
```

###### `icc`

```python
icc(n_subjects, n_sessions, method = 'icc2', parallel = None, n_jobs = -1, max_gpu_memory_gb = 4.0)
```

Calculate voxel-wise intraclass correlation coefficient.

ICC Formulas based on Shrout & Fleiss (1979).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_subjects` |  | Number of subjects in the data | *required*
`n_sessions` |  | Number of sessions per subject | *required*
`method` |  | Type of ICC ('icc1', 'icc2', 'icc3'). Default: 'icc2' | <code>'icc2'</code>
`parallel` |  | Parallelization method (None, 'cpu', 'gpu') | <code>None</code>
`n_jobs` |  | Number of CPU cores (-1 = all cores) | <code>-1</code>
`max_gpu_memory_gb` |  | GPU memory budget in GB | <code>4.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with ICC map (shape: (1, n_voxels))

**Examples:**

```pycon
>>> icc_map = data.icc(n_subjects=20, n_sessions=3, method='icc2')
```

###### `iplot`

```python
iplot(*, view: str = 'ortho', mode: str = 'symmetric', units: str = 'value', lower: float | None = None, upper: float | None = None, threshold: float | None = None, bg_img: float | None = None, cut_coords: float | None = None, cmap: str | None = None, symmetric_cmap: bool = True, **kwargs: bool)
```

Interactive HTML viewer with threshold panel (and volume slider for 4D).

Returns a `BrainViewerWidget` (anywidget) wrapping nilearn's HTML
ortho or surface viewer. Renders inline in Jupyter, marimo, and
Jupyter Book v2 / mystmd built sites.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`view` | <code>[str](#str)</code> | ``"ortho"`` (default, uses ``nilearn.view_img``) or ``"surface"`` (uses ``nilearn.view_img_on_surf``). | <code>'ortho'</code>
`mode` | <code>[str](#str)</code> | ``"symmetric"`` (default, single ``|x| ≥ upper`` slider) or ``"independent"`` (separate negative/positive cutoffs; voxels in ``(lower, upper)`` are masked). | <code>'symmetric'</code>
`units` | <code>[str](#str)</code> | ``"value"`` (default) or ``"percentile"``. Toggleable in the UI; this just sets the initial state. | <code>'value'</code>
`lower` | <code>[float](#float) \| None</code> | Initial negative cutoff for ``mode="independent"`` (must be ≤ 0). Ignored in symmetric mode. | <code>None</code>
`upper` | <code>[float](#float) \| None</code> | Initial threshold for symmetric mode (``|x| ≥ upper``) or positive cutoff for independent mode (must be ≥ 0). | <code>None</code>
`threshold` | <code>[float](#float) \| None</code> | Deprecated alias for ``upper`` (symmetric mode). | <code>None</code>
`bg_img` |  | Background image (ortho only). Defaults to nilearn's MNI152. | <code>None</code>
`cut_coords` |  | Initial cut coordinates (ortho only). | <code>None</code>
`cmap` | <code>[str](#str) \| None</code> | Colormap name. | <code>None</code>
`symmetric_cmap` | <code>[bool](#bool)</code> | Whether the colormap is symmetric around zero. | <code>True</code>
`**kwargs` |  | Forwarded to ``nilearn.view_img`` / ``nilearn.view_img_on_surf``. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | BrainViewerWidget

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

###### `multivariate_similarity`

```python
multivariate_similarity(images, method = 'ols')
```

Predict spatial distribution of BrainData() instance from linear
combination of other BrainData() instances or Nibabel images.

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
plot(method = 'glass', upper = None, lower = None, threshold = None, view = 'z', cut_coords = None, cmap = None, bg_img = None, ax = None, figsize = (8, 6), title = None, colorbar = True, save = None, stat = 'mean', limit = 3, **kwargs)
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

###### `plot_flatmap`

```python
plot_flatmap(threshold = None, cmap = 'RdBu_r', vmax = None, vmin = None, template = 'fsaverage5', with_curvature = True, curvature_contrast = 0.5, curvature_brightness = 0.5, transparency = 'auto', colorbar = True, colorbar_orientation = 'horizontal', figsize = (12, 6), title = None, radius_mm = 3.0, interpolation = 'linear', axes = None, save = None)
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

###### `plot_surf`

```python
plot_surf(*, hemi = 'both', view = 'montage', surface = 'pial', template = 'fsaverage5', threshold = None, cmap = 'RdBu_r', vmin = None, vmax = None, transparency = 'auto', bg_on_data = False, colorbar = True, colorbar_orientation = 'horizontal', figsize = (10, 8), title = None, radius_mm = 3.0, interpolation = 'linear', zoom = 1.2, axes = None, save = None)
```

Render this BrainData on fsaverage surfaces as a tight 2×2 montage.

Facade over :func:`nltools.plotting.plot_surf`. See that function's
docstring for the full argument reference. Notable defaults:
``surface="pial"``, ``zoom=1.2``, ``transparency="auto"`` (uses
this instance's ``.mask``).

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

###### `predict`

```python
predict(*, y: np.ndarray | None = None, X: np.ndarray | None = None, spatial_scale: str = 'whole_brain', model: str = 'svm', cv: int = 5, standardize: bool = True, reduce: str | None = None, n_components: int | None = None, scoring: str = 'auto', groups: np.ndarray | None = None, roi_mask: np.ndarray | None = None, radius_mm: float = 10.0, inplace: bool = False, n_jobs: int = 1, progress_bar: bool = False)
```

Predict voxel timeseries (encoding) or decode labels (MVPA).

Dispatched by which of ``X`` or ``y`` is provided:

1. **Timeseries prediction** (``X`` provided): use a fitted ridge /
   GLM encoding model on ``self`` to predict voxel responses.
   Returns a fresh ``BrainData`` whose ``.data`` holds the predicted
   timeseries (composes directly with ``.plot()``, ``.standardize()``
   etc.). ``inplace`` has no effect in this mode.
2. **MVPA decoding** (``y`` provided): train a classifier or
   regressor with cross-validation. Returns a :class:`Predict`
   dataclass. Spatial fields (``weight_map``, ``fold_weight_maps``,
   ``final_weight_map``, ``accuracy_map``) are :class:`BrainData`
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
`inplace` | <code>[bool](#bool)</code> | If ``True``, populate result fields as ``predict_*`` attributes on ``self`` and return ``self``. Default ``False`` returns a fresh :class:`Predict`. | <code>False</code>
`n_jobs` | <code>[int](#int)</code> | Parallel jobs for searchlight / ROI. Default ``1``; searchlight on a real brain at higher ``n_jobs`` can be memory-heavy. | <code>1</code>
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
False because we detect the Pipeline::

    from sklearn.feature_selection import SelectKBest
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    pipe = make_pipeline(StandardScaler(), SelectKBest(k=500),
                         LinearSVC())
    result = brain.predict(y=labels, model=pipe)

###### `predict_multi`

```python
predict_multi(*args, **kwargs)
```

Deprecated: removed in v0.6.0; will return in a future Model class.

Per the v0.6 migration guide, the multi-method MVPA wrapper has
been removed. Use :meth:`predict` for whole-brain MVPA, or compose
sklearn estimators directly via the new Model API.

###### `r_to_z`

```python
r_to_z()
```

Apply Fisher's r to z transformation to each element of the data
object.

###### `regions`

```python
regions(min_region_size = 1350, method = 'local_regions', smoothing_fwhm = 6, is_mask = False)
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

###### `regress`

```python
regress(design_matrix = None, method = 'ols', mode = None)
```

Deprecated: Use fit(model='glm', X=design_matrix) instead.

.. deprecated:: 0.6.0
    Use :meth:`fit` with ``model='glm'`` instead.

###### `resample_to`

```python
resample_to(img = None, resolution = None, interpolation = None)
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
similarity(image, method = 'correlation')
```

Calculate similarity of BrainData() instance with single
BrainData or Nibabel image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`image` |  | (BrainData, nifti) image to evaluate similarity | *required*
`method` |  | (str) Type of similarity     ['correlation','dot_product','cosine'] | <code>'correlation'</code>

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
standardize(axis = 0, method = 'center', verbose = True)
```

Standardize BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>[int](#int)</code> | 0 standardizes each voxel across observations (default). 1 standardizes each observation across voxels. | <code>0</code>
`method` | <code>[str](#str)</code> | 'center' subtracts the mean (default). 'zscore' subtracts the mean and divides by standard deviation. | <code>'center'</code>
`verbose` | <code>[bool](#bool)</code> | If False, suppress sklearn numerical warnings that occur when voxels have near-zero variance. Default: True. | <code>True</code>

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

###### `temporal_resample`

```python
temporal_resample(sampling_freq = None, target = None, target_type = 'hz')
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
threshold(upper = None, lower = None, binarize = False, coerce_nan = True, cluster_threshold = 0)
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

###### `to_nifti`

```python
to_nifti()
```

Convert BrainData Instance into Nifti Object.

**Returns:**

Type | Description
---- | -----------
 | nibabel.Nifti1Image: Brain data as a NIfTI image.

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
ttest(popmean = 0.0, permutation = False, n_permute = 5000, tail = 2, return_null = False, n_jobs = -1, random_state = None)
```

One-sample voxelwise t-test across images (axis 0).

Tests whether the per-voxel mean across images differs from
``popmean``. Operates on a stack of images (e.g. subject-level
contrast maps) with shape ``(n_samples, n_voxels)``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`popmean` |  | Population mean to test against. Default 0.0. | <code>0.0</code>
`permutation` |  | If True, use sign-flip permutation test via :func:`nltools.stats.one_sample_permutation_test`. | <code>False</code>
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

###### `upload_neurovault`

```python
upload_neurovault(access_token = None, collection_name = None, collection_id = None, img_type = None, img_modality = None, **kwargs)
```

Upload Data to Neurovault.  Will add any columns in self.X to image
    metadata. Index will be used as image name.

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

#### `DesignMatrix`

```python
DesignMatrix(data: DesignMatrix | pl.DataFrame | pd.DataFrame | np.ndarray | dict | str | Path | None = None, *, sampling_freq: float | None = None, TR: float | None = None, run_length: int | str | None = None, columns: list[str] | None = None, convolved: list[str] | None = None, confounds: list[str] | None = None, hrf_model: str | None = 'glover')
```

Polars-based design matrix for experimental designs in neuroimaging.

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
[`sampling_freq`](#sampling_freq) | <code>[float](#float) or None</code> | Sampling frequency in Hz
[`convolved`](#convolved) | <code>list of str</code> | Columns that have been convolved
[`confounds`](#confounds) | <code>list of str</code> | Nuisance/confound columns (intercept, polynomial trends, DCT bases, motion, physio, …) — these are skipped by ``.convolve()`` and kept separate per run on multi-run vertical append.
[`multi`](#multi) | <code>[bool](#bool)</code> | True if created from multi-run concatenation

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
[`add_dct_basis`](#add_dct_basis) | Add discrete cosine transform basis functions (high-pass filter).
[`add_poly`](#add_poly) | Add Legendre polynomial drift terms.
[`append`](#append) | Concatenate design matrices.
[`clean`](#clean) | Remove highly correlated columns.
[`convolve`](#convolve) | Convolve columns with HRF or custom kernel.
[`copy`](#copy) | Create a deep copy of the DesignMatrix.
[`corr`](#corr) | Correlation between columns as a similarity ``Adjacency``.
[`downsample`](#downsample) | Reduce temporal resolution to target frequency using Polars-native operations.
[`drop`](#drop) | Drop specified columns.
[`fillna`](#fillna) | Fill NaN/null values with specified value.
[`plot`](#plot) | Visualize the design matrix.
[`replace_data`](#replace_data) | Replace data columns while preserving confounds and metadata.
[`standardize`](#standardize) | Standardize columns using the specified method.
[`sum`](#sum) | Compute sum along axis.
[`to_numpy`](#to_numpy) | Convert DesignMatrix to numpy array.
[`to_pandas`](#to_pandas) | Convert DesignMatrix to pandas DataFrame.
[`upsample`](#upsample) | Increase temporal resolution to target frequency.
[`vif`](#vif) | Compute variance inflation factor for each column.
[`with_columns`](#with_columns) | Add or replace columns via Polars expressions.
[`write`](#write) | Write DesignMatrix to file.
[`zscore`](#zscore) | Z-score standardize columns (mean=0, std=1).

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

###### `add_dct_basis`

```python
add_dct_basis(duration: float = 180, drop: int = 0, *, include_constant: bool = True) -> DesignMatrix
```

Add discrete cosine transform basis functions (high-pass filter).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`duration` | <code>[float](#float)</code> | Filter duration in seconds. Default: 180. | <code>180</code>
`drop` | <code>[int](#int)</code> | Number of low-frequency bases to drop. Default: 0. | <code>0</code>
`include_constant` | <code>[bool](#bool)</code> | If True, also add a constant/intercept column named ``cosine_0`` (analogous to ``poly_0`` in :meth:`add_poly`). The underlying DCT basis drops the constant per SPM convention; set False to match SPM behavior. Default: True. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with DCT basis columns appended.

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
append(dm: DesignMatrix | list[DesignMatrix], *, axis: int = 0, keep_separate: bool = True, unique_cols: list[str] | None = None, fill_na: int | float = 0, as_confounds: bool = False, verbose: bool = False) -> DesignMatrix
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
`verbose` | <code>[bool](#bool)</code> | Print messages about confound separation. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Concatenated design matrix.

###### `clean`

```python
clean(fill_na: int | float | None = 0, exclude_confounds: bool = False, thresh: float = 0.95, verbose: bool = True) -> DesignMatrix
```

Remove highly correlated columns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fill_na` | <code>int, float, or None</code> | Fill NaN values before checking correlations (default 0) | <code>0</code>
`exclude_confounds` | <code>[bool](#bool)</code> | Skip confound/nuisance columns from correlation check | <code>False</code>
`thresh` | <code>[float](#float)</code> | Correlation threshold (drop if abs(r) >= thresh, default 0.95) | <code>0.95</code>
`verbose` | <code>[bool](#bool)</code> | Print dropped column names | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Cleaned matrix with highly correlated columns removed

###### `convolve`

```python
convolve(conv_func: str | np.ndarray = 'hrf', columns: list[str] | None = None) -> DesignMatrix
```

Convolve columns with HRF or custom kernel.

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

###### `corr`

```python
corr(*, metric: str = 'pearson', columns: list[str] | None = None)
```

Correlation between columns as a similarity ``Adjacency``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` | <code>[str](#str)</code> | ``'pearson'`` (default) or ``'spearman'``. | <code>'pearson'</code>
`columns` | <code>list of str</code> | Subset of columns to correlate. Defaults to all columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | Similarity matrix whose ``labels`` are the column names. The unit diagonal is dropped (self-correlation isn't an edge); use ``.plot(method='corr')`` for a heatmap with the diagonal restored.

###### `downsample`

```python
downsample(target: float, method: str = 'mean', **kwargs: str) -> DesignMatrix
```

Reduce temporal resolution to target frequency using Polars-native operations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` | <code>[float](#float)</code> | Target sampling frequency in Hz (must be < current sampling_freq) | *required*
`method` | <code>[str](#str)</code> | Aggregation method - 'mean' or 'median' (default: 'mean') | <code>'mean'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Downsampled DesignMatrix with updated sampling_freq

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
  :meth:`corr`; diagonal restored to 1.0 for display).

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
`columns` | <code>[list](#list)[[str](#str)] \| None</code> | Columns to standardize. If None, standardize all non-polynomial columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with standardized columns.

###### `sum`

```python
sum(axis: int = 0) -> pl.Series
```

Compute sum along axis.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>int, default=0</code> | 0: sum down columns, 1: sum across rows. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>[Series](#polars.Series)</code> | pl.Series: Sums along specified axis.

###### `to_numpy`

```python
to_numpy() -> np.ndarray
```

Convert DesignMatrix to numpy array.

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | np.ndarray: 2D array with shape (n_samples, n_columns)

###### `to_pandas`

```python
to_pandas() -> pd.DataFrame
```

Convert DesignMatrix to pandas DataFrame.

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#pandas.DataFrame)</code> | pd.DataFrame: Pandas DataFrame with same data and column names.

###### `upsample`

```python
upsample(target: float, method: str = 'linear', **kwargs: str) -> DesignMatrix
```

Increase temporal resolution to target frequency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` | <code>[float](#float)</code> | Target sampling frequency in Hz (must be > current sampling_freq) | *required*
`method` | <code>[str](#str)</code> | Interpolation method - 'linear' or 'nearest' (default: 'linear') | <code>'linear'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Upsampled DesignMatrix with updated sampling_freq

###### `vif`

```python
vif(exclude_confounds: bool = True) -> np.ndarray | None
```

Compute variance inflation factor for each column.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`exclude_confounds` | <code>[bool](#bool)</code> | Skip confound/nuisance columns. Default: True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray) \| None</code> | np.ndarray: VIF values for each included column. Returns None if the correlation matrix is singular.

###### `with_columns`

```python
with_columns(*exprs, **named_exprs) -> DesignMatrix
```

Add or replace columns via Polars expressions.

Mirrors :meth:`polars.DataFrame.with_columns`. Named kwargs become
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

###### `zscore`

```python
zscore(columns: list[str] | None = None) -> DesignMatrix
```

Z-score standardize columns (mean=0, std=1).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`columns` | <code>list of str</code> | Columns to standardize. If None, standardize all non-polynomial columns. | <code>None</code>

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

Attributes
----------
fitted_values : ndarray
    Fitted values or predictions, always present
weights : ndarray, optional
    Model coefficients (Ridge)
scores : ndarray, optional
    R² scores (Ridge)
betas : ndarray, optional
    Beta coefficients (GLM)
t_stats : ndarray, optional
    T-statistics (GLM)
p_values : ndarray, optional
    P-values (GLM)
se : ndarray, optional
    Standard errors (GLM)
residuals : ndarray, optional
    Residuals (GLM)
r2 : ndarray, optional
    R² values (GLM)
cv_scores : ndarray, optional
    Per-fold cross-validation scores
cv_mean_score : ndarray, optional
    Mean cross-validation score across folds
cv_predictions : ndarray, optional
    Out-of-fold predictions
cv_folds : ndarray, optional
    Fold indices for each sample
cv_best_alpha : float, optional
    Best alpha selected via cross-validation
cv_alpha_scores : ndarray, optional
    Cross-validation scores for each alpha tested

Methods
-------
available() : list
    Returns list of non-None attribute names (excludes private fields)
asdict(include_none=False) : dict
    Converts to dictionary, optionally excluding None values

Examples
--------
**Creating a Fit object (Ridge without CV):**

>>> import numpy as np
>>> from nltools.data.fitresults import Fit
>>> fit = Fit(
...     fitted_values=np.random.randn(100, 1000),
...     weights=np.random.randn(5, 1000),
...     scores=np.random.randn(1000)
... )
>>> fit.available()
['fitted_values', 'weights', 'scores']

**Creating a Fit object (Ridge with CV):**

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

**Immutability:**

>>> try:
...     fit.scores = np.zeros(1000)  # Will raise FrozenInstanceError
... except AttributeError:
...     print("Cannot modify frozen dataclass")
Cannot modify frozen dataclass

**Export/serialization:**

>>> # Save to .npz
>>> np.savez("results.npz", **fit.asdict())
>>>
>>> # Load and reconstruct
>>> loaded = np.load("results.npz")
>>> fit_reloaded = Fit(**{k: loaded[k] for k in loaded.files})

Notes
-----
- Frozen dataclass ensures results cannot be accidentally modified
- All attributes are numpy arrays (except cv_best_alpha which is float)
- None values indicate that field was not computed for this model/method
- Private fields (starting with _) are excluded from available() and asdict()

**Methods:**

Name | Description
---- | -----------
[`asdict`](#asdict) | Convert to dictionary.
[`available`](#available) | Return list of non-None attribute names.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`betas`](#betas) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`cv_alpha_scores`](#cv_alpha_scores) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`cv_best_alpha`](#cv_best_alpha) | <code>[float](#float) \| None</code> | 
[`cv_folds`](#cv_folds) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`cv_mean_score`](#cv_mean_score) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`cv_predictions`](#cv_predictions) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`cv_scores`](#cv_scores) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`fitted_values`](#fitted_values) | <code>[ndarray](#numpy.ndarray)</code> | 
[`p_values`](#p_values) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`r2`](#r2) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`residuals`](#residuals) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`scores`](#scores) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`se`](#se) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`t_stats`](#t_stats) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`weights`](#weights) | <code>[ndarray](#numpy.ndarray) \| None</code> | 

##### Methods

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

Examples
--------
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

Examples
--------
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

#### `Roc`

```python
Roc(input_values = None, binary_outcome = None, threshold_type = 'optimal_overall', forced_choice = None, **kwargs)
```

Roc Class

The Roc class is based on Tor Wager's Matlab roc_plot.m function and
allows a user to easily run different types of receiver operator
characteristic curves.  For example, one might be interested in single
interval or forced choice.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`input_values` |  | nibabel data instance | <code>None</code>
`binary_outcome` |  | vector of training labels | <code>None</code>
`threshold_type` |  | ['optimal_overall', 'optimal_balanced',             'minimum_sdt_bias'] | <code>'optimal_overall'</code>
`**kwargs` |  | Additional keyword arguments to pass to the prediction         algorithm | <code>{}</code>

**Methods:**

Name | Description
---- | -----------
[`calculate`](#calculate) | Calculate Receiver Operating Characteristic plot (ROC) for
[`plot`](#plot) | Create ROC Plot
[`summary`](#summary) | Display a formatted summary of ROC analysis.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`binary_outcome`](#binary_outcome) |  | 
[`forced_choice`](#forced_choice) |  | 
[`input_values`](#input_values) |  | 
[`threshold_type`](#threshold_type) |  | 

##### Methods

###### `calculate`

```python
calculate(input_values = None, binary_outcome = None, criterion_values = None, threshold_type = 'optimal_overall', forced_choice = None, balanced_acc = False)
```

Calculate Receiver Operating Characteristic plot (ROC) for
single-interval classification.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`input_values` |  | nibabel data instance | <code>None</code>
`binary_outcome` |  | vector of training labels | <code>None</code>
`criterion_values` |  | (optional) criterion values for calculating fpr             & tpr | <code>None</code>
`threshold_type` |  | ['optimal_overall', 'optimal_balanced',             'minimum_sdt_bias'] | <code>'optimal_overall'</code>
`forced_choice` |  | index indicating position for each unique subject             (default=None) | <code>None</code>
`balanced_acc` |  | balanced accuracy for single-interval classification             (bool). THIS IS NOT COMPLETELY IMPLEMENTED BECAUSE             IT AFFECTS ACCURACY ESTIMATES, BUT NOT P-VALUES OR             THRESHOLD AT WHICH TO EVALUATE SENS/SPEC | <code>False</code>
`**kwargs` |  | Additional keyword arguments to pass to the prediction             algorithm | *required*

###### `plot`

```python
plot(plot_method = 'gaussian', balanced_acc = False, **kwargs)
```

Create ROC Plot

Create a specific kind of ROC curve plot, based on input values
along a continuous distribution and a binary outcome variable (logical)

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`plot_method` |  | type of plot ['gaussian','observed'] | <code>'gaussian'</code>
`binary_outcome` |  | vector of training labels | *required*
`**kwargs` |  | Additional keyword arguments to pass to the prediction         algorithm | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | fig

###### `summary`

```python
summary()
```

Display a formatted summary of ROC analysis.

#### `SimulateGrid`

```python
SimulateGrid(grid_width = 100, signal_width = 20, n_subjects = 20, sigma = 1, signal_amplitude = None, random_state = None)
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
[`data`](#data) |  | The simulated data array of shape (n_subjects, grid_width, grid_width).
[`t_values`](#t_values) |  | T-statistic values after fitting.
[`p_values`](#p_values) |  | P-values after fitting.
[`thresholded`](#thresholded) |  | Thresholded statistical map.
[`isfit`](#isfit) |  | Whether fit() has been called.

**Examples:**

```pycon
>>> from nltools.data.simulator import SimulateGrid
>>> sim = SimulateGrid(signal_amplitude=0.5, random_state=42)
>>> sim.fit(n_permute=1000)
>>> sim.plot()
```

**Methods:**

Name | Description
---- | -----------
[`add_signal`](#add_signal) | Add rectangular signal to self.data
[`create_mask`](#create_mask) | Create a mask for where the signal is located in grid.
[`fit`](#fit) | Run ttest on self.data
[`plot_grid_simulation`](#plot_grid_simulation) | Create a plot of the simulations
[`run_multiple_simulations`](#run_multiple_simulations) | This method will run multiple simulations to calculate overall false positive rate
[`threshold_simulation`](#threshold_simulation) | Threshold simulation

##### Methods

###### `add_signal`

```python
add_signal(signal_width = 20, signal_amplitude = 1)
```

Add rectangular signal to self.data

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`signal_width` | <code>[int](#int)</code> | width of signal box | <code>20</code>
`signal_amplitude` | <code>[int](#int)</code> | intensity of signal | <code>1</code>

###### `create_mask`

```python
create_mask(signal_width)
```

Create a mask for where the signal is located in grid.

###### `fit`

```python
fit()
```

Run ttest on self.data

###### `plot_grid_simulation`

```python
plot_grid_simulation(threshold, threshold_type, n_simulations = 100, correction = None)
```

Create a plot of the simulations

###### `run_multiple_simulations`

```python
run_multiple_simulations(threshold, threshold_type, n_simulations = 100, correction = None)
```

This method will run multiple simulations to calculate overall false positive rate

###### `threshold_simulation`

```python
threshold_simulation(threshold, threshold_type, correction = None)
```

Threshold simulation

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`threshold` | <code>[float](#float)</code> | threshold to apply to simulation | *required*
`threshhold_type` | <code>[str](#str)</code> | type of threshold to use can be a specific t-value or p-value ['t', 'p', 'q'] | *required*

#### `Simulator`

```python
Simulator(brain_mask = None, output_dir = None, random_state = None)
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
[`brain_mask`](#brain_mask) |  | The brain mask image used for simulation.
[`output_dir`](#output_dir) |  | Output directory path.
[`random_state`](#random_state) |  | Random state for reproducible simulations.

**Examples:**

```pycon
>>> from nltools.data.simulator import Simulator
>>> sim = Simulator(random_state=42)
>>> # Create a dataset with signal in specific regions
>>> data = sim.create_data(y=[1, -1, 1, -1], sigma=1, n_reps=10)
```

**Methods:**

Name | Description
---- | -----------
[`create_cov_data`](#create_cov_data) | create continuous simulated data with covariance
[`create_data`](#create_data) | create simulated data with integers
[`create_ncov_data`](#create_ncov_data) | create continuous simulated data with covariance
[`gaussian`](#gaussian) | create a 3D gaussian signal normalized to a given intensity
[`n_spheres`](#n_spheres) | generate a set of spheres in the brain mask space
[`normal_noise`](#normal_noise) | produce a normal noise distribution for all all points in the brain mask
[`sphere`](#sphere) | create a sphere of given radius at some point p in the brain mask
[`to_nifti`](#to_nifti) | convert a numpy matrix to the nifti format and assign to it the brain_mask's affine matrix

##### Methods

###### `create_cov_data`

```python
create_cov_data(cor, cov, sigma, mask = None, reps = 1, n_sub = 1, output_dir = None)
```

create continuous simulated data with covariance

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cor` |  | amount of covariance between each voxel and Y variable | *required*
`cov` |  | amount of covariance between voxels | *required*
`sigma` |  | amount of noise to add | *required*
`radius` |  | vector of radius.  Will create multiple spheres if len(radius) > 1 | *required*
`center` |  | center(s) of sphere(s) of the form [px, py, pz] or [[px1, py1, pz1], ..., [pxn, pyn, pzn]] | *required*
`reps` |  | number of data repetitions | <code>1</code>
`n_sub` |  | number of subjects to simulate | <code>1</code>
`output_dir` |  | string path of directory to output data.  If None, no data will be written | <code>None</code>
`**kwargs` |  | Additional keyword arguments to pass to the prediction algorithm | *required*

###### `create_data`

```python
create_data(levels, sigma, radius = 5, center = None, reps = 1, output_dir = None)
```

create simulated data with integers

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`levels` |  | vector of intensities or class labels | *required*
`sigma` |  | amount of noise to add | *required*
`radius` |  | vector of radius.  Will create multiple spheres if len(radius) > 1 | <code>5</code>
`center` |  | center(s) of sphere(s) of the form [px, py, pz] or [[px1, py1, pz1], ..., [pxn, pyn, pzn]] | <code>None</code>
`reps` |  | number of data repetitions useful for trials or subjects | <code>1</code>
`output_dir` |  | string path of directory to output data.  If None, no data will be written | <code>None</code>
`**kwargs` |  | Additional keyword arguments to pass to the prediction algorithm | *required*

###### `create_ncov_data`

```python
create_ncov_data(cor, cov, sigma, masks = None, reps = 1, n_sub = 1, output_dir = None)
```

create continuous simulated data with covariance

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cor` |  | amount of covariance between each voxel and Y variable (an int or a vector) | *required*
`cov` |  | amount of covariance between voxels (an int or a matrix) | *required*
`sigma` |  | amount of noise to add | *required*
`mask` |  | region(s) where we will have activations (list if more than one) | *required*
`reps` |  | number of data repetitions | <code>1</code>
`n_sub` |  | number of subjects to simulate | <code>1</code>
`output_dir` |  | string path of directory to output data.  If None, no data will be written | <code>None</code>
`**kwargs` |  | Additional keyword arguments to pass to the prediction algorithm | *required*

###### `gaussian`

```python
gaussian(mu, sigma, i_tot)
```

create a 3D gaussian signal normalized to a given intensity

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mu` |  | average value of the gaussian signal (usually set to 0) | *required*
`sigma` |  | standard deviation | *required*
`i_tot` |  | sum total of activation (numerical integral over the gaussian returns this value) | *required*

###### `n_spheres`

```python
n_spheres(radius, center)
```

generate a set of spheres in the brain mask space

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`radius` |  | vector of radius.  Will create multiple spheres if len(radius) > 1 | *required*
`centers` |  | a vector of sphere centers of the form [px, py, pz] or [[px1, py1, pz1], ..., [pxn, pyn, pzn]] | *required*

###### `normal_noise`

```python
normal_noise(mu, sigma)
```

produce a normal noise distribution for all all points in the brain mask

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mu` |  | average value of the gaussian signal (usually set to 0) | *required*
`sigma` |  | standard deviation | *required*

###### `sphere`

```python
sphere(r, p)
```

create a sphere of given radius at some point p in the brain mask

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`r` |  | radius of the sphere | *required*
`p` |  | point (in coordinates of the brain mask) of the center of the sphere | *required*

###### `to_nifti`

```python
to_nifti(m)
```

convert a numpy matrix to the nifti format and assign to it the brain_mask's affine matrix

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`m` |  | the 3D numpy matrix we wish to convert to .nii | *required*



### Modules

#### `adjacency`

This data class is for working with similarity/dissimilarity matrices

**Modules:**

Name | Description
---- | -----------
[`io`](#io) | I/O functions for Adjacency objects.
[`modeling`](#modeling) | Standalone modeling/inference functions for Adjacency matrices.
[`plotting`](#plotting) | Plotting functions for Adjacency matrices.
[`spatial`](#spatial) | Spatial-scale provenance for stacked Adjacency matrices.
[`stats`](#stats) | Standalone statistical functions for Adjacency matrices.
[`utils`](#utils) | Shared helpers for Adjacency submodules.

**Classes:**

Name | Description
---- | -----------
[`Adjacency`](#Adjacency) | Adjacency is a class to represent Adjacency matrices as a vector rather

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`MAX_INT`](#MAX_INT) |  | 
[`nx`](#nx) |  | 

##### Methods

##### Modules

###### `io`

I/O functions for Adjacency objects.

**Methods:**

Name | Description
---- | -----------
[`to_graph`](#to_graph) | Convert Adjacency into networkx graph.
[`write`](#write) | Write out Adjacency object to csv file.



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

###### `modeling`

Standalone modeling/inference functions for Adjacency matrices.

Each function takes an Adjacency instance as its first argument (`adj`).

**Methods:**

Name | Description
---- | -----------
[`bootstrap`](#bootstrap) | Bootstrap statistics using efficient online algorithms.
[`convert_bootstrap_results_to_adjacency`](#convert_bootstrap_results_to_adjacency) | Convert bootstrap results dictionary to Adjacency format.
[`generate_permutations`](#generate_permutations) | Generate n_permute permutated versions of Adjacency in a lazy fashion. Useful for iterating against.
[`regress`](#regress) | Run a regression on an adjacency instance.
[`social_relations_model`](#social_relations_model) | Estimate the social relations model from a matrix for a round-robin design.



####### Functions##

###### `bootstrap`

```python
bootstrap(adj, stat, n_samples = 5000, save_boots = False, percentiles = (2.5, 97.5), n_jobs = -1, random_state = None)
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

Generate n_permute permutated versions of Adjacency in a lazy fashion. Useful for iterating against.

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

###### `plotting`

Plotting functions for Adjacency matrices.

**Methods:**

Name | Description
---- | -----------
[`plot_adjacency`](#plot_adjacency) | Create Heatmap of Adjacency Matrix.
[`plot_mds`](#plot_mds) | Plot Multidimensional Scaling.



####### Functions##

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
plot_mds(adj, n_components = 2, metric = True, labels = None, labels_color = None, cmap = None, view = (30, 20), figsize = None, ax = None, n_jobs = -1, *args, **kwargs)
```

Plot Multidimensional Scaling.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency object to plot (must be a distance matrix). | *required*
`n_components` | <code>[int](#int)</code> | Number of dimensions to project (can be 2 or 3). | <code>2</code>
`metric` | <code>[bool](#bool)</code> | Perform metric or non-metric dimensional scaling. Default True. | <code>True</code>
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

###### `spatial`

Spatial-scale provenance for stacked Adjacency matrices.

When a stack of Adjacency matrices comes from a per-parcel or per-searchlight
operation on a BrainData, attaching a :class:`SpatialScale` records the atlas,
the parcel labels in stack order, and the source mask — enough to project
per-matrix reductions back to a voxel-space :class:`BrainData` via
``Adjacency.to_brain()``.

See :class:`Adjacency` for the optional ``spatial_scale`` attribute, and
:meth:`BrainData.distance` (with ``spatial_scale='roi'|'searchlight'``) for
the canonical producer.

**Classes:**

Name | Description
---- | -----------
[`SpatialScale`](#SpatialScale) | Provenance for a stacked Adjacency that came from a per-parcel or



####### Classes##

###### `SpatialScale`

```python
SpatialScale(atlas: BrainData, roi_labels: np.ndarray, source_mask: Nifti1Image, kind: Literal['roi', 'searchlight'] = 'roi') -> None
```

Provenance for a stacked Adjacency that came from a per-parcel or
per-searchlight operation on a :class:`BrainData`.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`atlas`](#atlas) | <code>[BrainData](#nltools.data.BrainData)</code> | Labeled volume indicating parcel membership (or searchlight centers). One matrix in the stack per unique label.
[`roi_labels`](#roi_labels) | <code>[ndarray](#numpy.ndarray)</code> | Integer atlas IDs in stack order. ``len(roi_labels)`` must equal the number of matrices in the stack.
[`source_mask`](#source_mask) | <code>[Nifti1Image](#nibabel.Nifti1Image)</code> | The brain mask the atlas/values live in. Used as the target space for back-projection in ``Adjacency.to_brain()``.
[`kind`](#kind) | <code>[Literal](#typing.Literal)['roi', 'searchlight']</code> | Which spatial scale produced this stack — ``'roi'`` or ``'searchlight'``.



######### Attributes####

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

###### `stats`

Standalone statistical functions for Adjacency matrices.

Each function takes an Adjacency instance as its first argument (`adj`).

**Methods:**

Name | Description
---- | -----------
[`cluster_summary`](#cluster_summary) | This function provides summaries of clusters within Adjacency matrices.
[`plot_label_distance`](#plot_label_distance) | Create a violin plot indicating within and between label distance
[`plot_silhouette`](#plot_silhouette) | Create a silhouette plot.
[`r_to_z`](#r_to_z) | Apply Fisher's r to z transformation to each element of the data object.
[`similarity`](#similarity) | Calculate similarity between two Adjacency matrices. Default is to use spearman
[`stats_label_distance`](#stats_label_distance) | Calculate permutation tests on within and between label distance.
[`threshold`](#threshold) | Threshold Adjacency instance. Provide upper and lower values or
[`ttest`](#ttest) | Calculate ttest across samples.
[`z_to_r`](#z_to_r) | Convert z score back into r value for each element of data object.



####### Functions##

###### `cluster_summary`

```python
cluster_summary(adj, clusters = None, metric = 'mean', summary = 'within')
```

This function provides summaries of clusters within Adjacency matrices.

It can compute mean/median of within and between cluster values. Requires a
list of cluster ids indicating the row/column of each cluster.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency instance | *required*
`clusters` |  | (list) list of cluster labels | <code>None</code>
`metric` |  | (str) method to summarize mean or median. If 'None" then return all r values | <code>'mean'</code>
`summary` |  | (str) summarize within cluster or between clusters | <code>'within'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | (dict) within cluster means

######## `plot_label_distance`

```python
plot_label_distance(adj, labels = None, ax = None)
```

Create a violin plot indicating within and between label distance

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
plot_silhouette(adj, labels = None, ax = None, permutation_test = True, n_permute = 5000, **kwargs)
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
similarity(adj, data, plot = False, permutation_method = '2d', n_permute = 5000, metric = 'spearman', include_diag = False, nan_policy = 'omit', tail = 2, return_null = False, n_jobs = -1, random_state = None, *, project: bool = False)
```

Calculate similarity between two Adjacency matrices. Default is to use spearman
correlation and permutation test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency instance. | *required*
`data` | <code>[Adjacency](#nltools.data.adjacency.Adjacency) or [array](#array)</code> | Adjacency data, or 1-d array same size as adj.data. | *required*
`plot` | <code>[bool](#bool)</code> | If True, plot stacked adjacency matrices. Default False. | <code>False</code>
`permutation_method` | <code>[str](#str)</code> | '1d', '2d', or None. | <code>'2d'</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations. Default 5000. | <code>5000</code>
`metric` | <code>[str](#str)</code> | 'spearman', 'pearson', or 'kendall'. | <code>'spearman'</code>
`include_diag` | <code>[bool](#bool)</code> | Only applies to 'directed' Adjacency types using permutation_method=None or permutation_method='1d'. Default False (self-similarity is uninformative). Symmetric matrices never store the diagonal, so this flag is a no-op for them. | <code>False</code>
`nan_policy` | <code>[str](#str)</code> | How to handle NaN values. Options: - 'omit': Remove NaN values pairwise before computing correlation (default) - 'propagate': Allow NaN to propagate through calculations - 'raise': Raise an error if NaN values are present | <code>'omit'</code>
`tail` | <code>[int](#int)</code> | Tail of the test (1 or 2). Default 2. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, also return the null distribution. Default False. | <code>False</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. -1 means all cores. Default -1. | <code>-1</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | dict or list: Correlation result dict with keys 'r' and 'p', or a list of such dicts when adj contains multiple matrices.

######## `stats_label_distance`

```python
stats_label_distance(adj, labels = None, n_permute = 5000, n_jobs = -1)
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
threshold(adj, upper = None, lower = None, binarize = False)
```

Threshold Adjacency instance. Provide upper and lower values or
   percentages to perform two-sided thresholding. Binarize will return
   a mask image respecting thresholds if provided, otherwise respecting
   every non-zero value.

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
ttest(adj, permutation = False, n_permute = 5000, tail = 2, return_null = False, n_jobs = -1, random_state = None)
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

###### `utils`

Shared helpers for Adjacency submodules.

These are internal utilities used by the facade and submodules — not part of the
public API.

**Methods:**

Name | Description
---- | -----------
[`apply_stat`](#apply_stat) | Apply a statistical function along an axis.
[`import_single_data`](#import_single_data) | Import and validate a single adjacency data matrix.
[`perform_arithmetic`](#perform_arithmetic) | Perform arithmetic operation with validation.
[`test_is_single_matrix`](#test_is_single_matrix) | Check whether data represents a single matrix (1-D vector).



####### Functions##

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

#### `atlases`

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



##### Modules

###### `labeling`

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



####### Attributes##

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
`atlas` | <code>[str](#str) \| [Sequence](#collections.abc.Sequence)[[str](#str)]</code> | Atlas name or list of names from :func:`list_atlases`. One column is added to the output per atlas. | <code>'harvard_oxford'</code>
`prob_threshold` | <code>[float](#float)</code> | For probabilistic atlases only — drop regions with probability (in percent units) below this threshold. | <code>5.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#polars.DataFrame)</code> | Polars DataFrame with columns ``x``, ``y``, ``z`` plus one
<code>[DataFrame](#polars.DataFrame)</code> | column per atlas. All atlas columns are ``Utf8``.

###### `loading`

Lazy loading of atlas NIfTI + label CSV files from the HF dataset.

**Classes:**

Name | Description
---- | -----------
[`Atlas`](#Atlas) | A loaded atlas — image, labels, and metadata.

**Methods:**

Name | Description
---- | -----------
[`load_atlas`](#load_atlas) | Lazy-load an atlas by registry name.



####### Attributes

####### Classes##

###### `Atlas`

```python
Atlas(name: str, image: nb.Nifti1Image, labels: pl.DataFrame, kind: AtlasKind, citation: str) -> None
```

A loaded atlas — image, labels, and metadata.

Constructed by :func:`load_atlas`; users normally don't instantiate
directly.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`name`](#name) | <code>[str](#str)</code> | Registry key (e.g. ``"harvard_oxford"``).
[`image`](#image) | <code>[Nifti1Image](#nibabel.Nifti1Image)</code> | NIfTI volume. 3D for deterministic atlases, 4D for probabilistic ones (last axis indexes regions).
[`labels`](#labels) | <code>[DataFrame](#polars.DataFrame)</code> | Two-column ``index, name`` table. For deterministic atlases ``index`` is the integer voxel value; for probabilistic atlases ``index`` is the region index along the 4th dim of ``image``.
[`kind`](#kind) | <code>[AtlasKind](#nltools.data.atlases.registry.AtlasKind)</code> | ``"deterministic"`` or ``"probabilistic"``.
[`citation`](#citation) | <code>[str](#str)</code> | Short citation for the original atlas.



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
`name` | <code>[str](#str)</code> | Atlas key from :func:`list_atlases`. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`An` | <code>[Atlas](#nltools.data.atlases.loading.Atlas)</code> | class:`Atlas` with image, labels, and metadata loaded.

###### `registry`

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
[`kind`](#kind) | <code>[AtlasKind](#nltools.data.atlases.registry.AtlasKind)</code> | ``"deterministic"`` (3D integer-labeled) or ``"probabilistic"`` (4D, last axis indexes regions).
[`citation`](#citation) | <code>[str](#str)</code> | Short citation string for the original atlas.



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
<code>[list](#list)[[str](#str)]</code> | func:`nltools.data.atlases.load_atlas`.

###### `reporting`

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



####### Attributes

####### Classes##

###### `ClusterReport`

```python
ClusterReport(peaks: pl.DataFrame, clusters: pl.DataFrame, stat_img: BrainData) -> None
```

Result of :meth:`BrainData.cluster_report`.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`peaks`](#peaks) | <code>[DataFrame](#polars.DataFrame)</code> | Polars DataFrame, one row per peak (incl. sub-peaks). Columns ``cluster_id``, ``x``, ``y``, ``z`` (mm), ``peak_stat``, ``volume_mm3``, ``n_voxels``, then one Utf8 column per atlas.
[`clusters`](#clusters) | <code>[DataFrame](#polars.DataFrame)</code> | Polars DataFrame, one row per cluster. Columns ``cluster_id``, ``peak_x``, ``peak_y``, ``peak_z``, ``mean_stat``, ``volume_mm3``, ``n_voxels``, then one Utf8 column per atlas (mass-weighted top regions).
[`stat_img`](#stat_img) | <code>[BrainData](#nltools.data.BrainData)</code> | BrainData with the thresholded stat map (sub-cluster voxels and clusters smaller than ``cluster_threshold`` zeroed).

**Methods:**

Name | Description
---- | -----------
[`plot`](#plot) | Render an overview glass brain + one slice figure per cluster.
[`to_csv`](#to_csv) | Write ``peaks.csv`` and ``clusters.csv`` into ``output_dir``.



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

#### `braindata`

NeuroLearn Brain Data
=====================

Classes to represent brain image data.

**Modules:**

Name | Description
---- | -----------
[`analysis`](#analysis) | BrainData analysis functions.
[`bootstrap`](#bootstrap) | Bootstrap functions extracted from BrainData methods.
[`cache`](#cache) | Disk-based caching infrastructure for expensive computations.
[`io`](#io) | BrainData I/O and loading functions.
[`modeling`](#modeling) | BrainData modeling functions.
[`neighborhoods`](#neighborhoods) | Spatial neighborhood computation for neuroimaging analyses.
[`plotting`](#plotting) | BrainData plotting functions.
[`prediction`](#prediction) | BrainData prediction — timeseries (encoding) and MVPA (decoding).
[`utils`](#utils) | Shared helpers for BrainData submodules.
[`validation`](#validation) | Validation utilities for BrainData class.
[`widgets`](#widgets) | Anywidget-based interactive viewer for BrainData.

**Classes:**

Name | Description
---- | -----------
[`BrainData`](#BrainData) | BrainData is a class to represent neuroimaging data in python as a vector

##### Methods

##### Modules

###### `analysis`

BrainData analysis functions.

Standalone functions extracted from BrainData class methods for similarity,
distance, masking, ROI extraction, ICC, filtering, thresholding, decomposition,
alignment, smoothing, and other analytical operations.

**Methods:**

Name | Description
---- | -----------
[`align`](#align) | Align BrainData instance to target object using functional alignment
[`align_per_roi`](#align_per_roi) | Per-parcel functional alignment + voxel-space reassembly.
[`apply_mask`](#apply_mask) | Mask BrainData instance using nilearn functionality.
[`check_masks`](#check_masks) | Check to make sure masks are the same for each dataset and if not create a union mask
[`decompose`](#decompose) | Decompose BrainData object
[`detrend_data`](#detrend_data) | Remove linear trend from each voxel
[`distance`](#distance) | Calculate distance between images within a BrainData() instance.
[`extract_roi`](#extract_roi) | Extract activity from mask or ROI atlas using NiftiLabelsMasker.
[`filter_data`](#filter_data) | Apply butterworth filter to data. Wraps nilearn.signal.clean.
[`find_spikes_data`](#find_spikes_data) | Identify spikes from Time Series Data — see :func:`nltools.stats.find_spikes`.
[`icc`](#icc) | Calculate voxel-wise intraclass correlation coefficient for data within
[`multivariate_similarity`](#multivariate_similarity) | Predict spatial distribution of BrainData() instance from linear
[`r_to_z`](#r_to_z) | Apply Fisher's r to z transformation to each element of the data
[`reduce_per_roi`](#reduce_per_roi) | Apply ``reducer`` (e.g. ``np.mean``) within each parcel and paint the
[`regions`](#regions) | Extract brain connected regions into separate regions.
[`scale_data`](#scale_data) | Scale data via mean scaling.
[`similarity`](#similarity) | Calculate similarity of BrainData() instance with single
[`smooth`](#smooth) | Apply spatial smoothing using nilearn smooth_img()
[`standardize`](#standardize) | Standardize BrainData() instance.
[`temporal_resample`](#temporal_resample) | Resample BrainData timeseries to a new target frequency or number of samples
[`threshold_data`](#threshold_data) | Threshold BrainData instance with optional cluster filtering.
[`transform_pairwise_data`](#transform_pairwise_data) | Transform BrainData into pairwise comparisons.
[`z_to_r`](#z_to_r) | Convert z score back into r value for each element of data object.



####### Functions##

###### `align`

```python
align(bd, target, method = 'procrustes', axis = 0)
```

Align BrainData instance to target object using functional alignment

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
:class:`BrainData` of the same shape as the input (each voxel filled
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

Check to make sure masks are the same for each dataset and if not create a union mask

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

Decompose BrainData object

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

Remove linear trend from each voxel

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
`spatial_scale` | <code>[str](#str)</code> | ``'whole_brain'`` (default), ``'roi'``, or ``'searchlight'``. See :meth:`BrainData.distance`. | <code>'whole_brain'</code>
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
filter_data(bd, sampling_freq = None, high_pass = None, low_pass = None, **kwargs)
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

Identify spikes from Time Series Data — see :func:`nltools.stats.find_spikes`.

######## `icc`

```python
icc(bd, n_subjects, n_sessions, method = 'icc2', parallel = None, n_jobs = -1, max_gpu_memory_gb = 4.0)
```

Calculate voxel-wise intraclass correlation coefficient for data within
    BrainData class.

Computes ICC for each voxel independently, making it highly parallelizable.
Supports GPU acceleration for large voxel counts.

ICC Formulas are based on:
Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
assessing rater reliability. Psychological bulletin, 86(2), 420.

icc1:  x_ij = mu + beta_j + w_ij
icc2/3:  x_ij = mu + alpha_i + beta_j + (ab)_ij + epsilon_ij

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`n_subjects` |  | Number of subjects in the data | *required*
`n_sessions` |  | Number of sessions per subject | *required*
`method` |  | Type of ICC to calculate     - 'icc1': One-way random effects (subjects random, sessions treated as interchangeable)     - 'icc2': Two-way random effects (subjects and sessions random) (default)     - 'icc3': Two-way mixed effects (subjects random, sessions fixed) | <code>'icc2'</code>
`parallel` |  | Parallelization method     - None: Single-threaded vectorized NumPy (default, memory efficient)     - 'cpu': CPU parallelization via joblib (for medium-sized problems, 1K-10K voxels)     - 'gpu': GPU acceleration via PyTorch (recommended for large voxel counts >10K, 10-50x speedup) | <code>None</code>
`n_jobs` |  | Number of CPU cores (-1 = all cores). Only used when parallel='cpu' | <code>-1</code>
`max_gpu_memory_gb` |  | GPU memory budget in GB. Only used when parallel='gpu' | <code>4.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with ICC map (shape: (1, n_voxels))

**Examples:**

```pycon
>>> # Typical test-retest reliability analysis
>>> data = BrainData(...)  # Shape: (60, 238955) = 20 subjects x 3 sessions
>>> icc_map = data.icc(n_subjects=20, n_sessions=3, method='icc2')
>>> icc_map.shape
(1, 238955)
>>> # Visualize ICC map
>>> icc_map.plot()
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

Data must be organized such that n_images = n_subjects * n_sessions.
Images should be ordered as: [subject1_session1, subject1_session2, ...,
subject2_session1, ...]

</details>

######## `multivariate_similarity`

```python
multivariate_similarity(bd, images, method = 'ols')
```

Predict spatial distribution of BrainData() instance from linear
combination of other BrainData() instances or Nibabel images

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

Apply Fisher's r to z transformation to each element of the data
object.

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

Apply ``reducer`` (e.g. ``np.mean``) within each parcel and paint the
result back to voxel space — i.e. spatial smoothing via parcellation.

For each image ``i`` and each parcel ``p``, computes
``reducer(bd.data[i, voxels-in-p])`` and assigns that scalar to every
voxel in parcel ``p`` for image ``i``. Voxels outside any parcel get
NaN. Output is a :class:`BrainData` of the same shape as the input.

Used by ``BrainData.{mean,std,median}(spatial_scale='roi')``.

######## `regions`

```python
regions(bd, min_region_size = 1350, method = 'local_regions', smoothing_fwhm = 6, is_mask = False)
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
similarity(bd, image, method = 'correlation')
```

Calculate similarity of BrainData() instance with single
BrainData or Nibabel image

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`image` |  | (BrainData, nifti)  image to evaluate similarity | *required*
`method` |  | (str) Type of similarity     ['correlation', 'pearson', 'rank_correlation', 'spearman', 'dot_product', 'cosine'] | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
 | np.ndarray: Similarity values.

######## `smooth`

```python
smooth(bd, fwhm)
```

Apply spatial smoothing using nilearn smooth_img()

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
standardize(bd, axis = 0, method = 'center', verbose = True)
```

Standardize BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`axis` |  | 0 for observations 1 for voxels (default: 0) | <code>0</code>
`method` |  | ['center','zscore'] (default: 'center') | <code>'center'</code>
`verbose` |  | If False, suppress sklearn numerical warnings that occur when voxels have near-zero variance. (default: True) | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Standardized BrainData instance.

######## `temporal_resample`

```python
temporal_resample(bd, sampling_freq = None, target = None, target_type = 'hz')
```

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
threshold_data(bd, upper = None, lower = None, binarize = False, coerce_nan = True, cluster_threshold = 0)
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
[`bootstrap`](#bootstrap) | Bootstrap statistics using efficient online algorithms.
[`convert_bootstrap_results_to_brain_data`](#convert_bootstrap_results_to_brain_data) | Convert bootstrap results dictionary to BrainData format.



####### Functions##

###### `bootstrap`

```python
bootstrap(bd, stat, n_samples = 5000, save_boots = False, percentiles = (2.5, 97.5), X_test = None, backend = None, max_gpu_memory_gb = 4.0, n_jobs = -1, random_state = None)
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
[`CacheManager`](#CacheManager) | Manages disk-based caching for expensive computations.

**Methods:**

Name | Description
---- | -----------
[`clear_cache`](#clear_cache) | Clear the nltools cache.
[`get_cache_dir`](#get_cache_dir) | Get the nltools cache directory.
[`hash_mask`](#hash_mask) | Compute a stable hash for a NIfTI mask image.



####### Classes##

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
[`clear`](#clear) | Clear all cached files in this category.
[`delete`](#delete) | Delete a cached file.
[`exists`](#exists) | Check if a cache key exists.
[`get_path`](#get_path) | Get the file path for a cache key.
[`list_keys`](#list_keys) | List all cached keys in this category.
[`load`](#load) | Load cached data.
[`save`](#save) | Save arrays to cache.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`cache_dir`](#cache_dir) |  | 
[`category`](#category) |  | 



######### Attributes####

###### `cache_dir`

```python
cache_dir = get_cache_dir() / category
```

########## `category`

```python
category = category
```



######### Functions####

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
'a1b2c3d4e5f6g7h8'

</details>

###### `io`

BrainData I/O and loading functions.

Standalone functions extracted from BrainData class methods for mask initialization,
data loading (from files, lists, URLs, HDF5, other BrainData objects), resampling,
writing, and uploading.

**Methods:**

Name | Description
---- | -----------
[`check_space_match`](#check_space_match) | Check if data and mask are in same space.
[`detect_and_update_mask`](#detect_and_update_mask) | Detect best matching template from data and update mask if mask was None.
[`detect_space`](#detect_space) | Detect if mask is in MNI space or native space.
[`get_interpolation`](#get_interpolation) | Get the interpolation method to use for a given image.
[`initialize_mask`](#initialize_mask) | Initialize the mask image.
[`load_from_brain_data`](#load_from_brain_data) | Load data from another BrainData object.
[`load_from_file`](#load_from_file) | Load data from file path or nibabel object.
[`load_from_h5`](#load_from_h5) | Load data from HDF5 file.
[`load_from_list`](#load_from_list) | Load data from a list of BrainData objects or file paths.
[`load_from_url`](#load_from_url) | Load data from URL.
[`resample_to`](#resample_to) | Resample BrainData to match target image or resolution.
[`to_nifti`](#to_nifti) | Convert BrainData instance to a nibabel NIfTI image.
[`upload_neurovault`](#upload_neurovault) | Upload data to NeuroVault.
[`warn_if_resampling`](#warn_if_resampling) | Warn about resampling if verbose=True and resample=True.
[`write_brain_data`](#write_brain_data) | Write out BrainData object to Nifti or HDF5 File.



####### Functions##

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
detect_space(bd, mask)
```

Detect if mask is in MNI space or native space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance (unused, kept for API consistency). | *required*
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

######## `resample_to`

```python
resample_to(bd, img = None, resolution = None, interpolation = None)
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
upload_neurovault(bd, access_token = None, collection_name = None, collection_id = None, img_type = None, img_modality = None, **kwargs)
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
[`compute_contrasts`](#compute_contrasts) | Compute contrasts from a fitted GLM.
[`compute_ridge_cv`](#compute_ridge_cv) | Held-out CV scores under a fixed Ridge α.
[`fit`](#fit) | Fit a model to brain imaging data.
[`fit_glm`](#fit_glm) | Fit GLM model and extract results (same logic as current regress()).
[`fit_ridge`](#fit_ridge) | Fit Ridge model and extract results.
[`parse_contrast_string`](#parse_contrast_string) | Parse a contrast string into a numeric contrast vector.
[`regress`](#regress) | Deprecated: Use fit(model='glm', X=design_matrix) instead.
[`to_fit_dataclass`](#to_fit_dataclass) | Convert BrainData fit results to Fit dataclass.
[`ttest`](#ttest) | One-sample voxelwise t-test across images (axis 0).
[`ttest2`](#ttest2) | Two-sample voxelwise t-test between two BrainData stacks.



####### Functions##

###### `compute_contrasts`

```python
compute_contrasts(bd, contrasts, contrast_type = 't')
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
`contrast_type` | <code>[str](#str)</code> | What to return per contrast. One of:<br>- ``"t"`` (default): t-statistic map (for thresholding /   single-subject inference) - ``"z"``: z-score map - ``"p"``: p-value map - ``"beta"`` / ``"effect_size"``: effect-size (β) map — use this   when feeding into a second-level (group) analysis - ``"all"``: a bundle dict ``{"beta", "t", "z", "p", "se"}``   of BrainData maps for this one contrast. One fit, one call,   every view — effect size *and* inferential maps together so   group-level code never has to recompute beta separately. | <code>'t'</code>

**Returns:**

Type | Description
---- | -----------
 | Depends on inputs:<br>- single contrast (str or array) + scalar ``contrast_type``:   a single BrainData. - single contrast + ``contrast_type="all"``: a flat dict of five   BrainData keyed by ``"beta"``/``"t"``/``"z"``/``"p"``/``"se"``. - dict of contrasts + scalar ``contrast_type``: a dict   ``{name: BrainData}``. - dict of contrasts + ``contrast_type="all"``: a nested dict   ``{name: {"beta", "t", "z", "p", "se"}}``.

**Examples:**

```pycon
>>> data.fit(model="glm", X=dm)
>>> # Single-subject t-map, ready to threshold
>>> tmap = data.compute_contrasts("conditionA - conditionB")
>>> # Effect-size map for use as input to a group-level analysis
>>> beta = data.compute_contrasts(
...     "conditionA - conditionB", contrast_type="beta"
... )
>>> # Everything at once: threshold on res["t"], feed group on res["beta"]
>>> res = data.compute_contrasts(
...     "conditionA - conditionB", contrast_type="all"
... )
>>> res["t"].plot(threshold=3.09)
>>> group_effects.append(res["beta"])
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- String contrasts support coefficients: ``"2*A - B"`` or ``"0.5*A + 0.5*B"``.
- Column names must match design matrix columns exactly (case-sensitive).
- For group analysis, stack per-subject effect-size maps
  (``contrast_type="beta"`` or ``res["beta"]`` from ``contrast_type="all"``)
  and run a second-level test (e.g. ``BrainData.ttest``). Mixing first-level
  t-maps into a group one-sample test conflates effect magnitude with precision.

</details>

######## `compute_ridge_cv`

```python
compute_ridge_cv(bd, X, cv, alpha = None, backend = 'auto', **kwargs)
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
`**kwargs` |  | Additional kwargs (forward-compatibility). | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | ``{"scores", "mean_score", "predictions", "folds"}``.

######## `fit`

```python
fit(bd, model = 'glm', *, X = None, cv = None, local_alpha = True, fit_intercept = False, inplace = True, progress_bar = None, scale = True, scale_value = 100.0, design_clean = True, design_clean_thresh = 0.95, design_clean_exclude_confounds = False, design_clean_fill_na = 0, **kwargs)
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
`inplace` | <code>bool, default=True</code> | If True, mutate bd and return bd (backward compatible). If False, return Fit dataclass with results (bd unchanged). | <code>True</code>
`progress_bar` | <code>[bool](#bool)</code> | Display progress bar during fitting. - If None: Uses bd.verbose (default) - If True: Shows progress bar for long-running operations - If False: No progress bar | <code>None</code>
`scale` | <code>bool, default=True</code> | Apply grand-mean scaling before fitting. Calls bd.scale(scale_value) which divides all values by the global mean and multiplies by scale_value. This puts data in percent signal change units, which is standard for fMRI analysis. | <code>True</code>
`scale_value` | <code>float, default=100.0</code> | Target value for mean after scaling. Only used if scale=True. | <code>100.0</code>
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
[`The`](#The) | <code>following are set on bd when ``inplace=True``</code> | 
[```model_```](#nltools.data.braindata.modeling.fit.``model_``) | <code>[BaseModel](#BaseModel)</code> | Fitted model instance (Ridge, Glm, etc.)
[```X_```](#nltools.data.braindata.modeling.fit.``X_``) | <code>[ndarray](#ndarray)</code> | Training data X, stored for predict() default
[```cv_results_```](#nltools.data.braindata.modeling.fit.``cv_results_``) | <code>[dict](#dict)</code> | Cross-validation results dict with keys 'scores',
[`glm_betas`](#glm_betas) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Beta coefficients (for model='glm')
[`glm_t`](#glm_t) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | T-statistics (for model='glm')
[`glm_p`](#glm_p) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | P-values (for model='glm')
[`glm_se`](#glm_se) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Standard errors (for model='glm')
[`glm_residual`](#glm_residual) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Residuals (for model='glm')
[`glm_predicted`](#glm_predicted) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Fitted values (for model='glm')
[`glm_r2`](#glm_r2) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | R-squared values (for model='glm')
[`ridge_weights`](#ridge_weights) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Model coefficients (for model='ridge')
[`ridge_fitted_values`](#ridge_fitted_values) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Fitted values (for model='ridge')
[`ridge_scores`](#ridge_scores) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | R-squared scores (for model='ridge')

**Examples:**

```pycon
>>> # Old behavior (backward compatible): mutate self
>>> brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
>>> print(f"CV R2: {brain_data.cv_results_['mean_score'].mean():.3f}")
>>> weights = brain_data.ridge_weights  # Access as attribute
>>>
>>> # New behavior: return Fit dataclass (self unchanged)
>>> fit = brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features, inplace=False)
>>> assert isinstance(fit, Fit)
>>> assert 'weights' in fit.available()
>>> assert not hasattr(brain_data, 'ridge_weights')  # brain_data unchanged
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

######## `regress`

```python
regress(bd, design_matrix = None, method = 'ols', mode = None)
```

Deprecated: Use fit(model='glm', X=design_matrix) instead.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`design_matrix` |  | Design matrix (unused, raises error). | <code>None</code>
`method` |  | Noise model (unused, raises error). | <code>'ols'</code>
`mode` |  | Mode (unused, raises error). | <code>None</code>

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
ttest(bd, popmean = 0.0, permutation = False, n_permute = 5000, tail = 2, return_null = False, n_jobs = -1, random_state = None)
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
[`SphereNeighborhoods`](#SphereNeighborhoods) | Precomputed sphere neighborhoods for a brain mask.

**Methods:**

Name | Description
---- | -----------
[`compute_searchlight_neighborhoods`](#compute_searchlight_neighborhoods) | Compute sphere neighborhoods for all voxels in a brain mask.



####### Classes##

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
[`adjacency`](#adjacency) | <code>[csr_matrix](#scipy.sparse.csr_matrix)</code> | Sparse CSR matrix (n_voxels, n_voxels) where adjacency[i, j] is True if voxel j is within radius of voxel i
[`mask_hash`](#mask_hash) | <code>[str](#str)</code> | Hash of the source mask for validation
[`radius_mm`](#radius_mm) | <code>[float](#float)</code> | Radius in millimeters
[`n_voxels`](#n_voxels) | <code>[int](#int)</code> | Number of voxels in the mask

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
[`get_neighborhood_size`](#get_neighborhood_size) | Get the number of voxels in a neighborhood.
[`get_neighbors`](#get_neighbors) | Get indices of all voxels in the neighborhood of a given voxel.
[`iter_neighborhoods`](#iter_neighborhoods) | Iterate over all neighborhoods.



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
[`auto_select_colormap`](#auto_select_colormap) | Auto-select colormap based on data characteristics.
[`plot_brain`](#plot_brain) | Plot BrainData instance using nilearn visualization or matplotlib.
[`plot_flatmap_brain`](#plot_flatmap_brain) | Plot brain data on cortical flatmap.
[`prepare_save_paths`](#prepare_save_paths) | Prepare save paths for multiple plot outputs.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`DEFAULT_SLICE_CUT_COORDS`](#DEFAULT_SLICE_CUT_COORDS) |  | 



####### Attributes##

###### `DEFAULT_SLICE_CUT_COORDS`

```python
DEFAULT_SLICE_CUT_COORDS = {'x': list(range(-50, 51, 8)), 'y': list(range(-80, 50, 10)), 'z': list(range(-40, 71, 9))}
```



####### Functions##

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
plot_brain(bd, method = 'glass', upper = None, lower = None, threshold = None, view = 'z', cut_coords = None, cmap = None, bg_img = None, ax = None, figsize = (8, 6), title = None, colorbar = True, save = None, stat = 'mean', limit = 3, **kwargs)
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
plot_flatmap_brain(bd, threshold = None, cmap = 'RdBu_r', vmax = None, vmin = None, template = 'fsaverage5', with_curvature = True, curvature_contrast = 0.5, curvature_brightness = 0.5, transparency = 'auto', colorbar = True, colorbar_orientation = 'horizontal', figsize = (12, 6), title = None, radius_mm = 3.0, interpolation = 'linear', axes = None, save = None)
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

###### `prediction`

BrainData prediction — timeseries (encoding) and MVPA (decoding).

Single entry point: :func:`predict`. Returns :class:`nltools.data.fitresults.Predict`
with fields populated based on dispatch. Mirrors :meth:`BrainData.fit` /
:class:`Fit` patterns: frozen result dataclass, ``inplace=True`` mutates
self with attributes, ``inplace=False`` returns the dataclass.

**Methods:**

Name | Description
---- | -----------
[`build_pipeline`](#build_pipeline) | Build a per-fold sklearn pipeline: optional StandardScaler → optional
[`predict`](#predict) | Implementation of :meth:`BrainData.predict`. See class docstring for
[`predict_mvpa`](#predict_mvpa) | Cross-validated decoding. Returns Predict (or self if inplace=True).
[`predict_timeseries`](#predict_timeseries) | Predict voxel timeseries from a fitted encoding model.
[`resolve_model`](#resolve_model) | Resolve a string shortcut or pass through a sklearn estimator.
[`resolve_scoring`](#resolve_scoring) | Resolve scoring='auto' to 'accuracy' (classifier) or 'r2' (regressor).

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`VALID_SPATIAL_SCALES`](#VALID_SPATIAL_SCALES) |  | 



####### Attributes##

###### `VALID_SPATIAL_SCALES`

```python
VALID_SPATIAL_SCALES = {'whole_brain', 'searchlight', 'roi'}
```



####### Classes

####### Functions##

###### `build_pipeline`

```python
build_pipeline(model, standardize: bool, reduce: str | None, n_components: str | None)
```

Build a per-fold sklearn pipeline: optional StandardScaler → optional
PCA → model. If only the model is needed, returns the model itself.

######## `predict`

```python
predict(bd, *, y = None, X = None, spatial_scale: str = 'whole_brain', model: Any = 'svm', cv: int = 5, standardize: bool = True, reduce: str | None = None, n_components: int | None = None, scoring: str = 'auto', groups: str = None, roi_mask: str = None, radius_mm: float = 10.0, inplace: bool = False, n_jobs: int = 1, progress_bar: bool = False)
```

Implementation of :meth:`BrainData.predict`. See class docstring for
full parameter documentation.

######## `predict_mvpa`

```python
predict_mvpa(bd, *, y, spatial_scale: str, model: Any, cv: Any, standardize: bool, reduce: str | None, n_components: int | None, scoring: str, groups: str, roi_mask: str, radius_mm: float, inplace: bool, n_jobs: int, progress_bar: bool) -> Predict | Any
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
[`apply_func`](#apply_func) | Apply a statistical function to BrainData's ``.data`` attribute.
[`check_brain_data`](#check_brain_data) | Return *data* as a BrainData, coercing Niimg-like inputs if needed.
[`check_brain_data_is_single`](#check_brain_data_is_single) | Logical test if BrainData instance is a single image.
[`perform_arithmetic`](#perform_arithmetic) | Perform an arithmetic operation with validation.
[`shallow_copy`](#shallow_copy) | Create a shallow copy of a BrainData for efficient method chaining.



####### Functions##

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
:meth:`BrainData.apply_mask`.  Otherwise *data* is passed through
:class:`BrainData`, which dispatches on type (file path, list of paths,
URL, h5, ``nib.Nifti1Image``).  Unsupported types raise ``TypeError`` from
:func:`validate_data_type`.

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

###### `validation`

Validation utilities for BrainData class.

This module contains helper functions for validating inputs, shapes, and
compatibility between BrainData objects and other data types.

**Methods:**

Name | Description
---- | -----------
[`validate_append_shapes`](#validate_append_shapes) | Validate shape compatibility for appending BrainData objects.
[`validate_arithmetic_operand`](#validate_arithmetic_operand) | Validate operand type for arithmetic operations.
[`validate_brain_data_shapes`](#validate_brain_data_shapes) | Validate shape compatibility between two BrainData objects.
[`validate_data_type`](#validate_data_type) | Validate input data type for BrainData initialization.
[`validate_frame`](#validate_frame) | Validate and process X or Y dataframes for BrainData.
[`validate_index_operations`](#validate_index_operations) | Validate indexing operations for BrainData.
[`validate_list_data`](#validate_list_data) | Validate that all items in a list are the same type.



####### Functions##

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

###### `widgets`

Anywidget-based interactive viewer for BrainData.

`BrainViewerWidget` wraps nilearn's HTML viewers (`view_img`,
`view_img_on_surf`) inside an iframe with a threshold panel (with
symmetric/independent and value/percentile toggles), plus a volume
slider for 4D BrainData. Renders inline in Jupyter, marimo, and
Jupyter Book v2 (mystmd) static-built sites via the standard widget
mimebundle.

**Classes:**

Name | Description
---- | -----------
[`BrainViewerWidget`](#BrainViewerWidget) | Interactive ortho/surface viewer with threshold panel and (for 4D) volume slider.



####### Classes##

###### `BrainViewerWidget`

```python
BrainViewerWidget(bd, *, view: str = 'ortho', mode: str = 'symmetric', units: str = 'value', lower: float | None = None, upper: float | None = None, threshold: float | None = None, **view_kwargs: float | None)
```

Bases: <code>[AnyWidget](#anywidget.AnyWidget)</code>

Interactive ortho/surface viewer with threshold panel and (for 4D) volume slider.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`data_max`](#data_max) |  | 
[`data_min`](#data_min) |  | 
[`has_volume_slider`](#has_volume_slider) |  | 
[`html`](#html) |  | 
[`lower`](#lower) |  | 
[`mode_pct`](#mode_pct) |  | 
[`mode_signed`](#mode_signed) |  | 
[`n_volumes`](#n_volumes) |  | 
[`pct_table_abs`](#pct_table_abs) |  | 
[`pct_table_neg`](#pct_table_neg) |  | 
[`pct_table_pos`](#pct_table_pos) |  | 
[`upper`](#upper) |  | 
[`vmax_abs`](#vmax_abs) |  | 
[`volume_idx`](#volume_idx) |  | 



######### Attributes####

###### `data_max`

```python
data_max = traitlets.Float(0.0).tag(sync=True)
```

########## `data_min`

```python
data_min = traitlets.Float(0.0).tag(sync=True)
```

########## `has_volume_slider`

```python
has_volume_slider = n > 1
```

########## `html`

```python
html = traitlets.Unicode('').tag(sync=True)
```

########## `lower`

```python
lower = float(lower) if lower is not None else 0.0
```

########## `mode_pct`

```python
mode_pct = units == 'percentile'
```

########## `mode_signed`

```python
mode_signed = mode == 'independent'
```

########## `n_volumes`

```python
n_volumes = n
```

########## `pct_table_abs`

```python
pct_table_abs = traitlets.List(trait=(traitlets.Float())).tag(sync=True)
```

########## `pct_table_neg`

```python
pct_table_neg = traitlets.List(trait=(traitlets.Float())).tag(sync=True)
```

########## `pct_table_pos`

```python
pct_table_pos = traitlets.List(trait=(traitlets.Float())).tag(sync=True)
```

########## `upper`

```python
upper = float(upper) if upper is not None else 0.0
```

########## `vmax_abs`

```python
vmax_abs = traitlets.Float(1.0).tag(sync=True)
```

########## `volume_idx`

```python
volume_idx = traitlets.Int(0).tag(sync=True)
```



####### Functions

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
[`core`](#core) | Module-level helpers for BrainCollection.
[`execution`](#execution) | Parallel execution machinery for BrainCollection.
[`inference`](#inference) | Group-level reductions and cross-subject ops for BrainCollection.
[`io`](#io) | IO and constructors for BrainCollection.
[`pipeline`](#pipeline) | Pipeline classes for BrainCollection.

**Classes:**

Name | Description
---- | -----------
[`BrainCollection`](#BrainCollection) | Parallel, lazy iterator of ``BrainData`` whose API mirrors ``BrainData``.
[`BrainCollectionPipeline`](#BrainCollectionPipeline) | Pipeline for BrainCollection with multi-subject CV support.
[`BrainCollectionWorkerError`](#BrainCollectionWorkerError) | Raised in the parent process when a worker fails inside ``_apply``.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`BUNDLE_SCHEMA_VERSION`](#BUNDLE_SCHEMA_VERSION) |  | 

##### Methods

##### Modules

###### `append`

Standalone functions for DesignMatrix concatenation operations.

These functions implement the append/concatenation logic extracted from
DesignMatrix methods, following the "functional core" pattern.

**Methods:**

Name | Description
---- | -----------
[`append`](#append) | Concatenate design matrices.
[`append_horizontal`](#append_horizontal) | Horizontal concatenation (axis=1) - add columns from other matrices.
[`append_vertical`](#append_vertical) | Vertical concatenation (axis=0) - stack rows, with optional confound separation.
[`append_vertical_with_separation`](#append_vertical_with_separation) | Vertical concatenation with automatic confound separation.
[`get_starting_run_idx`](#get_starting_run_idx) | Determine next run index for multi-run appending.
[`identify_columns_to_separate`](#identify_columns_to_separate) | Identify which columns need run-specific separation.
[`match_column_pattern`](#match_column_pattern) | Match columns against pattern with wildcard support.



####### Classes

####### Functions##

###### `append`

```python
append(dm: DesignMatrix, other: DesignMatrix, axis: int = 0, keep_separate: bool = True, unique_cols: list[str] | None = None, fill_na: int | float | None = 0, as_confounds: bool = False, verbose: bool = False) -> DesignMatrix
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
`verbose` | <code>[bool](#bool)</code> | Print messages about confound separation. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Concatenated design matrix.

######## `append_horizontal`

```python
append_horizontal(dm: DesignMatrix, to_append: list[DesignMatrix], fill_na: int | float | None, as_confounds: bool = False) -> DesignMatrix
```

Horizontal concatenation (axis=1) - add columns from other matrices.

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
append_vertical(dm: DesignMatrix, to_append: list[DesignMatrix], keep_separate: bool, unique_cols: list[str] | None, fill_na: int | float | None, verbose: bool) -> DesignMatrix
```

Vertical concatenation (axis=0) - stack rows, with optional confound separation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Base DesignMatrix instance. | *required*
`to_append` | <code>list of DesignMatrix</code> | Matrices to stack below dm. | *required*
`keep_separate` | <code>[bool](#bool)</code> | Whether to separate confound columns across runs. | *required*
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | *required*
`fill_na` | <code>int, float, or None</code> | Value to fill NaN/null entries with. Pass ``None`` to preserve nulls. | *required*
`verbose` | <code>[bool](#bool)</code> | Print messages about confound separation. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with rows from all matrices.

######## `append_vertical_with_separation`

```python
append_vertical_with_separation(dm: DesignMatrix, to_append: list[DesignMatrix], unique_cols: list[str] | None, fill_na: int | float | None, verbose: bool) -> DesignMatrix
```

Vertical concatenation with automatic confound separation.

Creates run-specific columns (e.g., 0_poly_0, 1_poly_0) that are
active only in their respective runs (sparse representation).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Base DesignMatrix instance. | *required*
`to_append` | <code>list of DesignMatrix</code> | Matrices to stack below dm. | *required*
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | *required*
`fill_na` | <code>int, float, or None</code> | Value to fill NaN/null entries with. Pass ``None`` to preserve nulls. | *required*
`verbose` | <code>[bool](#bool)</code> | Print messages about confound separation. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Concatenated DesignMatrix with run-separated confound columns and multi=True.

######## `get_starting_run_idx`

```python
get_starting_run_idx(dm: DesignMatrix) -> int
```

Determine next run index for multi-run appending.

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

Identify which columns need run-specific separation.

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

Match columns against pattern with wildcard support.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`columns` | <code>list of str</code> | Column names to search. | *required*
`pattern` | <code>[str](#str)</code> | Pattern to match (supports '*' as wildcard). - 'motion*' matches motion_x, motion_y - '*_motion' matches x_motion, y_motion - 'exact' matches only 'exact' | *required*

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[str](#str)]</code> | list of str: Column names matching the pattern.

###### `diagnostics`

Diagnostic and utility functions for DesignMatrix.

**Methods:**

Name | Description
---- | -----------
[`clean`](#clean) | Remove highly correlated columns.
[`corr`](#corr) | Correlation between DesignMatrix columns as an Adjacency.
[`vif`](#vif) | Compute variance inflation factor for each column.



####### Classes

####### Functions##

###### `clean`

```python
clean(dm: DesignMatrix, fill_na: int | float | None = 0, exclude_confounds: bool = False, thresh: float = 0.95, verbose: bool = True) -> DesignMatrix
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
`verbose` | <code>[bool](#bool)</code> | Print dropped column names. Default: True. | <code>True</code>

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

Compute variance inflation factor for each column.

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

DesignMatrix I/O and visualization functions.

Standalone functions extracted from DesignMatrix methods.
Each takes a DesignMatrix instance (`dm`) as its first argument.

**Methods:**

Name | Description
---- | -----------
[`events_to_dm`](#events_to_dm) | Convert a BIDS events table to boxcar regressors aligned to TRs.
[`load_from_file`](#load_from_file) | Read a TSV/CSV into the frame a DesignMatrix wraps.
[`to_numpy`](#to_numpy) | Convert DesignMatrix to numpy array.
[`to_pandas`](#to_pandas) | Convert DesignMatrix to pandas DataFrame.
[`write`](#write) | Write DesignMatrix to file.
[`write_h5`](#write_h5) | Write DesignMatrix to HDF5 file with metadata.



####### Classes

####### Functions##

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

Convert DesignMatrix to numpy array.

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
[`plot_corr`](#plot_corr) | Render a labeled correlation heatmap of the columns.
[`plot_designmatrix`](#plot_designmatrix) | Visualize a DesignMatrix, dispatching over ``method``.
[`plot_matrix`](#plot_matrix) | Render the design matrix as an SPM-style heatmap (rows=TRs, cols=regressors).
[`plot_timeseries`](#plot_timeseries) | Plot regressor time courses as overlaid lines.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`VALID_PLOT_METHODS`](#VALID_PLOT_METHODS) |  | 



####### Attributes##

###### `VALID_PLOT_METHODS`

```python
VALID_PLOT_METHODS = ('matrix', 'timeseries', 'corr')
```



####### Classes

####### Functions##

###### `plot_corr`

```python
plot_corr(dm: DesignMatrix, *, columns: list[str] | None = None, metric: str = 'pearson', figsize: tuple | None = None, title: str | None = None, cmap: str | None = None, ax: plt.Axes | None = None, save: str | None = None, **kwargs: str | None)
```

Render a labeled correlation heatmap of the columns.

Reuses :meth:`DesignMatrix.corr`, which returns a similarity ``Adjacency``
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

See :meth:`DesignMatrix.plot` for the full argument documentation.

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

###### `regressors`

Standalone regressor functions for DesignMatrix.

Each function takes a DesignMatrix as its first argument (`dm`) and returns
a new DesignMatrix with the requested transformation applied.

**Methods:**

Name | Description
---- | -----------
[`add_dct_basis`](#add_dct_basis) | Add discrete cosine transform basis functions (high-pass filter).
[`add_poly`](#add_poly) | Add Legendre polynomial drift terms.
[`convolve`](#convolve) | Convolve columns with HRF or custom kernel.



####### Classes

####### Functions##

###### `add_dct_basis`

```python
add_dct_basis(dm: DesignMatrix, duration: float = 180, drop: int = 0, include_constant: bool = True) -> DesignMatrix
```

Add discrete cosine transform basis functions (high-pass filter).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix to add DCT basis to. | *required*
`duration` | <code>[float](#float)</code> | Filter duration in seconds. Default: 180. | <code>180</code>
`drop` | <code>[int](#int)</code> | Number of low-frequency bases to drop. Default: 0. | <code>0</code>
`include_constant` | <code>[bool](#bool)</code> | If True, also add a constant/intercept column named ``cosine_0`` (analogous to ``poly_0`` in :func:`add_poly`). The underlying DCT basis drops the constant per SPM convention; set False to match SPM behavior. Default: True. | <code>True</code>

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

Convolve columns with HRF or custom kernel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix to convolve. | *required*
`conv_func` | <code>[str](#str) or [ndarray](#ndarray)</code> | 'hrf' for canonical Glover HRF, or custom kernel(s). Can be 1D array (single kernel) or 2D (samples x kernels) | <code>'hrf'</code>
`columns` | <code>list of str</code> | Columns to convolve (default: all non-polynomial columns) | <code>None</code>

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

###### `transforms`

Standalone transform functions for DesignMatrix.

Each function takes a DesignMatrix instance as the first argument (`dm`)
and returns a new DesignMatrix via `copy_with(dm,...)`.

**Methods:**

Name | Description
---- | -----------
[`downsample`](#downsample) | Reduce temporal resolution to target frequency using Polars-native operations.
[`standardize`](#standardize) | Standardize columns using the specified method.
[`upsample`](#upsample) | Increase temporal resolution to target frequency using Polars-native interpolation.
[`zscore`](#zscore) | Z-score standardize columns (mean=0, std=1).



####### Classes

####### Functions##

###### `downsample`

```python
downsample(dm: DesignMatrix, target: float, **kwargs: float) -> DesignMatrix
```

Reduce temporal resolution to target frequency using Polars-native operations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance to transform. | *required*
`target` | <code>[float](#float)</code> | Target sampling frequency in Hz (must be < current sampling_freq). | *required*
`**kwargs` |  | Additional keyword arguments:<br>- **method** (str): Aggregation method - 'mean' or 'median'.   Default: 'mean'. | <code>{}</code>

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
`columns` | <code>[list](#list)[[str](#str)] \| None</code> | Columns to standardize. If None, standardize all non-polynomial columns. | <code>None</code>
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
upsample(dm: DesignMatrix, target: float, method: str = 'linear', **kwargs: str) -> DesignMatrix
```

Increase temporal resolution to target frequency using Polars-native interpolation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance to transform. | *required*
`target` | <code>[float](#float)</code> | Target sampling frequency in Hz (must be > current sampling_freq) | *required*
`method` | <code>[str](#str)</code> | Interpolation method - 'linear' or 'nearest' (default: 'linear') | <code>'linear'</code>
`**kwargs` |  | Reserved for future extensions | <code>{}</code>

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

Z-score standardize columns (mean=0, std=1).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance to transform. | *required*
`columns` | <code>list of str</code> | Columns to standardize. If None, standardize all non-polynomial columns. | <code>None</code>

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
[`copy_with`](#copy_with) | Create new DesignMatrix with updated data/metadata.
[`df_passthrough`](#df_passthrough) | Resolve ``name`` on ``dm.data``; re-wrap DataFrame results for allowlisted methods.
[`get_data_columns`](#get_data_columns) | Get column names, optionally excluding confound regressors.
[`get_metadata`](#get_metadata) | Extract metadata as dict (for copying).

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`WRAP_AS_DESIGNMATRIX`](#WRAP_AS_DESIGNMATRIX) |  | 



####### Attributes##

###### `WRAP_AS_DESIGNMATRIX`

```python
WRAP_AS_DESIGNMATRIX = frozenset({'slice', 'filter', 'select'})
```



####### Classes

####### Functions##

###### `copy_with`

```python
copy_with(dm: DesignMatrix, new_df: pl.DataFrame, **metadata_updates: pl.DataFrame) -> DesignMatrix
```

Create new DesignMatrix with updated data/metadata.

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

#### `fitresults`

Immutable container for model fitting results.

This module provides the Fit dataclass, which stores results from model fitting
operations in nltools. It uses pure numpy arrays and has no dependencies on
BrainData or other nltools data structures, making it suitable for standalone
use with inference algorithms.

Examples
--------
**Using with BrainData workflow:**

>>> from nltools.data import BrainData
>>> brain = BrainData(data="brain_data.nii.gz")
>>> fit = brain.fit(X=design_matrix, mode="ridge", cv_dict={"type": "kfold", "n_splits": 5})
>>> print(fit.available())
['fitted_values', 'weights', 'scores', 'cv_scores', 'cv_mean_score', 'cv_predictions', 'cv_folds']

**Using with inference algorithms directly:**

>>> from nltools.algorithms import ridge_cv
>>> import numpy as np
>>> X = np.random.randn(100, 5)
>>> y = np.random.randn(100, 1000)
>>> fit = ridge_cv(X, y, alpha=1.0, cv_dict={"type": "kfold", "n_splits": 5})
>>> fit.cv_mean_score.shape
(1000,)

**Serialization/deserialization:**

>>> # Save all non-None results
>>> np.savez("fit_results.npz", **fit.asdict())
>>>
>>> # Load and reconstruct
>>> loaded = np.load("fit_results.npz")
>>> fit_reconstructed = Fit(**{k: loaded[k] for k in loaded.files})

**Export to .npz:**

>>> # Export only specific fields
>>> import numpy as np
>>> np.savez("weights_and_scores.npz",
...          weights=fit.weights,
...          scores=fit.scores)

**Introspection:**

>>> # Check what's available
>>> if 'cv_scores' in fit.available():
...     print(f"CV R² range: [{fit.cv_mean_score.min():.3f}, {fit.cv_mean_score.max():.3f}]")
>>>
>>> # Get as dict and convert to a polars DataFrame (for scalar and 1D arrays)
>>> import polars as pl
>>> results_dict = fit.asdict()
>>> df = pl.DataFrame({k: v for k, v in results_dict.items() if v.ndim <= 1})

**Classes:**

Name | Description
---- | -----------
[`Fit`](#Fit) | Immutable container for model fitting results.
[`Predict`](#Predict) | Immutable container for prediction / MVPA decoding results.



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

Attributes
----------
fitted_values : ndarray
    Fitted values or predictions, always present
weights : ndarray, optional
    Model coefficients (Ridge)
scores : ndarray, optional
    R² scores (Ridge)
betas : ndarray, optional
    Beta coefficients (GLM)
t_stats : ndarray, optional
    T-statistics (GLM)
p_values : ndarray, optional
    P-values (GLM)
se : ndarray, optional
    Standard errors (GLM)
residuals : ndarray, optional
    Residuals (GLM)
r2 : ndarray, optional
    R² values (GLM)
cv_scores : ndarray, optional
    Per-fold cross-validation scores
cv_mean_score : ndarray, optional
    Mean cross-validation score across folds
cv_predictions : ndarray, optional
    Out-of-fold predictions
cv_folds : ndarray, optional
    Fold indices for each sample
cv_best_alpha : float, optional
    Best alpha selected via cross-validation
cv_alpha_scores : ndarray, optional
    Cross-validation scores for each alpha tested

Methods
-------
available() : list
    Returns list of non-None attribute names (excludes private fields)
asdict(include_none=False) : dict
    Converts to dictionary, optionally excluding None values

Examples
--------
**Creating a Fit object (Ridge without CV):**

>>> import numpy as np
>>> from nltools.data.fitresults import Fit
>>> fit = Fit(
...     fitted_values=np.random.randn(100, 1000),
...     weights=np.random.randn(5, 1000),
...     scores=np.random.randn(1000)
... )
>>> fit.available()
['fitted_values', 'weights', 'scores']

**Creating a Fit object (Ridge with CV):**

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

**Immutability:**

>>> try:
...     fit.scores = np.zeros(1000)  # Will raise FrozenInstanceError
... except AttributeError:
...     print("Cannot modify frozen dataclass")
Cannot modify frozen dataclass

**Export/serialization:**

>>> # Save to .npz
>>> np.savez("results.npz", **fit.asdict())
>>>
>>> # Load and reconstruct
>>> loaded = np.load("results.npz")
>>> fit_reloaded = Fit(**{k: loaded[k] for k in loaded.files})

Notes
-----
- Frozen dataclass ensures results cannot be accidentally modified
- All attributes are numpy arrays (except cv_best_alpha which is float)
- None values indicate that field was not computed for this model/method
- Private fields (starting with _) are excluded from available() and asdict()

**Methods:**

Name | Description
---- | -----------
[`asdict`](#asdict) | Convert to dictionary.
[`available`](#available) | Return list of non-None attribute names.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`betas`](#betas) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`cv_alpha_scores`](#cv_alpha_scores) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`cv_best_alpha`](#cv_best_alpha) | <code>[float](#float) \| None</code> | 
[`cv_folds`](#cv_folds) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`cv_mean_score`](#cv_mean_score) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`cv_predictions`](#cv_predictions) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`cv_scores`](#cv_scores) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`fitted_values`](#fitted_values) | <code>[ndarray](#numpy.ndarray)</code> | 
[`p_values`](#p_values) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`r2`](#r2) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`residuals`](#residuals) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`scores`](#scores) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`se`](#se) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`t_stats`](#t_stats) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`weights`](#weights) | <code>[ndarray](#numpy.ndarray) \| None</code> | 



####### Attributes##

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

Examples
--------
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

Examples
--------
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

###### `Predict`

```python
Predict(predictions: np.ndarray | None = None, scores: np.ndarray | None = None, mean_score: Any = None, std_score: Any = None, cv_folds: np.ndarray | None = None, roi_labels: np.ndarray | None = None, accuracy_map: Any = None, weight_map: Any = None, fold_weight_maps: Any = None, estimator: Any = None) -> None
```

Immutable container for prediction / MVPA decoding results.

Mirrors :class:`Fit`: frozen, all fields default to ``None``, populated
based on the dispatch path (``method``, ``y`` vs ``X``, ``refit``) used
by :meth:`BrainData.predict`. Fields not applicable to the call remain
``None`` and are filtered from :meth:`available` and :meth:`asdict`.

**Brain-space outputs are :class:`BrainData` objects**, not raw arrays —
so ``result.weight_map.plot()`` works directly. Drop down to numpy via
``result.weight_map.data`` if needed. Non-spatial fields (``predictions``,
``cv_folds``, scalar scores) stay as numpy.

Field shapes by dispatch:

**method='whole_brain'** (with ``y``):
    - ``predictions``: ``(n_samples,)`` ndarray, OOF predictions from CV
    - ``scores``: ``(n_folds,)`` ndarray, per-fold score
    - ``mean_score``: float, mean across folds
    - ``std_score``: float, std across folds
    - ``cv_folds``: ``(n_samples,)`` ndarray, fold index per sample
    - ``weight_map``: BrainData ``(1, n_voxels)``, ``coef_`` from one
      model fit on the **full** ``(X, y)``. The publishable map.
    - ``fold_weight_maps``: BrainData ``(n_folds, n_voxels)``, per-fold
      ``coef_`` stack — for stability analysis (e.g., across-fold std).
    - ``estimator``: the fitted all-data sklearn estimator (use for
      ``.predict()`` on new data).

**method='roi'** (with ``y``):
    - ``scores``: ``(n_folds, n_rois)`` ndarray
    - ``mean_score``: ``(n_rois,)`` ndarray, mean across folds per parcel
    - ``std_score``: ``(n_rois,)`` ndarray
    - ``roi_labels``: ``(n_rois,)`` ndarray of atlas integer IDs in the
      same order as ``mean_score`` / ``std_score`` / ``scores`` axis 1
    - ``accuracy_map``: BrainData ``(1, n_voxels)``, every voxel inside
      parcel *i* set to that parcel's mean accuracy (others NaN)
    - ``weight_map``: BrainData ``(1, n_voxels)``, per-parcel ``coef_``
      from each parcel's all-data fit, written back into voxel space
      (atlas is a label image so reassembly is disjoint). Voxels outside
      any parcel are NaN. Magnitudes across parcels are not directly
      comparable — different parcels live on different X distributions.
    - ``fold_weight_maps``: BrainData ``(n_folds, n_voxels)``
    - ``estimator``: ``dict[int, sklearn]`` keyed by atlas label

    If any parcel can't expose ``.coef_`` (non-linear model, ``SelectKBest``
    in pipeline), ``weight_map`` / ``fold_weight_maps`` / ``estimator``
    all collapse to ``None`` for the whole call.

**method='searchlight'** (with ``y``):
    - ``accuracy_map``: BrainData ``(1, n_voxels)``, sphere-centered
      accuracy at each voxel

Note: encoding-model timeseries prediction (``bd.predict(X=...)``) returns
a ``BrainData`` directly, not a ``Predict`` — the natural container for a
voxel timeseries.

Why the all-data fit is canonical: the CV mean of per-fold ``coef_``
vectors doesn't correspond to any actual fitted estimator (each fold
saw a different subset). The all-data refit is a single, real model
with all the information used. CV gives the honest *score*; the refit
gives the publishable *map*. ``fold_weight_maps`` is still exposed for
stability analysis, and the CV-mean is one line away if you want it
(``fold_weight_maps.data.mean(axis=0)``).

Methods
-------
available() : list
    Names of non-None fields (excludes private).
asdict(include_none=False) : dict
    Convert to dict for serialization (private fields always excluded).

**Methods:**

Name | Description
---- | -----------
[`asdict`](#asdict) | Convert to dictionary.
[`available`](#available) | Return names of non-None fields (excludes private).

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`accuracy_map`](#accuracy_map) | <code>[Any](#typing.Any)</code> | 
[`cv_folds`](#cv_folds) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`estimator`](#estimator) | <code>[Any](#typing.Any)</code> | 
[`fold_weight_maps`](#fold_weight_maps) | <code>[Any](#typing.Any)</code> | 
[`mean_score`](#mean_score) | <code>[Any](#typing.Any)</code> | 
[`predictions`](#predictions) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`roi_labels`](#roi_labels) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`scores`](#scores) | <code>[ndarray](#numpy.ndarray) \| None</code> | 
[`std_score`](#std_score) | <code>[Any](#typing.Any)</code> | 
[`weight_map`](#weight_map) | <code>[Any](#typing.Any)</code> | 



####### Attributes##

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

NeuroLearn Analysis Tools
=========================
These tools provide the ability to quickly run
machine-learning analyses on imaging data

**Classes:**

Name | Description
---- | -----------
[`Roc`](#Roc) | Roc Class



##### Classes

###### `Roc`

```python
Roc(input_values = None, binary_outcome = None, threshold_type = 'optimal_overall', forced_choice = None, **kwargs)
```

Roc Class

The Roc class is based on Tor Wager's Matlab roc_plot.m function and
allows a user to easily run different types of receiver operator
characteristic curves.  For example, one might be interested in single
interval or forced choice.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`input_values` |  | nibabel data instance | <code>None</code>
`binary_outcome` |  | vector of training labels | <code>None</code>
`threshold_type` |  | ['optimal_overall', 'optimal_balanced',             'minimum_sdt_bias'] | <code>'optimal_overall'</code>
`**kwargs` |  | Additional keyword arguments to pass to the prediction         algorithm | <code>{}</code>

**Methods:**

Name | Description
---- | -----------
[`calculate`](#calculate) | Calculate Receiver Operating Characteristic plot (ROC) for
[`plot`](#plot) | Create ROC Plot
[`summary`](#summary) | Display a formatted summary of ROC analysis.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`binary_outcome`](#binary_outcome) |  | 
[`forced_choice`](#forced_choice) |  | 
[`input_values`](#input_values) |  | 
[`threshold_type`](#threshold_type) |  | 



####### Attributes##

###### `binary_outcome`

```python
binary_outcome = np.asarray(binary_outcome).flatten()
```

######## `forced_choice`

```python
forced_choice = deepcopy(forced_choice)
```

######## `input_values`

```python
input_values = deepcopy(input_values)
```

######## `threshold_type`

```python
threshold_type = deepcopy(threshold_type)
```



####### Functions##

###### `calculate`

```python
calculate(input_values = None, binary_outcome = None, criterion_values = None, threshold_type = 'optimal_overall', forced_choice = None, balanced_acc = False)
```

Calculate Receiver Operating Characteristic plot (ROC) for
single-interval classification.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`input_values` |  | nibabel data instance | <code>None</code>
`binary_outcome` |  | vector of training labels | <code>None</code>
`criterion_values` |  | (optional) criterion values for calculating fpr             & tpr | <code>None</code>
`threshold_type` |  | ['optimal_overall', 'optimal_balanced',             'minimum_sdt_bias'] | <code>'optimal_overall'</code>
`forced_choice` |  | index indicating position for each unique subject             (default=None) | <code>None</code>
`balanced_acc` |  | balanced accuracy for single-interval classification             (bool). THIS IS NOT COMPLETELY IMPLEMENTED BECAUSE             IT AFFECTS ACCURACY ESTIMATES, BUT NOT P-VALUES OR             THRESHOLD AT WHICH TO EVALUATE SENS/SPEC | <code>False</code>
`**kwargs` |  | Additional keyword arguments to pass to the prediction             algorithm | *required*

######## `plot`

```python
plot(plot_method = 'gaussian', balanced_acc = False, **kwargs)
```

Create ROC Plot

Create a specific kind of ROC curve plot, based on input values
along a continuous distribution and a binary outcome variable (logical)

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`plot_method` |  | type of plot ['gaussian','observed'] | <code>'gaussian'</code>
`binary_outcome` |  | vector of training labels | *required*
`**kwargs` |  | Additional keyword arguments to pass to the prediction         algorithm | <code>{}</code>

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

NeuroLearn Simulator Tools
==========================

Tools to simulate multivariate data.

**Classes:**

Name | Description
---- | -----------
[`SimulateGrid`](#SimulateGrid) | Simulate 2D grid data for testing statistical methods.
[`Simulator`](#Simulator) | Simulate fMRI data with realistic spatial and temporal characteristics.



##### Classes

###### `SimulateGrid`

```python
SimulateGrid(grid_width = 100, signal_width = 20, n_subjects = 20, sigma = 1, signal_amplitude = None, random_state = None)
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
[`data`](#data) |  | The simulated data array of shape (n_subjects, grid_width, grid_width).
[`t_values`](#t_values) |  | T-statistic values after fitting.
[`p_values`](#p_values) |  | P-values after fitting.
[`thresholded`](#thresholded) |  | Thresholded statistical map.
[`isfit`](#isfit) |  | Whether fit() has been called.

**Examples:**

```pycon
>>> from nltools.data.simulator import SimulateGrid
>>> sim = SimulateGrid(signal_amplitude=0.5, random_state=42)
>>> sim.fit(n_permute=1000)
>>> sim.plot()
```

**Methods:**

Name | Description
---- | -----------
[`add_signal`](#add_signal) | Add rectangular signal to self.data
[`create_mask`](#create_mask) | Create a mask for where the signal is located in grid.
[`fit`](#fit) | Run ttest on self.data
[`plot_grid_simulation`](#plot_grid_simulation) | Create a plot of the simulations
[`run_multiple_simulations`](#run_multiple_simulations) | This method will run multiple simulations to calculate overall false positive rate
[`threshold_simulation`](#threshold_simulation) | Threshold simulation



####### Attributes##

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

Add rectangular signal to self.data

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

Run ttest on self.data

######## `plot_grid_simulation`

```python
plot_grid_simulation(threshold, threshold_type, n_simulations = 100, correction = None)
```

Create a plot of the simulations

######## `run_multiple_simulations`

```python
run_multiple_simulations(threshold, threshold_type, n_simulations = 100, correction = None)
```

This method will run multiple simulations to calculate overall false positive rate

######## `threshold_simulation`

```python
threshold_simulation(threshold, threshold_type, correction = None)
```

Threshold simulation

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`threshold` | <code>[float](#float)</code> | threshold to apply to simulation | *required*
`threshhold_type` | <code>[str](#str)</code> | type of threshold to use can be a specific t-value or p-value ['t', 'p', 'q'] | *required*

###### `Simulator`

```python
Simulator(brain_mask = None, output_dir = None, random_state = None)
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
[`brain_mask`](#brain_mask) |  | The brain mask image used for simulation.
[`output_dir`](#output_dir) |  | Output directory path.
[`random_state`](#random_state) |  | Random state for reproducible simulations.

**Examples:**

```pycon
>>> from nltools.data.simulator import Simulator
>>> sim = Simulator(random_state=42)
>>> # Create a dataset with signal in specific regions
>>> data = sim.create_data(y=[1, -1, 1, -1], sigma=1, n_reps=10)
```

**Methods:**

Name | Description
---- | -----------
[`create_cov_data`](#create_cov_data) | create continuous simulated data with covariance
[`create_data`](#create_data) | create simulated data with integers
[`create_ncov_data`](#create_ncov_data) | create continuous simulated data with covariance
[`gaussian`](#gaussian) | create a 3D gaussian signal normalized to a given intensity
[`n_spheres`](#n_spheres) | generate a set of spheres in the brain mask space
[`normal_noise`](#normal_noise) | produce a normal noise distribution for all all points in the brain mask
[`sphere`](#sphere) | create a sphere of given radius at some point p in the brain mask
[`to_nifti`](#to_nifti) | convert a numpy matrix to the nifti format and assign to it the brain_mask's affine matrix



####### Attributes##

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
create_cov_data(cor, cov, sigma, mask = None, reps = 1, n_sub = 1, output_dir = None)
```

create continuous simulated data with covariance

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cor` |  | amount of covariance between each voxel and Y variable | *required*
`cov` |  | amount of covariance between voxels | *required*
`sigma` |  | amount of noise to add | *required*
`radius` |  | vector of radius.  Will create multiple spheres if len(radius) > 1 | *required*
`center` |  | center(s) of sphere(s) of the form [px, py, pz] or [[px1, py1, pz1], ..., [pxn, pyn, pzn]] | *required*
`reps` |  | number of data repetitions | <code>1</code>
`n_sub` |  | number of subjects to simulate | <code>1</code>
`output_dir` |  | string path of directory to output data.  If None, no data will be written | <code>None</code>
`**kwargs` |  | Additional keyword arguments to pass to the prediction algorithm | *required*

######## `create_data`

```python
create_data(levels, sigma, radius = 5, center = None, reps = 1, output_dir = None)
```

create simulated data with integers

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`levels` |  | vector of intensities or class labels | *required*
`sigma` |  | amount of noise to add | *required*
`radius` |  | vector of radius.  Will create multiple spheres if len(radius) > 1 | <code>5</code>
`center` |  | center(s) of sphere(s) of the form [px, py, pz] or [[px1, py1, pz1], ..., [pxn, pyn, pzn]] | <code>None</code>
`reps` |  | number of data repetitions useful for trials or subjects | <code>1</code>
`output_dir` |  | string path of directory to output data.  If None, no data will be written | <code>None</code>
`**kwargs` |  | Additional keyword arguments to pass to the prediction algorithm | *required*

######## `create_ncov_data`

```python
create_ncov_data(cor, cov, sigma, masks = None, reps = 1, n_sub = 1, output_dir = None)
```

create continuous simulated data with covariance

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cor` |  | amount of covariance between each voxel and Y variable (an int or a vector) | *required*
`cov` |  | amount of covariance between voxels (an int or a matrix) | *required*
`sigma` |  | amount of noise to add | *required*
`mask` |  | region(s) where we will have activations (list if more than one) | *required*
`reps` |  | number of data repetitions | <code>1</code>
`n_sub` |  | number of subjects to simulate | <code>1</code>
`output_dir` |  | string path of directory to output data.  If None, no data will be written | <code>None</code>
`**kwargs` |  | Additional keyword arguments to pass to the prediction algorithm | *required*

######## `gaussian`

```python
gaussian(mu, sigma, i_tot)
```

create a 3D gaussian signal normalized to a given intensity

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

generate a set of spheres in the brain mask space

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`radius` |  | vector of radius.  Will create multiple spheres if len(radius) > 1 | *required*
`centers` |  | a vector of sphere centers of the form [px, py, pz] or [[px1, py1, pz1], ..., [pxn, pyn, pzn]] | *required*

######## `normal_noise`

```python
normal_noise(mu, sigma)
```

produce a normal noise distribution for all all points in the brain mask

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mu` |  | average value of the gaussian signal (usually set to 0) | *required*
`sigma` |  | standard deviation | *required*

######## `sphere`

```python
sphere(r, p)
```

create a sphere of given radius at some point p in the brain mask

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`r` |  | radius of the sphere | *required*
`p` |  | point (in coordinates of the brain mask) of the center of the sphere | *required*

######## `to_nifti`

```python
to_nifti(m)
```

convert a numpy matrix to the nifti format and assign to it the brain_mask's affine matrix

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`m` |  | the 3D numpy matrix we wish to convert to .nii | *required*



##### Methods