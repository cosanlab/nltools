## `data`

nltools data types.

**Modules:**

Name | Description
---- | -----------
[`adjacency`](#adjacency) | Provide data structures for working with similarity and dissimilarity matrices.
[`atlases`](#atlases) | Atlas registry, lazy loading, and coordinate labeling.
`braindata` | Represent brain image data with the BrainData class.
[`collection`](#collection) | BrainCollection — multi-subject brain-data container (v0.6.0).
[`designmatrix`](#designmatrix) | Provide a Polars-based design matrix for neuroimaging analysis.
[`fitresults`](#fitresults) | Immutable container for model fitting results.
[`roc`](#roc) | NeuroLearn Analysis Tools
[`simulator`](#simulator) | NeuroLearn Simulator Tools

**Classes:**

Name | Description
---- | -----------
[`Adjacency`](#adjacency) | Represent adjacency matrices in vectorized form.
[`BrainCollection`](#braincollection) | Parallel, lazy iterator of ``BrainData`` whose API mirrors ``BrainData``.
`BrainData` | Represent neuroimaging data as vectors instead of three-dimensional matrices.
[`DesignMatrix`](#designmatrix) | Represent experimental designs for neuroimaging with Polars.
[`Fit`](#fit) | Immutable container for model fitting results.
[`Roc`](#roc) | Roc Class
[`SimulateGrid`](#simulategrid) | Simulate 2D grid data for testing statistical methods.
[`Simulator`](#simulator) | Simulate fMRI data with realistic spatial and temporal characteristics.



### Classes

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
[`append`](#append) | Append data to an Adjacency instance.
[`bootstrap`](#bootstrap) | Bootstrap statistics using efficient online algorithms.
[`cluster_summary`](#cluster-summary) | Provide summaries of clusters within Adjacency matrices.
[`copy`](#copy) | Create a copy of Adjacency object.
[`distance`](#distance) | Calculate distance between images within an Adjacency() instance.
[`distance_to_similarity`](#distance-to-similarity) | Convert distance matrix to similarity matrix.
[`generate_permutations`](#generate-permutations) | Generate permuted versions of an Adjacency instance lazily.
[`mean`](#mean) | Calculate mean of Adjacency.
[`median`](#median) | Calculate median of Adjacency.
[`plot`](#plot) | Create a heatmap of an Adjacency matrix.
[`plot_label_distance`](#plot-label-distance) | Create a violin plot of within- and between-label distances.
[`plot_mds`](#plot-mds) | Plot multidimensional scaling.
[`plot_silhouette`](#plot-silhouette) | Create a silhouette plot.
[`r_to_z`](#r-to-z) | Apply Fisher's r-to-z transformation to each data element.
[`regress`](#regress) | Run a regression on an adjacency instance.
[`similarity`](#similarity) | Calculate similarity between two Adjacency matrices.
[`social_relations_model`](#social-relations-model) | Estimate the social relations model from a matrix for a round-robin design.
[`squareform`](#squareform) | Convert adjacency data back to square form.
[`stats_label_distance`](#stats-label-distance) | Calculate permutation tests on within and between label distance.
[`std`](#std) | Calculate standard deviation of Adjacency.
[`sum`](#sum) | Calculate sum of Adjacency.
[`threshold`](#threshold) | Threshold an Adjacency instance.
[`to_brain`](#to-brain) | Project per-matrix scalars back to voxel-space `BrainData`.
[`to_graph`](#to-graph) | Convert a single Adjacency matrix into a NetworkX graph.
[`to_square`](#to-square) | Convert adjacency back to square matrix format.
[`ttest`](#ttest) | Calculate ttest across samples.
[`write`](#write) | Write out Adjacency object to csv file.
[`z_to_r`](#z-to-r) | Convert each z score back into an r value.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`Y` | <code>[DataFrame](#polars.DataFrame)</code> | Training labels as a polars DataFrame (possibly empty).
[`data`](#data) |  | 
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

Create a heatmap of an Adjacency matrix.

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

Create a violin plot of within- and between-label distances.

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

Plot multidimensional scaling.

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

Create a silhouette plot.

###### `r_to_z`

```python
r_to_z()
```

Apply Fisher's r-to-z transformation to each data element.

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

Calculate similarity between two Adjacency matrices.

The default uses Spearman correlation and a permutation test.

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

Convert adjacency data back to square form.

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

###### `to_graph`

```python
to_graph()
```

Convert a single Adjacency matrix into a NetworkX graph.

This currently works only for ``single_matrix``.

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

Convert each z score back into an r value.

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
[`cleanup_all`](#cleanup-all) | Remove every ``.nltools_cache/{run_id}/`` under ``directory``.
[`compute_contrasts`](#compute-contrasts) | Compute per-subject contrast maps from fit-bundle items.
[`concat`](#concat) | 
[`cv`](#cv) | Build a cross-validation pipeline for cross-subject prediction.
[`detrend`](#detrend) | 
[`filter`](#filter) | Filter to a subset by predicate, mask, or boolean Series.
[`fit`](#fit) | Per-subject fit; returns a path-backed collection of HDF5 fit bundles.
[`from_bids`](#from-bids) | Auto-pair BOLD with events.tsv (→ ``DesignMatrix``) and confounds.tsv.
[`from_glob`](#from-glob) | 
[`from_paths`](#from-paths) | 
[`isc`](#isc) | 
[`isc_test`](#isc-test) | 
[`iter_pairs`](#iter-pairs) | Yield ``(BrainData, DesignMatrix | None)`` pairs.
[`load`](#load) | Materialize path-backed items in place.
[`map`](#map) | Apply an arbitrary ``fn(BrainData) -> BrainData`` to each item in parallel.
[`max`](#max) | 
[`mean`](#mean) | 
[`median`](#median) | 
[`memory_estimate`](#memory-estimate) | 
[`min`](#min) | 
[`permutation_test`](#permutation-test) | 
[`permutation_test2`](#permutation-test2) | 
[`predict`](#predict) | Dispatch prediction according to the provided target argument.
[`read`](#read) | Read a collection written by ``write()``.
[`resample`](#resample) | 
[`smooth`](#smooth) | 
[`standardize`](#standardize) | 
[`std`](#std) | 
[`steps`](#steps) | List step subdirs under ``cache_root``, oldest to newest (lex-sorted).
[`sum`](#sum) | 
[`threshold`](#threshold) | 
[`transform_designs`](#transform-designs) | Map a function over paired ``DesignMatrix``es.
[`ttest`](#ttest) | 
[`ttest2`](#ttest2) | 
[`unload`](#unload) | Drop in-memory data for items with backing paths.
[`var`](#var) | 
[`write`](#write) | 

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`cache_root` | <code>[Path](#pathlib.Path)</code> | 
`designs` | <code>[list](#list)</code> | 
`is_loaded` | <code>[list](#list)[[bool](#bool)]</code> | Per-item flag — True iff the slot holds a ``BrainData`` (not a path).
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image)</code> | 
`metadata` | <code>[DataFrame](#polars.DataFrame)</code> | 
`n_subjects` | <code>[int](#int)</code> | 
`n_voxels` | <code>[int](#int)</code> | Return the voxel count from the mask.
`shape` | <code>[tuple](#tuple)[[int](#int), [int](#int) \| None, [int](#int)]</code> | ``(n_subjects, n_obs_or_None_if_ragged, n_voxels)``.

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

Build a cross-validation pipeline for cross-subject prediction.

See ``pipeline.py`` for details.

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

Materialize path-backed items in place.

Returns ``self`` for chaining.

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

Dispatch prediction according to the provided target argument.

- ``y=`` only → group MVPA (subjects as samples) → ``BrainData``
- ``X_new=`` only → per-subject predict-after-fit → ``BrainCollection``
- both / neither → raise

``predict(y=...)`` requires single-map-per-subject items (run
``compute_contrasts(...)`` first if you have GLM/ridge bundles).

###### `read`

```python
read(directory: Path | str, *, mask: nib.Nifti1Image | Path | str, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Read a collection written by ``write()``.

This does not recover from cache subdirectories in v0.6.0.

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

Drop in-memory data for items with backing paths.

Returns ``self``.

###### `var`

```python
var() -> BrainData
```

###### `write`

```python
write(directory: Path | str, *, pattern: str = 'image_{i:04d}.nii.gz', metadata_file: str | None = 'metadata.csv') -> list[Path]
```

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
[`add_dct_basis`](#add-dct-basis) | Add discrete cosine transform basis functions for high-pass filtering.
[`add_poly`](#add-poly) | Add Legendre polynomial drift terms.
[`append`](#append) | Concatenate design matrices.
[`clean`](#clean) | Remove highly correlated columns.
[`convolve`](#convolve) | Convolve columns with an HRF or custom kernel.
[`copy`](#copy) | Create a deep copy of the DesignMatrix.
[`corr`](#corr) | Calculate column correlations as a similarity ``Adjacency``.
[`downsample`](#downsample) | Reduce temporal resolution using Polars-native operations.
[`drop`](#drop) | Drop specified columns.
[`fillna`](#fillna) | Fill NaN/null values with specified value.
[`plot`](#plot) | Visualize the design matrix.
[`replace_data`](#replace-data) | Replace data columns while preserving confounds and metadata.
[`standardize`](#standardize) | Standardize columns using the specified method.
[`sum`](#sum) | Compute the sum along an axis.
[`to_numpy`](#to-numpy) | Convert a DesignMatrix to a NumPy array.
[`to_pandas`](#to-pandas) | Convert DesignMatrix to pandas DataFrame.
[`upsample`](#upsample) | Increase temporal resolution to a target frequency.
[`vif`](#vif) | Compute the variance inflation factor for each column.
[`with_columns`](#with-columns) | Add or replace columns via Polars expressions.
[`write`](#write) | Write DesignMatrix to file.
[`zscore`](#zscore) | Z-score standardize columns to mean zero and unit variance.

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

###### `downsample`

```python
downsample(target: float, method: str = 'mean', **kwargs: str) -> DesignMatrix
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

Compute the sum along an axis.

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

Convert a DesignMatrix to a NumPy array.

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

###### `zscore`

```python
zscore(columns: list[str] | None = None) -> DesignMatrix
```

Z-score standardize columns to mean zero and unit variance.

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
`cv_alpha_scores` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`cv_best_alpha` | <code>[float](#float) \| None</code> | 
`cv_folds` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`cv_mean_score` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`cv_predictions` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`cv_scores` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`fitted_values` | <code>[ndarray](#numpy.ndarray)</code> | 
`p_values` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`r2` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`residuals` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`scores` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`se` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`t_stats` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`weights` | <code>[ndarray](#numpy.ndarray) \| None</code> | 

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
[`binary_outcome`](#binary-outcome) |  | 
`forced_choice` |  | 
`input_values` |  | 
`threshold_type` |  | 

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
`t_values` |  | T-statistic values after fitting.
`p_values` |  | P-values after fitting.
`thresholded` |  | Thresholded statistical map.
`isfit` |  | Whether fit() has been called.

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
[`add_signal`](#add-signal) | Add rectangular signal to self.data
[`create_mask`](#create-mask) | Create a mask for where the signal is located in grid.
[`fit`](#fit) | Run ttest on self.data
[`plot_grid_simulation`](#plot-grid-simulation) | Create a plot of the simulations
[`run_multiple_simulations`](#run-multiple-simulations) | This method will run multiple simulations to calculate overall false positive rate
[`threshold_simulation`](#threshold-simulation) | Threshold simulation

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
[`brain_mask`](#brain-mask) |  | The brain mask image used for simulation.
`output_dir` |  | Output directory path.
`random_state` |  | Random state for reproducible simulations.

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
[`create_cov_data`](#create-cov-data) | create continuous simulated data with covariance
[`create_data`](#create-data) | create simulated data with integers
[`create_ncov_data`](#create-ncov-data) | create continuous simulated data with covariance
[`gaussian`](#gaussian) | create a 3D gaussian signal normalized to a given intensity
[`n_spheres`](#n-spheres) | generate a set of spheres in the brain mask space
[`normal_noise`](#normal-noise) | produce a normal noise distribution for all all points in the brain mask
[`sphere`](#sphere) | create a sphere of given radius at some point p in the brain mask
[`to_nifti`](#to-nifti) | convert a numpy matrix to the nifti format and assign to it the brain_mask's affine matrix

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

Provide data structures for working with similarity and dissimilarity matrices.

**Modules:**

Name | Description
---- | -----------
[`io`](#io) | I/O functions for Adjacency objects.
[`modeling`](#modeling) | Provide standalone modeling and inference functions for Adjacency matrices.
[`plotting`](#plotting) | Plotting functions for Adjacency matrices.
[`spatial`](#spatial) | Spatial-scale provenance for stacked Adjacency matrices.
[`stats`](#stats) | Provide standalone statistical functions for Adjacency matrices.
[`utils`](#utils) | Shared helpers for Adjacency submodules.

**Classes:**

Name | Description
---- | -----------
[`Adjacency`](#adjacency) | Represent adjacency matrices in vectorized form.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`MAX_INT` |  | 
`nx` |  | 

##### Methods

##### Modules

###### `io`

I/O functions for Adjacency objects.

**Methods:**

Name | Description
---- | -----------
[`to_graph`](#to-graph) | Convert Adjacency into networkx graph.
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

Provide standalone modeling and inference functions for Adjacency matrices.

Each function takes an Adjacency instance as its first argument (`adj`).

**Methods:**

Name | Description
---- | -----------
[`bootstrap`](#bootstrap) | Bootstrap statistics using efficient online algorithms.
`convert_bootstrap_results_to_adjacency` | Convert bootstrap results dictionary to Adjacency format.
[`generate_permutations`](#generate-permutations) | Generate permuted versions of an Adjacency instance lazily.
[`regress`](#regress) | Run a regression on an adjacency instance.
[`social_relations_model`](#social-relations-model) | Estimate the social relations model from a matrix for a round-robin design.



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

###### `plotting`

Plotting functions for Adjacency matrices.

**Methods:**

Name | Description
---- | -----------
[`plot_adjacency`](#plot-adjacency) | Create Heatmap of Adjacency Matrix.
[`plot_mds`](#plot-mds) | Plot Multidimensional Scaling.



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
[`SpatialScale`](#spatialscale) | Record provenance for a per-parcel or per-searchlight Adjacency stack.



####### Classes##

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
[`atlas`](#atlas) | <code>[BrainData](#nltools.data.BrainData)</code> | Labeled volume indicating parcel membership (or searchlight centers). One matrix in the stack per unique label.
`roi_labels` | <code>[ndarray](#numpy.ndarray)</code> | Integer atlas IDs in stack order. ``len(roi_labels)`` must equal the number of matrices in the stack.
`source_mask` | <code>[Nifti1Image](#nibabel.Nifti1Image)</code> | The brain mask the atlas/values live in. Used as the target space for back-projection in ``Adjacency.to_brain()``.
`kind` | <code>[Literal](#typing.Literal)['roi', 'searchlight']</code> | Which spatial scale produced this stack — ``'roi'`` or ``'searchlight'``.



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

Provide standalone statistical functions for Adjacency matrices.

Each function takes an Adjacency instance as its first argument (`adj`).

**Methods:**

Name | Description
---- | -----------
[`cluster_summary`](#cluster-summary) | This function provides summaries of clusters within Adjacency matrices.
[`plot_label_distance`](#plot-label-distance) | Create a violin plot of within- and between-label distances.
[`plot_silhouette`](#plot-silhouette) | Create a silhouette plot.
[`r_to_z`](#r-to-z) | Apply Fisher's r to z transformation to each element of the data object.
[`similarity`](#similarity) | Calculate similarity between two Adjacency matrices.
[`stats_label_distance`](#stats-label-distance) | Calculate permutation tests on within and between label distance.
[`threshold`](#threshold) | Threshold an Adjacency instance.
[`ttest`](#ttest) | Calculate ttest across samples.
[`z_to_r`](#z-to-r) | Convert z score back into r value for each element of data object.



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

Calculate similarity between two Adjacency matrices.

The default uses Spearman correlation and a permutation test.

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
[`apply_stat`](#apply-stat) | Apply a statistical function along an axis.
`import_single_data` | Import and validate a single adjacency data matrix.
`perform_arithmetic` | Perform arithmetic operation with validation.
`test_is_single_matrix` | Check whether data represents a single matrix (1-D vector).



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
`fetch_resource`. Cached locally afterwards.

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
[`Atlas`](#atlas) | A loaded atlas — image, labels, and metadata.
[`AtlasMetadata`](#atlasmetadata) | Static description of a registered atlas.
[`ClusterReport`](#clusterreport) | Result of `BrainData.cluster_report`.

**Methods:**

Name | Description
---- | -----------
[`cluster_report_data`](#cluster-report-data) | Compute cluster report DataFrames + thresholded BrainData.
[`label_coords`](#label-coords) | Look up anatomical labels for a set of MNI mm coordinates.
[`list_atlases`](#list-atlases) | Return the sorted list of registered atlas names.
[`load_atlas`](#load-atlas) | Lazy-load an atlas by registry name.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`ATLASES`](#atlases) | <code>[dict](#dict)[[str](#str), [AtlasMetadata](#nltools.data.atlases.registry.AtlasMetadata)]</code> | 
`AtlasKind` |  | 
`DEFAULT_ATLASES` | <code>[tuple](#tuple)[[str](#str), ...]</code> | 

##### Methods

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

###### `labeling`

Coordinate-level atlas labeling.

Adapted from [atlasreader](https://github.com/miykael/atlasreader)
(BSD-3-Clause). Cite:

> Notter et al. (2019). AtlasReader. JOSS 4(34), 1257.

**Methods:**

Name | Description
---- | -----------
[`label_coords`](#label-coords) | Look up anatomical labels for a set of MNI mm coordinates.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`CoordsLike`](#coordslike) |  | 



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
`atlas` | <code>[str](#str) \| [Sequence](#collections.abc.Sequence)[[str](#str)]</code> | Atlas name or list of names from `list_atlases`. One column is added to the output per atlas. | <code>'harvard_oxford'</code>
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
[`Atlas`](#atlas) | A loaded atlas — image, labels, and metadata.

**Methods:**

Name | Description
---- | -----------
[`load_atlas`](#load-atlas) | Lazy-load an atlas by registry name.



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
`name` | <code>[str](#str)</code> | Atlas key from `list_atlases`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Atlas](#nltools.data.atlases.loading.Atlas)</code> | An `Atlas` with image, labels, and metadata loaded.

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
[`AtlasMetadata`](#atlasmetadata) | Static description of a registered atlas.

**Methods:**

Name | Description
---- | -----------
[`list_atlases`](#list-atlases) | Return the sorted list of registered atlas names.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`ATLASES`](#atlases) | <code>[dict](#dict)[[str](#str), [AtlasMetadata](#nltools.data.atlases.registry.AtlasMetadata)]</code> | 
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
<code>[list](#list)[[str](#str)]</code> | `load_atlas`.

###### `reporting`

Cluster reports — peak/cluster geometry plus atlas labels.

The peak/sub-peak geometry comes from `get_clusters_table`;
the cluster masks and mass-weighted labels are computed locally so we can
attribute every voxel of every cluster to one or more atlases.

**Classes:**

Name | Description
---- | -----------
[`ClusterReport`](#clusterreport) | Result of `BrainData.cluster_report`.

**Methods:**

Name | Description
---- | -----------
[`cluster_report_data`](#cluster-report-data) | Compute cluster report DataFrames + thresholded BrainData.



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
`peaks` | <code>[DataFrame](#polars.DataFrame)</code> | Polars DataFrame, one row per peak (incl. sub-peaks). Columns ``cluster_id``, ``x``, ``y``, ``z`` (mm), ``peak_stat``, ``volume_mm3``, ``n_voxels``, then one Utf8 column per atlas.
[`clusters`](#clusters) | <code>[DataFrame](#polars.DataFrame)</code> | Polars DataFrame, one row per cluster. Columns ``cluster_id``, ``peak_x``, ``peak_y``, ``peak_z``, ``mean_stat``, ``volume_mm3``, ``n_voxels``, then one Utf8 column per atlas (mass-weighted top regions).
`stat_img` | <code>[BrainData](#nltools.data.BrainData)</code> | BrainData with the thresholded stat map (sub-cluster voxels and clusters smaller than ``cluster_threshold`` zeroed).

**Methods:**

Name | Description
---- | -----------
[`plot`](#plot) | Render an overview glass brain + one slice figure per cluster.
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
`core` | Module-level helpers for BrainCollection.
`execution` | Parallel execution machinery for BrainCollection.
`inference` | Group-level reductions and cross-subject ops for BrainCollection.
[`io`](#io) | IO and constructors for BrainCollection.
`pipeline` | Pipeline classes for BrainCollection.

**Classes:**

Name | Description
---- | -----------
[`BrainCollection`](#braincollection) | Parallel, lazy iterator of ``BrainData`` whose API mirrors ``BrainData``.
`BrainCollectionPipeline` | Pipeline for BrainCollection with multi-subject CV support.
`BrainCollectionWorkerError` | Raised in the parent process when a worker fails inside ``_apply``.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`BUNDLE_SCHEMA_VERSION` |  | 

##### Methods

##### Modules

###### `append`

Provide standalone DesignMatrix concatenation functions.

These functions implement the append/concatenation logic extracted from
DesignMatrix methods, following the "functional core" pattern.

**Methods:**

Name | Description
---- | -----------
[`append`](#append) | Concatenate design matrices.
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
append_vertical(dm: DesignMatrix, to_append: list[DesignMatrix], keep_separate: bool, unique_cols: list[str] | None, fill_na: int | float | None, verbose: bool) -> DesignMatrix
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
`verbose` | <code>[bool](#bool)</code> | Print messages about confound separation. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with rows from all matrices.

######## `append_vertical_with_separation`

```python
append_vertical_with_separation(dm: DesignMatrix, to_append: list[DesignMatrix], unique_cols: list[str] | None, fill_na: int | float | None, verbose: bool) -> DesignMatrix
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
`verbose` | <code>[bool](#bool)</code> | Print messages about confound separation. | *required*

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

###### `diagnostics`

Diagnostic and utility functions for DesignMatrix.

**Methods:**

Name | Description
---- | -----------
[`clean`](#clean) | Remove highly correlated columns.
[`corr`](#corr) | Correlation between DesignMatrix columns as an Adjacency.
[`vif`](#vif) | Compute the variance inflation factor for each column.



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
[`events_to_dm`](#events-to-dm) | Convert a BIDS events table to boxcar regressors aligned to TRs.
`load_from_file` | Read a TSV/CSV into the frame a DesignMatrix wraps.
[`to_numpy`](#to-numpy) | Convert a DesignMatrix to a NumPy array.
[`to_pandas`](#to-pandas) | Convert DesignMatrix to pandas DataFrame.
[`write`](#write) | Write DesignMatrix to file.
`write_h5` | Write DesignMatrix to HDF5 file with metadata.



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
[`plot_corr`](#plot-corr) | Render a labeled correlation heatmap of the columns.
`plot_designmatrix` | Visualize a DesignMatrix, dispatching over ``method``.
`plot_matrix` | Render the design matrix as an SPM-style heatmap (rows=TRs, cols=regressors).
`plot_timeseries` | Plot regressor time courses as overlaid lines.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`VALID_PLOT_METHODS`](#valid-plot-methods) |  | 



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

###### `regressors`

Provide standalone regressor functions for DesignMatrix.

Each function takes a DesignMatrix as its first argument (`dm`) and returns
a new DesignMatrix with the requested transformation applied.

**Methods:**

Name | Description
---- | -----------
[`add_dct_basis`](#add-dct-basis) | Add discrete cosine transform basis functions for high-pass filtering.
[`add_poly`](#add-poly) | Add Legendre polynomial drift terms.
[`convolve`](#convolve) | Convolve columns with an HRF or custom kernel.



####### Classes

####### Functions##

###### `add_dct_basis`

```python
add_dct_basis(dm: DesignMatrix, duration: float = 180, drop: int = 0, include_constant: bool = True) -> DesignMatrix
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
[`downsample`](#downsample) | Reduce temporal resolution using Polars-native operations.
[`standardize`](#standardize) | Standardize columns using the specified method.
[`upsample`](#upsample) | Increase temporal resolution using Polars-native interpolation.
[`zscore`](#zscore) | Z-score standardize columns to mean zero and unit variance.



####### Classes

####### Functions##

###### `downsample`

```python
downsample(dm: DesignMatrix, target: float, **kwargs: float) -> DesignMatrix
```

Reduce temporal resolution using Polars-native operations.

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

Increase temporal resolution using Polars-native interpolation.

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

Z-score standardize columns to mean zero and unit variance.

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
[`copy_with`](#copy-with) | Create a new DesignMatrix with updated data and metadata.
`df_passthrough` | Resolve ``name`` on ``dm.data``; re-wrap DataFrame results for allowlisted methods.
`get_data_columns` | Get column names, optionally excluding confound regressors.
`get_metadata` | Extract metadata as dict (for copying).

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`WRAP_AS_DESIGNMATRIX`](#wrap-as-designmatrix) |  | 



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
[`Fit`](#fit) | Immutable container for model fitting results.
[`Predict`](#predict) | Immutable container for prediction / MVPA decoding results.



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
`cv_alpha_scores` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`cv_best_alpha` | <code>[float](#float) \| None</code> | 
`cv_folds` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`cv_mean_score` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`cv_predictions` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`cv_scores` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`fitted_values` | <code>[ndarray](#numpy.ndarray)</code> | 
`p_values` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`r2` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`residuals` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`scores` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`se` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`t_stats` | <code>[ndarray](#numpy.ndarray) \| None</code> | 
`weights` | <code>[ndarray](#numpy.ndarray) \| None</code> | 



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

Mirrors `Fit`: frozen, all fields default to ``None``, populated
based on the dispatch path (``method``, ``y`` vs ``X``, ``refit``) used
by `BrainData.predict`. Fields not applicable to the call remain
``None`` and are filtered from `available` and `asdict`.

**Brain-space outputs are `BrainData` objects**, not raw arrays —
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
[`accuracy_map`](#accuracy-map) | <code>[Any](#typing.Any)</code> | 
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
[`Roc`](#roc) | Roc Class



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
[`binary_outcome`](#binary-outcome) |  | 
`forced_choice` |  | 
`input_values` |  | 
`threshold_type` |  | 



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
[`SimulateGrid`](#simulategrid) | Simulate 2D grid data for testing statistical methods.
[`Simulator`](#simulator) | Simulate fMRI data with realistic spatial and temporal characteristics.



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
`t_values` |  | T-statistic values after fitting.
`p_values` |  | P-values after fitting.
`thresholded` |  | Thresholded statistical map.
`isfit` |  | Whether fit() has been called.

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
[`add_signal`](#add-signal) | Add rectangular signal to self.data
[`create_mask`](#create-mask) | Create a mask for where the signal is located in grid.
[`fit`](#fit) | Run ttest on self.data
[`plot_grid_simulation`](#plot-grid-simulation) | Create a plot of the simulations
[`run_multiple_simulations`](#run-multiple-simulations) | This method will run multiple simulations to calculate overall false positive rate
[`threshold_simulation`](#threshold-simulation) | Threshold simulation



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
[`brain_mask`](#brain-mask) |  | The brain mask image used for simulation.
`output_dir` |  | Output directory path.
`random_state` |  | Random state for reproducible simulations.

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
[`create_cov_data`](#create-cov-data) | create continuous simulated data with covariance
[`create_data`](#create-data) | create simulated data with integers
[`create_ncov_data`](#create-ncov-data) | create continuous simulated data with covariance
[`gaussian`](#gaussian) | create a 3D gaussian signal normalized to a given intensity
[`n_spheres`](#n-spheres) | generate a set of spheres in the brain mask space
[`normal_noise`](#normal-noise) | produce a normal noise distribution for all all points in the brain mask
[`sphere`](#sphere) | create a sphere of given radius at some point p in the brain mask
[`to_nifti`](#to-nifti) | convert a numpy matrix to the nifti format and assign to it the brain_mask's affine matrix



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