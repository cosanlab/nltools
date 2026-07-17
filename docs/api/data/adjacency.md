(data-adjacency-adjacency)=
## `Adjacency`

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
[`append`](#data-adjacency-append) | Append data to an Adjacency instance.
[`bootstrap`](#data-adjacency-bootstrap) | Bootstrap statistics using efficient online algorithms.
[`cluster_summary`](#data-adjacency-cluster-summary) | Provide summaries of clusters within Adjacency matrices.
[`copy`](#data-adjacency-copy) | Create a copy of Adjacency object.
[`distance`](#data-adjacency-distance) | Calculate distance between images within an Adjacency() instance.
[`distance_to_similarity`](#data-adjacency-distance-to-similarity) | Convert distance matrix to similarity matrix.
[`generate_permutations`](#data-adjacency-generate-permutations) | Generate permuted versions of an Adjacency instance lazily.
[`mean`](#data-adjacency-mean) | Calculate mean of Adjacency.
[`median`](#data-adjacency-median) | Calculate median of Adjacency.
[`plot`](#data-adjacency-plot) | Create a heatmap of an Adjacency matrix.
[`plot_label_distance`](#data-adjacency-plot-label-distance) | Create a violin plot of within- and between-label distances.
[`plot_mds`](#data-adjacency-plot-mds) | Plot multidimensional scaling.
[`plot_silhouette`](#data-adjacency-plot-silhouette) | Create a silhouette plot.
[`r_to_z`](#data-adjacency-r-to-z) | Apply Fisher's r-to-z transformation to each data element.
[`regress`](#data-adjacency-regress) | Run a regression on an adjacency instance.
[`similarity`](#data-adjacency-similarity) | Calculate similarity between two Adjacency matrices.
[`social_relations_model`](#data-adjacency-social-relations-model) | Estimate the social relations model from a matrix for a round-robin design.
[`squareform`](#data-adjacency-squareform) | Convert adjacency data back to square form.
[`stats_label_distance`](#data-adjacency-stats-label-distance) | Calculate permutation tests on within and between label distance.
[`std`](#data-adjacency-std) | Calculate standard deviation of Adjacency.
[`sum`](#data-adjacency-sum) | Calculate sum of Adjacency.
[`threshold`](#data-adjacency-threshold) | Threshold an Adjacency instance.
[`to_brain`](#data-adjacency-to-brain) | Project per-matrix scalars back to voxel-space `BrainData`.
[`to_graph`](#data-adjacency-to-graph) | Convert a single Adjacency matrix into a NetworkX graph.
[`to_square`](#data-adjacency-to-square) | Convert adjacency back to square matrix format.
[`ttest`](#data-adjacency-ttest) | Calculate ttest across samples.
[`write`](#data-adjacency-write) | Write out Adjacency object to csv file.
[`z_to_r`](#data-adjacency-z-to-r) | Convert each z score back into an r value.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`Y` | <code>[DataFrame](#polars.DataFrame)</code> | Training labels as a polars DataFrame (possibly empty).
`data` |  | 
`is_empty` | <code>[bool](#bool)</code> | Check if Adjacency object is empty.
`is_single_matrix` |  | 
`issymmetric` |  | 
`labels` |  | 
`matrix_type` |  | 
`n_nodes` |  | Return the number of nodes in the adjacency matrix.
`shape` |  | Return the logical shape of the adjacency matrix.
`spatial_scale` | <code>[SpatialScale](#nltools.data.adjacency.spatial.SpatialScale) \| None</code> | 
`vector_shape` |  | Return shape of internal vectorized representation.

### Methods

(data-adjacency-append)=
#### `append`

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

(data-adjacency-bootstrap)=
#### `bootstrap`

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

(data-adjacency-cluster-summary)=
#### `cluster_summary`

```python
cluster_summary(*, clusters = None, metric = 'mean', summary = 'within')
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

(data-adjacency-copy)=
#### `copy`

```python
copy()
```

Create a copy of Adjacency object.

(data-adjacency-distance)=
#### `distance`

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

(data-adjacency-distance-to-similarity)=
#### `distance_to_similarity`

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

(data-adjacency-generate-permutations)=
#### `generate_permutations`

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

(data-adjacency-mean)=
#### `mean`

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

(data-adjacency-median)=
#### `median`

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

(data-adjacency-plot)=
#### `plot`

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

(data-adjacency-plot-label-distance)=
#### `plot_label_distance`

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

(data-adjacency-plot-mds)=
#### `plot_mds`

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

(data-adjacency-plot-silhouette)=
#### `plot_silhouette`

```python
plot_silhouette(*, labels = None, ax = None, permutation_test = True, n_permute = 5000, **kwargs)
```

Create a silhouette plot.

(data-adjacency-r-to-z)=
#### `r_to_z`

```python
r_to_z()
```

Apply Fisher's r-to-z transformation to each data element.

(data-adjacency-regress)=
#### `regress`

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

(data-adjacency-similarity)=
#### `similarity`

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

(data-adjacency-social-relations-model)=
#### `social_relations_model`

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

(data-adjacency-squareform)=
#### `squareform`

```python
squareform()
```

Convert adjacency data back to square form.

(data-adjacency-stats-label-distance)=
#### `stats_label_distance`

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

(data-adjacency-std)=
#### `std`

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

(data-adjacency-sum)=
#### `sum`

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

(data-adjacency-threshold)=
#### `threshold`

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

(data-adjacency-to-brain)=
#### `to_brain`

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

(data-adjacency-to-graph)=
#### `to_graph`

```python
to_graph()
```

Convert a single Adjacency matrix into a NetworkX graph.

This currently works only for ``single_matrix``.

(data-adjacency-to-square)=
#### `to_square`

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

(data-adjacency-ttest)=
#### `ttest`

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

(data-adjacency-write)=
#### `write`

```python
write(file_name, method = 'long')
```

Write out Adjacency object to csv file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>[str](#str)</code> | name of file name to write | *required*
`method` | <code>[str](#str)</code> | method to write out data ['long','square'] | <code>'long'</code>

(data-adjacency-z-to-r)=
#### `z_to_r`

```python
z_to_r()
```

Convert each z score back into an r value.

