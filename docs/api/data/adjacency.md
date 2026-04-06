## `nltools.data.adjacency`

This data class is for working with similarity/dissimilarity matrices

**Modules:**

Name | Description
---- | -----------
[`io`](#nltools.data.adjacency.io) | I/O functions for Adjacency objects.
[`modeling`](#nltools.data.adjacency.modeling) | Standalone modeling/inference functions for Adjacency matrices.
[`plotting`](#nltools.data.adjacency.plotting) | Plotting functions for Adjacency matrices.
[`stats`](#nltools.data.adjacency.stats) | Standalone statistical functions for Adjacency matrices.
[`utils`](#nltools.data.adjacency.utils) | Shared helpers for Adjacency submodules.

**Classes:**

Name | Description
---- | -----------
[`Adjacency`](#nltools.data.adjacency.Adjacency) | Adjacency is a class to represent Adjacency matrices as a vector rather

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`MAX_INT`](#nltools.data.adjacency.MAX_INT) |  | 
[`nx`](#nltools.data.adjacency.nx) |  | 
[`tables`](#nltools.data.adjacency.tables) |  | 



### Attributes#### `nltools.data.adjacency.MAX_INT`

```python
MAX_INT = np.iinfo(np.int32).max
```

#### `nltools.data.adjacency.nx`

```python
nx = attempt_to_import('networkx', 'nx')
```

#### `nltools.data.adjacency.tables`

```python
tables = attempt_to_import('tables')
```



### Classes#### `nltools.data.adjacency.Adjacency`

```python
Adjacency(data = None, Y = None, matrix_type = None, labels = None, **kwargs)
```

Bases: <code>[object](#object)</code>

Adjacency is a class to represent Adjacency matrices as a vector rather
than a 2-dimensional matrix. This makes it easier to perform data
manipulation and analyses.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | pandas data instance or list of files | <code>None</code>
`matrix_type` |  | (str) type of matrix.  Possible values include:         ['distance','similarity','directed','distance_flat',         'similarity_flat','directed_flat'] | <code>None</code>
`Y` |  | Pandas DataFrame of training labels | <code>None</code>
`**kwargs` |  | Additional keyword arguments | <code>{}</code>

**Functions:**

Name | Description
---- | -----------
[`append`](#nltools.data.adjacency.Adjacency.append) | Append data to Adjacency instance
[`bootstrap`](#nltools.data.adjacency.Adjacency.bootstrap) | Bootstrap statistics using efficient online algorithms.
[`cluster_summary`](#nltools.data.adjacency.Adjacency.cluster_summary) | Provide summaries of clusters within Adjacency matrices.
[`copy`](#nltools.data.adjacency.Adjacency.copy) | Create a copy of Adjacency object.
[`distance`](#nltools.data.adjacency.Adjacency.distance) | Calculate distance between images within an Adjacency() instance.
[`distance_to_similarity`](#nltools.data.adjacency.Adjacency.distance_to_similarity) | Convert distance matrix to similarity matrix.
[`generate_permutations`](#nltools.data.adjacency.Adjacency.generate_permutations) | Generate n_perm permutated versions of Adjacency in a lazy fashion.
[`mean`](#nltools.data.adjacency.Adjacency.mean) | Calculate mean of Adjacency.
[`median`](#nltools.data.adjacency.Adjacency.median) | Calculate median of Adjacency.
[`plot`](#nltools.data.adjacency.Adjacency.plot) | Create Heatmap of Adjacency Matrix
[`plot_label_distance`](#nltools.data.adjacency.Adjacency.plot_label_distance) | Create a violin plot indicating within and between label distance
[`plot_mds`](#nltools.data.adjacency.Adjacency.plot_mds) | Plot Multidimensional Scaling
[`plot_silhouette`](#nltools.data.adjacency.Adjacency.plot_silhouette) | Create a silhouette plot
[`r_to_z`](#nltools.data.adjacency.Adjacency.r_to_z) | Apply Fisher's r to z transformation to each element of the data
[`regress`](#nltools.data.adjacency.Adjacency.regress) | Run a regression on an adjacency instance.
[`similarity`](#nltools.data.adjacency.Adjacency.similarity) | Calculate similarity between two Adjacency matrices. Default is to use spearman
[`social_relations_model`](#nltools.data.adjacency.Adjacency.social_relations_model) | Estimate the social relations model from a matrix for a round-robin design.
[`squareform`](#nltools.data.adjacency.Adjacency.squareform) | Convert adjacency back to squareform
[`stats_label_distance`](#nltools.data.adjacency.Adjacency.stats_label_distance) | Calculate permutation tests on within and between label distance.
[`std`](#nltools.data.adjacency.Adjacency.std) | Calculate standard deviation of Adjacency.
[`sum`](#nltools.data.adjacency.Adjacency.sum) | Calculate sum of Adjacency.
[`threshold`](#nltools.data.adjacency.Adjacency.threshold) | Threshold Adjacency instance. Provide upper and lower values or
[`to_graph`](#nltools.data.adjacency.Adjacency.to_graph) | Convert Adjacency into networkx graph.  only works on
[`to_square`](#nltools.data.adjacency.Adjacency.to_square) | Convert adjacency back to square matrix format.
[`ttest`](#nltools.data.adjacency.Adjacency.ttest) | Calculate ttest across samples.
[`write`](#nltools.data.adjacency.Adjacency.write) | Write out Adjacency object to csv file.
[`z_to_r`](#nltools.data.adjacency.Adjacency.z_to_r) | Convert z score back into r value for each element of data object

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`Y`](#nltools.data.adjacency.Adjacency.Y) |  | 
[`data`](#nltools.data.adjacency.Adjacency.data) |  | 
[`is_empty`](#nltools.data.adjacency.Adjacency.is_empty) | <code>[bool](#bool)</code> | Check if Adjacency object is empty.
[`is_single_matrix`](#nltools.data.adjacency.Adjacency.is_single_matrix) |  | 
[`issymmetric`](#nltools.data.adjacency.Adjacency.issymmetric) |  | 
[`labels`](#nltools.data.adjacency.Adjacency.labels) |  | 
[`matrix_type`](#nltools.data.adjacency.Adjacency.matrix_type) |  | 
[`n_nodes`](#nltools.data.adjacency.Adjacency.n_nodes) |  | Return the number of nodes in the adjacency matrix.
[`shape`](#nltools.data.adjacency.Adjacency.shape) |  | Return the logical shape of the adjacency matrix.
[`vector_shape`](#nltools.data.adjacency.Adjacency.vector_shape) |  | Return shape of internal vectorized representation.



##### Attributes###### `nltools.data.adjacency.Adjacency.Y`

```python
Y = f['Y']
```

###### `nltools.data.adjacency.Adjacency.data`

```python
data = np.array(f.root['data'])
```

###### `nltools.data.adjacency.Adjacency.is_empty`

```python
is_empty: bool
```

Check if Adjacency object is empty.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`bool` | <code>[bool](#bool)</code> | True if the adjacency matrix is empty, False otherwise.

###### `nltools.data.adjacency.Adjacency.is_single_matrix`

```python
is_single_matrix = f['is_single_matrix'][]
```

###### `nltools.data.adjacency.Adjacency.issymmetric`

```python
issymmetric = f['issymmetric'][]
```

###### `nltools.data.adjacency.Adjacency.labels`

```python
labels = list(f['labels'])
```

###### `nltools.data.adjacency.Adjacency.matrix_type`

```python
matrix_type = f['matrix_type'][].decode()
```

###### `nltools.data.adjacency.Adjacency.n_nodes`

```python
n_nodes
```

Return the number of nodes in the adjacency matrix.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`int` |  | Number of nodes (n) for an (n, n) matrix.

###### `nltools.data.adjacency.Adjacency.shape`

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

###### `nltools.data.adjacency.Adjacency.vector_shape`

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



##### Functions###### `nltools.data.adjacency.Adjacency.append`

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

###### `nltools.data.adjacency.Adjacency.bootstrap`

```python
bootstrap(stat, n_samples = 5000, save_boots = False, n_jobs = -1, random_state = None, percentiles = (2.5, 97.5))
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
`n_jobs` |  | (int) Number of CPU cores for parallelization. -1 means all CPUs. | <code>-1</code>
`random_state` |  | (int, optional) Random seed for reproducibility | <code>None</code>
`percentiles` |  | (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5) | <code>(2.5, 97.5)</code>

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

###### `nltools.data.adjacency.Adjacency.cluster_summary`

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

###### `nltools.data.adjacency.Adjacency.copy`

```python
copy()
```

Create a copy of Adjacency object.

###### `nltools.data.adjacency.Adjacency.distance`

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

###### `nltools.data.adjacency.Adjacency.distance_to_similarity`

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

###### `nltools.data.adjacency.Adjacency.generate_permutations`

```python
generate_permutations(n_perm, random_state = None)
```

Generate n_perm permutated versions of Adjacency in a lazy fashion.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_perm` | <code>[int](#int)</code> | number of permutations | *required*
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

###### `nltools.data.adjacency.Adjacency.mean`

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

###### `nltools.data.adjacency.Adjacency.median`

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

###### `nltools.data.adjacency.Adjacency.plot`

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

###### `nltools.data.adjacency.Adjacency.plot_label_distance`

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

###### `nltools.data.adjacency.Adjacency.plot_mds`

```python
plot_mds(n_components = 2, metric = True, labels = None, labels_color = None, cmap = None, n_jobs = -1, view = (30, 20), figsize = None, ax = None, *args, **kwargs)
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
`n_jobs` |  | (int) Number of parallel jobs | <code>-1</code>
`view` |  | (tuple) view for 3-Dimensional plot; default (30,20) | <code>(30, 20)</code>
`figsize` |  | (list) figure size; default [12, 8] | <code>None</code>
`ax` |  | matplotlib axis handle | <code>None</code>

###### `nltools.data.adjacency.Adjacency.plot_silhouette`

```python
plot_silhouette(labels = None, ax = None, permutation_test = True, n_permute = 5000, **kwargs)
```

Create a silhouette plot

###### `nltools.data.adjacency.Adjacency.r_to_z`

```python
r_to_z()
```

Apply Fisher's r to z transformation to each element of the data
object.

###### `nltools.data.adjacency.Adjacency.regress`

```python
regress(X, mode = 'ols', **kwargs)
```

Run a regression on an adjacency instance.
You can decompose an adjacency instance with another adjacency instance.
You can also decompose each pixel by passing a design_matrix instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` |  | Design matrix can be an Adjacency or DesignMatrix instance | *required*
`mode` |  | type of regression (default: ols) - only 'ols' is currently supported | <code>'ols'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`stats` |  | (dict) dictionary of stats outputs.

###### `nltools.data.adjacency.Adjacency.similarity`

```python
similarity(data, plot = False, perm_type = '2d', n_permute = 5000, metric = 'spearman', ignore_diagonal = False, nan_policy = 'omit', **kwargs)
```

Calculate similarity between two Adjacency matrices. Default is to use spearman
correlation and permutation test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Adjacency](#nltools.data.adjacency.Adjacency) or [array](#array)</code> | Adjacency data, or 1-d array same size as self.data | *required*
`perm_type` |  | (str) '1d','2d', or None | <code>'2d'</code>
`metric` |  | (str) 'spearman','pearson','kendall' | <code>'spearman'</code>
`ignore_diagonal` |  | (bool) only applies to 'directed' Adjacency types using perm_type=None or perm_type='1d' | <code>False</code>
`nan_policy` |  | (str) How to handle NaN values. Options: - 'omit': Remove NaN values pairwise before computing correlation (default) - 'propagate': Allow NaN to propagate through calculations - 'raise': Raise an error if NaN values are present | <code>'omit'</code>

###### `nltools.data.adjacency.Adjacency.social_relations_model`

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

###### `nltools.data.adjacency.Adjacency.squareform`

```python
squareform()
```

Convert adjacency back to squareform

###### `nltools.data.adjacency.Adjacency.stats_label_distance`

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

###### `nltools.data.adjacency.Adjacency.std`

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

###### `nltools.data.adjacency.Adjacency.sum`

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

###### `nltools.data.adjacency.Adjacency.threshold`

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

###### `nltools.data.adjacency.Adjacency.to_graph`

```python
to_graph()
```

Convert Adjacency into networkx graph.  only works on
single_matrix for now.

###### `nltools.data.adjacency.Adjacency.to_square`

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

###### `nltools.data.adjacency.Adjacency.ttest`

```python
ttest(permutation = False, **kwargs)
```

Calculate ttest across samples.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`permutation` |  | (bool) Run ttest as permutation. Note this can be very slow. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (dict) contains Adjacency instances of t values (or mean if  running permutation) and Adjacency instance of p values.

###### `nltools.data.adjacency.Adjacency.write`

```python
write(file_name, method = 'long')
```

Write out Adjacency object to csv file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>[str](#str)</code> | name of file name to write | *required*
`method` | <code>[str](#str)</code> | method to write out data ['long','square'] | <code>'long'</code>

###### `nltools.data.adjacency.Adjacency.z_to_r`

```python
z_to_r()
```

Convert z score back into r value for each element of data object



### Functions

### Modules#### `nltools.data.adjacency.io`

I/O functions for Adjacency objects.

**Functions:**

Name | Description
---- | -----------
[`to_graph`](#nltools.data.adjacency.io.to_graph) | Convert Adjacency into networkx graph.
[`write`](#nltools.data.adjacency.io.write) | Write out Adjacency object to csv file.



##### Functions###### `nltools.data.adjacency.io.to_graph`

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

###### `nltools.data.adjacency.io.write`

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

#### `nltools.data.adjacency.modeling`

Standalone modeling/inference functions for Adjacency matrices.

Each function takes an Adjacency instance as its first argument (`adj`).

**Functions:**

Name | Description
---- | -----------
[`bootstrap`](#nltools.data.adjacency.modeling.bootstrap) | Bootstrap statistics using efficient online algorithms.
[`convert_bootstrap_results_to_adjacency`](#nltools.data.adjacency.modeling.convert_bootstrap_results_to_adjacency) | Convert bootstrap results dictionary to Adjacency format.
[`generate_permutations`](#nltools.data.adjacency.modeling.generate_permutations) | Generate n_perm permutated versions of Adjacency in a lazy fashion. Useful for iterating against.
[`regress`](#nltools.data.adjacency.modeling.regress) | Run a regression on an adjacency instance.
[`social_relations_model`](#nltools.data.adjacency.modeling.social_relations_model) | Estimate the social relations model from a matrix for a round-robin design.



##### Functions###### `nltools.data.adjacency.modeling.bootstrap`

```python
bootstrap(adj, stat, n_samples = 5000, save_boots = False, n_jobs = -1, random_state = None, percentiles = (2.5, 97.5))
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
`n_jobs` |  | (int) Number of CPU cores for parallelization. -1 means all CPUs. | <code>-1</code>
`random_state` |  | (int, optional) Random seed for reproducibility | <code>None</code>
`percentiles` |  | (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5) | <code>(2.5, 97.5)</code>

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

###### `nltools.data.adjacency.modeling.convert_bootstrap_results_to_adjacency`

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

###### `nltools.data.adjacency.modeling.generate_permutations`

```python
generate_permutations(adj, n_perm, random_state = None)
```

Generate n_perm permutated versions of Adjacency in a lazy fashion. Useful for iterating against.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` |  | (Adjacency) Adjacency instance | *required*
`n_perm` | <code>[int](#int)</code> | number of permutations | *required*
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

###### `nltools.data.adjacency.modeling.regress`

```python
regress(adj, X, mode = 'ols', **kwargs)
```

Run a regression on an adjacency instance.
You can decompose an adjacency instance with another adjacency instance.
You can also decompose each pixel by passing a design_matrix instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` |  | (Adjacency) Adjacency instance | *required*
`X` |  | Design matrix can be an Adjacency or DesignMatrix instance | *required*
`mode` |  | type of regression (default: ols) - only 'ols' is currently supported | <code>'ols'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`stats` |  | (dict) dictionary of stats outputs.

###### `nltools.data.adjacency.modeling.social_relations_model`

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

#### `nltools.data.adjacency.plotting`

Plotting functions for Adjacency matrices.

**Functions:**

Name | Description
---- | -----------
[`plot`](#nltools.data.adjacency.plotting.plot) | Create Heatmap of Adjacency Matrix.
[`plot_mds`](#nltools.data.adjacency.plotting.plot_mds) | Plot Multidimensional Scaling.



##### Functions###### `nltools.data.adjacency.plotting.plot`

```python
plot(adj, limit = 3, axes = None, *args, **kwargs)
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

###### `nltools.data.adjacency.plotting.plot_mds`

```python
plot_mds(adj, n_components = 2, metric = True, labels = None, labels_color = None, cmap = None, n_jobs = -1, view = (30, 20), figsize = None, ax = None, *args, **kwargs)
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
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>-1</code>
`view` | <code>[tuple](#tuple)</code> | View for 3-Dimensional plot. Default (30, 20). | <code>(30, 20)</code>
`figsize` | <code>[list](#list)</code> | Figure size. Default [12, 8]. | <code>None</code>
`ax` |  | Matplotlib axis handle. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | None

#### `nltools.data.adjacency.stats`

Standalone statistical functions for Adjacency matrices.

Each function takes an Adjacency instance as its first argument (`adj`).

**Functions:**

Name | Description
---- | -----------
[`cluster_summary`](#nltools.data.adjacency.stats.cluster_summary) | This function provides summaries of clusters within Adjacency matrices.
[`plot_label_distance`](#nltools.data.adjacency.stats.plot_label_distance) | Create a violin plot indicating within and between label distance
[`plot_silhouette`](#nltools.data.adjacency.stats.plot_silhouette) | Create a silhouette plot.
[`r_to_z`](#nltools.data.adjacency.stats.r_to_z) | Apply Fisher's r to z transformation to each element of the data object.
[`similarity`](#nltools.data.adjacency.stats.similarity) | Calculate similarity between two Adjacency matrices. Default is to use spearman
[`stats_label_distance`](#nltools.data.adjacency.stats.stats_label_distance) | Calculate permutation tests on within and between label distance.
[`threshold`](#nltools.data.adjacency.stats.threshold) | Threshold Adjacency instance. Provide upper and lower values or
[`ttest`](#nltools.data.adjacency.stats.ttest) | Calculate ttest across samples.
[`z_to_r`](#nltools.data.adjacency.stats.z_to_r) | Convert z score back into r value for each element of data object.



##### Functions###### `nltools.data.adjacency.stats.cluster_summary`

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

###### `nltools.data.adjacency.stats.plot_label_distance`

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

###### `nltools.data.adjacency.stats.plot_silhouette`

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

###### `nltools.data.adjacency.stats.r_to_z`

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

###### `nltools.data.adjacency.stats.similarity`

```python
similarity(adj, data, plot = False, perm_type = '2d', n_permute = 5000, metric = 'spearman', ignore_diagonal = False, nan_policy = 'omit', **kwargs)
```

Calculate similarity between two Adjacency matrices. Default is to use spearman
correlation and permutation test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency instance. | *required*
`data` | <code>[Adjacency](#nltools.data.adjacency.Adjacency) or [array](#array)</code> | Adjacency data, or 1-d array same size as adj.data. | *required*
`plot` | <code>[bool](#bool)</code> | If True, plot stacked adjacency matrices. Default False. | <code>False</code>
`perm_type` | <code>[str](#str)</code> | '1d', '2d', or None. | <code>'2d'</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations. Default 5000. | <code>5000</code>
`metric` | <code>[str](#str)</code> | 'spearman', 'pearson', or 'kendall'. | <code>'spearman'</code>
`ignore_diagonal` | <code>[bool](#bool)</code> | Only applies to 'directed' Adjacency types using perm_type=None or perm_type='1d'. | <code>False</code>
`nan_policy` | <code>[str](#str)</code> | How to handle NaN values. Options: - 'omit': Remove NaN values pairwise before computing correlation (default) - 'propagate': Allow NaN to propagate through calculations - 'raise': Raise an error if NaN values are present | <code>'omit'</code>

**Returns:**

Type | Description
---- | -----------
 | dict or list: Correlation result dict with keys 'r' and 'p', or a list of such dicts when adj contains multiple matrices.

###### `nltools.data.adjacency.stats.stats_label_distance`

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

###### `nltools.data.adjacency.stats.threshold`

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

###### `nltools.data.adjacency.stats.ttest`

```python
ttest(adj, permutation = False, **kwargs)
```

Calculate ttest across samples.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#nltools.data.adjacency.Adjacency)</code> | Adjacency instance (must contain multiple matrices) | *required*
`permutation` |  | (bool) Run ttest as permutation. Note this can be very slow. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (dict) contains Adjacency instances of t values (or mean if  running permutation) and Adjacency instance of p values.

###### `nltools.data.adjacency.stats.z_to_r`

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

#### `nltools.data.adjacency.utils`

Shared helpers for Adjacency submodules.

These are internal utilities used by the facade and submodules — not part of the
public API.

**Functions:**

Name | Description
---- | -----------
[`apply_stat`](#nltools.data.adjacency.utils.apply_stat) | Apply a statistical function along an axis.
[`import_single_data`](#nltools.data.adjacency.utils.import_single_data) | Import and validate a single adjacency data matrix.
[`perform_arithmetic`](#nltools.data.adjacency.utils.perform_arithmetic) | Perform arithmetic operation with validation.
[`test_is_single_matrix`](#nltools.data.adjacency.utils.test_is_single_matrix) | Check whether data represents a single matrix (1-D vector).



##### Functions###### `nltools.data.adjacency.utils.apply_stat`

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

###### `nltools.data.adjacency.utils.import_single_data`

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

###### `nltools.data.adjacency.utils.perform_arithmetic`

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

###### `nltools.data.adjacency.utils.test_is_single_matrix`

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

