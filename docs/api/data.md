## `nltools.data`

nltools data types.

**Modules:**

Name | Description
---- | -----------
[`adjacency`](#nltools.data.adjacency) | This data class is for working with similarity/dissimilarity matrices
[`braindata`](#nltools.data.braindata) | NeuroLearn Brain Data
[`collection`](#nltools.data.collection) | BrainCollection: Multi-subject brain data container.
[`designmatrix`](#nltools.data.designmatrix) | DesignMatrix - Polars-based design matrix for neuroimaging analysis
[`fitresults`](#nltools.data.fitresults) | Immutable container for model fitting results.
[`roc`](#nltools.data.roc) | NeuroLearn Analysis Tools
[`simulator`](#nltools.data.simulator) | NeuroLearn Simulator Tools

**Classes:**

Name | Description
---- | -----------
[`Adjacency`](#nltools.data.Adjacency) | Adjacency is a class to represent Adjacency matrices as a vector rather
[`BrainCollection`](#nltools.data.BrainCollection) | Collection of brain images with tensor-like operations.
[`BrainData`](#nltools.data.BrainData) | BrainData is a class to represent neuroimaging data in python as a vector
[`DesignMatrix`](#nltools.data.DesignMatrix) | Polars-based design matrix for experimental designs in neuroimaging.
[`Fit`](#nltools.data.Fit) | Immutable container for model fitting results.
[`Roc`](#nltools.data.Roc) | Roc Class
[`SimulateGrid`](#nltools.data.SimulateGrid) | Simulate 2D grid data for testing statistical methods.
[`Simulator`](#nltools.data.Simulator) | Simulate fMRI data with realistic spatial and temporal characteristics.



### Classes#### `nltools.data.Adjacency`

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
[`append`](#nltools.data.Adjacency.append) | Append data to Adjacency instance
[`bootstrap`](#nltools.data.Adjacency.bootstrap) | Bootstrap statistics using efficient online algorithms.
[`cluster_summary`](#nltools.data.Adjacency.cluster_summary) | Provide summaries of clusters within Adjacency matrices.
[`copy`](#nltools.data.Adjacency.copy) | Create a copy of Adjacency object.
[`distance`](#nltools.data.Adjacency.distance) | Calculate distance between images within an Adjacency() instance.
[`distance_to_similarity`](#nltools.data.Adjacency.distance_to_similarity) | Convert distance matrix to similarity matrix.
[`generate_permutations`](#nltools.data.Adjacency.generate_permutations) | Generate n_perm permutated versions of Adjacency in a lazy fashion.
[`mean`](#nltools.data.Adjacency.mean) | Calculate mean of Adjacency.
[`median`](#nltools.data.Adjacency.median) | Calculate median of Adjacency.
[`plot`](#nltools.data.Adjacency.plot) | Create Heatmap of Adjacency Matrix
[`plot_label_distance`](#nltools.data.Adjacency.plot_label_distance) | Create a violin plot indicating within and between label distance
[`plot_mds`](#nltools.data.Adjacency.plot_mds) | Plot Multidimensional Scaling
[`plot_silhouette`](#nltools.data.Adjacency.plot_silhouette) | Create a silhouette plot
[`r_to_z`](#nltools.data.Adjacency.r_to_z) | Apply Fisher's r to z transformation to each element of the data
[`regress`](#nltools.data.Adjacency.regress) | Run a regression on an adjacency instance.
[`similarity`](#nltools.data.Adjacency.similarity) | Calculate similarity between two Adjacency matrices. Default is to use spearman
[`social_relations_model`](#nltools.data.Adjacency.social_relations_model) | Estimate the social relations model from a matrix for a round-robin design.
[`squareform`](#nltools.data.Adjacency.squareform) | Convert adjacency back to squareform
[`stats_label_distance`](#nltools.data.Adjacency.stats_label_distance) | Calculate permutation tests on within and between label distance.
[`std`](#nltools.data.Adjacency.std) | Calculate standard deviation of Adjacency.
[`sum`](#nltools.data.Adjacency.sum) | Calculate sum of Adjacency.
[`threshold`](#nltools.data.Adjacency.threshold) | Threshold Adjacency instance. Provide upper and lower values or
[`to_graph`](#nltools.data.Adjacency.to_graph) | Convert Adjacency into networkx graph.  only works on
[`to_square`](#nltools.data.Adjacency.to_square) | Convert adjacency back to square matrix format.
[`ttest`](#nltools.data.Adjacency.ttest) | Calculate ttest across samples.
[`write`](#nltools.data.Adjacency.write) | Write out Adjacency object to csv file.
[`z_to_r`](#nltools.data.Adjacency.z_to_r) | Convert z score back into r value for each element of data object

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`Y`](#nltools.data.Adjacency.Y) |  | 
[`data`](#nltools.data.Adjacency.data) |  | 
[`is_empty`](#nltools.data.Adjacency.is_empty) | <code>[bool](#bool)</code> | Check if Adjacency object is empty.
[`is_single_matrix`](#nltools.data.Adjacency.is_single_matrix) |  | 
[`issymmetric`](#nltools.data.Adjacency.issymmetric) |  | 
[`labels`](#nltools.data.Adjacency.labels) |  | 
[`matrix_type`](#nltools.data.Adjacency.matrix_type) |  | 
[`n_nodes`](#nltools.data.Adjacency.n_nodes) |  | Return the number of nodes in the adjacency matrix.
[`shape`](#nltools.data.Adjacency.shape) |  | Return the logical shape of the adjacency matrix.
[`vector_shape`](#nltools.data.Adjacency.vector_shape) |  | Return shape of internal vectorized representation.



##### Attributes###### `nltools.data.Adjacency.Y`

```python
Y = f['Y']
```

###### `nltools.data.Adjacency.data`

```python
data = np.array(f.root['data'])
```

###### `nltools.data.Adjacency.is_empty`

```python
is_empty: bool
```

Check if Adjacency object is empty.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`bool` | <code>[bool](#bool)</code> | True if the adjacency matrix is empty, False otherwise.

###### `nltools.data.Adjacency.is_single_matrix`

```python
is_single_matrix = f['is_single_matrix'][]
```

###### `nltools.data.Adjacency.issymmetric`

```python
issymmetric = f['issymmetric'][]
```

###### `nltools.data.Adjacency.labels`

```python
labels = list(f['labels'])
```

###### `nltools.data.Adjacency.matrix_type`

```python
matrix_type = f['matrix_type'][].decode()
```

###### `nltools.data.Adjacency.n_nodes`

```python
n_nodes
```

Return the number of nodes in the adjacency matrix.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`int` |  | Number of nodes (n) for an (n, n) matrix.

###### `nltools.data.Adjacency.shape`

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

###### `nltools.data.Adjacency.vector_shape`

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



##### Functions###### `nltools.data.Adjacency.append`

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

###### `nltools.data.Adjacency.bootstrap`

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

###### `nltools.data.Adjacency.cluster_summary`

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

###### `nltools.data.Adjacency.copy`

```python
copy()
```

Create a copy of Adjacency object.

###### `nltools.data.Adjacency.distance`

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

###### `nltools.data.Adjacency.distance_to_similarity`

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

###### `nltools.data.Adjacency.generate_permutations`

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

###### `nltools.data.Adjacency.mean`

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

###### `nltools.data.Adjacency.median`

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

###### `nltools.data.Adjacency.plot`

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

###### `nltools.data.Adjacency.plot_label_distance`

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

###### `nltools.data.Adjacency.plot_mds`

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

###### `nltools.data.Adjacency.plot_silhouette`

```python
plot_silhouette(labels = None, ax = None, permutation_test = True, n_permute = 5000, **kwargs)
```

Create a silhouette plot

###### `nltools.data.Adjacency.r_to_z`

```python
r_to_z()
```

Apply Fisher's r to z transformation to each element of the data
object.

###### `nltools.data.Adjacency.regress`

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

###### `nltools.data.Adjacency.similarity`

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

###### `nltools.data.Adjacency.social_relations_model`

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

###### `nltools.data.Adjacency.squareform`

```python
squareform()
```

Convert adjacency back to squareform

###### `nltools.data.Adjacency.stats_label_distance`

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

###### `nltools.data.Adjacency.std`

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

###### `nltools.data.Adjacency.sum`

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

###### `nltools.data.Adjacency.threshold`

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

###### `nltools.data.Adjacency.to_graph`

```python
to_graph()
```

Convert Adjacency into networkx graph.  only works on
single_matrix for now.

###### `nltools.data.Adjacency.to_square`

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

###### `nltools.data.Adjacency.ttest`

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

###### `nltools.data.Adjacency.write`

```python
write(file_name, method = 'long')
```

Write out Adjacency object to csv file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>[str](#str)</code> | name of file name to write | *required*
`method` | <code>[str](#str)</code> | method to write out data ['long','square'] | <code>'long'</code>

###### `nltools.data.Adjacency.z_to_r`

```python
z_to_r()
```

Convert z score back into r value for each element of data object

#### `nltools.data.BrainCollection`

```python
BrainCollection(items: list[Path | str | 'BrainData'], mask: nib.Nifti1Image | Path | str, metadata: pd.DataFrame | None = None, lazy: bool = True)
```

Collection of brain images with tensor-like operations.

BrainCollection provides a container for multiple brain images (e.g., multiple
subjects or runs) with numpy-style indexing and axis operations. It supports
lazy loading for memory efficiency and integrates with pybids for BIDS datasets.

<details class="shape-semantics" open markdown="1">
<summary>(n_images, n_observations, n_voxels)</summary>

- axis 0: images (subjects, runs, etc.)
- axis 1: observations (timepoints, TRs)
- axis 2: voxels (spatial locations)

</details>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`items` | <code>[list](#list)[[Path](#pathlib.Path) \| [str](#str) \| 'BrainData']</code> | List of file paths, BrainData objects, or mix of both. Paths are loaded lazily by default. | *required*
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str)</code> | Brain mask. Required. Can be: - nibabel Nifti1Image - Path to mask file - Template name (e.g., '2mm-MNI152-2009c') | *required*
`metadata` | <code>[DataFrame](#pandas.DataFrame) \| None</code> | Optional DataFrame with per-image metadata (subject, session, etc.). Index should match items order. | <code>None</code>
`lazy` | <code>[bool](#bool)</code> | If True (default), paths are not loaded until accessed. | <code>True</code>

**Examples:**

```pycon
>>> # Create from paths (lazy loading)
>>> bc = BrainCollection(
...     ['/data/sub-01.nii.gz', '/data/sub-02.nii.gz'],
...     mask='2mm-MNI152-2009c'
... )
>>> bc.shape
(2, 100, 228453)
```

```pycon
>>> # NumPy-style indexing
>>> bc[0]  # First subject -> BrainData
>>> bc[:, 0]  # First timepoint across all subjects -> BrainCollection
>>> bc[0:5, 10:20]  # 5 subjects, timepoints 10-20 -> BrainCollection
```

```pycon
>>> # Axis operations
>>> bc.mean(axis=0)  # Mean across subjects -> BrainData
>>> bc.mean(axis=1)  # Mean across time per subject -> BrainCollection
```

```pycon
>>> # From BIDS dataset
>>> bc = BrainCollection.from_bids('/data/bids', task='rest', mask=mask)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- All images must share the same mask/space. Heterogeneous masks are not
  supported; data is resampled to mask space on load.
- Some operations (e.g., to_tensor) require uniform observation counts
  across all images.

</details>

**Functions:**

Name | Description
---- | -----------
[`align`](#nltools.data.BrainCollection.align) | Align subjects using local functional alignment.
[`anova`](#nltools.data.BrainCollection.anova) | One-way ANOVA across groups defined by metadata.
[`compute_contrasts`](#nltools.data.BrainCollection.compute_contrasts) | Compute contrasts from fitted GLM beta coefficients.
[`cv`](#nltools.data.BrainCollection.cv) | Create a cross-validation pipeline for multi-subject analysis.
[`detrend`](#nltools.data.BrainCollection.detrend) | Remove trend from each image.
[`filter`](#nltools.data.BrainCollection.filter) | Filter collection by predicate.
[`fit`](#nltools.data.BrainCollection.fit) | Fit a model to each subject in the collection.
[`fit_from_events`](#nltools.data.BrainCollection.fit_from_events) | Build design matrices from events and fit GLM to each subject.
[`fit_glm`](#nltools.data.BrainCollection.fit_glm) | Fit GLM to each subject in collection.
[`fit_ridge`](#nltools.data.BrainCollection.fit_ridge) | Fit ridge regression to each subject in collection.
[`from_bids`](#nltools.data.BrainCollection.from_bids) | Create BrainCollection from a BIDS dataset.
[`from_glob`](#nltools.data.BrainCollection.from_glob) | Create BrainCollection from glob pattern.
[`from_stacked`](#nltools.data.BrainCollection.from_stacked) | Create BrainCollection by splitting a stacked BrainData.
[`isc`](#nltools.data.BrainCollection.isc) | Compute intersubject correlation (ISC) across the collection.
[`isc_test`](#nltools.data.BrainCollection.isc_test) | Compute ISC with permutation testing for statistical inference.
[`iter_batches`](#nltools.data.BrainCollection.iter_batches) | Iterate in batches along axis.
[`load`](#nltools.data.BrainCollection.load) | Load specified images into memory.
[`map`](#nltools.data.BrainCollection.map) | Apply function across specified axis.
[`max`](#nltools.data.BrainCollection.max) | Compute maximum along axis. See mean() for details.
[`mean`](#nltools.data.BrainCollection.mean) | Compute mean along axis.
[`median`](#nltools.data.BrainCollection.median) | Compute median along axis. See mean() for details.
[`memory_estimate`](#nltools.data.BrainCollection.memory_estimate) | Estimate memory usage for loading all images.
[`min`](#nltools.data.BrainCollection.min) | Compute minimum along axis. See mean() for details.
[`permutation_test`](#nltools.data.BrainCollection.permutation_test) | One-sample permutation test across images (sign-flipping).
[`permutation_test2`](#nltools.data.BrainCollection.permutation_test2) | Two-sample permutation test between collections.
[`predict`](#nltools.data.BrainCollection.predict) | Generate predictions for each subject in collection.
[`select_feature`](#nltools.data.BrainCollection.select_feature) | Select a single feature's weights across all subjects.
[`smooth`](#nltools.data.BrainCollection.smooth) | Spatially smooth each image.
[`standardize`](#nltools.data.BrainCollection.standardize) | Standardize each image.
[`std`](#nltools.data.BrainCollection.std) | Compute standard deviation along axis. See mean() for details.
[`sum`](#nltools.data.BrainCollection.sum) | Compute sum along axis. See mean() for details.
[`threshold`](#nltools.data.BrainCollection.threshold) | Threshold each image.
[`to_list`](#nltools.data.BrainCollection.to_list) | Return list of BrainData objects.
[`to_stacked`](#nltools.data.BrainCollection.to_stacked) | Stack all into single BrainData (n_total_obs, n_voxels).
[`to_tensor`](#nltools.data.BrainCollection.to_tensor) | Convert to numpy array (n_images, n_obs, n_voxels).
[`ttest`](#nltools.data.BrainCollection.ttest) | One-sample t-test across images.
[`ttest2`](#nltools.data.BrainCollection.ttest2) | Two-sample t-test between collections.
[`unload`](#nltools.data.BrainCollection.unload) | Free memory for specified images (keep paths for reloading).
[`var`](#nltools.data.BrainCollection.var) | Compute variance along axis. See mean() for details.
[`write`](#nltools.data.BrainCollection.write) | Write all images in collection to files.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`is_loaded`](#nltools.data.BrainCollection.is_loaded) | <code>[list](#list)[[bool](#bool)]</code> | List indicating which images are currently in memory.
[`mask`](#nltools.data.BrainCollection.mask) | <code>[Nifti1Image](#nibabel.Nifti1Image)</code> | Shared NIfTI brain mask image used to define the voxel space for the collection.
[`metadata`](#nltools.data.BrainCollection.metadata) | <code>[DataFrame](#pandas.DataFrame)</code> | Per-image metadata DataFrame.
[`n_images`](#nltools.data.BrainCollection.n_images) | <code>[int](#int)</code> | Number of images in collection.
[`n_voxels`](#nltools.data.BrainCollection.n_voxels) | <code>[int](#int)</code> | Number of voxels (from mask).
[`shape`](#nltools.data.BrainCollection.shape) | <code>[tuple](#tuple)[[int](#int), [int](#int) \| None, [int](#int)]</code> | Shape as (n_images, n_observations, n_voxels).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`items` | <code>[list](#list)[[Path](#pathlib.Path) \| [str](#str) \| 'BrainData']</code> | List of paths or BrainData objects. | *required*
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str)</code> | Shared mask (required). Path, nibabel image, or template name. | *required*
`metadata` | <code>[DataFrame](#pandas.DataFrame) \| None</code> | Optional per-image metadata DataFrame. | <code>None</code>
`lazy` | <code>[bool](#bool)</code> | If True, paths are loaded on demand. | <code>True</code>



##### Attributes###### `nltools.data.BrainCollection.is_loaded`

```python
is_loaded: list[bool]
```

List indicating which images are currently in memory.

###### `nltools.data.BrainCollection.mask`

```python
mask: nib.Nifti1Image
```

Shared NIfTI brain mask image used to define the voxel space for the collection.

###### `nltools.data.BrainCollection.metadata`

```python
metadata: pd.DataFrame
```

Per-image metadata DataFrame.

###### `nltools.data.BrainCollection.n_images`

```python
n_images: int
```

Number of images in collection.

###### `nltools.data.BrainCollection.n_voxels`

```python
n_voxels: int
```

Number of voxels (from mask).

###### `nltools.data.BrainCollection.shape`

```python
shape: tuple[int, int | None, int]
```

Shape as (n_images, n_observations, n_voxels).

n_observations is None if images have variable counts or not all are loaded.



##### Functions###### `nltools.data.BrainCollection.align`

```python
align(method: str = 'procrustes', scheme: str = 'searchlight', radius_mm: float = 10.0, parcellation: 'nib.Nifti1Image | None' = None, n_features: int | None = None, n_iter: int = 3, parallel: str | None = 'cpu', n_jobs: int = -1, return_model: bool = False, show_progress: bool = True) -> 'BrainCollection | tuple[BrainCollection, object]'
```

Align subjects using local functional alignment.

Performs neighborhood-based functional alignment across subjects using
LocalAlignment. Each subject's data is aligned to a common template space
using local transforms learned within searchlight spheres or parcels.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Alignment method. Options: - 'procrustes': Orthogonal Procrustes (default, preserves dimensions) - 'srm': Shared Response Model (dimensionality reduction) - 'hyperalignment': Hyperalignment (iterative Procrustes) | <code>'procrustes'</code>
`scheme` | <code>[str](#str)</code> | Spatial scheme. Options: - 'searchlight': Overlapping spheres with center-only aggregation - 'piecewise': Non-overlapping parcels (requires parcellation) | <code>'searchlight'</code>
`radius_mm` | <code>[float](#float)</code> | Sphere radius in millimeters for searchlight scheme. | <code>10.0</code>
`parcellation` | <code>'nib.Nifti1Image \| None'</code> | Parcellation image for piecewise scheme (required if scheme='piecewise'). | <code>None</code>
`n_features` | <code>[int](#int) \| None</code> | Number of features for SRM. None uses full dimensions. | <code>None</code>
`n_iter` | <code>[int](#int)</code> | Number of iterations for alignment refinement. | <code>3</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization mode. Options: - None: Single-threaded - 'cpu': CPU parallelization with joblib - 'gpu': GPU acceleration via PyTorch | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs for CPU mode (-1 = auto). | <code>-1</code>
`return_model` | <code>[bool](#bool)</code> | If True, return (aligned_collection, model) tuple for fit/transform workflow with new data. | <code>False</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| tuple[BrainCollection, object]'</code> | BrainCollection with aligned data. If return_model=True, returns
<code>'BrainCollection \| tuple[BrainCollection, object]'</code> | tuple of (aligned_collection, LocalAlignment_model).

**Examples:**

```pycon
>>> # Basic searchlight alignment
>>> aligned_bc = bc.align(method='procrustes', radius_mm=10.0)
```

```pycon
>>> # Piecewise alignment with parcellation
>>> aligned_bc = bc.align(
...     scheme='piecewise',
...     parcellation=parcellation_img,
...     method='srm',
...     n_features=50
... )
```

```pycon
>>> # Fit/transform workflow for train/test split
>>> aligned_train, model = train_bc.align(return_model=True)
>>> aligned_test = model.transform(test_data_list)
```

```pycon
>>> # GPU-accelerated alignment
>>> aligned_bc = bc.align(parallel='gpu')
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

Based on Bazeille et al. 2021 "An empirical evaluation of functional
alignment using inter-subject decoding". Center-only aggregation is
used for searchlight to preserve local orthogonality of transforms.

</details>

<details class="see-also" open markdown="1">
<summary>See Also</summary>

nltools.algorithms.alignment.LocalAlignment: Underlying alignment class.

</details>

###### `nltools.data.BrainCollection.anova`

```python
anova(groups: str | list | np.ndarray) -> tuple['BrainData', 'BrainData']
```

One-way ANOVA across groups defined by metadata.

Tests whether group means differ significantly. This is the
voxel-wise equivalent of scipy.stats.f_oneway.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`groups` | <code>[str](#str) \| [list](#list) \| [ndarray](#numpy.ndarray)</code> | Group assignment for each image. Can be: - str: Column name in metadata - list/array: Group labels of length n_images | *required*

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)['BrainData', 'BrainData']</code> | Tuple of (F_stat, p_value) as BrainData objects.

**Examples:**

```pycon
>>> # Groups from metadata column
>>> f_stat, p_val = bc.anova('condition')
```

```pycon
>>> # Explicit group labels
>>> groups = ['control'] * 10 + ['patient'] * 15
>>> f_stat, p_val = bc.anova(groups)
```

###### `nltools.data.BrainCollection.compute_contrasts`

```python
compute_contrasts(contrasts: 'str | dict | np.ndarray | list') -> 'BrainCollection | dict[str, BrainCollection]'
```

Compute contrasts from fitted GLM beta coefficients.

Applies contrast weights to each subject's betas and returns a
BrainCollection of contrast values suitable for group-level analysis.

Must be called on a BrainCollection created by fit_glm() which has
the _design_columns attribute set.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`contrasts` | <code>'str \| dict \| np.ndarray \| list'</code> | Can be: - str: Contrast string using column names, e.g., "face - house" - dict: Multiple contrasts, e.g., {"main": "face - house", "avg": [0.5, 0.5]} - array/list: Numeric contrast vector, e.g., [1, -1] | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection where each BrainData has shape (n_voxels,) containing
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | the contrast values. If dict input, returns dict of BrainCollections.

**Examples:**

```pycon
>>> # Fit GLM and compute contrast
>>> betas = bc.fit_glm(events=events_df, t_r=2.0)
>>> contrast = betas.compute_contrasts("face - house")
>>> # Group t-test on contrast
>>> group_result = contrast.ttest()
```

```pycon
>>> # Multiple contrasts
>>> contrasts = betas.compute_contrasts({
...     "face_vs_house": "face - house",
...     "face_vs_baseline": "face",
... })
>>> face_vs_house_ttest = contrasts["face_vs_house"].ttest()
```

###### `nltools.data.BrainCollection.cv`

```python
cv(k: int | None = None, scheme: str = 'kfold', split_by: str | None = None, groups: np.ndarray | None = None, random_state: int | None = None, **kwargs: int | None) -> 'BrainCollectionPipeline'
```

Create a cross-validation pipeline for multi-subject analysis.

Returns a pipeline object that enables fluent, chainable transforms
with cross-validation across subjects or runs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`k` | <code>[int](#int) \| None</code> | Number of folds (for kfold scheme). Defaults to 5. | <code>None</code>
`scheme` | <code>[str](#str)</code> | CV scheme type. Options: - 'kfold': k-fold cross-validation on pooled data - 'loso': leave-one-subject-out (one image held out per fold) - 'loro': leave-one-run-out (requires groups) | <code>'kfold'</code>
`split_by` | <code>[str](#str) \| None</code> | Metadata column for group splits. If provided and groups is None, gets groups from self.metadata[split_by]. | <code>None</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Explicit group labels for CV splits. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`**kwargs` |  | Additional arguments passed to CVScheme. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainCollectionPipeline` | <code>'BrainCollectionPipeline'</code> | Pipeline for method chaining.

**Examples:**

```pycon
>>> # Leave-one-subject-out classification
>>> result = bc.cv(scheme='loso').normalize().predict(subject_labels, algorithm='svm')
>>> print(f"Mean accuracy: {result.mean_score:.2%}")
```

```pycon
>>> # With preprocessing
>>> result = (bc
...     .cv(scheme='loso')
...     .normalize()
...     .reduce(n_components=50)
...     .predict(labels))
```

```pycon
>>> # Run-based CV with metadata
>>> result = bc.cv(scheme='loro', split_by='run').predict(y)
```

<details class="see-also" open markdown="1">
<summary>See Also</summary>

BrainCollectionPipeline: For available transforms and terminals.
CVScheme: For CV scheme configuration details.

</details>

###### `nltools.data.BrainCollection.detrend`

```python
detrend(method: str = 'linear', n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Remove trend from each image.

Delegates to BrainData.detrend() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | 'linear' or 'constant'. | <code>'linear'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with detrended images.

**Examples:**

```pycon
>>> bc.detrend()  # Remove linear trend
>>> bc.detrend(method='constant')  # Remove mean only
```

###### `nltools.data.BrainCollection.filter`

```python
filter(predicate: Callable | list | np.ndarray | 'pd.Series') -> 'BrainCollection'
```

Filter collection by predicate.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`predicate` | <code>[Callable](#collections.abc.Callable) \| [list](#list) \| [ndarray](#numpy.ndarray) \| 'pd.Series'</code> | Filter condition. Can be: - callable: fn(BrainData) → bool - list/ndarray: Boolean mask of length n_images - pd.Series: Boolean series (index ignored) | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with subset of images matching predicate.

**Examples:**

```pycon
>>> # Filter by callable
>>> bc.filter(lambda bd: bd.data.mean() > 0)
```

```pycon
>>> # Filter by boolean mask
>>> mask = [True, False, True]
>>> bc.filter(mask)
```

```pycon
>>> # Filter by metadata condition
>>> bc.filter(bc.metadata['group'] == 'control')
```

###### `nltools.data.BrainCollection.fit`

```python
fit(model: str, X: 'pd.DataFrame | np.ndarray | str | list', cv: int | None = None, scale: bool = True, scale_value: float = 100.0, show_progress: bool = True, **kwargs: bool) -> 'FittedBrainCollection'
```

Fit a model to each subject in the collection.

Unified fitting method that shadows BrainData.fit() API for multi-subject
analysis. Dispatches to model-specific implementations based on the
model parameter.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model` | <code>[str](#str)</code> | Model type - 'glm' or 'ridge' | *required*
`X` | <code>'pd.DataFrame \| np.ndarray \| str \| list'</code> | Design/feature matrix. Can be: - pd.DataFrame/DesignMatrix: Shared (used for all subjects) - np.ndarray: Shared array (used for all subjects) - str: Column name in metadata pointing to file paths - list: Per-subject list of DataFrames/arrays/paths | *required*
`cv` | <code>[int](#int) \| None</code> | Cross-validation folds (Ridge only). Default is None for GLM, 5 for Ridge when output='scores'. | <code>None</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`**kwargs` |  | Model-specific arguments passed to _fit_glm or _fit_ridge: - GLM: return_stats, save - Ridge: alpha, output, save, backend, random_state | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'FittedBrainCollection'</code> | FittedBrainCollection wrapping the fitted results. Supports:
<code>'FittedBrainCollection'</code> | - ``.results``: Access underlying BrainCollection(s) directly
<code>'FittedBrainCollection'</code> | - ``.betas``: Convenience accessor for beta coefficients (GLM)
<code>'FittedBrainCollection'</code> | - ``.pool()``: Aggregate across subjects for group analysis
<code>'FittedBrainCollection'</code> | The underlying results contain:
<code>'FittedBrainCollection'</code> | - GLM: Beta coefficients (n_regressors, n_voxels) per subject
<code>'FittedBrainCollection'</code> | - Ridge: Scores or weights depending on 'output' kwarg
<code>'FittedBrainCollection'</code> | If return_stats (GLM) or output='both' (Ridge), results is a dict.

**Examples:**

```pycon
>>> # GLM with shared design matrix
>>> fitted = bc.fit(model='glm', X=dm)
>>> betas = fitted.results  # Access BrainCollection directly
>>>
>>> # Two-stage analysis with pool()
>>> pool = bc.fit(model='glm', X=dm).pool(param='beta')
>>> t_map = pool.fit(model='ttest', contrast='A-B')
>>>
>>> # GLM with per-subject design matrices
>>> fitted = bc.fit(model='glm', X=[dm1, dm2, dm3])
>>>
>>> # Ridge encoding model with CV scores
>>> fitted = bc.fit(model='ridge', X=features, cv=5)
>>> scores = fitted.results
```

<details class="see-also" open markdown="1">
<summary>See Also</summary>

fit_from_events: Convenience method for event-based GLM workflows
fit_glm: Legacy GLM fitting (use fit_from_events instead)
fit_ridge: Legacy Ridge fitting (use fit(..., model='ridge') instead)

</details>

###### `nltools.data.BrainCollection.fit_from_events`

```python
fit_from_events(events: pd.DataFrame, t_r: float, confounds: str | list[pd.DataFrame | Path | str] | None = None, confound_columns: list[str] | None = None, hrf_model: str = 'spm', drift_model: str = 'cosine', high_pass: float = 0.01, scale: bool = True, scale_value: float = 100.0, return_stats: list[str] | None = None, return_residuals: bool = False, save: dict[str, str] | None = None, show_progress: bool = True, by_run: bool = False, run_column: str = 'run', run_lengths: int | list[int] | None = None) -> 'BrainCollection | dict[str, BrainCollection]'
```

Build design matrices from events and fit GLM to each subject.

Convenience method for event-based experimental designs. Builds
nilearn-compatible design matrices from the events DataFrame and
fits a GLM to each subject in the collection.

This is the recommended method for typical task-based fMRI analysis
where you have event timing information. For more control, use
fit(model='glm', X=design_matrices) with pre-built design matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`events` | <code>[DataFrame](#pandas.DataFrame)</code> | Task events DataFrame with onset, duration, trial_type columns. This is shared across all subjects (same experimental paradigm). If by_run=True, must also have a run column. | *required*
`t_r` | <code>[float](#float)</code> | Repetition time (TR) in seconds. | *required*
`confounds` | <code>[str](#str) \| [list](#list)[[DataFrame](#pandas.DataFrame) \| [Path](#pathlib.Path) \| [str](#str)] \| None</code> | Subject-specific confounds. Can be: - str: Column name in metadata pointing to confound file paths - list: List of DataFrames or paths, one per subject - None: No confounds (only task + drift terms) | <code>None</code>
`confound_columns` | <code>[list](#list)[[str](#str)] \| None</code> | Columns to extract from confound files. If None and confounds provided, uses all columns. | <code>None</code>
`hrf_model` | <code>[str](#str)</code> | HRF model for convolution ('spm', 'glover', 'fir', etc.) | <code>'spm'</code>
`drift_model` | <code>[str](#str)</code> | Drift model ('cosine', 'polynomial', None) | <code>'cosine'</code>
`high_pass` | <code>[float](#float)</code> | High-pass filter cutoff in Hz (default 0.01) | <code>0.01</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`return_stats` | <code>[list](#list)[[str](#str)] \| None</code> | Optional list of statistics to return as separate BrainCollections. Options: 't', 'r2', 'p', 'se', 'residual'. | <code>None</code>
`return_residuals` | <code>[bool](#bool)</code> | If True, return residuals (same as return_stats=['residual']). | <code>False</code>
`save` | <code>[dict](#dict)[[str](#str), [str](#str)] \| None</code> | Dict mapping output type to path template. | <code>None</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`by_run` | <code>[bool](#bool)</code> | If True, fit GLM separately per run and return run-level betas. This enables MVPA decoding with leave-one-run-out CV. | <code>False</code>
`run_column` | <code>[str](#str)</code> | Column name in events identifying runs (default 'run'). | <code>'run'</code>
`run_lengths` | <code>[int](#int) \| [list](#list)[[int](#int)] \| None</code> | Number of TRs per run. Required when by_run=True. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection of beta coefficients for task regressors.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | If return_stats specified, returns dict with keys 'betas', 't', etc.

**Examples:**

```pycon
>>> # Basic GLM fit from events
>>> betas = bc.fit_from_events(events=events_df, t_r=2.0)
>>> group_t = betas.ttest()
>>>
>>> # With confounds from metadata column
>>> betas = bc.fit_from_events(
...     events=events_df,
...     t_r=2.0,
...     confounds='confound_file',
...     confound_columns=['trans_x', 'trans_y', 'trans_z']
... )
>>>
>>> # Run-level betas for MVPA
>>> betas = bc.fit_from_events(events=events_df, t_r=2.0, by_run=True)
```

<details class="see-also" open markdown="1">
<summary>See Also</summary>

fit: Unified fit method that accepts pre-built design matrices
_fit_glm: Internal method for design matrix-based fitting

</details>

###### `nltools.data.BrainCollection.fit_glm`

```python
fit_glm(events: pd.DataFrame, t_r: float, confounds: str | list[pd.DataFrame | Path | str] | None = None, confound_columns: list[str] | None = None, hrf_model: str = 'spm', drift_model: str = 'cosine', high_pass: float = 0.01, scale: bool = True, scale_value: float = 100.0, return_stats: list[str] | None = None, return_residuals: bool = False, save: dict[str, str] | None = None, show_progress: bool = True, by_run: bool = False, run_column: str = 'run', run_lengths: int | list[int] | None = None) -> 'BrainCollection | dict[str, BrainCollection]'
```

Fit GLM to each subject in collection.

Memory-efficient first-level GLM analysis that processes subjects
one at a time. Returns a BrainCollection of beta coefficients for
task regressors (confounds and drift terms are fit but not returned).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`events` | <code>[DataFrame](#pandas.DataFrame)</code> | Task events DataFrame with onset, duration, trial_type columns. This is shared across all subjects (same experimental paradigm). If by_run=True, must also have a run column. | *required*
`t_r` | <code>[float](#float)</code> | Repetition time (TR) in seconds. | *required*
`confounds` | <code>[str](#str) \| [list](#list)[[DataFrame](#pandas.DataFrame) \| [Path](#pathlib.Path) \| [str](#str)] \| None</code> | Subject-specific confounds. Can be: - str: Column name in metadata pointing to confound file paths - list: List of DataFrames or paths, one per subject - None: No confounds (only task + drift terms) | <code>None</code>
`confound_columns` | <code>[list](#list)[[str](#str)] \| None</code> | Columns to extract from confound files. If None and confounds provided, uses all columns. | <code>None</code>
`hrf_model` | <code>[str](#str)</code> | HRF model for convolution ('spm', 'glover', 'fir', etc.) | <code>'spm'</code>
`drift_model` | <code>[str](#str)</code> | Drift model ('cosine', 'polynomial', None) | <code>'cosine'</code>
`high_pass` | <code>[float](#float)</code> | High-pass filter cutoff in Hz (default 0.01) | <code>0.01</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`return_stats` | <code>[list](#list)[[str](#str)] \| None</code> | Optional list of statistics to return as separate BrainCollections. Options: 't', 'r2', 'p', 'se', 'residual'. | <code>None</code>
`return_residuals` | <code>[bool](#bool)</code> | If True, return residuals (same as return_stats=['residual']). | <code>False</code>
`save` | <code>[dict](#dict)[[str](#str), [str](#str)] \| None</code> | Dict mapping output type to path template, e.g. ``{'betas': 'output/{subject}_betas.nii.gz', 't': 'output/{subject}_tstat.nii.gz'}``. Supports {subject}, {session}, {idx}, and other metadata columns. | <code>None</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`by_run` | <code>[bool](#bool)</code> | If True, fit GLM separately per run and return run-level betas. This enables MVPA decoding with leave-one-run-out CV. Each subject will have (n_runs * n_conditions, n_voxels) betas. | <code>False</code>
`run_column` | <code>[str](#str)</code> | Column name in events identifying runs (default 'run'). | <code>'run'</code>
`run_lengths` | <code>[int](#int) \| [list](#list)[[int](#int)] \| None</code> | Number of TRs per run. Required when by_run=True.<br>- int: All runs have same length - list of int: Different length per run - None: Will attempt to infer equal-length runs from total scans | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection where each BrainData has shape:
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | - (n_task_regressors, n_voxels) if by_run=False (default)
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | - (n_runs * n_task_regressors, n_voxels) if by_run=True
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | The ``._design_columns`` attribute stores task regressor names.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | If by_run=True, also stores ``._condition_labels`` and ``._run_labels``.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | If return_stats specified, returns dict with keys 'betas', 't', etc.

**Examples:**

```pycon
>>> # Basic GLM fit
>>> betas = bc.fit_glm(events=events_df, t_r=2.0)
>>> # Group t-test on first regressor
>>> group_t = betas[:, 0].ttest()
```

```pycon
>>> # Run-level betas for MVPA decoding
>>> betas = bc.fit_glm(events=events_df, t_r=2.0, by_run=True)
>>> # betas._condition_labels = ['face', 'house', 'face', 'house', ...]
>>> # betas._run_labels = [1, 1, 2, 2, 3, 3, ...]
>>> accuracy = betas.predict(y=None, method='searchlight')
```

```pycon
>>> # With confounds from metadata column
>>> betas = bc.fit_glm(
...     events=events_df,
...     t_r=2.0,
...     confounds='confound_file',  # column name in metadata
...     confound_columns=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
... )
```

###### `nltools.data.BrainCollection.fit_ridge`

```python
fit_ridge(X: 'np.ndarray | str | list', alpha: float | str = 1.0, cv: int | None = 5, scale: bool = True, scale_value: float = 100.0, output: str = 'scores', save: dict[str, str] | None = None, show_progress: bool = True, **ridge_kwargs: bool) -> 'BrainCollection | dict[str, BrainCollection]'
```

Fit ridge regression to each subject in collection.

Memory-efficient encoding model fitting that processes subjects one at a
time. Default behavior returns cross-validated R² scores per voxel,
suitable for group-level inference on encoding model performance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>'np.ndarray \| str \| list'</code> | Feature matrix. Can be: - np.ndarray: Shared features (n_samples, n_features) used for all subjects - str: Column name in metadata pointing to feature file paths - list: List of arrays/DataFrames, one per subject | *required*
`alpha` | <code>[float](#float) \| [str](#str)</code> | Ridge regularization parameter. Can be: - float: Fixed regularization strength - 'auto': Use cross-validation to select optimal alpha | <code>1.0</code>
`cv` | <code>[int](#int) \| None</code> | Cross-validation folds for computing scores. Default is 5. Required when output='scores' or 'both'. Set to None only when output='weights'. | <code>5</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`output` | <code>[str](#str)</code> | What to return. Options: - 'scores': CV R² scores per voxel (default, for encoding workflow) - 'weights': Model weights (n_features, n_voxels) - 'both': Dict with both 'scores' and 'weights' | <code>'scores'</code>
`save` | <code>[dict](#dict)[[str](#str), [str](#str)] \| None</code> | Dict mapping output type to path template, e.g. ``{'weights': 'output/{subject}_weights.nii.gz', 'scores': 'output/{subject}_scores.nii.gz'}``. Supports {subject}, {session}, {idx}, and other metadata columns. | <code>None</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`**ridge_kwargs` |  | Additional arguments passed to Ridge model (e.g., backend='torch', random_state=42). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection of scores or weights, or dict with both if output='both'.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | Each BrainData will have ``cv_results_`` attribute when cv is used.

**Examples:**

```pycon
>>> # Encoding model workflow: get CV scores for group analysis
>>> scores = bc.fit_ridge(X=features, alpha=1.0)
>>> group_ttest = scores.ttest()  # Test encoding accuracy vs chance
```

```pycon
>>> # Get both scores and weights
>>> results = bc.fit_ridge(X=features, alpha=1.0, output='both')
>>> scores = results['scores']
>>> weights = results['weights']
```

```pycon
>>> # Auto-select alpha with CV
>>> scores = bc.fit_ridge(X=features, alpha='auto', cv=5)
```

```pycon
>>> # Get weights only (no CV needed)
>>> weights = bc.fit_ridge(X=features, alpha=1.0, output='weights', cv=None)
```

###### `nltools.data.BrainCollection.from_bids`

```python
from_bids(layout: Any, mask: nib.Nifti1Image | Path | str, *, task: str | None = None, subject: str | list[str] | None = None, session: str | list[str] | None = None, run: int | list[int] | None = None, space: str | None = None, suffix: str = 'bold', extension: str = 'nii.gz', **bids_filters: str) -> 'BrainCollection'
```

Create BrainCollection from a BIDS dataset.

Requires pybids to be installed: `pip install pybids`

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`layout` | <code>[Any](#typing.Any)</code> | pybids BIDSLayout object or path to BIDS dataset. | *required*
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str)</code> | Shared mask (required). | *required*
`task` | <code>[str](#str) \| None</code> | BIDS task filter. | <code>None</code>
`subject` | <code>[str](#str) \| [list](#list)[[str](#str)] \| None</code> | Subject ID(s) to include. | <code>None</code>
`session` | <code>[str](#str) \| [list](#list)[[str](#str)] \| None</code> | Session ID(s) to include. | <code>None</code>
`run` | <code>[int](#int) \| [list](#list)[[int](#int)] \| None</code> | Run number(s) to include. | <code>None</code>
`space` | <code>[str](#str) \| None</code> | BIDS space filter (e.g., 'MNI152NLin2009cAsym'). | <code>None</code>
`suffix` | <code>[str](#str)</code> | BIDS suffix (default 'bold'). | <code>'bold'</code>
`extension` | <code>[str](#str)</code> | File extension (default 'nii.gz'). | <code>'nii.gz'</code>
`**bids_filters` |  | Additional BIDS entity filters. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with metadata extracted from BIDS entities.

**Examples:**

```pycon
>>> bc = BrainCollection.from_bids(
...     '/data/bids_dataset',
...     mask='2mm-MNI152-2009c',
...     task='rest',
...     space='MNI152NLin2009cAsym'
... )
```

###### `nltools.data.BrainCollection.from_glob`

```python
from_glob(pattern: str, mask: nib.Nifti1Image | Path | str, *, pattern_groups: dict[str, int] | str | None = None, sort: bool = True) -> 'BrainCollection'
```

Create BrainCollection from glob pattern.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`pattern` | <code>[str](#str)</code> | Glob pattern (e.g., ``'/data/*/func/*_bold.nii.gz'``). | *required*
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str)</code> | Shared mask (required). | *required*
`pattern_groups` | <code>[dict](#dict)[[str](#str), [int](#int)] \| [str](#str) \| None</code> | Regex pattern with named groups for metadata extraction. Example: ``r'sub-(?P<subject>\w+)/.*run-(?P<run>\d+)'`` | <code>None</code>
`sort` | <code>[bool](#bool)</code> | Sort files alphabetically (default True). | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with optional metadata from pattern groups.

**Examples:**

```pycon
>>> bc = BrainCollection.from_glob(
...     '/data/sub-*/func/*_bold.nii.gz',
...     mask=mask,
...     pattern_groups=r'sub-(?P<subject>\w+)'
... )
```

###### `nltools.data.BrainCollection.from_stacked`

```python
from_stacked(brain_data: 'BrainData', splits: list[int] | None = None, n_images: int | None = None) -> 'BrainCollection'
```

Create BrainCollection by splitting a stacked BrainData.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_data` | <code>'BrainData'</code> | BrainData with shape (n_total_obs, n_voxels). | *required*
`splits` | <code>[list](#list)[[int](#int)] \| None</code> | List of observation counts per image. Must sum to n_total_obs. | <code>None</code>
`n_images` | <code>[int](#int) \| None</code> | Number of images (splits evenly). Mutually exclusive with splits. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with data split according to specification.

**Examples:**

```pycon
>>> # Split evenly into 3 images
>>> bc = BrainCollection.from_stacked(bd, n_images=3)
```

```pycon
>>> # Split with explicit counts
>>> bc = BrainCollection.from_stacked(bd, splits=[100, 100, 150])
```

###### `nltools.data.BrainCollection.isc`

```python
isc(method: str = 'loo', roi_mask: 'nib.Nifti1Image | Path | str | None' = None, radius: float | None = 6.0, metric: str = 'median', parallel: str = 'cpu', n_jobs: int = -1, show_progress: bool = True) -> dict
```

Compute intersubject correlation (ISC) across the collection.

ISC measures the similarity of brain responses across subjects,
computed by correlating each subject's timeseries with others.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ISC computation method: - 'loo': Leave-one-out (correlate each subject with mean of others) - 'pairwise': All pairwise correlations between subjects | <code>'loo'</code>
`roi_mask` | <code>'nib.Nifti1Image \| Path \| str \| None'</code> | If provided, compute ISC per ROI. Can be: - NIfTI image with integer labels (atlas/parcellation) - Path to parcellation file | <code>None</code>
`radius` | <code>[float](#float) \| None</code> | Searchlight radius in mm. If None, use voxelwise mode. Ignored if roi_mask is provided. | <code>6.0</code>
`metric` | <code>[str](#str)</code> | Summary statistic for aggregating ISC values: - 'median': Robust to outliers (default) - 'mean': Fisher z-transformed mean | <code>'median'</code>
`parallel` | <code>[str](#str)</code> | Parallelization method ('cpu', 'gpu', or None). | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs (-1 = all cores). | <code>-1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during extraction. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with: - 'isc': BrainData with ISC values - 'method': ISC method used ('loo' or 'pairwise') - 'extraction': Extraction mode ('roi', 'searchlight', 'voxelwise') - 'n_subjects': Number of subjects - 'extraction_info': Dict with extraction metadata

**Examples:**

```pycon
>>> # ROI-based ISC using atlas
>>> result = bc.isc(roi_mask="atlas.nii.gz")
>>> result['isc'].plot()
```

```pycon
>>> # Searchlight ISC
>>> result = bc.isc(radius=10.0)
```

```pycon
>>> # Voxelwise ISC
>>> result = bc.isc(radius=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

For permutation testing, see BrainCollection.isc_test() (requires
discussion of statistical methodology first).

</details>

###### `nltools.data.BrainCollection.isc_test`

```python
isc_test(method: str = 'loo', roi_mask: 'nib.Nifti1Image | Path | str | None' = None, radius: float | None = 6.0, n_permute: int = 5000, permutation_method: str = 'bootstrap', metric: str = 'median', tail: int = 2, ci_percentile: float = 95, parallel: str = 'cpu', n_jobs: int = -1, random_state: int | None = None, return_null: bool = False, show_progress: bool = True) -> dict
```

Compute ISC with permutation testing for statistical inference.

This method combines ISC computation with permutation testing to
determine statistical significance. It uses the same extraction
pipeline as isc() and wraps isc_permutation_test().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ISC computation method: - 'loo': Leave-one-out (correlate each subject with mean of others) - 'pairwise': All pairwise correlations between subjects | <code>'loo'</code>
`roi_mask` | <code>'nib.Nifti1Image \| Path \| str \| None'</code> | If provided, compute ISC per ROI. Can be: - NIfTI image with integer labels (atlas/parcellation) - Path to parcellation file | <code>None</code>
`radius` | <code>[float](#float) \| None</code> | Searchlight radius in mm. If None, use voxelwise mode. Ignored if roi_mask is provided. | <code>6.0</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations/bootstrap iterations. Default 5000. | <code>5000</code>
`permutation_method` | <code>[str](#str)</code> | Method for null distribution:<br>- 'bootstrap': Subject-wise bootstrap (default, Chen et al. 2016).   Tests whether observed ISC differs from random groupings. - 'circle_shift': Circular time-shift (preserves autocorrelation).   Tests for temporally-locked shared signal. - 'phase_randomize': FFT phase randomization (preserves power spectrum).   Tests for nonlinear temporal coupling. | <code>'bootstrap'</code>
`metric` | <code>[str](#str)</code> | Summary statistic for aggregating ISC values: - 'median': Robust to outliers (default) - 'mean': Fisher z-transformed mean | <code>'median'</code>
`tail` | <code>[int](#int)</code> | One-tailed (1) or two-tailed (2) test. Default 2. | <code>2</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95). Default 95. | <code>95</code>
`parallel` | <code>[str](#str)</code> | Parallelization method ('cpu', 'gpu', or None). | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs (-1 = all cores). | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, include null distribution in results. | <code>False</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during extraction and permutation. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with: - 'isc': BrainData with ISC values - 'p': BrainData with p-values (Phipson-Smyth corrected) - 'ci': Tuple of (lower, upper) BrainData confidence intervals - 'method': ISC method used ('loo' or 'pairwise') - 'permutation_method': Permutation method used - 'extraction': Extraction mode ('roi', 'searchlight', 'voxelwise') - 'n_subjects': Number of subjects - 'n_permute': Number of permutations - 'null_dist': (optional) Null distribution array if return_null=True

**Examples:**

```pycon
>>> # ROI-based ISC with permutation testing
>>> result = bc.isc_test(roi_mask="atlas.nii.gz", n_permute=5000)
>>> sig_mask = result['p'].data < 0.05
>>> print(f"Significant ROIs: {sig_mask.sum()}")
```

```pycon
>>> # Searchlight ISC testing
>>> result = bc.isc_test(radius=10.0)
>>> result['isc'].plot()  # Show ISC values
>>> result['p'].plot()    # Show p-values
```

```pycon
>>> # Voxelwise with phase randomization (tests temporal coupling)
>>> result = bc.isc_test(
...     radius=None,
...     permutation_method='phase_randomize',
...     parallel='gpu'
... )
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Bootstrap (default) is recommended for standard ISC inference
  (Chen et al. 2016). It tests whether ISC is significant at
  the group level.
- Circle_shift and phase_randomize are more specialized - they
  test for temporally-structured shared signal beyond what's
  explained by autocorrelation or spectral structure alone.
- For large voxelwise analyses, bootstrap is much faster as it
  resamples pre-computed values rather than recomputing ISC.

</details>

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., et al. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

###### `nltools.data.BrainCollection.iter_batches`

```python
iter_batches(batch_size: int, axis: int = 0, show_progress: bool = True) -> Generator['BrainCollection', None, None]
```

Iterate in batches along axis.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`batch_size` | <code>[int](#int)</code> | Number of items per batch. | *required*
`axis` | <code>[int](#int)</code> | Axis to batch along: - 0: Batches of images (default) - 1: Batches of timepoints (within each image) | <code>0</code>
`show_progress` | <code>[bool](#bool)</code> | Show tqdm progress bar. | <code>True</code>

**Yields:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection for each batch.

**Examples:**

```pycon
>>> # Batch over images
>>> for batch in bc.iter_batches(batch_size=5):
...     process(batch)  # batch is BrainCollection with 5 images
```

```pycon
>>> # Batch over time
>>> for batch in bc.iter_batches(batch_size=10, axis=1):
...     process(batch)  # batch has 10 timepoints per image
```

###### `nltools.data.BrainCollection.load`

```python
load(indices: list[int] | None = None) -> 'BrainCollection'
```

Load specified images into memory.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`indices` | <code>[list](#list)[[int](#int)] \| None</code> | List of indices to load. If None, loads all. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | self (for chaining)

###### `nltools.data.BrainCollection.map`

```python
map(fn: Callable, axis: int | str = 0, n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Apply function across specified axis.

This is the general-purpose transformation method. For common operations,
use convenience methods like standardize(), smooth(), etc.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fn` | <code>[Callable](#collections.abc.Callable)</code> | Function to apply. Signature depends on axis: - axis=0: fn(BrainData) → BrainData (per image) - axis=1: fn(BrainData) → BrainData (per timepoint slice) - axis=2: fn(ndarray[n_obs]) → ndarray (per voxel timeseries) | *required*
`axis` | <code>[int](#int) \| [str](#str)</code> | Axis to iterate over: - 0 or 'images': Apply fn to each image independently - 1 or 'time': Apply fn to each timepoint across images - 2 or 'voxels': Apply fn to each voxel timeseries per image | <code>0</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. -1 for all cores. Default 1. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show tqdm progress bar. Default True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with transformed data.

**Examples:**

```pycon
>>> # Per-image operation
>>> bc.map(lambda bd: bd.standardize())
```

```pycon
>>> # Per-voxel timeseries (e.g., detrend each voxel)
>>> from scipy.signal import detrend
>>> bc.map(detrend, axis=2)
```

```pycon
>>> # Parallel processing
>>> bc.map(expensive_fn, n_jobs=-1)
```

###### `nltools.data.BrainCollection.max`

```python
max(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute maximum along axis. See mean() for details.

###### `nltools.data.BrainCollection.mean`

```python
mean(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute mean along axis.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>[int](#int) \| [str](#str) \| [tuple](#tuple)[[int](#int), ...]</code> | Axis or axes to aggregate: - 0 or 'images': Mean across images -> BrainData (n_obs, n_voxels) - 1 or 'time': Mean across time -> BrainCollection (n_images, n_voxels) - 2 or 'voxels': Mean across voxels -> np.ndarray (n_images, n_obs) - (0, 1): Mean across images and time -> BrainData (n_voxels,) | <code>0</code>
`batch_size` | <code>[int](#int) \| None</code> | Number of images to process at once (for memory efficiency). If None, uses streaming algorithm. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainData \| BrainCollection \| np.ndarray'</code> | BrainData, BrainCollection, or np.ndarray depending on axis.

**Examples:**

```pycon
>>> bc.mean(axis=0)  # Mean across subjects
>>> bc.mean(axis='images')  # Same as above
>>> bc.mean(axis=1)  # Mean across time per subject
>>> bc.mean(axis=(0, 1))  # Grand mean
```

###### `nltools.data.BrainCollection.median`

```python
median(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute median along axis. See mean() for details.

###### `nltools.data.BrainCollection.memory_estimate`

```python
memory_estimate() -> str
```

Estimate memory usage for loading all images.

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | Human-readable string like "12.4 GB total (1.2 GB per image avg)"

###### `nltools.data.BrainCollection.min`

```python
min(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute minimum along axis. See mean() for details.

###### `nltools.data.BrainCollection.permutation_test`

```python
permutation_test(n_permute: int = 5000, tail: int = 2, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None, return_null: bool = False) -> dict
```

One-sample permutation test across images (sign-flipping).

Tests whether the mean across images is significantly different from
zero using sign-flipping permutation. More robust than parametric
t-test for non-normal distributions.

This is a collection-level interface to
nltools.algorithms.inference.one_sample_permutation_test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000). | <code>5000</code>
`tail` | <code>[int](#int)</code> | Test type - 1 for one-tailed, 2 for two-tailed (default). | <code>2</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization method: - 'cpu': CPU parallelization via joblib (default) - 'gpu': GPU acceleration via PyTorch - None: Single-threaded (for debugging) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores (default: -1 = all cores). | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget (default: 4.0 GB). | <code>4.0</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, include null distribution in result. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | dict with keys: - 'mean': BrainData with observed mean across images - 'p': BrainData with p-values - 'null_dist': np.ndarray (if return_null=True) - 'parallel': parallelization method used

**Examples:**

```pycon
>>> result = bc.permutation_test(n_permute=5000)
>>> mean_bd, p_bd = result['mean'], result['p']
```

```pycon
>>> # With GPU acceleration
>>> result = bc.permutation_test(parallel='gpu')
```

###### `nltools.data.BrainCollection.permutation_test2`

```python
permutation_test2(other: 'BrainCollection', n_permute: int = 5000, tail: int = 2, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None, return_null: bool = False) -> dict
```

Two-sample permutation test between collections.

Tests whether two collections have different means using group
label permutation. More robust than parametric t-test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` | <code>'BrainCollection'</code> | Another BrainCollection to compare against. | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000). | <code>5000</code>
`tail` | <code>[int](#int)</code> | Test type - 1 for one-tailed, 2 for two-tailed (default). | <code>2</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization method ('cpu', 'gpu', or None). | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores (default: -1 = all cores). | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget (default: 4.0 GB). | <code>4.0</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, include null distribution in result. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | dict with keys: - 'mean_diff': BrainData with observed mean difference - 'p': BrainData with p-values - 'null_dist': np.ndarray (if return_null=True) - 'parallel': parallelization method used

**Examples:**

```pycon
>>> result = patients.permutation_test2(controls)
>>> diff_bd, p_bd = result['mean_diff'], result['p']
```

###### `nltools.data.BrainCollection.predict`

```python
predict(X: 'np.ndarray | str | list | None' = None, y: 'np.ndarray | None' = None, method: str = 'whole_brain', estimator: str = 'svm', cv: str = 5, groups: 'np.ndarray | None' = None, roi_mask: 'np.ndarray | None' = None, radius: float = 10.0, scoring: str = 'accuracy', standardize: bool = True, n_jobs: int = -1, show_progress: bool = True) -> 'BrainCollection'
```

Generate predictions for each subject in collection.

This method supports two prediction modes determined by which parameter
is provided:

1. **Timeseries prediction** (X provided): Use fitted ridge model to
   predict voxel responses for new feature data.

2. **MVPA decoding** (y provided): Train a classifier to predict labels
   from brain patterns using cross-validation.

For MVPA, if this collection was created with by_run=True, you can
use y=None to infer labels from _condition_labels and groups from
_run_labels (leave-one-run-out CV).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>'np.ndarray \| str \| list \| None'</code> | Features for timeseries prediction. Can be: - np.ndarray: Shared features (same for all subjects) - str: Metadata column with per-subject feature paths - list: Per-subject feature arrays | <code>None</code>
`y` | <code>'np.ndarray \| None'</code> | Labels for MVPA decoding. If None and _condition_labels exists, will use stored condition labels (from fit_glm with by_run=True). | <code>None</code>
`method` | <code>[str](#str)</code> | MVPA method - 'whole_brain', 'searchlight', or 'roi'. | <code>'whole_brain'</code>
`estimator` |  | Classifier - 'svm', 'logistic', 'ridge', 'lda' or sklearn estimator instance. | <code>'svm'</code>
`cv` |  | Cross-validation strategy. If None and _run_labels exists, uses leave-one-group-out with run labels. | <code>5</code>
`groups` | <code>'np.ndarray \| None'</code> | Group labels for GroupKFold/LeaveOneGroupOut. If None and _run_labels exists, uses stored run labels. | <code>None</code>
`roi_mask` |  | Mask for ROI-based MVPA. Required if method='roi'. | <code>None</code>
`radius` | <code>[float](#float)</code> | Searchlight radius in mm (default 10.0). | <code>10.0</code>
`scoring` | <code>[str](#str)</code> | Scoring metric (default 'accuracy'). | <code>'accuracy'</code>
`standardize` | <code>[bool](#bool)</code> | If True, standardize features before classification. | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Parallel jobs for searchlight (-1 = all cores). | <code>-1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with prediction results:
<code>'BrainCollection'</code> | - For timeseries: (n_timepoints, n_voxels) predicted responses
<code>'BrainCollection'</code> | - For MVPA: (1, n_voxels) accuracy values

**Examples:**

```pycon
>>> # MVPA workflow with run-level betas
>>> betas = bc.fit_glm(events=events, t_r=2.0, by_run=True)
>>> accuracy = betas.predict(y=None, method='whole_brain')
>>> # y and groups inferred from _condition_labels, _run_labels
```

```pycon
>>> # Explicit labels
>>> accuracy = betas.predict(y=labels, method='searchlight')
```

```pycon
>>> # Timeseries prediction with ridge weights
>>> weights = bc.fit_ridge(X=features, output='weights')
>>> predictions = weights.predict(X=new_features)
```

###### `nltools.data.BrainCollection.select_feature`

```python
select_feature(feature: 'int | str') -> 'BrainCollection'
```

Select a single feature's weights across all subjects.

Used after fit_ridge() to extract weights for a specific feature
for group-level analysis (e.g., t-test on feature weights).

Must be called on a BrainCollection created by fit_ridge() where
each subject has shape (n_features, n_voxels).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`feature` | <code>'int \| str'</code> | Feature to select. Can be: - int: Feature index (0-based) - str: Feature name (requires _feature_names attribute) | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection where each BrainData has shape (n_voxels,)
<code>'BrainCollection'</code> | containing the weights for the specified feature.

**Examples:**

```pycon
>>> # Fit ridge and select feature
>>> weights = bc.fit_ridge(X=features, alpha=1.0)
>>> feature_0 = weights.select_feature(0)
>>> # Group t-test on first feature's weights
>>> group_result = feature_0.ttest()
```

```pycon
>>> # By name (if features had column names)
>>> face_weights = weights.select_feature("face_response")
```

###### `nltools.data.BrainCollection.smooth`

```python
smooth(fwhm: float, n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Spatially smooth each image.

Delegates to BrainData.smooth() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fwhm` | <code>[float](#float)</code> | Full width at half maximum of Gaussian kernel in mm. | *required*
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with smoothed images.

**Examples:**

```pycon
>>> bc.smooth(fwhm=6)  # 6mm FWHM smoothing
```

###### `nltools.data.BrainCollection.standardize`

```python
standardize(axis: int = 0, method: str = 'center', n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Standardize each image.

Delegates to BrainData.standardize() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>[int](#int)</code> | Axis for standardization within each image: - 0: Standardize across observations (time) per voxel - 1: Standardize across voxels per observation | <code>0</code>
`method` | <code>[str](#str)</code> | 'center' (subtract mean) or 'zscore' (subtract mean, divide std) | <code>'center'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with standardized images.

**Examples:**

```pycon
>>> bc.standardize()  # Center each image across time
>>> bc.standardize(method='zscore')  # Z-score each image
>>> bc.standardize(axis=1)  # Standardize across voxels
```

###### `nltools.data.BrainCollection.std`

```python
std(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute standard deviation along axis. See mean() for details.

###### `nltools.data.BrainCollection.sum`

```python
sum(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute sum along axis. See mean() for details.

###### `nltools.data.BrainCollection.threshold`

```python
threshold(upper: float | str | None = None, lower: float | str | None = None, binarize: bool = False, coerce_nan: bool = True, n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Threshold each image.

Delegates to BrainData.threshold() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`upper` | <code>[float](#float) \| [str](#str) \| None</code> | Upper cutoff. String interpreted as percentile. | <code>None</code>
`lower` | <code>[float](#float) \| [str](#str) \| None</code> | Lower cutoff. String interpreted as percentile. | <code>None</code>
`binarize` | <code>[bool](#bool)</code> | Return binary mask. | <code>False</code>
`coerce_nan` | <code>[bool](#bool)</code> | Replace NaN with 0. | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with thresholded images.

**Examples:**

```pycon
>>> bc.threshold(lower=0)  # Zero out negative values
>>> bc.threshold(upper='95%')  # Keep top 5%
>>> bc.threshold(lower=2, binarize=True)  # Binary mask
```

###### `nltools.data.BrainCollection.to_list`

```python
to_list() -> list['BrainData']
```

Return list of BrainData objects.

Loads all items if not already loaded.

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)['BrainData']</code> | List of BrainData objects.

###### `nltools.data.BrainCollection.to_stacked`

```python
to_stacked() -> 'BrainData'
```

Stack all into single BrainData (n_total_obs, n_voxels).

**Returns:**

Type | Description
---- | -----------
<code>'BrainData'</code> | Single BrainData with all observations concatenated.

**Examples:**

```pycon
>>> bc = BrainCollection([bd1, bd2, bd3], mask=mask)
>>> stacked = bc.to_stacked()
>>> stacked.shape
(300, 50000)  # 3 images * 100 obs each
```

###### `nltools.data.BrainCollection.to_tensor`

```python
to_tensor(batch_size: int | None = None) -> np.ndarray | Generator[np.ndarray, None, None]
```

Convert to numpy array (n_images, n_obs, n_voxels).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`batch_size` | <code>[int](#int) \| None</code> | If specified, returns generator yielding batches. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray) \| [Generator](#typing.Generator)[[ndarray](#numpy.ndarray), None, None]</code> | Full tensor if batch_size is None, otherwise generator.

**Examples:**

```pycon
>>> tensor = bc.to_tensor()  # Full array
>>> tensor.shape
(3, 100, 50000)
```

```pycon
>>> # Batched iteration
>>> for batch in bc.to_tensor(batch_size=10):
...     process(batch)  # batch.shape = (10, 100, 50000)
```

###### `nltools.data.BrainCollection.ttest`

```python
ttest(popmean: float = 0.0, axis: int | str = 0) -> tuple['BrainData', 'BrainData']
```

One-sample t-test across images.

Tests whether the mean across images is significantly different from
a population mean (default: 0). This is the voxel-wise equivalent of
scipy.stats.ttest_1samp.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`popmean` | <code>[float](#float)</code> | Population mean to test against (default: 0). | <code>0.0</code>
`axis` | <code>[int](#int) \| [str](#str)</code> | Axis to test across. Only axis=0 (images) supported. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainData'</code> | Tuple of (t_stat, p_value) as BrainData objects.
<code>'BrainData'</code> | Both have shape (n_obs, n_voxels) if uniform obs counts.

**Examples:**

```pycon
>>> t_stat, p_val = bc.ttest()  # Test mean != 0
>>> t_stat, p_val = bc.ttest(popmean=0.5)  # Test mean != 0.5
```

```pycon
>>> # Threshold significant voxels
>>> sig_mask = p_val.data < 0.05
```

###### `nltools.data.BrainCollection.ttest2`

```python
ttest2(other: 'BrainCollection', equal_var: bool = True) -> tuple['BrainData', 'BrainData']
```

Two-sample t-test between collections.

Tests whether two collections have different means. This is the
voxel-wise equivalent of scipy.stats.ttest_ind.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` | <code>'BrainCollection'</code> | Another BrainCollection to compare against. | *required*
`equal_var` | <code>[bool](#bool)</code> | If True (default), perform standard t-test assuming equal variances. If False, use Welch's t-test. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)['BrainData', 'BrainData']</code> | Tuple of (t_stat, p_value) as BrainData objects.

**Examples:**

```pycon
>>> t_stat, p_val = patients.ttest2(controls)
>>> t_stat, p_val = group1.ttest2(group2, equal_var=False)  # Welch's
```

###### `nltools.data.BrainCollection.unload`

```python
unload(indices: list[int] | None = None) -> 'BrainCollection'
```

Free memory for specified images (keep paths for reloading).

Only works for items that were originally loaded from paths.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`indices` | <code>[list](#list)[[int](#int)] \| None</code> | List of indices to unload. If None, unloads all possible. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | self (for chaining)

###### `nltools.data.BrainCollection.var`

```python
var(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute variance along axis. See mean() for details.

###### `nltools.data.BrainCollection.write`

```python
write(directory: str | Path, pattern: str = 'image_{i:04d}.nii.gz', metadata_file: str | None = 'metadata.csv') -> list[Path]
```

Write all images in collection to files.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`directory` | <code>[str](#str) \| [Path](#pathlib.Path)</code> | Output directory path. Will be created if it doesn't exist. | *required*
`pattern` | <code>[str](#str)</code> | Filename pattern with {i} placeholder for image index. Default: "image_{i:04d}.nii.gz" produces image_0000.nii.gz, etc. | <code>'image_{i:04d}.nii.gz'</code>
`metadata_file` | <code>[str](#str) \| None</code> | Optional filename for metadata CSV. Set to None to skip. Default: "metadata.csv" | <code>'metadata.csv'</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[Path](#pathlib.Path)]</code> | List of paths to written files.

**Examples:**

```pycon
>>> bc = BrainCollection([bd1, bd2, bd3], mask=mask)
>>> paths = bc.write("output/")
>>> # Creates: output/image_0000.nii.gz, image_0001.nii.gz, etc.
```

```pycon
>>> # Custom pattern
>>> bc.write("output/", pattern="sub-{i:02d}_bold.nii.gz")
>>> # Creates: output/sub-00_bold.nii.gz, sub-01_bold.nii.gz, etc.
```

```pycon
>>> # With BIDS-style naming using metadata
>>> bc.metadata["filename"] = [f"sub-{s}_bold.nii.gz" for s in subjects]
>>> for i, bd in enumerate(bc):
...     bd.write(f"output/{bc.metadata.loc[i, 'filename']}")
```

#### `nltools.data.BrainData`

```python
BrainData(data = None, Y = None, X = None, mask = None, masker = None, **kwargs)
```

Bases: <code>[object](#object)</code>

BrainData is a class to represent neuroimaging data in python as a vector
rather than a 3-dimensional matrix. This makes it easier to perform data
manipulation and analyses.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | nibabel data instance or list of files | <code>None</code>
`Y` |  | Pandas DataFrame of training labels | <code>None</code>
`X` |  | Pandas DataFrame Design Matrix for running univariate models | <code>None</code>
`mask` |  | binary nifti file to mask brain data | <code>None</code>
`masker` |  | nilearn masker object (e.g., ROI or searchlight extractor). Default uses voxel-level masking. | <code>None</code>
`**kwargs` |  | Additional keyword arguments passed to NiftiMasker | <code>{}</code>

**Functions:**

Name | Description
---- | -----------
[`align`](#nltools.data.BrainData.align) | Align BrainData instance to target object using functional alignment.
[`append`](#nltools.data.BrainData.append) | Append data to BrainData instance.
[`apply_mask`](#nltools.data.BrainData.apply_mask) | Mask BrainData instance using nilearn functionality.
[`astype`](#nltools.data.BrainData.astype) | Cast BrainData.data as type.
[`bootstrap`](#nltools.data.BrainData.bootstrap) | Bootstrap statistics using efficient online algorithms.
[`compute_contrasts`](#nltools.data.BrainData.compute_contrasts) | Compute contrasts from fitted GLM results.
[`copy`](#nltools.data.BrainData.copy) | Create a deep copy of a BrainData instance.
[`create_empty`](#nltools.data.BrainData.create_empty) | Create a copy of BrainData with empty data array.
[`cv`](#nltools.data.BrainData.cv) | Create a cross-validation pipeline for this BrainData.
[`decompose`](#nltools.data.BrainData.decompose) | Decompose BrainData object.
[`detrend`](#nltools.data.BrainData.detrend) | Remove linear trend from each voxel.
[`distance`](#nltools.data.BrainData.distance) | Calculate distance between images within a BrainData() instance.
[`extract_roi`](#nltools.data.BrainData.extract_roi) | Extract activity from mask or ROI atlas using NiftiLabelsMasker.
[`filter`](#nltools.data.BrainData.filter) | Apply butterworth filter to data. Wraps nilearn.signal.clean.
[`find_spikes`](#nltools.data.BrainData.find_spikes) | Identify spikes from Time Series Data.
[`fit`](#nltools.data.BrainData.fit) | Fit a model to brain imaging data.
[`icc`](#nltools.data.BrainData.icc) | Calculate voxel-wise intraclass correlation coefficient.
[`mean`](#nltools.data.BrainData.mean) | Get mean of each voxel or image.
[`median`](#nltools.data.BrainData.median) | Get median of each voxel or image.
[`multivariate_similarity`](#nltools.data.BrainData.multivariate_similarity) | Predict spatial distribution of BrainData() instance from linear
[`plot`](#nltools.data.BrainData.plot) | Plot BrainData instance using nilearn visualization or matplotlib.
[`plot_flatmap`](#nltools.data.BrainData.plot_flatmap) | Plot brain data on cortical flatmap.
[`predict`](#nltools.data.BrainData.predict) | Generate predictions using fitted model OR classify patterns (MVPA).
[`r_to_z`](#nltools.data.BrainData.r_to_z) | Apply Fisher's r to z transformation to each element of the data
[`regions`](#nltools.data.BrainData.regions) | Extract brain connected regions into separate regions.
[`regress`](#nltools.data.BrainData.regress) | Deprecated: Use fit(model='glm', X=design_matrix) instead.
[`resample_to`](#nltools.data.BrainData.resample_to) | Resample BrainData to match target image or resolution.
[`scale`](#nltools.data.BrainData.scale) | Scale data via mean scaling.
[`similarity`](#nltools.data.BrainData.similarity) | Calculate similarity of BrainData() instance with single
[`smooth`](#nltools.data.BrainData.smooth) | Apply spatial smoothing using nilearn smooth_img().
[`standardize`](#nltools.data.BrainData.standardize) | Standardize BrainData() instance.
[`std`](#nltools.data.BrainData.std) | Get standard deviation of each voxel or image.
[`sum`](#nltools.data.BrainData.sum) | Get sum of each voxel or image.
[`temporal_resample`](#nltools.data.BrainData.temporal_resample) | Resample BrainData timeseries to a new target frequency or number of samples.
[`threshold`](#nltools.data.BrainData.threshold) | Threshold BrainData instance with optional cluster filtering.
[`to_nifti`](#nltools.data.BrainData.to_nifti) | Convert BrainData Instance into Nifti Object.
[`transform_pairwise`](#nltools.data.BrainData.transform_pairwise) | Transform data into pairwise comparisons.
[`upload_neurovault`](#nltools.data.BrainData.upload_neurovault) | Upload Data to Neurovault.  Will add any columns in self.X to image
[`write`](#nltools.data.BrainData.write) | Write out BrainData object to Nifti or HDF5 File.
[`z_to_r`](#nltools.data.BrainData.z_to_r) | Convert z score back into r value for each element of data object.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`X`](#nltools.data.BrainData.X) |  | 
[`Y`](#nltools.data.BrainData.Y) |  | 
[`data`](#nltools.data.BrainData.data) |  | 
[`design_matrix`](#nltools.data.BrainData.design_matrix) |  | 
[`dtype`](#nltools.data.BrainData.dtype) |  | Get data type of BrainData.data.
[`is_empty`](#nltools.data.BrainData.is_empty) | <code>[bool](#bool)</code> | Check if BrainData.data is empty.
[`masker`](#nltools.data.BrainData.masker) |  | 
[`shape`](#nltools.data.BrainData.shape) |  | Get images by voxels shape.
[`verbose`](#nltools.data.BrainData.verbose) |  | 

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | Neuroimaging data. Can be: - None (empty BrainData) - BrainData object - List of BrainData objects or file paths - File path (str/Path) to .nii/.nii.gz/.h5/.hdf5 - nibabel Nifti1Image object - URL to download data from | <code>None</code>
`mask` |  | Brain mask. Can be None (uses MNI template), a nibabel Nifti1Image, a file path (str/Path) to a mask file, or a template name string like ``'2mm-MNI152-2009c'`` (version: 'fsl' for default/, 'a' for nilearn/, 'c' for fmriprep/). | <code>None</code>
`masker` |  | nilearn masker object (e.g. ROI or searchlight extractor). Default will load data as voxels. | <code>None</code>
`resample` | <code>bool, default=True</code> | Whether to automatically resample data to mask space. If True, data is resampled to match mask spatial characteristics. If False, data must already be in mask space. Default True preserves backward compatibility with v0.5.1. | *required*
`interpolation` | <code>str, default='auto'</code> | Interpolation method for resampling. Options: 'auto' (detect based on data type; uses 'nearest' for discrete data like atlases/masks and 'continuous' for stat maps), 'nearest' (nearest-neighbor, preserves discrete values), 'linear' (linear interpolation), 'continuous' (higher-order spline, use for stat maps). | *required*
`**kwargs` |  | Additional arguments passed to NiftiMasker. | <code>{}</code>



##### Attributes###### `nltools.data.BrainData.X`

```python
X = X
```

###### `nltools.data.BrainData.Y`

```python
Y = Y
```

###### `nltools.data.BrainData.data`

```python
data = np.array([])
```

###### `nltools.data.BrainData.design_matrix`

```python
design_matrix = None
```

###### `nltools.data.BrainData.dtype`

```python
dtype
```

Get data type of BrainData.data.

###### `nltools.data.BrainData.is_empty`

```python
is_empty: bool
```

Check if BrainData.data is empty.

###### `nltools.data.BrainData.masker`

```python
masker = masker
```

###### `nltools.data.BrainData.shape`

```python
shape
```

Get images by voxels shape.

###### `nltools.data.BrainData.verbose`

```python
verbose = kwargs.pop('verbose', False)
```



##### Functions###### `nltools.data.BrainData.align`

```python
align(target, method = 'procrustes', axis = 0, *args, **kwargs)
```

Align BrainData instance to target object using functional alignment.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` |  | (BrainData) object to align to. | *required*
`method` |  | (str) alignment method to use ['probabilistic_srm','deterministic_srm','procrustes'] | <code>'procrustes'</code>
`axis` |  | (int) axis to align on | <code>0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (dict) a dictionary containing transformed object, transformation matrix, and the shared response matrix

**Examples:**

```pycon
>>> out = data.align(target, method='procrustes')
>>> out = data.align(target, method='probabilistic_srm', n_features=None)
```

###### `nltools.data.BrainData.append`

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

###### `nltools.data.BrainData.apply_mask`

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

###### `nltools.data.BrainData.astype`

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

###### `nltools.data.BrainData.bootstrap`

```python
bootstrap(stat, n_samples = 5000, save_boots = False, n_jobs = -1, random_state = None, percentiles = (2.5, 97.5), X_test = None, **kwargs)
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
`n_jobs` |  | (int) Number of CPU cores for parallelization. -1 means all CPUs. | <code>-1</code>
`random_state` |  | (int, optional) Random seed for reproducibility | <code>None</code>
`percentiles` |  | (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5) | <code>(2.5, 97.5)</code>
`X_test` |  | (np.ndarray, optional) Test features for 'predict' bootstrap. | <code>None</code>
`**kwargs` |  | Additional parameters (backend, max_gpu_memory_gb, etc.) | <code>{}</code>

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

###### `nltools.data.BrainData.compute_contrasts`

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

###### `nltools.data.BrainData.copy`

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

###### `nltools.data.BrainData.create_empty`

```python
create_empty()
```

Create a copy of BrainData with empty data array.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | A copy of this object with an empty data array.

###### `nltools.data.BrainData.cv`

```python
cv(k: int | None = None, scheme: str = 'kfold', split_by: str | None = None, groups: np.ndarray | None = None, random_state: int | None = None, **kwargs: int | None) -> BrainDataPipeline
```

Create a cross-validation pipeline for this BrainData.

Returns a Pipeline object that enables fluent, chainable transforms
with cross-validation. Terminal methods like .predict() execute the
pipeline and return results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`k` | <code>[int](#int) \| None</code> | Number of folds (for kfold scheme). Defaults to 5. | <code>None</code>
`scheme` | <code>[str](#str)</code> | CV scheme type. Options: - 'kfold': k-fold cross-validation (default) - 'loro': leave-one-run-out (requires split_by='runs' or groups) - 'bootstrap': bootstrap with out-of-bag test sets | <code>'kfold'</code>
`split_by` | <code>[str](#str) \| None</code> | Attribute name for group splits (e.g., 'runs'). | <code>None</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Explicit group labels for CV splits. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`**kwargs` |  | Additional arguments passed to CVScheme. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainDataPipeline` | <code>[BrainDataPipeline](#nltools.data.braindata.pipeline.BrainDataPipeline)</code> | A pipeline object for method chaining.

**Examples:**

```pycon
>>> result = brain.cv(k=5).predict(y, algorithm='ridge')
>>> result = brain.cv(scheme='loro', groups=run_labels).predict(y)
```

###### `nltools.data.BrainData.decompose`

```python
decompose(algorithm = 'pca', axis = 'voxels', n_components = None, *args, **kwargs)
```

Decompose BrainData object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`algorithm` |  | (str) Algorithm to perform decomposition         types=['pca','ica','nnmf','fa','dictionary','kernelpca'] | <code>'pca'</code>
`axis` |  | dimension to decompose ['voxels','images'] | <code>'voxels'</code>
`n_components` |  | (int) number of components. If None then retain         as many as possible. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`output` |  | a dictionary of decomposition parameters

###### `nltools.data.BrainData.detrend`

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

###### `nltools.data.BrainData.distance`

```python
distance(metric = 'euclidean', **kwargs)
```

Calculate distance between images within a BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` |  | (str) type of distance metric (can use any scipy.spatial.distance     metric supported by cdist) | <code>'euclidean'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | Pairwise distance matrix.

###### `nltools.data.BrainData.extract_roi`

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

###### `nltools.data.BrainData.filter`

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

###### `nltools.data.BrainData.find_spikes`

```python
find_spikes(global_spike_cutoff = 3, diff_spike_cutoff = 3)
```

Identify spikes from Time Series Data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`global_spike_cutoff` | <code>[int](#int) or None</code> | cutoff to identify spikes in global signal in standard deviations, or None to skip. | <code>3</code>
`diff_spike_cutoff` | <code>[int](#int) or None</code> | cutoff to identify spikes in average frame difference in standard deviations, or None to skip. | <code>3</code>

**Returns:**

Type | Description
---- | -----------
 | pandas dataframe with spikes as indicator variables

###### `nltools.data.BrainData.fit`

```python
fit(model = None, X = None, cv = None, inplace = True, progress_bar = None, scale = True, scale_value = 100.0, **kwargs)
```

Fit a model to brain imaging data.

Creates and fits a model from string specification. The brain data
(self.data) is always used as the target variable. Model and results
are stored for later use with predict().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model` | <code>[str](#str)</code> | Model type: 'ridge', 'glm', or future model names | <code>None</code>
`X` | <code>[array](#array) - [like](#like) or [DataFrame](#DataFrame)</code> | Design matrix or feature matrix | <code>None</code>
`cv` | <code>int, 'auto', or sklearn CV splitter</code> | Cross-validation specification (Ridge only) | <code>None</code>
`inplace` | <code>bool, default=True</code> | If True, mutate self and return self. If False, return Fit dataclass with results (self unchanged). | <code>True</code>
`progress_bar` | <code>[bool](#bool)</code> | Display progress bar during fitting. | <code>None</code>
`scale` | <code>bool, default=True</code> | Apply grand-mean scaling before fitting. | <code>True</code>
`scale_value` | <code>float, default=100.0</code> | Target value for mean after scaling. | <code>100.0</code>
`**kwargs` | <code>[dict](#dict)</code> | Additional arguments passed to model constructor | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or Fit: If ``inplace=True``, returns self (fitted BrainData). If ``inplace=False``, returns Fit dataclass with results.

**Examples:**

```pycon
>>> brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
>>> fit = brain_data.fit(model='ridge', alpha=1.0, X=features, inplace=False)
```

###### `nltools.data.BrainData.icc`

```python
icc(n_subjects, n_sessions, icc_type = 'icc2', parallel = None, n_jobs = -1, max_gpu_memory_gb = 4.0)
```

Calculate voxel-wise intraclass correlation coefficient.

ICC Formulas based on Shrout & Fleiss (1979).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_subjects` |  | Number of subjects in the data | *required*
`n_sessions` |  | Number of sessions per subject | *required*
`icc_type` |  | Type of ICC ('icc1', 'icc2', 'icc3'). Default: 'icc2' | <code>'icc2'</code>
`parallel` |  | Parallelization method (None, 'cpu', 'gpu') | <code>None</code>
`n_jobs` |  | Number of CPU cores (-1 = all cores) | <code>-1</code>
`max_gpu_memory_gb` |  | GPU memory budget in GB | <code>4.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with ICC map (shape: (1, n_voxels))

**Examples:**

```pycon
>>> icc_map = data.icc(n_subjects=20, n_sessions=3, icc_type='icc2')
```

###### `nltools.data.BrainData.mean`

```python
mean(axis = 0)
```

Get mean of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array) | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Mean values.

###### `nltools.data.BrainData.median`

```python
median(axis = 0)
```

Get median of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array) | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Median values.

###### `nltools.data.BrainData.multivariate_similarity`

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

###### `nltools.data.BrainData.plot`

```python
plot(kind = 'glass', thr_upper = None, thr_lower = None, threshold = None, cut_coords = None, cmap = None, bg_img = None, ax = None, title = None, colorbar = True, save = None, stat = 'mean', **kwargs)
```

Plot BrainData instance using nilearn visualization or matplotlib.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`kind` | <code>[str](#str)</code> | Visualization type: 'glass', 'slices', 'timeseries', 'histogram' | <code>'glass'</code>
`thr_upper` | <code>[str](#str) / [float](#float)</code> | Upper threshold. | <code>None</code>
`thr_lower` | <code>[str](#str) / [float](#float)</code> | Lower threshold. | <code>None</code>
`threshold` | <code>[float](#float)</code> | Convenience parameter for thresholding. | <code>None</code>
`cut_coords` | <code>[list](#list)</code> | Cut coordinates for multi-slice views. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Colormap name. | <code>None</code>
`bg_img` | <code>str/nibabel image</code> | Background image. | <code>None</code>
`ax` | <code>[Axes](#matplotlib.axes.Axes)</code> | Matplotlib axis. | <code>None</code>
`title` | <code>[str](#str)</code> | Plot title. | <code>None</code>
`colorbar` | <code>[bool](#bool)</code> | Whether to show colorbar. Default: True. | <code>True</code>
`save` | <code>[str](#str)</code> | Path to save figure(s). | <code>None</code>
`stat` | <code>[str](#str)</code> | Statistic for timeseries plots. Default: 'mean'. | <code>'mean'</code>
`**kwargs` |  | Additional arguments passed to nilearn plot functions. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | Display or matplotlib Figure.

###### `nltools.data.BrainData.plot_flatmap`

```python
plot_flatmap(threshold = None, cmap = 'RdBu_r', vmax = None, vmin = None, template = 'fsaverage5', with_curvature = True, curvature_contrast = 0.5, curvature_brightness = 0.5, colorbar = True, colorbar_orientation = 'horizontal', figsize = (12, 6), title = None, radius = 3.0, interpolation = 'linear', axes = None, save = None)
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
`colorbar` | <code>[bool](#bool)</code> | Show colorbar. Default: True. | <code>True</code>
`colorbar_orientation` | <code>[str](#str)</code> | 'horizontal' or 'vertical'. Default: 'horizontal'. | <code>'horizontal'</code>
`figsize` | <code>[tuple](#tuple)</code> | Figure size as (width, height). Default: (12, 6). | <code>(12, 6)</code>
`title` | <code>[str](#str)</code> | Figure title. | <code>None</code>
`radius` | <code>[float](#float)</code> | Sampling radius in mm. Default: 3.0. | <code>3.0</code>
`interpolation` | <code>[str](#str)</code> | Interpolation method. Default: 'linear'. | <code>'linear'</code>
`axes` | <code>[Axes](#matplotlib.axes.Axes)</code> | Existing axes to plot on. | <code>None</code>
`save` | <code>[str](#str)</code> | File path to save figure. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

###### `nltools.data.BrainData.predict`

```python
predict(X: np.ndarray | None = None, y: np.ndarray | None = None, method: str = 'whole_brain', estimator: str = 'svm', cv: str = 5, groups: np.ndarray | None = None, roi_mask: np.ndarray | None = None, radius: float = 10.0, scoring: str = 'accuracy', standardize: bool = True, n_jobs: int = -1, show_progress: bool = True)
```

Generate predictions using fitted model OR classify patterns (MVPA).

Two modes:
1. **Timeseries prediction** (X provided): Use fitted ridge model to predict voxel responses.
2. **MVPA decoding** (y provided): Train a classifier to predict labels from brain patterns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray) \| None</code> | Features for timeseries prediction, shape (n_samples, n_features). | <code>None</code>
`y` | <code>[ndarray](#numpy.ndarray) \| None</code> | Labels for MVPA decoding, shape (n_samples,). | <code>None</code>
`method` | <code>[str](#str)</code> | Decoding method - 'whole_brain', 'searchlight', or 'roi'. | <code>'whole_brain'</code>
`estimator` |  | Classifier ('svm', 'logistic', 'ridge', 'lda', or sklearn estimator). | <code>'svm'</code>
`cv` |  | Cross-validation specification. | <code>5</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Group labels for CV. | <code>None</code>
`roi_mask` |  | Atlas/parcellation for ROI-based decoding. | <code>None</code>
`radius` | <code>[float](#float)</code> | Searchlight radius in mm (default 10.0). | <code>10.0</code>
`scoring` | <code>[str](#str)</code> | Metric for evaluation. | <code>'accuracy'</code>
`standardize` | <code>[bool](#bool)</code> | Z-score features before classification (default True). | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs (-1 = all cores). | <code>-1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar for searchlight. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Predicted timeseries or accuracy map.

**Examples:**

```pycon
>>> brain_data.fit(model='ridge', X=features)
>>> predictions = brain_data.predict(X=new_features)
>>> accuracy = brain_data.predict(y=labels, method='searchlight')
```

###### `nltools.data.BrainData.r_to_z`

```python
r_to_z()
```

Apply Fisher's r to z transformation to each element of the data
object.

###### `nltools.data.BrainData.regions`

```python
regions(min_region_size = 1350, extract_type = 'local_regions', smoothing_fwhm = 6, is_mask = False)
```

Extract brain connected regions into separate regions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`min_region_size` | <code>[int](#int)</code> | Minimum volume in mm3 for a region to be kept. | <code>1350</code>
`extract_type` | <code>[str](#str)</code> | Type of extraction method                 ['connected_components', 'local_regions']. | <code>'local_regions'</code>
`smoothing_fwhm` | <code>[scalar](#scalar)</code> | Smooth an image to extract more sparser regions. | <code>6</code>
`is_mask` | <code>[bool](#bool)</code> | Whether to treat as boolean mask. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with extracted ROIs as data.

###### `nltools.data.BrainData.regress`

```python
regress(design_matrix = None, noise_model = 'ols', mode = None, **kwargs)
```

Deprecated: Use fit(model='glm', X=design_matrix) instead.

.. deprecated:: 0.6.0
    Use :meth:`fit` with ``model='glm'`` instead.

###### `nltools.data.BrainData.resample_to`

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

###### `nltools.data.BrainData.scale`

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

###### `nltools.data.BrainData.similarity`

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

###### `nltools.data.BrainData.smooth`

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

###### `nltools.data.BrainData.standardize`

```python
standardize(axis = 0, method = 'center')
```

Standardize BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>[int](#int)</code> | 0 standardizes each voxel across observations (default). 1 standardizes each observation across voxels. | <code>0</code>
`method` | <code>[str](#str)</code> | 'center' subtracts the mean (default). 'zscore' subtracts the mean and divides by standard deviation. | <code>'center'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Standardized BrainData instance.

###### `nltools.data.BrainData.std`

```python
std(axis = 0)
```

Get standard deviation of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array) | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Standard deviation values.

###### `nltools.data.BrainData.sum`

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

###### `nltools.data.BrainData.temporal_resample`

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

###### `nltools.data.BrainData.threshold`

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

###### `nltools.data.BrainData.to_nifti`

```python
to_nifti()
```

Convert BrainData Instance into Nifti Object.

**Returns:**

Type | Description
---- | -----------
 | nibabel.Nifti1Image: Brain data as a NIfTI image.

###### `nltools.data.BrainData.transform_pairwise`

```python
transform_pairwise()
```

Transform data into pairwise comparisons.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance transformed into pairwise comparisons

###### `nltools.data.BrainData.upload_neurovault`

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

###### `nltools.data.BrainData.write`

```python
write(file_name)
```

Write out BrainData object to Nifti or HDF5 File.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>[str](#str) or [Path](#Path)</code> | Output file path (.nii/.nii.gz for NIfTI, .h5/.hdf5 for HDF5). | *required*

###### `nltools.data.BrainData.z_to_r`

```python
z_to_r()
```

Convert z score back into r value for each element of data object.

#### `nltools.data.DesignMatrix`

```python
DesignMatrix(data: Union[pl.DataFrame, pd.DataFrame, np.ndarray, dict, None] = None, *, sampling_freq: Optional[float] = None, columns: Optional[List[str]] = None, convolved: Optional[List[str]] = None, polys: Optional[List[str]] = None)
```

Polars-based design matrix for experimental designs in neuroimaging.

Wraps a Polars DataFrame with neuroimaging-specific metadata and methods.
Uses composition pattern (not subclassing) for clean metadata preservation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>DataFrame, ndarray, dict, or None</code> | Input data. Accepts: - Polars DataFrame (zero-copy) - pandas DataFrame (converted to Polars) - numpy ndarray - dict (keys=columns, values=data) - None (empty initialization) | <code>None</code>
`sampling_freq` | <code>[float](#float)</code> | Sampling frequency in Hz (1/TR for fMRI data) | <code>None</code>
`columns` | <code>list of str</code> | Column names (used with ndarray input) | <code>None</code>
`convolved` | <code>list of str</code> | Names of convolved columns (tracked internally) | <code>None</code>
`polys` | <code>list of str</code> | Names of polynomial columns (tracked internally) | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`sampling_freq`](#nltools.data.DesignMatrix.sampling_freq) | <code>[float](#float) or None</code> | Sampling frequency in Hz
[`convolved`](#nltools.data.DesignMatrix.convolved) | <code>list of str</code> | Columns that have been convolved
[`polys`](#nltools.data.DesignMatrix.polys) | <code>list of str</code> | Polynomial/nuisance columns (intercept, trends, DCT bases)
[`multi`](#nltools.data.DesignMatrix.multi) | <code>[bool](#bool)</code> | True if created from multi-run concatenation

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
>>> # Convolve with HRF
>>> dm_conv = dm.convolve()
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

**Functions:**

Name | Description
---- | -----------
[`add_dct_basis`](#nltools.data.DesignMatrix.add_dct_basis) | Add discrete cosine transform basis functions (high-pass filter).
[`add_poly`](#nltools.data.DesignMatrix.add_poly) | Add Legendre polynomial drift terms.
[`append`](#nltools.data.DesignMatrix.append) | Concatenate design matrices.
[`clean`](#nltools.data.DesignMatrix.clean) | Remove highly correlated columns.
[`convolve`](#nltools.data.DesignMatrix.convolve) | Convolve columns with HRF or custom kernel.
[`copy`](#nltools.data.DesignMatrix.copy) | Create a deep copy of the DesignMatrix.
[`details`](#nltools.data.DesignMatrix.details) | Return human-readable metadata summary.
[`downsample`](#nltools.data.DesignMatrix.downsample) | Reduce temporal resolution to target frequency using Polars-native operations.
[`drop`](#nltools.data.DesignMatrix.drop) | Drop specified columns.
[`fillna`](#nltools.data.DesignMatrix.fillna) | Fill NaN/null values with specified value.
[`heatmap`](#nltools.data.DesignMatrix.heatmap) | Visualize design matrix as heatmap (SPM-style).
[`replace_data`](#nltools.data.DesignMatrix.replace_data) | Replace data columns while preserving polynomials and metadata.
[`reset_index`](#nltools.data.DesignMatrix.reset_index) | Reset index (pandas compatibility method).
[`standardize`](#nltools.data.DesignMatrix.standardize) | Standardize columns using the specified method.
[`sum`](#nltools.data.DesignMatrix.sum) | Compute sum along axis.
[`to_numpy`](#nltools.data.DesignMatrix.to_numpy) | Convert DesignMatrix to numpy array.
[`to_pandas`](#nltools.data.DesignMatrix.to_pandas) | Convert DesignMatrix to pandas DataFrame.
[`upsample`](#nltools.data.DesignMatrix.upsample) | Increase temporal resolution to target frequency.
[`vif`](#nltools.data.DesignMatrix.vif) | Compute variance inflation factor for each column.
[`write`](#nltools.data.DesignMatrix.write) | Write DesignMatrix to file.
[`zscore`](#nltools.data.DesignMatrix.zscore) | Z-score standardize columns (mean=0, std=1).



##### Attributes###### `nltools.data.DesignMatrix.columns`

```python
columns: List[str]
```

Column names of the design matrix as a list of strings.

###### `nltools.data.DesignMatrix.convolved`

```python
convolved = convolved if convolved is not None else []
```

###### `nltools.data.DesignMatrix.is_empty`

```python
is_empty: bool
```

Check if DesignMatrix has no data.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`bool` | <code>[bool](#bool)</code> | True if the design matrix is empty, False otherwise.

###### `nltools.data.DesignMatrix.multi`

```python
multi = False
```

###### `nltools.data.DesignMatrix.polys`

```python
polys = polys if polys is not None else []
```

###### `nltools.data.DesignMatrix.sampling_freq`

```python
sampling_freq = sampling_freq
```

###### `nltools.data.DesignMatrix.shape`

```python
shape: tuple
```

Return (n_rows, n_cols) tuple.



##### Functions###### `nltools.data.DesignMatrix.add_dct_basis`

```python
add_dct_basis(duration: float = 180, drop: int = 0) -> DesignMatrix
```

Add discrete cosine transform basis functions (high-pass filter).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`duration` | <code>[float](#float)</code> | Filter duration in seconds. Default: 180. | <code>180</code>
`drop` | <code>[int](#int)</code> | Number of low-frequency bases to drop. Default: 0. | <code>0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with DCT basis columns appended.

###### `nltools.data.DesignMatrix.add_poly`

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

###### `nltools.data.DesignMatrix.append`

```python
append(dm: Union[DesignMatrix, List[DesignMatrix]], axis: int = 0, keep_separate: bool = True, unique_cols: Optional[List[str]] = None, fill_na: Union[int, float] = 0, verbose: bool = False) -> DesignMatrix
```

Concatenate design matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>DesignMatrix or list of DesignMatrix</code> | Design matrix/matrices to append. | *required*
`axis` | <code>[int](#int)</code> | 0 for row-wise (vertical), 1 for column-wise (horizontal). Default: 0. | <code>0</code>
`keep_separate` | <code>[bool](#bool)</code> | Whether to separate polynomial columns across runs (only applies when axis=0). Default: True. | <code>True</code>
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | <code>None</code>
`fill_na` | <code>[int](#int) or [float](#float)</code> | Value to fill NaN values during vertical concatenation. Default: 0. | <code>0</code>
`verbose` | <code>[bool](#bool)</code> | Print messages about polynomial separation. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Concatenated design matrix.

###### `nltools.data.DesignMatrix.clean`

```python
clean(fill_na: Union[int, float, None] = 0, exclude_polys: bool = False, thresh: float = 0.95, verbose: bool = True) -> DesignMatrix
```

Remove highly correlated columns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fill_na` | <code>int, float, or None</code> | Fill NaN values before checking correlations (default 0) | <code>0</code>
`exclude_polys` | <code>[bool](#bool)</code> | Skip polynomial columns from correlation check | <code>False</code>
`thresh` | <code>[float](#float)</code> | Correlation threshold (drop if abs(r) >= thresh, default 0.95) | <code>0.95</code>
`verbose` | <code>[bool](#bool)</code> | Print dropped column names | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Cleaned matrix with highly correlated columns removed

###### `nltools.data.DesignMatrix.convolve`

```python
convolve(conv_func: Union[str, np.ndarray] = 'hrf', columns: Optional[List[str]] = None) -> DesignMatrix
```

Convolve columns with HRF or custom kernel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`conv_func` | <code>[str](#str) or [ndarray](#ndarray)</code> | 'hrf' for canonical Glover HRF, or custom kernel(s). Can be 1D array (single kernel) or 2D (samples x kernels) | <code>'hrf'</code>
`columns` | <code>list of str</code> | Columns to convolve (default: all non-polynomial columns) | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with convolved columns

###### `nltools.data.DesignMatrix.copy`

```python
copy() -> DesignMatrix
```

Create a deep copy of the DesignMatrix.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Copy of the current DesignMatrix

###### `nltools.data.DesignMatrix.details`

```python
details() -> str
```

Return human-readable metadata summary.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` | <code>[str](#str)</code> | Formatted string showing sampling_freq, shape, convolved columns, and polynomial columns

###### `nltools.data.DesignMatrix.downsample`

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

###### `nltools.data.DesignMatrix.drop`

```python
drop(columns: List[str]) -> DesignMatrix
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

###### `nltools.data.DesignMatrix.fillna`

```python
fillna(value: Union[int, float]) -> DesignMatrix
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

###### `nltools.data.DesignMatrix.heatmap`

```python
heatmap(figsize: tuple = (8, 6), **kwargs: tuple)
```

Visualize design matrix as heatmap (SPM-style).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`figsize` | <code>tuple, default=(8, 6)</code> | Figure size (width, height) in inches | <code>(8, 6)</code>
`**kwargs` |  | Additional keyword arguments passed to seaborn.heatmap() | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.axes.Axes: The axes object containing the heatmap

###### `nltools.data.DesignMatrix.replace_data`

```python
replace_data(data: np.ndarray, column_names: Optional[List[str]] = None) -> DesignMatrix
```

Replace data columns while preserving polynomials and metadata.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#ndarray)</code> | New data array (must match number of rows in current DesignMatrix) | *required*
`column_names` | <code>list of str</code> | Names for new data columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with replaced data columns, preserved polynomials

###### `nltools.data.DesignMatrix.reset_index`

```python
reset_index(drop: bool = True) -> DesignMatrix
```

Reset index (pandas compatibility method).

Polars DataFrames don't have row indexes, so this is a no-op.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`drop` | <code>bool, default=True</code> | Ignored. Kept for API compatibility. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Returns self unchanged

###### `nltools.data.DesignMatrix.standardize`

```python
standardize(method: str = 'zscore', columns: Optional[List[str]] = None) -> DesignMatrix
```

Standardize columns using the specified method.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Standardization method ('zscore' or 'center'). Default: 'zscore'. | <code>'zscore'</code>
`columns` | <code>[Optional](#typing.Optional)[[List](#typing.List)[[str](#str)]]</code> | Columns to standardize. If None, standardize all non-polynomial columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with standardized columns.

###### `nltools.data.DesignMatrix.sum`

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

###### `nltools.data.DesignMatrix.to_numpy`

```python
to_numpy() -> np.ndarray
```

Convert DesignMatrix to numpy array.

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | np.ndarray: 2D array with shape (n_samples, n_columns)

###### `nltools.data.DesignMatrix.to_pandas`

```python
to_pandas() -> pd.DataFrame
```

Convert DesignMatrix to pandas DataFrame.

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#pandas.DataFrame)</code> | pd.DataFrame: Pandas DataFrame with same data and column names.

###### `nltools.data.DesignMatrix.upsample`

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

###### `nltools.data.DesignMatrix.vif`

```python
vif(exclude_polys: bool = True) -> np.ndarray | None
```

Compute variance inflation factor for each column.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`exclude_polys` | <code>[bool](#bool)</code> | Skip polynomial columns. Default: True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray) \| None</code> | np.ndarray: VIF values for each included column. Returns None if the correlation matrix is singular.

###### `nltools.data.DesignMatrix.write`

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

###### `nltools.data.DesignMatrix.zscore`

```python
zscore(columns: Optional[List[str]] = None) -> DesignMatrix
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

#### `nltools.data.Fit`

```python
Fit(fitted_values: np.ndarray, weights: Optional[np.ndarray] = None, scores: Optional[np.ndarray] = None, betas: Optional[np.ndarray] = None, t_stats: Optional[np.ndarray] = None, p_values: Optional[np.ndarray] = None, se: Optional[np.ndarray] = None, residuals: Optional[np.ndarray] = None, r2: Optional[np.ndarray] = None, cv_scores: Optional[np.ndarray] = None, cv_mean_score: Optional[np.ndarray] = None, cv_predictions: Optional[np.ndarray] = None, cv_folds: Optional[np.ndarray] = None, cv_best_alpha: Optional[float] = None, cv_alpha_scores: Optional[np.ndarray] = None) -> None
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

**Functions:**

Name | Description
---- | -----------
[`asdict`](#nltools.data.Fit.asdict) | Convert to dictionary.
[`available`](#nltools.data.Fit.available) | Return list of non-None attribute names.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`betas`](#nltools.data.Fit.betas) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`cv_alpha_scores`](#nltools.data.Fit.cv_alpha_scores) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`cv_best_alpha`](#nltools.data.Fit.cv_best_alpha) | <code>[Optional](#typing.Optional)[[float](#float)]</code> | 
[`cv_folds`](#nltools.data.Fit.cv_folds) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`cv_mean_score`](#nltools.data.Fit.cv_mean_score) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`cv_predictions`](#nltools.data.Fit.cv_predictions) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`cv_scores`](#nltools.data.Fit.cv_scores) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`fitted_values`](#nltools.data.Fit.fitted_values) | <code>[ndarray](#numpy.ndarray)</code> | 
[`p_values`](#nltools.data.Fit.p_values) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`r2`](#nltools.data.Fit.r2) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`residuals`](#nltools.data.Fit.residuals) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`scores`](#nltools.data.Fit.scores) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`se`](#nltools.data.Fit.se) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`t_stats`](#nltools.data.Fit.t_stats) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`weights`](#nltools.data.Fit.weights) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 



##### Attributes###### `nltools.data.Fit.betas`

```python
betas: Optional[np.ndarray] = None
```

###### `nltools.data.Fit.cv_alpha_scores`

```python
cv_alpha_scores: Optional[np.ndarray] = None
```

###### `nltools.data.Fit.cv_best_alpha`

```python
cv_best_alpha: Optional[float] = None
```

###### `nltools.data.Fit.cv_folds`

```python
cv_folds: Optional[np.ndarray] = None
```

###### `nltools.data.Fit.cv_mean_score`

```python
cv_mean_score: Optional[np.ndarray] = None
```

###### `nltools.data.Fit.cv_predictions`

```python
cv_predictions: Optional[np.ndarray] = None
```

###### `nltools.data.Fit.cv_scores`

```python
cv_scores: Optional[np.ndarray] = None
```

###### `nltools.data.Fit.fitted_values`

```python
fitted_values: np.ndarray
```

###### `nltools.data.Fit.p_values`

```python
p_values: Optional[np.ndarray] = None
```

###### `nltools.data.Fit.r2`

```python
r2: Optional[np.ndarray] = None
```

###### `nltools.data.Fit.residuals`

```python
residuals: Optional[np.ndarray] = None
```

###### `nltools.data.Fit.scores`

```python
scores: Optional[np.ndarray] = None
```

###### `nltools.data.Fit.se`

```python
se: Optional[np.ndarray] = None
```

###### `nltools.data.Fit.t_stats`

```python
t_stats: Optional[np.ndarray] = None
```

###### `nltools.data.Fit.weights`

```python
weights: Optional[np.ndarray] = None
```



##### Functions###### `nltools.data.Fit.asdict`

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

###### `nltools.data.Fit.available`

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

#### `nltools.data.Roc`

```python
Roc(input_values = None, binary_outcome = None, threshold_type = 'optimal_overall', forced_choice = None, **kwargs)
```

Bases: <code>[object](#object)</code>

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

**Functions:**

Name | Description
---- | -----------
[`calculate`](#nltools.data.Roc.calculate) | Calculate Receiver Operating Characteristic plot (ROC) for
[`plot`](#nltools.data.Roc.plot) | Create ROC Plot
[`summary`](#nltools.data.Roc.summary) | Display a formatted summary of ROC analysis.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`binary_outcome`](#nltools.data.Roc.binary_outcome) |  | 
[`forced_choice`](#nltools.data.Roc.forced_choice) |  | 
[`input_values`](#nltools.data.Roc.input_values) |  | 
[`threshold_type`](#nltools.data.Roc.threshold_type) |  | 



##### Attributes###### `nltools.data.Roc.binary_outcome`

```python
binary_outcome = deepcopy(binary_outcome)
```

###### `nltools.data.Roc.forced_choice`

```python
forced_choice = deepcopy(forced_choice)
```

###### `nltools.data.Roc.input_values`

```python
input_values = deepcopy(input_values)
```

###### `nltools.data.Roc.threshold_type`

```python
threshold_type = deepcopy(threshold_type)
```



##### Functions###### `nltools.data.Roc.calculate`

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

###### `nltools.data.Roc.plot`

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

###### `nltools.data.Roc.summary`

```python
summary()
```

Display a formatted summary of ROC analysis.

#### `nltools.data.SimulateGrid`

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
[`data`](#nltools.data.SimulateGrid.data) |  | The simulated data array of shape (n_subjects, grid_width, grid_width).
[`t_values`](#nltools.data.SimulateGrid.t_values) |  | T-statistic values after fitting.
[`p_values`](#nltools.data.SimulateGrid.p_values) |  | P-values after fitting.
[`thresholded`](#nltools.data.SimulateGrid.thresholded) |  | Thresholded statistical map.
[`isfit`](#nltools.data.SimulateGrid.isfit) |  | Whether fit() has been called.

**Examples:**

```pycon
>>> from nltools.data.simulator import SimulateGrid
>>> sim = SimulateGrid(signal_amplitude=0.5, random_state=42)
>>> sim.fit(n_permute=1000)
>>> sim.plot()
```

**Functions:**

Name | Description
---- | -----------
[`add_signal`](#nltools.data.SimulateGrid.add_signal) | Add rectangular signal to self.data
[`create_mask`](#nltools.data.SimulateGrid.create_mask) | Create a mask for where the signal is located in grid.
[`fit`](#nltools.data.SimulateGrid.fit) | Run ttest on self.data
[`plot_grid_simulation`](#nltools.data.SimulateGrid.plot_grid_simulation) | Create a plot of the simulations
[`run_multiple_simulations`](#nltools.data.SimulateGrid.run_multiple_simulations) | This method will run multiple simulations to calculate overall false positive rate
[`threshold_simulation`](#nltools.data.SimulateGrid.threshold_simulation) | Threshold simulation



##### Attributes###### `nltools.data.SimulateGrid.correction`

```python
correction = None
```

###### `nltools.data.SimulateGrid.data`

```python
data = self._create_noise()
```

###### `nltools.data.SimulateGrid.grid_width`

```python
grid_width = grid_width
```

###### `nltools.data.SimulateGrid.isfit`

```python
isfit = False
```

###### `nltools.data.SimulateGrid.n_subjects`

```python
n_subjects = n_subjects
```

###### `nltools.data.SimulateGrid.p_values`

```python
p_values = None
```

###### `nltools.data.SimulateGrid.random_state`

```python
random_state = check_random_state(random_state)
```

###### `nltools.data.SimulateGrid.sigma`

```python
sigma = sigma
```

###### `nltools.data.SimulateGrid.signal_amplitude`

```python
signal_amplitude = None
```

###### `nltools.data.SimulateGrid.signal_mask`

```python
signal_mask = None
```

###### `nltools.data.SimulateGrid.t_values`

```python
t_values = None
```

###### `nltools.data.SimulateGrid.threshold`

```python
threshold = None
```

###### `nltools.data.SimulateGrid.threshold_type`

```python
threshold_type = None
```

###### `nltools.data.SimulateGrid.thresholded`

```python
thresholded = None
```



##### Functions###### `nltools.data.SimulateGrid.add_signal`

```python
add_signal(signal_width = 20, signal_amplitude = 1)
```

Add rectangular signal to self.data

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`signal_width` | <code>[int](#int)</code> | width of signal box | <code>20</code>
`signal_amplitude` | <code>[int](#int)</code> | intensity of signal | <code>1</code>

###### `nltools.data.SimulateGrid.create_mask`

```python
create_mask(signal_width)
```

Create a mask for where the signal is located in grid.

###### `nltools.data.SimulateGrid.fit`

```python
fit()
```

Run ttest on self.data

###### `nltools.data.SimulateGrid.plot_grid_simulation`

```python
plot_grid_simulation(threshold, threshold_type, n_simulations = 100, correction = None)
```

Create a plot of the simulations

###### `nltools.data.SimulateGrid.run_multiple_simulations`

```python
run_multiple_simulations(threshold, threshold_type, n_simulations = 100, correction = None)
```

This method will run multiple simulations to calculate overall false positive rate

###### `nltools.data.SimulateGrid.threshold_simulation`

```python
threshold_simulation(threshold, threshold_type, correction = None)
```

Threshold simulation

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`threshold` | <code>[float](#float)</code> | threshold to apply to simulation | *required*
`threshhold_type` | <code>[str](#str)</code> | type of threshold to use can be a specific t-value or p-value ['t', 'p', 'q'] | *required*

#### `nltools.data.Simulator`

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
[`brain_mask`](#nltools.data.Simulator.brain_mask) |  | The brain mask image used for simulation.
[`nifti_masker`](#nltools.data.Simulator.nifti_masker) |  | Fitted NiftiMasker for converting between 4D data and 2D arrays.
[`output_dir`](#nltools.data.Simulator.output_dir) |  | Output directory path.
[`random_state`](#nltools.data.Simulator.random_state) |  | Random state for reproducible simulations.

**Examples:**

```pycon
>>> from nltools.data.simulator import Simulator
>>> sim = Simulator(random_state=42)
>>> # Create a dataset with signal in specific regions
>>> data = sim.create_data(y=[1, -1, 1, -1], sigma=1, n_reps=10)
```

**Functions:**

Name | Description
---- | -----------
[`create_cov_data`](#nltools.data.Simulator.create_cov_data) | create continuous simulated data with covariance
[`create_data`](#nltools.data.Simulator.create_data) | create simulated data with integers
[`create_ncov_data`](#nltools.data.Simulator.create_ncov_data) | create continuous simulated data with covariance
[`gaussian`](#nltools.data.Simulator.gaussian) | create a 3D gaussian signal normalized to a given intensity
[`n_spheres`](#nltools.data.Simulator.n_spheres) | generate a set of spheres in the brain mask space
[`normal_noise`](#nltools.data.Simulator.normal_noise) | produce a normal noise distribution for all all points in the brain mask
[`sphere`](#nltools.data.Simulator.sphere) | create a sphere of given radius at some point p in the brain mask
[`to_nifti`](#nltools.data.Simulator.to_nifti) | convert a numpy matrix to the nifti format and assign to it the brain_mask's affine matrix



##### Attributes###### `nltools.data.Simulator.brain_mask`

```python
brain_mask = brain_mask
```

###### `nltools.data.Simulator.nifti_masker`

```python
nifti_masker = NiftiMasker(mask_img=(self.brain_mask))
```

###### `nltools.data.Simulator.output_dir`

```python
output_dir = os.path.join(os.getcwd())
```

###### `nltools.data.Simulator.random_state`

```python
random_state = check_random_state(random_state)
```



##### Functions###### `nltools.data.Simulator.create_cov_data`

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

###### `nltools.data.Simulator.create_data`

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

###### `nltools.data.Simulator.create_ncov_data`

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

###### `nltools.data.Simulator.gaussian`

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

###### `nltools.data.Simulator.n_spheres`

```python
n_spheres(radius, center)
```

generate a set of spheres in the brain mask space

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`radius` |  | vector of radius.  Will create multiple spheres if len(radius) > 1 | *required*
`centers` |  | a vector of sphere centers of the form [px, py, pz] or [[px1, py1, pz1], ..., [pxn, pyn, pzn]] | *required*

###### `nltools.data.Simulator.normal_noise`

```python
normal_noise(mu, sigma)
```

produce a normal noise distribution for all all points in the brain mask

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mu` |  | average value of the gaussian signal (usually set to 0) | *required*
`sigma` |  | standard deviation | *required*

###### `nltools.data.Simulator.sphere`

```python
sphere(r, p)
```

create a sphere of given radius at some point p in the brain mask

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`r` |  | radius of the sphere | *required*
`p` |  | point (in coordinates of the brain mask) of the center of the sphere | *required*

###### `nltools.data.Simulator.to_nifti`

```python
to_nifti(m)
```

convert a numpy matrix to the nifti format and assign to it the brain_mask's affine matrix

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`m` |  | the 3D numpy matrix we wish to convert to .nii | *required*



### Modules#### `nltools.data.adjacency`

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



##### Attributes###### `nltools.data.adjacency.MAX_INT`

```python
MAX_INT = np.iinfo(np.int32).max
```

###### `nltools.data.adjacency.nx`

```python
nx = attempt_to_import('networkx', 'nx')
```

###### `nltools.data.adjacency.tables`

```python
tables = attempt_to_import('tables')
```



##### Classes###### `nltools.data.adjacency.Adjacency`

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



####### Attributes######## `nltools.data.adjacency.Adjacency.Y`

```python
Y = f['Y']
```

######## `nltools.data.adjacency.Adjacency.data`

```python
data = np.array(f.root['data'])
```

######## `nltools.data.adjacency.Adjacency.is_empty`

```python
is_empty: bool
```

Check if Adjacency object is empty.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`bool` | <code>[bool](#bool)</code> | True if the adjacency matrix is empty, False otherwise.

######## `nltools.data.adjacency.Adjacency.is_single_matrix`

```python
is_single_matrix = f['is_single_matrix'][]
```

######## `nltools.data.adjacency.Adjacency.issymmetric`

```python
issymmetric = f['issymmetric'][]
```

######## `nltools.data.adjacency.Adjacency.labels`

```python
labels = list(f['labels'])
```

######## `nltools.data.adjacency.Adjacency.matrix_type`

```python
matrix_type = f['matrix_type'][].decode()
```

######## `nltools.data.adjacency.Adjacency.n_nodes`

```python
n_nodes
```

Return the number of nodes in the adjacency matrix.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`int` |  | Number of nodes (n) for an (n, n) matrix.

######## `nltools.data.adjacency.Adjacency.shape`

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

######## `nltools.data.adjacency.Adjacency.vector_shape`

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



####### Functions######## `nltools.data.adjacency.Adjacency.append`

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

######## `nltools.data.adjacency.Adjacency.bootstrap`

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

######## `nltools.data.adjacency.Adjacency.cluster_summary`

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

######## `nltools.data.adjacency.Adjacency.copy`

```python
copy()
```

Create a copy of Adjacency object.

######## `nltools.data.adjacency.Adjacency.distance`

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

######## `nltools.data.adjacency.Adjacency.distance_to_similarity`

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

######## `nltools.data.adjacency.Adjacency.generate_permutations`

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

######## `nltools.data.adjacency.Adjacency.mean`

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

######## `nltools.data.adjacency.Adjacency.median`

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

######## `nltools.data.adjacency.Adjacency.plot`

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

######## `nltools.data.adjacency.Adjacency.plot_label_distance`

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

######## `nltools.data.adjacency.Adjacency.plot_mds`

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

######## `nltools.data.adjacency.Adjacency.plot_silhouette`

```python
plot_silhouette(labels = None, ax = None, permutation_test = True, n_permute = 5000, **kwargs)
```

Create a silhouette plot

######## `nltools.data.adjacency.Adjacency.r_to_z`

```python
r_to_z()
```

Apply Fisher's r to z transformation to each element of the data
object.

######## `nltools.data.adjacency.Adjacency.regress`

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

######## `nltools.data.adjacency.Adjacency.similarity`

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

######## `nltools.data.adjacency.Adjacency.social_relations_model`

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

######## `nltools.data.adjacency.Adjacency.squareform`

```python
squareform()
```

Convert adjacency back to squareform

######## `nltools.data.adjacency.Adjacency.stats_label_distance`

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

######## `nltools.data.adjacency.Adjacency.std`

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

######## `nltools.data.adjacency.Adjacency.sum`

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

######## `nltools.data.adjacency.Adjacency.threshold`

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

######## `nltools.data.adjacency.Adjacency.to_graph`

```python
to_graph()
```

Convert Adjacency into networkx graph.  only works on
single_matrix for now.

######## `nltools.data.adjacency.Adjacency.to_square`

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

######## `nltools.data.adjacency.Adjacency.ttest`

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

######## `nltools.data.adjacency.Adjacency.write`

```python
write(file_name, method = 'long')
```

Write out Adjacency object to csv file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>[str](#str)</code> | name of file name to write | *required*
`method` | <code>[str](#str)</code> | method to write out data ['long','square'] | <code>'long'</code>

######## `nltools.data.adjacency.Adjacency.z_to_r`

```python
z_to_r()
```

Convert z score back into r value for each element of data object



##### Functions

##### Modules###### `nltools.data.adjacency.io`

I/O functions for Adjacency objects.

**Functions:**

Name | Description
---- | -----------
[`to_graph`](#nltools.data.adjacency.io.to_graph) | Convert Adjacency into networkx graph.
[`write`](#nltools.data.adjacency.io.write) | Write out Adjacency object to csv file.



####### Functions######## `nltools.data.adjacency.io.to_graph`

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

######## `nltools.data.adjacency.io.write`

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

###### `nltools.data.adjacency.modeling`

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



####### Functions######## `nltools.data.adjacency.modeling.bootstrap`

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

######## `nltools.data.adjacency.modeling.convert_bootstrap_results_to_adjacency`

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

######## `nltools.data.adjacency.modeling.generate_permutations`

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

######## `nltools.data.adjacency.modeling.regress`

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

######## `nltools.data.adjacency.modeling.social_relations_model`

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

###### `nltools.data.adjacency.plotting`

Plotting functions for Adjacency matrices.

**Functions:**

Name | Description
---- | -----------
[`plot`](#nltools.data.adjacency.plotting.plot) | Create Heatmap of Adjacency Matrix.
[`plot_mds`](#nltools.data.adjacency.plotting.plot_mds) | Plot Multidimensional Scaling.



####### Functions######## `nltools.data.adjacency.plotting.plot`

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

######## `nltools.data.adjacency.plotting.plot_mds`

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

###### `nltools.data.adjacency.stats`

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



####### Functions######## `nltools.data.adjacency.stats.cluster_summary`

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

######## `nltools.data.adjacency.stats.plot_label_distance`

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

######## `nltools.data.adjacency.stats.plot_silhouette`

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

######## `nltools.data.adjacency.stats.r_to_z`

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

######## `nltools.data.adjacency.stats.similarity`

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

######## `nltools.data.adjacency.stats.stats_label_distance`

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

######## `nltools.data.adjacency.stats.threshold`

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

######## `nltools.data.adjacency.stats.ttest`

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

######## `nltools.data.adjacency.stats.z_to_r`

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

###### `nltools.data.adjacency.utils`

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



####### Functions######## `nltools.data.adjacency.utils.apply_stat`

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

######## `nltools.data.adjacency.utils.import_single_data`

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

######## `nltools.data.adjacency.utils.perform_arithmetic`

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

######## `nltools.data.adjacency.utils.test_is_single_matrix`

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

#### `nltools.data.braindata`

NeuroLearn Brain Data
=====================

Classes to represent brain image data.

**Modules:**

Name | Description
---- | -----------
[`analysis`](#nltools.data.braindata.analysis) | BrainData analysis functions.
[`bootstrap`](#nltools.data.braindata.bootstrap) | Bootstrap functions extracted from BrainData methods.
[`cache`](#nltools.data.braindata.cache) | Disk-based caching infrastructure for expensive computations.
[`io`](#nltools.data.braindata.io) | BrainData I/O and loading functions.
[`modeling`](#nltools.data.braindata.modeling) | BrainData modeling functions.
[`neighborhoods`](#nltools.data.braindata.neighborhoods) | Spatial neighborhood computation for neuroimaging analyses.
[`pipeline`](#nltools.data.braindata.pipeline) | BrainData pipeline and cross-validation result classes.
[`plotting`](#nltools.data.braindata.plotting) | BrainData plotting functions.
[`prediction`](#nltools.data.braindata.prediction) | BrainData prediction functions.
[`utils`](#nltools.data.braindata.utils) | Shared helpers for BrainData submodules.
[`validation`](#nltools.data.braindata.validation) | Validation utilities for BrainData class.

**Classes:**

Name | Description
---- | -----------
[`BrainData`](#nltools.data.braindata.BrainData) | BrainData is a class to represent neuroimaging data in python as a vector



##### Attributes

##### Classes###### `nltools.data.braindata.BrainData`

```python
BrainData(data = None, Y = None, X = None, mask = None, masker = None, **kwargs)
```

Bases: <code>[object](#object)</code>

BrainData is a class to represent neuroimaging data in python as a vector
rather than a 3-dimensional matrix. This makes it easier to perform data
manipulation and analyses.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | nibabel data instance or list of files | <code>None</code>
`Y` |  | Pandas DataFrame of training labels | <code>None</code>
`X` |  | Pandas DataFrame Design Matrix for running univariate models | <code>None</code>
`mask` |  | binary nifti file to mask brain data | <code>None</code>
`masker` |  | nilearn masker object (e.g., ROI or searchlight extractor). Default uses voxel-level masking. | <code>None</code>
`**kwargs` |  | Additional keyword arguments passed to NiftiMasker | <code>{}</code>

**Functions:**

Name | Description
---- | -----------
[`align`](#nltools.data.braindata.BrainData.align) | Align BrainData instance to target object using functional alignment.
[`append`](#nltools.data.braindata.BrainData.append) | Append data to BrainData instance.
[`apply_mask`](#nltools.data.braindata.BrainData.apply_mask) | Mask BrainData instance using nilearn functionality.
[`astype`](#nltools.data.braindata.BrainData.astype) | Cast BrainData.data as type.
[`bootstrap`](#nltools.data.braindata.BrainData.bootstrap) | Bootstrap statistics using efficient online algorithms.
[`compute_contrasts`](#nltools.data.braindata.BrainData.compute_contrasts) | Compute contrasts from fitted GLM results.
[`copy`](#nltools.data.braindata.BrainData.copy) | Create a deep copy of a BrainData instance.
[`create_empty`](#nltools.data.braindata.BrainData.create_empty) | Create a copy of BrainData with empty data array.
[`cv`](#nltools.data.braindata.BrainData.cv) | Create a cross-validation pipeline for this BrainData.
[`decompose`](#nltools.data.braindata.BrainData.decompose) | Decompose BrainData object.
[`detrend`](#nltools.data.braindata.BrainData.detrend) | Remove linear trend from each voxel.
[`distance`](#nltools.data.braindata.BrainData.distance) | Calculate distance between images within a BrainData() instance.
[`extract_roi`](#nltools.data.braindata.BrainData.extract_roi) | Extract activity from mask or ROI atlas using NiftiLabelsMasker.
[`filter`](#nltools.data.braindata.BrainData.filter) | Apply butterworth filter to data. Wraps nilearn.signal.clean.
[`find_spikes`](#nltools.data.braindata.BrainData.find_spikes) | Identify spikes from Time Series Data.
[`fit`](#nltools.data.braindata.BrainData.fit) | Fit a model to brain imaging data.
[`icc`](#nltools.data.braindata.BrainData.icc) | Calculate voxel-wise intraclass correlation coefficient.
[`mean`](#nltools.data.braindata.BrainData.mean) | Get mean of each voxel or image.
[`median`](#nltools.data.braindata.BrainData.median) | Get median of each voxel or image.
[`multivariate_similarity`](#nltools.data.braindata.BrainData.multivariate_similarity) | Predict spatial distribution of BrainData() instance from linear
[`plot`](#nltools.data.braindata.BrainData.plot) | Plot BrainData instance using nilearn visualization or matplotlib.
[`plot_flatmap`](#nltools.data.braindata.BrainData.plot_flatmap) | Plot brain data on cortical flatmap.
[`predict`](#nltools.data.braindata.BrainData.predict) | Generate predictions using fitted model OR classify patterns (MVPA).
[`r_to_z`](#nltools.data.braindata.BrainData.r_to_z) | Apply Fisher's r to z transformation to each element of the data
[`regions`](#nltools.data.braindata.BrainData.regions) | Extract brain connected regions into separate regions.
[`regress`](#nltools.data.braindata.BrainData.regress) | Deprecated: Use fit(model='glm', X=design_matrix) instead.
[`resample_to`](#nltools.data.braindata.BrainData.resample_to) | Resample BrainData to match target image or resolution.
[`scale`](#nltools.data.braindata.BrainData.scale) | Scale data via mean scaling.
[`similarity`](#nltools.data.braindata.BrainData.similarity) | Calculate similarity of BrainData() instance with single
[`smooth`](#nltools.data.braindata.BrainData.smooth) | Apply spatial smoothing using nilearn smooth_img().
[`standardize`](#nltools.data.braindata.BrainData.standardize) | Standardize BrainData() instance.
[`std`](#nltools.data.braindata.BrainData.std) | Get standard deviation of each voxel or image.
[`sum`](#nltools.data.braindata.BrainData.sum) | Get sum of each voxel or image.
[`temporal_resample`](#nltools.data.braindata.BrainData.temporal_resample) | Resample BrainData timeseries to a new target frequency or number of samples.
[`threshold`](#nltools.data.braindata.BrainData.threshold) | Threshold BrainData instance with optional cluster filtering.
[`to_nifti`](#nltools.data.braindata.BrainData.to_nifti) | Convert BrainData Instance into Nifti Object.
[`transform_pairwise`](#nltools.data.braindata.BrainData.transform_pairwise) | Transform data into pairwise comparisons.
[`upload_neurovault`](#nltools.data.braindata.BrainData.upload_neurovault) | Upload Data to Neurovault.  Will add any columns in self.X to image
[`write`](#nltools.data.braindata.BrainData.write) | Write out BrainData object to Nifti or HDF5 File.
[`z_to_r`](#nltools.data.braindata.BrainData.z_to_r) | Convert z score back into r value for each element of data object.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`X`](#nltools.data.braindata.BrainData.X) |  | 
[`Y`](#nltools.data.braindata.BrainData.Y) |  | 
[`data`](#nltools.data.braindata.BrainData.data) |  | 
[`design_matrix`](#nltools.data.braindata.BrainData.design_matrix) |  | 
[`dtype`](#nltools.data.braindata.BrainData.dtype) |  | Get data type of BrainData.data.
[`is_empty`](#nltools.data.braindata.BrainData.is_empty) | <code>[bool](#bool)</code> | Check if BrainData.data is empty.
[`masker`](#nltools.data.braindata.BrainData.masker) |  | 
[`shape`](#nltools.data.braindata.BrainData.shape) |  | Get images by voxels shape.
[`verbose`](#nltools.data.braindata.BrainData.verbose) |  | 

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | Neuroimaging data. Can be: - None (empty BrainData) - BrainData object - List of BrainData objects or file paths - File path (str/Path) to .nii/.nii.gz/.h5/.hdf5 - nibabel Nifti1Image object - URL to download data from | <code>None</code>
`mask` |  | Brain mask. Can be None (uses MNI template), a nibabel Nifti1Image, a file path (str/Path) to a mask file, or a template name string like ``'2mm-MNI152-2009c'`` (version: 'fsl' for default/, 'a' for nilearn/, 'c' for fmriprep/). | <code>None</code>
`masker` |  | nilearn masker object (e.g. ROI or searchlight extractor). Default will load data as voxels. | <code>None</code>
`resample` | <code>bool, default=True</code> | Whether to automatically resample data to mask space. If True, data is resampled to match mask spatial characteristics. If False, data must already be in mask space. Default True preserves backward compatibility with v0.5.1. | *required*
`interpolation` | <code>str, default='auto'</code> | Interpolation method for resampling. Options: 'auto' (detect based on data type; uses 'nearest' for discrete data like atlases/masks and 'continuous' for stat maps), 'nearest' (nearest-neighbor, preserves discrete values), 'linear' (linear interpolation), 'continuous' (higher-order spline, use for stat maps). | *required*
`**kwargs` |  | Additional arguments passed to NiftiMasker. | <code>{}</code>



####### Attributes######## `nltools.data.braindata.BrainData.X`

```python
X = X
```

######## `nltools.data.braindata.BrainData.Y`

```python
Y = Y
```

######## `nltools.data.braindata.BrainData.data`

```python
data = np.array([])
```

######## `nltools.data.braindata.BrainData.design_matrix`

```python
design_matrix = None
```

######## `nltools.data.braindata.BrainData.dtype`

```python
dtype
```

Get data type of BrainData.data.

######## `nltools.data.braindata.BrainData.is_empty`

```python
is_empty: bool
```

Check if BrainData.data is empty.

######## `nltools.data.braindata.BrainData.masker`

```python
masker = masker
```

######## `nltools.data.braindata.BrainData.shape`

```python
shape
```

Get images by voxels shape.

######## `nltools.data.braindata.BrainData.verbose`

```python
verbose = kwargs.pop('verbose', False)
```



####### Functions######## `nltools.data.braindata.BrainData.align`

```python
align(target, method = 'procrustes', axis = 0, *args, **kwargs)
```

Align BrainData instance to target object using functional alignment.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` |  | (BrainData) object to align to. | *required*
`method` |  | (str) alignment method to use ['probabilistic_srm','deterministic_srm','procrustes'] | <code>'procrustes'</code>
`axis` |  | (int) axis to align on | <code>0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (dict) a dictionary containing transformed object, transformation matrix, and the shared response matrix

**Examples:**

```pycon
>>> out = data.align(target, method='procrustes')
>>> out = data.align(target, method='probabilistic_srm', n_features=None)
```

######## `nltools.data.braindata.BrainData.append`

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

######## `nltools.data.braindata.BrainData.apply_mask`

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

######## `nltools.data.braindata.BrainData.astype`

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

######## `nltools.data.braindata.BrainData.bootstrap`

```python
bootstrap(stat, n_samples = 5000, save_boots = False, n_jobs = -1, random_state = None, percentiles = (2.5, 97.5), X_test = None, **kwargs)
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
`n_jobs` |  | (int) Number of CPU cores for parallelization. -1 means all CPUs. | <code>-1</code>
`random_state` |  | (int, optional) Random seed for reproducibility | <code>None</code>
`percentiles` |  | (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5) | <code>(2.5, 97.5)</code>
`X_test` |  | (np.ndarray, optional) Test features for 'predict' bootstrap. | <code>None</code>
`**kwargs` |  | Additional parameters (backend, max_gpu_memory_gb, etc.) | <code>{}</code>

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

######## `nltools.data.braindata.BrainData.compute_contrasts`

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

######## `nltools.data.braindata.BrainData.copy`

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

######## `nltools.data.braindata.BrainData.create_empty`

```python
create_empty()
```

Create a copy of BrainData with empty data array.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | A copy of this object with an empty data array.

######## `nltools.data.braindata.BrainData.cv`

```python
cv(k: int | None = None, scheme: str = 'kfold', split_by: str | None = None, groups: np.ndarray | None = None, random_state: int | None = None, **kwargs: int | None) -> BrainDataPipeline
```

Create a cross-validation pipeline for this BrainData.

Returns a Pipeline object that enables fluent, chainable transforms
with cross-validation. Terminal methods like .predict() execute the
pipeline and return results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`k` | <code>[int](#int) \| None</code> | Number of folds (for kfold scheme). Defaults to 5. | <code>None</code>
`scheme` | <code>[str](#str)</code> | CV scheme type. Options: - 'kfold': k-fold cross-validation (default) - 'loro': leave-one-run-out (requires split_by='runs' or groups) - 'bootstrap': bootstrap with out-of-bag test sets | <code>'kfold'</code>
`split_by` | <code>[str](#str) \| None</code> | Attribute name for group splits (e.g., 'runs'). | <code>None</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Explicit group labels for CV splits. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`**kwargs` |  | Additional arguments passed to CVScheme. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainDataPipeline` | <code>[BrainDataPipeline](#nltools.data.braindata.pipeline.BrainDataPipeline)</code> | A pipeline object for method chaining.

**Examples:**

```pycon
>>> result = brain.cv(k=5).predict(y, algorithm='ridge')
>>> result = brain.cv(scheme='loro', groups=run_labels).predict(y)
```

######## `nltools.data.braindata.BrainData.decompose`

```python
decompose(algorithm = 'pca', axis = 'voxels', n_components = None, *args, **kwargs)
```

Decompose BrainData object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`algorithm` |  | (str) Algorithm to perform decomposition         types=['pca','ica','nnmf','fa','dictionary','kernelpca'] | <code>'pca'</code>
`axis` |  | dimension to decompose ['voxels','images'] | <code>'voxels'</code>
`n_components` |  | (int) number of components. If None then retain         as many as possible. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`output` |  | a dictionary of decomposition parameters

######## `nltools.data.braindata.BrainData.detrend`

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

######## `nltools.data.braindata.BrainData.distance`

```python
distance(metric = 'euclidean', **kwargs)
```

Calculate distance between images within a BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` |  | (str) type of distance metric (can use any scipy.spatial.distance     metric supported by cdist) | <code>'euclidean'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | Pairwise distance matrix.

######## `nltools.data.braindata.BrainData.extract_roi`

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

######## `nltools.data.braindata.BrainData.filter`

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

######## `nltools.data.braindata.BrainData.find_spikes`

```python
find_spikes(global_spike_cutoff = 3, diff_spike_cutoff = 3)
```

Identify spikes from Time Series Data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`global_spike_cutoff` | <code>[int](#int) or None</code> | cutoff to identify spikes in global signal in standard deviations, or None to skip. | <code>3</code>
`diff_spike_cutoff` | <code>[int](#int) or None</code> | cutoff to identify spikes in average frame difference in standard deviations, or None to skip. | <code>3</code>

**Returns:**

Type | Description
---- | -----------
 | pandas dataframe with spikes as indicator variables

######## `nltools.data.braindata.BrainData.fit`

```python
fit(model = None, X = None, cv = None, inplace = True, progress_bar = None, scale = True, scale_value = 100.0, **kwargs)
```

Fit a model to brain imaging data.

Creates and fits a model from string specification. The brain data
(self.data) is always used as the target variable. Model and results
are stored for later use with predict().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model` | <code>[str](#str)</code> | Model type: 'ridge', 'glm', or future model names | <code>None</code>
`X` | <code>[array](#array) - [like](#like) or [DataFrame](#DataFrame)</code> | Design matrix or feature matrix | <code>None</code>
`cv` | <code>int, 'auto', or sklearn CV splitter</code> | Cross-validation specification (Ridge only) | <code>None</code>
`inplace` | <code>bool, default=True</code> | If True, mutate self and return self. If False, return Fit dataclass with results (self unchanged). | <code>True</code>
`progress_bar` | <code>[bool](#bool)</code> | Display progress bar during fitting. | <code>None</code>
`scale` | <code>bool, default=True</code> | Apply grand-mean scaling before fitting. | <code>True</code>
`scale_value` | <code>float, default=100.0</code> | Target value for mean after scaling. | <code>100.0</code>
`**kwargs` | <code>[dict](#dict)</code> | Additional arguments passed to model constructor | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or Fit: If ``inplace=True``, returns self (fitted BrainData). If ``inplace=False``, returns Fit dataclass with results.

**Examples:**

```pycon
>>> brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
>>> fit = brain_data.fit(model='ridge', alpha=1.0, X=features, inplace=False)
```

######## `nltools.data.braindata.BrainData.icc`

```python
icc(n_subjects, n_sessions, icc_type = 'icc2', parallel = None, n_jobs = -1, max_gpu_memory_gb = 4.0)
```

Calculate voxel-wise intraclass correlation coefficient.

ICC Formulas based on Shrout & Fleiss (1979).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_subjects` |  | Number of subjects in the data | *required*
`n_sessions` |  | Number of sessions per subject | *required*
`icc_type` |  | Type of ICC ('icc1', 'icc2', 'icc3'). Default: 'icc2' | <code>'icc2'</code>
`parallel` |  | Parallelization method (None, 'cpu', 'gpu') | <code>None</code>
`n_jobs` |  | Number of CPU cores (-1 = all cores) | <code>-1</code>
`max_gpu_memory_gb` |  | GPU memory budget in GB | <code>4.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with ICC map (shape: (1, n_voxels))

**Examples:**

```pycon
>>> icc_map = data.icc(n_subjects=20, n_sessions=3, icc_type='icc2')
```

######## `nltools.data.braindata.BrainData.mean`

```python
mean(axis = 0)
```

Get mean of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array) | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Mean values.

######## `nltools.data.braindata.BrainData.median`

```python
median(axis = 0)
```

Get median of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array) | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Median values.

######## `nltools.data.braindata.BrainData.multivariate_similarity`

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

######## `nltools.data.braindata.BrainData.plot`

```python
plot(kind = 'glass', thr_upper = None, thr_lower = None, threshold = None, cut_coords = None, cmap = None, bg_img = None, ax = None, title = None, colorbar = True, save = None, stat = 'mean', **kwargs)
```

Plot BrainData instance using nilearn visualization or matplotlib.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`kind` | <code>[str](#str)</code> | Visualization type: 'glass', 'slices', 'timeseries', 'histogram' | <code>'glass'</code>
`thr_upper` | <code>[str](#str) / [float](#float)</code> | Upper threshold. | <code>None</code>
`thr_lower` | <code>[str](#str) / [float](#float)</code> | Lower threshold. | <code>None</code>
`threshold` | <code>[float](#float)</code> | Convenience parameter for thresholding. | <code>None</code>
`cut_coords` | <code>[list](#list)</code> | Cut coordinates for multi-slice views. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Colormap name. | <code>None</code>
`bg_img` | <code>str/nibabel image</code> | Background image. | <code>None</code>
`ax` | <code>[Axes](#matplotlib.axes.Axes)</code> | Matplotlib axis. | <code>None</code>
`title` | <code>[str](#str)</code> | Plot title. | <code>None</code>
`colorbar` | <code>[bool](#bool)</code> | Whether to show colorbar. Default: True. | <code>True</code>
`save` | <code>[str](#str)</code> | Path to save figure(s). | <code>None</code>
`stat` | <code>[str](#str)</code> | Statistic for timeseries plots. Default: 'mean'. | <code>'mean'</code>
`**kwargs` |  | Additional arguments passed to nilearn plot functions. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | Display or matplotlib Figure.

######## `nltools.data.braindata.BrainData.plot_flatmap`

```python
plot_flatmap(threshold = None, cmap = 'RdBu_r', vmax = None, vmin = None, template = 'fsaverage5', with_curvature = True, curvature_contrast = 0.5, curvature_brightness = 0.5, colorbar = True, colorbar_orientation = 'horizontal', figsize = (12, 6), title = None, radius = 3.0, interpolation = 'linear', axes = None, save = None)
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
`colorbar` | <code>[bool](#bool)</code> | Show colorbar. Default: True. | <code>True</code>
`colorbar_orientation` | <code>[str](#str)</code> | 'horizontal' or 'vertical'. Default: 'horizontal'. | <code>'horizontal'</code>
`figsize` | <code>[tuple](#tuple)</code> | Figure size as (width, height). Default: (12, 6). | <code>(12, 6)</code>
`title` | <code>[str](#str)</code> | Figure title. | <code>None</code>
`radius` | <code>[float](#float)</code> | Sampling radius in mm. Default: 3.0. | <code>3.0</code>
`interpolation` | <code>[str](#str)</code> | Interpolation method. Default: 'linear'. | <code>'linear'</code>
`axes` | <code>[Axes](#matplotlib.axes.Axes)</code> | Existing axes to plot on. | <code>None</code>
`save` | <code>[str](#str)</code> | File path to save figure. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

######## `nltools.data.braindata.BrainData.predict`

```python
predict(X: np.ndarray | None = None, y: np.ndarray | None = None, method: str = 'whole_brain', estimator: str = 'svm', cv: str = 5, groups: np.ndarray | None = None, roi_mask: np.ndarray | None = None, radius: float = 10.0, scoring: str = 'accuracy', standardize: bool = True, n_jobs: int = -1, show_progress: bool = True)
```

Generate predictions using fitted model OR classify patterns (MVPA).

Two modes:
1. **Timeseries prediction** (X provided): Use fitted ridge model to predict voxel responses.
2. **MVPA decoding** (y provided): Train a classifier to predict labels from brain patterns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray) \| None</code> | Features for timeseries prediction, shape (n_samples, n_features). | <code>None</code>
`y` | <code>[ndarray](#numpy.ndarray) \| None</code> | Labels for MVPA decoding, shape (n_samples,). | <code>None</code>
`method` | <code>[str](#str)</code> | Decoding method - 'whole_brain', 'searchlight', or 'roi'. | <code>'whole_brain'</code>
`estimator` |  | Classifier ('svm', 'logistic', 'ridge', 'lda', or sklearn estimator). | <code>'svm'</code>
`cv` |  | Cross-validation specification. | <code>5</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Group labels for CV. | <code>None</code>
`roi_mask` |  | Atlas/parcellation for ROI-based decoding. | <code>None</code>
`radius` | <code>[float](#float)</code> | Searchlight radius in mm (default 10.0). | <code>10.0</code>
`scoring` | <code>[str](#str)</code> | Metric for evaluation. | <code>'accuracy'</code>
`standardize` | <code>[bool](#bool)</code> | Z-score features before classification (default True). | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs (-1 = all cores). | <code>-1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar for searchlight. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Predicted timeseries or accuracy map.

**Examples:**

```pycon
>>> brain_data.fit(model='ridge', X=features)
>>> predictions = brain_data.predict(X=new_features)
>>> accuracy = brain_data.predict(y=labels, method='searchlight')
```

######## `nltools.data.braindata.BrainData.r_to_z`

```python
r_to_z()
```

Apply Fisher's r to z transformation to each element of the data
object.

######## `nltools.data.braindata.BrainData.regions`

```python
regions(min_region_size = 1350, extract_type = 'local_regions', smoothing_fwhm = 6, is_mask = False)
```

Extract brain connected regions into separate regions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`min_region_size` | <code>[int](#int)</code> | Minimum volume in mm3 for a region to be kept. | <code>1350</code>
`extract_type` | <code>[str](#str)</code> | Type of extraction method                 ['connected_components', 'local_regions']. | <code>'local_regions'</code>
`smoothing_fwhm` | <code>[scalar](#scalar)</code> | Smooth an image to extract more sparser regions. | <code>6</code>
`is_mask` | <code>[bool](#bool)</code> | Whether to treat as boolean mask. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with extracted ROIs as data.

######## `nltools.data.braindata.BrainData.regress`

```python
regress(design_matrix = None, noise_model = 'ols', mode = None, **kwargs)
```

Deprecated: Use fit(model='glm', X=design_matrix) instead.

.. deprecated:: 0.6.0
    Use :meth:`fit` with ``model='glm'`` instead.

######## `nltools.data.braindata.BrainData.resample_to`

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

######## `nltools.data.braindata.BrainData.scale`

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

######## `nltools.data.braindata.BrainData.similarity`

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

######## `nltools.data.braindata.BrainData.smooth`

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

######## `nltools.data.braindata.BrainData.standardize`

```python
standardize(axis = 0, method = 'center')
```

Standardize BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>[int](#int)</code> | 0 standardizes each voxel across observations (default). 1 standardizes each observation across voxels. | <code>0</code>
`method` | <code>[str](#str)</code> | 'center' subtracts the mean (default). 'zscore' subtracts the mean and divides by standard deviation. | <code>'center'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Standardized BrainData instance.

######## `nltools.data.braindata.BrainData.std`

```python
std(axis = 0)
```

Get standard deviation of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array) | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Standard deviation values.

######## `nltools.data.braindata.BrainData.sum`

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

######## `nltools.data.braindata.BrainData.temporal_resample`

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

######## `nltools.data.braindata.BrainData.threshold`

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

######## `nltools.data.braindata.BrainData.to_nifti`

```python
to_nifti()
```

Convert BrainData Instance into Nifti Object.

**Returns:**

Type | Description
---- | -----------
 | nibabel.Nifti1Image: Brain data as a NIfTI image.

######## `nltools.data.braindata.BrainData.transform_pairwise`

```python
transform_pairwise()
```

Transform data into pairwise comparisons.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance transformed into pairwise comparisons

######## `nltools.data.braindata.BrainData.upload_neurovault`

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

######## `nltools.data.braindata.BrainData.write`

```python
write(file_name)
```

Write out BrainData object to Nifti or HDF5 File.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>[str](#str) or [Path](#Path)</code> | Output file path (.nii/.nii.gz for NIfTI, .h5/.hdf5 for HDF5). | *required*

######## `nltools.data.braindata.BrainData.z_to_r`

```python
z_to_r()
```

Convert z score back into r value for each element of data object.



##### Functions

##### Modules###### `nltools.data.braindata.analysis`

BrainData analysis functions.

Standalone functions extracted from BrainData class methods for similarity,
distance, masking, ROI extraction, ICC, filtering, thresholding, decomposition,
alignment, smoothing, and other analytical operations.

**Functions:**

Name | Description
---- | -----------
[`align`](#nltools.data.braindata.analysis.align) | Align BrainData instance to target object using functional alignment
[`apply_mask`](#nltools.data.braindata.analysis.apply_mask) | Mask BrainData instance using nilearn functionality.
[`check_masks`](#nltools.data.braindata.analysis.check_masks) | Check to make sure masks are the same for each dataset and if not create a union mask
[`decompose`](#nltools.data.braindata.analysis.decompose) | Decompose BrainData object
[`detrend_data`](#nltools.data.braindata.analysis.detrend_data) | Remove linear trend from each voxel
[`distance`](#nltools.data.braindata.analysis.distance) | Calculate distance between images within a BrainData() instance.
[`extract_roi`](#nltools.data.braindata.analysis.extract_roi) | Extract activity from mask or ROI atlas using NiftiLabelsMasker.
[`filter_data`](#nltools.data.braindata.analysis.filter_data) | Apply butterworth filter to data. Wraps nilearn.signal.clean.
[`find_spikes_data`](#nltools.data.braindata.analysis.find_spikes_data) | Function to identify spikes from Time Series Data
[`icc`](#nltools.data.braindata.analysis.icc) | Calculate voxel-wise intraclass correlation coefficient for data within
[`multivariate_similarity`](#nltools.data.braindata.analysis.multivariate_similarity) | Predict spatial distribution of BrainData() instance from linear
[`r_to_z`](#nltools.data.braindata.analysis.r_to_z) | Apply Fisher's r to z transformation to each element of the data
[`regions`](#nltools.data.braindata.analysis.regions) | Extract brain connected regions into separate regions.
[`scale_data`](#nltools.data.braindata.analysis.scale_data) | Scale data via mean scaling.
[`similarity`](#nltools.data.braindata.analysis.similarity) | Calculate similarity of BrainData() instance with single
[`smooth`](#nltools.data.braindata.analysis.smooth) | Apply spatial smoothing using nilearn smooth_img()
[`standardize`](#nltools.data.braindata.analysis.standardize) | Standardize BrainData() instance.
[`temporal_resample`](#nltools.data.braindata.analysis.temporal_resample) | Resample BrainData timeseries to a new target frequency or number of samples
[`threshold_data`](#nltools.data.braindata.analysis.threshold_data) | Threshold BrainData instance with optional cluster filtering.
[`transform_pairwise_data`](#nltools.data.braindata.analysis.transform_pairwise_data) | Transform BrainData into pairwise comparisons.
[`z_to_r`](#nltools.data.braindata.analysis.z_to_r) | Convert z score back into r value for each element of data object.



####### Functions######## `nltools.data.braindata.analysis.align`

```python
align(bd, target, method = 'procrustes', axis = 0, *args, **kwargs)
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
`**kwargs` |  | Additional keyword arguments passed to the alignment function. | <code>{}</code>

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

######## `nltools.data.braindata.analysis.apply_mask`

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

######## `nltools.data.braindata.analysis.check_masks`

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

######## `nltools.data.braindata.analysis.decompose`

```python
decompose(bd, algorithm = 'pca', axis = 'voxels', n_components = None, *args, **kwargs)
```

Decompose BrainData object

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`algorithm` |  | (str) Algorithm to perform decomposition         types=['pca','ica','nnmf','fa','dictionary','kernelpca'] | <code>'pca'</code>
`axis` |  | dimension to decompose ['voxels','images'] | <code>'voxels'</code>
`n_components` |  | (int) number of components. If None then retain         as many as possible (default: None). | <code>None</code>
`**kwargs` |  | Additional keyword arguments passed to the decomposition algorithm. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`output` |  | a dictionary of decomposition parameters

######## `nltools.data.braindata.analysis.detrend_data`

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

######## `nltools.data.braindata.analysis.distance`

```python
distance(bd, metric = 'euclidean', **kwargs)
```

Calculate distance between images within a BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`metric` |  | (str) type of distance metric (can use any scipy.spatial.distance     metric supported by cdist, e.g., 'euclidean', 'cityblock', 'cosine',     'correlation', 'hamming', 'jaccard', etc.) | <code>'euclidean'</code>
`**kwargs` |  | Additional arguments passed to scipy.spatial.distance.cdist. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dist` |  | (Adjacency) Outputs a 2D distance matrix.

######## `nltools.data.braindata.analysis.extract_roi`

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

######## `nltools.data.braindata.analysis.filter_data`

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

######## `nltools.data.braindata.analysis.find_spikes_data`

```python
find_spikes_data(bd, global_spike_cutoff = 3, diff_spike_cutoff = 3)
```

Function to identify spikes from Time Series Data

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`global_spike_cutoff` |  | (int,None) cutoff to identify spikes in global signal                  in standard deviations, None indicates do not calculate. | <code>3</code>
`diff_spike_cutoff` |  | (int,None) cutoff to identify spikes in average frame difference                  in standard deviations, None indicates do not calculate. | <code>3</code>

**Returns:**

Type | Description
---- | -----------
 | pd.DataFrame: DataFrame with spikes as indicator variables.

######## `nltools.data.braindata.analysis.icc`

```python
icc(bd, n_subjects, n_sessions, icc_type = 'icc2', parallel = None, n_jobs = -1, max_gpu_memory_gb = 4.0)
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
`icc_type` |  | Type of ICC to calculate     - 'icc1': One-way random effects (subjects random, sessions treated as interchangeable)     - 'icc2': Two-way random effects (subjects and sessions random) (default)     - 'icc3': Two-way mixed effects (subjects random, sessions fixed) | <code>'icc2'</code>
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
>>> icc_map = data.icc(n_subjects=20, n_sessions=3, icc_type='icc2')
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

######## `nltools.data.braindata.analysis.multivariate_similarity`

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

######## `nltools.data.braindata.analysis.r_to_z`

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

######## `nltools.data.braindata.analysis.regions`

```python
regions(bd, min_region_size = 1350, extract_type = 'local_regions', smoothing_fwhm = 6, is_mask = False)
```

Extract brain connected regions into separate regions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`min_region_size` | <code>[int](#int)</code> | Minimum volume in mm3 for a region to be                 kept. | <code>1350</code>
`extract_type` | <code>[str](#str)</code> | Type of extraction method                 ['connected_components', 'local_regions'].                 If 'connected_components', each component/region                 in the image is extracted automatically by                 labelling each region based upon the presence of                 unique features in their respective regions.                 If 'local_regions', each component/region is                 extracted based on their maximum peak value to                 define a seed marker and then using random                 walker segementation algorithm on these                 markers for region separation. | <code>'local_regions'</code>
`smoothing_fwhm` | <code>[scalar](#scalar)</code> | Smooth an image to extract more sparser                 regions. Only works for extract_type                 'local_regions'. | <code>6</code>
`is_mask` | <code>[bool](#bool)</code> | Whether the BrainData instance should be treated             as a boolean mask and if so, calls             connected_label_regions instead. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with extracted ROIs as data.

######## `nltools.data.braindata.analysis.scale_data`

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

######## `nltools.data.braindata.analysis.similarity`

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

######## `nltools.data.braindata.analysis.smooth`

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

######## `nltools.data.braindata.analysis.standardize`

```python
standardize(bd, axis = 0, method = 'center')
```

Standardize BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`axis` |  | 0 for observations 1 for voxels (default: 0) | <code>0</code>
`method` |  | ['center','zscore'] (default: 'center') | <code>'center'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Standardized BrainData instance.

######## `nltools.data.braindata.analysis.temporal_resample`

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

######## `nltools.data.braindata.analysis.threshold_data`

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

######## `nltools.data.braindata.analysis.transform_pairwise_data`

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

######## `nltools.data.braindata.analysis.z_to_r`

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

###### `nltools.data.braindata.bootstrap`

Bootstrap functions extracted from BrainData methods.

**Functions:**

Name | Description
---- | -----------
[`bootstrap`](#nltools.data.braindata.bootstrap.bootstrap) | Bootstrap statistics using efficient online algorithms.
[`convert_bootstrap_results_to_brain_data`](#nltools.data.braindata.bootstrap.convert_bootstrap_results_to_brain_data) | Convert bootstrap results dictionary to BrainData format.



####### Functions######## `nltools.data.braindata.bootstrap.bootstrap`

```python
bootstrap(bd, stat, n_samples = 5000, save_boots = False, n_jobs = -1, random_state = None, percentiles = (2.5, 97.5), X_test = None, **kwargs)
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
`n_jobs` |  | (int) Number of CPU cores for parallelization. Default: -1 (all CPUs). | <code>-1</code>
`random_state` |  | (int, optional) Random seed for reproducibility | <code>None</code>
`percentiles` |  | (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5) | <code>(2.5, 97.5)</code>
`X_test` |  | (np.ndarray, optional) Test features for 'predict' bootstrap.    Required if stat='predict' | <code>None</code>
`backend` |  | (str, optional) Backend for computation ('numpy', 'torch', 'auto').     If 'torch' and GPU available, uses optimized GPU acceleration with     inline Ridge computation (no CPU round-trips). Default: None (CPU). | *required*
`max_gpu_memory_gb` |  | (float) Maximum GPU memory to use in GB. Default: 4.0 | *required*
`**kwargs` |  | Additional keyword arguments passed to the underlying bootstrap computation functions. Also accepts ``backend`` (str) and ``max_gpu_memory_gb`` (float). | <code>{}</code>

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

This method replaces the deprecated `summarize_bootstrap()` function from
`nltools.stats`. To reproduce `summarize_bootstrap()` functionality:

**Old API (deprecated):**
>>> from nltools.stats import summarize_bootstrap
>>> bootstrap_samples = BrainData(list_of_samples)  # Multiple samples
>>> result = summarize_bootstrap(bootstrap_samples, save_weights=False)
>>> # Returns: {'mean': BrainData, 'Z': BrainData, 'p': BrainData}

**New API (recommended):**
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

######## `nltools.data.braindata.bootstrap.convert_bootstrap_results_to_brain_data`

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

###### `nltools.data.braindata.cache`

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
[`CacheManager`](#nltools.data.braindata.cache.CacheManager) | Manages disk-based caching for expensive computations.

**Functions:**

Name | Description
---- | -----------
[`clear_cache`](#nltools.data.braindata.cache.clear_cache) | Clear the nltools cache.
[`get_cache_dir`](#nltools.data.braindata.cache.get_cache_dir) | Get the nltools cache directory.
[`hash_mask`](#nltools.data.braindata.cache.hash_mask) | Compute a stable hash for a NIfTI mask image.



####### Classes######## `nltools.data.braindata.cache.CacheManager`

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

**Functions:**

Name | Description
---- | -----------
[`clear`](#nltools.data.braindata.cache.CacheManager.clear) | Clear all cached files in this category.
[`delete`](#nltools.data.braindata.cache.CacheManager.delete) | Delete a cached file.
[`exists`](#nltools.data.braindata.cache.CacheManager.exists) | Check if a cache key exists.
[`get_path`](#nltools.data.braindata.cache.CacheManager.get_path) | Get the file path for a cache key.
[`list_keys`](#nltools.data.braindata.cache.CacheManager.list_keys) | List all cached keys in this category.
[`load`](#nltools.data.braindata.cache.CacheManager.load) | Load cached data.
[`save`](#nltools.data.braindata.cache.CacheManager.save) | Save arrays to cache.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`cache_dir`](#nltools.data.braindata.cache.CacheManager.cache_dir) |  | 
[`category`](#nltools.data.braindata.cache.CacheManager.category) |  | 



######### Attributes########## `nltools.data.braindata.cache.CacheManager.cache_dir`

```python
cache_dir = get_cache_dir() / category
```

########## `nltools.data.braindata.cache.CacheManager.category`

```python
category = category
```



######### Functions########## `nltools.data.braindata.cache.CacheManager.clear`

```python
clear() -> int
```

Clear all cached files in this category.

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of files deleted

########## `nltools.data.braindata.cache.CacheManager.delete`

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

########## `nltools.data.braindata.cache.CacheManager.exists`

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

########## `nltools.data.braindata.cache.CacheManager.get_path`

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

########## `nltools.data.braindata.cache.CacheManager.list_keys`

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

########## `nltools.data.braindata.cache.CacheManager.load`

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

########## `nltools.data.braindata.cache.CacheManager.save`

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



####### Functions######## `nltools.data.braindata.cache.clear_cache`

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

######## `nltools.data.braindata.cache.get_cache_dir`

```python
get_cache_dir() -> Path
```

Get the nltools cache directory.

Returns ~/.nltools/cache, creating it if necessary.

**Returns:**

Type | Description
---- | -----------
<code>[Path](#pathlib.Path)</code> | Path to cache directory

######## `nltools.data.braindata.cache.hash_mask`

```python
hash_mask(mask_img: 'Nifti1Image') -> str
```

Compute a stable hash for a NIfTI mask image.

The hash is based on the mask's shape, affine transformation, and the
actual voxel positions. This ensures that masks with the same shape but
different voxel locations (or different affines) produce different hashes.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask_img` | <code>'Nifti1Image'</code> | NIfTI image to hash (typically a binary mask) | *required*

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

###### `nltools.data.braindata.io`

BrainData I/O and loading functions.

Standalone functions extracted from BrainData class methods for mask initialization,
data loading (from files, lists, URLs, HDF5, other BrainData objects), resampling,
writing, and uploading.

**Functions:**

Name | Description
---- | -----------
[`check_space_match`](#nltools.data.braindata.io.check_space_match) | Check if data and mask are in same space.
[`detect_and_update_mask`](#nltools.data.braindata.io.detect_and_update_mask) | Detect best matching template from data and update mask if mask was None.
[`detect_space`](#nltools.data.braindata.io.detect_space) | Detect if mask is in MNI space or native space.
[`get_interpolation`](#nltools.data.braindata.io.get_interpolation) | Get the interpolation method to use for a given image.
[`initialize_mask`](#nltools.data.braindata.io.initialize_mask) | Initialize the mask and NiftiMasker.
[`load_from_brain_data`](#nltools.data.braindata.io.load_from_brain_data) | Load data from another BrainData object.
[`load_from_file`](#nltools.data.braindata.io.load_from_file) | Load data from file path or nibabel object.
[`load_from_h5`](#nltools.data.braindata.io.load_from_h5) | Load data from HDF5 file.
[`load_from_list`](#nltools.data.braindata.io.load_from_list) | Load data from a list of BrainData objects or file paths.
[`load_from_url`](#nltools.data.braindata.io.load_from_url) | Load data from URL.
[`resample_to`](#nltools.data.braindata.io.resample_to) | Resample BrainData to match target image or resolution.
[`to_nifti`](#nltools.data.braindata.io.to_nifti) | Convert BrainData instance to a nibabel NIfTI image.
[`upload_neurovault`](#nltools.data.braindata.io.upload_neurovault) | Upload data to NeuroVault.
[`warn_if_resampling`](#nltools.data.braindata.io.warn_if_resampling) | Warn about resampling if verbose=True and resample=True.
[`write_brain_data`](#nltools.data.braindata.io.write_brain_data) | Write out BrainData object to Nifti or HDF5 File.



####### Functions######## `nltools.data.braindata.io.check_space_match`

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

######## `nltools.data.braindata.io.detect_and_update_mask`

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

######## `nltools.data.braindata.io.detect_space`

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

######## `nltools.data.braindata.io.get_interpolation`

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

######## `nltools.data.braindata.io.initialize_mask`

```python
initialize_mask(bd, mask, **kwargs)
```

Initialize the mask and NiftiMasker.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`mask` |  | Brain mask as nibabel object, file path, template name string, or None. Template name strings supported: '{res}mm-MNI152-2009{version}' (e.g., '2mm-MNI152-2009c', '3mm-MNI152-2009a', '2mm-MNI152-2009fsl') | *required*
`**kwargs` |  | Additional arguments passed to NiftiMasker. | <code>{}</code>

######## `nltools.data.braindata.io.load_from_brain_data`

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

######## `nltools.data.braindata.io.load_from_file`

```python
load_from_file(bd, data)
```

Load data from file path or nibabel object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`data` |  | File path or nibabel object. | *required*

######## `nltools.data.braindata.io.load_from_h5`

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

######## `nltools.data.braindata.io.load_from_list`

```python
load_from_list(bd, data_list)
```

Load data from a list of BrainData objects or file paths.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`data_list` |  | List of BrainData objects or file paths. | *required*

######## `nltools.data.braindata.io.load_from_url`

```python
load_from_url(bd, url)
```

Load data from URL.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`url` |  | URL to download data from. | *required*

######## `nltools.data.braindata.io.resample_to`

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

######## `nltools.data.braindata.io.to_nifti`

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

######## `nltools.data.braindata.io.upload_neurovault`

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

######## `nltools.data.braindata.io.warn_if_resampling`

```python
warn_if_resampling(bd, context = '')
```

Warn about resampling if verbose=True and resample=True.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`context` | <code>[str](#str)</code> | Context string to include in warning. Default: empty string. | <code>''</code>

######## `nltools.data.braindata.io.write_brain_data`

```python
write_brain_data(bd, file_name)
```

Write out BrainData object to Nifti or HDF5 File.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`file_name` | <code>[str](#str) or [Path](#pathlib.Path)</code> | Output file path. Supports .nii/.nii.gz (NIfTI) and .h5/.hdf5 (HDF5) formats. | *required*

###### `nltools.data.braindata.modeling`

BrainData modeling functions.

Standalone functions extracted from BrainData class methods for model fitting,
cross-validation, GLM estimation, Ridge regression, and contrast computation.

**Functions:**

Name | Description
---- | -----------
[`compute_contrasts`](#nltools.data.braindata.modeling.compute_contrasts) | Compute contrasts from fitted GLM results.
[`compute_ridge_cv`](#nltools.data.braindata.modeling.compute_ridge_cv) | Compute cross-validation results for Ridge regression.
[`cv`](#nltools.data.braindata.modeling.cv) | Create a cross-validation pipeline for this BrainData.
[`fit`](#nltools.data.braindata.modeling.fit) | Fit a model to brain imaging data.
[`fit_glm`](#nltools.data.braindata.modeling.fit_glm) | Fit GLM model and extract results (same logic as current regress()).
[`fit_ridge`](#nltools.data.braindata.modeling.fit_ridge) | Fit Ridge model and extract results.
[`parse_contrast_string`](#nltools.data.braindata.modeling.parse_contrast_string) | Parse a contrast string into a numeric contrast vector.
[`regress`](#nltools.data.braindata.modeling.regress) | Deprecated: Use fit(model='glm', X=design_matrix) instead.
[`to_fit_dataclass`](#nltools.data.braindata.modeling.to_fit_dataclass) | Convert BrainData fit results to Fit dataclass.



####### Functions######## `nltools.data.braindata.modeling.compute_contrasts`

```python
compute_contrasts(bd, contrasts, contrast_type = 't')
```

Compute contrasts from fitted GLM results.

This method computes contrasts as linear combinations of the GLM beta coefficients.
Must be called after .fit(model='glm', X=design_matrix) has been run.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`contrasts` |  | Can be:<br>- str: A string specifying the contrast using column names   e.g., "conditionA - conditionB" or "2*conditionA - conditionB - conditionC" - dict: Dictionary with contrast names as keys and contrast strings/vectors as values   e.g., {"main_effect": "conditionA - conditionB", "interaction": [1, -1, -1, 1]} - array: Numeric contrast vector matching the number of regressors   e.g., [1, -1, 0, 0] for a 4-regressor model | *required*
`contrast_type` | <code>[str](#str)</code> | Type of contrast statistic ('t' or 'F'). Default: 't' Note: Currently only 't' contrasts are supported. | <code>'t'</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or dict: If single contrast, returns BrainData object with contrast map.                If multiple contrasts (dict input), returns dict of BrainData objects.

**Examples:**

```pycon
>>> # Fit GLM model
>>> design_matrix = pd.DataFrame({
...     'intercept': np.ones(n_samples),
...     'conditionA': signal_a,
...     'conditionB': signal_b
... })
>>> brain.fit(model='glm', X=design_matrix)
>>>
>>> # Simple numeric contrast: A - B
>>> contrast1 = brain.compute_contrasts([0, 1, -1])
>>>
>>> # String-based contrast (more readable)
>>> contrast2 = brain.compute_contrasts("conditionA - conditionB")
>>>
>>> # Multiple contrasts at once
>>> contrasts = {
...     "A_vs_B": "conditionA - conditionB",
...     "avg_effect": [0, 0.5, 0.5],
...     "weighted": "2*conditionA - conditionB"
... }
>>> results = brain.compute_contrasts(contrasts)
>>> # results is a dict: {"A_vs_B": BrainData, "avg_effect": BrainData, ...}
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- String contrasts support coefficients: "2*A - B" or "0.5*A + 0.5*B"
- Column names must match design matrix columns exactly (case-sensitive)
- Contrast weights should sum to zero for proper inference in most cases

</details>

######## `nltools.data.braindata.modeling.compute_ridge_cv`

```python
compute_ridge_cv(bd, X, cv, alpha = None, alphas = None, backend = 'auto', **kwargs)
```

Compute cross-validation results for Ridge regression.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`X` | <code>[ndarray](#ndarray) or [list](#list)</code> | Training features. If ndarray, shape (n_samples, n_features). If list, list of feature spaces for banded ridge. | *required*
`cv` | <code>int, 'auto', or sklearn CV splitter</code> | Cross-validation specification | *required*
`alpha` | <code>[float](#float) or [auto](#auto)</code> | Regularization strength (extracted from model if not provided) | <code>None</code>
`alphas` | <code>[array](#array) - [like](#like)</code> | Alpha values to try for alpha selection | <code>None</code>
`backend` | <code>[str](#str)</code> | Computational backend ('numpy', 'torch', 'auto'). Default: 'auto' | <code>'auto'</code>
`**kwargs` | <code>[dict](#dict)</code> | Additional arguments (currently unused, for future extensibility) | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | Dictionary containing: - 'scores': (n_folds, n_voxels) array of R-squared per fold - 'mean_score': (n_voxels,) array of mean R-squared across folds - 'predictions': BrainData of out-of-fold predictions - 'folds': (n_samples,) array of fold indices - 'best_alpha': Selected alpha (if alpha selection performed) - 'alpha_scores': (n_folds, n_alphas, n_voxels) array (if alpha selection)

######## `nltools.data.braindata.modeling.cv`

```python
cv(bd, k = None, scheme = 'kfold', split_by = None, groups = None, random_state = None, **kwargs)
```

Create a cross-validation pipeline for this BrainData.

Returns a Pipeline object that enables fluent, chainable transforms
with cross-validation. Terminal methods like .predict() execute the
pipeline and return results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`k` |  | Number of folds (for kfold scheme). Defaults to 5. | <code>None</code>
`scheme` |  | CV scheme type. Options: - 'kfold': k-fold cross-validation (default) - 'loro': leave-one-run-out (requires split_by='runs' or groups) - 'bootstrap': bootstrap with out-of-bag test sets | <code>'kfold'</code>
`split_by` |  | Attribute name for group splits (e.g., 'runs'). If provided and groups is None, will try to get groups from bd.X[split_by] if bd.X is a DataFrame. | <code>None</code>
`groups` |  | Explicit group labels for CV splits. | <code>None</code>
`random_state` |  | Random seed for reproducibility. | <code>None</code>
`**kwargs` |  | Additional arguments passed to CVScheme. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainDataPipeline` |  | A pipeline object for method chaining.

**Examples:**

```pycon
>>> # Simple 5-fold CV with prediction
>>> result = brain.cv(k=5).predict(y, algorithm='ridge')
>>> print(f"Mean score: {result.mean_score:.3f}")
```

```pycon
>>> # With preprocessing
>>> result = (brain
...     .cv(k=5)
...     .normalize()
...     .reduce(n_components=50)
...     .predict(y))
```

```pycon
>>> # Leave-one-run-out CV
>>> result = brain.cv(scheme='loro', groups=run_labels).predict(y)
```

<details class="see-also" open markdown="1">
<summary>See Also</summary>

BrainDataPipeline: For available transforms and terminal methods.
CVScheme: For CV scheme configuration details.

</details>

######## `nltools.data.braindata.modeling.fit`

```python
fit(bd, model = None, X = None, cv = None, inplace = True, progress_bar = None, scale = True, scale_value = 100.0, **kwargs)
```

Fit a model to brain imaging data.

Creates and fits a model from string specification. The brain data
(bd.data) is always used as the target variable. Model and results
are stored for later use with predict().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`model` | <code>[str](#str)</code> | Model type: 'ridge', 'glm', or future model names | <code>None</code>
`X` | <code>[array](#array) - [like](#like) or [DataFrame](#DataFrame)</code> | Design matrix or feature matrix, shape (n_samples, n_features) - For GLM: Design matrix with regressors (n_samples must match bd.data) - For Ridge: Feature matrix for prediction (n_samples must match bd.data) | <code>None</code>
`cv` | <code>int, 'auto', or sklearn CV splitter</code> | Cross-validation specification (Ridge only): - int: Number of folds for k-fold CV (returns CV scores) - 'auto': Triggers alpha selection via CV (implies alpha='auto') - sklearn CV object: Custom CV splitter (e.g., KFold(3, shuffle=True)) - None: No CV (default, backward compatible) | <code>None</code>
`inplace` | <code>bool, default=True</code> | If True, mutate bd and return bd (backward compatible). If False, return Fit dataclass with results (bd unchanged). | <code>True</code>
`progress_bar` | <code>[bool](#bool)</code> | Display progress bar during fitting. - If None: Uses bd.verbose (default) - If True: Shows progress bar for long-running operations - If False: No progress bar | <code>None</code>
`scale` | <code>bool, default=True</code> | Apply grand-mean scaling before fitting. Calls bd.scale(scale_value) which divides all values by the global mean and multiplies by scale_value. This puts data in percent signal change units, which is standard for fMRI analysis. | <code>True</code>
`scale_value` | <code>float, default=100.0</code> | Target value for mean after scaling. Only used if scale=True. | <code>100.0</code>
`**kwargs` | <code>[dict](#dict)</code> | Additional arguments passed to model constructor - Ridge: alpha, alphas, backend, random_state - Glm: noise_model, minimize_memory, etc. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or Fit: If ``inplace=True``, returns bd (fitted BrainData). If ``inplace=False``, returns Fit dataclass with results.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`The`](#nltools.data.braindata.modeling.fit.The) | <code>following are set on bd when ``inplace=True``</code> | 
[```model_```](#nltools.data.braindata.modeling.fit.``model_``) | <code>[BaseModel](#BaseModel)</code> | Fitted model instance (Ridge, Glm, etc.)
[```X_```](#nltools.data.braindata.modeling.fit.``X_``) | <code>[ndarray](#ndarray)</code> | Training data X, stored for predict() default
[```cv_results_```](#nltools.data.braindata.modeling.fit.``cv_results_``) | <code>[dict](#dict)</code> | Cross-validation results dict with keys 'scores',
[`glm_betas`](#nltools.data.braindata.modeling.fit.glm_betas) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Beta coefficients (for model='glm')
[`glm_t`](#nltools.data.braindata.modeling.fit.glm_t) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | T-statistics (for model='glm')
[`glm_p`](#nltools.data.braindata.modeling.fit.glm_p) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | P-values (for model='glm')
[`glm_se`](#nltools.data.braindata.modeling.fit.glm_se) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Standard errors (for model='glm')
[`glm_residual`](#nltools.data.braindata.modeling.fit.glm_residual) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Residuals (for model='glm')
[`glm_predicted`](#nltools.data.braindata.modeling.fit.glm_predicted) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Fitted values (for model='glm')
[`glm_r2`](#nltools.data.braindata.modeling.fit.glm_r2) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | R-squared values (for model='glm')
[`ridge_weights`](#nltools.data.braindata.modeling.fit.ridge_weights) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Model coefficients (for model='ridge')
[`ridge_fitted_values`](#nltools.data.braindata.modeling.fit.ridge_fitted_values) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Fitted values (for model='ridge')
[`ridge_scores`](#nltools.data.braindata.modeling.fit.ridge_scores) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | R-squared scores (for model='ridge')

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

######## `nltools.data.braindata.modeling.fit_glm`

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

######## `nltools.data.braindata.modeling.fit_ridge`

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

######## `nltools.data.braindata.modeling.parse_contrast_string`

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

######## `nltools.data.braindata.modeling.regress`

```python
regress(bd, design_matrix = None, noise_model = 'ols', mode = None, **kwargs)
```

Deprecated: Use fit(model='glm', X=design_matrix) instead.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`design_matrix` |  | Design matrix (unused, raises error). | <code>None</code>
`noise_model` |  | Noise model (unused, raises error). | <code>'ols'</code>
`mode` |  | Mode (unused, raises error). | <code>None</code>
`**kwargs` |  | Additional arguments (unused, raises error). | <code>{}</code>

######## `nltools.data.braindata.modeling.to_fit_dataclass`

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

###### `nltools.data.braindata.neighborhoods`

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
[`SphereNeighborhoods`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods) | Precomputed sphere neighborhoods for a brain mask.

**Functions:**

Name | Description
---- | -----------
[`compute_searchlight_neighborhoods`](#nltools.data.braindata.neighborhoods.compute_searchlight_neighborhoods) | Compute sphere neighborhoods for all voxels in a brain mask.



####### Classes######## `nltools.data.braindata.neighborhoods.SphereNeighborhoods`

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
[`adjacency`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.adjacency) | <code>[csr_matrix](#scipy.sparse.csr_matrix)</code> | Sparse CSR matrix (n_voxels, n_voxels) where adjacency[i, j] is True if voxel j is within radius of voxel i
[`mask_hash`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.mask_hash) | <code>[str](#str)</code> | Hash of the source mask for validation
[`radius_mm`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.radius_mm) | <code>[float](#float)</code> | Radius in millimeters
[`n_voxels`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.n_voxels) | <code>[int](#int)</code> | Number of voxels in the mask

<details class="example" open markdown="1">
<summary>Example</summary>

>>> neighborhoods = compute_searchlight_neighborhoods(mask, radius_mm=10.0)
>>> print(f"Mean neighborhood size: {neighborhoods.mean_size:.1f} voxels")
>>>
>>> # Get neighbors of a specific voxel
>>> neighbor_idx = neighborhoods.get_neighbors(100)
>>> print(f"Voxel 100 has {len(neighbor_idx)} neighbors")

</details>

**Functions:**

Name | Description
---- | -----------
[`get_neighborhood_size`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.get_neighborhood_size) | Get the number of voxels in a neighborhood.
[`get_neighbors`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.get_neighbors) | Get indices of all voxels in the neighborhood of a given voxel.
[`iter_neighborhoods`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.iter_neighborhoods) | Iterate over all neighborhoods.



######### Attributes########## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.adjacency`

```python
adjacency: sparse.csr_matrix
```

########## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.mask_hash`

```python
mask_hash: str
```

########## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.max_size`

```python
max_size: int
```

Maximum neighborhood size.

########## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.mean_size`

```python
mean_size: float
```

Mean neighborhood size in voxels.

########## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.min_size`

```python
min_size: int
```

Minimum neighborhood size.

########## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.n_voxels`

```python
n_voxels: int
```

########## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.radius_mm`

```python
radius_mm: float
```



######### Functions########## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.get_neighborhood_size`

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

########## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.get_neighbors`

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

########## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.iter_neighborhoods`

```python
iter_neighborhoods(show_progress: bool = False) -> Iterator[tuple[int, np.ndarray]]
```

Iterate over all neighborhoods.

**Yields:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[int](#int), [ndarray](#numpy.ndarray)]</code> | Tuple of (center_voxel_idx, neighbor_indices) for each voxel

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`show_progress` | <code>[bool](#bool)</code> | If True, wrap iterator with tqdm progress bar | <code>False</code>



####### Functions######## `nltools.data.braindata.neighborhoods.compute_searchlight_neighborhoods`

```python
compute_searchlight_neighborhoods(mask_img: 'Nifti1Image', radius_mm: float = 10.0, use_cache: bool = True) -> SphereNeighborhoods
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
`mask_img` | <code>'Nifti1Image'</code> | NIfTI mask image defining the brain region | *required*
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

###### `nltools.data.braindata.pipeline`

BrainData pipeline and cross-validation result classes.

**Classes:**

Name | Description
---- | -----------
[`BrainDataCVResult`](#nltools.data.braindata.pipeline.BrainDataCVResult) | Cross-validation results for BrainData pipelines.
[`BrainDataPipeline`](#nltools.data.braindata.pipeline.BrainDataPipeline) | Pipeline specialized for BrainData with CV support.



####### Classes######## `nltools.data.braindata.pipeline.BrainDataCVResult`

```python
BrainDataCVResult(fold_results: list, pipeline: list)
```

Cross-validation results for BrainData pipelines.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`fold_results`](#nltools.data.braindata.pipeline.BrainDataCVResult.fold_results) |  | 
[`mean_score`](#nltools.data.braindata.pipeline.BrainDataCVResult.mean_score) | <code>[float](#float)</code> | Mean score across folds.
[`pipeline`](#nltools.data.braindata.pipeline.BrainDataCVResult.pipeline) |  | 
[`predictions`](#nltools.data.braindata.pipeline.BrainDataCVResult.predictions) | <code>[ndarray](#numpy.ndarray)</code> | All predictions in original sample order.
[`scores`](#nltools.data.braindata.pipeline.BrainDataCVResult.scores) | <code>[ndarray](#numpy.ndarray)</code> | Per-fold prediction scores as a numpy array.
[`std_score`](#nltools.data.braindata.pipeline.BrainDataCVResult.std_score) | <code>[float](#float)</code> | Standard deviation of scores.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fold_results` | <code>[list](#list)</code> | List of dicts, one per fold, each containing 'score', 'predictions', 'train_idx', 'test_idx', and 'fitted_stack'. | *required*
`pipeline` | <code>[BrainDataPipeline](#nltools.data.braindata.pipeline.BrainDataPipeline)</code> | The pipeline that produced these results. | *required*



######### Attributes########## `nltools.data.braindata.pipeline.BrainDataCVResult.fold_results`

```python
fold_results = fold_results
```

########## `nltools.data.braindata.pipeline.BrainDataCVResult.mean_score`

```python
mean_score: float
```

Mean score across folds.

########## `nltools.data.braindata.pipeline.BrainDataCVResult.pipeline`

```python
pipeline = pipeline
```

########## `nltools.data.braindata.pipeline.BrainDataCVResult.predictions`

```python
predictions: np.ndarray
```

All predictions in original sample order.

########## `nltools.data.braindata.pipeline.BrainDataCVResult.scores`

```python
scores: np.ndarray
```

Per-fold prediction scores as a numpy array.

########## `nltools.data.braindata.pipeline.BrainDataCVResult.std_score`

```python
std_score: float
```

Standard deviation of scores.

######## `nltools.data.braindata.pipeline.BrainDataPipeline`

```python
BrainDataPipeline(brain_data, cv = None, groups = None)
```

Pipeline specialized for BrainData with CV support.

Wraps the base Pipeline to handle BrainData-specific operations
like splitting by samples and accessing the underlying data array.

**Functions:**

Name | Description
---- | -----------
[`normalize`](#nltools.data.braindata.pipeline.BrainDataPipeline.normalize) | Add normalization step.
[`pipe`](#nltools.data.braindata.pipeline.BrainDataPipeline.pipe) | Add custom sklearn transformer.
[`predict`](#nltools.data.braindata.pipeline.BrainDataPipeline.predict) | Execute pipeline with CV and return prediction results.
[`reduce`](#nltools.data.braindata.pipeline.BrainDataPipeline.reduce) | Add dimensionality reduction step.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`cv`](#nltools.data.braindata.pipeline.BrainDataPipeline.cv) |  | Cross-validation splitter for this pipeline.
[`data`](#nltools.data.braindata.pipeline.BrainDataPipeline.data) |  | Get underlying data array.
[`n_steps`](#nltools.data.braindata.pipeline.BrainDataPipeline.n_steps) | <code>[int](#int)</code> | Number of processing steps in this pipeline.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_data` |  | BrainData instance to build the pipeline on. | *required*
`cv` |  | Cross-validation splitter (e.g., CVScheme instance) or None. | <code>None</code>
`groups` | <code>[array](#array) - [like](#like)</code> | Group labels for CV splits (e.g., run IDs for leave-one-run-out). | <code>None</code>



######### Attributes########## `nltools.data.braindata.pipeline.BrainDataPipeline.cv`

```python
cv
```

Cross-validation splitter for this pipeline.

**Returns:**

Type | Description
---- | -----------
 | sklearn cross-validator or None: The cross-validation strategy
 | set for this pipeline, or None if not configured.

########## `nltools.data.braindata.pipeline.BrainDataPipeline.data`

```python
data
```

Get underlying data array.

########## `nltools.data.braindata.pipeline.BrainDataPipeline.n_steps`

```python
n_steps: int
```

Number of processing steps in this pipeline.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`int` | <code>[int](#int)</code> | The count of steps added to this pipeline.



######### Functions########## `nltools.data.braindata.pipeline.BrainDataPipeline.normalize`

```python
normalize(method: str = 'zscore', **kwargs: str) -> BrainDataPipeline
```

Add normalization step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Normalization method. Default: 'zscore'. | <code>'zscore'</code>
`**kwargs` |  | Additional arguments passed to NormalizeStep. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainDataPipeline` | <code>[BrainDataPipeline](#nltools.data.braindata.pipeline.BrainDataPipeline)</code> | New pipeline with the normalization step appended.

########## `nltools.data.braindata.pipeline.BrainDataPipeline.pipe`

```python
pipe(transformer) -> BrainDataPipeline
```

Add custom sklearn transformer.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`transformer` |  | An sklearn-compatible transformer with fit/transform methods. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainDataPipeline` | <code>[BrainDataPipeline](#nltools.data.braindata.pipeline.BrainDataPipeline)</code> | New pipeline with the custom step appended.

########## `nltools.data.braindata.pipeline.BrainDataPipeline.predict`

```python
predict(y, algorithm: str = 'ridge', **kwargs: str)
```

Execute pipeline with CV and return prediction results.

This is a terminal method that executes the full pipeline.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` |  | Target variable (labels or continuous values). | *required*
`algorithm` | <code>[str](#str)</code> | Prediction algorithm. Options: - 'ridge': Ridge regression (continuous targets) - 'lasso': Lasso regression (continuous targets) - 'svr': Support Vector Regression (continuous targets) - 'svm': Support Vector Classification (categorical targets) | <code>'ridge'</code>
`**kwargs` |  | Additional arguments passed to sklearn model constructor. For classification (svm), use class_weight='balanced' to handle imbalanced classes. See sklearn documentation for all options. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | BrainDataCVResult with scores, predictions, and fold information.

**Examples:**

Basic regression::

    result = brain.cv(5).predict(continuous_y, algorithm='ridge', alpha=1.0)

Classification with balanced classes::

    result = brain.cv(5).predict(labels, algorithm='svm', class_weight='balanced')

########## `nltools.data.braindata.pipeline.BrainDataPipeline.reduce`

```python
reduce(method: str = 'pca', n_components: int | None = None, **kwargs: int | None) -> BrainDataPipeline
```

Add dimensionality reduction step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Reduction method (e.g., 'pca'). Default: 'pca'. | <code>'pca'</code>
`n_components` | <code>[int](#int)</code> | Number of components to keep. | <code>None</code>
`**kwargs` |  | Additional arguments passed to ReduceStep. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainDataPipeline` | <code>[BrainDataPipeline](#nltools.data.braindata.pipeline.BrainDataPipeline)</code> | New pipeline with the reduction step appended.

###### `nltools.data.braindata.plotting`

BrainData plotting functions.

**Functions:**

Name | Description
---- | -----------
[`auto_select_colormap`](#nltools.data.braindata.plotting.auto_select_colormap) | Auto-select colormap based on data characteristics.
[`plot_brain`](#nltools.data.braindata.plotting.plot_brain) | Plot BrainData instance using nilearn visualization or matplotlib.
[`plot_flatmap_brain`](#nltools.data.braindata.plotting.plot_flatmap_brain) | Plot brain data on cortical flatmap.
[`plot_matplotlib`](#nltools.data.braindata.plotting.plot_matplotlib) | Plot using matplotlib (timeseries or histogram).
[`prepare_save_paths`](#nltools.data.braindata.plotting.prepare_save_paths) | Prepare save paths for multiple plot outputs.



####### Functions######## `nltools.data.braindata.plotting.auto_select_colormap`

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

######## `nltools.data.braindata.plotting.plot_brain`

```python
plot_brain(bd, kind = 'glass', thr_upper = None, thr_lower = None, threshold = None, cut_coords = None, cmap = None, bg_img = None, ax = None, title = None, colorbar = True, save = None, stat = 'mean', **kwargs)
```

Plot BrainData instance using nilearn visualization or matplotlib.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`kind` | <code>[str](#str)</code> | Visualization type ('glass', 'slices', 'timeseries', 'histogram'). | <code>'glass'</code>
`thr_upper` | <code>[str](#str) / [float](#float)</code> | Upper threshold. | <code>None</code>
`thr_lower` | <code>[str](#str) / [float](#float)</code> | Lower threshold. | <code>None</code>
`threshold` | <code>[float](#float)</code> | Convenience parameter. If positive, sets thr_upper (shows values above threshold). If negative, sets thr_lower (shows values below threshold). | <code>None</code>
`cut_coords` | <code>[list](#list)</code> | Cut coordinates for multi-slice views. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Colormap name. | <code>None</code>
`bg_img` | <code>[Nifti1Image](#Nifti1Image) or [str](#str)</code> | Background image for slice views. | <code>None</code>
`ax` | <code>[Axes](#matplotlib.axes.Axes)</code> | Matplotlib axis to plot on. | <code>None</code>
`title` | <code>[str](#str)</code> | Plot title. | <code>None</code>
`colorbar` | <code>[bool](#bool)</code> | Whether to show colorbar. Default: True. | <code>True</code>
`save` | <code>[str](#str)</code> | Path to save figure(s). | <code>None</code>
`stat` | <code>[str](#str)</code> | Statistic for timeseries plots. Valid options: 'mean', 'median', 'std'. | <code>'mean'</code>
`**kwargs` |  | Additional arguments passed to nilearn plot functions. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | Display or matplotlib Figure.

######## `nltools.data.braindata.plotting.plot_flatmap_brain`

```python
plot_flatmap_brain(bd, threshold = None, cmap = 'RdBu_r', vmax = None, vmin = None, template = 'fsaverage5', with_curvature = True, curvature_contrast = 0.5, curvature_brightness = 0.5, colorbar = True, colorbar_orientation = 'horizontal', figsize = (12, 6), title = None, radius = 3.0, interpolation = 'linear', axes = None, save = None)
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
`radius` | <code>[float](#float)</code> | Sampling radius in mm for vol_to_surf. Default: 3.0. | <code>3.0</code>
`interpolation` | <code>[str](#str)</code> | Interpolation for vol_to_surf. Default: 'linear'. | <code>'linear'</code>
`axes` | <code>[Axes](#matplotlib.axes.Axes)</code> | Existing axes to plot on. | <code>None</code>
`save` | <code>[str](#str)</code> | File path to save figure. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

######## `nltools.data.braindata.plotting.plot_matplotlib`

```python
plot_matplotlib(bd, kind, stat = 'mean', ax = None, title = None, save = None)
```

Plot using matplotlib (timeseries or histogram).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`kind` | <code>[str](#str)</code> | 'timeseries' or 'histogram' | *required*
`stat` | <code>[str](#str)</code> | Statistic for timeseries ('mean', 'median', 'std') | <code>'mean'</code>
`ax` |  | Matplotlib axis. | <code>None</code>
`title` | <code>[str](#str)</code> | Plot title. | <code>None</code>
`save` | <code>[str](#str)</code> | Path to save figure. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

######## `nltools.data.braindata.plotting.prepare_save_paths`

```python
prepare_save_paths(save)
```

Prepare save paths for multiple plot outputs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`save` |  | Base save path (str or Path) | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | Dictionary with 'glass' and 'slices' keys containing save paths

###### `nltools.data.braindata.prediction`

BrainData prediction functions.

Standalone functions extracted from BrainData class methods for timeseries
prediction (encoding models) and MVPA decoding (pattern classification).

**Functions:**

Name | Description
---- | -----------
[`mvpa_roi`](#nltools.data.braindata.prediction.mvpa_roi) | ROI-based MVPA - accuracy per ROI.
[`mvpa_searchlight`](#nltools.data.braindata.prediction.mvpa_searchlight) | Searchlight MVPA - accuracy per voxel neighborhood.
[`mvpa_whole_brain`](#nltools.data.braindata.prediction.mvpa_whole_brain) | Whole-brain MVPA - single accuracy across all voxels.
[`mvpa_whole_brain_pipeline`](#nltools.data.braindata.prediction.mvpa_whole_brain_pipeline) | Whole-brain MVPA using Pipeline infrastructure.
[`predict`](#nltools.data.braindata.prediction.predict) | Generate predictions using fitted model OR classify patterns (MVPA).
[`predict_mvpa`](#nltools.data.braindata.prediction.predict_mvpa) | Perform MVPA decoding using cross-validation.
[`predict_timeseries`](#nltools.data.braindata.prediction.predict_timeseries) | Generate timeseries predictions using fitted model.
[`resolve_estimator`](#nltools.data.braindata.prediction.resolve_estimator) | Resolve string shortcut to sklearn estimator.



####### Functions######## `nltools.data.braindata.prediction.mvpa_roi`

```python
mvpa_roi(bd, X, y, pipe, cv, groups, scoring, roi_mask, n_jobs, show_progress)
```

ROI-based MVPA - accuracy per ROI.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`X` |  | Feature matrix, shape (n_samples, n_voxels). | *required*
`y` |  | Labels, shape (n_samples,). | *required*
`pipe` |  | sklearn pipeline or estimator. | *required*
`cv` |  | Cross-validation splitter. | *required*
`groups` |  | Group labels for CV. | *required*
`scoring` |  | Scoring metric string. | *required*
`roi_mask` |  | Atlas/parcellation image or path. | *required*
`n_jobs` |  | Number of parallel jobs. | *required*
`show_progress` |  | Whether to show progress bar. | *required*

**Returns:**

Type | Description
---- | -----------
 | np.ndarray of accuracy values per ROI.

######## `nltools.data.braindata.prediction.mvpa_searchlight`

```python
mvpa_searchlight(bd, X, y, pipe, cv, groups, scoring, radius, n_jobs, show_progress)
```

Searchlight MVPA - accuracy per voxel neighborhood.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`X` |  | Feature matrix, shape (n_samples, n_voxels). | *required*
`y` |  | Labels, shape (n_samples,). | *required*
`pipe` |  | sklearn pipeline or estimator. | *required*
`cv` |  | Cross-validation splitter. | *required*
`groups` |  | Group labels for CV. | *required*
`scoring` |  | Scoring metric string. | *required*
`radius` |  | Searchlight radius in mm. | *required*
`n_jobs` |  | Number of parallel jobs. | *required*
`show_progress` |  | Whether to show progress bar. | *required*

**Returns:**

Type | Description
---- | -----------
 | np.ndarray of accuracy values per voxel.

######## `nltools.data.braindata.prediction.mvpa_whole_brain`

```python
mvpa_whole_brain(bd, X, y, pipe, cv, groups, scoring)
```

Whole-brain MVPA - single accuracy across all voxels.

Legacy implementation using sklearn cross_val_score directly.
Kept for searchlight/ROI methods that still use sklearn pipelines.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance (unused, kept for API consistency). | *required*
`X` |  | Feature matrix, shape (n_samples, n_voxels). | *required*
`y` |  | Labels, shape (n_samples,). | *required*
`pipe` |  | sklearn pipeline or estimator. | *required*
`cv` |  | Cross-validation splitter. | *required*
`groups` |  | Group labels for CV. | *required*
`scoring` |  | Scoring metric string. | *required*

**Returns:**

Type | Description
---- | -----------
 | np.ndarray with single mean accuracy value.

######## `nltools.data.braindata.prediction.mvpa_whole_brain_pipeline`

```python
mvpa_whole_brain_pipeline(bd, y, estimator, cv, groups, standardize)
```

Whole-brain MVPA using Pipeline infrastructure.

Delegates to the fluent pipeline API for whole-brain classification,
then extracts mean accuracy for backward compatibility.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`y` |  | Labels to predict. | *required*
`estimator` |  | Estimator name ('svm', 'logistic', etc.). | *required*
`cv` |  | Cross-validation splitter or int. | *required*
`groups` |  | Group labels for CV. | *required*
`standardize` |  | Whether to z-score features. | *required*

**Returns:**

Type | Description
---- | -----------
 | np.ndarray with single mean accuracy value.

######## `nltools.data.braindata.prediction.predict`

```python
predict(bd, X = None, y = None, method = 'whole_brain', estimator = 'svm', cv = 5, groups = None, roi_mask = None, radius = 10.0, scoring = 'accuracy', standardize = True, n_jobs = -1, show_progress = True)
```

Generate predictions using fitted model OR classify patterns (MVPA).

This method supports two prediction modes determined by which parameter
is provided:

1. **Timeseries prediction** (X provided): Use fitted ridge model to
   predict voxel responses for new feature data.

2. **MVPA decoding** (y provided): Train a classifier to predict labels
   from brain patterns using cross-validation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`X` | <code>[array](#array) - [like](#like)</code> | Features for timeseries prediction, shape (n_samples, n_features). If None and y is None, uses training data from fit(). | <code>None</code>
`y` | <code>[array](#array) - [like](#like)</code> | Labels for MVPA decoding, shape (n_samples,). If provided, performs pattern classification instead of timeseries prediction. | <code>None</code>
`MVPA-specific parameters` | <code>only used when y is provided</code> |  | *required*
`method` | <code>[str](#str)</code> | Decoding method - 'whole_brain', 'searchlight', or 'roi'. | <code>'whole_brain'</code>
`estimator` | <code>str or sklearn estimator</code> | Classifier to use. Can be: - 'svm': LinearSVC (default) - 'logistic': LogisticRegression - 'ridge': RidgeClassifier - 'lda': LinearDiscriminantAnalysis - Any sklearn-compatible estimator with fit/predict | <code>'svm'</code>
`cv` | <code>int or sklearn CV splitter</code> | Cross-validation specification. Int for k-fold or sklearn CV object. | <code>5</code>
`groups` | <code>[array](#array) - [like](#like)</code> | Group labels for CV (e.g., run IDs for leave-one-run-out). | <code>None</code>
`roi_mask` | <code>[Nifti1Image](#Nifti1Image) or [str](#str)</code> | Atlas/parcellation for ROI-based decoding. | <code>None</code>
`radius` | <code>[float](#float)</code> | Searchlight radius in mm. Default: 10.0. | <code>10.0</code>
`scoring` | <code>[str](#str)</code> | Metric for evaluation ('accuracy', 'balanced_accuracy', 'roc_auc'). | <code>'accuracy'</code>
`standardize` | <code>[bool](#bool)</code> | Z-score features before classification. Default: True. | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs for searchlight (-1 = all cores). | <code>-1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar for searchlight. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | For timeseries prediction, shape (n_samples, n_voxels). For MVPA, shape (1, n_voxels) with accuracy per voxel/ROI.

**Examples:**

```pycon
>>> # Timeseries prediction (encoding model)
>>> brain_data.fit(model='ridge', X=features)
>>> predictions = brain_data.predict(X=new_features)
```

```pycon
>>> # MVPA decoding (pattern classification)
>>> # brain_data.data has shape (n_trials, n_voxels)
>>> accuracy = brain_data.predict(y=labels, method='searchlight')
>>> print(accuracy.shape)  # (1, n_voxels)
```

######## `nltools.data.braindata.prediction.predict_mvpa`

```python
predict_mvpa(bd, y, method = 'whole_brain', estimator = 'svm', cv = 5, groups = None, roi_mask = None, radius = 10.0, scoring = 'accuracy', standardize = True, n_jobs = -1, show_progress = True)
```

Perform MVPA decoding using cross-validation.

Internal function for pattern classification.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`y` | <code>[array](#array) - [like](#like)</code> | Labels to predict, shape (n_samples,). | *required*
`method` | <code>[str](#str)</code> | 'whole_brain', 'searchlight', or 'roi'. | <code>'whole_brain'</code>
`estimator` | <code>str or sklearn estimator</code> | Classifier (string shortcut or sklearn estimator). | <code>'svm'</code>
`cv` | <code>int or sklearn CV splitter</code> | Cross-validation specification. | <code>5</code>
`groups` | <code>[array](#array) - [like](#like)</code> | Group labels for CV. | <code>None</code>
`roi_mask` | <code>[Nifti1Image](#Nifti1Image) or [str](#str)</code> | Atlas for ROI-based decoding. | <code>None</code>
`radius` | <code>[float](#float)</code> | Searchlight radius in mm. | <code>10.0</code>
`scoring` | <code>[str](#str)</code> | Scoring metric. | <code>'accuracy'</code>
`standardize` | <code>[bool](#bool)</code> | Whether to z-score features. | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Parallel jobs for searchlight. | <code>-1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData with accuracy values.

######## `nltools.data.braindata.prediction.predict_timeseries`

```python
predict_timeseries(bd, X = None)
```

Generate timeseries predictions using fitted model.

Internal function for encoding model prediction.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`X` | <code>[array](#array) - [like](#like)</code> | Features to predict on. If None, uses training data. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData with predicted timeseries.

######## `nltools.data.braindata.prediction.resolve_estimator`

```python
resolve_estimator(bd, estimator)
```

Resolve string shortcut to sklearn estimator.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance (unused, kept for API consistency). | *required*
`estimator` |  | String shortcut or sklearn estimator object. | *required*

**Returns:**

Type | Description
---- | -----------
 | Instantiated sklearn estimator.

###### `nltools.data.braindata.utils`

Shared helpers for BrainData submodules.

These are internal utilities used by the facade and submodules — not part of the
public API.

**Functions:**

Name | Description
---- | -----------
[`apply_func`](#nltools.data.braindata.utils.apply_func) | Apply a statistical function to BrainData's ``.data`` attribute.
[`check_brain_data`](#nltools.data.braindata.utils.check_brain_data) | Check if data is a BrainData Instance, coercing Nifti1Image if needed.
[`check_brain_data_is_single`](#nltools.data.braindata.utils.check_brain_data_is_single) | Logical test if BrainData instance is a single image.
[`perform_arithmetic`](#nltools.data.braindata.utils.perform_arithmetic) | Perform an arithmetic operation with validation.
[`shallow_copy`](#nltools.data.braindata.utils.shallow_copy) | Create a shallow copy of a BrainData for efficient method chaining.



####### Functions######## `nltools.data.braindata.utils.apply_func`

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

######## `nltools.data.braindata.utils.check_brain_data`

```python
check_brain_data(data, mask = None)
```

Check if data is a BrainData Instance, coercing Nifti1Image if needed.

######## `nltools.data.braindata.utils.check_brain_data_is_single`

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

######## `nltools.data.braindata.utils.perform_arithmetic`

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

######## `nltools.data.braindata.utils.shallow_copy`

```python
shallow_copy(bd)
```

Create a shallow copy of a BrainData for efficient method chaining.

Creates a new BrainData instance that shares immutable objects (mask,
nifti_masker) but copies mutable attributes.  The data array is NOT
copied — callers should handle data copying as needed.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance to copy. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | New instance with shared/copied attributes.

###### `nltools.data.braindata.validation`

Validation utilities for BrainData class.

This module contains helper functions for validating inputs, shapes, and
compatibility between BrainData objects and other data types.

**Functions:**

Name | Description
---- | -----------
[`validate_append_shapes`](#nltools.data.braindata.validation.validate_append_shapes) | Validate shape compatibility for appending BrainData objects.
[`validate_arithmetic_operand`](#nltools.data.braindata.validation.validate_arithmetic_operand) | Validate operand type for arithmetic operations.
[`validate_brain_data_shapes`](#nltools.data.braindata.validation.validate_brain_data_shapes) | Validate shape compatibility between two BrainData objects.
[`validate_data_type`](#nltools.data.braindata.validation.validate_data_type) | Validate input data type for BrainData initialization.
[`validate_frame`](#nltools.data.braindata.validation.validate_frame) | Validate and process X or Y dataframes for BrainData.
[`validate_index_operations`](#nltools.data.braindata.validation.validate_index_operations) | Validate indexing operations for BrainData.
[`validate_list_data`](#nltools.data.braindata.validation.validate_list_data) | Validate that all items in a list are the same type.



####### Functions######## `nltools.data.braindata.validation.validate_append_shapes`

```python
validate_append_shapes(data1_shape, data2_shape)
```

Validate shape compatibility for appending BrainData objects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1_shape` |  | Shape of first BrainData. | *required*
`data2_shape` |  | Shape of second BrainData to append. | *required*

######## `nltools.data.braindata.validation.validate_arithmetic_operand`

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

######## `nltools.data.braindata.validation.validate_brain_data_shapes`

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

######## `nltools.data.braindata.validation.validate_data_type`

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

######## `nltools.data.braindata.validation.validate_frame`

```python
validate_frame(frame, data_shape = None, frame_type = 'DataFrame')
```

Validate and process X or Y dataframes for BrainData.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`frame` |  | Input to validate. Can be str, Path, pd.DataFrame, or None. | *required*
`data_shape` |  | Optional tuple of data shape to validate against. | <code>None</code>
`frame_type` |  | Type of frame for error messages (e.g., "X", "Y"). | <code>'DataFrame'</code>

**Returns:**

Type | Description
---- | -----------
 | pd.DataFrame: Validated and processed dataframe.

######## `nltools.data.braindata.validation.validate_index_operations`

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

######## `nltools.data.braindata.validation.validate_list_data`

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

#### `nltools.data.collection`

BrainCollection: Multi-subject brain data container.

Provides tensor-like semantics for efficient group analyses with lazy loading
and memory-efficient operations.

<details class="shape-semantics" open markdown="1">
<summary>(n_images, n_observations, n_voxels)</summary>

- axis 0: images (subjects, runs, etc.)
- axis 1: observations (timepoints, TRs)
- axis 2: voxels (spatial)

</details>

**Modules:**

Name | Description
---- | -----------
[`constructors`](#nltools.data.collection.constructors) | Constructor functions for BrainCollection.
[`inference`](#nltools.data.collection.inference) | BrainCollection inference functions.
[`io`](#nltools.data.collection.io) | I/O functions for BrainCollection.
[`modeling`](#nltools.data.collection.modeling) | Modeling functions extracted from BrainCollection.
[`pipeline`](#nltools.data.collection.pipeline) | Pipeline classes for BrainCollection.
[`prediction`](#nltools.data.collection.prediction) | Prediction functions extracted from BrainCollection.
[`transforms`](#nltools.data.collection.transforms) | BrainCollection transform functions.

**Classes:**

Name | Description
---- | -----------
[`BrainCollection`](#nltools.data.collection.BrainCollection) | Collection of brain images with tensor-like operations.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`T`](#nltools.data.collection.T) |  | 
[`tqdm`](#nltools.data.collection.tqdm) |  | 



##### Attributes###### `nltools.data.collection.T`

```python
T = TypeVar('T')
```

###### `nltools.data.collection.tqdm`

```python
tqdm = attempt_to_import('tqdm', 'tqdm')
```



##### Classes###### `nltools.data.collection.BrainCollection`

```python
BrainCollection(items: list[Path | str | 'BrainData'], mask: nib.Nifti1Image | Path | str, metadata: pd.DataFrame | None = None, lazy: bool = True)
```

Collection of brain images with tensor-like operations.

BrainCollection provides a container for multiple brain images (e.g., multiple
subjects or runs) with numpy-style indexing and axis operations. It supports
lazy loading for memory efficiency and integrates with pybids for BIDS datasets.

<details class="shape-semantics" open markdown="1">
<summary>(n_images, n_observations, n_voxels)</summary>

- axis 0: images (subjects, runs, etc.)
- axis 1: observations (timepoints, TRs)
- axis 2: voxels (spatial locations)

</details>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`items` | <code>[list](#list)[[Path](#pathlib.Path) \| [str](#str) \| 'BrainData']</code> | List of file paths, BrainData objects, or mix of both. Paths are loaded lazily by default. | *required*
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str)</code> | Brain mask. Required. Can be: - nibabel Nifti1Image - Path to mask file - Template name (e.g., '2mm-MNI152-2009c') | *required*
`metadata` | <code>[DataFrame](#pandas.DataFrame) \| None</code> | Optional DataFrame with per-image metadata (subject, session, etc.). Index should match items order. | <code>None</code>
`lazy` | <code>[bool](#bool)</code> | If True (default), paths are not loaded until accessed. | <code>True</code>

**Examples:**

```pycon
>>> # Create from paths (lazy loading)
>>> bc = BrainCollection(
...     ['/data/sub-01.nii.gz', '/data/sub-02.nii.gz'],
...     mask='2mm-MNI152-2009c'
... )
>>> bc.shape
(2, 100, 228453)
```

```pycon
>>> # NumPy-style indexing
>>> bc[0]  # First subject -> BrainData
>>> bc[:, 0]  # First timepoint across all subjects -> BrainCollection
>>> bc[0:5, 10:20]  # 5 subjects, timepoints 10-20 -> BrainCollection
```

```pycon
>>> # Axis operations
>>> bc.mean(axis=0)  # Mean across subjects -> BrainData
>>> bc.mean(axis=1)  # Mean across time per subject -> BrainCollection
```

```pycon
>>> # From BIDS dataset
>>> bc = BrainCollection.from_bids('/data/bids', task='rest', mask=mask)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- All images must share the same mask/space. Heterogeneous masks are not
  supported; data is resampled to mask space on load.
- Some operations (e.g., to_tensor) require uniform observation counts
  across all images.

</details>

**Functions:**

Name | Description
---- | -----------
[`align`](#nltools.data.collection.BrainCollection.align) | Align subjects using local functional alignment.
[`anova`](#nltools.data.collection.BrainCollection.anova) | One-way ANOVA across groups defined by metadata.
[`compute_contrasts`](#nltools.data.collection.BrainCollection.compute_contrasts) | Compute contrasts from fitted GLM beta coefficients.
[`cv`](#nltools.data.collection.BrainCollection.cv) | Create a cross-validation pipeline for multi-subject analysis.
[`detrend`](#nltools.data.collection.BrainCollection.detrend) | Remove trend from each image.
[`filter`](#nltools.data.collection.BrainCollection.filter) | Filter collection by predicate.
[`fit`](#nltools.data.collection.BrainCollection.fit) | Fit a model to each subject in the collection.
[`fit_from_events`](#nltools.data.collection.BrainCollection.fit_from_events) | Build design matrices from events and fit GLM to each subject.
[`fit_glm`](#nltools.data.collection.BrainCollection.fit_glm) | Fit GLM to each subject in collection.
[`fit_ridge`](#nltools.data.collection.BrainCollection.fit_ridge) | Fit ridge regression to each subject in collection.
[`from_bids`](#nltools.data.collection.BrainCollection.from_bids) | Create BrainCollection from a BIDS dataset.
[`from_glob`](#nltools.data.collection.BrainCollection.from_glob) | Create BrainCollection from glob pattern.
[`from_stacked`](#nltools.data.collection.BrainCollection.from_stacked) | Create BrainCollection by splitting a stacked BrainData.
[`isc`](#nltools.data.collection.BrainCollection.isc) | Compute intersubject correlation (ISC) across the collection.
[`isc_test`](#nltools.data.collection.BrainCollection.isc_test) | Compute ISC with permutation testing for statistical inference.
[`iter_batches`](#nltools.data.collection.BrainCollection.iter_batches) | Iterate in batches along axis.
[`load`](#nltools.data.collection.BrainCollection.load) | Load specified images into memory.
[`map`](#nltools.data.collection.BrainCollection.map) | Apply function across specified axis.
[`max`](#nltools.data.collection.BrainCollection.max) | Compute maximum along axis. See mean() for details.
[`mean`](#nltools.data.collection.BrainCollection.mean) | Compute mean along axis.
[`median`](#nltools.data.collection.BrainCollection.median) | Compute median along axis. See mean() for details.
[`memory_estimate`](#nltools.data.collection.BrainCollection.memory_estimate) | Estimate memory usage for loading all images.
[`min`](#nltools.data.collection.BrainCollection.min) | Compute minimum along axis. See mean() for details.
[`permutation_test`](#nltools.data.collection.BrainCollection.permutation_test) | One-sample permutation test across images (sign-flipping).
[`permutation_test2`](#nltools.data.collection.BrainCollection.permutation_test2) | Two-sample permutation test between collections.
[`predict`](#nltools.data.collection.BrainCollection.predict) | Generate predictions for each subject in collection.
[`select_feature`](#nltools.data.collection.BrainCollection.select_feature) | Select a single feature's weights across all subjects.
[`smooth`](#nltools.data.collection.BrainCollection.smooth) | Spatially smooth each image.
[`standardize`](#nltools.data.collection.BrainCollection.standardize) | Standardize each image.
[`std`](#nltools.data.collection.BrainCollection.std) | Compute standard deviation along axis. See mean() for details.
[`sum`](#nltools.data.collection.BrainCollection.sum) | Compute sum along axis. See mean() for details.
[`threshold`](#nltools.data.collection.BrainCollection.threshold) | Threshold each image.
[`to_list`](#nltools.data.collection.BrainCollection.to_list) | Return list of BrainData objects.
[`to_stacked`](#nltools.data.collection.BrainCollection.to_stacked) | Stack all into single BrainData (n_total_obs, n_voxels).
[`to_tensor`](#nltools.data.collection.BrainCollection.to_tensor) | Convert to numpy array (n_images, n_obs, n_voxels).
[`ttest`](#nltools.data.collection.BrainCollection.ttest) | One-sample t-test across images.
[`ttest2`](#nltools.data.collection.BrainCollection.ttest2) | Two-sample t-test between collections.
[`unload`](#nltools.data.collection.BrainCollection.unload) | Free memory for specified images (keep paths for reloading).
[`var`](#nltools.data.collection.BrainCollection.var) | Compute variance along axis. See mean() for details.
[`write`](#nltools.data.collection.BrainCollection.write) | Write all images in collection to files.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`is_loaded`](#nltools.data.collection.BrainCollection.is_loaded) | <code>[list](#list)[[bool](#bool)]</code> | List indicating which images are currently in memory.
[`mask`](#nltools.data.collection.BrainCollection.mask) | <code>[Nifti1Image](#nibabel.Nifti1Image)</code> | Shared NIfTI brain mask image used to define the voxel space for the collection.
[`metadata`](#nltools.data.collection.BrainCollection.metadata) | <code>[DataFrame](#pandas.DataFrame)</code> | Per-image metadata DataFrame.
[`n_images`](#nltools.data.collection.BrainCollection.n_images) | <code>[int](#int)</code> | Number of images in collection.
[`n_voxels`](#nltools.data.collection.BrainCollection.n_voxels) | <code>[int](#int)</code> | Number of voxels (from mask).
[`shape`](#nltools.data.collection.BrainCollection.shape) | <code>[tuple](#tuple)[[int](#int), [int](#int) \| None, [int](#int)]</code> | Shape as (n_images, n_observations, n_voxels).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`items` | <code>[list](#list)[[Path](#pathlib.Path) \| [str](#str) \| 'BrainData']</code> | List of paths or BrainData objects. | *required*
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str)</code> | Shared mask (required). Path, nibabel image, or template name. | *required*
`metadata` | <code>[DataFrame](#pandas.DataFrame) \| None</code> | Optional per-image metadata DataFrame. | <code>None</code>
`lazy` | <code>[bool](#bool)</code> | If True, paths are loaded on demand. | <code>True</code>



####### Attributes######## `nltools.data.collection.BrainCollection.is_loaded`

```python
is_loaded: list[bool]
```

List indicating which images are currently in memory.

######## `nltools.data.collection.BrainCollection.mask`

```python
mask: nib.Nifti1Image
```

Shared NIfTI brain mask image used to define the voxel space for the collection.

######## `nltools.data.collection.BrainCollection.metadata`

```python
metadata: pd.DataFrame
```

Per-image metadata DataFrame.

######## `nltools.data.collection.BrainCollection.n_images`

```python
n_images: int
```

Number of images in collection.

######## `nltools.data.collection.BrainCollection.n_voxels`

```python
n_voxels: int
```

Number of voxels (from mask).

######## `nltools.data.collection.BrainCollection.shape`

```python
shape: tuple[int, int | None, int]
```

Shape as (n_images, n_observations, n_voxels).

n_observations is None if images have variable counts or not all are loaded.



####### Functions######## `nltools.data.collection.BrainCollection.align`

```python
align(method: str = 'procrustes', scheme: str = 'searchlight', radius_mm: float = 10.0, parcellation: 'nib.Nifti1Image | None' = None, n_features: int | None = None, n_iter: int = 3, parallel: str | None = 'cpu', n_jobs: int = -1, return_model: bool = False, show_progress: bool = True) -> 'BrainCollection | tuple[BrainCollection, object]'
```

Align subjects using local functional alignment.

Performs neighborhood-based functional alignment across subjects using
LocalAlignment. Each subject's data is aligned to a common template space
using local transforms learned within searchlight spheres or parcels.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Alignment method. Options: - 'procrustes': Orthogonal Procrustes (default, preserves dimensions) - 'srm': Shared Response Model (dimensionality reduction) - 'hyperalignment': Hyperalignment (iterative Procrustes) | <code>'procrustes'</code>
`scheme` | <code>[str](#str)</code> | Spatial scheme. Options: - 'searchlight': Overlapping spheres with center-only aggregation - 'piecewise': Non-overlapping parcels (requires parcellation) | <code>'searchlight'</code>
`radius_mm` | <code>[float](#float)</code> | Sphere radius in millimeters for searchlight scheme. | <code>10.0</code>
`parcellation` | <code>'nib.Nifti1Image \| None'</code> | Parcellation image for piecewise scheme (required if scheme='piecewise'). | <code>None</code>
`n_features` | <code>[int](#int) \| None</code> | Number of features for SRM. None uses full dimensions. | <code>None</code>
`n_iter` | <code>[int](#int)</code> | Number of iterations for alignment refinement. | <code>3</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization mode. Options: - None: Single-threaded - 'cpu': CPU parallelization with joblib - 'gpu': GPU acceleration via PyTorch | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs for CPU mode (-1 = auto). | <code>-1</code>
`return_model` | <code>[bool](#bool)</code> | If True, return (aligned_collection, model) tuple for fit/transform workflow with new data. | <code>False</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| tuple[BrainCollection, object]'</code> | BrainCollection with aligned data. If return_model=True, returns
<code>'BrainCollection \| tuple[BrainCollection, object]'</code> | tuple of (aligned_collection, LocalAlignment_model).

**Examples:**

```pycon
>>> # Basic searchlight alignment
>>> aligned_bc = bc.align(method='procrustes', radius_mm=10.0)
```

```pycon
>>> # Piecewise alignment with parcellation
>>> aligned_bc = bc.align(
...     scheme='piecewise',
...     parcellation=parcellation_img,
...     method='srm',
...     n_features=50
... )
```

```pycon
>>> # Fit/transform workflow for train/test split
>>> aligned_train, model = train_bc.align(return_model=True)
>>> aligned_test = model.transform(test_data_list)
```

```pycon
>>> # GPU-accelerated alignment
>>> aligned_bc = bc.align(parallel='gpu')
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

Based on Bazeille et al. 2021 "An empirical evaluation of functional
alignment using inter-subject decoding". Center-only aggregation is
used for searchlight to preserve local orthogonality of transforms.

</details>

<details class="see-also" open markdown="1">
<summary>See Also</summary>

nltools.algorithms.alignment.LocalAlignment: Underlying alignment class.

</details>

######## `nltools.data.collection.BrainCollection.anova`

```python
anova(groups: str | list | np.ndarray) -> tuple['BrainData', 'BrainData']
```

One-way ANOVA across groups defined by metadata.

Tests whether group means differ significantly. This is the
voxel-wise equivalent of scipy.stats.f_oneway.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`groups` | <code>[str](#str) \| [list](#list) \| [ndarray](#numpy.ndarray)</code> | Group assignment for each image. Can be: - str: Column name in metadata - list/array: Group labels of length n_images | *required*

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)['BrainData', 'BrainData']</code> | Tuple of (F_stat, p_value) as BrainData objects.

**Examples:**

```pycon
>>> # Groups from metadata column
>>> f_stat, p_val = bc.anova('condition')
```

```pycon
>>> # Explicit group labels
>>> groups = ['control'] * 10 + ['patient'] * 15
>>> f_stat, p_val = bc.anova(groups)
```

######## `nltools.data.collection.BrainCollection.compute_contrasts`

```python
compute_contrasts(contrasts: 'str | dict | np.ndarray | list') -> 'BrainCollection | dict[str, BrainCollection]'
```

Compute contrasts from fitted GLM beta coefficients.

Applies contrast weights to each subject's betas and returns a
BrainCollection of contrast values suitable for group-level analysis.

Must be called on a BrainCollection created by fit_glm() which has
the _design_columns attribute set.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`contrasts` | <code>'str \| dict \| np.ndarray \| list'</code> | Can be: - str: Contrast string using column names, e.g., "face - house" - dict: Multiple contrasts, e.g., {"main": "face - house", "avg": [0.5, 0.5]} - array/list: Numeric contrast vector, e.g., [1, -1] | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection where each BrainData has shape (n_voxels,) containing
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | the contrast values. If dict input, returns dict of BrainCollections.

**Examples:**

```pycon
>>> # Fit GLM and compute contrast
>>> betas = bc.fit_glm(events=events_df, t_r=2.0)
>>> contrast = betas.compute_contrasts("face - house")
>>> # Group t-test on contrast
>>> group_result = contrast.ttest()
```

```pycon
>>> # Multiple contrasts
>>> contrasts = betas.compute_contrasts({
...     "face_vs_house": "face - house",
...     "face_vs_baseline": "face",
... })
>>> face_vs_house_ttest = contrasts["face_vs_house"].ttest()
```

######## `nltools.data.collection.BrainCollection.cv`

```python
cv(k: int | None = None, scheme: str = 'kfold', split_by: str | None = None, groups: np.ndarray | None = None, random_state: int | None = None, **kwargs: int | None) -> 'BrainCollectionPipeline'
```

Create a cross-validation pipeline for multi-subject analysis.

Returns a pipeline object that enables fluent, chainable transforms
with cross-validation across subjects or runs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`k` | <code>[int](#int) \| None</code> | Number of folds (for kfold scheme). Defaults to 5. | <code>None</code>
`scheme` | <code>[str](#str)</code> | CV scheme type. Options: - 'kfold': k-fold cross-validation on pooled data - 'loso': leave-one-subject-out (one image held out per fold) - 'loro': leave-one-run-out (requires groups) | <code>'kfold'</code>
`split_by` | <code>[str](#str) \| None</code> | Metadata column for group splits. If provided and groups is None, gets groups from self.metadata[split_by]. | <code>None</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Explicit group labels for CV splits. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`**kwargs` |  | Additional arguments passed to CVScheme. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainCollectionPipeline` | <code>'BrainCollectionPipeline'</code> | Pipeline for method chaining.

**Examples:**

```pycon
>>> # Leave-one-subject-out classification
>>> result = bc.cv(scheme='loso').normalize().predict(subject_labels, algorithm='svm')
>>> print(f"Mean accuracy: {result.mean_score:.2%}")
```

```pycon
>>> # With preprocessing
>>> result = (bc
...     .cv(scheme='loso')
...     .normalize()
...     .reduce(n_components=50)
...     .predict(labels))
```

```pycon
>>> # Run-based CV with metadata
>>> result = bc.cv(scheme='loro', split_by='run').predict(y)
```

<details class="see-also" open markdown="1">
<summary>See Also</summary>

BrainCollectionPipeline: For available transforms and terminals.
CVScheme: For CV scheme configuration details.

</details>

######## `nltools.data.collection.BrainCollection.detrend`

```python
detrend(method: str = 'linear', n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Remove trend from each image.

Delegates to BrainData.detrend() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | 'linear' or 'constant'. | <code>'linear'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with detrended images.

**Examples:**

```pycon
>>> bc.detrend()  # Remove linear trend
>>> bc.detrend(method='constant')  # Remove mean only
```

######## `nltools.data.collection.BrainCollection.filter`

```python
filter(predicate: Callable | list | np.ndarray | 'pd.Series') -> 'BrainCollection'
```

Filter collection by predicate.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`predicate` | <code>[Callable](#collections.abc.Callable) \| [list](#list) \| [ndarray](#numpy.ndarray) \| 'pd.Series'</code> | Filter condition. Can be: - callable: fn(BrainData) → bool - list/ndarray: Boolean mask of length n_images - pd.Series: Boolean series (index ignored) | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with subset of images matching predicate.

**Examples:**

```pycon
>>> # Filter by callable
>>> bc.filter(lambda bd: bd.data.mean() > 0)
```

```pycon
>>> # Filter by boolean mask
>>> mask = [True, False, True]
>>> bc.filter(mask)
```

```pycon
>>> # Filter by metadata condition
>>> bc.filter(bc.metadata['group'] == 'control')
```

######## `nltools.data.collection.BrainCollection.fit`

```python
fit(model: str, X: 'pd.DataFrame | np.ndarray | str | list', cv: int | None = None, scale: bool = True, scale_value: float = 100.0, show_progress: bool = True, **kwargs: bool) -> 'FittedBrainCollection'
```

Fit a model to each subject in the collection.

Unified fitting method that shadows BrainData.fit() API for multi-subject
analysis. Dispatches to model-specific implementations based on the
model parameter.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model` | <code>[str](#str)</code> | Model type - 'glm' or 'ridge' | *required*
`X` | <code>'pd.DataFrame \| np.ndarray \| str \| list'</code> | Design/feature matrix. Can be: - pd.DataFrame/DesignMatrix: Shared (used for all subjects) - np.ndarray: Shared array (used for all subjects) - str: Column name in metadata pointing to file paths - list: Per-subject list of DataFrames/arrays/paths | *required*
`cv` | <code>[int](#int) \| None</code> | Cross-validation folds (Ridge only). Default is None for GLM, 5 for Ridge when output='scores'. | <code>None</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`**kwargs` |  | Model-specific arguments passed to _fit_glm or _fit_ridge: - GLM: return_stats, save - Ridge: alpha, output, save, backend, random_state | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'FittedBrainCollection'</code> | FittedBrainCollection wrapping the fitted results. Supports:
<code>'FittedBrainCollection'</code> | - ``.results``: Access underlying BrainCollection(s) directly
<code>'FittedBrainCollection'</code> | - ``.betas``: Convenience accessor for beta coefficients (GLM)
<code>'FittedBrainCollection'</code> | - ``.pool()``: Aggregate across subjects for group analysis
<code>'FittedBrainCollection'</code> | The underlying results contain:
<code>'FittedBrainCollection'</code> | - GLM: Beta coefficients (n_regressors, n_voxels) per subject
<code>'FittedBrainCollection'</code> | - Ridge: Scores or weights depending on 'output' kwarg
<code>'FittedBrainCollection'</code> | If return_stats (GLM) or output='both' (Ridge), results is a dict.

**Examples:**

```pycon
>>> # GLM with shared design matrix
>>> fitted = bc.fit(model='glm', X=dm)
>>> betas = fitted.results  # Access BrainCollection directly
>>>
>>> # Two-stage analysis with pool()
>>> pool = bc.fit(model='glm', X=dm).pool(param='beta')
>>> t_map = pool.fit(model='ttest', contrast='A-B')
>>>
>>> # GLM with per-subject design matrices
>>> fitted = bc.fit(model='glm', X=[dm1, dm2, dm3])
>>>
>>> # Ridge encoding model with CV scores
>>> fitted = bc.fit(model='ridge', X=features, cv=5)
>>> scores = fitted.results
```

<details class="see-also" open markdown="1">
<summary>See Also</summary>

fit_from_events: Convenience method for event-based GLM workflows
fit_glm: Legacy GLM fitting (use fit_from_events instead)
fit_ridge: Legacy Ridge fitting (use fit(..., model='ridge') instead)

</details>

######## `nltools.data.collection.BrainCollection.fit_from_events`

```python
fit_from_events(events: pd.DataFrame, t_r: float, confounds: str | list[pd.DataFrame | Path | str] | None = None, confound_columns: list[str] | None = None, hrf_model: str = 'spm', drift_model: str = 'cosine', high_pass: float = 0.01, scale: bool = True, scale_value: float = 100.0, return_stats: list[str] | None = None, return_residuals: bool = False, save: dict[str, str] | None = None, show_progress: bool = True, by_run: bool = False, run_column: str = 'run', run_lengths: int | list[int] | None = None) -> 'BrainCollection | dict[str, BrainCollection]'
```

Build design matrices from events and fit GLM to each subject.

Convenience method for event-based experimental designs. Builds
nilearn-compatible design matrices from the events DataFrame and
fits a GLM to each subject in the collection.

This is the recommended method for typical task-based fMRI analysis
where you have event timing information. For more control, use
fit(model='glm', X=design_matrices) with pre-built design matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`events` | <code>[DataFrame](#pandas.DataFrame)</code> | Task events DataFrame with onset, duration, trial_type columns. This is shared across all subjects (same experimental paradigm). If by_run=True, must also have a run column. | *required*
`t_r` | <code>[float](#float)</code> | Repetition time (TR) in seconds. | *required*
`confounds` | <code>[str](#str) \| [list](#list)[[DataFrame](#pandas.DataFrame) \| [Path](#pathlib.Path) \| [str](#str)] \| None</code> | Subject-specific confounds. Can be: - str: Column name in metadata pointing to confound file paths - list: List of DataFrames or paths, one per subject - None: No confounds (only task + drift terms) | <code>None</code>
`confound_columns` | <code>[list](#list)[[str](#str)] \| None</code> | Columns to extract from confound files. If None and confounds provided, uses all columns. | <code>None</code>
`hrf_model` | <code>[str](#str)</code> | HRF model for convolution ('spm', 'glover', 'fir', etc.) | <code>'spm'</code>
`drift_model` | <code>[str](#str)</code> | Drift model ('cosine', 'polynomial', None) | <code>'cosine'</code>
`high_pass` | <code>[float](#float)</code> | High-pass filter cutoff in Hz (default 0.01) | <code>0.01</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`return_stats` | <code>[list](#list)[[str](#str)] \| None</code> | Optional list of statistics to return as separate BrainCollections. Options: 't', 'r2', 'p', 'se', 'residual'. | <code>None</code>
`return_residuals` | <code>[bool](#bool)</code> | If True, return residuals (same as return_stats=['residual']). | <code>False</code>
`save` | <code>[dict](#dict)[[str](#str), [str](#str)] \| None</code> | Dict mapping output type to path template. | <code>None</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`by_run` | <code>[bool](#bool)</code> | If True, fit GLM separately per run and return run-level betas. This enables MVPA decoding with leave-one-run-out CV. | <code>False</code>
`run_column` | <code>[str](#str)</code> | Column name in events identifying runs (default 'run'). | <code>'run'</code>
`run_lengths` | <code>[int](#int) \| [list](#list)[[int](#int)] \| None</code> | Number of TRs per run. Required when by_run=True. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection of beta coefficients for task regressors.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | If return_stats specified, returns dict with keys 'betas', 't', etc.

**Examples:**

```pycon
>>> # Basic GLM fit from events
>>> betas = bc.fit_from_events(events=events_df, t_r=2.0)
>>> group_t = betas.ttest()
>>>
>>> # With confounds from metadata column
>>> betas = bc.fit_from_events(
...     events=events_df,
...     t_r=2.0,
...     confounds='confound_file',
...     confound_columns=['trans_x', 'trans_y', 'trans_z']
... )
>>>
>>> # Run-level betas for MVPA
>>> betas = bc.fit_from_events(events=events_df, t_r=2.0, by_run=True)
```

<details class="see-also" open markdown="1">
<summary>See Also</summary>

fit: Unified fit method that accepts pre-built design matrices
_fit_glm: Internal method for design matrix-based fitting

</details>

######## `nltools.data.collection.BrainCollection.fit_glm`

```python
fit_glm(events: pd.DataFrame, t_r: float, confounds: str | list[pd.DataFrame | Path | str] | None = None, confound_columns: list[str] | None = None, hrf_model: str = 'spm', drift_model: str = 'cosine', high_pass: float = 0.01, scale: bool = True, scale_value: float = 100.0, return_stats: list[str] | None = None, return_residuals: bool = False, save: dict[str, str] | None = None, show_progress: bool = True, by_run: bool = False, run_column: str = 'run', run_lengths: int | list[int] | None = None) -> 'BrainCollection | dict[str, BrainCollection]'
```

Fit GLM to each subject in collection.

Memory-efficient first-level GLM analysis that processes subjects
one at a time. Returns a BrainCollection of beta coefficients for
task regressors (confounds and drift terms are fit but not returned).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`events` | <code>[DataFrame](#pandas.DataFrame)</code> | Task events DataFrame with onset, duration, trial_type columns. This is shared across all subjects (same experimental paradigm). If by_run=True, must also have a run column. | *required*
`t_r` | <code>[float](#float)</code> | Repetition time (TR) in seconds. | *required*
`confounds` | <code>[str](#str) \| [list](#list)[[DataFrame](#pandas.DataFrame) \| [Path](#pathlib.Path) \| [str](#str)] \| None</code> | Subject-specific confounds. Can be: - str: Column name in metadata pointing to confound file paths - list: List of DataFrames or paths, one per subject - None: No confounds (only task + drift terms) | <code>None</code>
`confound_columns` | <code>[list](#list)[[str](#str)] \| None</code> | Columns to extract from confound files. If None and confounds provided, uses all columns. | <code>None</code>
`hrf_model` | <code>[str](#str)</code> | HRF model for convolution ('spm', 'glover', 'fir', etc.) | <code>'spm'</code>
`drift_model` | <code>[str](#str)</code> | Drift model ('cosine', 'polynomial', None) | <code>'cosine'</code>
`high_pass` | <code>[float](#float)</code> | High-pass filter cutoff in Hz (default 0.01) | <code>0.01</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`return_stats` | <code>[list](#list)[[str](#str)] \| None</code> | Optional list of statistics to return as separate BrainCollections. Options: 't', 'r2', 'p', 'se', 'residual'. | <code>None</code>
`return_residuals` | <code>[bool](#bool)</code> | If True, return residuals (same as return_stats=['residual']). | <code>False</code>
`save` | <code>[dict](#dict)[[str](#str), [str](#str)] \| None</code> | Dict mapping output type to path template, e.g. ``{'betas': 'output/{subject}_betas.nii.gz', 't': 'output/{subject}_tstat.nii.gz'}``. Supports {subject}, {session}, {idx}, and other metadata columns. | <code>None</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`by_run` | <code>[bool](#bool)</code> | If True, fit GLM separately per run and return run-level betas. This enables MVPA decoding with leave-one-run-out CV. Each subject will have (n_runs * n_conditions, n_voxels) betas. | <code>False</code>
`run_column` | <code>[str](#str)</code> | Column name in events identifying runs (default 'run'). | <code>'run'</code>
`run_lengths` | <code>[int](#int) \| [list](#list)[[int](#int)] \| None</code> | Number of TRs per run. Required when by_run=True.<br>- int: All runs have same length - list of int: Different length per run - None: Will attempt to infer equal-length runs from total scans | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection where each BrainData has shape:
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | - (n_task_regressors, n_voxels) if by_run=False (default)
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | - (n_runs * n_task_regressors, n_voxels) if by_run=True
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | The ``._design_columns`` attribute stores task regressor names.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | If by_run=True, also stores ``._condition_labels`` and ``._run_labels``.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | If return_stats specified, returns dict with keys 'betas', 't', etc.

**Examples:**

```pycon
>>> # Basic GLM fit
>>> betas = bc.fit_glm(events=events_df, t_r=2.0)
>>> # Group t-test on first regressor
>>> group_t = betas[:, 0].ttest()
```

```pycon
>>> # Run-level betas for MVPA decoding
>>> betas = bc.fit_glm(events=events_df, t_r=2.0, by_run=True)
>>> # betas._condition_labels = ['face', 'house', 'face', 'house', ...]
>>> # betas._run_labels = [1, 1, 2, 2, 3, 3, ...]
>>> accuracy = betas.predict(y=None, method='searchlight')
```

```pycon
>>> # With confounds from metadata column
>>> betas = bc.fit_glm(
...     events=events_df,
...     t_r=2.0,
...     confounds='confound_file',  # column name in metadata
...     confound_columns=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
... )
```

######## `nltools.data.collection.BrainCollection.fit_ridge`

```python
fit_ridge(X: 'np.ndarray | str | list', alpha: float | str = 1.0, cv: int | None = 5, scale: bool = True, scale_value: float = 100.0, output: str = 'scores', save: dict[str, str] | None = None, show_progress: bool = True, **ridge_kwargs: bool) -> 'BrainCollection | dict[str, BrainCollection]'
```

Fit ridge regression to each subject in collection.

Memory-efficient encoding model fitting that processes subjects one at a
time. Default behavior returns cross-validated R² scores per voxel,
suitable for group-level inference on encoding model performance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>'np.ndarray \| str \| list'</code> | Feature matrix. Can be: - np.ndarray: Shared features (n_samples, n_features) used for all subjects - str: Column name in metadata pointing to feature file paths - list: List of arrays/DataFrames, one per subject | *required*
`alpha` | <code>[float](#float) \| [str](#str)</code> | Ridge regularization parameter. Can be: - float: Fixed regularization strength - 'auto': Use cross-validation to select optimal alpha | <code>1.0</code>
`cv` | <code>[int](#int) \| None</code> | Cross-validation folds for computing scores. Default is 5. Required when output='scores' or 'both'. Set to None only when output='weights'. | <code>5</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`output` | <code>[str](#str)</code> | What to return. Options: - 'scores': CV R² scores per voxel (default, for encoding workflow) - 'weights': Model weights (n_features, n_voxels) - 'both': Dict with both 'scores' and 'weights' | <code>'scores'</code>
`save` | <code>[dict](#dict)[[str](#str), [str](#str)] \| None</code> | Dict mapping output type to path template, e.g. ``{'weights': 'output/{subject}_weights.nii.gz', 'scores': 'output/{subject}_scores.nii.gz'}``. Supports {subject}, {session}, {idx}, and other metadata columns. | <code>None</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`**ridge_kwargs` |  | Additional arguments passed to Ridge model (e.g., backend='torch', random_state=42). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection of scores or weights, or dict with both if output='both'.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | Each BrainData will have ``cv_results_`` attribute when cv is used.

**Examples:**

```pycon
>>> # Encoding model workflow: get CV scores for group analysis
>>> scores = bc.fit_ridge(X=features, alpha=1.0)
>>> group_ttest = scores.ttest()  # Test encoding accuracy vs chance
```

```pycon
>>> # Get both scores and weights
>>> results = bc.fit_ridge(X=features, alpha=1.0, output='both')
>>> scores = results['scores']
>>> weights = results['weights']
```

```pycon
>>> # Auto-select alpha with CV
>>> scores = bc.fit_ridge(X=features, alpha='auto', cv=5)
```

```pycon
>>> # Get weights only (no CV needed)
>>> weights = bc.fit_ridge(X=features, alpha=1.0, output='weights', cv=None)
```

######## `nltools.data.collection.BrainCollection.from_bids`

```python
from_bids(layout: Any, mask: nib.Nifti1Image | Path | str, *, task: str | None = None, subject: str | list[str] | None = None, session: str | list[str] | None = None, run: int | list[int] | None = None, space: str | None = None, suffix: str = 'bold', extension: str = 'nii.gz', **bids_filters: str) -> 'BrainCollection'
```

Create BrainCollection from a BIDS dataset.

Requires pybids to be installed: `pip install pybids`

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`layout` | <code>[Any](#typing.Any)</code> | pybids BIDSLayout object or path to BIDS dataset. | *required*
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str)</code> | Shared mask (required). | *required*
`task` | <code>[str](#str) \| None</code> | BIDS task filter. | <code>None</code>
`subject` | <code>[str](#str) \| [list](#list)[[str](#str)] \| None</code> | Subject ID(s) to include. | <code>None</code>
`session` | <code>[str](#str) \| [list](#list)[[str](#str)] \| None</code> | Session ID(s) to include. | <code>None</code>
`run` | <code>[int](#int) \| [list](#list)[[int](#int)] \| None</code> | Run number(s) to include. | <code>None</code>
`space` | <code>[str](#str) \| None</code> | BIDS space filter (e.g., 'MNI152NLin2009cAsym'). | <code>None</code>
`suffix` | <code>[str](#str)</code> | BIDS suffix (default 'bold'). | <code>'bold'</code>
`extension` | <code>[str](#str)</code> | File extension (default 'nii.gz'). | <code>'nii.gz'</code>
`**bids_filters` |  | Additional BIDS entity filters. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with metadata extracted from BIDS entities.

**Examples:**

```pycon
>>> bc = BrainCollection.from_bids(
...     '/data/bids_dataset',
...     mask='2mm-MNI152-2009c',
...     task='rest',
...     space='MNI152NLin2009cAsym'
... )
```

######## `nltools.data.collection.BrainCollection.from_glob`

```python
from_glob(pattern: str, mask: nib.Nifti1Image | Path | str, *, pattern_groups: dict[str, int] | str | None = None, sort: bool = True) -> 'BrainCollection'
```

Create BrainCollection from glob pattern.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`pattern` | <code>[str](#str)</code> | Glob pattern (e.g., ``'/data/*/func/*_bold.nii.gz'``). | *required*
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str)</code> | Shared mask (required). | *required*
`pattern_groups` | <code>[dict](#dict)[[str](#str), [int](#int)] \| [str](#str) \| None</code> | Regex pattern with named groups for metadata extraction. Example: ``r'sub-(?P<subject>\w+)/.*run-(?P<run>\d+)'`` | <code>None</code>
`sort` | <code>[bool](#bool)</code> | Sort files alphabetically (default True). | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with optional metadata from pattern groups.

**Examples:**

```pycon
>>> bc = BrainCollection.from_glob(
...     '/data/sub-*/func/*_bold.nii.gz',
...     mask=mask,
...     pattern_groups=r'sub-(?P<subject>\w+)'
... )
```

######## `nltools.data.collection.BrainCollection.from_stacked`

```python
from_stacked(brain_data: 'BrainData', splits: list[int] | None = None, n_images: int | None = None) -> 'BrainCollection'
```

Create BrainCollection by splitting a stacked BrainData.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_data` | <code>'BrainData'</code> | BrainData with shape (n_total_obs, n_voxels). | *required*
`splits` | <code>[list](#list)[[int](#int)] \| None</code> | List of observation counts per image. Must sum to n_total_obs. | <code>None</code>
`n_images` | <code>[int](#int) \| None</code> | Number of images (splits evenly). Mutually exclusive with splits. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with data split according to specification.

**Examples:**

```pycon
>>> # Split evenly into 3 images
>>> bc = BrainCollection.from_stacked(bd, n_images=3)
```

```pycon
>>> # Split with explicit counts
>>> bc = BrainCollection.from_stacked(bd, splits=[100, 100, 150])
```

######## `nltools.data.collection.BrainCollection.isc`

```python
isc(method: str = 'loo', roi_mask: 'nib.Nifti1Image | Path | str | None' = None, radius: float | None = 6.0, metric: str = 'median', parallel: str = 'cpu', n_jobs: int = -1, show_progress: bool = True) -> dict
```

Compute intersubject correlation (ISC) across the collection.

ISC measures the similarity of brain responses across subjects,
computed by correlating each subject's timeseries with others.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ISC computation method: - 'loo': Leave-one-out (correlate each subject with mean of others) - 'pairwise': All pairwise correlations between subjects | <code>'loo'</code>
`roi_mask` | <code>'nib.Nifti1Image \| Path \| str \| None'</code> | If provided, compute ISC per ROI. Can be: - NIfTI image with integer labels (atlas/parcellation) - Path to parcellation file | <code>None</code>
`radius` | <code>[float](#float) \| None</code> | Searchlight radius in mm. If None, use voxelwise mode. Ignored if roi_mask is provided. | <code>6.0</code>
`metric` | <code>[str](#str)</code> | Summary statistic for aggregating ISC values: - 'median': Robust to outliers (default) - 'mean': Fisher z-transformed mean | <code>'median'</code>
`parallel` | <code>[str](#str)</code> | Parallelization method ('cpu', 'gpu', or None). | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs (-1 = all cores). | <code>-1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during extraction. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with: - 'isc': BrainData with ISC values - 'method': ISC method used ('loo' or 'pairwise') - 'extraction': Extraction mode ('roi', 'searchlight', 'voxelwise') - 'n_subjects': Number of subjects - 'extraction_info': Dict with extraction metadata

**Examples:**

```pycon
>>> # ROI-based ISC using atlas
>>> result = bc.isc(roi_mask="atlas.nii.gz")
>>> result['isc'].plot()
```

```pycon
>>> # Searchlight ISC
>>> result = bc.isc(radius=10.0)
```

```pycon
>>> # Voxelwise ISC
>>> result = bc.isc(radius=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

For permutation testing, see BrainCollection.isc_test() (requires
discussion of statistical methodology first).

</details>

######## `nltools.data.collection.BrainCollection.isc_test`

```python
isc_test(method: str = 'loo', roi_mask: 'nib.Nifti1Image | Path | str | None' = None, radius: float | None = 6.0, n_permute: int = 5000, permutation_method: str = 'bootstrap', metric: str = 'median', tail: int = 2, ci_percentile: float = 95, parallel: str = 'cpu', n_jobs: int = -1, random_state: int | None = None, return_null: bool = False, show_progress: bool = True) -> dict
```

Compute ISC with permutation testing for statistical inference.

This method combines ISC computation with permutation testing to
determine statistical significance. It uses the same extraction
pipeline as isc() and wraps isc_permutation_test().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ISC computation method: - 'loo': Leave-one-out (correlate each subject with mean of others) - 'pairwise': All pairwise correlations between subjects | <code>'loo'</code>
`roi_mask` | <code>'nib.Nifti1Image \| Path \| str \| None'</code> | If provided, compute ISC per ROI. Can be: - NIfTI image with integer labels (atlas/parcellation) - Path to parcellation file | <code>None</code>
`radius` | <code>[float](#float) \| None</code> | Searchlight radius in mm. If None, use voxelwise mode. Ignored if roi_mask is provided. | <code>6.0</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations/bootstrap iterations. Default 5000. | <code>5000</code>
`permutation_method` | <code>[str](#str)</code> | Method for null distribution:<br>- 'bootstrap': Subject-wise bootstrap (default, Chen et al. 2016).   Tests whether observed ISC differs from random groupings. - 'circle_shift': Circular time-shift (preserves autocorrelation).   Tests for temporally-locked shared signal. - 'phase_randomize': FFT phase randomization (preserves power spectrum).   Tests for nonlinear temporal coupling. | <code>'bootstrap'</code>
`metric` | <code>[str](#str)</code> | Summary statistic for aggregating ISC values: - 'median': Robust to outliers (default) - 'mean': Fisher z-transformed mean | <code>'median'</code>
`tail` | <code>[int](#int)</code> | One-tailed (1) or two-tailed (2) test. Default 2. | <code>2</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95). Default 95. | <code>95</code>
`parallel` | <code>[str](#str)</code> | Parallelization method ('cpu', 'gpu', or None). | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs (-1 = all cores). | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, include null distribution in results. | <code>False</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during extraction and permutation. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with: - 'isc': BrainData with ISC values - 'p': BrainData with p-values (Phipson-Smyth corrected) - 'ci': Tuple of (lower, upper) BrainData confidence intervals - 'method': ISC method used ('loo' or 'pairwise') - 'permutation_method': Permutation method used - 'extraction': Extraction mode ('roi', 'searchlight', 'voxelwise') - 'n_subjects': Number of subjects - 'n_permute': Number of permutations - 'null_dist': (optional) Null distribution array if return_null=True

**Examples:**

```pycon
>>> # ROI-based ISC with permutation testing
>>> result = bc.isc_test(roi_mask="atlas.nii.gz", n_permute=5000)
>>> sig_mask = result['p'].data < 0.05
>>> print(f"Significant ROIs: {sig_mask.sum()}")
```

```pycon
>>> # Searchlight ISC testing
>>> result = bc.isc_test(radius=10.0)
>>> result['isc'].plot()  # Show ISC values
>>> result['p'].plot()    # Show p-values
```

```pycon
>>> # Voxelwise with phase randomization (tests temporal coupling)
>>> result = bc.isc_test(
...     radius=None,
...     permutation_method='phase_randomize',
...     parallel='gpu'
... )
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Bootstrap (default) is recommended for standard ISC inference
  (Chen et al. 2016). It tests whether ISC is significant at
  the group level.
- Circle_shift and phase_randomize are more specialized - they
  test for temporally-structured shared signal beyond what's
  explained by autocorrelation or spectral structure alone.
- For large voxelwise analyses, bootstrap is much faster as it
  resamples pre-computed values rather than recomputing ISC.

</details>

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., et al. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

######## `nltools.data.collection.BrainCollection.iter_batches`

```python
iter_batches(batch_size: int, axis: int = 0, show_progress: bool = True) -> Generator['BrainCollection', None, None]
```

Iterate in batches along axis.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`batch_size` | <code>[int](#int)</code> | Number of items per batch. | *required*
`axis` | <code>[int](#int)</code> | Axis to batch along: - 0: Batches of images (default) - 1: Batches of timepoints (within each image) | <code>0</code>
`show_progress` | <code>[bool](#bool)</code> | Show tqdm progress bar. | <code>True</code>

**Yields:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection for each batch.

**Examples:**

```pycon
>>> # Batch over images
>>> for batch in bc.iter_batches(batch_size=5):
...     process(batch)  # batch is BrainCollection with 5 images
```

```pycon
>>> # Batch over time
>>> for batch in bc.iter_batches(batch_size=10, axis=1):
...     process(batch)  # batch has 10 timepoints per image
```

######## `nltools.data.collection.BrainCollection.load`

```python
load(indices: list[int] | None = None) -> 'BrainCollection'
```

Load specified images into memory.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`indices` | <code>[list](#list)[[int](#int)] \| None</code> | List of indices to load. If None, loads all. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | self (for chaining)

######## `nltools.data.collection.BrainCollection.map`

```python
map(fn: Callable, axis: int | str = 0, n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Apply function across specified axis.

This is the general-purpose transformation method. For common operations,
use convenience methods like standardize(), smooth(), etc.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fn` | <code>[Callable](#collections.abc.Callable)</code> | Function to apply. Signature depends on axis: - axis=0: fn(BrainData) → BrainData (per image) - axis=1: fn(BrainData) → BrainData (per timepoint slice) - axis=2: fn(ndarray[n_obs]) → ndarray (per voxel timeseries) | *required*
`axis` | <code>[int](#int) \| [str](#str)</code> | Axis to iterate over: - 0 or 'images': Apply fn to each image independently - 1 or 'time': Apply fn to each timepoint across images - 2 or 'voxels': Apply fn to each voxel timeseries per image | <code>0</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. -1 for all cores. Default 1. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show tqdm progress bar. Default True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with transformed data.

**Examples:**

```pycon
>>> # Per-image operation
>>> bc.map(lambda bd: bd.standardize())
```

```pycon
>>> # Per-voxel timeseries (e.g., detrend each voxel)
>>> from scipy.signal import detrend
>>> bc.map(detrend, axis=2)
```

```pycon
>>> # Parallel processing
>>> bc.map(expensive_fn, n_jobs=-1)
```

######## `nltools.data.collection.BrainCollection.max`

```python
max(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute maximum along axis. See mean() for details.

######## `nltools.data.collection.BrainCollection.mean`

```python
mean(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute mean along axis.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>[int](#int) \| [str](#str) \| [tuple](#tuple)[[int](#int), ...]</code> | Axis or axes to aggregate: - 0 or 'images': Mean across images -> BrainData (n_obs, n_voxels) - 1 or 'time': Mean across time -> BrainCollection (n_images, n_voxels) - 2 or 'voxels': Mean across voxels -> np.ndarray (n_images, n_obs) - (0, 1): Mean across images and time -> BrainData (n_voxels,) | <code>0</code>
`batch_size` | <code>[int](#int) \| None</code> | Number of images to process at once (for memory efficiency). If None, uses streaming algorithm. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainData \| BrainCollection \| np.ndarray'</code> | BrainData, BrainCollection, or np.ndarray depending on axis.

**Examples:**

```pycon
>>> bc.mean(axis=0)  # Mean across subjects
>>> bc.mean(axis='images')  # Same as above
>>> bc.mean(axis=1)  # Mean across time per subject
>>> bc.mean(axis=(0, 1))  # Grand mean
```

######## `nltools.data.collection.BrainCollection.median`

```python
median(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute median along axis. See mean() for details.

######## `nltools.data.collection.BrainCollection.memory_estimate`

```python
memory_estimate() -> str
```

Estimate memory usage for loading all images.

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | Human-readable string like "12.4 GB total (1.2 GB per image avg)"

######## `nltools.data.collection.BrainCollection.min`

```python
min(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute minimum along axis. See mean() for details.

######## `nltools.data.collection.BrainCollection.permutation_test`

```python
permutation_test(n_permute: int = 5000, tail: int = 2, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None, return_null: bool = False) -> dict
```

One-sample permutation test across images (sign-flipping).

Tests whether the mean across images is significantly different from
zero using sign-flipping permutation. More robust than parametric
t-test for non-normal distributions.

This is a collection-level interface to
nltools.algorithms.inference.one_sample_permutation_test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000). | <code>5000</code>
`tail` | <code>[int](#int)</code> | Test type - 1 for one-tailed, 2 for two-tailed (default). | <code>2</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization method: - 'cpu': CPU parallelization via joblib (default) - 'gpu': GPU acceleration via PyTorch - None: Single-threaded (for debugging) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores (default: -1 = all cores). | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget (default: 4.0 GB). | <code>4.0</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, include null distribution in result. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | dict with keys: - 'mean': BrainData with observed mean across images - 'p': BrainData with p-values - 'null_dist': np.ndarray (if return_null=True) - 'parallel': parallelization method used

**Examples:**

```pycon
>>> result = bc.permutation_test(n_permute=5000)
>>> mean_bd, p_bd = result['mean'], result['p']
```

```pycon
>>> # With GPU acceleration
>>> result = bc.permutation_test(parallel='gpu')
```

######## `nltools.data.collection.BrainCollection.permutation_test2`

```python
permutation_test2(other: 'BrainCollection', n_permute: int = 5000, tail: int = 2, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None, return_null: bool = False) -> dict
```

Two-sample permutation test between collections.

Tests whether two collections have different means using group
label permutation. More robust than parametric t-test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` | <code>'BrainCollection'</code> | Another BrainCollection to compare against. | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000). | <code>5000</code>
`tail` | <code>[int](#int)</code> | Test type - 1 for one-tailed, 2 for two-tailed (default). | <code>2</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization method ('cpu', 'gpu', or None). | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores (default: -1 = all cores). | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget (default: 4.0 GB). | <code>4.0</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, include null distribution in result. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | dict with keys: - 'mean_diff': BrainData with observed mean difference - 'p': BrainData with p-values - 'null_dist': np.ndarray (if return_null=True) - 'parallel': parallelization method used

**Examples:**

```pycon
>>> result = patients.permutation_test2(controls)
>>> diff_bd, p_bd = result['mean_diff'], result['p']
```

######## `nltools.data.collection.BrainCollection.predict`

```python
predict(X: 'np.ndarray | str | list | None' = None, y: 'np.ndarray | None' = None, method: str = 'whole_brain', estimator: str = 'svm', cv: str = 5, groups: 'np.ndarray | None' = None, roi_mask: 'np.ndarray | None' = None, radius: float = 10.0, scoring: str = 'accuracy', standardize: bool = True, n_jobs: int = -1, show_progress: bool = True) -> 'BrainCollection'
```

Generate predictions for each subject in collection.

This method supports two prediction modes determined by which parameter
is provided:

1. **Timeseries prediction** (X provided): Use fitted ridge model to
   predict voxel responses for new feature data.

2. **MVPA decoding** (y provided): Train a classifier to predict labels
   from brain patterns using cross-validation.

For MVPA, if this collection was created with by_run=True, you can
use y=None to infer labels from _condition_labels and groups from
_run_labels (leave-one-run-out CV).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>'np.ndarray \| str \| list \| None'</code> | Features for timeseries prediction. Can be: - np.ndarray: Shared features (same for all subjects) - str: Metadata column with per-subject feature paths - list: Per-subject feature arrays | <code>None</code>
`y` | <code>'np.ndarray \| None'</code> | Labels for MVPA decoding. If None and _condition_labels exists, will use stored condition labels (from fit_glm with by_run=True). | <code>None</code>
`method` | <code>[str](#str)</code> | MVPA method - 'whole_brain', 'searchlight', or 'roi'. | <code>'whole_brain'</code>
`estimator` |  | Classifier - 'svm', 'logistic', 'ridge', 'lda' or sklearn estimator instance. | <code>'svm'</code>
`cv` |  | Cross-validation strategy. If None and _run_labels exists, uses leave-one-group-out with run labels. | <code>5</code>
`groups` | <code>'np.ndarray \| None'</code> | Group labels for GroupKFold/LeaveOneGroupOut. If None and _run_labels exists, uses stored run labels. | <code>None</code>
`roi_mask` |  | Mask for ROI-based MVPA. Required if method='roi'. | <code>None</code>
`radius` | <code>[float](#float)</code> | Searchlight radius in mm (default 10.0). | <code>10.0</code>
`scoring` | <code>[str](#str)</code> | Scoring metric (default 'accuracy'). | <code>'accuracy'</code>
`standardize` | <code>[bool](#bool)</code> | If True, standardize features before classification. | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Parallel jobs for searchlight (-1 = all cores). | <code>-1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with prediction results:
<code>'BrainCollection'</code> | - For timeseries: (n_timepoints, n_voxels) predicted responses
<code>'BrainCollection'</code> | - For MVPA: (1, n_voxels) accuracy values

**Examples:**

```pycon
>>> # MVPA workflow with run-level betas
>>> betas = bc.fit_glm(events=events, t_r=2.0, by_run=True)
>>> accuracy = betas.predict(y=None, method='whole_brain')
>>> # y and groups inferred from _condition_labels, _run_labels
```

```pycon
>>> # Explicit labels
>>> accuracy = betas.predict(y=labels, method='searchlight')
```

```pycon
>>> # Timeseries prediction with ridge weights
>>> weights = bc.fit_ridge(X=features, output='weights')
>>> predictions = weights.predict(X=new_features)
```

######## `nltools.data.collection.BrainCollection.select_feature`

```python
select_feature(feature: 'int | str') -> 'BrainCollection'
```

Select a single feature's weights across all subjects.

Used after fit_ridge() to extract weights for a specific feature
for group-level analysis (e.g., t-test on feature weights).

Must be called on a BrainCollection created by fit_ridge() where
each subject has shape (n_features, n_voxels).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`feature` | <code>'int \| str'</code> | Feature to select. Can be: - int: Feature index (0-based) - str: Feature name (requires _feature_names attribute) | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection where each BrainData has shape (n_voxels,)
<code>'BrainCollection'</code> | containing the weights for the specified feature.

**Examples:**

```pycon
>>> # Fit ridge and select feature
>>> weights = bc.fit_ridge(X=features, alpha=1.0)
>>> feature_0 = weights.select_feature(0)
>>> # Group t-test on first feature's weights
>>> group_result = feature_0.ttest()
```

```pycon
>>> # By name (if features had column names)
>>> face_weights = weights.select_feature("face_response")
```

######## `nltools.data.collection.BrainCollection.smooth`

```python
smooth(fwhm: float, n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Spatially smooth each image.

Delegates to BrainData.smooth() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fwhm` | <code>[float](#float)</code> | Full width at half maximum of Gaussian kernel in mm. | *required*
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with smoothed images.

**Examples:**

```pycon
>>> bc.smooth(fwhm=6)  # 6mm FWHM smoothing
```

######## `nltools.data.collection.BrainCollection.standardize`

```python
standardize(axis: int = 0, method: str = 'center', n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Standardize each image.

Delegates to BrainData.standardize() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>[int](#int)</code> | Axis for standardization within each image: - 0: Standardize across observations (time) per voxel - 1: Standardize across voxels per observation | <code>0</code>
`method` | <code>[str](#str)</code> | 'center' (subtract mean) or 'zscore' (subtract mean, divide std) | <code>'center'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with standardized images.

**Examples:**

```pycon
>>> bc.standardize()  # Center each image across time
>>> bc.standardize(method='zscore')  # Z-score each image
>>> bc.standardize(axis=1)  # Standardize across voxels
```

######## `nltools.data.collection.BrainCollection.std`

```python
std(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute standard deviation along axis. See mean() for details.

######## `nltools.data.collection.BrainCollection.sum`

```python
sum(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute sum along axis. See mean() for details.

######## `nltools.data.collection.BrainCollection.threshold`

```python
threshold(upper: float | str | None = None, lower: float | str | None = None, binarize: bool = False, coerce_nan: bool = True, n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Threshold each image.

Delegates to BrainData.threshold() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`upper` | <code>[float](#float) \| [str](#str) \| None</code> | Upper cutoff. String interpreted as percentile. | <code>None</code>
`lower` | <code>[float](#float) \| [str](#str) \| None</code> | Lower cutoff. String interpreted as percentile. | <code>None</code>
`binarize` | <code>[bool](#bool)</code> | Return binary mask. | <code>False</code>
`coerce_nan` | <code>[bool](#bool)</code> | Replace NaN with 0. | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with thresholded images.

**Examples:**

```pycon
>>> bc.threshold(lower=0)  # Zero out negative values
>>> bc.threshold(upper='95%')  # Keep top 5%
>>> bc.threshold(lower=2, binarize=True)  # Binary mask
```

######## `nltools.data.collection.BrainCollection.to_list`

```python
to_list() -> list['BrainData']
```

Return list of BrainData objects.

Loads all items if not already loaded.

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)['BrainData']</code> | List of BrainData objects.

######## `nltools.data.collection.BrainCollection.to_stacked`

```python
to_stacked() -> 'BrainData'
```

Stack all into single BrainData (n_total_obs, n_voxels).

**Returns:**

Type | Description
---- | -----------
<code>'BrainData'</code> | Single BrainData with all observations concatenated.

**Examples:**

```pycon
>>> bc = BrainCollection([bd1, bd2, bd3], mask=mask)
>>> stacked = bc.to_stacked()
>>> stacked.shape
(300, 50000)  # 3 images * 100 obs each
```

######## `nltools.data.collection.BrainCollection.to_tensor`

```python
to_tensor(batch_size: int | None = None) -> np.ndarray | Generator[np.ndarray, None, None]
```

Convert to numpy array (n_images, n_obs, n_voxels).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`batch_size` | <code>[int](#int) \| None</code> | If specified, returns generator yielding batches. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray) \| [Generator](#typing.Generator)[[ndarray](#numpy.ndarray), None, None]</code> | Full tensor if batch_size is None, otherwise generator.

**Examples:**

```pycon
>>> tensor = bc.to_tensor()  # Full array
>>> tensor.shape
(3, 100, 50000)
```

```pycon
>>> # Batched iteration
>>> for batch in bc.to_tensor(batch_size=10):
...     process(batch)  # batch.shape = (10, 100, 50000)
```

######## `nltools.data.collection.BrainCollection.ttest`

```python
ttest(popmean: float = 0.0, axis: int | str = 0) -> tuple['BrainData', 'BrainData']
```

One-sample t-test across images.

Tests whether the mean across images is significantly different from
a population mean (default: 0). This is the voxel-wise equivalent of
scipy.stats.ttest_1samp.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`popmean` | <code>[float](#float)</code> | Population mean to test against (default: 0). | <code>0.0</code>
`axis` | <code>[int](#int) \| [str](#str)</code> | Axis to test across. Only axis=0 (images) supported. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainData'</code> | Tuple of (t_stat, p_value) as BrainData objects.
<code>'BrainData'</code> | Both have shape (n_obs, n_voxels) if uniform obs counts.

**Examples:**

```pycon
>>> t_stat, p_val = bc.ttest()  # Test mean != 0
>>> t_stat, p_val = bc.ttest(popmean=0.5)  # Test mean != 0.5
```

```pycon
>>> # Threshold significant voxels
>>> sig_mask = p_val.data < 0.05
```

######## `nltools.data.collection.BrainCollection.ttest2`

```python
ttest2(other: 'BrainCollection', equal_var: bool = True) -> tuple['BrainData', 'BrainData']
```

Two-sample t-test between collections.

Tests whether two collections have different means. This is the
voxel-wise equivalent of scipy.stats.ttest_ind.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` | <code>'BrainCollection'</code> | Another BrainCollection to compare against. | *required*
`equal_var` | <code>[bool](#bool)</code> | If True (default), perform standard t-test assuming equal variances. If False, use Welch's t-test. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)['BrainData', 'BrainData']</code> | Tuple of (t_stat, p_value) as BrainData objects.

**Examples:**

```pycon
>>> t_stat, p_val = patients.ttest2(controls)
>>> t_stat, p_val = group1.ttest2(group2, equal_var=False)  # Welch's
```

######## `nltools.data.collection.BrainCollection.unload`

```python
unload(indices: list[int] | None = None) -> 'BrainCollection'
```

Free memory for specified images (keep paths for reloading).

Only works for items that were originally loaded from paths.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`indices` | <code>[list](#list)[[int](#int)] \| None</code> | List of indices to unload. If None, unloads all possible. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | self (for chaining)

######## `nltools.data.collection.BrainCollection.var`

```python
var(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute variance along axis. See mean() for details.

######## `nltools.data.collection.BrainCollection.write`

```python
write(directory: str | Path, pattern: str = 'image_{i:04d}.nii.gz', metadata_file: str | None = 'metadata.csv') -> list[Path]
```

Write all images in collection to files.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`directory` | <code>[str](#str) \| [Path](#pathlib.Path)</code> | Output directory path. Will be created if it doesn't exist. | *required*
`pattern` | <code>[str](#str)</code> | Filename pattern with {i} placeholder for image index. Default: "image_{i:04d}.nii.gz" produces image_0000.nii.gz, etc. | <code>'image_{i:04d}.nii.gz'</code>
`metadata_file` | <code>[str](#str) \| None</code> | Optional filename for metadata CSV. Set to None to skip. Default: "metadata.csv" | <code>'metadata.csv'</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[Path](#pathlib.Path)]</code> | List of paths to written files.

**Examples:**

```pycon
>>> bc = BrainCollection([bd1, bd2, bd3], mask=mask)
>>> paths = bc.write("output/")
>>> # Creates: output/image_0000.nii.gz, image_0001.nii.gz, etc.
```

```pycon
>>> # Custom pattern
>>> bc.write("output/", pattern="sub-{i:02d}_bold.nii.gz")
>>> # Creates: output/sub-00_bold.nii.gz, sub-01_bold.nii.gz, etc.
```

```pycon
>>> # With BIDS-style naming using metadata
>>> bc.metadata["filename"] = [f"sub-{s}_bold.nii.gz" for s in subjects]
>>> for i, bd in enumerate(bc):
...     bd.write(f"output/{bc.metadata.loc[i, 'filename']}")
```



##### Functions

##### Modules###### `nltools.data.collection.constructors`

Constructor functions for BrainCollection.

Standalone functions that create BrainCollection instances from various sources
(BIDS datasets, glob patterns, stacked BrainData).

**Functions:**

Name | Description
---- | -----------
[`from_bids`](#nltools.data.collection.constructors.from_bids) | Create BrainCollection from a BIDS dataset.
[`from_glob`](#nltools.data.collection.constructors.from_glob) | Create BrainCollection from glob pattern.
[`from_stacked`](#nltools.data.collection.constructors.from_stacked) | Create BrainCollection by splitting a stacked BrainData.



####### Classes

####### Functions######## `nltools.data.collection.constructors.from_bids`

```python
from_bids(layout: Any, mask: 'nib.Nifti1Image | Path | str', *, task: str | None = None, subject: str | list[str] | None = None, session: str | list[str] | None = None, run: int | list[int] | None = None, space: str | None = None, suffix: str = 'bold', extension: str = 'nii.gz', **bids_filters: str) -> 'BrainCollection'
```

Create BrainCollection from a BIDS dataset.

Requires pybids to be installed: `pip install pybids`

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`layout` | <code>[Any](#typing.Any)</code> | pybids BIDSLayout object or path to BIDS dataset. | *required*
`mask` | <code>'nib.Nifti1Image \| Path \| str'</code> | Shared mask (required). | *required*
`task` | <code>[str](#str) \| None</code> | BIDS task filter. | <code>None</code>
`subject` | <code>[str](#str) \| [list](#list)[[str](#str)] \| None</code> | Subject ID(s) to include. | <code>None</code>
`session` | <code>[str](#str) \| [list](#list)[[str](#str)] \| None</code> | Session ID(s) to include. | <code>None</code>
`run` | <code>[int](#int) \| [list](#list)[[int](#int)] \| None</code> | Run number(s) to include. | <code>None</code>
`space` | <code>[str](#str) \| None</code> | BIDS space filter (e.g., 'MNI152NLin2009cAsym'). | <code>None</code>
`suffix` | <code>[str](#str)</code> | BIDS suffix (default 'bold'). | <code>'bold'</code>
`extension` | <code>[str](#str)</code> | File extension (default 'nii.gz'). | <code>'nii.gz'</code>
`**bids_filters` |  | Additional BIDS entity filters. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with metadata extracted from BIDS entities.

**Examples:**

```pycon
>>> bc = from_bids(
...     '/data/bids_dataset',
...     mask='2mm-MNI152-2009c',
...     task='rest',
...     space='MNI152NLin2009cAsym'
... )
```

######## `nltools.data.collection.constructors.from_glob`

```python
from_glob(pattern: str, mask: 'nib.Nifti1Image | Path | str', *, pattern_groups: 'dict[str, int] | str | None' = None, sort: bool = True) -> 'BrainCollection'
```

Create BrainCollection from glob pattern.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`pattern` | <code>[str](#str)</code> | Glob pattern (e.g., ``'/data/*/func/*_bold.nii.gz'``). | *required*
`mask` | <code>'nib.Nifti1Image \| Path \| str'</code> | Shared mask (required). | *required*
`pattern_groups` | <code>'dict[str, int] \| str \| None'</code> | Regex pattern with named groups for metadata extraction. Example: ``r'sub-(?P<subject>\w+)/.*run-(?P<run>\d+)'`` | <code>None</code>
`sort` | <code>[bool](#bool)</code> | Sort files alphabetically (default True). | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with optional metadata from pattern groups.

**Examples:**

```pycon
>>> bc = from_glob(
...     '/data/sub-*/func/*_bold.nii.gz',
...     mask=mask,
...     pattern_groups=r'sub-(?P<subject>\w+)'
... )
```

######## `nltools.data.collection.constructors.from_stacked`

```python
from_stacked(brain_data: 'BrainData', splits: list[int] | None = None, n_images: int | None = None) -> 'BrainCollection'
```

Create BrainCollection by splitting a stacked BrainData.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_data` | <code>'BrainData'</code> | BrainData with shape (n_total_obs, n_voxels). | *required*
`splits` | <code>[list](#list)[[int](#int)] \| None</code> | List of observation counts per image. Must sum to n_total_obs. | <code>None</code>
`n_images` | <code>[int](#int) \| None</code> | Number of images (splits evenly). Mutually exclusive with splits. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with data split according to specification.

**Examples:**

```pycon
>>> # Split evenly into 3 images
>>> bc = from_stacked(bd, n_images=3)
```

```pycon
>>> # Split with explicit counts
>>> bc = from_stacked(bd, splits=[100, 100, 150])
```

###### `nltools.data.collection.inference`

BrainCollection inference functions.

Extracted from BrainCollection methods — each function takes a BrainCollection
as its first argument (``bc``) instead of ``self``.

**Functions:**

Name | Description
---- | -----------
[`anova`](#nltools.data.collection.inference.anova) | One-way ANOVA across groups defined by metadata.
[`extract_for_isc`](#nltools.data.collection.inference.extract_for_isc) | Extract data for ISC computation.
[`extract_roi`](#nltools.data.collection.inference.extract_roi) | Extract mean signal per ROI.
[`extract_searchlight`](#nltools.data.collection.inference.extract_searchlight) | Extract mean signal per searchlight sphere.
[`extract_voxelwise`](#nltools.data.collection.inference.extract_voxelwise) | Extract raw voxel data.
[`isc`](#nltools.data.collection.inference.isc) | Compute intersubject correlation (ISC) across the collection.
[`isc_test`](#nltools.data.collection.inference.isc_test) | Compute ISC with permutation testing for statistical inference.
[`permutation_test`](#nltools.data.collection.inference.permutation_test) | One-sample permutation test across images (sign-flipping).
[`permutation_test2`](#nltools.data.collection.inference.permutation_test2) | Two-sample permutation test between collections.
[`project_to_brain`](#nltools.data.collection.inference.project_to_brain) | Project ISC values back to brain space.
[`ttest`](#nltools.data.collection.inference.ttest) | One-sample t-test across images.
[`ttest2`](#nltools.data.collection.inference.ttest2) | Two-sample t-test between collections.



####### Classes

####### Functions######## `nltools.data.collection.inference.anova`

```python
anova(bc: 'BrainCollection', groups: str | list | np.ndarray) -> tuple
```

One-way ANOVA across groups defined by metadata.

Tests whether group means differ significantly. This is the
voxel-wise equivalent of scipy.stats.f_oneway.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to test. | *required*
`groups` | <code>[str](#str) \| [list](#list) \| [ndarray](#numpy.ndarray)</code> | Group assignment for each image. Can be: - str: Column name in metadata - list/array: Group labels of length n_images | *required*

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)</code> | Tuple of (F_stat, p_value) as BrainData objects.

**Examples:**

```pycon
>>> # Groups from metadata column
>>> f_stat, p_val = bc.anova('condition')
```

```pycon
>>> # Explicit group labels
>>> groups = ['control'] * 10 + ['patient'] * 15
>>> f_stat, p_val = bc.anova(groups)
```

######## `nltools.data.collection.inference.extract_for_isc`

```python
extract_for_isc(bc: 'BrainCollection', roi_mask: 'nib.Nifti1Image | Path | str | None' = None, radius: float | None = 6.0, show_progress: bool = True) -> tuple[np.ndarray, dict]
```

Extract data for ISC computation.

Memory-efficient extraction that processes one subject at a time.
Returns data in ISC-compatible format: (n_obs, n_subjects, n_features).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to extract from. | *required*
`roi_mask` | <code>'nib.Nifti1Image \| Path \| str \| None'</code> | If provided, extract mean per ROI. Can be: - NIfTI image with integer labels (atlas/parcellation) - Path to parcellation file | <code>None</code>
`radius` | <code>[float](#float) \| None</code> | Searchlight radius in mm. If None, use voxelwise mode. Ignored if roi_mask is provided. | <code>6.0</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during extraction. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[ndarray](#numpy.ndarray), [dict](#dict)]</code> | Tuple of: - extracted_data: Array of shape (n_obs, n_subjects, n_features) - extraction_info: Dict with metadata for projection back:     - 'mode': 'roi', 'searchlight', or 'voxelwise'     - 'n_features': Number of features     - 'roi_mask': ROI mask if mode='roi'     - 'neighborhoods': SphereNeighborhoods if mode='searchlight'

######## `nltools.data.collection.inference.extract_roi`

```python
extract_roi(bc: 'BrainCollection', roi_mask: 'nib.Nifti1Image | Path | str', show_progress: bool = True) -> tuple[np.ndarray, dict]
```

Extract mean signal per ROI.

######## `nltools.data.collection.inference.extract_searchlight`

```python
extract_searchlight(bc: 'BrainCollection', radius: float, show_progress: bool = True) -> tuple[np.ndarray, dict]
```

Extract mean signal per searchlight sphere.

######## `nltools.data.collection.inference.extract_voxelwise`

```python
extract_voxelwise(bc: 'BrainCollection', show_progress: bool = True) -> tuple[np.ndarray, dict]
```

Extract raw voxel data.

######## `nltools.data.collection.inference.isc`

```python
isc(bc: 'BrainCollection', method: str = 'loo', roi_mask: 'nib.Nifti1Image | Path | str | None' = None, radius: float | None = 6.0, metric: str = 'median', parallel: str = 'cpu', n_jobs: int = -1, show_progress: bool = True) -> dict
```

Compute intersubject correlation (ISC) across the collection.

ISC measures the similarity of brain responses across subjects,
computed by correlating each subject's timeseries with others.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to compute ISC on. | *required*
`method` | <code>[str](#str)</code> | ISC computation method: - 'loo': Leave-one-out (correlate each subject with mean of others) - 'pairwise': All pairwise correlations between subjects | <code>'loo'</code>
`roi_mask` | <code>'nib.Nifti1Image \| Path \| str \| None'</code> | If provided, compute ISC per ROI. Can be: - NIfTI image with integer labels (atlas/parcellation) - Path to parcellation file | <code>None</code>
`radius` | <code>[float](#float) \| None</code> | Searchlight radius in mm. If None, use voxelwise mode. Ignored if roi_mask is provided. | <code>6.0</code>
`metric` | <code>[str](#str)</code> | Summary statistic for aggregating ISC values: - 'median': Robust to outliers (default) - 'mean': Fisher z-transformed mean | <code>'median'</code>
`parallel` | <code>[str](#str)</code> | Parallelization method ('cpu', 'gpu', or None). | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs (-1 = all cores). | <code>-1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during extraction. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with: - 'isc': BrainData with ISC values - 'method': ISC method used ('loo' or 'pairwise') - 'extraction': Extraction mode ('roi', 'searchlight', 'voxelwise') - 'n_subjects': Number of subjects - 'extraction_info': Dict with extraction metadata

**Examples:**

```pycon
>>> # ROI-based ISC using atlas
>>> result = bc.isc(roi_mask="atlas.nii.gz")
>>> result['isc'].plot()
```

```pycon
>>> # Searchlight ISC
>>> result = bc.isc(radius=10.0)
```

```pycon
>>> # Voxelwise ISC
>>> result = bc.isc(radius=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

For permutation testing, see BrainCollection.isc_test() (requires
discussion of statistical methodology first).

</details>

######## `nltools.data.collection.inference.isc_test`

```python
isc_test(bc: 'BrainCollection', method: str = 'loo', roi_mask: 'nib.Nifti1Image | Path | str | None' = None, radius: float | None = 6.0, n_permute: int = 5000, permutation_method: str = 'bootstrap', metric: str = 'median', tail: int = 2, ci_percentile: float = 95, parallel: str = 'cpu', n_jobs: int = -1, random_state: int | None = None, return_null: bool = False, show_progress: bool = True) -> dict
```

Compute ISC with permutation testing for statistical inference.

This method combines ISC computation with permutation testing to
determine statistical significance. It uses the same extraction
pipeline as isc() and wraps isc_permutation_test().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to test. | *required*
`method` | <code>[str](#str)</code> | ISC computation method: - 'loo': Leave-one-out (correlate each subject with mean of others) - 'pairwise': All pairwise correlations between subjects | <code>'loo'</code>
`roi_mask` | <code>'nib.Nifti1Image \| Path \| str \| None'</code> | If provided, compute ISC per ROI. Can be: - NIfTI image with integer labels (atlas/parcellation) - Path to parcellation file | <code>None</code>
`radius` | <code>[float](#float) \| None</code> | Searchlight radius in mm. If None, use voxelwise mode. Ignored if roi_mask is provided. | <code>6.0</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations/bootstrap iterations. Default 5000. | <code>5000</code>
`permutation_method` | <code>[str](#str)</code> | Method for null distribution:<br>- 'bootstrap': Subject-wise bootstrap (default, Chen et al. 2016).   Tests whether observed ISC differs from random groupings. - 'circle_shift': Circular time-shift (preserves autocorrelation).   Tests for temporally-locked shared signal. - 'phase_randomize': FFT phase randomization (preserves power spectrum).   Tests for nonlinear temporal coupling. | <code>'bootstrap'</code>
`metric` | <code>[str](#str)</code> | Summary statistic for aggregating ISC values: - 'median': Robust to outliers (default) - 'mean': Fisher z-transformed mean | <code>'median'</code>
`tail` | <code>[int](#int)</code> | One-tailed (1) or two-tailed (2) test. Default 2. | <code>2</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95). Default 95. | <code>95</code>
`parallel` | <code>[str](#str)</code> | Parallelization method ('cpu', 'gpu', or None). | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs (-1 = all cores). | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, include null distribution in results. | <code>False</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during extraction and permutation. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with: - 'isc': BrainData with ISC values - 'p': BrainData with p-values (Phipson-Smyth corrected) - 'ci': Tuple of (lower, upper) BrainData confidence intervals - 'method': ISC method used ('loo' or 'pairwise') - 'permutation_method': Permutation method used - 'extraction': Extraction mode ('roi', 'searchlight', 'voxelwise') - 'n_subjects': Number of subjects - 'n_permute': Number of permutations - 'null_dist': (optional) Null distribution array if return_null=True

**Examples:**

```pycon
>>> # ROI-based ISC with permutation testing
>>> result = bc.isc_test(roi_mask="atlas.nii.gz", n_permute=5000)
>>> sig_mask = result['p'].data < 0.05
>>> print(f"Significant ROIs: {sig_mask.sum()}")
```

```pycon
>>> # Searchlight ISC testing
>>> result = bc.isc_test(radius=10.0)
>>> result['isc'].plot()  # Show ISC values
>>> result['p'].plot()    # Show p-values
```

```pycon
>>> # Voxelwise with phase randomization (tests temporal coupling)
>>> result = bc.isc_test(
...     radius=None,
...     permutation_method='phase_randomize',
...     parallel='gpu'
... )
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Bootstrap (default) is recommended for standard ISC inference
  (Chen et al. 2016). It tests whether ISC is significant at
  the group level.
- Circle_shift and phase_randomize are more specialized - they
  test for temporally-structured shared signal beyond what's
  explained by autocorrelation or spectral structure alone.
- For large voxelwise analyses, bootstrap is much faster as it
  resamples pre-computed values rather than recomputing ISC.

</details>

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., et al. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

######## `nltools.data.collection.inference.permutation_test`

```python
permutation_test(bc: 'BrainCollection', n_permute: int = 5000, tail: int = 2, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None, return_null: bool = False) -> dict
```

One-sample permutation test across images (sign-flipping).

Tests whether the mean across images is significantly different from
zero using sign-flipping permutation. More robust than parametric
t-test for non-normal distributions.

This is a collection-level interface to
nltools.algorithms.inference.one_sample_permutation_test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to test. | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000). | <code>5000</code>
`tail` | <code>[int](#int)</code> | Test type - 1 for one-tailed, 2 for two-tailed (default). | <code>2</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization method: - 'cpu': CPU parallelization via joblib (default) - 'gpu': GPU acceleration via PyTorch - None: Single-threaded (for debugging) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores (default: -1 = all cores). | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget (default: 4.0 GB). | <code>4.0</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, include null distribution in result. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | dict with keys: - 'mean': BrainData with observed mean across images - 'p': BrainData with p-values - 'null_dist': np.ndarray (if return_null=True) - 'parallel': parallelization method used

**Examples:**

```pycon
>>> result = bc.permutation_test(n_permute=5000)
>>> mean_bd, p_bd = result['mean'], result['p']
```

```pycon
>>> # With GPU acceleration
>>> result = bc.permutation_test(parallel='gpu')
```

######## `nltools.data.collection.inference.permutation_test2`

```python
permutation_test2(bc: 'BrainCollection', other: 'BrainCollection', n_permute: int = 5000, tail: int = 2, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None, return_null: bool = False) -> dict
```

Two-sample permutation test between collections.

Tests whether two collections have different means using group
label permutation. More robust than parametric t-test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | First BrainCollection. | *required*
`other` | <code>'BrainCollection'</code> | Another BrainCollection to compare against. | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000). | <code>5000</code>
`tail` | <code>[int](#int)</code> | Test type - 1 for one-tailed, 2 for two-tailed (default). | <code>2</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization method ('cpu', 'gpu', or None). | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores (default: -1 = all cores). | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget (default: 4.0 GB). | <code>4.0</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, include null distribution in result. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | dict with keys: - 'mean_diff': BrainData with observed mean difference - 'p': BrainData with p-values - 'null_dist': np.ndarray (if return_null=True) - 'parallel': parallelization method used

**Examples:**

```pycon
>>> result = patients.permutation_test2(controls)
>>> diff_bd, p_bd = result['mean_diff'], result['p']
```

######## `nltools.data.collection.inference.project_to_brain`

```python
project_to_brain(bc: 'BrainCollection', values: np.ndarray, extraction_info: dict)
```

Project ISC values back to brain space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection (used for mask). | *required*
`values` | <code>[ndarray](#numpy.ndarray)</code> | ISC values, shape depends on extraction mode: - ROI mode: (n_rois,) - Searchlight/voxelwise: (n_voxels,) | *required*
`extraction_info` | <code>[dict](#dict)</code> | Dict from extract_for_isc with mode info. | *required*

**Returns:**

Type | Description
---- | -----------
 | BrainData with ISC values in brain space.

######## `nltools.data.collection.inference.ttest`

```python
ttest(bc: 'BrainCollection', popmean: float = 0.0, axis: int | str = 0) -> tuple
```

One-sample t-test across images.

Tests whether the mean across images is significantly different from
a population mean (default: 0). This is the voxel-wise equivalent of
scipy.stats.ttest_1samp.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to test. | *required*
`popmean` | <code>[float](#float)</code> | Population mean to test against (default: 0). | <code>0.0</code>
`axis` | <code>[int](#int) \| [str](#str)</code> | Axis to test across. Only axis=0 (images) supported. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)</code> | Tuple of (t_stat, p_value) as BrainData objects.
<code>[tuple](#tuple)</code> | Both have shape (n_obs, n_voxels) if uniform obs counts.

**Examples:**

```pycon
>>> t_stat, p_val = bc.ttest()  # Test mean != 0
>>> t_stat, p_val = bc.ttest(popmean=0.5)  # Test mean != 0.5
```

```pycon
>>> # Threshold significant voxels
>>> sig_mask = p_val.data < 0.05
```

######## `nltools.data.collection.inference.ttest2`

```python
ttest2(bc: 'BrainCollection', other: 'BrainCollection', equal_var: bool = True) -> tuple
```

Two-sample t-test between collections.

Tests whether two collections have different means. This is the
voxel-wise equivalent of scipy.stats.ttest_ind.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | First BrainCollection. | *required*
`other` | <code>'BrainCollection'</code> | Another BrainCollection to compare against. | *required*
`equal_var` | <code>[bool](#bool)</code> | If True (default), perform standard t-test assuming equal variances. If False, use Welch's t-test. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)</code> | Tuple of (t_stat, p_value) as BrainData objects.

**Examples:**

```pycon
>>> t_stat, p_val = patients.ttest2(controls)
>>> t_stat, p_val = group1.ttest2(group2, equal_var=False)  # Welch's
```

###### `nltools.data.collection.io`

I/O functions for BrainCollection.

Provides save path resolution and write functionality extracted from BrainCollection.

**Functions:**

Name | Description
---- | -----------
[`write`](#nltools.data.collection.io.write) | Write all images in collection to files.



####### Classes

####### Functions######## `nltools.data.collection.io.write`

```python
write(bc: 'BrainCollection', directory: 'str | Path', pattern: str = 'image_{i:04d}.nii.gz', metadata_file: str | None = 'metadata.csv') -> list['Path']
```

Write all images in collection to files.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to write. | *required*
`directory` | <code>'str \| Path'</code> | Output directory path. Will be created if it doesn't exist. | *required*
`pattern` | <code>[str](#str)</code> | Filename pattern with {i} placeholder for image index. Default: "image_{i:04d}.nii.gz" produces image_0000.nii.gz, etc. | <code>'image_{i:04d}.nii.gz'</code>
`metadata_file` | <code>[str](#str) \| None</code> | Optional filename for metadata CSV. Set to None to skip. Default: "metadata.csv" | <code>'metadata.csv'</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)['Path']</code> | List of paths to written files.

**Examples:**

```pycon
>>> bc = BrainCollection([bd1, bd2, bd3], mask=mask)
>>> paths = write(bc, "output/")
>>> # Creates: output/image_0000.nii.gz, image_0001.nii.gz, etc.
```

```pycon
>>> # Custom pattern
>>> write(bc, "output/", pattern="sub-{i:02d}_bold.nii.gz")
>>> # Creates: output/sub-00_bold.nii.gz, sub-01_bold.nii.gz, etc.
```

```pycon
>>> # With BIDS-style naming using metadata
>>> bc.metadata["filename"] = [f"sub-{s}_bold.nii.gz" for s in subjects]
>>> for i, bd in enumerate(bc):
...     bd.write(f"output/{bc.metadata.loc[i, 'filename']}")
```

###### `nltools.data.collection.modeling`

Modeling functions extracted from BrainCollection.

Contains GLM fitting, Ridge fitting, design matrix building, and related helpers.
All BrainCollection methods converted to functions taking `bc` as first argument.

**Functions:**

Name | Description
---- | -----------
[`cv`](#nltools.data.collection.modeling.cv) | Create a cross-validation pipeline for multi-subject analysis.
[`fit`](#nltools.data.collection.modeling.fit) | Fit a model to each subject in the collection.
[`fit_from_events`](#nltools.data.collection.modeling.fit_from_events) | Build design matrices from events and fit GLM to each subject.
[`fit_glm`](#nltools.data.collection.modeling.fit_glm) | Fit GLM to each subject in collection.
[`fit_glm_internal`](#nltools.data.collection.modeling.fit_glm_internal) | Internal GLM fitting with design matrix input.
[`fit_ridge`](#nltools.data.collection.modeling.fit_ridge) | Fit ridge regression to each subject in collection.
[`load_design_matrix`](#nltools.data.collection.modeling.load_design_matrix) | Load design matrix from a file path.
[`load_features`](#nltools.data.collection.modeling.load_features) | Load features from a file path.
[`resolve_X`](#nltools.data.collection.modeling.resolve_X) | Resolve design/feature matrix X to per-subject list.
[`resolve_confounds`](#nltools.data.collection.modeling.resolve_confounds) | Resolve confounds argument to per-subject list.



####### Classes

####### Functions######## `nltools.data.collection.modeling.cv`

```python
cv(bc, k: int | None = None, scheme: str = 'kfold', split_by: str | None = None, groups: np.ndarray | None = None, random_state: int | None = None, **kwargs: int | None) -> 'BrainCollectionPipeline'
```

Create a cross-validation pipeline for multi-subject analysis.

Returns a pipeline object that enables fluent, chainable transforms
with cross-validation across subjects or runs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`k` | <code>[int](#int) \| None</code> | Number of folds (for kfold scheme). Defaults to 5. | <code>None</code>
`scheme` | <code>[str](#str)</code> | CV scheme type. Options: - 'kfold': k-fold cross-validation on pooled data - 'loso': leave-one-subject-out (one image held out per fold) - 'loro': leave-one-run-out (requires groups) | <code>'kfold'</code>
`split_by` | <code>[str](#str) \| None</code> | Metadata column for group splits. If provided and groups is None, gets groups from bc.metadata[split_by]. | <code>None</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Explicit group labels for CV splits. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`**kwargs` |  | Additional arguments passed to CVScheme. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainCollectionPipeline` | <code>'BrainCollectionPipeline'</code> | Pipeline for method chaining.

**Examples:**

```pycon
>>> # Leave-one-subject-out classification
>>> result = bc.cv(scheme='loso').normalize().predict(subject_labels, algorithm='svm')
>>> print(f"Mean accuracy: {result.mean_score:.2%}")
```

```pycon
>>> # With preprocessing
>>> result = (bc
...     .cv(scheme='loso')
...     .normalize()
...     .reduce(n_components=50)
...     .predict(labels))
```

```pycon
>>> # Run-based CV with metadata
>>> result = bc.cv(scheme='loro', split_by='run').predict(y)
```

<details class="see-also" open markdown="1">
<summary>See Also</summary>

BrainCollectionPipeline: For available transforms and terminals.
CVScheme: For CV scheme configuration details.

</details>

######## `nltools.data.collection.modeling.fit`

```python
fit(bc, model: str, X: 'pd.DataFrame | np.ndarray | str | list', cv: int | None = None, scale: bool = True, scale_value: float = 100.0, show_progress: bool = True, **kwargs: bool) -> 'FittedBrainCollection'
```

Fit a model to each subject in the collection.

Unified fitting method that shadows BrainData.fit() API for multi-subject
analysis. Dispatches to model-specific implementations based on the
model parameter.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`model` | <code>[str](#str)</code> | Model type - 'glm' or 'ridge' | *required*
`X` | <code>'pd.DataFrame \| np.ndarray \| str \| list'</code> | Design/feature matrix. Can be: - pd.DataFrame/DesignMatrix: Shared (used for all subjects) - np.ndarray: Shared array (used for all subjects) - str: Column name in metadata pointing to file paths - list: Per-subject list of DataFrames/arrays/paths | *required*
`cv` | <code>[int](#int) \| None</code> | Cross-validation folds (Ridge only). Default is None for GLM, 5 for Ridge when output='scores'. | <code>None</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`**kwargs` |  | Model-specific arguments passed to _fit_glm or _fit_ridge: - GLM: return_stats, save - Ridge: alpha, output, save, backend, random_state | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'FittedBrainCollection'</code> | FittedBrainCollection wrapping the fitted results. Supports:
<code>'FittedBrainCollection'</code> | - ``.results``: Access underlying BrainCollection(s) directly
<code>'FittedBrainCollection'</code> | - ``.betas``: Convenience accessor for beta coefficients (GLM)
<code>'FittedBrainCollection'</code> | - ``.pool()``: Aggregate across subjects for group analysis
<code>'FittedBrainCollection'</code> | The underlying results contain:
<code>'FittedBrainCollection'</code> | - GLM: Beta coefficients (n_regressors, n_voxels) per subject
<code>'FittedBrainCollection'</code> | - Ridge: Scores or weights depending on 'output' kwarg
<code>'FittedBrainCollection'</code> | If return_stats (GLM) or output='both' (Ridge), results is a dict.

**Examples:**

```pycon
>>> # GLM with shared design matrix
>>> fitted = bc.fit(model='glm', X=dm)
>>> betas = fitted.results  # Access BrainCollection directly
>>>
>>> # Two-stage analysis with pool()
>>> pool = bc.fit(model='glm', X=dm).pool(param='beta')
>>> t_map = pool.fit(model='ttest', contrast='A-B')
>>>
>>> # GLM with per-subject design matrices
>>> fitted = bc.fit(model='glm', X=[dm1, dm2, dm3])
>>>
>>> # Ridge encoding model with CV scores
>>> fitted = bc.fit(model='ridge', X=features, cv=5)
>>> scores = fitted.results
```

<details class="see-also" open markdown="1">
<summary>See Also</summary>

fit_from_events: Convenience method for event-based GLM workflows
fit_glm: Legacy GLM fitting (use fit_from_events instead)
fit_ridge: Legacy Ridge fitting (use fit(..., model='ridge') instead)

</details>

######## `nltools.data.collection.modeling.fit_from_events`

```python
fit_from_events(bc, events: pd.DataFrame, t_r: float, confounds: str | list[pd.DataFrame | Path | str] | None = None, confound_columns: list[str] | None = None, hrf_model: str = 'spm', drift_model: str = 'cosine', high_pass: float = 0.01, scale: bool = True, scale_value: float = 100.0, return_stats: list[str] | None = None, return_residuals: bool = False, save: dict[str, str] | None = None, show_progress: bool = True, by_run: bool = False, run_column: str = 'run', run_lengths: int | list[int] | None = None) -> 'BrainCollection | dict[str, BrainCollection]'
```

Build design matrices from events and fit GLM to each subject.

Convenience method for event-based experimental designs. Builds
nilearn-compatible design matrices from the events DataFrame and
fits a GLM to each subject in the collection.

This is the recommended method for typical task-based fMRI analysis
where you have event timing information. For more control, use
fit(model='glm', X=design_matrices) with pre-built design matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`events` | <code>[DataFrame](#pandas.DataFrame)</code> | Task events DataFrame with onset, duration, trial_type columns. This is shared across all subjects (same experimental paradigm). If by_run=True, must also have a run column. | *required*
`t_r` | <code>[float](#float)</code> | Repetition time (TR) in seconds. | *required*
`confounds` | <code>[str](#str) \| [list](#list)[[DataFrame](#pandas.DataFrame) \| [Path](#pathlib.Path) \| [str](#str)] \| None</code> | Subject-specific confounds. Can be: - str: Column name in metadata pointing to confound file paths - list: List of DataFrames or paths, one per subject - None: No confounds (only task + drift terms) | <code>None</code>
`confound_columns` | <code>[list](#list)[[str](#str)] \| None</code> | Columns to extract from confound files. If None and confounds provided, uses all columns. | <code>None</code>
`hrf_model` | <code>[str](#str)</code> | HRF model for convolution ('spm', 'glover', 'fir', etc.) | <code>'spm'</code>
`drift_model` | <code>[str](#str)</code> | Drift model ('cosine', 'polynomial', None) | <code>'cosine'</code>
`high_pass` | <code>[float](#float)</code> | High-pass filter cutoff in Hz (default 0.01) | <code>0.01</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`return_stats` | <code>[list](#list)[[str](#str)] \| None</code> | Optional list of statistics to return as separate BrainCollections. Options: 't', 'r2', 'p', 'se', 'residual'. | <code>None</code>
`return_residuals` | <code>[bool](#bool)</code> | If True, return residuals (same as return_stats=['residual']). | <code>False</code>
`save` | <code>[dict](#dict)[[str](#str), [str](#str)] \| None</code> | Dict mapping output type to path template. | <code>None</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`by_run` | <code>[bool](#bool)</code> | If True, fit GLM separately per run and return run-level betas. This enables MVPA decoding with leave-one-run-out CV. | <code>False</code>
`run_column` | <code>[str](#str)</code> | Column name in events identifying runs (default 'run'). | <code>'run'</code>
`run_lengths` | <code>[int](#int) \| [list](#list)[[int](#int)] \| None</code> | Number of TRs per run. Required when by_run=True. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection of beta coefficients for task regressors.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | If return_stats specified, returns dict with keys 'betas', 't', etc.

**Examples:**

```pycon
>>> # Basic GLM fit from events
>>> betas = bc.fit_from_events(events=events_df, t_r=2.0)
>>> group_t = betas.ttest()
>>>
>>> # With confounds from metadata column
>>> betas = bc.fit_from_events(
...     events=events_df,
...     t_r=2.0,
...     confounds='confound_file',
...     confound_columns=['trans_x', 'trans_y', 'trans_z']
... )
>>>
>>> # Run-level betas for MVPA
>>> betas = bc.fit_from_events(events=events_df, t_r=2.0, by_run=True)
```

<details class="see-also" open markdown="1">
<summary>See Also</summary>

fit: Unified fit method that accepts pre-built design matrices
_fit_glm: Internal method for design matrix-based fitting

</details>

######## `nltools.data.collection.modeling.fit_glm`

```python
fit_glm(bc, events: pd.DataFrame, t_r: float, confounds: str | list[pd.DataFrame | Path | str] | None = None, confound_columns: list[str] | None = None, hrf_model: str = 'spm', drift_model: str = 'cosine', high_pass: float = 0.01, scale: bool = True, scale_value: float = 100.0, return_stats: list[str] | None = None, return_residuals: bool = False, save: dict[str, str] | None = None, show_progress: bool = True, by_run: bool = False, run_column: str = 'run', run_lengths: int | list[int] | None = None) -> 'BrainCollection | dict[str, BrainCollection]'
```

Fit GLM to each subject in collection.

Memory-efficient first-level GLM analysis that processes subjects
one at a time. Returns a BrainCollection of beta coefficients for
task regressors (confounds and drift terms are fit but not returned).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`events` | <code>[DataFrame](#pandas.DataFrame)</code> | Task events DataFrame with onset, duration, trial_type columns. This is shared across all subjects (same experimental paradigm). If by_run=True, must also have a run column. | *required*
`t_r` | <code>[float](#float)</code> | Repetition time (TR) in seconds. | *required*
`confounds` | <code>[str](#str) \| [list](#list)[[DataFrame](#pandas.DataFrame) \| [Path](#pathlib.Path) \| [str](#str)] \| None</code> | Subject-specific confounds. Can be: - str: Column name in metadata pointing to confound file paths - list: List of DataFrames or paths, one per subject - None: No confounds (only task + drift terms) | <code>None</code>
`confound_columns` | <code>[list](#list)[[str](#str)] \| None</code> | Columns to extract from confound files. If None and confounds provided, uses all columns. | <code>None</code>
`hrf_model` | <code>[str](#str)</code> | HRF model for convolution ('spm', 'glover', 'fir', etc.) | <code>'spm'</code>
`drift_model` | <code>[str](#str)</code> | Drift model ('cosine', 'polynomial', None) | <code>'cosine'</code>
`high_pass` | <code>[float](#float)</code> | High-pass filter cutoff in Hz (default 0.01) | <code>0.01</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`return_stats` | <code>[list](#list)[[str](#str)] \| None</code> | Optional list of statistics to return as separate BrainCollections. Options: 't', 'r2', 'p', 'se', 'residual'. | <code>None</code>
`return_residuals` | <code>[bool](#bool)</code> | If True, return residuals (same as return_stats=['residual']). | <code>False</code>
`save` | <code>[dict](#dict)[[str](#str), [str](#str)] \| None</code> | Dict mapping output type to path template, e.g. ``{'betas': 'output/{subject}_betas.nii.gz', 't': 'output/{subject}_tstat.nii.gz'}``. Supports {subject}, {session}, {idx}, and other metadata columns. | <code>None</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`by_run` | <code>[bool](#bool)</code> | If True, fit GLM separately per run and return run-level betas. This enables MVPA decoding with leave-one-run-out CV. Each subject will have (n_runs * n_conditions, n_voxels) betas. | <code>False</code>
`run_column` | <code>[str](#str)</code> | Column name in events identifying runs (default 'run'). | <code>'run'</code>
`run_lengths` | <code>[int](#int) \| [list](#list)[[int](#int)] \| None</code> | Number of TRs per run. Required when by_run=True.<br>- int: All runs have same length - list of int: Different length per run - None: Will attempt to infer equal-length runs from total scans | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection where each BrainData has shape:
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | - (n_task_regressors, n_voxels) if by_run=False (default)
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | - (n_runs * n_task_regressors, n_voxels) if by_run=True
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | The ``._design_columns`` attribute stores task regressor names.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | If by_run=True, also stores ``._condition_labels`` and ``._run_labels``.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | If return_stats specified, returns dict with keys 'betas', 't', etc.

**Examples:**

```pycon
>>> # Basic GLM fit
>>> betas = bc.fit_glm(events=events_df, t_r=2.0)
>>> # Group t-test on first regressor
>>> group_t = betas[:, 0].ttest()
```

```pycon
>>> # Run-level betas for MVPA decoding
>>> betas = bc.fit_glm(events=events_df, t_r=2.0, by_run=True)
>>> # betas._condition_labels = ['face', 'house', 'face', 'house', ...]
>>> # betas._run_labels = [1, 1, 2, 2, 3, 3, ...]
>>> accuracy = betas.predict(y=None, method='searchlight')
```

```pycon
>>> # With confounds from metadata column
>>> betas = bc.fit_glm(
...     events=events_df,
...     t_r=2.0,
...     confounds='confound_file',  # column name in metadata
...     confound_columns=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
... )
```

######## `nltools.data.collection.modeling.fit_glm_internal`

```python
fit_glm_internal(bc, X: 'pd.DataFrame | np.ndarray | str | list', scale: bool = True, scale_value: float = 100.0, return_stats: list[str] | None = None, save: dict[str, str] | None = None, show_progress: bool = True) -> 'BrainCollection | dict[str, BrainCollection]'
```

Internal GLM fitting with design matrix input.

Core implementation that accepts DesignMatrix/DataFrame directly.
Called by fit(model='glm') and fit_from_events().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`X` | <code>'pd.DataFrame \| np.ndarray \| str \| list'</code> | Design matrix. Can be: - pd.DataFrame/DesignMatrix: Shared (used for all subjects) - np.ndarray: Shared array (converted to DataFrame internally) - str: Column name in metadata pointing to file paths - list: Per-subject list of DataFrames/arrays/paths | *required*
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`return_stats` | <code>[list](#list)[[str](#str)] \| None</code> | Optional list of statistics to return as separate BrainCollections. Options: 't', 'r2', 'p', 'se', 'residual'. | <code>None</code>
`save` | <code>[dict](#dict)[[str](#str), [str](#str)] \| None</code> | Dict mapping output type to path template. | <code>None</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection of betas, or dict with betas + requested stats.

######## `nltools.data.collection.modeling.fit_ridge`

```python
fit_ridge(bc, X: 'np.ndarray | str | list', alpha: float | str = 1.0, cv: int | None = 5, scale: bool = True, scale_value: float = 100.0, output: str = 'scores', save: dict[str, str] | None = None, show_progress: bool = True, **ridge_kwargs: bool) -> 'BrainCollection | dict[str, BrainCollection]'
```

Fit ridge regression to each subject in collection.

Memory-efficient encoding model fitting that processes subjects one at a
time. Default behavior returns cross-validated R-squared scores per voxel,
suitable for group-level inference on encoding model performance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`X` | <code>'np.ndarray \| str \| list'</code> | Feature matrix. Can be: - np.ndarray: Shared features (n_samples, n_features) used for all subjects - str: Column name in metadata pointing to feature file paths - list: List of arrays/DataFrames, one per subject | *required*
`alpha` | <code>[float](#float) \| [str](#str)</code> | Ridge regularization parameter. Can be: - float: Fixed regularization strength - 'auto': Use cross-validation to select optimal alpha | <code>1.0</code>
`cv` | <code>[int](#int) \| None</code> | Cross-validation folds for computing scores. Default is 5. Required when output='scores' or 'both'. Set to None only when output='weights'. | <code>5</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`output` | <code>[str](#str)</code> | What to return. Options: - 'scores': CV R-squared scores per voxel (default, for encoding workflow) - 'weights': Model weights (n_features, n_voxels) - 'both': Dict with both 'scores' and 'weights' | <code>'scores'</code>
`save` | <code>[dict](#dict)[[str](#str), [str](#str)] \| None</code> | Dict mapping output type to path template, e.g. ``{'weights': 'output/{subject}_weights.nii.gz', 'scores': 'output/{subject}_scores.nii.gz'}``. Supports {subject}, {session}, {idx}, and other metadata columns. | <code>None</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`**ridge_kwargs` |  | Additional arguments passed to Ridge model (e.g., backend='torch', random_state=42). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection of scores or weights, or dict with both if output='both'.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | Each BrainData will have ``cv_results_`` attribute when cv is used.

**Examples:**

```pycon
>>> # Encoding model workflow: get CV scores for group analysis
>>> scores = bc.fit_ridge(X=features, alpha=1.0)
>>> group_ttest = scores.ttest()  # Test encoding accuracy vs chance
```

```pycon
>>> # Get both scores and weights
>>> results = bc.fit_ridge(X=features, alpha=1.0, output='both')
>>> scores = results['scores']
>>> weights = results['weights']
```

```pycon
>>> # Auto-select alpha with CV
>>> scores = bc.fit_ridge(X=features, alpha='auto', cv=5)
```

```pycon
>>> # Get weights only (no CV needed)
>>> weights = bc.fit_ridge(X=features, alpha=1.0, output='weights', cv=None)
```

######## `nltools.data.collection.modeling.load_design_matrix`

```python
load_design_matrix(bc, path: str | Path) -> pd.DataFrame
```

Load design matrix from a file path.

Supports common formats: .csv, .tsv, .txt

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance (unused, kept for API consistency). | *required*
`path` | <code>[str](#str) \| [Path](#pathlib.Path)</code> | Path to design matrix file. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#pandas.DataFrame)</code> | DataFrame with design matrix contents.

######## `nltools.data.collection.modeling.load_features`

```python
load_features(bc, path: str | Path) -> np.ndarray
```

Load features from a file path.

Supports common formats: .npy, .csv, .tsv, .txt

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance (unused, kept for API consistency). | *required*
`path` | <code>[str](#str) \| [Path](#pathlib.Path)</code> | Path to feature file. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | NumPy array of feature values.

######## `nltools.data.collection.modeling.resolve_X`

```python
resolve_X(bc, X: 'np.ndarray | pd.DataFrame | str | list | None') -> list | None
```

Resolve design/feature matrix X to per-subject list.

Unified helper for resolving X parameter across fit methods. Supports
three input patterns:
1. Shared matrix (array/DataFrame/DesignMatrix): Same X for all subjects
2. Per-subject list: List of matrices, one per subject
3. Metadata column: String column name pointing to file paths

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`X` | <code>'np.ndarray \| pd.DataFrame \| str \| list \| None'</code> | Design/feature matrix. Can be: - np.ndarray: Shared array (used for all subjects) - pd.DataFrame: Shared DataFrame/DesignMatrix (used for all subjects) - str: Column name in metadata containing file paths - list: Per-subject list of arrays/DataFrames/paths - None: Error | *required*

**Returns:**

Type | Description
---- | -----------
<code>[list](#list) \| None</code> | list | None: Per-subject list if X varies by subject, None if shared. Caller should use: `X_subj = X_list[i] if X_list else X`

######## `nltools.data.collection.modeling.resolve_confounds`

```python
resolve_confounds(bc, confounds: str | list[pd.DataFrame | Path | str] | None) -> list[pd.DataFrame | Path | str] | None
```

Resolve confounds argument to per-subject list.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`confounds` | <code>[str](#str) \| [list](#list)[[DataFrame](#pandas.DataFrame) \| [Path](#pathlib.Path) \| [str](#str)] \| None</code> | Either: - str: Column name in metadata containing confound paths - list: Already per-subject list of DataFrames or paths - None: No confounds | *required*

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[DataFrame](#pandas.DataFrame) \| [Path](#pathlib.Path) \| [str](#str)] \| None</code> | List of confounds (one per subject) or None

###### `nltools.data.collection.pipeline`

Pipeline classes for BrainCollection.

Provides BrainCollectionPipeline for fluent pipeline API with cross-validation,
BrainCollectionCVResult for storing CV results, and FittedBrainCollection for
chaining pool() after fit().

**Classes:**

Name | Description
---- | -----------
[`BrainCollectionCVResult`](#nltools.data.collection.pipeline.BrainCollectionCVResult) | Cross-validation results for BrainCollection pipelines.
[`BrainCollectionPipeline`](#nltools.data.collection.pipeline.BrainCollectionPipeline) | Pipeline for BrainCollection with multi-subject CV support.
[`FittedBrainCollection`](#nltools.data.collection.pipeline.FittedBrainCollection) | Wrapper for fitted BrainCollection enabling pool() chaining.



####### Classes######## `nltools.data.collection.pipeline.BrainCollectionCVResult`

```python
BrainCollectionCVResult(fold_results: list, pipeline: BrainCollectionPipeline)
```

Cross-validation results for BrainCollection pipelines.

Contains fold-by-fold results from cross-validated prediction,
with convenience properties for accessing scores and predictions.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`fold_results`](#nltools.data.collection.pipeline.BrainCollectionCVResult.fold_results) |  | List of dictionaries with per-fold results.
[`pipeline`](#nltools.data.collection.pipeline.BrainCollectionCVResult.pipeline) |  | The pipeline that generated these results.
[`scores`](#nltools.data.collection.pipeline.BrainCollectionCVResult.scores) | <code>[ndarray](#numpy.ndarray)</code> | Per-fold prediction scores.
[`mean_score`](#nltools.data.collection.pipeline.BrainCollectionCVResult.mean_score) | <code>[float](#float)</code> | Mean score across all folds.
[`std_score`](#nltools.data.collection.pipeline.BrainCollectionCVResult.std_score) | <code>[float](#float)</code> | Standard deviation of scores.
[`n_folds`](#nltools.data.collection.pipeline.BrainCollectionCVResult.n_folds) | <code>[int](#int)</code> | Number of CV folds.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fold_results` | <code>[list](#list)</code> | List of fold result dictionaries. | *required*
`pipeline` | <code>[BrainCollectionPipeline](#nltools.data.collection.pipeline.BrainCollectionPipeline)</code> | The pipeline that generated these results. | *required*



######### Attributes########## `nltools.data.collection.pipeline.BrainCollectionCVResult.fold_results`

```python
fold_results = fold_results
```

########## `nltools.data.collection.pipeline.BrainCollectionCVResult.mean_score`

```python
mean_score: float
```

Mean score across folds.

########## `nltools.data.collection.pipeline.BrainCollectionCVResult.n_folds`

```python
n_folds: int
```

Number of cross-validation folds.

########## `nltools.data.collection.pipeline.BrainCollectionCVResult.pipeline`

```python
pipeline = pipeline
```

########## `nltools.data.collection.pipeline.BrainCollectionCVResult.scores`

```python
scores: np.ndarray
```

Per-fold prediction scores as a numpy array.

########## `nltools.data.collection.pipeline.BrainCollectionCVResult.std_score`

```python
std_score: float
```

Standard deviation of scores.

######## `nltools.data.collection.pipeline.BrainCollectionPipeline`

```python
BrainCollectionPipeline(brain_collection: 'BrainCollection', cv: 'BrainCollection' = None, groups: np.ndarray | None = None)
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
[`n_subjects`](#nltools.data.collection.pipeline.BrainCollectionPipeline.n_subjects) | <code>[int](#int)</code> | Number of subjects/images in the collection.
[`cv`](#nltools.data.collection.pipeline.BrainCollectionPipeline.cv) |  | The cross-validation scheme configuration.
[`n_steps`](#nltools.data.collection.pipeline.BrainCollectionPipeline.n_steps) | <code>[int](#int)</code> | Number of transform steps in the pipeline.

**Examples:**

```pycon
>>> # Leave-one-subject-out with preprocessing
>>> result = (bc
...     .cv(scheme='loso')
...     .normalize()
...     .reduce(n_components=50)
...     .predict(labels, algorithm='svm'))
>>> print(f"Mean accuracy: {result.mean_score:.2%}")
```

**Functions:**

Name | Description
---- | -----------
[`normalize`](#nltools.data.collection.pipeline.BrainCollectionPipeline.normalize) | Add normalization step.
[`pipe`](#nltools.data.collection.pipeline.BrainCollectionPipeline.pipe) | Add custom sklearn transformer.
[`predict`](#nltools.data.collection.pipeline.BrainCollectionPipeline.predict) | Execute pipeline with CV and return prediction results.
[`reduce`](#nltools.data.collection.pipeline.BrainCollectionPipeline.reduce) | Add dimensionality reduction step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_collection` | <code>'BrainCollection'</code> | BrainCollection to wrap. | *required*
`cv` |  | CVScheme configuration. | <code>None</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Group labels for CV splits. | <code>None</code>



######### Attributes########## `nltools.data.collection.pipeline.BrainCollectionPipeline.cv`

```python
cv
```

Cross-validation scheme.

########## `nltools.data.collection.pipeline.BrainCollectionPipeline.n_steps`

```python
n_steps: int
```

Number of transform steps.

########## `nltools.data.collection.pipeline.BrainCollectionPipeline.n_subjects`

```python
n_subjects: int
```

Number of subjects/images.



######### Functions########## `nltools.data.collection.pipeline.BrainCollectionPipeline.normalize`

```python
normalize(method: str = 'zscore', **kwargs: str) -> 'BrainCollectionPipeline'
```

Add normalization step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Normalization method ('zscore', 'minmax'). | <code>'zscore'</code>
`**kwargs` |  | Additional arguments for NormalizeStep. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollectionPipeline'</code> | New pipeline with normalization step added.

########## `nltools.data.collection.pipeline.BrainCollectionPipeline.pipe`

```python
pipe(transformer) -> 'BrainCollectionPipeline'
```

Add custom sklearn transformer.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`transformer` |  | sklearn-compatible transformer with fit/transform interface. | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollectionPipeline'</code> | New pipeline with custom step added.

########## `nltools.data.collection.pipeline.BrainCollectionPipeline.predict`

```python
predict(y, algorithm: str = 'ridge', **kwargs: str) -> 'BrainCollectionCVResult'
```

Execute pipeline with CV and return prediction results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` |  | Target variable. For LOSO, shape should be (n_subjects,). | *required*
`algorithm` | <code>[str](#str)</code> | Prediction algorithm ('ridge', 'svm', 'logistic', etc.) | <code>'ridge'</code>
`**kwargs` |  | Passed to model constructor. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollectionCVResult'</code> | Cross-validation results with scores and predictions.

########## `nltools.data.collection.pipeline.BrainCollectionPipeline.reduce`

```python
reduce(method: str = 'pca', n_components: int | None = None, **kwargs: int | None) -> 'BrainCollectionPipeline'
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
<code>'BrainCollectionPipeline'</code> | New pipeline with reduction step added.

######## `nltools.data.collection.pipeline.FittedBrainCollection`

```python
FittedBrainCollection(brain_collection: 'BrainCollection', fitted_results: 'BrainCollection | dict[str, BrainCollection]', model: str, condition_names: list[str] | None = None)
```

Wrapper for fitted BrainCollection enabling pool() chaining.

This class wraps the results of bc.fit() and provides the .pool()
method for aggregating across subjects.

The execution model:
- fit() executes immediately (eager)
- pool() aggregates the fitted parameters
- pool() returns PooledData for second-level analysis

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_collection` | <code>'BrainCollection'</code> | The original collection that was fitted. | *required*
`fitted_results` | <code>'BrainCollection \| dict[str, BrainCollection]'</code> | The fitted results. Can be a BrainCollection (betas or scores) or a dict mapping stat names to BrainCollections (e.g., {'betas': ..., 't': ...}). | *required*
`model` | <code>[str](#str)</code> | The model type that was fitted ('glm' or 'ridge'). | *required*
`condition_names` | <code>[list](#list)[[str](#str)] \| None</code> | Names of conditions/regressors from the design matrix. | <code>None</code>

Examples
--------
>>> fitted = bc.fit(model='glm', X=dm)
>>> pool = fitted.pool(param='beta')
>>> result = pool.fit(model='ttest', contrast='A-B')

**Functions:**

Name | Description
---- | -----------
[`pool`](#nltools.data.collection.pipeline.FittedBrainCollection.pool) | Pool fitted parameters across subjects.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`betas`](#nltools.data.collection.pipeline.FittedBrainCollection.betas) | <code>'BrainCollection'</code> | Convenience accessor for beta coefficients from a GLM fit.
[`n_subjects`](#nltools.data.collection.pipeline.FittedBrainCollection.n_subjects) | <code>[int](#int)</code> | Number of subjects in the fitted collection.
[`results`](#nltools.data.collection.pipeline.FittedBrainCollection.results) | <code>'BrainCollection \| dict[str, BrainCollection]'</code> | Access the fitted results directly.



######### Attributes########## `nltools.data.collection.pipeline.FittedBrainCollection.betas`

```python
betas: 'BrainCollection'
```

Convenience accessor for beta coefficients from a GLM fit.

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | Beta coefficients from GLM fit.

########## `nltools.data.collection.pipeline.FittedBrainCollection.n_subjects`

```python
n_subjects: int
```

Number of subjects in the fitted collection.

########## `nltools.data.collection.pipeline.FittedBrainCollection.results`

```python
results: 'BrainCollection | dict[str, BrainCollection]'
```

Access the fitted results directly.

Returns the underlying BrainCollection or dict of BrainCollections.
Use this for backward compatibility or when pool() is not needed.



######### Functions########## `nltools.data.collection.pipeline.FittedBrainCollection.pool`

```python
pool(param: str = 'beta', contrast: str | None = None, save: str | None = None, save_fitted: bool = False)
```

Pool fitted parameters across subjects.

Aggregates per-subject fitted results for group-level analysis.
Returns a PooledData object that can be passed to second-level
statistical tests.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`param` | <code>[str](#str)</code> | Parameter to pool. GLM options: 'beta', 't', 'r2', 'p', 'se', 'residual'. Ridge options: 'scores', 'weights'. Default is 'beta'. | <code>'beta'</code>
`contrast` | <code>[str](#str) \| None</code> | Apply contrast before pooling. Format: 'A-B' or 'A+B'. Requires condition_names to be available. | <code>None</code>
`save` | <code>[str](#str) \| None</code> | Path template to save per-subject results before pooling. Supports {subject}, {idx} placeholders. | <code>None</code>
`save_fitted` | <code>[bool](#bool)</code> | If True, save full fitted state for later repool(). | <code>False</code>

**Returns:**

Type | Description
---- | -----------
 | Pooled data ready for second-level analysis.

Examples
--------
>>> pool = bc.fit(model='glm', X=designs).pool(param='beta')
>>> result = pool.fit(model='ttest', contrast='face-house')

>>> # Pool t-statistics instead of betas
>>> pool = bc.fit(model='glm', X=dm, return_stats=['t']).pool(param='t')

###### `nltools.data.collection.prediction`

Prediction functions extracted from BrainCollection.

Contains predict, compute_contrasts, select_feature, and related helpers.
All BrainCollection methods converted to functions taking `bc` as first argument.

**Functions:**

Name | Description
---- | -----------
[`compute_contrasts`](#nltools.data.collection.prediction.compute_contrasts) | Compute contrasts from fitted GLM beta coefficients.
[`compute_single_contrast`](#nltools.data.collection.prediction.compute_single_contrast) | Compute a single contrast across all subjects.
[`parse_contrast_string`](#nltools.data.collection.prediction.parse_contrast_string) | Parse a contrast string into a numeric contrast vector.
[`predict`](#nltools.data.collection.prediction.predict) | Generate predictions for each subject in collection.
[`select_feature`](#nltools.data.collection.prediction.select_feature) | Select a single feature's weights across all subjects.



####### Classes

####### Functions######## `nltools.data.collection.prediction.compute_contrasts`

```python
compute_contrasts(bc, contrasts: 'str | dict | np.ndarray | list') -> 'BrainCollection | dict[str, BrainCollection]'
```

Compute contrasts from fitted GLM beta coefficients.

Applies contrast weights to each subject's betas and returns a
BrainCollection of contrast values suitable for group-level analysis.

Must be called on a BrainCollection created by fit_glm() which has
the _design_columns attribute set.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`contrasts` | <code>'str \| dict \| np.ndarray \| list'</code> | Can be: - str: Contrast string using column names, e.g., "face - house" - dict: Multiple contrasts, e.g., {"main": "face - house", "avg": [0.5, 0.5]} - array/list: Numeric contrast vector, e.g., [1, -1] | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection where each BrainData has shape (n_voxels,) containing
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | the contrast values. If dict input, returns dict of BrainCollections.

**Examples:**

```pycon
>>> # Fit GLM and compute contrast
>>> betas = bc.fit_glm(events=events_df, t_r=2.0)
>>> contrast = betas.compute_contrasts("face - house")
>>> # Group t-test on contrast
>>> group_result = contrast.ttest()
```

```pycon
>>> # Multiple contrasts
>>> contrasts = betas.compute_contrasts({
...     "face_vs_house": "face - house",
...     "face_vs_baseline": "face",
... })
>>> face_vs_house_ttest = contrasts["face_vs_house"].ttest()
```

######## `nltools.data.collection.prediction.compute_single_contrast`

```python
compute_single_contrast(bc, contrast: 'str | np.ndarray | list', design_columns: list[str]) -> 'BrainCollection'
```

Compute a single contrast across all subjects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`contrast` | <code>'str \| np.ndarray \| list'</code> | Contrast specification (string, array, or list) | *required*
`design_columns` | <code>[list](#list)[[str](#str)]</code> | List of regressor names from fit_glm | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with contrast values for each subject

######## `nltools.data.collection.prediction.parse_contrast_string`

```python
parse_contrast_string(bc, contrast_str: str, design_columns: list[str]) -> np.ndarray
```

Parse a contrast string into a numeric contrast vector.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance (unused, kept for API consistency). | *required*
`contrast_str` | <code>[str](#str)</code> | Contrast string like "A - B" or "2*A - B" | *required*
`design_columns` | <code>[list](#list)[[str](#str)]</code> | List of regressor column names | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Numeric contrast vector

######## `nltools.data.collection.prediction.predict`

```python
predict(bc, X: 'np.ndarray | str | list | None' = None, y: 'np.ndarray | None' = None, method: str = 'whole_brain', estimator: str = 'svm', cv: str = 5, groups: 'np.ndarray | None' = None, roi_mask: 'np.ndarray | None' = None, radius: float = 10.0, scoring: str = 'accuracy', standardize: bool = True, n_jobs: int = -1, show_progress: bool = True) -> 'BrainCollection'
```

Generate predictions for each subject in collection.

This method supports two prediction modes determined by which parameter
is provided:

1. **Timeseries prediction** (X provided): Use fitted ridge model to
   predict voxel responses for new feature data.

2. **MVPA decoding** (y provided): Train a classifier to predict labels
   from brain patterns using cross-validation.

For MVPA, if this collection was created with by_run=True, you can
use y=None to infer labels from _condition_labels and groups from
_run_labels (leave-one-run-out CV).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`X` | <code>'np.ndarray \| str \| list \| None'</code> | Features for timeseries prediction. Can be: - np.ndarray: Shared features (same for all subjects) - str: Metadata column with per-subject feature paths - list: Per-subject feature arrays | <code>None</code>
`y` | <code>'np.ndarray \| None'</code> | Labels for MVPA decoding. If None and _condition_labels exists, will use stored condition labels (from fit_glm with by_run=True). | <code>None</code>
`method` | <code>[str](#str)</code> | MVPA method - 'whole_brain', 'searchlight', or 'roi'. | <code>'whole_brain'</code>
`estimator` |  | Classifier - 'svm', 'logistic', 'ridge', 'lda' or sklearn estimator instance. | <code>'svm'</code>
`cv` |  | Cross-validation strategy. If None and _run_labels exists, uses leave-one-group-out with run labels. | <code>5</code>
`groups` | <code>'np.ndarray \| None'</code> | Group labels for GroupKFold/LeaveOneGroupOut. If None and _run_labels exists, uses stored run labels. | <code>None</code>
`roi_mask` |  | Mask for ROI-based MVPA. Required if method='roi'. | <code>None</code>
`radius` | <code>[float](#float)</code> | Searchlight radius in mm (default 10.0). | <code>10.0</code>
`scoring` | <code>[str](#str)</code> | Scoring metric (default 'accuracy'). | <code>'accuracy'</code>
`standardize` | <code>[bool](#bool)</code> | If True, standardize features before classification. | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Parallel jobs for searchlight (-1 = all cores). | <code>-1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with prediction results:
<code>'BrainCollection'</code> | - For timeseries: (n_timepoints, n_voxels) predicted responses
<code>'BrainCollection'</code> | - For MVPA: (1, n_voxels) accuracy values

**Examples:**

```pycon
>>> # MVPA workflow with run-level betas
>>> betas = bc.fit_glm(events=events, t_r=2.0, by_run=True)
>>> accuracy = betas.predict(y=None, method='whole_brain')
>>> # y and groups inferred from _condition_labels, _run_labels
```

```pycon
>>> # Explicit labels
>>> accuracy = betas.predict(y=labels, method='searchlight')
```

```pycon
>>> # Timeseries prediction with ridge weights
>>> weights = bc.fit_ridge(X=features, output='weights')
>>> predictions = weights.predict(X=new_features)
```

######## `nltools.data.collection.prediction.select_feature`

```python
select_feature(bc, feature: 'int | str') -> 'BrainCollection'
```

Select a single feature's weights across all subjects.

Used after fit_ridge() to extract weights for a specific feature
for group-level analysis (e.g., t-test on feature weights).

Must be called on a BrainCollection created by fit_ridge() where
each subject has shape (n_features, n_voxels).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`feature` | <code>'int \| str'</code> | Feature to select. Can be: - int: Feature index (0-based) - str: Feature name (requires _feature_names attribute) | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection where each BrainData has shape (n_voxels,)
<code>'BrainCollection'</code> | containing the weights for the specified feature.

**Examples:**

```pycon
>>> # Fit ridge and select feature
>>> weights = bc.fit_ridge(X=features, alpha=1.0)
>>> feature_0 = weights.select_feature(0)
>>> # Group t-test on first feature's weights
>>> group_result = feature_0.ttest()
```

```pycon
>>> # By name (if features had column names)
>>> face_weights = weights.select_feature("face_response")
```

###### `nltools.data.collection.transforms`

BrainCollection transform functions.

Extracted from BrainCollection methods — each function takes a BrainCollection
as its first argument (``bc``) instead of ``self``.

**Functions:**

Name | Description
---- | -----------
[`align`](#nltools.data.collection.transforms.align) | Align subjects using local functional alignment.
[`detrend`](#nltools.data.collection.transforms.detrend) | Remove trend from each image.
[`filter_collection`](#nltools.data.collection.transforms.filter_collection) | Filter collection by predicate.
[`map_axis0`](#nltools.data.collection.transforms.map_axis0) | Map function over images (axis=0).
[`map_axis1`](#nltools.data.collection.transforms.map_axis1) | Map function over timepoints (axis=1).
[`map_axis2`](#nltools.data.collection.transforms.map_axis2) | Map function over voxels (axis=2) per image.
[`map_collection`](#nltools.data.collection.transforms.map_collection) | Apply function across specified axis.
[`smooth`](#nltools.data.collection.transforms.smooth) | Spatially smooth each image.
[`standardize`](#nltools.data.collection.transforms.standardize) | Standardize each image.
[`threshold`](#nltools.data.collection.transforms.threshold) | Threshold each image.



####### Classes

####### Functions######## `nltools.data.collection.transforms.align`

```python
align(bc: 'BrainCollection', method: str = 'procrustes', scheme: str = 'searchlight', radius_mm: float = 10.0, parcellation: 'nib.Nifti1Image | None' = None, n_features: int | None = None, n_iter: int = 3, parallel: str | None = 'cpu', n_jobs: int = -1, return_model: bool = False, show_progress: bool = True) -> 'BrainCollection | tuple[BrainCollection, object]'
```

Align subjects using local functional alignment.

Performs neighborhood-based functional alignment across subjects using
LocalAlignment. Each subject's data is aligned to a common template space
using local transforms learned within searchlight spheres or parcels.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to align. | *required*
`method` | <code>[str](#str)</code> | Alignment method. Options: - 'procrustes': Orthogonal Procrustes (default, preserves dimensions) - 'srm': Shared Response Model (dimensionality reduction) - 'hyperalignment': Hyperalignment (iterative Procrustes) | <code>'procrustes'</code>
`scheme` | <code>[str](#str)</code> | Spatial scheme. Options: - 'searchlight': Overlapping spheres with center-only aggregation - 'piecewise': Non-overlapping parcels (requires parcellation) | <code>'searchlight'</code>
`radius_mm` | <code>[float](#float)</code> | Sphere radius in millimeters for searchlight scheme. | <code>10.0</code>
`parcellation` | <code>'nib.Nifti1Image \| None'</code> | Parcellation image for piecewise scheme (required if scheme='piecewise'). | <code>None</code>
`n_features` | <code>[int](#int) \| None</code> | Number of features for SRM. None uses full dimensions. | <code>None</code>
`n_iter` | <code>[int](#int)</code> | Number of iterations for alignment refinement. | <code>3</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization mode. Options: - None: Single-threaded - 'cpu': CPU parallelization with joblib - 'gpu': GPU acceleration via PyTorch | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs for CPU mode (-1 = auto). | <code>-1</code>
`return_model` | <code>[bool](#bool)</code> | If True, return (aligned_collection, model) tuple for fit/transform workflow with new data. | <code>False</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| tuple[BrainCollection, object]'</code> | BrainCollection with aligned data. If return_model=True, returns
<code>'BrainCollection \| tuple[BrainCollection, object]'</code> | tuple of (aligned_collection, LocalAlignment_model).

**Examples:**

```pycon
>>> # Basic searchlight alignment
>>> aligned_bc = bc.align(method='procrustes', radius_mm=10.0)
```

```pycon
>>> # Piecewise alignment with parcellation
>>> aligned_bc = bc.align(
...     scheme='piecewise',
...     parcellation=parcellation_img,
...     method='srm',
...     n_features=50
... )
```

```pycon
>>> # Fit/transform workflow for train/test split
>>> aligned_train, model = train_bc.align(return_model=True)
>>> aligned_test = model.transform(test_data_list)
```

```pycon
>>> # GPU-accelerated alignment
>>> aligned_bc = bc.align(parallel='gpu')
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

Based on Bazeille et al. 2021 "An empirical evaluation of functional
alignment using inter-subject decoding". Center-only aggregation is
used for searchlight to preserve local orthogonality of transforms.

</details>

<details class="see-also" open markdown="1">
<summary>See Also</summary>

nltools.algorithms.alignment.LocalAlignment: Underlying alignment class.

</details>

######## `nltools.data.collection.transforms.detrend`

```python
detrend(bc: 'BrainCollection', method: str = 'linear', n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Remove trend from each image.

Delegates to BrainData.detrend() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to detrend. | *required*
`method` | <code>[str](#str)</code> | 'linear' or 'constant'. | <code>'linear'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with detrended images.

**Examples:**

```pycon
>>> bc.detrend()  # Remove linear trend
>>> bc.detrend(method='constant')  # Remove mean only
```

######## `nltools.data.collection.transforms.filter_collection`

```python
filter_collection(bc: 'BrainCollection', predicate: 'Callable | list | np.ndarray') -> 'BrainCollection'
```

Filter collection by predicate.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to filter. | *required*
`predicate` | <code>'Callable \| list \| np.ndarray'</code> | Filter condition. Can be: - callable: fn(BrainData) -> bool - list/ndarray: Boolean mask of length n_images - pd.Series: Boolean series (index ignored) | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with subset of images matching predicate.

**Examples:**

```pycon
>>> # Filter by callable
>>> bc.filter(lambda bd: bd.data.mean() > 0)
```

```pycon
>>> # Filter by boolean mask
>>> mask = [True, False, True]
>>> bc.filter(mask)
```

```pycon
>>> # Filter by metadata condition
>>> bc.filter(bc.metadata['group'] == 'control')
```

######## `nltools.data.collection.transforms.map_axis0`

```python
map_axis0(bc: 'BrainCollection', fn: Callable, n_jobs: int, show_progress: bool) -> 'BrainCollection'
```

Map function over images (axis=0).

######## `nltools.data.collection.transforms.map_axis1`

```python
map_axis1(bc: 'BrainCollection', fn: Callable, n_jobs: int, show_progress: bool) -> 'BrainCollection'
```

Map function over timepoints (axis=1).

######## `nltools.data.collection.transforms.map_axis2`

```python
map_axis2(bc: 'BrainCollection', fn: Callable, n_jobs: int, show_progress: bool) -> 'BrainCollection'
```

Map function over voxels (axis=2) per image.

######## `nltools.data.collection.transforms.map_collection`

```python
map_collection(bc: 'BrainCollection', fn: Callable, axis: int | str = 0, n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Apply function across specified axis.

This is the general-purpose transformation method. For common operations,
use convenience methods like standardize(), smooth(), etc.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to transform. | *required*
`fn` | <code>[Callable](#collections.abc.Callable)</code> | Function to apply. Signature depends on axis: - axis=0: fn(BrainData) -> BrainData (per image) - axis=1: fn(BrainData) -> BrainData (per timepoint slice) - axis=2: fn(ndarray[n_obs]) -> ndarray (per voxel timeseries) | *required*
`axis` | <code>[int](#int) \| [str](#str)</code> | Axis to iterate over: - 0 or 'images': Apply fn to each image independently - 1 or 'time': Apply fn to each timepoint across images - 2 or 'voxels': Apply fn to each voxel timeseries per image | <code>0</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. -1 for all cores. Default 1. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show tqdm progress bar. Default True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with transformed data.

**Examples:**

```pycon
>>> # Per-image operation
>>> bc.map(lambda bd: bd.standardize())
```

```pycon
>>> # Per-voxel timeseries (e.g., detrend each voxel)
>>> from scipy.signal import detrend
>>> bc.map(detrend, axis=2)
```

```pycon
>>> # Parallel processing
>>> bc.map(expensive_fn, n_jobs=-1)
```

######## `nltools.data.collection.transforms.smooth`

```python
smooth(bc: 'BrainCollection', fwhm: float, n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Spatially smooth each image.

Delegates to BrainData.smooth() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to smooth. | *required*
`fwhm` | <code>[float](#float)</code> | Full width at half maximum of Gaussian kernel in mm. | *required*
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with smoothed images.

**Examples:**

```pycon
>>> bc.smooth(fwhm=6)  # 6mm FWHM smoothing
```

######## `nltools.data.collection.transforms.standardize`

```python
standardize(bc: 'BrainCollection', axis: int = 0, method: str = 'center', n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Standardize each image.

Delegates to BrainData.standardize() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to standardize. | *required*
`axis` | <code>[int](#int)</code> | Axis for standardization within each image: - 0: Standardize across observations (time) per voxel - 1: Standardize across voxels per observation | <code>0</code>
`method` | <code>[str](#str)</code> | 'center' (subtract mean) or 'zscore' (subtract mean, divide std) | <code>'center'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with standardized images.

**Examples:**

```pycon
>>> bc.standardize()  # Center each image across time
>>> bc.standardize(method='zscore')  # Z-score each image
>>> bc.standardize(axis=1)  # Standardize across voxels
```

######## `nltools.data.collection.transforms.threshold`

```python
threshold(bc: 'BrainCollection', upper: float | str | None = None, lower: float | str | None = None, binarize: bool = False, coerce_nan: bool = True, n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Threshold each image.

Delegates to BrainData.threshold() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to threshold. | *required*
`upper` | <code>[float](#float) \| [str](#str) \| None</code> | Upper cutoff. String interpreted as percentile. | <code>None</code>
`lower` | <code>[float](#float) \| [str](#str) \| None</code> | Lower cutoff. String interpreted as percentile. | <code>None</code>
`binarize` | <code>[bool](#bool)</code> | Return binary mask. | <code>False</code>
`coerce_nan` | <code>[bool](#bool)</code> | Replace NaN with 0. | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with thresholded images.

**Examples:**

```pycon
>>> bc.threshold(lower=0)  # Zero out negative values
>>> bc.threshold(upper='95%')  # Keep top 5%
>>> bc.threshold(lower=2, binarize=True)  # Binary mask
```

#### `nltools.data.designmatrix`

DesignMatrix - Polars-based design matrix for neuroimaging analysis

Efficient design matrix implementation using Polars for fast DataFrame operations.
Provides HRF convolution, resampling, polynomial regressors, and diagnostic tools.

Uses composition pattern (wrapping pl.DataFrame) for clean metadata preservation.

**Modules:**

Name | Description
---- | -----------
[`append`](#nltools.data.designmatrix.append) | Standalone functions for DesignMatrix concatenation operations.
[`diagnostics`](#nltools.data.designmatrix.diagnostics) | Diagnostic and utility functions for DesignMatrix.
[`io`](#nltools.data.designmatrix.io) | DesignMatrix I/O and visualization functions.
[`regressors`](#nltools.data.designmatrix.regressors) | Standalone regressor functions for DesignMatrix.
[`transforms`](#nltools.data.designmatrix.transforms) | Standalone transform functions for DesignMatrix.
[`utils`](#nltools.data.designmatrix.utils) | Shared helpers for DesignMatrix submodules.

**Classes:**

Name | Description
---- | -----------
[`DesignMatrix`](#nltools.data.designmatrix.DesignMatrix) | Polars-based design matrix for experimental designs in neuroimaging.



##### Classes###### `nltools.data.designmatrix.DesignMatrix`

```python
DesignMatrix(data: Union[pl.DataFrame, pd.DataFrame, np.ndarray, dict, None] = None, *, sampling_freq: Optional[float] = None, columns: Optional[List[str]] = None, convolved: Optional[List[str]] = None, polys: Optional[List[str]] = None)
```

Polars-based design matrix for experimental designs in neuroimaging.

Wraps a Polars DataFrame with neuroimaging-specific metadata and methods.
Uses composition pattern (not subclassing) for clean metadata preservation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>DataFrame, ndarray, dict, or None</code> | Input data. Accepts: - Polars DataFrame (zero-copy) - pandas DataFrame (converted to Polars) - numpy ndarray - dict (keys=columns, values=data) - None (empty initialization) | <code>None</code>
`sampling_freq` | <code>[float](#float)</code> | Sampling frequency in Hz (1/TR for fMRI data) | <code>None</code>
`columns` | <code>list of str</code> | Column names (used with ndarray input) | <code>None</code>
`convolved` | <code>list of str</code> | Names of convolved columns (tracked internally) | <code>None</code>
`polys` | <code>list of str</code> | Names of polynomial columns (tracked internally) | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`sampling_freq`](#nltools.data.designmatrix.DesignMatrix.sampling_freq) | <code>[float](#float) or None</code> | Sampling frequency in Hz
[`convolved`](#nltools.data.designmatrix.DesignMatrix.convolved) | <code>list of str</code> | Columns that have been convolved
[`polys`](#nltools.data.designmatrix.DesignMatrix.polys) | <code>list of str</code> | Polynomial/nuisance columns (intercept, trends, DCT bases)
[`multi`](#nltools.data.designmatrix.DesignMatrix.multi) | <code>[bool](#bool)</code> | True if created from multi-run concatenation

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
>>> # Convolve with HRF
>>> dm_conv = dm.convolve()
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

**Functions:**

Name | Description
---- | -----------
[`add_dct_basis`](#nltools.data.designmatrix.DesignMatrix.add_dct_basis) | Add discrete cosine transform basis functions (high-pass filter).
[`add_poly`](#nltools.data.designmatrix.DesignMatrix.add_poly) | Add Legendre polynomial drift terms.
[`append`](#nltools.data.designmatrix.DesignMatrix.append) | Concatenate design matrices.
[`clean`](#nltools.data.designmatrix.DesignMatrix.clean) | Remove highly correlated columns.
[`convolve`](#nltools.data.designmatrix.DesignMatrix.convolve) | Convolve columns with HRF or custom kernel.
[`copy`](#nltools.data.designmatrix.DesignMatrix.copy) | Create a deep copy of the DesignMatrix.
[`details`](#nltools.data.designmatrix.DesignMatrix.details) | Return human-readable metadata summary.
[`downsample`](#nltools.data.designmatrix.DesignMatrix.downsample) | Reduce temporal resolution to target frequency using Polars-native operations.
[`drop`](#nltools.data.designmatrix.DesignMatrix.drop) | Drop specified columns.
[`fillna`](#nltools.data.designmatrix.DesignMatrix.fillna) | Fill NaN/null values with specified value.
[`heatmap`](#nltools.data.designmatrix.DesignMatrix.heatmap) | Visualize design matrix as heatmap (SPM-style).
[`replace_data`](#nltools.data.designmatrix.DesignMatrix.replace_data) | Replace data columns while preserving polynomials and metadata.
[`reset_index`](#nltools.data.designmatrix.DesignMatrix.reset_index) | Reset index (pandas compatibility method).
[`standardize`](#nltools.data.designmatrix.DesignMatrix.standardize) | Standardize columns using the specified method.
[`sum`](#nltools.data.designmatrix.DesignMatrix.sum) | Compute sum along axis.
[`to_numpy`](#nltools.data.designmatrix.DesignMatrix.to_numpy) | Convert DesignMatrix to numpy array.
[`to_pandas`](#nltools.data.designmatrix.DesignMatrix.to_pandas) | Convert DesignMatrix to pandas DataFrame.
[`upsample`](#nltools.data.designmatrix.DesignMatrix.upsample) | Increase temporal resolution to target frequency.
[`vif`](#nltools.data.designmatrix.DesignMatrix.vif) | Compute variance inflation factor for each column.
[`write`](#nltools.data.designmatrix.DesignMatrix.write) | Write DesignMatrix to file.
[`zscore`](#nltools.data.designmatrix.DesignMatrix.zscore) | Z-score standardize columns (mean=0, std=1).



####### Attributes######## `nltools.data.designmatrix.DesignMatrix.columns`

```python
columns: List[str]
```

Column names of the design matrix as a list of strings.

######## `nltools.data.designmatrix.DesignMatrix.convolved`

```python
convolved = convolved if convolved is not None else []
```

######## `nltools.data.designmatrix.DesignMatrix.is_empty`

```python
is_empty: bool
```

Check if DesignMatrix has no data.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`bool` | <code>[bool](#bool)</code> | True if the design matrix is empty, False otherwise.

######## `nltools.data.designmatrix.DesignMatrix.multi`

```python
multi = False
```

######## `nltools.data.designmatrix.DesignMatrix.polys`

```python
polys = polys if polys is not None else []
```

######## `nltools.data.designmatrix.DesignMatrix.sampling_freq`

```python
sampling_freq = sampling_freq
```

######## `nltools.data.designmatrix.DesignMatrix.shape`

```python
shape: tuple
```

Return (n_rows, n_cols) tuple.



####### Functions######## `nltools.data.designmatrix.DesignMatrix.add_dct_basis`

```python
add_dct_basis(duration: float = 180, drop: int = 0) -> DesignMatrix
```

Add discrete cosine transform basis functions (high-pass filter).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`duration` | <code>[float](#float)</code> | Filter duration in seconds. Default: 180. | <code>180</code>
`drop` | <code>[int](#int)</code> | Number of low-frequency bases to drop. Default: 0. | <code>0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with DCT basis columns appended.

######## `nltools.data.designmatrix.DesignMatrix.add_poly`

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

######## `nltools.data.designmatrix.DesignMatrix.append`

```python
append(dm: Union[DesignMatrix, List[DesignMatrix]], axis: int = 0, keep_separate: bool = True, unique_cols: Optional[List[str]] = None, fill_na: Union[int, float] = 0, verbose: bool = False) -> DesignMatrix
```

Concatenate design matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>DesignMatrix or list of DesignMatrix</code> | Design matrix/matrices to append. | *required*
`axis` | <code>[int](#int)</code> | 0 for row-wise (vertical), 1 for column-wise (horizontal). Default: 0. | <code>0</code>
`keep_separate` | <code>[bool](#bool)</code> | Whether to separate polynomial columns across runs (only applies when axis=0). Default: True. | <code>True</code>
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | <code>None</code>
`fill_na` | <code>[int](#int) or [float](#float)</code> | Value to fill NaN values during vertical concatenation. Default: 0. | <code>0</code>
`verbose` | <code>[bool](#bool)</code> | Print messages about polynomial separation. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Concatenated design matrix.

######## `nltools.data.designmatrix.DesignMatrix.clean`

```python
clean(fill_na: Union[int, float, None] = 0, exclude_polys: bool = False, thresh: float = 0.95, verbose: bool = True) -> DesignMatrix
```

Remove highly correlated columns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fill_na` | <code>int, float, or None</code> | Fill NaN values before checking correlations (default 0) | <code>0</code>
`exclude_polys` | <code>[bool](#bool)</code> | Skip polynomial columns from correlation check | <code>False</code>
`thresh` | <code>[float](#float)</code> | Correlation threshold (drop if abs(r) >= thresh, default 0.95) | <code>0.95</code>
`verbose` | <code>[bool](#bool)</code> | Print dropped column names | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Cleaned matrix with highly correlated columns removed

######## `nltools.data.designmatrix.DesignMatrix.convolve`

```python
convolve(conv_func: Union[str, np.ndarray] = 'hrf', columns: Optional[List[str]] = None) -> DesignMatrix
```

Convolve columns with HRF or custom kernel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`conv_func` | <code>[str](#str) or [ndarray](#ndarray)</code> | 'hrf' for canonical Glover HRF, or custom kernel(s). Can be 1D array (single kernel) or 2D (samples x kernels) | <code>'hrf'</code>
`columns` | <code>list of str</code> | Columns to convolve (default: all non-polynomial columns) | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with convolved columns

######## `nltools.data.designmatrix.DesignMatrix.copy`

```python
copy() -> DesignMatrix
```

Create a deep copy of the DesignMatrix.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Copy of the current DesignMatrix

######## `nltools.data.designmatrix.DesignMatrix.details`

```python
details() -> str
```

Return human-readable metadata summary.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` | <code>[str](#str)</code> | Formatted string showing sampling_freq, shape, convolved columns, and polynomial columns

######## `nltools.data.designmatrix.DesignMatrix.downsample`

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

######## `nltools.data.designmatrix.DesignMatrix.drop`

```python
drop(columns: List[str]) -> DesignMatrix
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

######## `nltools.data.designmatrix.DesignMatrix.fillna`

```python
fillna(value: Union[int, float]) -> DesignMatrix
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

######## `nltools.data.designmatrix.DesignMatrix.heatmap`

```python
heatmap(figsize: tuple = (8, 6), **kwargs: tuple)
```

Visualize design matrix as heatmap (SPM-style).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`figsize` | <code>tuple, default=(8, 6)</code> | Figure size (width, height) in inches | <code>(8, 6)</code>
`**kwargs` |  | Additional keyword arguments passed to seaborn.heatmap() | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.axes.Axes: The axes object containing the heatmap

######## `nltools.data.designmatrix.DesignMatrix.replace_data`

```python
replace_data(data: np.ndarray, column_names: Optional[List[str]] = None) -> DesignMatrix
```

Replace data columns while preserving polynomials and metadata.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#ndarray)</code> | New data array (must match number of rows in current DesignMatrix) | *required*
`column_names` | <code>list of str</code> | Names for new data columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with replaced data columns, preserved polynomials

######## `nltools.data.designmatrix.DesignMatrix.reset_index`

```python
reset_index(drop: bool = True) -> DesignMatrix
```

Reset index (pandas compatibility method).

Polars DataFrames don't have row indexes, so this is a no-op.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`drop` | <code>bool, default=True</code> | Ignored. Kept for API compatibility. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Returns self unchanged

######## `nltools.data.designmatrix.DesignMatrix.standardize`

```python
standardize(method: str = 'zscore', columns: Optional[List[str]] = None) -> DesignMatrix
```

Standardize columns using the specified method.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Standardization method ('zscore' or 'center'). Default: 'zscore'. | <code>'zscore'</code>
`columns` | <code>[Optional](#typing.Optional)[[List](#typing.List)[[str](#str)]]</code> | Columns to standardize. If None, standardize all non-polynomial columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with standardized columns.

######## `nltools.data.designmatrix.DesignMatrix.sum`

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

######## `nltools.data.designmatrix.DesignMatrix.to_numpy`

```python
to_numpy() -> np.ndarray
```

Convert DesignMatrix to numpy array.

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | np.ndarray: 2D array with shape (n_samples, n_columns)

######## `nltools.data.designmatrix.DesignMatrix.to_pandas`

```python
to_pandas() -> pd.DataFrame
```

Convert DesignMatrix to pandas DataFrame.

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#pandas.DataFrame)</code> | pd.DataFrame: Pandas DataFrame with same data and column names.

######## `nltools.data.designmatrix.DesignMatrix.upsample`

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

######## `nltools.data.designmatrix.DesignMatrix.vif`

```python
vif(exclude_polys: bool = True) -> np.ndarray | None
```

Compute variance inflation factor for each column.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`exclude_polys` | <code>[bool](#bool)</code> | Skip polynomial columns. Default: True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray) \| None</code> | np.ndarray: VIF values for each included column. Returns None if the correlation matrix is singular.

######## `nltools.data.designmatrix.DesignMatrix.write`

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

######## `nltools.data.designmatrix.DesignMatrix.zscore`

```python
zscore(columns: Optional[List[str]] = None) -> DesignMatrix
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



##### Functions

##### Modules###### `nltools.data.designmatrix.append`

Standalone functions for DesignMatrix concatenation operations.

These functions implement the append/concatenation logic extracted from
DesignMatrix methods, following the "functional core" pattern.

**Functions:**

Name | Description
---- | -----------
[`append`](#nltools.data.designmatrix.append.append) | Concatenate design matrices.
[`append_horizontal`](#nltools.data.designmatrix.append.append_horizontal) | Horizontal concatenation (axis=1) - add columns from other matrices.
[`append_vertical`](#nltools.data.designmatrix.append.append_vertical) | Vertical concatenation (axis=0) - stack rows, with optional polynomial separation.
[`append_vertical_with_separation`](#nltools.data.designmatrix.append.append_vertical_with_separation) | Vertical concatenation with automatic polynomial separation.
[`get_starting_run_idx`](#nltools.data.designmatrix.append.get_starting_run_idx) | Determine next run index for multi-run appending.
[`identify_columns_to_separate`](#nltools.data.designmatrix.append.identify_columns_to_separate) | Identify which columns need run-specific separation.
[`match_column_pattern`](#nltools.data.designmatrix.append.match_column_pattern) | Match columns against pattern with wildcard support.



####### Classes

####### Functions######## `nltools.data.designmatrix.append.append`

```python
append(dm: DesignMatrix, other: Union[DesignMatrix, List[DesignMatrix]], axis: int = 0, keep_separate: bool = True, unique_cols: Optional[List[str]] = None, fill_na: Union[int, float] = 0, verbose: bool = False) -> DesignMatrix
```

Concatenate design matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | The base design matrix. | *required*
`other` | <code>DesignMatrix or list of DesignMatrix</code> | Design matrix/matrices to append. | *required*
`axis` | <code>[int](#int)</code> | 0 for row-wise (vertical), 1 for column-wise (horizontal). | <code>0</code>
`keep_separate` | <code>[bool](#bool)</code> | Whether to separate polynomial columns across runs (only axis=0). | <code>True</code>
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | <code>None</code>
`fill_na` | <code>[int](#int) or [float](#float)</code> | Value to fill NaN values during vertical concatenation. Default: 0. | <code>0</code>
`verbose` | <code>[bool](#bool)</code> | Print messages about polynomial separation. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Concatenated design matrix.

######## `nltools.data.designmatrix.append.append_horizontal`

```python
append_horizontal(dm: DesignMatrix, to_append: List[DesignMatrix], fill_na: Union[int, float]) -> DesignMatrix
```

Horizontal concatenation (axis=1) - add columns from other matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Base DesignMatrix instance. | *required*
`to_append` | <code>list of DesignMatrix</code> | Matrices whose columns to add. | *required*
`fill_na` | <code>[int](#int) or [float](#float)</code> | Value to fill NaN/null entries with. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with columns from all matrices.

######## `nltools.data.designmatrix.append.append_vertical`

```python
append_vertical(dm: DesignMatrix, to_append: List[DesignMatrix], keep_separate: bool, unique_cols: Optional[List[str]], fill_na: Union[int, float], verbose: bool) -> DesignMatrix
```

Vertical concatenation (axis=0) - stack rows, with optional polynomial separation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Base DesignMatrix instance. | *required*
`to_append` | <code>list of DesignMatrix</code> | Matrices to stack below dm. | *required*
`keep_separate` | <code>[bool](#bool)</code> | Whether to separate polynomial columns across runs. | *required*
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | *required*
`fill_na` | <code>[int](#int) or [float](#float)</code> | Value to fill NaN/null entries with. | *required*
`verbose` | <code>[bool](#bool)</code> | Print messages about polynomial separation. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with rows from all matrices.

######## `nltools.data.designmatrix.append.append_vertical_with_separation`

```python
append_vertical_with_separation(dm: DesignMatrix, to_append: List[DesignMatrix], unique_cols: Optional[List[str]], fill_na: Union[int, float], verbose: bool) -> DesignMatrix
```

Vertical concatenation with automatic polynomial separation.

Creates run-specific columns (e.g., 0_poly_0, 1_poly_0) that are
active only in their respective runs (sparse representation).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Base DesignMatrix instance. | *required*
`to_append` | <code>list of DesignMatrix</code> | Matrices to stack below dm. | *required*
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | *required*
`fill_na` | <code>[int](#int) or [float](#float)</code> | Value to fill NaN/null entries with. | *required*
`verbose` | <code>[bool](#bool)</code> | Print messages about polynomial separation. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Concatenated DesignMatrix with run-separated polynomial columns and multi=True.

######## `nltools.data.designmatrix.append.get_starting_run_idx`

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

######## `nltools.data.designmatrix.append.identify_columns_to_separate`

```python
identify_columns_to_separate(dm: DesignMatrix, all_dms: List[DesignMatrix], unique_cols: Optional[List[str]]) -> set
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

######## `nltools.data.designmatrix.append.match_column_pattern`

```python
match_column_pattern(columns: List[str], pattern: str) -> List[str]
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
<code>[List](#typing.List)[[str](#str)]</code> | list of str: Column names matching the pattern.

###### `nltools.data.designmatrix.diagnostics`

Diagnostic and utility functions for DesignMatrix.

**Functions:**

Name | Description
---- | -----------
[`clean`](#nltools.data.designmatrix.diagnostics.clean) | Remove highly correlated columns.
[`details`](#nltools.data.designmatrix.diagnostics.details) | Return human-readable metadata summary.
[`vif`](#nltools.data.designmatrix.diagnostics.vif) | Compute variance inflation factor for each column.



####### Classes

####### Functions######## `nltools.data.designmatrix.diagnostics.clean`

```python
clean(dm: DesignMatrix, fill_na: Union[int, float, None] = 0, exclude_polys: bool = False, thresh: float = 0.95, verbose: bool = True) -> DesignMatrix
```

Remove highly correlated columns.

Removes columns with correlation >= threshold. Keeps first instance
of correlated pair, drops duplicates.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`fill_na` | <code>int, float, or None</code> | Fill NaN values before checking correlations. Default: 0. | <code>0</code>
`exclude_polys` | <code>[bool](#bool)</code> | Skip polynomial columns from correlation check. Default: False. | <code>False</code>
`thresh` | <code>[float](#float)</code> | Correlation threshold (drop if abs(r) >= thresh). Default: 0.95. | <code>0.95</code>
`verbose` | <code>[bool](#bool)</code> | Print dropped column names. Default: True. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Cleaned matrix with highly correlated columns removed

######## `nltools.data.designmatrix.diagnostics.details`

```python
details(dm: DesignMatrix) -> str
```

Return human-readable metadata summary.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` | <code>[str](#str)</code> | Formatted string showing sampling_freq, shape, convolved columns, and polynomial columns.

######## `nltools.data.designmatrix.diagnostics.vif`

```python
vif(dm: DesignMatrix, exclude_polys: bool = True) -> np.ndarray | None
```

Compute variance inflation factor for each column.

Uses diagonal elements of inverted correlation matrix
(same method as Matlab and R).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`exclude_polys` | <code>[bool](#bool)</code> | Skip polynomial columns. Default: True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray) \| None</code> | np.ndarray: VIF values for each included column. Returns None if the correlation matrix is singular (perfect collinearity detected).

###### `nltools.data.designmatrix.io`

DesignMatrix I/O and visualization functions.

Standalone functions extracted from DesignMatrix methods.
Each takes a DesignMatrix instance (`dm`) as its first argument.

**Functions:**

Name | Description
---- | -----------
[`heatmap`](#nltools.data.designmatrix.io.heatmap) | Visualize design matrix as heatmap (SPM-style).
[`to_numpy`](#nltools.data.designmatrix.io.to_numpy) | Convert DesignMatrix to numpy array.
[`to_pandas`](#nltools.data.designmatrix.io.to_pandas) | Convert DesignMatrix to pandas DataFrame.
[`write`](#nltools.data.designmatrix.io.write) | Write DesignMatrix to file.
[`write_h5`](#nltools.data.designmatrix.io.write_h5) | Write DesignMatrix to HDF5 file with metadata.



####### Classes

####### Functions######## `nltools.data.designmatrix.io.heatmap`

```python
heatmap(dm: DesignMatrix, figsize: tuple = (8, 6), **kwargs: tuple)
```

Visualize design matrix as heatmap (SPM-style).

Creates a heatmap visualization of the design matrix columns.
Uses seaborn + matplotlib under the hood.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`figsize` | <code>[tuple](#tuple)</code> | Figure size (width, height) in inches. Default: (8, 6). | <code>(8, 6)</code>
`**kwargs` |  | Additional keyword arguments passed to seaborn.heatmap(). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.axes.Axes: The axes object containing the heatmap.

**Examples:**

```pycon
>>> dm = DesignMatrix(np.random.randn(100, 3), columns=['a', 'b', 'c'])
>>> heatmap(dm)
```

######## `nltools.data.designmatrix.io.to_numpy`

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

######## `nltools.data.designmatrix.io.to_pandas`

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

######## `nltools.data.designmatrix.io.write`

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
HDF5 format preserves metadata (sampling_freq, convolved, polys).

</details>

######## `nltools.data.designmatrix.io.write_h5`

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

###### `nltools.data.designmatrix.regressors`

Standalone regressor functions for DesignMatrix.

Each function takes a DesignMatrix as its first argument (`dm`) and returns
a new DesignMatrix with the requested transformation applied.

**Functions:**

Name | Description
---- | -----------
[`add_dct_basis`](#nltools.data.designmatrix.regressors.add_dct_basis) | Add discrete cosine transform basis functions (high-pass filter).
[`add_poly`](#nltools.data.designmatrix.regressors.add_poly) | Add Legendre polynomial drift terms.
[`convolve`](#nltools.data.designmatrix.regressors.convolve) | Convolve columns with HRF or custom kernel.



####### Classes

####### Functions######## `nltools.data.designmatrix.regressors.add_dct_basis`

```python
add_dct_basis(dm: DesignMatrix, duration: float = 180, drop: int = 0) -> DesignMatrix
```

Add discrete cosine transform basis functions (high-pass filter).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix to add DCT basis to. | *required*
`duration` | <code>[float](#float)</code> | Filter duration in seconds. Default: 180. | <code>180</code>
`drop` | <code>[int](#int)</code> | Number of low-frequency bases to drop. Default: 0. | <code>0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with DCT basis columns appended.

######## `nltools.data.designmatrix.regressors.add_poly`

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

######## `nltools.data.designmatrix.regressors.convolve`

```python
convolve(dm: DesignMatrix, conv_func: Union[str, np.ndarray] = 'hrf', columns: Optional[List[str]] = None) -> DesignMatrix
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
>>> # Default HRF convolution
>>> dm_conv = convolve(dm)
```

```pycon
>>> # Custom kernel
>>> kernel = np.array([0.5, 1.0, 0.5])
>>> dm_conv = convolve(dm, conv_func=kernel)
```

```pycon
>>> # Multiple kernels (FIR model)
>>> kernels = np.array([[1.0, 0.5], [0.5, 1.0]]).T  # 2 kernels
>>> dm_conv = convolve(dm, conv_func=kernels)  # Creates col_c0, col_c1
```

###### `nltools.data.designmatrix.transforms`

Standalone transform functions for DesignMatrix.

Each function takes a DesignMatrix instance as the first argument (`dm`)
and returns a new DesignMatrix via `copy_with(dm,...)`.

**Functions:**

Name | Description
---- | -----------
[`downsample`](#nltools.data.designmatrix.transforms.downsample) | Reduce temporal resolution to target frequency using Polars-native operations.
[`standardize`](#nltools.data.designmatrix.transforms.standardize) | Standardize columns using the specified method.
[`upsample`](#nltools.data.designmatrix.transforms.upsample) | Increase temporal resolution to target frequency using Polars-native interpolation.
[`zscore`](#nltools.data.designmatrix.transforms.zscore) | Z-score standardize columns (mean=0, std=1).



####### Classes

####### Functions######## `nltools.data.designmatrix.transforms.downsample`

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

######## `nltools.data.designmatrix.transforms.standardize`

```python
standardize(dm: DesignMatrix, columns: Optional[List[str]] = None, method: str = 'zscore') -> DesignMatrix
```

Standardize columns using the specified method.

This method provides a consistent API with BrainData and Collection
for data normalization.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance to transform. | *required*
`columns` | <code>[Optional](#typing.Optional)[[List](#typing.List)[[str](#str)]]</code> | Columns to standardize. If None, standardize all non-polynomial columns. | <code>None</code>
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

######## `nltools.data.designmatrix.transforms.upsample`

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

######## `nltools.data.designmatrix.transforms.zscore`

```python
zscore(dm: DesignMatrix, columns: Optional[List[str]] = None) -> DesignMatrix
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

###### `nltools.data.designmatrix.utils`

Shared helpers for DesignMatrix submodules.

These are internal utilities used by the facade and submodules — not part of the
public API.

**Functions:**

Name | Description
---- | -----------
[`copy_with`](#nltools.data.designmatrix.utils.copy_with) | Create new DesignMatrix with updated data/metadata.
[`get_data_columns`](#nltools.data.designmatrix.utils.get_data_columns) | Get column names, optionally excluding polynomials.
[`get_metadata`](#nltools.data.designmatrix.utils.get_metadata) | Extract metadata as dict (for copying).



####### Classes

####### Functions######## `nltools.data.designmatrix.utils.copy_with`

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

######## `nltools.data.designmatrix.utils.get_data_columns`

```python
get_data_columns(dm: DesignMatrix, exclude_polys: bool = True) -> List[str]
```

Get column names, optionally excluding polynomials.

This helper reduces code duplication across methods that need to
distinguish between data columns and polynomial/nuisance columns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`exclude_polys` | <code>bool, default=True</code> | If True, exclude polynomial/nuisance columns from the result. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[List](#typing.List)[[str](#str)]</code> | list of str: Column names (excluding polys if requested).

######## `nltools.data.designmatrix.utils.get_metadata`

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
`dict` | <code>[dict](#dict)</code> | Dictionary with keys 'sampling_freq', 'convolved', 'polys', 'multi'.

#### `nltools.data.fitresults`

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
>>> # Get as dict for pandas DataFrame
>>> import pandas as pd
>>> results_dict = fit.asdict()
>>> # Convert to DataFrame (for scalar and 1D arrays)
>>> df = pd.DataFrame({k: v for k, v in results_dict.items() if v.ndim <= 1})

**Classes:**

Name | Description
---- | -----------
[`Fit`](#nltools.data.fitresults.Fit) | Immutable container for model fitting results.



##### Classes###### `nltools.data.fitresults.Fit`

```python
Fit(fitted_values: np.ndarray, weights: Optional[np.ndarray] = None, scores: Optional[np.ndarray] = None, betas: Optional[np.ndarray] = None, t_stats: Optional[np.ndarray] = None, p_values: Optional[np.ndarray] = None, se: Optional[np.ndarray] = None, residuals: Optional[np.ndarray] = None, r2: Optional[np.ndarray] = None, cv_scores: Optional[np.ndarray] = None, cv_mean_score: Optional[np.ndarray] = None, cv_predictions: Optional[np.ndarray] = None, cv_folds: Optional[np.ndarray] = None, cv_best_alpha: Optional[float] = None, cv_alpha_scores: Optional[np.ndarray] = None) -> None
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

**Functions:**

Name | Description
---- | -----------
[`asdict`](#nltools.data.fitresults.Fit.asdict) | Convert to dictionary.
[`available`](#nltools.data.fitresults.Fit.available) | Return list of non-None attribute names.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`betas`](#nltools.data.fitresults.Fit.betas) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`cv_alpha_scores`](#nltools.data.fitresults.Fit.cv_alpha_scores) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`cv_best_alpha`](#nltools.data.fitresults.Fit.cv_best_alpha) | <code>[Optional](#typing.Optional)[[float](#float)]</code> | 
[`cv_folds`](#nltools.data.fitresults.Fit.cv_folds) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`cv_mean_score`](#nltools.data.fitresults.Fit.cv_mean_score) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`cv_predictions`](#nltools.data.fitresults.Fit.cv_predictions) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`cv_scores`](#nltools.data.fitresults.Fit.cv_scores) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`fitted_values`](#nltools.data.fitresults.Fit.fitted_values) | <code>[ndarray](#numpy.ndarray)</code> | 
[`p_values`](#nltools.data.fitresults.Fit.p_values) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`r2`](#nltools.data.fitresults.Fit.r2) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`residuals`](#nltools.data.fitresults.Fit.residuals) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`scores`](#nltools.data.fitresults.Fit.scores) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`se`](#nltools.data.fitresults.Fit.se) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`t_stats`](#nltools.data.fitresults.Fit.t_stats) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 
[`weights`](#nltools.data.fitresults.Fit.weights) | <code>[Optional](#typing.Optional)[[ndarray](#numpy.ndarray)]</code> | 



####### Attributes######## `nltools.data.fitresults.Fit.betas`

```python
betas: Optional[np.ndarray] = None
```

######## `nltools.data.fitresults.Fit.cv_alpha_scores`

```python
cv_alpha_scores: Optional[np.ndarray] = None
```

######## `nltools.data.fitresults.Fit.cv_best_alpha`

```python
cv_best_alpha: Optional[float] = None
```

######## `nltools.data.fitresults.Fit.cv_folds`

```python
cv_folds: Optional[np.ndarray] = None
```

######## `nltools.data.fitresults.Fit.cv_mean_score`

```python
cv_mean_score: Optional[np.ndarray] = None
```

######## `nltools.data.fitresults.Fit.cv_predictions`

```python
cv_predictions: Optional[np.ndarray] = None
```

######## `nltools.data.fitresults.Fit.cv_scores`

```python
cv_scores: Optional[np.ndarray] = None
```

######## `nltools.data.fitresults.Fit.fitted_values`

```python
fitted_values: np.ndarray
```

######## `nltools.data.fitresults.Fit.p_values`

```python
p_values: Optional[np.ndarray] = None
```

######## `nltools.data.fitresults.Fit.r2`

```python
r2: Optional[np.ndarray] = None
```

######## `nltools.data.fitresults.Fit.residuals`

```python
residuals: Optional[np.ndarray] = None
```

######## `nltools.data.fitresults.Fit.scores`

```python
scores: Optional[np.ndarray] = None
```

######## `nltools.data.fitresults.Fit.se`

```python
se: Optional[np.ndarray] = None
```

######## `nltools.data.fitresults.Fit.t_stats`

```python
t_stats: Optional[np.ndarray] = None
```

######## `nltools.data.fitresults.Fit.weights`

```python
weights: Optional[np.ndarray] = None
```



####### Functions######## `nltools.data.fitresults.Fit.asdict`

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

######## `nltools.data.fitresults.Fit.available`

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

#### `nltools.data.roc`

NeuroLearn Analysis Tools
=========================
These tools provide the ability to quickly run
machine-learning analyses on imaging data

**Classes:**

Name | Description
---- | -----------
[`Roc`](#nltools.data.roc.Roc) | Roc Class



##### Classes###### `nltools.data.roc.Roc`

```python
Roc(input_values = None, binary_outcome = None, threshold_type = 'optimal_overall', forced_choice = None, **kwargs)
```

Bases: <code>[object](#object)</code>

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

**Functions:**

Name | Description
---- | -----------
[`calculate`](#nltools.data.roc.Roc.calculate) | Calculate Receiver Operating Characteristic plot (ROC) for
[`plot`](#nltools.data.roc.Roc.plot) | Create ROC Plot
[`summary`](#nltools.data.roc.Roc.summary) | Display a formatted summary of ROC analysis.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`binary_outcome`](#nltools.data.roc.Roc.binary_outcome) |  | 
[`forced_choice`](#nltools.data.roc.Roc.forced_choice) |  | 
[`input_values`](#nltools.data.roc.Roc.input_values) |  | 
[`threshold_type`](#nltools.data.roc.Roc.threshold_type) |  | 



####### Attributes######## `nltools.data.roc.Roc.binary_outcome`

```python
binary_outcome = deepcopy(binary_outcome)
```

######## `nltools.data.roc.Roc.forced_choice`

```python
forced_choice = deepcopy(forced_choice)
```

######## `nltools.data.roc.Roc.input_values`

```python
input_values = deepcopy(input_values)
```

######## `nltools.data.roc.Roc.threshold_type`

```python
threshold_type = deepcopy(threshold_type)
```



####### Functions######## `nltools.data.roc.Roc.calculate`

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

######## `nltools.data.roc.Roc.plot`

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

######## `nltools.data.roc.Roc.summary`

```python
summary()
```

Display a formatted summary of ROC analysis.



##### Functions#### `nltools.data.simulator`

NeuroLearn Simulator Tools
==========================

Tools to simulate multivariate data.

**Classes:**

Name | Description
---- | -----------
[`SimulateGrid`](#nltools.data.simulator.SimulateGrid) | Simulate 2D grid data for testing statistical methods.
[`Simulator`](#nltools.data.simulator.Simulator) | Simulate fMRI data with realistic spatial and temporal characteristics.



##### Attributes

##### Classes###### `nltools.data.simulator.SimulateGrid`

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
[`data`](#nltools.data.simulator.SimulateGrid.data) |  | The simulated data array of shape (n_subjects, grid_width, grid_width).
[`t_values`](#nltools.data.simulator.SimulateGrid.t_values) |  | T-statistic values after fitting.
[`p_values`](#nltools.data.simulator.SimulateGrid.p_values) |  | P-values after fitting.
[`thresholded`](#nltools.data.simulator.SimulateGrid.thresholded) |  | Thresholded statistical map.
[`isfit`](#nltools.data.simulator.SimulateGrid.isfit) |  | Whether fit() has been called.

**Examples:**

```pycon
>>> from nltools.data.simulator import SimulateGrid
>>> sim = SimulateGrid(signal_amplitude=0.5, random_state=42)
>>> sim.fit(n_permute=1000)
>>> sim.plot()
```

**Functions:**

Name | Description
---- | -----------
[`add_signal`](#nltools.data.simulator.SimulateGrid.add_signal) | Add rectangular signal to self.data
[`create_mask`](#nltools.data.simulator.SimulateGrid.create_mask) | Create a mask for where the signal is located in grid.
[`fit`](#nltools.data.simulator.SimulateGrid.fit) | Run ttest on self.data
[`plot_grid_simulation`](#nltools.data.simulator.SimulateGrid.plot_grid_simulation) | Create a plot of the simulations
[`run_multiple_simulations`](#nltools.data.simulator.SimulateGrid.run_multiple_simulations) | This method will run multiple simulations to calculate overall false positive rate
[`threshold_simulation`](#nltools.data.simulator.SimulateGrid.threshold_simulation) | Threshold simulation



####### Attributes######## `nltools.data.simulator.SimulateGrid.correction`

```python
correction = None
```

######## `nltools.data.simulator.SimulateGrid.data`

```python
data = self._create_noise()
```

######## `nltools.data.simulator.SimulateGrid.grid_width`

```python
grid_width = grid_width
```

######## `nltools.data.simulator.SimulateGrid.isfit`

```python
isfit = False
```

######## `nltools.data.simulator.SimulateGrid.n_subjects`

```python
n_subjects = n_subjects
```

######## `nltools.data.simulator.SimulateGrid.p_values`

```python
p_values = None
```

######## `nltools.data.simulator.SimulateGrid.random_state`

```python
random_state = check_random_state(random_state)
```

######## `nltools.data.simulator.SimulateGrid.sigma`

```python
sigma = sigma
```

######## `nltools.data.simulator.SimulateGrid.signal_amplitude`

```python
signal_amplitude = None
```

######## `nltools.data.simulator.SimulateGrid.signal_mask`

```python
signal_mask = None
```

######## `nltools.data.simulator.SimulateGrid.t_values`

```python
t_values = None
```

######## `nltools.data.simulator.SimulateGrid.threshold`

```python
threshold = None
```

######## `nltools.data.simulator.SimulateGrid.threshold_type`

```python
threshold_type = None
```

######## `nltools.data.simulator.SimulateGrid.thresholded`

```python
thresholded = None
```



####### Functions######## `nltools.data.simulator.SimulateGrid.add_signal`

```python
add_signal(signal_width = 20, signal_amplitude = 1)
```

Add rectangular signal to self.data

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`signal_width` | <code>[int](#int)</code> | width of signal box | <code>20</code>
`signal_amplitude` | <code>[int](#int)</code> | intensity of signal | <code>1</code>

######## `nltools.data.simulator.SimulateGrid.create_mask`

```python
create_mask(signal_width)
```

Create a mask for where the signal is located in grid.

######## `nltools.data.simulator.SimulateGrid.fit`

```python
fit()
```

Run ttest on self.data

######## `nltools.data.simulator.SimulateGrid.plot_grid_simulation`

```python
plot_grid_simulation(threshold, threshold_type, n_simulations = 100, correction = None)
```

Create a plot of the simulations

######## `nltools.data.simulator.SimulateGrid.run_multiple_simulations`

```python
run_multiple_simulations(threshold, threshold_type, n_simulations = 100, correction = None)
```

This method will run multiple simulations to calculate overall false positive rate

######## `nltools.data.simulator.SimulateGrid.threshold_simulation`

```python
threshold_simulation(threshold, threshold_type, correction = None)
```

Threshold simulation

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`threshold` | <code>[float](#float)</code> | threshold to apply to simulation | *required*
`threshhold_type` | <code>[str](#str)</code> | type of threshold to use can be a specific t-value or p-value ['t', 'p', 'q'] | *required*

###### `nltools.data.simulator.Simulator`

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
[`brain_mask`](#nltools.data.simulator.Simulator.brain_mask) |  | The brain mask image used for simulation.
[`nifti_masker`](#nltools.data.simulator.Simulator.nifti_masker) |  | Fitted NiftiMasker for converting between 4D data and 2D arrays.
[`output_dir`](#nltools.data.simulator.Simulator.output_dir) |  | Output directory path.
[`random_state`](#nltools.data.simulator.Simulator.random_state) |  | Random state for reproducible simulations.

**Examples:**

```pycon
>>> from nltools.data.simulator import Simulator
>>> sim = Simulator(random_state=42)
>>> # Create a dataset with signal in specific regions
>>> data = sim.create_data(y=[1, -1, 1, -1], sigma=1, n_reps=10)
```

**Functions:**

Name | Description
---- | -----------
[`create_cov_data`](#nltools.data.simulator.Simulator.create_cov_data) | create continuous simulated data with covariance
[`create_data`](#nltools.data.simulator.Simulator.create_data) | create simulated data with integers
[`create_ncov_data`](#nltools.data.simulator.Simulator.create_ncov_data) | create continuous simulated data with covariance
[`gaussian`](#nltools.data.simulator.Simulator.gaussian) | create a 3D gaussian signal normalized to a given intensity
[`n_spheres`](#nltools.data.simulator.Simulator.n_spheres) | generate a set of spheres in the brain mask space
[`normal_noise`](#nltools.data.simulator.Simulator.normal_noise) | produce a normal noise distribution for all all points in the brain mask
[`sphere`](#nltools.data.simulator.Simulator.sphere) | create a sphere of given radius at some point p in the brain mask
[`to_nifti`](#nltools.data.simulator.Simulator.to_nifti) | convert a numpy matrix to the nifti format and assign to it the brain_mask's affine matrix



####### Attributes######## `nltools.data.simulator.Simulator.brain_mask`

```python
brain_mask = brain_mask
```

######## `nltools.data.simulator.Simulator.nifti_masker`

```python
nifti_masker = NiftiMasker(mask_img=(self.brain_mask))
```

######## `nltools.data.simulator.Simulator.output_dir`

```python
output_dir = os.path.join(os.getcwd())
```

######## `nltools.data.simulator.Simulator.random_state`

```python
random_state = check_random_state(random_state)
```



####### Functions######## `nltools.data.simulator.Simulator.create_cov_data`

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

######## `nltools.data.simulator.Simulator.create_data`

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

######## `nltools.data.simulator.Simulator.create_ncov_data`

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

######## `nltools.data.simulator.Simulator.gaussian`

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

######## `nltools.data.simulator.Simulator.n_spheres`

```python
n_spheres(radius, center)
```

generate a set of spheres in the brain mask space

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`radius` |  | vector of radius.  Will create multiple spheres if len(radius) > 1 | *required*
`centers` |  | a vector of sphere centers of the form [px, py, pz] or [[px1, py1, pz1], ..., [pxn, pyn, pzn]] | *required*

######## `nltools.data.simulator.Simulator.normal_noise`

```python
normal_noise(mu, sigma)
```

produce a normal noise distribution for all all points in the brain mask

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mu` |  | average value of the gaussian signal (usually set to 0) | *required*
`sigma` |  | standard deviation | *required*

######## `nltools.data.simulator.Simulator.sphere`

```python
sphere(r, p)
```

create a sphere of given radius at some point p in the brain mask

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`r` |  | radius of the sphere | *required*
`p` |  | point (in coordinates of the brain mask) of the center of the sphere | *required*

######## `nltools.data.simulator.Simulator.to_nifti`

```python
to_nifti(m)
```

convert a numpy matrix to the nifti format and assign to it the brain_mask's affine matrix

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`m` |  | the 3D numpy matrix we wish to convert to .nii | *required*



##### Functions