---
title: Adjacency
label: page-data-adjacency
---

```python
Adjacency(data = None, *, Y = None, matrix_type = None, labels = None)
```

Represent adjacency matrices in vectorized form.

Store distance/similarity matrices as strict upper triangles and directed
matrices as full row-major vectors. Symmetric reconstruction always has a
zero diagonal; input diagonals are discarded. Flat rectangular stacks require
an explicit `*_flat` matrix type. A list or 2-D flat array retains stack rank,
including one matrix. A zero-length symmetric vector represents one node.
Construction and result methods return independently owned mutable state.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Adjacency](#page-data-adjacency) \| ndarray \| DataFrame \| DataFrame \| str \| Path \| list</code> | A square matrix, a flattened vector, a `.csv`/`.h5` path, or a list of matrices/`Adjacency` instances/`.csv` paths to stack. | <code>None</code>
`Y` | <code>DataFrame \| DataFrame</code> | Matrix metadata, one row per matrix. None inherits metadata during copy construction. | <code>None</code>
`matrix_type` | <code>str</code> | Type of matrix. One of `'distance'`, `'similarity'`, `'directed'`, `'distance_flat'`, `'similarity_flat'`, `'directed_flat'`. For copy construction, this confirms the existing kind without reinterpreting it. | <code>None</code>
`labels` | <code>list</code> | Shared node labels, or a nested matrix-by-node label grid for a stack. None inherits labels during copy construction. | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`data` | <code>ndarray</code> | Vectorized matrix values. Shape `(vector_length,)` for a single matrix or `(n_matrices, vector_length)` for a stack; symmetric matrices store only the upper triangle without the diagonal.
`matrix_type` | <code>str</code> | One of `'distance'`, `'similarity'`, `'directed'`, or `'empty'` (the `'_flat'` input variants are normalized to their base type).
`is_single_matrix` | <code>bool</code> | True for single storage; a one-row stack is False.
`issymmetric` | <code>bool</code> | True for distance/similarity matrices, False for directed.
`labels` | <code>list</code> | Node labels (empty list when none were given).
`Y` | <code>DataFrame</code> | Training labels as a polars DataFrame (possibly empty).
`is_empty` | <code>bool</code> | True if the instance holds no matrices.
`n_nodes` | <code>int</code> | Number of nodes `n` for an `(n, n)` matrix.
`shape` | <code>tuple</code> | Logical shape — `(n_nodes, n_nodes)` for a single matrix, `(n_matrices, n_nodes, n_nodes)` for a stack, including typed empty stacks; `(0, 0)` for an untyped empty constructor.
`vector_shape` | <code>tuple</code> | Shape of the internal vectorized storage (`data.shape`).

**Methods:**

Name | Description
---- | -----------
[`append`](#data-adjacency-append) | Append data to an Adjacency instance.
[`bootstrap`](#data-adjacency-bootstrap) | Bootstrap statistics using efficient online algorithms.
[`cluster_summary`](#data-adjacency-cluster-summary) | Provide summaries of clusters within Adjacency matrices.
[`copy`](#data-adjacency-copy) | Return an independently owned copy, preserving internal aliases and cycles.
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
[`to_graph`](#data-adjacency-to-graph) | Convert a single Adjacency matrix into a NetworkX graph.
[`to_square`](#data-adjacency-to-square) | Convert adjacency back to square matrix format.
[`ttest`](#data-adjacency-ttest) | Run a one-sample t-test across stacked matrices.
[`write`](#data-adjacency-write) | Write the Adjacency to a `.csv` or `.h5` file.
[`z_to_r`](#data-adjacency-z-to-r) | Convert each z score back into an r value.

## Methods

(data-adjacency-append)=
### `append`

```python
append(data)
```

Append data to an Adjacency instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Adjacency](#page-data-adjacency)</code> | Adjacency instance to append. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Adjacency](#page-data-adjacency)</code> | New appended Adjacency instance.

(data-adjacency-bootstrap)=
### `bootstrap`

```python
bootstrap(stat, *, n_samples = 5000, save_boots = False, percentiles = (2.5, 97.5), tail = 2, n_jobs = -1, random_state = None, progress_bar: bool = False)
```

Bootstrap statistics using efficient online algorithms.

Uses memory-efficient bootstrap infrastructure with CPU parallelization.
Supports simple aggregation statistics (mean, std, median, sum, min, max).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stat` | <code>str</code> | Statistic to bootstrap: `'mean'`, `'median'`, `'std'`, `'sum'`, `'min'`, or `'max'`. | *required*
`n_samples` | <code>int</code> | Number of bootstrap iterations. Default 5000. | <code>5000</code>
`save_boots` | <code>bool</code> | If True, store all bootstrap samples (memory intensive). Default False. | <code>False</code>
`percentiles` | <code>tuple</code> | Percentiles for confidence intervals. Default (2.5, 97.5). | <code>(2.5, 97.5)</code>
`tail` | <code>int \| str</code> | `2`/`'two'` for two-tailed (default); `1`/`'one'` for one-tailed (statistic > 0; negate the data for the other direction). | <code>2</code>
`n_jobs` | <code>int</code> | Number of CPU cores for parallelization. -1 means all CPUs. | <code>-1</code>
`random_state` | <code>int</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. Default False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Dictionary with keys `'Z'`, `'p'`, `'mean'`, `'std'`, `'ci_lower'`,     `'ci_upper'` (all Adjacency objects). If `save_boots=True`, also     includes `'samples'`.

**Examples:**

```python
boot = adj.bootstrap(stat="mean", n_samples=1000)
boot["mean"]  # → Adjacency
```

(data-adjacency-cluster-summary)=
### `cluster_summary`

```python
cluster_summary(*, clusters = None, summary = 'mean', scope = 'within')
```

Provide summaries of clusters within Adjacency matrices.

Computes mean/median of within and between cluster values. Requires a
list of cluster ids indicating the row/column of each cluster.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`clusters` | <code>list</code> | Cluster label for each row/column. | <code>None</code>
`summary` | <code>str \| None</code> | Central tendency, `'mean'` or `'median'`. If None, return all values instead of a summary. | <code>'mean'</code>
`scope` | <code>str</code> | Summarize `'within'` cluster or `'between'` clusters. | <code>'within'</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Per-cluster summaries keyed by cluster label.

(data-adjacency-copy)=
### `copy`

```python
copy()
```

Return an independently owned copy, preserving internal aliases and cycles.

(data-adjacency-distance)=
### `distance`

```python
distance(metric = 'correlation', include_diag = False, **kwargs)
```

Calculate distance between images within an Adjacency() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` | <code>str</code> | Distance metric; any metric accepted by `sklearn.metrics.pairwise_distances` (scikit-learn or scipy). | <code>'correlation'</code>
`include_diag` | <code>bool</code> | Whether to include the main diagonal when computing distances between adjacency matrices. Only applies to symmetric matrices. Default False (consistent with how symmetric matrices are stored without the diagonal). | <code>False</code>
`**kwargs` | <code>dict</code> | Forwarded to `sklearn.metrics.pairwise_distances`. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[Adjacency](#page-data-adjacency)</code> | A 2D distance matrix.

(data-adjacency-distance-to-similarity)=
### `distance_to_similarity`

```python
distance_to_similarity(metric = 'correlation', beta = 1)
```

Convert distance matrix to similarity matrix.

Currently only implemented for the 'correlation' and 'euclidean' metrics.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` | <code>str</code> | Either 'correlation' or 'euclidean'. | <code>'correlation'</code>
`beta` | <code>float</code> | Scale parameter of the exponential used for 'euclidean' (default: 1). | <code>1</code>

**Returns:**

Type | Description
---- | -----------
<code>[Adjacency](#page-data-adjacency)</code> | The converted similarity matrix.

(data-adjacency-generate-permutations)=
### `generate_permutations`

```python
generate_permutations(n_permute, random_state = None)
```

Generate permuted versions of an Adjacency instance lazily.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_permute` | <code>int</code> | Number of permutations. | *required*
`random_state` | <code>int \| RandomState</code> | Random seed for reproducibility. | <code>None</code>

**Yields:**

Type | Description
---- | -----------
<code>[Adjacency](#page-data-adjacency)</code> | Permuted version of self.

**Examples:**

```python
for perm in adj.generate_permutations(1000):
    out = neural_distance_mat.similarity(perm)
```

(data-adjacency-mean)=
### `mean`

```python
mean(axis = 0)
```

Calculate mean of Adjacency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>int</code> | Calculate mean over matrices (0) or upper triangle (1). | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>float \| [Adjacency](#page-data-adjacency) \| ndarray</code> | A float for a single matrix; an     Adjacency when `axis=0`; an array when `axis=1`.

(data-adjacency-median)=
### `median`

```python
median(axis = 0)
```

Calculate median of Adjacency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>int</code> | Calculate median over matrices (0) or upper triangle (1). | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>float \| [Adjacency](#page-data-adjacency) \| ndarray</code> | A float for a single matrix; an     Adjacency when `axis=0`; an array when `axis=1`.

(data-adjacency-plot)=
### `plot`

```python
plot(limit = 3, axes = None, *args, **kwargs)
```

Create a heatmap of an Adjacency matrix.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`limit` | <code>int</code> | Number of heatmaps to plot if the object contains multiple matrices. Default 3. | <code>3</code>
`axes` | <code>Axes</code> | Axis to draw on (single matrix only). | <code>None</code>
`*args` | <code>tuple</code> | Forwarded positionally to `seaborn.heatmap`. | <code>()</code>
`**kwargs` | <code>dict</code> | Forwarded to `seaborn.heatmap`. | <code>{}</code>

(data-adjacency-plot-label-distance)=
### `plot_label_distance`

```python
plot_label_distance(labels = None, ax = None)
```

Create a violin plot of within- and between-label distances.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`labels` | <code>ndarray</code> | Group label per node; defaults to the stored labels. | <code>None</code>
`ax` | <code>Axes</code> | Axis to draw on. | <code>None</code>

(data-adjacency-plot-mds)=
### `plot_mds`

```python
plot_mds(*, n_components = 2, metric_mds = True, labels = None, labels_color = None, cmap = None, view = (30, 20), figsize = None, ax = None, n_jobs = -1, **kwargs)
```

Plot multidimensional scaling.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_components` | <code>int</code> | Number of dimensions to project (2 or 3). | <code>2</code>
`metric_mds` | <code>bool</code> | Perform metric (True) or non-metric (False) scaling. Default True. | <code>True</code>
`labels` | <code>list</code> | Overrides the labels stored on the instance. | <code>None</code>
`labels_color` | <code>list</code> | One color per label. | <code>None</code>
`cmap` | <code>Colormap</code> | Colormap. Default `plt.cm.hot_r`. | <code>None</code>
`view` | <code>tuple</code> | Elevation/azimuth for a 3-D plot. Default (30, 20). | <code>(30, 20)</code>
`figsize` | <code>list</code> | Figure size. Default [12, 8]. | <code>None</code>
`ax` | <code>Axes</code> | Axis to draw on. | <code>None</code>
`n_jobs` | <code>int</code> | Number of parallel jobs. | <code>-1</code>
`**kwargs` | <code>dict</code> | Forwarded to `sklearn.manifold.MDS`. | <code>{}</code>

(data-adjacency-plot-silhouette)=
### `plot_silhouette`

```python
plot_silhouette(*, labels = None, ax = None, permutation_test = True, n_permute = 5000, colors = None, figsize = (6, 4))
```

Create a silhouette plot.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`labels` | <code>ndarray</code> | Cluster/group label per node (overrides stored labels). | <code>None</code>
`ax` | <code>Axes</code> | Axis to draw on. | <code>None</code>
`permutation_test` | <code>bool</code> | Whether to run a permutation test. Default True. | <code>True</code>
`n_permute` | <code>int</code> | Number of permutations for the test. Default 5000. | <code>5000</code>
`colors` | <code>list</code> | RGB triplets, one per cluster. Default: seaborn `'hls'` palette. | <code>None</code>
`figsize` | <code>tuple</code> | Figure size. Default (6, 4). | <code>(6, 4)</code>

**Returns:**

Type | Description
---- | -----------
<code>DataFrame</code> | Columns `label` and `mean_silhouette`, plus `p` when     `permutation_test=True`.

(data-adjacency-r-to-z)=
### `r_to_z`

```python
r_to_z()
```

Apply Fisher's r-to-z transformation to each data element.

(data-adjacency-regress)=
### `regress`

```python
regress(X, method = 'ols', tail = 2)
```

Run a regression on an adjacency instance.

Pass an `Adjacency` as `X` to decompose this matrix with other matrices, or a
`DesignMatrix` to regress each cell across a stack of matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[Adjacency](#page-data-adjacency) \| [DesignMatrix](#page-data-design-matrix)</code> | Design matrix. | *required*
`method` | <code>str</code> | Type of regression; only `'ols'` is currently supported. | <code>'ols'</code>
`tail` | <code>int \| str</code> | `2`/`'two'` (two-tailed, default) or `1`/`'one'` (one-tailed: beta > 0; negate a regressor for the other direction). | <code>2</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `beta`, `sigma` (coefficient standard error), `t`, `p`,     `df`, and `residual`. With DesignMatrix predictors, coefficient     fields are Adjacency maps per predictor (single for one predictor).     With Adjacency predictors, a single response is required and     coefficient fields are native predictor arrays or scalars.     `df` is a scalar; `residual` retains response shape and metadata.

(data-adjacency-similarity)=
### `similarity`

```python
similarity(data, *, plot = False, method = '2d', n_permute = 5000, metric = 'spearman', include_diag = False, nan_policy = 'omit', tail = 2, return_null = False, n_jobs = -1, random_state = None, progress_bar: bool = False)
```

Calculate similarity between two Adjacency matrices.

The default uses Spearman correlation and a permutation test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Adjacency](#page-data-adjacency) \| ndarray</code> | Adjacency to compare against, or a 1-D array the same size as `self.data`. | *required*
`plot` | <code>bool</code> | Plot the two stacked adjacency matrices being compared. Default False. | <code>False</code>
`method` | <code>str \| None</code> | Permutation scheme, `'1d'`, `'2d'`, or None (no permutation test). | <code>'2d'</code>
`n_permute` | <code>int</code> | Number of permutations for the p-value. Default 5000. | <code>5000</code>
`metric` | <code>str</code> | `'spearman'`, `'pearson'`, or `'kendall'`. | <code>'spearman'</code>
`include_diag` | <code>bool</code> | Only applies to `'directed'` matrices with `method=None` or `method='1d'`. Default False (self-similarity is uninformative). Symmetric matrices never store the diagonal, so this flag is a no-op for them. | <code>False</code>
`nan_policy` | <code>str</code> | How to handle NaN values: `'omit'` removes NaN pairwise before computing the correlation (default), `'propagate'` lets NaN flow through, `'raise'` errors if any NaN is present. | <code>'omit'</code>
`tail` | <code>int \| str</code> | `2`/`'two'` (two-tailed, default) or `1`/`'one'` (one-tailed, positive direction). | <code>2</code>
`return_null` | <code>bool</code> | If True, also return the null distribution. Default False. | <code>False</code>
`n_jobs` | <code>int</code> | Number of parallel jobs. Default -1 (all cores). | <code>-1</code>
`random_state` | <code>int</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. Default False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict \| list[dict]</code> | A correlation result dict with keys 'correlation',     'p', and 'device' for a single matrix, or a list of these dicts     for a stack.

(data-adjacency-social-relations-model)=
### `social_relations_model`

```python
social_relations_model(summarize_results = True, nan_replace = True)
```

Estimate the social relations model from a matrix for a round-robin design.

$$X_{ij} = m + \alpha_i + \beta_j + g_{ij} + \epsilon_{ijl}$$

where $X_{ij}$ is the score for person i rating person j, $m$ is the group mean,
$\alpha_i$ is person i's actor effect, $\beta_j$ is person j's partner effect, $g_{ij}$
is the relationship effect and $\epsilon_{ijl}$ is the error in measure l for actor i and partner j.

This model is primarily concerned with partitioning the variance of the various
effects. The implementation follows Chapter 8 of Kenny, Kashy, & Cook (2006) and
the tests replicate the book's examples. Actor scores are rows (lower triangle)
and partner scores are columns (upper triangle). The minimal sample size to
estimate these effects is 4.

**Model assumptions:** social interactions are exclusively dyadic; people are
randomly sampled from the population; there are no order effects; the effects
combine additively and relationships are linear.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`summarize_results` | <code>bool</code> | If True, print a formatted summary of model results. | <code>True</code>
`nan_replace` | <code>bool</code> | If True, replace NaN values with row and column means. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>Series \| DataFrame</code> | All of the effects estimated using SRM, as a     Series (single matrix) or DataFrame (one row per matrix).

<details class="references" open markdown="1">
<summary>References</summary>

Kenny, D. A., Kashy, D. A., & Cook, W. L. (2006). *Dyadic data analysis*.
Guilford Press.

</details>

(data-adjacency-squareform)=
### `squareform`

```python
squareform()
```

Convert adjacency data back to square form.

**Returns:**

Type | Description
---- | -----------
<code>ndarray \| list[ndarray]</code> | Detached square matrix, or a list of     detached matrices for a stack. Symmetric diagonals are zero.

(data-adjacency-stats-label-distance)=
### `stats_label_distance`

```python
stats_label_distance(*, labels = None, n_permute = 5000, n_jobs = -1)
```

Calculate permutation tests on within and between label distance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`labels` | <code>ndarray</code> | Group label per node; defaults to the stored labels. | <code>None</code>
`n_permute` | <code>int</code> | Number of permutations to run. Default 5000. | <code>5000</code>
`n_jobs` | <code>int</code> | Number of parallel jobs. Default -1 (all cores). | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Per-group within-vs-between distance differences and p-values, keyed     by group label.

(data-adjacency-std)=
### `std`

```python
std(axis = 0)
```

Calculate standard deviation of Adjacency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>int</code> | Calculate std over matrices (0) or upper triangle (1). | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>float \| [Adjacency](#page-data-adjacency) \| ndarray</code> | A float for a single matrix; an     Adjacency when `axis=0`; an array when `axis=1`.

(data-adjacency-sum)=
### `sum`

```python
sum(axis = 0)
```

Calculate sum of Adjacency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>int</code> | Calculate sum over matrices (0) or upper triangle (1). | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>float \| [Adjacency](#page-data-adjacency) \| ndarray</code> | A float for a single matrix; an     Adjacency when `axis=0`; an array when `axis=1`.

(data-adjacency-threshold)=
### `threshold`

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
`upper` | <code>float \| str</code> | Upper cutoff. A string such as `'95%'` is interpreted as a percentile; None for one-sided thresholding. | <code>None</code>
`lower` | <code>float \| str</code> | Lower cutoff. A string such as `'5%'` is interpreted as a percentile; None for one-sided thresholding. | <code>None</code>
`binarize` | <code>bool</code> | Return a binarized matrix respecting the thresholds if provided, otherwise binarize on every non-zero value. Default False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[Adjacency](#page-data-adjacency)</code> | Thresholded Adjacency instance.

(data-adjacency-to-graph)=
### `to_graph`

```python
to_graph()
```

Convert a single Adjacency matrix into a NetworkX graph.

This currently works only when `is_single_matrix` is True.

**Returns:**

Type | Description
---- | -----------
<code>Graph \| DiGraph</code> | `DiGraph` for directed matrices,     `Graph` otherwise; nodes are relabeled with `labels` when set.

(data-adjacency-to-square)=
### `to_square`

```python
to_square()
```

Convert adjacency back to square matrix format.

This is an alias for `squareform`.

**Returns:**

Type | Description
---- | -----------
<code>ndarray \| list[ndarray]</code> | Square matrix representation, or a list     of them if this object contains multiple adjacency matrices.

(data-adjacency-ttest)=
### `ttest`

```python
ttest(*, popmean = 0.0, permutation = False, n_permute = 5000, tail = 2, return_null = False, n_jobs = -1, random_state = None, progress_bar: bool = False)
```

Run a one-sample t-test across stacked matrices.

Tests every stored edge against `popmean` across the matrices in the
stack.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`popmean` | <code>float</code> | Population mean to test against. Default 0.0. | <code>0.0</code>
`permutation` | <code>bool</code> | If True, take p from a sign-flip permutation test on `matrices - popmean`. The reported `t` stays the observed parametric statistic. Default False. | <code>False</code>
`n_permute` | <code>int</code> | Number of permutations, used only when `permutation=True`. Default 5000. | <code>5000</code>
`tail` | <code>int \| str</code> | `2`/`'two'` (two-tailed, default) or `1`/`'one'` (one-tailed: mean > `popmean`). Applies to both paths. | <code>2</code>
`return_null` | <code>bool</code> | If True, also return the permutation null. Has no effect on the parametric path, which computes no null. Default False. | <code>False</code>
`n_jobs` | <code>int</code> | Number of parallel jobs. Default -1 (all cores). | <code>-1</code>
`random_state` | <code>int</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. Default False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | `'mean'`, `'t'`, `'z'` and `'p'` as independent single-matrix     `Adjacency` results that retain the node count, storage kind     (including directed) and shared node labels, with matrix     metadata cleared. `'mean'` is the edgewise mean minus `popmean`;     `'t'` is the observed one-sample t-statistic on both paths;     `'p'` is parametric, or the empirical sign-flip p-value when     `permutation=True`; `'z'` is the tail-aware normal score of `p`.     With `permutation=True` and `return_null=True` the dict also     holds `'null_dist'`, an owned `(n_permute, n_edges)` array of     centered means in flat storage order and in the units of     `'mean'`. Maps are unthresholded. Apply a cutoff or a     multiple-comparison correction afterwards.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If this Adjacency holds fewer than two matrices.

**Examples:**

```python
result = stacked.ttest()
result["mean"]  # effect size per edge
result["t"].squareform()  # back to a node-by-node matrix

# Threshold after testing, never inside it
import numpy as np

significant = result["t"].copy()
significant.data = np.where(result["p"].data < 0.05, result["t"].data, 0.0)
```

(data-adjacency-write)=
### `write`

```python
write(file_name, method = 'long')
```

Write the Adjacency to a `.csv` or `.h5` file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>str \| Path</code> | Output path; an `.h5`/`.hdf5` suffix writes HDF5. | *required*
`method` | <code>str</code> | Layout for CSV output, `'long'` (vectorized rows) or `'square'` (single matrix only). | <code>'long'</code>

(data-adjacency-z-to-r)=
### `z_to_r`

```python
z_to_r()
```

Convert each z score back into an r value.
