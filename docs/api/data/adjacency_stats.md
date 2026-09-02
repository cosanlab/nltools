---
title: data.adjacency.stats
label: data-adjacency-stats
---

Provide standalone statistical functions for Adjacency matrices.

Each function takes an Adjacency instance as its first argument (`adj`).

**Functions:**

Name | Description
---- | -----------
[`cluster_summary`](#data-adjacency-stats-cluster-summary) | This function provides summaries of clusters within Adjacency matrices.
[`plot_label_distance`](#data-adjacency-stats-plot-label-distance) | Create a violin plot of within- and between-label distances.
[`plot_silhouette`](#data-adjacency-stats-plot-silhouette) | Create a silhouette plot.
[`r_to_z`](#data-adjacency-stats-r-to-z) | Apply Fisher's r to z transformation to each element of the data object.
[`similarity`](#data-adjacency-stats-similarity) | Calculate similarity between two Adjacency matrices.
[`stats_label_distance`](#data-adjacency-stats-stats-label-distance) | Calculate permutation tests on within and between label distance.
[`threshold`](#data-adjacency-stats-threshold) | Threshold an Adjacency instance.
[`ttest`](#data-adjacency-stats-ttest) | Calculate ttest across samples.
[`z_to_r`](#data-adjacency-stats-z-to-r) | Convert z score back into r value for each element of data object.



## Functions

(data-adjacency-stats-cluster-summary)=
### `cluster_summary`

```python
cluster_summary(adj, *, clusters = None, summary = 'mean', scope = 'within')
```

This function provides summaries of clusters within Adjacency matrices.

It can compute mean/median of within and between cluster values. Requires a
list of cluster ids indicating the row/column of each cluster.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#data-adjacency)</code> | Adjacency instance | *required*
`clusters` |  | (list) list of cluster labels | <code>None</code>
`summary` |  | (str) central tendency, 'mean' or 'median'. If `None` then return all r values | <code>'mean'</code>
`scope` |  | (str) summarize 'within' cluster or 'between' clusters | <code>'within'</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | (dict) per-cluster summaries

(data-adjacency-stats-plot-label-distance)=
### `plot_label_distance`

```python
plot_label_distance(adj, labels = None, ax = None)
```

Create a violin plot of within- and between-label distances.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#data-adjacency)</code> | Adjacency instance (must be a single matrix) | *required*
`labels` | <code>array</code> | numpy array of labels to plot | <code>None</code>

(data-adjacency-stats-plot-silhouette)=
### `plot_silhouette`

```python
plot_silhouette(adj, *, labels = None, ax = None, permutation_test = True, n_permute = 5000, colors = None, figsize = (6, 4))
```

Create a silhouette plot.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#data-adjacency)</code> | Adjacency instance (must be a single matrix). | *required*
`labels` | <code>array</code> | Numpy array of cluster/group labels. | <code>None</code>
`ax` |  | Matplotlib axis handle. | <code>None</code>
`permutation_test` | <code>bool</code> | Whether to run a permutation test. Default True. | <code>True</code>
`n_permute` | <code>int</code> | Number of permutations for the test. Default 5000. | <code>5000</code>
`colors` |  | Optional list of RGB triplets, one per cluster (default: seaborn 'hls' palette). | <code>None</code>
`figsize` |  | Figure size tuple. Default (6, 4). | <code>(6, 4)</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Silhouette plot results including scores and optional permutation p-value.

(data-adjacency-stats-r-to-z)=
### `r_to_z`

```python
r_to_z(adj)
```

Apply Fisher's r to z transformation to each element of the data object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#data-adjacency)</code> | Adjacency instance. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Adjacency](#data-adjacency)</code> | New Adjacency with z-transformed values.

(data-adjacency-stats-similarity)=
### `similarity`

```python
similarity(adj, data, plot = False, method = '2d', n_permute = 5000, metric = 'spearman', include_diag = False, nan_policy = 'omit', tail = 2, return_null = False, n_jobs = -1, random_state = None, *, project: bool = False, progress_bar: bool = False)
```

Calculate similarity between two Adjacency matrices.

The default uses Spearman correlation and a permutation test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#data-adjacency)</code> | Adjacency instance. | *required*
`data` | <code>[Adjacency](#data-adjacency) or array</code> | Adjacency data, or 1-d array same size as adj.data. | *required*
`plot` | <code>bool</code> | If True, plot stacked adjacency matrices. Default False. | <code>False</code>
`method` | <code>str</code> | permutation scheme '1d', '2d', or None. | <code>'2d'</code>
`n_permute` | <code>int</code> | Number of permutations. Default 5000. | <code>5000</code>
`metric` | <code>str</code> | 'spearman', 'pearson', or 'kendall'. | <code>'spearman'</code>
`include_diag` | <code>bool</code> | Only applies to 'directed' Adjacency types using method=None or method='1d'. Default False (self-similarity is uninformative). Symmetric matrices never store the diagonal, so this flag is a no-op for them. | <code>False</code>
`nan_policy` | <code>str</code> | How to handle NaN values. Options: - 'omit': Remove NaN values pairwise before computing correlation (default) - 'propagate': Allow NaN to propagate through calculations - 'raise': Raise an error if NaN values are present | <code>'omit'</code>
`tail` | <code>int \| str</code> | `2`/`'two'` (two-tailed, default) or `1`/`'one'` (one-tailed, positive direction). | <code>2</code>
`return_null` | <code>bool</code> | If True, also return the null distribution. Default False. | <code>False</code>
`n_jobs` | <code>int</code> | Number of parallel jobs. -1 means all cores. Default -1. | <code>-1</code>
`random_state` | <code>int</code> | Random seed for reproducibility. | <code>None</code>
`project` | <code>bool</code> | If True and adj has a spatial_scale, project the per-matrix correlations back into brain space. Default False. | <code>False</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. Default False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict \| list[dict] \| [BrainData](#data-brain-data)</code> | A correlation result dict with keys     'correlation', 'p', and 'device' (or a list of such dicts when adj     contains multiple matrices); a `BrainData` when `project=True`,     holding the per-matrix correlations projected back into brain space     via the spatial_scale.

(data-adjacency-stats-stats-label-distance)=
### `stats_label_distance`

```python
stats_label_distance(adj, *, labels = None, n_permute = 5000, n_jobs = -1, progress_bar = False)
```

Calculate permutation tests on within and between label distance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#data-adjacency)</code> | Adjacency instance (must be a single matrix) | *required*
`labels` | <code>array</code> | numpy array of labels to plot | <code>None</code>
`n_permute` | <code>int</code> | number of permutations to run (default=5000) | <code>5000</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | dictionary of within and between group differences         and p-values

(data-adjacency-stats-threshold)=
### `threshold`

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
`adj` | <code>[Adjacency](#data-adjacency)</code> | Adjacency instance | *required*
`upper` |  | (float or str) Upper cutoff for thresholding. If string     will interpret as percentile; can be None for one-sided     thresholding. | <code>None</code>
`lower` |  | (float or str) Lower cutoff for thresholding. If string     will interpret as percentile; can be None for one-sided     thresholding. | <code>None</code>
`binarize` | <code>bool</code> | return binarized image respecting thresholds if     provided, otherwise binarize on every non-zero value;     default False | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[Adjacency](#data-adjacency)</code> | thresholded Adjacency instance

(data-adjacency-stats-ttest)=
### `ttest`

```python
ttest(adj, *, permutation = False, n_permute = 5000, tail = 2, return_null = False, n_jobs = -1, random_state = None, progress_bar = False)
```

Calculate ttest across samples.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#data-adjacency)</code> | Adjacency instance (must contain multiple matrices) | *required*
`permutation` |  | (bool) Run ttest as permutation. Note this can be very slow. | <code>False</code>
`n_permute` |  | Number of permutations (used only when ``permutation=True``). Default 5000. | <code>5000</code>
`tail` |  | `2`/`'two'` (two-tailed, default) or `1`/`'one'` (one-tailed, positive direction). | <code>2</code>
`return_null` |  | If True, also return the null distribution. Default False. | <code>False</code>
`n_jobs` |  | Number of parallel jobs. Default -1 (all cores). | <code>-1</code>
`random_state` |  | Random seed for reproducibility. | <code>None</code>
`progress_bar` |  | If True, show a progress bar. Default False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Contains Adjacency instances of t values (or mean if running     permutation) and Adjacency instance of p values.

(data-adjacency-stats-z-to-r)=
### `z_to_r`

```python
z_to_r(adj)
```

Convert z score back into r value for each element of data object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#data-adjacency)</code> | Adjacency instance. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Adjacency](#data-adjacency)</code> | New Adjacency with r values.
