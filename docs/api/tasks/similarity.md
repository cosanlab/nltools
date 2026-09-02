---
title: Similarity & RSA
label: page-tasks-similarity
---

Compare patterns and matrices. `compute_similarity` scores two arrays under a `metric=`; the Fisher transforms make correlations averageable. `matrix_permutation_test` (Mantel), `correlation_permutation_test`, and `distance_correlation` compare whole matrices. The plots summarize stacks of [Adjacency](../data/adjacency.md) matrices, and `SpatialScale` records which ROI or searchlight each matrix in a stack came from so a reduction can be painted back onto the brain.

**Classes:**

Name | Description
---- | -----------
[`SpatialScale`](#tasks-similarity-spatialscale) | Record provenance for a per-parcel or per-searchlight Adjacency stack.

**Functions:**

Name | Description
---- | -----------
[`compute_similarity`](#tasks-similarity-compute-similarity) | Compute row-wise similarity between two data arrays.
[`compute_multivariate_similarity`](#tasks-similarity-compute-multivariate-similarity) | Compute multivariate similarity by regressing one pattern on several.
[`transform_pairwise`](#tasks-similarity-transform-pairwise) | Transform data into pairwise differences with balanced labels for ranking.
[`fisher_r_to_z`](#tasks-similarity-fisher-r-to-z) | Convert correlation coefficients to Fisher z values.
[`fisher_z_to_r`](#tasks-similarity-fisher-z-to-r) | Convert Fisher z back to a correlation coefficient.
[`matrix_permutation_test`](#tasks-similarity-matrix-permutation-test) | Matrix permutation test (Mantel test) for correlating two square matrices.
[`correlation_permutation_test`](#tasks-similarity-correlation-permutation-test) | Permutation test for whether the correlation between two arrays differs from zero.
[`distance_correlation`](#tasks-similarity-distance-correlation) | Compute the distance correlation between two arrays to test for multivariate dependence.
[`double_center`](#tasks-similarity-double-center) | Double center a 2d array.
[`u_center`](#tasks-similarity-u-center) | U-center a 2d array.
[`plot_stacked_adjacency`](#tasks-similarity-plot-stacked-adjacency) | Create stacked adjacency to illustrate similarity.
[`plot_mean_label_distance`](#tasks-similarity-plot-mean-label-distance) | Violin plot of within- vs between-label distances.
[`plot_between_label_distance`](#tasks-similarity-plot-between-label-distance) | Heatmap of average pairwise distance between every label pair.
[`plot_silhouette`](#tasks-similarity-plot-silhouette) | Silhouette plot indicating between- vs within-label distance.

## Classes

(tasks-similarity-spatialscale)=
### `SpatialScale`

```python
SpatialScale(atlas: BrainData, roi_labels: np.ndarray, source_mask: Nifti1Image, kind: Literal['roi', 'searchlight'] = 'roi')
```

Record provenance for a per-parcel or per-searchlight Adjacency stack.

The stack comes from a per-parcel or per-searchlight operation on a
`BrainData`.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`atlas` | <code>[BrainData](#page-data-brain-data)</code> | Labeled volume indicating parcel membership (or searchlight centers). One matrix in the stack per unique label.
`roi_labels` | <code>ndarray</code> | Integer atlas IDs in stack order. ``len(roi_labels)`` must equal the number of matrices in the stack.
`source_mask` | <code>Nifti1Image</code> | The brain mask the atlas/values live in. Used as the target space for back-projection in ``Adjacency.to_brain()``.
`kind` | <code>Literal['roi', 'searchlight']</code> | Which spatial scale produced this stack — ``'roi'`` or ``'searchlight'``.

## Functions

(tasks-similarity-compute-similarity)=
### `compute_similarity`

```python
compute_similarity(data1, data2, metric = 'correlation')
```

Compute row-wise similarity between two data arrays.

The array engine behind `BrainData.similarity`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>ndarray</code> | First data array, shape (n_samples1, n_features). | *required*
`data2` | <code>ndarray</code> | Second data array, shape (n_samples2, n_features). | *required*
`metric` | <code>str</code> | 'correlation' (or 'pearson'), 'spearman' (or 'rank_correlation'), 'dot_product', or 'cosine'. Defaults to 'correlation'. | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Similarities of shape (n_samples1, n_samples2), squeezed: a 1D     array when either input has a single row, a scalar when both do.

**Examples:**

```python
data1 = np.random.randn(10, 100)
data2 = np.random.randn(5, 100)
sim = compute_similarity(data1, data2, metric="correlation")
sim.shape  # → (10, 5)
```

(tasks-similarity-compute-multivariate-similarity)=
### `compute_multivariate_similarity`

```python
compute_multivariate_similarity(y, X, method = 'ols', tail = 2)
```

Compute multivariate similarity by regressing one pattern on several.

The array engine behind `BrainData.multivariate_similarity`: predicts the
spatial pattern `y` from a linear combination of the columns of `X` and
returns the OLS coefficients, t-statistics, p-values, and residuals.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` | <code>ndarray</code> | Target pattern, shape (n_features,). | *required*
`X` | <code>ndarray</code> | Predictor patterns, shape (n_features, n_predictors) (the transpose is accepted). An intercept column is always prepended, so do not include one. | *required*
`method` | <code>str</code> | Regression method; only 'ols' is implemented. Defaults to 'ols'. | <code>'ols'</code>
`tail` | <code>int</code> | 2 for two-sided p-values, 1 for an upper-tail test. Defaults to 2. | <code>2</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys 'beta' (coefficients, intercept first, shape (n_predictors + 1,)),     't' (t-statistics, same shape), 'p' (p-values, same shape), 'df'     (residual degrees of freedom), 'sigma' (residual standard deviation),     and 'residual' (residuals, shape (n_features,)).

**Examples:**

```python
y = np.random.randn(100)
X = np.random.randn(100, 5)
result = compute_multivariate_similarity(y, X, method="ols")
result["beta"].shape  # → (6,)  5 predictors + intercept
```

(tasks-similarity-transform-pairwise)=
### `transform_pairwise`

```python
transform_pairwise(X, y)
```

Transform data into pairwise differences with balanced labels for ranking.

Turns an n-class ranking problem into a two-class classification problem:
every pair of samples with different target values becomes one difference
row, and signs are flipped so that the -1 and +1 classes are balanced.

Reference: Herbrich, R., Graepel, T., & Obermayer, K. "Large Margin Rank
Boundaries for Ordinal Regression".

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray</code> | Data, shape (n_samples, n_features). | *required*
`y` | <code>ndarray</code> | Target labels, shape (n_samples,) or (n_samples, 2). A second column groups the samples; pairs from different groups are skipped. | *required*

**Returns:**

Type | Description
---- | -----------
<code>tuple[ndarray, ndarray]</code> | `(X_trans, y_trans)`. `X_trans` has shape     (k, n_features) with one row per retained pair (k is at most     n_samples * (n_samples - 1) / 2; pairs are formed within groups when     given). `y_trans` holds the labels in {-1, +1}, shape (k,), or (k, 2)     with the group in the second column when `y` had two columns.

(tasks-similarity-fisher-r-to-z)=
### `fisher_r_to_z`

```python
fisher_r_to_z(r)
```

Convert correlation coefficients to Fisher z values.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`r` | <code>float \| ndarray</code> | Correlation coefficient(s). | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Fisher z-transformed correlation(s).

(tasks-similarity-fisher-z-to-r)=
### `fisher_z_to_r`

```python
fisher_z_to_r(z)
```

Convert Fisher z back to a correlation coefficient.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`z` | <code>float \| ndarray</code> | Fisher z value(s). | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Correlation coefficient(s).

(tasks-similarity-matrix-permutation-test)=
### `matrix_permutation_test`

```python
matrix_permutation_test(data1: np.ndarray, data2: np.ndarray, *, n_permute: int = 5000, metric: str = 'pearson', how: str = 'upper', include_diag: bool = False, tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Matrix permutation test (Mantel test) for correlating two square matrices.

Tests whether the correlation between the elements of two matrices is
significant by permuting the rows and columns of one matrix together
(`data1[perm][:, perm]`) while keeping the other fixed. Each permutation
preserves the matrix's structure (including symmetry) but destroys its
relationship to `data2`; the p-value is the fraction of permuted correlations
at least as extreme as the observed one. Assumes both matrices are square and
the same size, and that row/column ordering is exchangeable under the null.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>ndarray</code> | First square matrix (n×n). | *required*
`data2` | <code>ndarray</code> | Second square matrix (n×n). | *required*
`n_permute` | <code>int</code> | Number of permutations. Defaults to 5000. | <code>5000</code>
`metric` | <code>str</code> | Correlation metric, one of 'pearson', 'spearman', or 'kendall'. Defaults to 'pearson'. | <code>'pearson'</code>
`how` | <code>str</code> | Which elements to compare: 'upper' (upper triangle; assumes symmetric matrices), 'lower' (lower triangle), or 'full' (all elements, see `include_diag`). Defaults to 'upper'. | <code>'upper'</code>
`include_diag` | <code>bool</code> | Include diagonal elements (only when `how='full'`). Defaults to False. | <code>False</code>
`tail` | <code>int \| str</code> | `2` or `'two'` for a two-tailed test (r != 0); `1` or `'one'` for a one-tailed test of r > 0 (negate one matrix for the other direction). Defaults to 2. | <code>2</code>
`return_null` | <code>bool</code> | Also return the null distribution. Defaults to False. | <code>False</code>
`device` | <code>str \| None</code> | `'cpu'` parallelizes permutations across `n_jobs` joblib workers (4-8× speedup); `None` runs single-threaded NumPy (for debugging or small problems). Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of parallel workers, -1 = all cores; only used when `device='cpu'`. Defaults to -1. | <code>-1</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | Show a progress bar over permutations. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys 'correlation' (float, observed correlation), 'p' (float,     Phipson-Smyth corrected p-value), 'device' (`'cpu'` or `None`, the     execution path used), and 'null_dist' (np.ndarray) when     `return_null=True`.

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G. et al. (2016). Untangling the relatedness among correlations,
part I: nonparametric approaches to inter-subject correlation analysis
at the group level. NeuroImage, 142, 248-259.

Mantel, N. (1967). The detection of disease clustering and a generalized
regression approach. Cancer Research, 27(2), 209-220.

</details>

**Examples:**

```python
import numpy as np
from nltools.algorithms.inference import matrix_permutation_test

# Two 20×20 similarity matrices sharing a common pattern
rng = np.random.default_rng(42)
pattern = rng.standard_normal((20, 10))
data1 = np.corrcoef(pattern + rng.standard_normal((20, 10)) * 0.5)
data2 = np.corrcoef(pattern + rng.standard_normal((20, 10)) * 0.5)

result = matrix_permutation_test(data1, data2, n_permute=1000)
print(f"Correlation: {result['correlation']:.3f}, p = {result['p']:.4f}")
```

(tasks-similarity-correlation-permutation-test)=
### `correlation_permutation_test`

```python
correlation_permutation_test(data1: np.ndarray, data2: np.ndarray, *, n_permute: int = 5000, metric: str = 'pearson', tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Permutation test for whether the correlation between two arrays differs from zero.

Builds the null distribution by randomly permuting the observations of `data1`
and re-correlating with `data2`. Assumes observations are independent (i.i.d.);
for autocorrelated time series use `timeseries_correlation_permutation_test`,
whose `'circle_shift'` and `'phase_randomize'` methods preserve temporal
structure. With 2D inputs each column of `data1` is tested against the
matching column of `data2`, independently.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>ndarray</code> | Data to permute, shape (n_samples,) for a single feature or (n_samples, n_features) for several. | *required*
`data2` | <code>ndarray</code> | Data to correlate with, same shape as `data1`. | *required*
`n_permute` | <code>int</code> | Number of permutations. Defaults to 5000. | <code>5000</code>
`metric` | <code>str</code> | 'pearson' (linear), 'spearman' (rank-based, monotonic), or 'kendall' (tau-b, ordinal association, tie-corrected). Defaults to 'pearson'. | <code>'pearson'</code>
`tail` | <code>int \| str</code> | `2` or `'two'` for a two-tailed test (r != 0); `1` or `'one'` for a one-tailed test of r > 0 (negate one variable for the other direction; the fixed direction keeps multiple-comparison correction valid). Defaults to 2. | <code>2</code>
`return_null` | <code>bool</code> | Also return the full null distribution. Defaults to False. | <code>False</code>
`device` | <code>str \| None</code> | Execution path. `'cpu'` parallelizes permutations across `n_jobs` joblib workers (4-8× speedup); `'gpu'` vectorizes them with PyTorch in memory-bounded batches (fastest for large problems; Pearson and Spearman run 5-20× faster on multi-feature data, Kendall needs O(n²) memory per permutation so its batches are smaller); `None` runs single-threaded NumPy (for debugging or small problems). Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of CPU workers, -1 = all cores; only used when `device='cpu'`. Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>float \| None</code> | GPU memory budget in GB that sizes the permutation batches; only used when `device='gpu'`. None (default) measures the device's available memory. Larger values fit more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | Show a progress bar over permutations. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys 'correlation' (float, or np.ndarray of shape (n_features,) for     2D inputs: the observed correlation), 'p' (float or np.ndarray, the     matching p-values), 'device' (the execution path used: `'cpu'`,     `'gpu'`, or `None`), and 'null_dist' (np.ndarray of shape     (n_permute,) or (n_permute, n_features)) when `return_null=True`.

**Examples:**

```python
import numpy as np
from nltools.algorithms import correlation_permutation_test

# Single feature (default: CPU parallel)
x = np.random.randn(100)
y = x + np.random.randn(100) * 0.5
result = correlation_permutation_test(x, y, n_permute=5000)
result["correlation"]  # → 0.85 (approximately)
result["p"]  # → 0.0002

# Multi-feature: each column pair tested independently
data1 = np.random.randn(100, 10)
data2 = data1 + np.random.randn(100, 10) * 0.3
result = correlation_permutation_test(data1, data2, n_permute=5000)
result["correlation"].shape  # → (10,)
result["p"].shape  # → (10,)

# GPU acceleration
result = correlation_permutation_test(data1, data2, n_permute=5000, device="gpu")
```

<details class="note" open markdown="1">
<summary>Note</summary>

Kendall's tau is O(n²) in the number of samples on every path, so it is
markedly slower than Pearson or Spearman for large samples.

</details>

(tasks-similarity-distance-correlation)=
### `distance_correlation`

```python
distance_correlation(x: np.ndarray, y: np.ndarray, bias_corrected: bool = True, ttest: bool = False) -> dict
```

Compute the distance correlation between two arrays to test for multivariate dependence.

Distance correlation detects linear and non-linear dependence. The arrays must
match on their first dimension. Prefer the bias-corrected version (the default),
which can also perform a t-test; that test operates on a statistic that is
approximately the squared distance correlation, which is also returned.

Distance correlation is the normalized covariance of two centered Euclidean
distance matrices. Each distance matrix holds the distances between rows (if x
or y is 2d) or scalars (if 1d). Each matrix is centered before the covariance
is computed, either by double-centering or by U-centering, which corrects the
bias that grows with the number of dimensions. U-centering is almost always
preferable and also permits a one-tailed directional t-test on the normalized
covariance (Szekely & Rizzo, 2013). Distance correlation is normally bounded
between 0 and 1, but U-centering can produce negative estimates, which are
never significant.

Validated against `dcor` and `dcor.ttest` in the R package *energy* and
`dcor.distance_correlation`, `dcor.u_distance_correlation_sqr`, and
`dcor.independence.distance_correlation_t_test` in the Python package *dcor*.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` | <code>ndarray</code> | 1d or 2d array of observations by features. | *required*
`y` | <code>ndarray</code> | 1d or 2d array of observations by features. | *required*
`bias_corrected` | <code>bool</code> | If True, U-center the distance matrices; if False, double-center them, which gives a biased estimate that converges to 1 as the number of dimensions grows. Must be True when `ttest=True`. Defaults to True. | <code>True</code>
`ttest` | <code>bool</code> | Perform a t-test on the bias-corrected distance correlation. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Key 'dcorr' (float, distance correlation); with `bias_corrected=True`     also 'dcorr_squared' (float, the U-centered statistic, which can be     negative); with `ttest=True` also 't', 'p', and 'df'.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If arrays are not 1d or 2d, or if `ttest=True` and `bias_corrected=False`.

**Examples:**

```python
import numpy as np

x = np.random.randn(20, 3)
y = x + np.random.randn(20, 3) * 0.1  # strongly dependent
result = distance_correlation(x, y, bias_corrected=True)
"dcorr" in result  # → True
0 <= result["dcorr"] <= 1  # → True
```

(tasks-similarity-double-center)=
### `double_center`

```python
double_center(mat: np.ndarray) -> np.ndarray
```

Double center a 2d array.

Double-centering subtracts row means, column means, and adds the grand mean.
This centers both rows and columns around zero.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mat` | <code>ndarray</code> | 2d numpy array. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Double-centered version of the input.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If input is not 2D.

**Examples:**

```python
mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
result = double_center(mat)
np.allclose(result.mean(axis=0), 0)  # → True
np.allclose(result.mean(axis=1), 0)  # → True
```

(tasks-similarity-u-center)=
### `u_center`

```python
u_center(mat: np.ndarray) -> np.ndarray
```

U-center a 2d array.

U-centering is a bias-corrected form of double-centering: it corrects for the
bias that grows with the number of dimensions under plain double-centering.
The diagonal is explicitly set to zero.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mat` | <code>ndarray</code> | 2d numpy array. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | U-centered version of the input.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If input is not 2D.

**Examples:**

```python
mat = np.random.randn(5, 5)
result = u_center(mat)
np.allclose(np.diag(result), 0)  # → True
```

(tasks-similarity-plot-stacked-adjacency)=
### `plot_stacked_adjacency`

```python
plot_stacked_adjacency(adjacency1, adjacency2, normalize = True, **kwargs)
```

Create stacked adjacency to illustrate similarity.

`adjacency1` is drawn in the upper triangle and `adjacency2` in the lower,
consistently whether or not `normalize` is set.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adjacency1` | <code>[Adjacency](#page-data-adjacency)</code> | Adjacency instance shown in the upper triangle. | *required*
`adjacency2` | <code>[Adjacency](#page-data-adjacency)</code> | Adjacency instance shown in the lower triangle. | *required*
`normalize` | <code>bool</code> | Normalize matrices before stacking. Default True. | <code>True</code>
`**kwargs` | <code>dict</code> | Forwarded to `seaborn.heatmap`. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>Axes</code> | Axes holding the stacked heatmap.

(tasks-similarity-plot-mean-label-distance)=
### `plot_mean_label_distance`

```python
plot_mean_label_distance(distance, labels, *, ax = None, permutation_test = False, n_permute = 5000, fontsize = 18, **kwargs)
```

Violin plot of within- vs between-label distances.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`distance` | <code>ndarray \| DataFrame \| DataFrame</code> | Square pairwise distance matrix. | *required*
`labels` | <code>array - like</code> | Group label for each row/column (length N). | *required*
`ax` | <code>Axes</code> | Axis to draw on. | <code>None</code>
`permutation_test` | <code>bool</code> | If True, run a two-sample permutation test per group. Default False. | <code>False</code>
`n_permute` | <code>int</code> | Number of permutations. Default 5000. | <code>5000</code>
`fontsize` | <code>int</code> | Font size for the axis label and title. Default 18. | <code>18</code>
`**kwargs` | <code>dict</code> | Forwarded to `seaborn.violinplot`. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>DataFrame \| tuple[DataFrame, dict]</code> | A long-format frame with columns     `Distance`, `Type`, `Group`. If `permutation_test=True`, a tuple     `(long_df, stats)` where `stats` maps each group label to its     permutation-test result.

(tasks-similarity-plot-between-label-distance)=
### `plot_between_label_distance`

```python
plot_between_label_distance(distance, labels, *, ax = None, permutation_test = True, n_permute = 5000, **kwargs)
```

Heatmap of average pairwise distance between every label pair.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`distance` | <code>ndarray \| DataFrame \| DataFrame</code> | Square pairwise distance matrix. | *required*
`labels` | <code>array - like</code> | Group label for each row/column (length N). | *required*
`ax` | <code>Axes</code> | Axis to draw on. | <code>None</code>
`permutation_test` | <code>bool</code> | If True, also compute mean-difference and p-value matrices. Default True. | <code>True</code>
`n_permute` | <code>int</code> | Number of permutations. Default 5000. | <code>5000</code>
`**kwargs` | <code>dict</code> | Forwarded to `seaborn.heatmap`. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>tuple[DataFrame, ...]</code> | `(long_df, within_mean_df)` without     `permutation_test`, or `(long_df, within_mean_df, mean_diff_df, p_df)`     with it. All frames are polars DataFrames. `long_df` has columns     `Distance`, `Group`, `Comparison`. The three square-matrix-like frames are     long format with columns `label1`, `label2`, and a value column so they     can be pivoted to a matrix if needed.

(tasks-similarity-plot-silhouette)=
### `plot_silhouette`

```python
plot_silhouette(distance, labels, *, ax = None, permutation_test = True, n_permute = 5000, colors = None, figsize = (6, 4))
```

Silhouette plot indicating between- vs within-label distance.

Uses the simplified silhouette definition from the original nltools
implementation: within(i) = mean distance to other points in the same
cluster; between(i) = mean distance to all points in other clusters
(not the strict Rousseeuw min-over-clusters). Score is
(between - within) / max(between, within).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`distance` | <code>ndarray \| DataFrame \| DataFrame</code> | Square pairwise distance matrix. | *required*
`labels` | <code>array - like</code> | Cluster label for each row/column (length N). | *required*
`ax` | <code>Axes</code> | Axis to draw on. | <code>None</code>
`permutation_test` | <code>bool</code> | If True, run a one-sample permutation test per cluster on positive-mean silhouette scores. Default True. | <code>True</code>
`n_permute` | <code>int</code> | Number of permutations. Default 5000. | <code>5000</code>
`colors` | <code>list</code> | RGB triplets, one per cluster. Default: seaborn `'hls'` palette. | <code>None</code>
`figsize` | <code>tuple</code> | Figure size. Default (6, 4). | <code>(6, 4)</code>

**Returns:**

Type | Description
---- | -----------
<code>DataFrame</code> | Frame with columns `label` and `mean_silhouette`. If     `permutation_test` is True, adds a `p` column (1.0 for clusters with     non-positive mean).
