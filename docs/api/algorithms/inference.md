---
title: algorithms.inference
label: page-algorithms-inference
---

Permutation tests, bootstrap resampling, and intersubject statistics.

Every test here runs on plain numpy arrays and returns a dict of results. The
one-, two-sample, correlation, matrix, and timeseries permutation tests share
one execution model: `device=None` runs single-threaded numpy, `device='cpu'`
(the default) parallelizes permutation batches with joblib across `n_jobs`
workers, and `device='gpu'` batches permutations through PyTorch (10-100×
faster for voxel-wise problems with many permutations). The intersubject
statistics (`isc`, `isc_group`, `isfc`, `isps` in `nltools.algorithms`) are
built on the same engine.

**Classes:**

Name | Description
---- | -----------
[`OnlineBootstrapStats`](#algorithms-inference-onlinebootstrapstats) | Memory-efficient online statistics aggregator for bootstrap samples.

**Functions:**

Name | Description
---- | -----------
[`circle_shift`](#algorithms-inference-circle-shift) | Circular shift for time-series data.
[`correlation_permutation_test`](#algorithms-inference-correlation-permutation-test) | Permutation test for whether the correlation between two arrays differs from zero.
[`distance_correlation`](#algorithms-inference-distance-correlation) | Compute the distance correlation between two arrays to test for multivariate dependence.
[`double_center`](#algorithms-inference-double-center) | Double center a 2d array.
[`isc_group_permutation_test`](#algorithms-inference-isc-group-permutation-test) | Test the difference in intersubject correlation between two groups.
[`isc_permutation_test`](#algorithms-inference-isc-permutation-test) | Compute intersubject correlation with bootstrap or permutation inference.
[`matrix_permutation_test`](#algorithms-inference-matrix-permutation-test) | Matrix permutation test (Mantel test) for correlating two square matrices.
[`one_sample_permutation_test`](#algorithms-inference-one-sample-permutation-test) | One-sample permutation test using sign flipping.
[`phase_randomize`](#algorithms-inference-phase-randomize) | FFT-based phase randomization for time-series data.
[`timeseries_correlation_permutation_test`](#algorithms-inference-timeseries-correlation-permutation-test) | Permutation test for the correlation between two autocorrelated time series.
[`two_sample_permutation_test`](#algorithms-inference-two-sample-permutation-test) | Two-sample permutation test using group-label shuffling.
[`u_center`](#algorithms-inference-u-center) | U-center a 2d array.



**Examples:**

```python
import numpy as np
from nltools.algorithms.inference import one_sample_permutation_test

data = np.random.randn(30)  # 30 subjects
result = one_sample_permutation_test(data, n_permute=5000)
result["p"]  # → two-sided p-value

# Voxel-wise test on the GPU
data = np.random.randn(30, 50000)  # 30 subjects, 50K voxels
result = one_sample_permutation_test(data, n_permute=10000, device="gpu")
(result["p"] < 0.05).sum()  # → number of significant voxels
```

<details class="references" open markdown="1">
<summary>References</summary>

Eklund, A., Dufort, P., Villani, M., & LaConte, S. M. (2014).
BROCCOLI: Software for fast fMRI analysis on many-core CPUs and GPUs.
Frontiers in Neuroinformatics, 8, 24.

</details>

<details class="note" open markdown="1">
<summary>Note</summary>

These are the functional core. The data classes wrap them —
`BrainData.ttest`, `BrainData.bootstrap`, `Adjacency.ttest` — and handle
masking and result reshaping for you.

</details>

## Classes

(algorithms-inference-onlinebootstrapstats)=
### `OnlineBootstrapStats`

```python
OnlineBootstrapStats(shape: tuple[int, ...], save_samples: bool = False, percentiles: tuple[float, float] = (2.5, 97.5))
```

Memory-efficient online statistics aggregator for bootstrap samples.

Accumulates the running mean and variance with Welford's algorithm, so the
summary is numerically stable without holding every sample in memory.
Optionally stores all samples for exact percentile confidence intervals.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`shape` | <code>tuple[int, ...]</code> | Shape of each bootstrap sample. | *required*
`save_samples` | <code>bool</code> | If True, store all samples for exact percentile confidence intervals; if False, use the normal approximation (much more memory efficient). Defaults to False. | <code>False</code>
`percentiles` | <code>tuple[float, float]</code> | Percentiles for confidence intervals, e.g. (2.5, 97.5) for a 95% CI. Defaults to (2.5, 97.5). | <code>(2.5, 97.5)</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`n` | <code>int</code> | Number of samples seen so far.
`mean` | <code>ndarray</code> | Running mean, shape `shape`.
`M2` | <code>ndarray</code> | Running sum of squared deviations from the mean.
`samples` | <code>list[ndarray] \| None</code> | Stored samples when `save_samples=True`, else None.

**Methods:**

Name | Description
---- | -----------
[`get_results`](#algorithms-inference-get-results) | Compute final bootstrap statistics.
[`update`](#algorithms-inference-update) | Fold one bootstrap sample into the running statistics.



**Examples:**

```python
stats = OnlineBootstrapStats(shape=(100,), save_samples=False)
for _ in range(1000):
    stats.update(np.random.randn(100))
results = stats.get_results()
results.keys()  # → dict_keys(['mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper'])
```

#### Methods

(algorithms-inference-get-results)=
##### `get_results`

```python
get_results(tail: int | str = 2) -> dict[str, np.ndarray]
```

Compute final bootstrap statistics.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`tail` | <code>int \| str</code> | `2` or `'two'` for two-tailed (default); `1` or `'one'` for one-tailed (statistic > 0; negate the data for the other direction). | <code>2</code>

**Returns:**

Type | Description
---- | -----------
<code>dict[str, ndarray]</code> | Keys 'mean' (bootstrap mean), 'std' (bootstrap     standard deviation), 'Z' (z-scores, mean/std), 'p' (p-values per     `tail`), 'ci_lower' and 'ci_upper' (confidence bounds; exact     percentiles when samples were saved, else a normal approximation),     and 'samples' (all samples, only when `save_samples=True`).

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If fewer than 2 samples have been seen.

**Examples:**

```python
stats = OnlineBootstrapStats(shape=(100,), save_samples=False)
for _ in range(1000):
    stats.update(np.random.randn(100))
results = stats.get_results()
results.keys()  # → dict_keys(['mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper'])
```

(algorithms-inference-update)=
##### `update`

```python
update(sample: np.ndarray) -> None
```

Fold one bootstrap sample into the running statistics.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sample` | <code>ndarray</code> | New bootstrap sample with shape matching `self.shape`. | *required*

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If the sample's shape does not match `self.shape`.



## Functions

(algorithms-inference-circle-shift)=
### `circle_shift`

```python
circle_shift(data: np.ndarray, shift_amount: int | np.ndarray | None = None, random_state: int | np.random.RandomState | None = None) -> np.ndarray
```

Circular shift for time-series data.

Performs a circular shift that preserves autocorrelation structure.
Useful for permutation tests on autocorrelated time series (e.g., fMRI).
For 1D data, shifts by a single amount. For 2D data, shifts each
feature (column) independently.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>ndarray</code> | Time series, shape (n_samples,) or (n_samples, n_features). | *required*
`shift_amount` | <code>int \| ndarray \| None</code> | Shift amount: an int for 1D data, or an array of length n_features (one shift per column) for 2D data. None draws random shift(s). Defaults to None. | <code>None</code>
`random_state` | <code>int \| RandomState \| None</code> | Random seed used when `shift_amount` is None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Circularly shifted data with the same shape as the input.

**Examples:**

```python
x = np.array([1, 2, 3, 4, 5])
circle_shift(x, shift_amount=2)  # → array([4, 5, 1, 2, 3])

X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
circle_shift(X, shift_amount=np.array([1, 2]))
# → array([[ 4, 30],
#          [ 1, 40],
#          [ 2, 10],
#          [ 3, 20]])
```

(algorithms-inference-correlation-permutation-test)=
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

(algorithms-inference-distance-correlation)=
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

(algorithms-inference-double-center)=
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

(algorithms-inference-isc-group-permutation-test)=
### `isc_group_permutation_test`

```python
isc_group_permutation_test(group1: np.ndarray, group2: np.ndarray, *, n_permute: int = 5000, summary: Literal['median', 'mean'] = 'median', method: Literal['permute', 'bootstrap'] = 'permute', summary_statistic: Literal['leave-one-out', 'pairwise'] = 'pairwise', ci_percentile: float = 95, tail: int | str = 2, device: Literal['cpu', 'gpu'] | None = 'cpu', n_jobs: int = -1, random_state: int | None = None, return_null: bool = False, progress_bar: bool = False, exclude_self_corr: bool = True, metric: str = 'correlation') -> dict[str, Any]
```

Test the difference in intersubject correlation between two groups.

Computes ISC within each group, takes `group1 - group2`, and builds a null
distribution by either subject-wise permutation (pool the subjects and
reshuffle the group labels — the Chen et al. 2016 recommendation) or
subject-wise bootstrap (resample subjects within each group; the bootstrap
draws are centered on the observed difference before the p-value is
computed). The confidence interval brackets the observed difference for
`method='bootstrap'` and describes the null spread for `method='permute'`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`group1` | <code>ndarray</code> | First group, shape `(n_observations, n_subjects1)` for a single feature or `(n_observations, n_subjects1, n_voxels)` for voxel-wise data. | *required*
`group2` | <code>ndarray</code> | Second group, shape `(n_observations, n_subjects2)` or `(n_observations, n_subjects2, n_voxels)`; `n_observations` must match `group1`. | *required*
`n_permute` | <code>int</code> | Number of permutations or bootstrap draws. Defaults to 5000. | <code>5000</code>
`summary` | <code>str</code> | How ISC values are aggregated: `'median'` (default, robust to outliers) or `'mean'` (Fisher z-transformed mean). | <code>'median'</code>
`method` | <code>str</code> | `'permute'` (default; pool subjects and permute labels) or `'bootstrap'` (resample subjects within each group). | <code>'permute'</code>
`summary_statistic` | <code>str</code> | `'pairwise'` (default; summarize all pairwise correlations) or `'leave-one-out'` (correlate each subject with the mean of the others). | <code>'pairwise'</code>
`ci_percentile` | <code>float</code> | Confidence-interval width in percent (95 gives a 95% CI). Defaults to 95. | <code>95</code>
`tail` | <code>int \| str</code> | `2` or `'two'` (default) for a two-tailed p-value; `1` or `'one'` for one-tailed (group1 > group2). | <code>2</code>
`device` | <code>str \| None</code> | Execution path. `'cpu'` (default) parallelizes the resamples with joblib; `'gpu'` computes the observed voxel-wise ISC through PyTorch (10-30× speedup; the resamples still run on the CPU); None runs single-threaded numpy. | <code>'cpu'</code>
`n_jobs` | <code>int</code> | CPU cores for the resamples when `device` is not None. -1 (default) picks the worker count from available memory. | <code>-1</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>bool</code> | If True, include the null distribution in the result. Defaults to False. | <code>False</code>
`progress_bar` | <code>bool</code> | Show a progress bar over the resamples. Defaults to False. | <code>False</code>
`exclude_self_corr` | <code>bool</code> | In the bootstrap, mask the perfect correlations a duplicated subject produces (pairwise only). Defaults to True. | <code>True</code>
`metric` | <code>str</code> | Similarity metric for pairwise ISC; any metric accepted by `sklearn.metrics.pairwise_distances`. Ignored for `summary_statistic='leave-one-out'`. Defaults to `'correlation'`. | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'isc_group_difference'` (float or np.ndarray, observed     difference), `'p'` (float or np.ndarray, p-value with the     `(count + 1) / (n + 1)` correction), `'ci'` (tuple     `(lower, upper)`), `'device'` (the execution path used), and — when     `return_null=True` — `'null_dist'` (np.ndarray).

**Examples:**

```python
# Single-feature comparison
group1 = np.random.randn(100, 10)  # 10 subjects
group2 = np.random.randn(100, 10)
result = isc_group_permutation_test(group1, group2, n_permute=1000)
result["isc_group_difference"], result["p"]

# Voxel-wise comparison, observed ISC on the GPU
group1_voxels = np.random.randn(100, 10, 5000)  # 5K voxels
group2_voxels = np.random.randn(100, 10, 5000)
result = isc_group_permutation_test(
    group1_voxels,
    group2_voxels,
    summary_statistic="leave-one-out",
    device="gpu",
    n_permute=5000,
)
(result["p"] < 0.05).sum()  # → number of significant voxels
```

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

(algorithms-inference-isc-permutation-test)=
### `isc_permutation_test`

```python
isc_permutation_test(data: np.ndarray, *, n_permute: int = 5000, summary: Literal['median', 'mean'] = 'median', summary_statistic: Literal['leave-one-out', 'pairwise'] = 'pairwise', method: Literal['bootstrap', 'circle_shift', 'phase_randomize'] = 'bootstrap', ci_percentile: float = 95, tail: int | str = 2, return_null: bool = False, progress_bar: bool = False, exclude_self_corr: bool = True, metric: str = 'correlation', device: Literal['cpu', 'gpu'] | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> dict[str, Any]
```

Compute intersubject correlation with bootstrap or permutation inference.

Summarizes how similarly subjects respond over time — either leave-one-out
(each subject against the mean of the others; O(n_subjects), unbiased) or
pairwise (all subject pairs; O(n_subjects²), full correlation structure).
The two are monotonically but non-linearly related and statistically
different (Chen et al. 2016, Figure 3). The null distribution comes from a
subject-wise bootstrap (centered on the observed ISC, so the p-value tests
H0: ISC = 0) or from surrogate time series that preserve each subject's
autocorrelation (circular shift) or power spectrum (phase randomization).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>ndarray</code> | Shape `(n_observations, n_subjects)` for a single feature or `(n_observations, n_subjects, n_voxels)` for voxel-wise ISC. | *required*
`n_permute` | <code>int</code> | Number of bootstrap draws or permutations. Defaults to 5000. | <code>5000</code>
`summary` | <code>str</code> | How ISC values are aggregated: `'median'` (default, robust to outliers) or `'mean'` (Fisher z-transformed mean). | <code>'median'</code>
`summary_statistic` | <code>str</code> | `'pairwise'` (default) or `'leave-one-out'`. | <code>'pairwise'</code>
`method` | <code>str</code> | `'bootstrap'` (default; subject-wise bootstrap, Chen et al. 2016), `'circle_shift'` (circular time-series shift), or `'phase_randomize'` (FFT phase randomization). | <code>'bootstrap'</code>
`ci_percentile` | <code>float</code> | Confidence-interval width in percent (95 gives a 95% CI). Defaults to 95. | <code>95</code>
`tail` | <code>int \| str</code> | `2` or `'two'` (default) for a two-tailed p-value; `1` or `'one'` for one-tailed (ISC > 0). | <code>2</code>
`return_null` | <code>bool</code> | If True, include the null distribution in the result. Defaults to False. | <code>False</code>
`progress_bar` | <code>bool</code> | Show a progress bar over the resamples. Defaults to False. | <code>False</code>
`exclude_self_corr` | <code>bool</code> | In the pairwise bootstrap, mask the perfect correlations a duplicated subject produces as NaN. Defaults to True. | <code>True</code>
`metric` | <code>str</code> | Similarity metric for pairwise ISC; any metric accepted by `sklearn.metrics.pairwise_distances` (`'correlation'`, `'spearman'`, `'cosine'`, and `'euclidean'` take fast paths). Ignored for `summary_statistic='leave-one-out'`; the GPU pairwise path supports only `'correlation'`. Defaults to `'correlation'`. | <code>'correlation'</code>
`device` | <code>str \| None</code> | Execution path. `'cpu'` (default) parallelizes the resamples with joblib; `'gpu'` computes voxel-wise ISC through PyTorch (10-30× speedup) and, for the pairwise bootstrap, runs the resamples on the device too; None runs single-threaded numpy. | <code>'cpu'</code>
`n_jobs` | <code>int</code> | CPU cores for the resamples when `device` is not None. -1 (default) picks the worker count from available memory. | <code>-1</code>
`max_gpu_memory_gb` | <code>float \| None</code> | GPU working-set budget in GB for the pairwise GPU bootstrap (`device='gpu'`, `summary_statistic='pairwise'`, `method='bootstrap'`): bounds the `(perm_batch, voxel_chunk, n_subjects, n_subjects)` resample tensor, chunking voxels and permutations to fit. Not used by the leave-one-out or surrogate paths. None (default) measures the device. | <code>None</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'isc'` (float or np.ndarray, observed ISC), `'p'` (float or     np.ndarray, p-value with the `(count + 1) / (n + 1)` correction),     `'ci'` (tuple `(lower, upper)` percentiles of the resamples),     `'device'` (the execution path used), and — when     `return_null=True` — `'null_dist'` (np.ndarray).

**Examples:**

```python
# Single-feature ISC
data = np.random.randn(100, 10)  # 100 timepoints, 10 subjects
result = isc_permutation_test(data, n_permute=1000)
result["isc"], result["p"]

# Voxel-wise leave-one-out ISC on the GPU
data_voxels = np.random.randn(100, 50, 5000)  # 5K voxels
result = isc_permutation_test(
    data_voxels,
    summary_statistic="leave-one-out",
    device="gpu",
    n_permute=5000,
)
(result["p"] < 0.05).sum()  # → number of significant voxels

# Leave-one-out vs pairwise
result_loo = isc_permutation_test(data, summary_statistic="leave-one-out")
result_pair = isc_permutation_test(data, summary_statistic="pairwise")
```

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

(algorithms-inference-matrix-permutation-test)=
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

(algorithms-inference-one-sample-permutation-test)=
### `one_sample_permutation_test`

```python
one_sample_permutation_test(data: np.ndarray, *, n_permute: int = 5000, tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None, progress_bar: bool = False) -> dict
```

One-sample permutation test using sign flipping.

Tests whether the mean of `data` differs from zero by randomly flipping the
sign of each observation — the permutation analogue of a one-sample t-test.
Multi-feature (voxel-wise) data tests each column independently against
the same permutations.

Assumes errors are distributed symmetrically around zero. For strongly
skewed data, prefer bootstrap resampling.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>ndarray</code> | Data to test, shape `(n_samples,)` for a single feature or `(n_samples, n_features)` for voxel-wise data. | *required*
`n_permute` | <code>int</code> | Number of permutations. Defaults to 5000. | <code>5000</code>
`tail` | <code>int \| str</code> | `2` or `'two'` (default) for a two-tailed test (mean != 0); `1` or `'one'` for a one-tailed test of mean > 0 (negate the data for the other direction — the fixed direction keeps multiple-comparison correction valid). | <code>2</code>
`return_null` | <code>bool</code> | If True, include the full null distribution in the result. Defaults to False. | <code>False</code>
`device` | <code>str \| None</code> | Execution path. `'cpu'` (default) parallelizes with joblib across `n_jobs` cores (4-8× speedup); `'gpu'` batches permutations through PyTorch (fastest for large problems); None runs single-threaded numpy (small problems, debugging). | <code>'cpu'</code>
`n_jobs` | <code>int</code> | CPU cores for `device='cpu'`. Defaults to -1 (all cores). | <code>-1</code>
`max_gpu_memory_gb` | <code>float \| None</code> | GPU memory budget in GB for `device='gpu'`; controls automatic batching. None (default) measures the device's available memory. | <code>None</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | Whether to display a progress bar. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'mean'` (float or np.ndarray, observed mean(s)), `'p'`     (float or np.ndarray, p-value(s)), `'device'` (the execution path     used), and — when `return_null=True` — `'null_dist'` (np.ndarray,     shape `(n_permute,)` or `(n_permute, n_features)`).

**Examples:**

```python
# Single feature (default CPU parallelization)
data = np.random.randn(30)
result = one_sample_permutation_test(data, n_permute=5000)
result["p"]  # → 0.23

# Voxel-wise test on the GPU
data = np.random.randn(30, 10000)  # 30 subjects, 10K voxels
result = one_sample_permutation_test(data, n_permute=5000, device="gpu")
result["mean"].shape  # → (10000,)
result["p"].shape  # → (10000,)

# Single-threaded (for debugging)
result = one_sample_permutation_test(data, n_permute=5000, device=None)
```

(algorithms-inference-phase-randomize)=
### `phase_randomize`

```python
phase_randomize(data: np.ndarray, *, device: str | None = 'cpu', random_state: int | np.random.RandomState | None = None) -> np.ndarray
```

FFT-based phase randomization for time-series data.

Preserves the power spectrum (and therefore the autocorrelation) exactly, up
to numerical precision, while destroying nonlinear temporal structure. Used to
test whether data was generated by a linear Gaussian process or contains
nonlinear dynamics.

The signal is transformed with an FFT, each positive frequency is multiplied
by `exp(iφ)` with φ drawn uniformly from [0, 2π], the matching negative
frequency by the conjugate `exp(-iφ)` so the inverse FFT is real, and the
result is transformed back.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>ndarray</code> | Time series, shape (n_samples,) or (n_samples, n_features). | *required*
`device` | <code>str \| None</code> | `'cpu'` or None runs NumPy's FFT in float64; `'gpu'` runs PyTorch's FFT on CUDA/MPS in float32 (5-20× faster for large data); `'auto'` uses a GPU if present, else CPU. Defaults to 'cpu'. | <code>'cpu'</code>
`random_state` | <code>int \| RandomState \| None</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Phase-randomized data with the same shape as the input.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If `device` is not None, 'cpu', 'gpu', or 'auto'. An explicit `'gpu'` runs on the GPU or raises; it never silently falls back to CPU.

**Examples:**

```python
x = np.sin(np.linspace(0, 10 * np.pi, 100))
x_rand = phase_randomize(x, random_state=42)
# Power spectrum is preserved
np.allclose(np.abs(np.fft.rfft(x)) ** 2, np.abs(np.fft.rfft(x_rand)) ** 2)  # → True

# GPU acceleration for large data
x_large = np.random.randn(10000)
x_rand_gpu = phase_randomize(x_large, device="gpu", random_state=42)
```

(algorithms-inference-timeseries-correlation-permutation-test)=
### `timeseries_correlation_permutation_test`

```python
timeseries_correlation_permutation_test(data1: np.ndarray, data2: np.ndarray, *, method: Literal['circle_shift', 'phase_randomize'] = 'circle_shift', n_permute: int = 5000, metric: Literal['pearson', 'spearman', 'kendall'] = 'pearson', tail: int | str = 2, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, return_null: bool = False, random_state: int | np.random.RandomState | None = None, progress_bar: bool = False) -> dict
```

Permutation test for the correlation between two autocorrelated time series.

Standard permutation tests shuffle samples independently, which destroys
autocorrelation and inflates Type I error on time series. This test instead
builds the null from surrogates of `data1` that preserve temporal structure
— circular shifts (`'circle_shift'`) or Fourier phase randomization
(`'phase_randomize'`) — while `data2` stays fixed. For independent
observations use `correlation_permutation_test`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>ndarray</code> | First time series, shape (n_samples,) or (n_samples, 1). | *required*
`data2` | <code>ndarray</code> | Second time series, shape (n_samples,) or (n_samples, 1). | *required*
`method` | <code>str</code> | `'circle_shift'` rotates the series (preserves autocorrelation; fast, and suitable for most fMRI time series); `'phase_randomize'` randomizes Fourier phases (preserves the power spectrum exactly; tests for nonlinear structure). Defaults to 'circle_shift'. | <code>'circle_shift'</code>
`n_permute` | <code>int</code> | Number of permutations. Defaults to 5000. | <code>5000</code>
`metric` | <code>str</code> | Correlation type, one of 'pearson', 'spearman', or 'kendall'. Defaults to 'pearson'. | <code>'pearson'</code>
`tail` | <code>int \| str</code> | `2` or `'two'` for a two-tailed test; `1` or `'one'` for a one-tailed test in the positive direction (negate one series for the other direction). Defaults to 2. | <code>2</code>
`device` | <code>str \| None</code> | Execution path. `'cpu'` parallelizes permutations across `n_jobs` joblib workers (4-8× speedup); `'gpu'` generates surrogates with PyTorch in memory-bounded batches (5-20× faster for n_samples > 1000; `'phase_randomize'` benefits most from the GPU FFT); `None` runs single-threaded NumPy (for debugging or small problems). Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of CPU workers, -1 = all cores; only used when `device='cpu'`. Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>float \| None</code> | GPU memory budget in GB that sizes the permutation batches; only used when `device='gpu'`. None (default) measures the device's available memory. Larger values fit more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`return_null` | <code>bool</code> | Also return the null distribution. Defaults to False. | <code>False</code>
`random_state` | <code>int \| RandomState \| None</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | Show a progress bar over permutations. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys 'correlation' (float, observed correlation), 'p' (float),     'device' (the execution path used: `'cpu'`, `'gpu'`, or `None`), and     'null_dist' (np.ndarray of shape (n_permute,)) when     `return_null=True`.

**Examples:**

```python
import numpy as np
from nltools.algorithms import timeseries_correlation_permutation_test

rng = np.random.default_rng(0)
x = np.sin(np.linspace(0, 10 * np.pi, 100))  # strongly autocorrelated
y = x + rng.standard_normal(100) * 0.5
result = timeseries_correlation_permutation_test(
    x, y, method="circle_shift", n_permute=1000, random_state=42
)
result["correlation"]  # → 0.853
result["p"]  # → 0.078 — the autocorrelation-aware null is far wider
#   than a sample-shuffling null would be

# GPU acceleration
result = timeseries_correlation_permutation_test(
    x, y, method="phase_randomize", device="gpu", n_permute=5000
)
```

(algorithms-inference-two-sample-permutation-test)=
### `two_sample_permutation_test`

```python
two_sample_permutation_test(data1: np.ndarray, data2: np.ndarray, *, n_permute: int = 5000, tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Two-sample permutation test using group-label shuffling.

Tests whether two independent groups have different means by randomly
reassigning observations to groups — the permutation analogue of an
independent-samples t-test. Group sizes may differ. Multi-feature
(voxel-wise) data tests each column independently against the same
permutations.

Assumes exchangeability under the null (group assignment is arbitrary):
independent samples from similarly shaped distributions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>ndarray</code> | Group 1 data, shape `(n_samples1,)` for a single feature or `(n_samples1, n_features)` for voxel-wise data. | *required*
`data2` | <code>ndarray</code> | Group 2 data, shape `(n_samples2,)` or `(n_samples2, n_features)`; must have the same number of features as `data1`. | *required*
`n_permute` | <code>int</code> | Number of permutations. Defaults to 5000. | <code>5000</code>
`tail` | <code>int \| str</code> | `2` or `'two'` (default) for a two-tailed test (mean1 != mean2); `1` or `'one'` for a one-tailed test of mean1 > mean2 (swap the groups for the other direction — the fixed direction keeps multiple-comparison correction valid). | <code>2</code>
`return_null` | <code>bool</code> | If True, include the full null distribution in the result. Defaults to False. | <code>False</code>
`device` | <code>str \| None</code> | Execution path. `'cpu'` (default) parallelizes with joblib across `n_jobs` cores (4-8× speedup); `'gpu'` batches permutations through PyTorch (fastest for large problems); None runs single-threaded numpy (small problems, debugging). | <code>'cpu'</code>
`n_jobs` | <code>int</code> | CPU cores for `device='cpu'`. Defaults to -1 (all cores). | <code>-1</code>
`max_gpu_memory_gb` | <code>float \| None</code> | GPU memory budget in GB for `device='gpu'`; controls automatic batching. None (default) measures the device's available memory. | <code>None</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | Whether to display a progress bar. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'mean_diff'` (float or np.ndarray, observed     `mean(data1) - mean(data2)`), `'p'` (float or np.ndarray,     p-value(s)), `'device'` (the execution path used), and — when     `return_null=True` — `'null_dist'` (np.ndarray, shape     `(n_permute,)` or `(n_permute, n_features)`).

**Examples:**

```python
# Single feature (default CPU parallelization)
data1 = np.random.randn(20)  # Group 1: 20 subjects
data2 = np.random.randn(25)  # Group 2: 25 subjects
result = two_sample_permutation_test(data1, data2, n_permute=5000)
result["p"]  # → 0.45

# Voxel-wise test on the GPU
data1 = np.random.randn(20, 10000)  # 20 subjects, 10K voxels
data2 = np.random.randn(25, 10000)  # 25 subjects, 10K voxels
result = two_sample_permutation_test(data1, data2, n_permute=5000, device="gpu")
result["mean_diff"].shape  # → (10000,)
result["p"].shape  # → (10000,)

# Single-threaded (for debugging)
result = two_sample_permutation_test(data1, data2, n_permute=5000, device=None)
```

(algorithms-inference-u-center)=
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
