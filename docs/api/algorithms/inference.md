---
title: algorithms.inference
---

GPU-accelerated statistical inference for neuroimaging.

This module provides fast permutation testing and bootstrap resampling using
optional GPU acceleration via PyTorch. When GPU is unavailable, efficiently
uses CPU parallelization.

Inspired by BROCCOLI's GPU permutation testing (Eklund et al. 2014).

<details class="key-features" open markdown="1">
<summary>Key Features</summary>

- 10-100× speedup for permutation tests with GPU
- Efficient CPU parallelization when GPU unavailable
- Transparent CPU/GPU support via Backend abstraction
- Intersubject statistics (`isc`, `isc_group`, `isfc`, `isps`) built on the
  same permutation/bootstrap engine

</details>

**Classes:**

Name | Description
---- | -----------
[`OnlineBootstrapStats`](#algorithms-inference-onlinebootstrapstats) | Memory-efficient online statistics aggregator for bootstrap samples.

**Methods:**

Name | Description
---- | -----------
[`circle_shift`](#algorithms-inference-circle-shift) | Circular shift for time-series data.
[`correlation_permutation_test`](#algorithms-inference-correlation-permutation-test) | Correlation permutation test.
[`distance_correlation`](#algorithms-inference-distance-correlation) | Compute the distance correlation between 2 arrays to test for multivariate dependence (linear or non-linear).
[`double_center`](#algorithms-inference-double-center) | Double center a 2d array.
[`isc_group_permutation_test`](#algorithms-inference-isc-group-permutation-test) | Compute ISC difference between groups with permutation testing.
[`isc_permutation_test`](#algorithms-inference-isc-permutation-test) | Compute intersubject correlation with permutation testing.
[`matrix_permutation_test`](#algorithms-inference-matrix-permutation-test) | Matrix permutation test (Mantel test) for correlating two square matrices.
[`one_sample_permutation_test`](#algorithms-inference-one-sample-permutation-test) | One-sample permutation test using sign-flipping.
[`phase_randomize`](#algorithms-inference-phase-randomize) | FFT-based phase randomization for time-series data.
[`timeseries_correlation_permutation_test`](#algorithms-inference-timeseries-correlation-permutation-test) | Time-series correlation permutation test.
[`two_sample_permutation_test`](#algorithms-inference-two-sample-permutation-test) | Two-sample permutation test using group label shuffling.
[`u_center`](#algorithms-inference-u-center) | U-center a 2d array.



**Examples:**

```pycon
>>> import numpy as np
>>> from nltools.algorithms.inference import one_sample_permutation_test
```

```pycon
>>> # Simple one-sample test
>>> data = np.random.randn(30)  # 30 subjects
>>> result = one_sample_permutation_test(data, n_permute=5000)
>>> print(f"p-value: {result['p']:.3f}")
```

```pycon
>>> # Voxel-wise test with GPU acceleration
>>> data = np.random.randn(30, 50000)  # 30 subjects, 50K voxels
>>> result = one_sample_permutation_test(data, n_permute=10000, device='gpu')
>>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")
```

<details class="performance" open markdown="1">
<summary>Performance</summary>

- CPU (NumPy): Good for small problems (< 5K permutations)
- GPU (PyTorch): Excellent for large problems (> 5K permutations)
- CPU Parallel (joblib): Efficient fallback when GPU unavailable
- Select with device='cpu' | 'gpu' | None (no 'auto' selector)

</details>

<details class="references" open markdown="1">
<summary>References</summary>

Eklund, A., Dufort, P., Villani, M., & LaConte, S. M. (2014).
BROCCOLI: Software for fast fMRI analysis on many-core CPUs and GPUs.
Frontiers in Neuroinformatics, 8, 24.

</details>

<details class="notes" open markdown="1">
<summary>Notes</summary>

This module is part of the "functional core" of nltools. For integration
with BrainData objects, see nltools.data.brain_data.

</details>

## Classes

(algorithms-inference-onlinebootstrapstats)=
### `OnlineBootstrapStats`

```python
OnlineBootstrapStats(shape: tuple[int, ...], save_samples: bool = False, percentiles: tuple[float, float] = (2.5, 97.5))
```

Memory-efficient online statistics aggregator for bootstrap samples.

Uses Welford's algorithm for numerically stable online computation of
mean and variance. Optionally stores all samples for exact percentile CIs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`shape` | <code>[tuple](#tuple)[[int](#int), ...]</code> | Shape of each bootstrap sample. | *required*
`save_samples` | <code>[bool](#bool)</code> | If True, store all samples for exact percentile confidence intervals. If False, use normal approximation (much more memory efficient). Defaults to False. | <code>False</code>
`percentiles` | <code>[tuple](#tuple)[[float](#float), [float](#float)]</code> | Percentiles for confidence intervals (e.g., (2.5, 97.5) for 95% CI). Defaults to (2.5, 97.5). | <code>(2.5, 97.5)</code>

**Methods:**

Name | Description
---- | -----------
[`get_results`](#algorithms-inference-get-results) | Compute final bootstrap statistics.
[`update`](#algorithms-inference-update) | Update statistics with a new bootstrap sample.



**Examples:**

```pycon
>>> stats = OnlineBootstrapStats(shape=(100,), save_samples=False)
>>> for i in range(1000):
...     sample = np.random.randn(100)
...     stats.update(sample)
>>> results = stats.get_results()
>>> print(results.keys())
dict_keys(['mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper'])
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
`tail` | <code>[int](#int) \| [str](#str)</code> | `2`/`'two'` (two-tailed, default) or `1`/`'one'` (one-tailed: statistic > 0; negate the data for the other direction). | <code>2</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | Dictionary with keys 'mean' (bootstrap mean), 'std' (bootstrap standard     deviation), 'Z' (z-scores, mean/std), 'p' (p-values per ``tail``),     'ci_lower' and 'ci_upper' (confidence bounds), and 'samples' (all     samples, only if ``save_samples=True``).

**Examples:**

```python
stats = OnlineBootstrapStats(shape=(100,), save_samples=False)
for _ in range(1000):
    stats.update(np.random.randn(100))
results = stats.get_results()
# results.keys() -> mean, std, Z, p, ci_lower, ci_upper
```

(algorithms-inference-update)=
##### `update`

```python
update(sample: np.ndarray) -> None
```

Update statistics with a new bootstrap sample.

Uses Welford's algorithm for numerical stability.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sample` | <code>[ndarray](#numpy.ndarray)</code> | New bootstrap sample with shape matching self.shape. | *required*



## Methods

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
`data` | <code>[ndarray](#numpy.ndarray)</code> | Time series data, shape (n_samples,) or (n_samples, n_features) | *required*
`shift_amount` | <code>[int](#int) \| [ndarray](#numpy.ndarray) \| None</code> | Shift amount(s). If None, random shift is used. For 1D: int specifying shift amount For 2D: array of length n_features with shift per feature | <code>None</code>
`random_state` | <code>[int](#int) \| [RandomState](#numpy.random.RandomState) \| None</code> | Random seed for reproducibility (if shift_amount is None) | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Circularly shifted data with same shape as input

**Examples:**

```pycon
>>> x = np.array([1, 2, 3, 4, 5])
>>> circle_shift(x, shift_amount=2)
array([4, 5, 1, 2, 3])
```

```pycon
>>> X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
>>> circle_shift(X, shift_amount=np.array([1, 2]))
array([[ 4, 30],
       [ 1, 40],
       [ 2, 10],
       [ 3, 20]])
```

(algorithms-inference-correlation-permutation-test)=
### `correlation_permutation_test`

```python
correlation_permutation_test(data1: np.ndarray, data2: np.ndarray, *, n_permute: int = 5000, metric: str = 'pearson', tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Correlation permutation test.

Tests whether the correlation between data1 and data2 is significantly
different from zero by randomly permuting data1 and computing correlations.

Assumption: Observations are independent (i.i.d.). For autocorrelated time
series, use timeseries_correlation_permutation_test with circle_shift or
phase_randomize methods instead.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | Data to permute - shape (n_samples,) for single feature - shape (n_samples, n_features) for multi-feature | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Data to correlate with - shape (n_samples,) for single feature - shape (n_samples, n_features) for multi-feature | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000) | <code>5000</code>
`metric` | <code>[str](#str)</code> | Correlation metric (default: 'pearson') - 'pearson': Pearson correlation (linear relationships) - 'spearman': Spearman rank correlation (monotonic relationships) - 'kendall': Kendall tau rank correlation (ordinal association, robust to ties) | <code>'pearson'</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | `2` or `'two'` for two-tailed (default); `1` or `'one'` for one-tailed (positive direction). - `2`/`'two'`: Two-tailed test (r != 0) - `1`/`'one'`: One-tailed (r > 0; negate one variable for the other   direction). The fixed direction keeps MCP correction valid. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show a progress bar over permutations (default: False) | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with keys:     - 'correlation' (float or np.ndarray): Observed correlation(s)     - 'p' (float or np.ndarray): P-value(s)     - 'null_dist' (np.ndarray): Null distribution (if return_null=True)     - 'device' (str): Parallelization method used

**Examples:**

```pycon
>>> # Single feature (default CPU parallelization)
>>> x = np.random.randn(100)
>>> y = x + np.random.randn(100) * 0.5  # Correlated
>>> result = correlation_permutation_test(x, y, n_permute=5000)
>>> result['correlation']
0.85
>>> result['p']
0.001
```

```pycon
>>> # Multi-feature (2D arrays)
>>> data1 = np.random.randn(100, 10)  # 100 samples, 10 features
>>> data2 = data1 + np.random.randn(100, 10) * 0.3  # Correlated
>>> result = correlation_permutation_test(data1, data2, n_permute=5000)
>>> result['correlation'].shape
(10,)
>>> result['p'].shape
(10,)
```

```pycon
>>> # GPU acceleration
>>> result = correlation_permutation_test(data1, data2, n_permute=5000, device='gpu')
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (device='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
    - Pearson: Fully vectorized across all features (5-20× speedup for multi-feature)
    - Spearman: GPU rank transform (average ties) + vectorized Pearson on ranks
    - Kendall: tie-corrected tau-b via pre-computed pairwise sign tensors;
      O(n²) memory per permutation, so batches are sized accordingly
- Single-threaded (device=None): Use for small problems or debugging
- For multi-feature data, each feature pair tested independently
- Kendall is O(n^2) complexity, slower than Pearson/Spearman for large samples

</details>

(algorithms-inference-distance-correlation)=
### `distance_correlation`

```python
distance_correlation(x: np.ndarray, y: np.ndarray, bias_corrected: bool = True, ttest: bool = False) -> dict
```

Compute the distance correlation between 2 arrays to test for multivariate dependence (linear or non-linear).

Arrays must match on their first dimension. It's almost always preferable to compute the bias_corrected
version which can also optionally perform a ttest. This ttest operates on a statistic thats ~dcorr^2
and will be also returned.

Explanation:
Distance correlation involves computing the normalized covariance of two centered euclidean distance
matrices. Each distance matrix is the euclidean distance between rows (if x or y are 2d) or scalars
(if x or y are 1d). Each matrix is centered prior to computing the covariance either using double-centering
or u-centering, which corrects for bias as the number of dimensions increases. U-centering is almost always
preferred in all cases. It also permits inference of the normalized covariance between each distance matrix
using a one-tailed directional t-test. (Szekely & Rizzo, 2013). While distance correlation is normally
bounded between 0 and 1, u-centering can produce negative estimates, which are never significant.

Validated against the dcor and dcor.ttest functions in the 'energy' R package and the
dcor.distance_correlation, dcor.udistance_correlation_sqr, and dcor.independence.distance_correlation_t_test
functions in the dcor Python package.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` | <code>[ndarray](#ndarray)</code> | 1d or 2d numpy array of observations by features | *required*
`y` | <code>[ndarray](#ndarray)</code> | 1d or 2d numpy array of observations by features | *required*
`bias_corrected` | <code>[bool](#bool)</code> | if false use double-centering which produces a biased-estimate that converges to 1 as the number of dimensions increase. Otherwise used u-centering to correct this bias. **Note** this must be True if ttest=True; default True | <code>True</code>
`ttest` | <code>[bool](#bool)</code> | perform a ttest using the bias_corrected distance correlation; default False | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary of results (correlation, t, p, and df); optionally also     covariance, x variance, and y variance.

**Examples:**

```pycon
>>> import numpy as np
>>> x = np.random.randn(20, 3)
>>> y = x + np.random.randn(20, 3) * 0.1  # Strongly correlated
>>> result = distance_correlation(x, y, bias_corrected=True)
>>> 'dcorr' in result
True
>>> 0 <= result['dcorr'] <= 1
True
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
`mat` | <code>[ndarray](#ndarray)</code> | 2d numpy array | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Double-centered version of the input.

**Examples:**

```pycon
>>> mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
>>> result = double_center(mat)
>>> np.allclose(result.mean(axis=0), 0)
True
>>> np.allclose(result.mean(axis=1), 0)
True
```

(algorithms-inference-isc-group-permutation-test)=
### `isc_group_permutation_test`

```python
isc_group_permutation_test(group1: np.ndarray, group2: np.ndarray, *, n_permute: int = 5000, summary: Literal['median', 'mean'] = 'median', method: Literal['permute', 'bootstrap'] = 'permute', summary_statistic: Literal['leave-one-out', 'pairwise'] = 'pairwise', ci_percentile: float = 95, tail: int | str = 2, device: Literal['cpu', 'gpu'] | None = 'cpu', n_jobs: int = -1, random_state: int | None = None, return_null: bool = False, progress_bar: bool = False, exclude_self_corr: bool = True, metric: str = 'correlation') -> dict[str, Any]
```

Compute ISC difference between groups with permutation testing.

Supports both subject-wise permutation and bootstrap methods with efficient
CPU-parallel and optional GPU acceleration. Follows the statistical methods
from Chen et al. (2016) for correct group comparison inference.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`group1` | <code>[ndarray](#numpy.ndarray)</code> | First group data with one of the following shapes: - (n_observations, n_subjects1): Single feature - (n_observations, n_subjects1, n_voxels): Voxel-wise | *required*
`group2` | <code>[ndarray](#numpy.ndarray)</code> | Second group data with one of the following shapes: - (n_observations, n_subjects2): Single feature - (n_observations, n_subjects2, n_voxels): Voxel-wise | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations/bootstrap iterations. Defaults to 5000. | <code>5000</code>
`summary` | <code>[Literal](#typing.Literal)['median', 'mean']</code> | Summary statistic for aggregating ISC values: - 'median': Direct median (robust to outliers) - 'mean': Fisher z-transformed mean (unbiased averaging) Defaults to 'median'. | <code>'median'</code>
`method` | <code>[Literal](#typing.Literal)['permute', 'bootstrap']</code> | Resampling method for p-value computation: - 'permute': Subject-wise permutation (combines groups, permutes labels) - 'bootstrap': Subject-wise bootstrap (resamples within each group) Defaults to 'permute'. | <code>'permute'</code>
`summary_statistic` | <code>[Literal](#typing.Literal)['leave-one-out', 'pairwise']</code> | ISC computation method: - 'pairwise': Average all pairwise correlations - 'leave-one-out': Correlate each subject with mean of others Defaults to 'pairwise'. | <code>'pairwise'</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95 for 95% CI). Defaults to 95. | <code>95</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Two-tailed (2 or 'two', default) or one-tailed (1 or 'one', positive direction) p-value. | <code>2</code>
`device` | <code>[Literal](#typing.Literal)['cpu', 'gpu'] \| None</code> | Parallelization method: - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (10-30× speedup for voxel-wise LOO) - None: Single-threaded NumPy (for debugging/small problems) Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when device='cpu'. Defaults to -1. | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, return null distribution in result dict. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during bootstrap/permutation. Defaults to False. | <code>False</code>
`exclude_self_corr` | <code>[bool](#bool)</code> | Mask self-correlations in bootstrap (pairwise only). Defaults to True. | <code>True</code>
`metric` | <code>[str](#str)</code> | Similarity metric for pairwise ISC computation. See sklearn.metrics.pairwise_distances for valid options. Only applies when summary_statistic='pairwise'. Defaults to 'correlation'. | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with keys 'isc_group_difference' (observed ISC difference, float or     array per voxel), 'p' (Phipson-Smyth corrected p-value), 'ci' (confidence     interval tuple `(lower, upper)`), 'device' (parallelization method used),     and optionally 'null_dist' (bootstrap/permutation distribution).

**Examples:**

```pycon
>>> # Single-feature ISC group comparison
>>> group1 = np.random.randn(100, 10)  # 10 subjects
>>> group2 = np.random.randn(100, 10)
>>> result = isc_group_permutation_test(group1, group2, n_permute=1000)
>>> print(f"ISC difference: {result['isc_group_difference']:.3f}, p: {result['p']:.3f}")
```

```pycon
>>> # Voxel-wise ISC group comparison with GPU acceleration
>>> group1_voxels = np.random.randn(100, 10, 5000)  # 5K voxels
>>> group2_voxels = np.random.randn(100, 10, 5000)
>>> result = isc_group_permutation_test(
...     group1_voxels,
...     group2_voxels,
...     summary_statistic='leave-one-out',
...     device='gpu',  # GPU for LOO computation
...     n_permute=5000
... )
>>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")
```

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Permutation method combines groups and permutes labels (Chen et al. 2016)
- Bootstrap method resamples subjects within each group independently
- Bootstrap distribution is centered by subtracting observed difference
- GPU acceleration available for voxel-wise LOO computation

</details>

(algorithms-inference-isc-permutation-test)=
### `isc_permutation_test`

```python
isc_permutation_test(data: np.ndarray, *, n_permute: int = 5000, summary: Literal['median', 'mean'] = 'median', summary_statistic: Literal['leave-one-out', 'pairwise'] = 'pairwise', method: Literal['bootstrap', 'circle_shift', 'phase_randomize'] = 'bootstrap', ci_percentile: float = 95, tail: int | str = 2, return_null: bool = False, progress_bar: bool = False, exclude_self_corr: bool = True, metric: str = 'correlation', device: Literal['cpu', 'gpu'] | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> dict[str, Any]
```

Compute intersubject correlation with permutation testing.

Supports both leave-one-out and pairwise ISC computation modes with
GPU acceleration for large voxel-wise problems and CPU-parallel
bootstrap resampling.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Data array with one of the following shapes: - (n_observations, n_subjects): Single feature ISC - (n_observations, n_subjects, n_voxels): Voxel-wise ISC | *required*
`n_permute` | <code>[int](#int)</code> | Number of bootstrap iterations or permutations. Defaults to 5000. | <code>5000</code>
`summary` | <code>[Literal](#typing.Literal)['median', 'mean']</code> | Summary statistic to aggregate ISC values. - 'median': Direct median (robust to outliers) - 'mean': Fisher z-transformed mean (unbiased averaging) Defaults to 'median'. | <code>'median'</code>
`summary_statistic` | <code>[Literal](#typing.Literal)['leave-one-out', 'pairwise']</code> | ISC computation method. Options: - 'leave-one-out': Correlate each subject with mean of others. O(n_subjects), unbiased, recommended by Chen et al. 2016. - 'pairwise': Average all pairwise correlations. O(n_subjects²), captures full correlation structure. Note: These methods are statistically different and monotonically but non-linearly related (see Chen et al. 2016, Figure 3). Defaults to 'pairwise'. | <code>'pairwise'</code>
`method` | <code>[Literal](#typing.Literal)['bootstrap', 'circle_shift', 'phase_randomize']</code> | Resampling method for p-value computation: - 'bootstrap': Subject-wise bootstrap (default, Chen et al. 2016) - 'circle_shift': Circular time-series shift (preserves autocorrelation) - 'phase_randomize': FFT phase randomization (preserves power spectrum) Defaults to 'bootstrap'. | <code>'bootstrap'</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95 for 95% CI). Defaults to 95. | <code>95</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Two-tailed (2 or 'two', default) or one-tailed (1 or 'one', positive direction) p-value. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return bootstrap/permutation distribution in result dict. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during bootstrap/permutation. Defaults to False. | <code>False</code>
`exclude_self_corr` | <code>[bool](#bool)</code> | If True, mask self-correlations (perfect correlations from duplicate subjects in bootstrap samples) as NaN. If False, include them in the summary statistic. Only applies when method='bootstrap' and summary_statistic='pairwise'. Defaults to True. | <code>True</code>
`metric` | <code>[str](#str)</code> | Similarity metric for pairwise ISC computation. See sklearn.metrics.pairwise_distances for valid options. Only applies when summary_statistic='pairwise'. For 'correlation', uses optimized np.corrcoef. Other metrics use pairwise_distances. Defaults to 'correlation'. | <code>'correlation'</code>
`device` | <code>[Literal](#typing.Literal)['cpu', 'gpu'] \| None</code> | Parallelization method: - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (10-30× speedup for voxel-wise LOO) - None: Single-threaded NumPy (for debugging/small problems) Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when device='cpu'. Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float) \| None</code> | GPU working-set budget in GB. For the pairwise GPU bootstrap (``device='gpu'``, ``summary_statistic='pairwise'``, ``method='bootstrap'``) this bounds the ``(perm_batch, voxel_chunk, n_subjects, n_subjects)`` resample tensor, chunking voxels and permutations to fit — so whole-brain runs stay within budget. Not used by the LOO or surrogate (circle_shift/phase_randomize) paths. Defaults to 4. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with keys 'isc' (observed ISC value, float or array per voxel),     'p' (Phipson-Smyth corrected p-value), 'ci' (confidence interval tuple     `(lower, upper)`), 'device' (parallelization method used), and     optionally 'null_dist' (bootstrap/permutation distribution).

**Examples:**

```pycon
>>> # Single-feature ISC
>>> data = np.random.randn(100, 10)  # 100 timepoints, 10 subjects
>>> result = isc_permutation_test(data, n_permute=1000)
>>> print(f"ISC: {result['isc']:.3f}, p: {result['p']:.3f}")
```

```pycon
>>> # Voxel-wise ISC with GPU acceleration
>>> data_voxels = np.random.randn(100, 50, 5000)  # 5K voxels
>>> result = isc_permutation_test(
...     data_voxels,
...     summary_statistic='leave-one-out',
...     device='gpu',  # GPU for LOO computation
...     n_permute=5000
... )
>>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")
```

```pycon
>>> # Compare LOO vs pairwise
>>> result_loo = isc_permutation_test(data, summary_statistic='leave-one-out')
>>> result_pair = isc_permutation_test(data, summary_statistic='pairwise')
>>> print(f"LOO: {result_loo['isc']:.3f}, Pairwise: {result_pair['isc']:.3f}")
```

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Leave-one-out is 20-30× faster than pairwise for large n_subjects
- GPU acceleration helps most for voxel-wise LOO (10-30× speedup)
- Pairwise bootstrap uses correct subject-wise resampling (Chen 2016)
- Bootstrap distribution is centered by subtracting observed ISC

</details>

(algorithms-inference-matrix-permutation-test)=
### `matrix_permutation_test`

```python
matrix_permutation_test(data1: np.ndarray, data2: np.ndarray, *, n_permute: int = 5000, metric: str = 'pearson', how: str = 'upper', include_diag: bool = False, tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Matrix permutation test (Mantel test) for correlating two square matrices.

Tests whether the correlation between elements of two matrices is significant
by permuting rows and columns of one matrix symmetrically while keeping the
other fixed.

**Statistical Method**:
For each permutation, create random permutation `perm`, then apply:
`matrix1[perm][:, perm]`. This preserves matrix structure while destroying
correlation. Count how often permuted correlation is as extreme as observed.

**Assumptions**:
- Matrices are square and same size
- Under H₀, row/column ordering is exchangeable
- Symmetric permutation preserves matrix properties (e.g., symmetry)

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | First square matrix (n×n) | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Second square matrix (n×n) | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000) | <code>5000</code>
`metric` | <code>[str](#str)</code> | Correlation metric, one of 'pearson', 'spearman', or 'kendall' (default: 'pearson') | <code>'pearson'</code>
`how` | <code>[str](#str)</code> | Which elements to compare, one of 'upper', 'lower', or 'full' (default: 'upper') - 'upper': Upper triangle only (assumes symmetric matrices) - 'lower': Lower triangle only - 'full': All elements (see include_diag) | <code>'upper'</code>
`include_diag` | <code>[bool](#bool)</code> | Include diagonal elements (only applies if how='full') (default: False) | <code>False</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | `2` or `'two'` for two-tailed (default); `1` or `'one'` for one-tailed (positive direction). - `2`/`'two'`: Two-tailed test (r != 0) - `1`/`'one'`: One-tailed (r > 0; negate the data for the other direction) | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | Return null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel workers, -1 = all cores (default: -1) Only used when device='cpu' | <code>-1</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show a progress bar over permutations (default: False) | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with keys:     - 'correlation' (float): Observed correlation coefficient     - 'p' (float): P-value using Phipson-Smyth correction     - 'device' (str): Parallelization method used ('cpu' or None)     - 'null_dist' (np.ndarray): Null distribution (if return_null=True)

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G. et al. (2016). Untangling the relatedness among correlations,
part I: nonparametric approaches to inter-subject correlation analysis
at the group level. NeuroImage, 142, 248-259.

Mantel, N. (1967). The detection of disease clustering and a generalized
regression approach. Cancer Research, 27(2), 209-220.

</details>

**Examples:**

```pycon
>>> import numpy as np
>>> from nltools.algorithms.inference import matrix_permutation_test
>>>
>>> # Create two correlated similarity matrices
>>> np.random.seed(42)
>>> n = 50
>>> true_pattern = np.random.randn(n)
>>> data1 = np.corrcoef(true_pattern + np.random.randn(n) * 0.1)
>>> data2 = np.corrcoef(true_pattern + np.random.randn(n) * 0.1)
>>>
>>> # Test if matrices are correlated
>>> result = matrix_permutation_test(data1, data2, n_permute=1000)
>>> print(f"Correlation: {result['correlation']:.3f}, p = {result['p']:.4f}")
```

(algorithms-inference-one-sample-permutation-test)=
### `one_sample_permutation_test`

```python
one_sample_permutation_test(data: np.ndarray, *, n_permute: int = 5000, tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None, progress_bar: bool = False) -> dict
```

One-sample permutation test using sign-flipping.

Tests whether the mean of data is significantly different from zero
by randomly flipping the sign of each observation. This is the
permutation test equivalent of a one-sample t-test.

Assumption: Symmetric error distribution around zero. For highly skewed
distributions, consider alternative methods (e.g., bootstrap resampling).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Data to test - shape (n_samples,) for single feature - shape (n_samples, n_features) for multi-feature (voxel-wise) | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000) | <code>5000</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | `2` or `'two'` for two-tailed (default); `1` or `'one'` for one-tailed (positive direction). - `2`/`'two'`: Two-tailed test (mean != 0) - `1`/`'one'`: One-tailed (mean > 0; negate the data for the other   direction). The fixed direction keeps MCP correction valid. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Whether to display a progress bar (default: False) | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with keys:     - 'mean' (float or np.ndarray): Observed mean(s)     - 'p' (float or np.ndarray): P-value(s)     - 'null_dist' (np.ndarray): Null distribution (if return_null=True)     - 'device' (str): Parallelization method used

**Examples:**

```pycon
>>> # Single feature (default CPU parallelization)
>>> data = np.random.randn(30)
>>> result = one_sample_permutation_test(data, n_permute=5000)
>>> result['p']
0.23
```

```pycon
>>> # Voxel-wise test with GPU
>>> data = np.random.randn(30, 10000)  # 30 subjects, 10K voxels
>>> result = one_sample_permutation_test(data, n_permute=5000, device='gpu')
>>> result['mean'].shape
(10000,)
>>> result['p'].shape
(10000,)
```

```pycon
>>> # Single-threaded (for debugging)
>>> result = one_sample_permutation_test(data, n_permute=5000, device=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (device='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
- Single-threaded (device=None): Use for small problems or debugging
- For voxel-wise tests, each voxel tested independently
- Progress bars show completion for both CPU parallel and GPU batched modes

</details>

(algorithms-inference-phase-randomize)=
### `phase_randomize`

```python
phase_randomize(data: np.ndarray, *, device: str | None = 'cpu', random_state: int | np.random.RandomState | None = None) -> np.ndarray
```

FFT-based phase randomization for time-series data.

Preserves the power spectrum (autocorrelation) but destroys nonlinear
temporal structure by randomizing Fourier phases. Used to test whether
data was generated by a linear Gaussian process or contains nonlinear
dynamics.

<details class="algorithm" open markdown="1">
<summary>Algorithm</summary>

1. Compute FFT of input signal
2. Generate random phases [0, 2π] for positive frequencies
3. Apply phase shifts to positive frequencies: multiply by exp(i*φ)
4. Apply conjugate phase shifts to negative frequencies (for real output)
5. Compute inverse FFT to get phase-randomized signal

</details>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Time series data, shape (n_samples,) or (n_samples, n_features) | *required*
`device` | <code>[str](#str) \| None</code> | Compute device. - 'cpu' / None: NumPy FFT (default, float64 precision) - 'gpu': PyTorch FFT on CUDA/MPS (float32 precision, 5-20× faster for large data) - 'auto': use a GPU if present, else CPU | <code>'cpu'</code>
`random_state` | <code>[int](#int) \| [RandomState](#numpy.random.RandomState) \| None</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Phase-randomized data with same shape as input

<details class="notes" open markdown="1">
<summary>Notes</summary>

- **CRITICAL**: Preserves power spectrum exactly (within numerical precision)
- Precision: the CPU path uses float64, the GPU path float32
- Conjugate symmetry is maintained for real-valued output

</details>

**Examples:**

```pycon
>>> x = np.sin(np.linspace(0, 10*np.pi, 100))  # Sine wave
>>> x_rand = phase_randomize(x, random_state=42)
>>> # Power spectrum preserved:
>>> np.allclose(np.abs(np.fft.rfft(x))**2, np.abs(np.fft.rfft(x_rand))**2)
True
```

```pycon
>>> # GPU acceleration for large datasets:
>>> x_large = np.random.randn(10000)
>>> x_rand_gpu = phase_randomize(x_large, device='gpu', random_state=42)
```

(algorithms-inference-timeseries-correlation-permutation-test)=
### `timeseries_correlation_permutation_test`

```python
timeseries_correlation_permutation_test(data1: np.ndarray, data2: np.ndarray, *, method: Literal['circle_shift', 'phase_randomize'] = 'circle_shift', n_permute: int = 5000, metric: Literal['pearson', 'spearman', 'kendall'] = 'pearson', tail: int | str = 2, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, return_null: bool = False, random_state: int | np.random.RandomState | None = None, progress_bar: bool = False) -> dict
```

Time-series correlation permutation test.

Unlike standard permutation tests that shuffle data independently,
this test uses time-series-aware permutation methods that preserve
temporal structure (circle_shift) or power spectrum (phase_randomize).

Use this test when data contains temporal autocorrelation. Standard
permutation tests inflate Type I error for autocorrelated data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | First time series, shape (n_samples,) or (n_samples, 1) | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Second time series, shape (n_samples,) or (n_samples, 1) | *required*
`method` | <code>[Literal](#typing.Literal)['circle_shift', 'phase_randomize']</code> | Permutation method: - 'circle_shift': Circular shift (preserves autocorrelation) - 'phase_randomize': FFT-based (preserves power spectrum) | <code>'circle_shift'</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations | <code>5000</code>
`metric` | <code>[Literal](#typing.Literal)['pearson', 'spearman', 'kendall']</code> | Correlation type ('pearson', 'spearman', 'kendall') | <code>'pearson'</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type (default: 2) - 2 or 'two': Two-tailed test (default) - 1 or 'one': One-tailed test in the test's positive direction   (to test the negative direction, negate the data / swap groups) | <code>2</code>
`device` | <code>[str](#str) \| None</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs (-1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float) \| None</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | Whether to return null distribution | <code>False</code>
`random_state` | <code>[int](#int) \| [RandomState](#numpy.random.RandomState) \| None</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show a progress bar over permutations (default: False) | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>Dictionary with keys</code> | - 'correlation': Observed correlation coefficient     - 'p': P-value     - 'null_dist': (if return_null=True) Null distribution     - 'device': Parallelization method used

**Examples:**

```pycon
>>> x = np.sin(np.linspace(0, 10*np.pi, 100))
>>> y = np.cos(np.linspace(0, 10*np.pi, 100))
>>> result = timeseries_correlation_permutation_test(
...     x, y, method='circle_shift', n_permute=1000, random_state=42
... )
>>> result['correlation']  # Strong negative correlation
-0.999...
>>> result['p'] < 0.05  # Significant
True
```

```pycon
>>> # GPU acceleration
>>> result = timeseries_correlation_permutation_test(
...     x, y, method='phase_randomize', device='gpu', n_permute=5000
... )
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (device='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): 5-20× faster for large problems (n_samples > 1000)
- Single-threaded (device=None): Use for small problems or debugging
- For independent data, use regular correlation_permutation_test
- circle_shift is faster and suitable for most fMRI time series
- phase_randomize preserves power spectrum exactly (tests nonlinearity)
- Only data1 is randomized; data2 remains fixed to test correlation
- phase_randomize benefits most from GPU (FFT acceleration)

</details>

(algorithms-inference-two-sample-permutation-test)=
### `two_sample_permutation_test`

```python
two_sample_permutation_test(data1: np.ndarray, data2: np.ndarray, *, n_permute: int = 5000, tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Two-sample permutation test using group label shuffling.

Tests whether two independent groups have different means by randomly
permuting group labels. This is the permutation test equivalent of an
independent samples t-test.

Assumption: Exchangeability under the null hypothesis (group assignments
are arbitrary). Valid for independent samples from similar distributions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | Group 1 data - shape (n_samples1,) for single feature - shape (n_samples1, n_features) for multi-feature (voxel-wise) | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Group 2 data - shape (n_samples2,) for single feature - shape (n_samples2, n_features) for multi-feature (voxel-wise) | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000) | <code>5000</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | `2` or `'two'` for two-tailed (default); `1` or `'one'` for one-tailed (positive direction). - `2`/`'two'`: Two-tailed test (mean1 != mean2) - `1`/`'one'`: One-tailed (mean1 > mean2; swap the groups for the   other direction). The fixed direction keeps MCP correction valid. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with keys:     - 'mean_diff' (float or np.ndarray): Observed mean difference (data1 - data2)     - 'p' (float or np.ndarray): P-value(s)     - 'null_dist' (np.ndarray): Null distribution (if return_null=True)     - 'device' (str): Parallelization method used

**Examples:**

```pycon
>>> # Single feature (default CPU parallelization)
>>> data1 = np.random.randn(20)  # Group 1: 20 subjects
>>> data2 = np.random.randn(25)  # Group 2: 25 subjects
>>> result = two_sample_permutation_test(data1, data2, n_permute=5000)
>>> result['p']
0.45
```

```pycon
>>> # Voxel-wise test with GPU
>>> data1 = np.random.randn(20, 10000)  # 20 subjects, 10K voxels
>>> data2 = np.random.randn(25, 10000)  # 25 subjects, 10K voxels
>>> result = two_sample_permutation_test(data1, data2, n_permute=5000, device='gpu')
>>> result['mean_diff'].shape
(10000,)
>>> result['p'].shape
(10000,)
```

```pycon
>>> # Single-threaded (for debugging)
>>> result = two_sample_permutation_test(data1, data2, n_permute=5000, device=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (device='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
- Single-threaded (device=None): Use for small problems or debugging
- For voxel-wise tests, each voxel tested independently
- Group sizes can be unequal

</details>

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
`mat` | <code>[ndarray](#ndarray)</code> | 2d numpy array | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#ndarray)</code> | U-centered version of the input.

**Examples:**

```pycon
>>> mat = np.random.randn(5, 5)
>>> result = u_center(mat)
>>> np.allclose(np.diag(result), 0)
True
```
