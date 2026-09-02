---
title: Statistics & inference
label: page-tasks-inference
---

Non-parametric group statistics. The one-sample, two-sample, and timeseries permutation tests run on CPU or GPU (`device=`); `phase_randomize` and `circle_shift` are the timeseries null models. `OnlineBootstrapStats` keeps running mean and variance over bootstrap draws instead of storing them. `fdr`, `holm_bonf`, `threshold`, and `multi_threshold` correct or threshold the resulting p-maps. `BrainData.ttest`, `Adjacency.ttest`, and `BrainData.bootstrap` call these.

**Classes:**

Name | Description
---- | -----------
[`OnlineBootstrapStats`](#tasks-inference-onlinebootstrapstats) | Memory-efficient online statistics aggregator for bootstrap samples.

**Functions:**

Name | Description
---- | -----------
[`one_sample_permutation_test`](#tasks-inference-one-sample-permutation-test) | One-sample permutation test using sign flipping.
[`two_sample_permutation_test`](#tasks-inference-two-sample-permutation-test) | Two-sample permutation test using group-label shuffling.
[`timeseries_correlation_permutation_test`](#tasks-inference-timeseries-correlation-permutation-test) | Permutation test for the correlation between two autocorrelated time series.
[`phase_randomize`](#tasks-inference-phase-randomize) | FFT-based phase randomization for time-series data.
[`circle_shift`](#tasks-inference-circle-shift) | Circular shift for time-series data.
[`fdr`](#tasks-inference-fdr) | Determine an FDR threshold for an array of p-values.
[`holm_bonf`](#tasks-inference-holm-bonf) | Determine a Holm-Bonferroni (step-down) threshold for an array of p-values.
[`threshold`](#tasks-inference-threshold) | Threshold a statistic image by the p-values in a separate image.
[`multi_threshold`](#tasks-inference-multi-threshold) | Threshold a statistic image at several p-values and count the passes per voxel.

## Classes

(tasks-inference-onlinebootstrapstats)=
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
[`get_results`](#tasks-inference-get-results) | Compute final bootstrap statistics.
[`update`](#tasks-inference-update) | Fold one bootstrap sample into the running statistics.



**Examples:**

```python
stats = OnlineBootstrapStats(shape=(100,), save_samples=False)
for _ in range(1000):
    stats.update(np.random.randn(100))
results = stats.get_results()
results.keys()  # → dict_keys(['mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper'])
```

#### Methods

(tasks-inference-get-results)=
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

(tasks-inference-update)=
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

(tasks-inference-one-sample-permutation-test)=
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

(tasks-inference-two-sample-permutation-test)=
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

(tasks-inference-timeseries-correlation-permutation-test)=
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

(tasks-inference-phase-randomize)=
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

(tasks-inference-circle-shift)=
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

(tasks-inference-fdr)=
### `fdr`

```python
fdr(p, q = 0.05)
```

Determine an FDR threshold for an array of p-values.

Benjamini-Hochberg procedure at false discovery rate `q` (valid under
independence or positive dependence). Written by Tal Yarkoni.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`p` | <code>ndarray</code> | Vector of p-values. | *required*
`q` | <code>float</code> | False discovery rate level. Defaults to 0.05. | <code>0.05</code>

**Returns:**

Type | Description
---- | -----------
<code>float</code> | The p-value threshold; `-1` when no p-value survives correction.

(tasks-inference-holm-bonf)=
### `holm_bonf`

```python
holm_bonf(p, alpha = 0.05)
```

Determine a Holm-Bonferroni (step-down) threshold for an array of p-values.

The step-down procedure applies progressively less correction to larger
p-values. It is more conservative than FDR but much more powerful than plain
Bonferroni correction.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`p` | <code>ndarray</code> | Vector of p-values. | *required*
`alpha` | <code>float</code> | Family-wise alpha level. Defaults to 0.05. | <code>0.05</code>

**Returns:**

Type | Description
---- | -----------
<code>float</code> | The p-value threshold; `-1` when no p-value survives correction.

(tasks-inference-threshold)=
### `threshold`

```python
threshold(stat, p, thr = 0.05, return_mask = False)
```

Threshold a statistic image by the p-values in a separate image.

Voxels whose p-value is at or above `thr` are set to zero in a copy of `stat`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stat` | <code>[BrainData](#page-data-brain-data)</code> | Statistic image (e.g. betas or t-values). | *required*
`p` | <code>[BrainData](#page-data-brain-data)</code> | P-value image with the same voxels as `stat`. | *required*
`thr` | <code>float</code> | P-value threshold; voxels with `p < thr` are kept. Defaults to 0.05. | <code>0.05</code>
`return_mask` | <code>bool</code> | Also return the binary thresholding mask. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data) \| tuple[[BrainData](#page-data-brain-data), [BrainData](#page-data-brain-data)]</code> | The thresholded image, or the     tuple `(thresholded, mask)` when `return_mask=True`.

<details class="note" open markdown="1">
<summary>Note</summary>

`BrainData.threshold` and `nilearn.image.threshold_img` threshold an image
by its own values; this function is the only one that thresholds one
image by the p-values of another.

</details>

(tasks-inference-multi-threshold)=
### `multi_threshold`

```python
multi_threshold(t_map, p_map, thresh)
```

Threshold a statistic image at several p-values and count the passes per voxel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_map` | <code>[BrainData](#page-data-brain-data)</code> | Statistic image (e.g. t-values or betas). | *required*
`p_map` | <code>[BrainData](#page-data-brain-data)</code> | P-value image with the same voxels as `t_map`. | *required*
`thresh` | <code>list[float]</code> | P-value thresholds to apply. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Cumulative map. Positive values count how many thresholds a     positive statistic passed; negative values count the same for negative     statistics.

<details class="note" open markdown="1">
<summary>Note</summary>

Calling `threshold` once per level gives separate images; this returns a
single map of the threshold hierarchy, which `nilearn.image.threshold_img`
cannot produce.

</details>
