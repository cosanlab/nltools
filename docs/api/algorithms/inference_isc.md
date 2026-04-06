## `isc`

Intersubject Correlation (ISC) with GPU-Accelerated Permutation Testing.

This module provides both leave-one-out (LOO) and pairwise ISC computation
with efficient CPU-parallel and GPU-batched implementations. Follows the
statistical methods from Chen et al. (2016) for correct bootstrap resampling
of correlation matrices.

<details class="key-features" open markdown="1">
<summary>Key Features</summary>

- Two ISC modes: leave-one-out and pairwise (statistically different)
- GPU acceleration for voxel-wise computation (10-30× speedup)
- CPU-parallel bootstrap with joblib
- Correct subject-wise bootstrap (Chen et al. 2016)
- Memory-efficient condensed matrix storage

</details>

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

<details class="notes" open markdown="1">
<summary>Notes</summary>

Leave-one-out and pairwise ISC are monotonically correlated but
statistically different. LOO is computationally more efficient
and provides unbiased estimates. Pairwise captures full correlation
structure but is O(n²) in subjects.

</details>

**Methods:**

Name | Description
---- | -----------
[`isc_group_permutation_test`](#isc_group_permutation_test) | Compute ISC difference between groups with permutation testing.
[`isc_permutation_test`](#isc_permutation_test) | Compute intersubject correlation with permutation testing.

### Methods

#### `isc_group_permutation_test`

```python
isc_group_permutation_test(group1: np.ndarray, group2: np.ndarray, n_permute: int = 5000, metric: Literal['median', 'mean'] = 'median', method: Literal['permute', 'bootstrap'] = 'permute', summary_statistic: Literal['leave-one-out', 'pairwise'] = 'pairwise', ci_percentile: float = 95, tail: Literal[1, 2] = 2, parallel: Optional[Literal['cpu', 'gpu']] = 'cpu', n_jobs: int = -1, random_state: Optional[int] = None, return_null: bool = False, progress_bar: bool = True, exclude_self_corr: bool = True, sim_metric: str = 'correlation') -> Dict[str, Any]
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
`metric` | <code>[Literal](#typing.Literal)['median', 'mean']</code> | Summary statistic for aggregating ISC values: - 'median': Direct median (robust to outliers) - 'mean': Fisher z-transformed mean (unbiased averaging) Defaults to 'median'. | <code>'median'</code>
`method` | <code>[Literal](#typing.Literal)['permute', 'bootstrap']</code> | Resampling method for p-value computation: - 'permute': Subject-wise permutation (combines groups, permutes labels) - 'bootstrap': Subject-wise bootstrap (resamples within each group) Defaults to 'permute'. | <code>'permute'</code>
`summary_statistic` | <code>[Literal](#typing.Literal)['leave-one-out', 'pairwise']</code> | ISC computation method: - 'pairwise': Average all pairwise correlations - 'leave-one-out': Correlate each subject with mean of others Defaults to 'pairwise'. | <code>'pairwise'</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95 for 95% CI). Defaults to 95. | <code>95</code>
`tail` | <code>[Literal](#typing.Literal)[1, 2]</code> | One-tailed (1) or two-tailed (2) p-value. Defaults to 2. | <code>2</code>
`parallel` | <code>[Optional](#typing.Optional)[[Literal](#typing.Literal)['cpu', 'gpu']]</code> | Parallelization method: - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (10-30× speedup for voxel-wise LOO) - None: Single-threaded NumPy (for debugging/small problems) Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when parallel='cpu'. Defaults to -1. | <code>-1</code>
`random_state` | <code>[Optional](#typing.Optional)[[int](#int)]</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, return null distribution in result dict. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during bootstrap/permutation. Defaults to True. | <code>True</code>
`exclude_self_corr` | <code>[bool](#bool)</code> | Mask self-correlations in bootstrap (pairwise only). Defaults to True. | <code>True</code>
`sim_metric` | <code>[str](#str)</code> | Similarity metric for pairwise ISC computation. See sklearn.metrics.pairwise_distances for valid options. Only applies when summary_statistic='pairwise'. Defaults to 'correlation'. | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
<code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with the following keys:
<code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | - 'isc_group_difference': Observed ISC difference (float or array per voxel)
<code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | - 'p': P-value (Phipson-Smyth corrected)
<code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | - 'ci': Confidence interval tuple (lower, upper)
<code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | - 'parallel': Parallelization method used
<code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | - 'null_dist': (optional) Bootstrap/permutation distribution

Examples:
>>> # Single-feature ISC group comparison
>>> group1 = np.random.randn(100, 10)  # 10 subjects
>>> group2 = np.random.randn(100, 10)
>>> result = isc_group_permutation_test(group1, group2, n_permute=1000)
>>> print(f"ISC difference: {result['isc_group_difference']:.3f}, p: {result['p']:.3f}")

>>> # Voxel-wise ISC group comparison with GPU acceleration
>>> group1_voxels = np.random.randn(100, 10, 5000)  # 5K voxels
>>> group2_voxels = np.random.randn(100, 10, 5000)
>>> result = isc_group_permutation_test(
...     group1_voxels,
...     group2_voxels,
...     summary_statistic='leave-one-out',
...     parallel='gpu',  # GPU for LOO computation
...     n_permute=5000
... )
>>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")

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

#### `isc_permutation_test`

```python
isc_permutation_test(data: np.ndarray, n_permute: int = 5000, metric: Literal['median', 'mean'] = 'median', summary_statistic: Literal['leave-one-out', 'pairwise'] = 'pairwise', method: Literal['bootstrap', 'circle_shift', 'phase_randomize'] = 'bootstrap', ci_percentile: float = 95, tail: Literal[1, 2] = 2, return_null: bool = False, progress_bar: bool = True, exclude_self_corr: bool = True, sim_metric: str = 'correlation', parallel: Optional[Literal['cpu', 'gpu']] = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: Optional[int] = None) -> Dict[str, Any]
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
`metric` | <code>[Literal](#typing.Literal)['median', 'mean']</code> | Summary statistic to aggregate ISC values. - 'median': Direct median (robust to outliers) - 'mean': Fisher z-transformed mean (unbiased averaging) Defaults to 'median'. | <code>'median'</code>
`summary_statistic` | <code>[Literal](#typing.Literal)['leave-one-out', 'pairwise']</code> | ISC computation method. Options: - 'leave-one-out': Correlate each subject with mean of others. O(n_subjects), unbiased, recommended by Chen et al. 2016. - 'pairwise': Average all pairwise correlations. O(n_subjects²), captures full correlation structure. Note: These methods are statistically different and monotonically but non-linearly related (see Chen et al. 2016, Figure 3). Defaults to 'pairwise'. | <code>'pairwise'</code>
`method` | <code>[Literal](#typing.Literal)['bootstrap', 'circle_shift', 'phase_randomize']</code> | Resampling method for p-value computation: - 'bootstrap': Subject-wise bootstrap (default, Chen et al. 2016) - 'circle_shift': Circular time-series shift (preserves autocorrelation) - 'phase_randomize': FFT phase randomization (preserves power spectrum) Defaults to 'bootstrap'. | <code>'bootstrap'</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95 for 95% CI). Defaults to 95. | <code>95</code>
`tail` | <code>[Literal](#typing.Literal)[1, 2]</code> | One-tailed (1) or two-tailed (2) p-value. Defaults to 2. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return bootstrap/permutation distribution in result dict. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during bootstrap/permutation. Defaults to True. | <code>True</code>
`exclude_self_corr` | <code>[bool](#bool)</code> | If True, mask self-correlations (perfect correlations from duplicate subjects in bootstrap samples) as NaN. If False, include them in the summary statistic. Only applies when method='bootstrap' and summary_statistic='pairwise'. Defaults to True. | <code>True</code>
`sim_metric` | <code>[str](#str)</code> | Similarity metric for pairwise ISC computation. See sklearn.metrics.pairwise_distances for valid options. Only applies when summary_statistic='pairwise'. For 'correlation', uses optimized np.corrcoef. Other metrics use pairwise_distances. Defaults to 'correlation'. | <code>'correlation'</code>
`parallel` | <code>[Optional](#typing.Optional)[[Literal](#typing.Literal)['cpu', 'gpu']]</code> | Parallelization method: - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (10-30× speedup for voxel-wise LOO) - None: Single-threaded NumPy (for debugging/small problems) Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when parallel='cpu'. Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel='gpu'). Defaults to 4. | <code>4.0</code>
`random_state` | <code>[Optional](#typing.Optional)[[int](#int)]</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with the following keys:
<code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | - 'isc': Observed ISC value (float or array per voxel)
<code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | - 'p': P-value (Phipson-Smyth corrected)
<code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | - 'ci': Confidence interval tuple (lower, upper)
<code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | - 'parallel': Parallelization method used
<code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | - 'null_dist': (optional) Bootstrap/permutation distribution

Examples:
>>> # Single-feature ISC
>>> data = np.random.randn(100, 10)  # 100 timepoints, 10 subjects
>>> result = isc_permutation_test(data, n_permute=1000)
>>> print(f"ISC: {result['isc']:.3f}, p: {result['p']:.3f}")

>>> # Voxel-wise ISC with GPU acceleration
>>> data_voxels = np.random.randn(100, 50, 5000)  # 5K voxels
>>> result = isc_permutation_test(
...     data_voxels,
...     summary_statistic='leave-one-out',
...     parallel='gpu',  # GPU for LOO computation
...     n_permute=5000
... )
>>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")

>>> # Compare LOO vs pairwise
>>> result_loo = isc_permutation_test(data, summary_statistic='leave-one-out')
>>> result_pair = isc_permutation_test(data, summary_statistic='pairwise')
>>> print(f"LOO: {result_loo['isc']:.3f}, Pairwise: {result_pair['isc']:.3f}")

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

