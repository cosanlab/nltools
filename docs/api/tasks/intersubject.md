---
title: Intersubject correlation
label: page-tasks-intersubject
---

Measure time-locked responses shared across subjects. `isc` correlates each subject's timeseries with the rest of the group and bootstraps a confidence interval. `isfc` does the same across regions, `isps` measures phase synchrony, and `isc_group` compares two groups by permutation. The two `*_permutation_test` functions are the CPU/GPU engines (`device=`) underneath, shared with the [permutation tests](inference.md).

**Functions:**

Name | Description
---- | -----------
[`isc`](#tasks-intersubject-isc) | Compute pairwise intersubject correlation from an observations-by-subjects array.
[`isfc`](#tasks-intersubject-isfc) | Compute intersubject functional connectivity (ISFC) from per-subject matrices.
[`isps`](#tasks-intersubject-isps) | Compute dynamic intersubject phase synchrony (ISPS) from an observations-by-subjects array.
[`isc_group`](#tasks-intersubject-isc-group) | Test the difference in pairwise intersubject correlation between two groups.
[`isc_permutation_test`](#tasks-intersubject-isc-permutation-test) | Compute intersubject correlation with bootstrap or permutation inference.
[`isc_group_permutation_test`](#tasks-intersubject-isc-group-permutation-test) | Test the difference in intersubject correlation between two groups.

## Functions

(tasks-intersubject-isc)=
### `isc`

```python
isc(data, *, n_samples = 5000, summary = 'median', method = 'bootstrap', ci_percentile = 95, exclude_self_corr = True, tail = 2, metric = 'correlation', return_null = False, n_jobs = -1, random_state = None, progress_bar = False)
```

Compute pairwise intersubject correlation from an observations-by-subjects array.

Pairwise ISC is summarized with the median, as Chen et al. (2016) recommend;
`summary='mean'` instead averages after the Fisher r-to-z transform and
converts back, which avoids inflating the estimate.

Three null distributions are available. The default subject-wise bootstrap
(Chen et al., 2016) resamples subjects with replacement and recomputes the
pairwise similarity matrix; a subject drawn twice correlates perfectly with
itself, so those entries are set to NaN when `exclude_self_corr=True`.
P-values use the percentile method, as in Brainiak. The classic surrogate
methods instead circle-shift or phase-randomize each time series (Lancaster
et al., 2018), preserving its temporal autocorrelation, and recompute ISC.

Runs on plain arrays; `BrainCollection.isc` wraps it for brain data.
`isc_permutation_test` exposes the same engine with `device='gpu'` and
leave-one-out ISC.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>ndarray \| DataFrame \| DataFrame</code> | Observations by subjects; ISC is computed across the columns. | *required*
`n_samples` | <code>int</code> | Number of bootstrap draws or surrogate permutations. Defaults to 5000. | <code>5000</code>
`summary` | <code>str</code> | `'median'` (default) or `'mean'`. | <code>'median'</code>
`method` | <code>str</code> | `'bootstrap'` (default), `'circle_shift'`, or `'phase_randomize'`. | <code>'bootstrap'</code>
`ci_percentile` | <code>int</code> | Confidence-interval width in percent. Defaults to 95. | <code>95</code>
`exclude_self_corr` | <code>bool</code> | Set self-correlations (the same subject bootstrapped twice) to NaN. Defaults to True. | <code>True</code>
`tail` | <code>int \| str</code> | `2` or `'two'` (two-tailed, default) or `1` or `'one'` (one-tailed, ISC > 0). | <code>2</code>
`metric` | <code>str</code> | Pairwise similarity metric; any metric accepted by sklearn's `pairwise_distances`. Defaults to `'correlation'`. | <code>'correlation'</code>
`return_null` | <code>bool</code> | Include the null distribution in the result. Defaults to False. | <code>False</code>
`n_jobs` | <code>int</code> | CPU workers for the resamples; -1 (default) picks the count from available memory. | <code>-1</code>
`random_state` | <code>int \| RandomState \| None</code> | Seed or generator for the resampling. | <code>None</code>
`progress_bar` | <code>bool</code> | Display a progress bar. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'isc'` (float, observed ISC), `'p'` (float), `'ci'` (tuple     `(lower, upper)`), `'device'`, and — when `return_null=True` —     `'null_dist'` (np.ndarray).

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

Hall, P., & Wilson, S. R. (1991). Two guidelines for bootstrap
hypothesis testing. Biometrics, 757-762.

Lancaster, G., Iatsenko, D., Pidde, A., Ticcinelli, V., & Stefanovska,
A. (2018). Surrogate data for hypothesis testing of physical systems.
Physics Reports, 748, 1-60.

</details>

(tasks-intersubject-isfc)=
### `isfc`

```python
isfc(data, method = 'average', n_jobs = -1)
```

Compute intersubject functional connectivity (ISFC) from per-subject matrices.

Uses the leave-one-out approach of Simony et al. (2016): for each subject,
average the other subjects' data and correlate every voxel/ROI time series
of the target subject with every voxel/ROI time series of that average.
Subjects are independent, so they are processed in parallel with joblib
unless `n_jobs=1`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>list[ndarray]</code> | One matrix per subject, each `(n_observations, n_features)` with identical shapes. | *required*
`method` | <code>str</code> | Only `'average'` (leave-one-out) is implemented. | <code>'average'</code>
`n_jobs` | <code>int</code> | Parallel workers; -1 (default) uses all cores, 1 runs serially. | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
<code>list[ndarray]</code> | One `(n_features, n_features)` ISFC matrix per     subject.

<details class="references" open markdown="1">
<summary>References</summary>

Simony, E., Honey, C. J., Chen, J., Lositsky, O., Yeshurun, Y., Wiesel,
A., & Hasson, U. (2016). Dynamic reconfiguration of the default mode
network during narrative comprehension. Nature Communications, 7, 12141.

</details>

(tasks-intersubject-isps)=
### `isps`

```python
isps(data, *, sampling_freq = 0.5, low_cut = 0.04, high_cut = 0.07, order = 5, pairwise = False)
```

Compute dynamic intersubject phase synchrony (ISPS) from an observations-by-subjects array.

Instantaneous phase synchrony across subjects for a single voxel/ROI time
series, after Glerean et al. (2012): the data are narrow-band filtered
(Butterworth) and Hilbert-transformed to get each subject's instantaneous
phase angle at every time point. Across subjects, the result gives the
mean phase angle, the mean resultant vector length, and a parametric
p-value from the Rayleigh test for circular uniformity (Fisher, 1995).
With `pairwise=True` these are computed on pairwise phase-angle differences
(inter-site phase coupling in the EEG literature) rather than on the raw
angles (inter-trial phase coupling).

The default band, 0.04-0.07 Hz, follows Glerean et al. (2012). It is close
to the "slow-4" band (0.025-0.067 Hz; Zuo et al., 2010; Penttonen &
Buzsáki, 2003) but excludes ~0.03 Hz, which carries aliased respiration
(Birn et al., 2006).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>ndarray \| DataFrame \| DataFrame</code> | Observations by subjects. | *required*
`sampling_freq` | <code>float</code> | Sampling frequency in Hz. Defaults to 0.5. | <code>0.5</code>
`low_cut` | <code>float</code> | Lower band-pass cutoff in Hz. Defaults to 0.04. | <code>0.04</code>
`high_cut` | <code>float</code> | Upper band-pass cutoff in Hz. Defaults to 0.07. | <code>0.07</code>
`order` | <code>int</code> | Butterworth filter order. Defaults to 5. | <code>5</code>
`pairwise` | <code>bool</code> | Compute on pairwise phase-angle differences instead of the raw phase angles. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'average_angle'` (np.ndarray, mean phase angle per time     point), `'vector_length'` (np.ndarray, mean resultant length per     time point), and `'p'` (np.ndarray, Rayleigh-test p-value per time     point).

<details class="references" open markdown="1">
<summary>References</summary>

Birn, R. M., Smith, M. A., Bandettini, P. A., & Diamond, J. B. (2006).
Separating respiratory-variation-related fluctuations from
neuronal-activity-related fluctuations in fMRI. NeuroImage, 31,
1536-1548.

Buzsáki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical
networks. Science, 304(5679), 1926-1929.

Fisher, N. I. (1995). Statistical analysis of circular data. Cambridge
University Press.

Glerean, E., Salmi, J., Lahnakoski, J. M., Jääskeläinen, I. P., & Sams,
M. (2012). Functional magnetic resonance imaging phase synchronization
as a measure of dynamic functional connectivity. Brain Connectivity,
2(2), 91-101.

</details>

(tasks-intersubject-isc-group)=
### `isc_group`

```python
isc_group(group1, group2, *, n_samples = 5000, summary = 'median', method = 'permute', ci_percentile = 95, exclude_self_corr = True, return_null = False, tail = 2, metric = 'correlation', n_jobs = -1, random_state = None, progress_bar = False)
```

Test the difference in pairwise intersubject correlation between two groups.

ISC within each group is summarized with the median, as Chen et al. (2016)
recommend (`summary='mean'` averages after the Fisher r-to-z transform), and
the observed statistic is `group1 - group2`.

Two null distributions are available. The default subject-wise permutation
(Chen et al., 2016) pools the subjects, computes pairwise similarity within
and between groups, then reshuffles the group labels and recomputes the
difference. The subject-wise bootstrap instead resamples subjects with
replacement within each group; a subject drawn twice correlates perfectly
with itself, so those entries are set to NaN when `exclude_self_corr=True`.
P-values use the percentile method (Hall & Wilson, 1991).

Runs on plain arrays; `isc_group_permutation_test` exposes the same engine
with `device='gpu'` and leave-one-out ISC.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`group1` | <code>ndarray \| DataFrame \| DataFrame</code> | Observations by subjects for the first group. | *required*
`group2` | <code>ndarray \| DataFrame \| DataFrame</code> | Observations by subjects for the second group (same number of observations). | *required*
`n_samples` | <code>int</code> | Number of permutations or bootstrap draws. Defaults to 5000. | <code>5000</code>
`summary` | <code>str</code> | `'median'` (default) or `'mean'`. | <code>'median'</code>
`method` | <code>str</code> | `'permute'` (default) or `'bootstrap'`. | <code>'permute'</code>
`ci_percentile` | <code>float</code> | Confidence-interval width in percent. Defaults to 95. | <code>95</code>
`exclude_self_corr` | <code>bool</code> | In the bootstrap, set self-correlations to NaN. Defaults to True. | <code>True</code>
`return_null` | <code>bool</code> | Include the null distribution in the result. Defaults to False. | <code>False</code>
`tail` | <code>int \| str</code> | `2` or `'two'` (two-tailed, default) or `1` or `'one'` (one-tailed, group1 > group2). | <code>2</code>
`metric` | <code>str</code> | Pairwise similarity metric; any metric accepted by sklearn's `pairwise_distances`. Defaults to `'correlation'`. | <code>'correlation'</code>
`n_jobs` | <code>int</code> | CPU workers for the resamples; -1 (default) picks the count from available memory. | <code>-1</code>
`random_state` | <code>int \| RandomState \| None</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | Display a progress bar. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'isc_group_difference'` (float, observed difference), `'p'`     (float), `'ci'` (tuple `(lower, upper)`), `'device'`, and — when     `return_null=True` — `'null_dist'` (np.ndarray).

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

Hall, P., & Wilson, S. R. (1991). Two guidelines for bootstrap
hypothesis testing. Biometrics, 757-762.

</details>

(tasks-intersubject-isc-permutation-test)=
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

(tasks-intersubject-isc-group-permutation-test)=
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
