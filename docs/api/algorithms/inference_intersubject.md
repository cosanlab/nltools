(algorithms-inference-intersubject-intersubject)=
## `intersubject`

Intersubject correlation, functional connectivity, and phase synchrony.

**Methods:**

Name | Description
---- | -----------
[`isc`](#algorithms-inference-intersubject-isc) | Compute pairwise intersubject correlation from observations by subjects array.
[`isc_group`](#algorithms-inference-intersubject-isc-group) | Compute difference in intersubject correlation between groups.
[`isfc`](#algorithms-inference-intersubject-isfc) | Compute intersubject functional connectivity (ISFC) from a list of observation x feature matrices.
[`isps`](#algorithms-inference-intersubject-isps) | Compute dynamic intersubject phase synchrony (ISPS) from an observations-by-subjects array.



### Methods

(algorithms-inference-intersubject-isc)=
#### `isc`

```python
isc(data, *, n_samples = 5000, summary = 'median', method = 'bootstrap', ci_percentile = 95, exclude_self_corr = True, tail = 2, metric = 'correlation', return_null = False, n_jobs = -1, random_state = None, progress_bar = False)
```

Compute pairwise intersubject correlation from observations by subjects array.

This function computes pairwise intersubject correlations (ISC) using the median as recommended by Chen
et al., 2016). However, if the mean is preferred, we compute the mean correlation after performing
the fisher r-to-z transformation and then convert back to correlations to minimize artificially
inflating the correlation values.

There are currently three different methods to compute p-values. These include the classic methods for
computing permuted time-series by either circle-shifting the data or phase-randomizing the data
(see Lancaster et al., 2018). These methods create random surrogate data while preserving the temporal
autocorrelation inherent to the signal. By default, we use the subject-wise bootstrap method from
Chen et al., 2016. Instead of recomputing the pairwise ISC using circle_shift or phase_randomization methods,
this approach uses the computationally more efficient method of bootstrapping the subjects
and computing a new pairwise similarity matrix with randomly selected subjects with replacement.
If the same subject is selected multiple times, we set the perfect correlation to a nan with
(exclude_self_corr=True). We compute the p-values using the percentile method using the same
method in Brainiak.

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C., Israel, R. B.,
& Cox, R. W. (2016). Untangling the relatedness among correlations, part I:
nonparametric approaches to inter-subject correlation analysis at the group level.
NeuroImage, 142, 248-259.

Hall, P., & Wilson, S. R. (1991). Two guidelines for bootstrap hypothesis testing.
Biometrics, 757-762.

Lancaster, G., Iatsenko, D., Pidde, A., Ticcinelli, V., & Stefanovska, A. (2018).
Surrogate data for hypothesis testing of physical systems. Physics Reports, 748, 1-60.

This function is a wrapper around `isc_permutation_test` from the inference module,
which provides optimized implementations with CPU-parallel and GPU acceleration support.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pd.DataFrame, np.array) observations by subjects where isc is computed across subjects | *required*
`n_samples` |  | (int) number of random samples/bootstraps | <code>5000</code>
`summary` |  | (str) type of isc summary statistic ['mean','median'] (default: median) | <code>'median'</code>
`method` |  | (str) method to compute p-values ['bootstrap', 'circle_shift','phase_randomize'] (default: bootstrap) | <code>'bootstrap'</code>
`ci_percentile` |  | (int) confidence-interval width in percent for the bootstrap CI (default: 95) | <code>95</code>
`exclude_self_corr` |  | (bool) set self-correlations (same subject bootstrapped twice) to nan (default: True) | <code>True</code>
`tail` |  | (int | str) 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction) | <code>2</code>
`metric` |  | (str) pairwise distance metric. See sklearn's pairwise_distances for valid inputs (default: correlation) | <code>'correlation'</code>
`return_null` |  | (bool) Return the permutation distribution along with the p-value; default False | <code>False</code>
`n_jobs` |  | (int) The number of CPUs to use to do the computation. -1 means all CPUs. | <code>-1</code>
`random_state` |  | (int, np.random.RandomState, or None) seed or generator for the resampling; default None | <code>None</code>
`progress_bar` |  | (bool) If True, display a progress bar. Default False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`stats` |  | (dict) dictionary of permutation results ['isc', 'p', 'ci', 'null_dist']

(algorithms-inference-intersubject-isc-group)=
#### `isc_group`

```python
isc_group(group1, group2, *, n_samples = 5000, summary = 'median', method = 'permute', ci_percentile = 95, exclude_self_corr = True, return_null = False, tail = 2, metric = 'correlation', n_jobs = -1, random_state = None, progress_bar = False)
```

Compute difference in intersubject correlation between groups.

This function computes pairwise intersubject correlations (ISC) using the median as recommended by Chen
et al., 2016). However, if the mean is preferred, we compute the mean correlation after performing
the fisher r-to-z transformation and then convert back to correlations to minimize artificially
inflating the correlation values.

There are currently two different methods to compute p-values. By default, we use the subject-wise permutation
method recommended Chen et al., 2016. This method combines the two groups and computes pairwise similarity both
within and between the groups. Then the group labels are permuted and the mean difference between the two groups
are recomputed to generate a null distribution. The second method uses subject-wise bootstrapping, where a new
pairwise similarity matrix with randomly selected subjects with replacement is created separately for each group
and the ISC difference between these groups is used to generate a null distribution. If the same subject is
selected multiple times, we set the perfect correlation to a nan with (exclude_self_corr=True). We compute the
p-values using the percentile method (Hall & Wilson, 1991).

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C., Israel, R. B.,
& Cox, R. W. (2016). Untangling the relatedness among correlations, part I:
nonparametric approaches to inter-subject correlation analysis at the group level.
NeuroImage, 142, 248-259.

Hall, P., & Wilson, S. R. (1991). Two guidelines for bootstrap hypothesis testing.
Biometrics, 757-762.

This function is a thin wrapper around `isc_group_permutation_test` from the inference
module (which provides optimized CPU parallelization and optional GPU acceleration),
pinning the classic pairwise behavior and the `n_samples` vocabulary.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`group1` |  | (pd.DataFrame, np.array) observations by subjects where isc is computed across subjects | *required*
`group2` |  | (pd.DataFrame, np.array) observations by subjects where isc is computed across subjects | *required*
`n_samples` |  | (int) number of samples for permutation or bootstrapping | <code>5000</code>
`summary` |  | (str) type of isc summary statistic ['mean','median'] (default: median) | <code>'median'</code>
`method` |  | (str) method to compute p-values ['permute', 'bootstrap'] (default: permute) | <code>'permute'</code>
`ci_percentile` |  | (float) confidence interval percentile (default: 95) | <code>95</code>
`exclude_self_corr` |  | (bool) exclude self-correlations in bootstrap (default: True) | <code>True</code>
`return_null` |  | (bool) Return the permutation distribution along with the p-value; default False | <code>False</code>
`tail` |  | (int | str) 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction) | <code>2</code>
`metric` |  | (str) pairwise distance metric. See sklearn's pairwise_distances for valid inputs (default: correlation) | <code>'correlation'</code>
`n_jobs` |  | (int) The number of CPUs to use to do the computation. -1 means all CPUs. | <code>-1</code>
`random_state` |  | (int or RandomState) Random seed for reproducibility | <code>None</code>
`progress_bar` |  | (bool) If True, display a progress bar. Default False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`stats` |  | (dict) dictionary of permutation results with keys: - 'isc_group_difference': Observed ISC difference (float or array) - 'p': P-value (float or array) - 'ci': Confidence interval tuple (lower, upper) - 'null_dist': Null distribution (if return_null=True)

(algorithms-inference-intersubject-isfc)=
#### `isfc`

```python
isfc(data, method = 'average', n_jobs = -1)
```

Compute intersubject functional connectivity (ISFC) from a list of observation x feature matrices.

This function uses the leave one out approach to compute ISFC (Simony et al., 2016).
For each subject, compute the cross-correlation between each voxel/roi
with the average of the rest of the subjects data. In other words,
compute the mean voxel/ROI response for all participants except the
target subject. Then compute the correlation between each ROI within
the target subject with the mean ROI response in the group average.

Simony, E., Honey, C. J., Chen, J., Lositsky, O., Yeshurun, Y., Wiesel, A., & Hasson, U. (2016).
Dynamic reconfiguration of the default mode network during narrative comprehension.
Nature communications, 7, 12141.

This function now uses the optimized implementation from the inference module,
which provides efficient cross-correlation computation between matrix columns.
CPU parallelization is available via joblib when n_jobs > 1 or n_jobs=-1.
Each subject's ISFC computation is independent and can be parallelized efficiently.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | list of subject matrices (observations x voxels/rois) | *required*
`method` |  | approach to computing ISFC. 'average' uses leave one out | <code>'average'</code>
`n_jobs` |  | (int) Number of parallel jobs to use. -1 means all available cores.     Default is -1 (parallel execution by default, consistent with other stats functions). | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
 | list of subject ISFC matrices

(algorithms-inference-intersubject-isps)=
#### `isps`

```python
isps(data, *, sampling_freq = 0.5, low_cut = 0.04, high_cut = 0.07, order = 5, pairwise = False)
```

Compute dynamic intersubject phase synchrony (ISPS) from an observations-by-subjects array.

This function computes the instantaneous intersubject phase synchrony for a single voxel/roi
timeseries. Requires multiple subjects. This method is largely based on that described by Glerean
et al., 2012 and performs a hilbert transform on narrow bandpass filtered timeseries (butterworth)
data to get the instantaneous phase angle. The function returns a dictionary containing the
average phase angle, the average vector length, and parametric p-values computed using the rayleigh test using circular
statistics (Fisher, 1993). If pairwise=True, then it will compute these on the pairwise phase angle differences,
if pairwise=False, it will compute these on the actual phase angles. This is called inter-site phase coupling
or inter-trial phase coupling respectively in the EEG literatures.

This function requires narrow band filtering your data. As a default we use the recommendations
by (Glerean et al., 2012) of .04-.07Hz. This is similar to the "slow-4" band (0.025–0.067 Hz)
described by (Zuo et al., 2010; Penttonen & Buzsáki, 2003), but excludes the .03 band, which has been
demonstrated to contain aliased respiration signals (Birn, 2006).

Birn RM, Smith MA, Bandettini PA, Diamond JB. 2006. Separating respiratory-variation-related
fluctuations from neuronal-activity- related fluctuations in fMRI. Neuroimage 31:1536–1548.

Buzsáki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical networks. Science,
304(5679), 1926-1929.

Fisher, N. I. (1995). Statistical analysis of circular data. cambridge university press.

Glerean, E., Salmi, J., Lahnakoski, J. M., Jääskeläinen, I. P., & Sams, M. (2012).
Functional magnetic resonance imaging phase synchronization as a measure of dynamic
functional connectivity. Brain connectivity, 2(2), 91-101.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pd.DataFrame, np.ndarray) observations x subjects data | *required*
`sampling_freq` |  | (float) sampling freqency of data in Hz | <code>0.5</code>
`low_cut` |  | (float) lower bound cutoff for high pass filter | <code>0.04</code>
`high_cut` |  | (float) upper bound cutoff for low pass filter | <code>0.07</code>
`order` |  | (int) filter order for butterworth bandpass | <code>5</code>
`pairwise` |  | (bool) compute phase angle coherence on pairwise phase angle differences     or on raw phase angle. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
 | dictionary with mean phase angle, vector length, and rayleigh statistic

