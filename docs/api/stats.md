## `nltools.stats`

NeuroLearn Statistics Tools
===========================

Tools to help with statistical analyses.

**Functions:**

Name | Description
---- | -----------
[`align`](#nltools.stats.align) | Align subject data into a common response model. This function is a convenience wrapper around `HyperAlignment` and `SRM` classes
[`align_states`](#nltools.stats.align_states) | Align state weight maps using hungarian algorithm by minimizing pairwise distance between group states.This function uses the Hungarian algorithm for state alignment, which is different from aligning multiple subjects' data.
[`calc_bpm`](#nltools.stats.calc_bpm) | Calculate instantaneous BPM from beat to beat interval
[`compute_icc`](#nltools.stats.compute_icc) | Compute intraclass correlation coefficient (ICC).
[`compute_multivariate_similarity`](#nltools.stats.compute_multivariate_similarity) | Compute multivariate similarity via OLS regression.
[`compute_similarity`](#nltools.stats.compute_similarity) | Compute similarity between two data arrays.
[`correlation_permutation`](#nltools.stats.correlation_permutation) | Compute correlation and calculate p-value using permutation methods.
[`distance_correlation`](#nltools.stats.distance_correlation) | Compute the distance correlation between 2 arrays to test for multivariate dependence (linear or non-linear).
[`double_center`](#nltools.stats.double_center) | Double center a 2d array.
[`downsample`](#nltools.stats.downsample) | Downsample Polars or pandas DataFrame/Series to a new target frequency or number of samples using averaging.
[`fdr`](#nltools.stats.fdr) | Determine FDR threshold given a p value array and desired false
[`find_spikes`](#nltools.stats.find_spikes) | Function to identify spikes from fMRI Time Series Data
[`fisher_r_to_z`](#nltools.stats.fisher_r_to_z) | Use Fisher transformation to convert correlation to z score
[`fisher_z_to_r`](#nltools.stats.fisher_z_to_r) | Use Fisher transformation to convert correlation to z score
[`holm_bonf`](#nltools.stats.holm_bonf) | Compute corrected p-values based on the Holm-Bonferroni method, i.e. step-down procedure applying iteratively less correction to highest p-values. A bit more conservative than fdr, but much more powerful thanvanilla bonferroni.
[`isc`](#nltools.stats.isc) | Compute pairwise intersubject correlation from observations by subjects array.
[`isc_group`](#nltools.stats.isc_group) | Compute difference in intersubject correlation between groups.
[`isfc`](#nltools.stats.isfc) | Compute intersubject functional connectivity (ISFC) from a list of observation x feature matrices
[`isps`](#nltools.stats.isps) | Compute Dynamic Intersubject Phase Synchrony (ISPS from a observation by subject array)
[`make_cosine_basis`](#nltools.stats.make_cosine_basis) | Create a series of cosine basis functions for a discrete cosine
[`matrix_permutation`](#nltools.stats.matrix_permutation) | Permute 2-dimensional matrix correlation (mantel test).
[`multi_threshold`](#nltools.stats.multi_threshold) | Threshold test image by multiple p-values from p image.
[`one_sample_permutation`](#nltools.stats.one_sample_permutation) | One sample permutation test using randomization.
[`procrustes`](#nltools.stats.procrustes) | Procrustes analysis, a similarity test for two data sets. For more comprehensive procrustes-based alignment tasks, use `HyperAlignment` and `align()` instead.
[`procrustes_distance`](#nltools.stats.procrustes_distance) | Use procrustes super-position to perform a similarity test between 2 matrices. Matrices need to match in size on their first dimension only, as the smaller matrix on the second dimension will be padded with zeros. After aligning two matrices using the procrustes transformation, use the computed disparity between them (sum of squared error of elements) as a similarity metric. Shuffle the rows of one of the matrices and recompute the disparity to perform inference (Peres-Neto & Jackson, 2001).
[`threshold`](#nltools.stats.threshold) | Threshold test image by p-value from p image.
[`transform_pairwise`](#nltools.stats.transform_pairwise) | Transforms data into pairs with balanced labels for ranking
[`trim`](#nltools.stats.trim) | Trim a Polars or pandas DataFrame/Series by replacing outlier values with NaNs.
[`two_sample_permutation`](#nltools.stats.two_sample_permutation) | Independent sample permutation test.
[`u_center`](#nltools.stats.u_center) | U-center a 2d array. U-centering is a bias-corrected form of double-centering.
[`upsample`](#nltools.stats.upsample) | Upsample Polars or pandas DataFrame/Series to a new target frequency or number of samples using interpolation.
[`winsorize`](#nltools.stats.winsorize) | Winsorize a Polars or pandas DataFrame/Series with the largest/lowest value not considered outlier.
[`zscore`](#nltools.stats.zscore) | zscore every column in a pandas dataframe or series.



### Attributes

### Classes

### Functions#### `nltools.stats.align`

```python
align(data, method = 'deterministic_srm', n_features = None, axis = 0, *args, **kwargs)
```

Align subject data into a common response model. This function is a convenience wrapper around `HyperAlignment` and `SRM` classes

Can be used to hyperalign source data to target data using
Hyperalignment from Dartmouth (i.e., procrustes transformation; see
nltools.stats.procrustes) or Shared Response Model from Princeton (see
nltools.algorithms.srm). (see nltools.data.BrainData.align for aligning
a single Brain object to another). Common Model is shared response
model or centered target data. Transformed data can be back projected to
original data using Tranformation matrix. Inputs must be a list of BrainData
instances or numpy arrays (observations by features).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (list) A list of BrainData objects | *required*
`method` |  | (str) alignment method to use ['probabilistic_srm','deterministic_srm','procrustes'] | <code>'deterministic_srm'</code>
`n_features` |  | (int) number of features to align to common space. If None then will select number of voxels | <code>None</code>
`axis` |  | (int) axis to align on | <code>0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (dict) a dictionary containing a list of transformed subject matrices, a list of transformation matrices, the shared response matrix, and the intersubject correlation of the shared resposnes

**Examples:**

- Hyperalign using procrustes transform:
    >>> out = align(data, method='procrustes')
- Align using shared response model:
    >>> out = align(data, method='probabilistic_srm', n_features=None)
- Project aligned data into original data:
    >>> original_data = [np.dot(t.data,tm.T) for t,tm in zip(out['transformed'], out['transformation_matrix'])]

#### `nltools.stats.align_states`

```python
align_states(reference, target, metric = 'correlation', return_index = False, replace_zero_variance = False)
```

Align state weight maps using hungarian algorithm by minimizing pairwise distance between group states.This function uses the Hungarian algorithm for state alignment, which is different from aligning multiple subjects' data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`reference` |  | (np.array) reference pattern x state matrix | *required*
`target` |  | (np.array) target pattern x state matrix to align to reference | *required*
`metric` |  | (str) distance metric to use | <code>'correlation'</code>
`return_index` |  | (bool) return index if True, return remapped data if False | <code>False</code>
`replace_zero_variance` |  | (bool) transform a vector with zero variance to random numbers from a uniform distribution.                     Useful for when using correlation as a distance metric to avoid NaNs. | <code>False</code>

Returns:
    ordered_weights: (list) a list of reordered state X pattern matrices

#### `nltools.stats.calc_bpm`

```python
calc_bpm(beat_interval, sampling_freq)
```

Calculate instantaneous BPM from beat to beat interval

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`beat_interval` |  | (int) number of samples in between each beat             (typically R-R Interval) | *required*
`sampling_freq` |  | (float) sampling frequency in Hz | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`bpm` |  | (float) beats per minute for time interval

#### `nltools.stats.compute_icc`

```python
compute_icc(Y, icc_type = 'icc2')
```

Compute intraclass correlation coefficient (ICC).

This is the functional core implementation for ICC computation.
Used by BrainData.icc() to delegate computation to the functional core.

ICC Formulas are based on:
Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
assessing rater reliability. Psychological bulletin, 86(2), 420.

Code modified from nipype algorithms.icc
https://github.com/nipy/nipype/blob/master/nipype/algorithms/icc.py

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`Y` | <code>[ndarray](#numpy.ndarray)</code> | Data array, shape (n_features, n_subjects) or (n_subjects, n_features) If 2D with shape (n_subjects, n_features), will be transposed internally. Final shape should be (n_subjects, n_sessions) where: - n_subjects: number of subjects/rows - n_sessions: number of sessions/columns | *required*
`icc_type` | <code>[str](#str)</code> | Type of ICC to calculate - 'icc1': One-way random effects (subjects random, sessions treated as interchangeable) - 'icc2': Two-way random effects (subjects and sessions random) (default) - 'icc3': Two-way mixed effects (subjects random, sessions fixed) | <code>'icc2'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`float` |  | Intraclass correlation coefficient

**Examples:**

```pycon
>>> Y = np.random.randn(10, 5)  # 10 subjects, 5 sessions
>>> icc = compute_icc(Y, icc_type='icc2')
>>> isinstance(icc, (float, np.floating))
True
```

#### `nltools.stats.compute_multivariate_similarity`

```python
compute_multivariate_similarity(y, X, method = 'ols')
```

Compute multivariate similarity via OLS regression.

This is the functional core implementation for multivariate similarity computation.
Used by BrainData.multivariate_similarity() to delegate computation to the functional core.

Predicts spatial distribution of y from linear combination of X columns.
Computes OLS regression statistics including beta coefficients, t-statistics,
p-values, and residuals.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` | <code>[ndarray](#numpy.ndarray)</code> | Target data, shape (n_features,) - single image | *required*
`X` | <code>[ndarray](#numpy.ndarray)</code> | Predictor data, shape (n_features, n_predictors) where first column should be intercept (ones) if intercept is desired. If X does not include intercept, an intercept will be added automatically. | *required*
`method` | <code>[str](#str)</code> | Regression method (currently only 'ols' supported) | <code>'ols'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | Dictionary with keys: - 'beta': Regression coefficients including intercept, shape (n_predictors+1,) - 't': t-statistics, shape (n_predictors+1,) - 'p': p-values, shape (n_predictors+1,) - 'df': Degrees of freedom (int) - 'sigma': Residual standard deviation (float) - 'residual': Residuals, shape (n_features,)

**Examples:**

```pycon
>>> y = np.random.randn(100)
>>> X = np.random.randn(100, 5)
>>> result = compute_multivariate_similarity(y, X, method='ols')
>>> 'beta' in result
True
>>> result['beta'].shape
(6,)  # 5 predictors + intercept
```

#### `nltools.stats.compute_similarity`

```python
compute_similarity(data1, data2, method = 'correlation')
```

Compute similarity between two data arrays.

This is the functional core implementation for similarity computation.
Used by BrainData.similarity() to delegate computation to the functional core.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | First data array, shape (n_samples1, n_features) | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Second data array, shape (n_samples2, n_features) | *required*
`method` | <code>[str](#str)</code> | Type of similarity metric - 'correlation' or 'pearson': Pearson correlation - 'spearman' or 'rank_correlation': Spearman rank correlation - 'dot_product': Dot product - 'cosine': Cosine similarity | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
 | np.ndarray: Similarity matrix or vector - If data1.shape[0] == 1 and data2.shape[0] == 1: scalar - If data1.shape[0] == 1 or data2.shape[0] == 1: 1D array - Otherwise: 2D array shape (n_samples1, n_samples2)

**Examples:**

```pycon
>>> data1 = np.random.randn(10, 100)
>>> data2 = np.random.randn(5, 100)
>>> sim = compute_similarity(data1, data2, method='correlation')
>>> sim.shape
(10, 5)
```

#### `nltools.stats.correlation_permutation`

```python
correlation_permutation(data1, data2, method = 'permute', n_permute = 5000, metric = 'spearman', tail = 2, n_jobs = -1, return_perms = False, random_state = None)
```

Compute correlation and calculate p-value using permutation methods.

.. deprecated:: 0.6.0
    Use ``nltools.algorithms.inference.correlation_permutation_test`` or
    ``nltools.algorithms.inference.timeseries_correlation_permutation_test`` instead.

This function is a wrapper around `nltools.algorithms.inference.correlation_permutation_test`
and `nltools.algorithms.inference.timeseries_correlation_permutation_test` for backward
compatibility. The underlying implementation provides optimized CPU parallelization
and optional GPU acceleration.

'permute' method randomly shuffles one of the vectors. This method is recommended
for independent data. For timeseries data we recommend using 'circle_shift' or
'phase_randomize' methods.

Args:

    data1: (pd.DataFrame, pd.Series, np.array) dataset 1 to permute
    data2: (pd.DataFrame, pd.Series, np.array) dataset 2 to permute
    n_permute: (int) number of permutations
    metric: (str) type of association metric ['spearman','pearson',
            'kendall']
    method: (str) type of permutation ['permute', 'circle_shift', 'phase_randomize']
    random_state: (int, None, or np.random.RandomState) Initial random seed (default: None)
    tail: (int) either 1 for one-tail or 2 for two-tailed test (default: 2)
    n_jobs: (int) The number of CPUs to use to do the computation.
            -1 means all CPUs.
    return_parms: (bool) Return the permutation distribution along with the p-value; default False

Returns:

    stats: (dict) dictionary of permutation results ['correlation','p']

<details class="notes" open markdown="1">
<summary>Notes</summary>

This function uses the optimized inference module implementation which provides:
- 4-8× speedup with CPU parallelization (default)
- GPU acceleration available for 'permute' method with Pearson correlation
- Identical results to the original implementation

</details>

#### `nltools.stats.distance_correlation`

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

Name | Type | Description
---- | ---- | -----------
`results` | <code>[dict](#dict)</code> | dictionary of results (correlation, t, p, and df.) Optionally, covariance, x variance, and y variance

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

#### `nltools.stats.double_center`

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

Name | Type | Description
---- | ---- | -----------
`mat` | <code>[ndarray](#ndarray)</code> | double-centered version of input

**Examples:**

```pycon
>>> mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
>>> result = double_center(mat)
>>> np.allclose(result.mean(axis=0), 0)
True
>>> np.allclose(result.mean(axis=1), 0)
True
```

#### `nltools.stats.downsample`

```python
downsample(data, sampling_freq = None, target = None, target_type = 'samples', method = 'mean')
```

Downsample Polars or pandas DataFrame/Series to a new target frequency or number of samples using averaging.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pl.DataFrame, pl.Series, pd.DataFrame, pd.Series) data to downsample | *required*
`sampling_freq` |  | (float) Sampling frequency of data in hertz | <code>None</code>
`target` |  | (float) downsampling target | <code>None</code>
`target_type` |  | type of target can be [samples,seconds,hz] | <code>'samples'</code>
`method` |  | (str) type of downsample method ['mean','median'],     default: mean | <code>'mean'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (pl.DataFrame, pl.Series) downsampled data (same type as input)

#### `nltools.stats.fdr`

```python
fdr(p, q = 0.05)
```

Determine FDR threshold given a p value array and desired false
discovery rate q. Written by Tal Yarkoni

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`p` |  | (np.array) vector of p-values | *required*
`q` |  | (float) false discovery rate level | <code>0.05</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`fdr_p` |  | (float) p-value threshold based on independence or positive     dependence

#### `nltools.stats.find_spikes`

```python
find_spikes(data, global_spike_cutoff = 3, diff_spike_cutoff = 3)
```

Function to identify spikes from fMRI Time Series Data

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | BrainData or nibabel instance | *required*
`global_spike_cutoff` |  | (int,None) cutoff to identify spikes in global signal                  in standard deviations, None indicates do not calculate. | <code>3</code>
`diff_spike_cutoff` |  | (int,None) cutoff to identify spikes in average frame difference                  in standard deviations, None indicates do not calculate. | <code>3</code>

Returns:
    Polars DataFrame with spikes as indicator variables

#### `nltools.stats.fisher_r_to_z`

```python
fisher_r_to_z(r)
```

Use Fisher transformation to convert correlation to z score

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`r` |  | correlation coefficient(s) | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`z` |  | Fisher z-transformed correlation(s)

<details class="note" open markdown="1">
<summary>Note</summary>

Clips r values to (-1, 1) range to avoid invalid arctanh inputs

</details>

#### `nltools.stats.fisher_z_to_r`

```python
fisher_z_to_r(z)
```

Use Fisher transformation to convert correlation to z score

#### `nltools.stats.holm_bonf`

```python
holm_bonf(p, alpha = 0.05)
```

Compute corrected p-values based on the Holm-Bonferroni method, i.e. step-down procedure applying iteratively less correction to highest p-values. A bit more conservative than fdr, but much more powerful thanvanilla bonferroni.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`p` |  | (np.array) vector of p-values | *required*
`alpha` |  | (float) alpha level | <code>0.05</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`bonf_p` |  | (float) p-value threshold based on bonferroni     step-down procedure

#### `nltools.stats.isc`

```python
isc(data, n_samples = 5000, metric = 'median', method = 'bootstrap', ci_percentile = 95, exclude_self_corr = True, return_null = False, tail = 2, n_jobs = -1, random_state = None, sim_metric = 'correlation')
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

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pd.DataFrame, np.array) observations by subjects where isc is computed across subjects | *required*
`n_samples` |  | (int) number of random samples/bootstraps | <code>5000</code>
`metric` |  | (str) type of isc summary metric ['mean','median'] | <code>'median'</code>
`method` |  | (str) method to compute p-values ['bootstrap', 'circle_shift','phase_randomize'] (default: bootstrap) | <code>'bootstrap'</code>
`tail` |  | (int) either 1 for one-tail or 2 for two-tailed test (default: 2) | <code>2</code>
`n_jobs` |  | (int) The number of CPUs to use to do the computation. -1 means all CPUs. | <code>-1</code>
`return_null` |  | (bool) Return the permutation distribution along with the p-value; default False | <code>False</code>
`sim_metric` |  | (str) pairwise distance metric. See sklearn's pairwise_distances for valid inputs (default: correlation) | <code>'correlation'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`stats` |  | (dict) dictionary of permutation results ['isc', 'p', 'ci', 'null_distribution']

Notes
-----
This function is a wrapper around `isc_permutation_test` from the inference module,
which provides optimized implementations with CPU-parallel and GPU acceleration support.
Performance improvements: 4-8× speedup for CPU-parallel operations, 10-100× speedup
for GPU operations. See `nltools.algorithms.inference.isc.isc_permutation_test` for details.

#### `nltools.stats.isc_group`

```python
isc_group(group1, group2, n_samples = 5000, metric = 'median', method = 'permute', ci_percentile = 95, exclude_self_corr = True, return_null = False, tail = 2, n_jobs = -1, random_state = None)
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

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`group1` |  | (pd.DataFrame, np.array) observations by subjects where isc is computed across subjects | *required*
`group2` |  | (pd.DataFrame, np.array) observations by subjects where isc is computed across subjects | *required*
`n_samples` |  | (int) number of samples for permutation or bootstrapping | <code>5000</code>
`metric` |  | (str) type of isc summary metric ['mean','median'] | <code>'median'</code>
`method` |  | (str) method to compute p-values ['permute', 'bootstrap'] (default: permute) | <code>'permute'</code>
`ci_percentile` |  | (float) confidence interval percentile (default: 95) | <code>95</code>
`exclude_self_corr` |  | (bool) exclude self-correlations in bootstrap (default: True) | <code>True</code>
`return_null` |  | (bool) Return the permutation distribution along with the p-value; default False | <code>False</code>
`tail` |  | (int) either 1 for one-tail or 2 for two-tailed test (default: 2) | <code>2</code>
`n_jobs` |  | (int) The number of CPUs to use to do the computation. -1 means all CPUs. | <code>-1</code>
`random_state` |  | (int or RandomState) Random seed for reproducibility | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`stats` |  | (dict) dictionary of permutation results with keys: - 'isc_group_difference': Observed ISC difference (float or array) - 'p': P-value (float or array) - 'ci': Confidence interval tuple (lower, upper) - 'null_distribution': Null distribution (if return_null=True)

Notes
-----
This function is a wrapper around `nltools.algorithms.inference.isc.isc_group_permutation_test`
for backward compatibility. The underlying implementation provides optimized CPU parallelization
and optional GPU acceleration. For new code, consider using `isc_group_permutation_test` directly.

Performance improvements:
- 4-8× speedup with CPU-parallel backend (default)
- 10-30× speedup with GPU backend for voxel-wise LOO computation
- More memory efficient (no Adjacency object overhead)

#### `nltools.stats.isfc`

```python
isfc(data, method = 'average', n_jobs = -1)
```

Compute intersubject functional connectivity (ISFC) from a list of observation x feature matrices

This function uses the leave one out approach to compute ISFC (Simony et al., 2016).
For each subject, compute the cross-correlation between each voxel/roi
with the average of the rest of the subjects data. In other words,
compute the mean voxel/ROI response for all participants except the
target subject. Then compute the correlation between each ROI within
the target subject with the mean ROI response in the group average.

Simony, E., Honey, C. J., Chen, J., Lositsky, O., Yeshurun, Y., Wiesel, A., & Hasson, U. (2016).
Dynamic reconfiguration of the default mode network during narrative comprehension.
Nature communications, 7, 12141.

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

Notes
-----
This function now uses the optimized implementation from the inference module,
which provides efficient cross-correlation computation between matrix columns.

CPU parallelization is available via joblib when n_jobs > 1 or n_jobs=-1,
providing 4-8× speedup on multi-core machines. Each subject's ISFC computation
is independent and can be parallelized efficiently.

#### `nltools.stats.isps`

```python
isps(data, sampling_freq = 0.5, low_cut = 0.04, high_cut = 0.07, order = 5, pairwise = False)
```

Compute Dynamic Intersubject Phase Synchrony (ISPS from a observation by subject array)

This function computes the instantaneous intersubject phase synchrony for a single voxel/roi
timeseries. Requires multiple subjects. This method is largely based on that described by Glerean
et al., 2012 and performs a hilbert transform on narrow bandpass filtered timeseries (butterworth)
data to get the instantaneous phase angle. The function returns a dictionary containing the
average phase angle, the average vector length, and parametric p-values computed using the rayleigh test using circular
statistics (Fisher, 1993). If pairwise=True, then it will compute these on the pairwise phase angle differences,
if pairwise=False, it will compute these on the actual phase angles. This is called inter-site phase coupling
or inter-trial phase coupling respectively in the EEG literatures.

This function requires narrow band filtering your data. As a default we use the recommendations
by (Glerean et al., 2012) of .04-.07Hz. This is similar to the "slow-4" band (0.025–0.067 Hz)
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

#### `nltools.stats.make_cosine_basis`

```python
make_cosine_basis(nsamples, sampling_freq, filter_length, unit_scale = True, drop = 0)
```

Create a series of cosine basis functions for a discrete cosine
    transform. Based off of implementation in spm_filter and spm_dctmtx
    because scipy dct can only apply transforms but not return the basis
    functions. Like SPM, does not add constant (i.e. intercept), but does
    retain first basis (i.e. sigmoidal/linear drift)

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`nsamples` | <code>[int](#int)</code> | number of observations (e.g. TRs) | *required*
`sampling_freq` | <code>[float](#float)</code> | sampling frequency in hertz (i.e. 1 / TR) | *required*
`filter_length` | <code>[int](#int)</code> | length of filter in seconds | *required*
`unit_scale` | <code>[true](#true)</code> | assure that the basis functions are on the normalized range [-1, 1]; default True | <code>True</code>
`drop` | <code>[int](#int)</code> | index of which early/slow bases to drop if any; default is to drop constant (i.e. intercept) like SPM. Unlike SPM, retains first basis (i.e. linear/sigmoidal). Will cumulatively drop bases up to and inclusive of index provided (e.g. 2, drops bases 1 and 2) | <code>0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` | <code>[ndarray](#ndarray)</code> | nsamples x number of basis sets numpy array

#### `nltools.stats.matrix_permutation`

```python
matrix_permutation(data1, data2, n_permute = 5000, metric = 'spearman', how = 'upper', include_diag = False, tail = 2, n_jobs = -1, return_perms = False, random_state = None)
```

Permute 2-dimensional matrix correlation (mantel test).

.. deprecated:: 0.6.0
    Use ``nltools.algorithms.inference.matrix_permutation_test`` instead.

This function is a wrapper around `nltools.algorithms.inference.matrix_permutation_test`
for backward compatibility. The underlying implementation provides optimized CPU parallelization.

Chen, G. et al. (2016). Untangling the relatedness among correlations,
part I: nonparametric approaches to inter-subject correlation analysis
at the group level. Neuroimage, 142, 248-259.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` |  | (pd.DataFrame, np.array) square matrix | *required*
`data2` |  | (pd.DataFrame, np.array) square matrix | *required*
`n_permute` |  | (int) number of permutations | <code>5000</code>
`metric` |  | (str) type of association metric ['spearman','pearson',     'kendall'] | <code>'spearman'</code>
`how` |  | (str) whether to use the 'upper' (default), 'lower', or 'full' matrix. The default of 'upper' assumes both matrices are symmetric | <code>'upper'</code>
`include_diag` | <code>[bool](#bool)</code> | only applies if `how='full'`. Whether to include the diagonal elements in the comparison | <code>False</code>
`tail` |  | (int) either 1 for one-tail or 2 for two-tailed test   (default: 2) | <code>2</code>
`n_jobs` |  | (int) The number of CPUs to use to do the computation.     -1 means all CPUs. | <code>-1</code>
`return_parms` |  | (bool) Return the permutation distribution along with the p-value; default False | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`stats` |  | (dict) dictionary of permutation results ['correlation','p']

<details class="notes" open markdown="1">
<summary>Notes</summary>

This function uses the optimized inference module implementation which provides:
- 4-8× speedup with CPU parallelization (default)
- Identical results to the original implementation

</details>

#### `nltools.stats.multi_threshold`

```python
multi_threshold(t_map, p_map, thresh)
```

Threshold test image by multiple p-values from p image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_map` |  | (BrainData) BrainData instance of statistic metric (e.g., t-statistic, beta, etc) | *required*
`p_map` |  | (BrainData) BrainData instance of p-values | *required*
`thresh` |  | (list) list of p-values to threshold stat image | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | Thresholded BrainData instance with cumulative map - Positive values indicate how many thresholds were passed for positive stats - Negative values indicate how many thresholds were passed for negative stats

<details class="note" open markdown="1">
<summary>Note</summary>

This function provides unique cumulative threshold map functionality:
- Creates a single map showing which thresholds were passed
- Different from calling threshold() multiple times (which would give separate images)
- Useful for visualizing threshold hierarchies
- nilearn.threshold_img() does not support cumulative multi-threshold maps

</details>

#### `nltools.stats.one_sample_permutation`

```python
one_sample_permutation(data, n_permute = 5000, tail = 2, n_jobs = -1, return_perms = False, random_state = None)
```

One sample permutation test using randomization.

.. deprecated:: 0.6.0
    Use ``nltools.algorithms.inference.one_sample_permutation_test`` instead.

This function is a wrapper around `nltools.algorithms.inference.one_sample_permutation_test`
for backward compatibility. The underlying implementation provides optimized CPU parallelization
and optional GPU acceleration.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pd.DataFrame, pd.Series, np.array) data to permute | *required*
`n_permute` |  | (int) number of permutations | <code>5000</code>
`tail` |  | (int) either 1 for one-tail or 2 for two-tailed test (default: 2) | <code>2</code>
`n_jobs` |  | (int) The number of CPUs to use to do the computation.     -1 means all CPUs. | <code>-1</code>
`return_parms` |  | (bool) Return the permutation distribution along with the p-value; default False | *required*
`random_state` |  | (int, None, or np.random.RandomState) Initial random seed (default: None) | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`stats` |  | (dict) dictionary of permutation results ['mean','p']

<details class="notes" open markdown="1">
<summary>Notes</summary>

This function uses the optimized inference module implementation which provides:
- 4-8× speedup with CPU parallelization (default)
- 10-100× speedup with GPU acceleration (via backend='torch')
- Identical results to the original implementation

</details>

#### `nltools.stats.procrustes`

```python
procrustes(data1, data2)
```

Procrustes analysis, a similarity test for two data sets. For more comprehensive procrustes-based alignment tasks, use `HyperAlignment` and `align()` instead.

Each input matrix is a set of points or vectors (the rows of the matrix).
The dimension of the space is the number of columns of each matrix. Given
two identically sized matrices, procrustes standardizes both such that:
- :math:`tr(AA^{T}) = 1`.
- Both sets of points are centered around the origin.
Procrustes then applies the optimal transform to the second
matrix (including scaling/dilation, rotations, and reflections) to minimize
:math:`M^{2}=\sum(data1-data2)^{2}`, or the sum of the squares of the
pointwise differences between the two input datasets.
This function was not designed to handle datasets with different numbers of
datapoints (rows).  If two data sets have different dimensionality
(different number of columns), this function will add columns of zeros to
the smaller of the two.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1 ` |  | array_like Matrix, n rows represent points in k (columns) space `data1` is the reference data, after it is standardised, the data from `data2` will be transformed to fit the pattern in `data1` (must have >1 unique points). | *required*
`data2 ` |  | array_like n rows of data in k space to be fit to `data1`.  Must be the  same shape ``(numrows, numcols)`` as data1 (must have >1 unique points). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`mtx1` |  | array_like A standardized version of `data1`.
`mtx2` |  | array_like The orientation of `data2` that best fits `data1`. Centered, but not necessarily :math:`tr(AA^{T}) = 1`.
`disparity` |  | float :math:`M^{2}` as defined above.
`R` |  | (N, N) ndarray The matrix solution of the orthogonal Procrustes problem. Minimizes the Frobenius norm of dot(data1, R) - data2, subject to dot(R.T, R) == I.
`scale` |  | float Sum of the singular values of ``dot(data1.T, data2)``.

#### `nltools.stats.procrustes_distance`

```python
procrustes_distance(mat1, mat2, n_permute = 5000, tail = 2, n_jobs = -1, random_state = None)
```

Use procrustes super-position to perform a similarity test between 2 matrices. Matrices need to match in size on their first dimension only, as the smaller matrix on the second dimension will be padded with zeros. After aligning two matrices using the procrustes transformation, use the computed disparity between them (sum of squared error of elements) as a similarity metric. Shuffle the rows of one of the matrices and recompute the disparity to perform inference (Peres-Neto & Jackson, 2001).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mat1` | <code>[ndarray](#ndarray)</code> | 2d numpy array; must have same number of rows as mat2 | *required*
`mat2` | <code>[ndarray](#ndarray)</code> | 1d or 2d numpy array; must have same number of rows as mat1 | *required*
`n_permute` | <code>[int](#int)</code> | number of permutation iterations to perform | <code>5000</code>
`tail` | <code>[int](#int)</code> | either 1 for one-tailed or 2 for two-tailed test; default 2 | <code>2</code>
`n_jobs` | <code>[int](#int)</code> | The number of CPUs to use to do permutation; default -1 (all) | <code>-1</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`similarity` | <code>[float](#float)</code> | similarity between matrices bounded between 0 and 1
`pval` | <code>[float](#float)</code> | permuted p-value

#### `nltools.stats.threshold`

```python
threshold(stat, p, thr = 0.05, return_mask = False)
```

Threshold test image by p-value from p image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stat` |  | (BrainData) BrainData instance of arbitrary statistic metric   (e.g., beta, t, etc) | *required*
`p` |  | (BrainData) BrainData instance of p-values | *required*
`thr` |  | (float) p-value threshold to apply | <code>0.05</code>
`return_mask` |  | (bool) optionally return the thresholding mask; default False | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | Thresholded BrainData instance
`mask` |  | (optional) BrainData instance of thresholding mask if return_mask=True

<details class="note" open markdown="1">
<summary>Note</summary>

This function provides unique functionality not available in nilearn:
- Thresholds stat image based on p-values from separate p-value image
- Neither nilearn.threshold_img nor BrainData.threshold() support this
- BrainData.threshold() thresholds based on stat values themselves
- nilearn.threshold_img() thresholds based on image intensity values

</details>

#### `nltools.stats.transform_pairwise`

```python
transform_pairwise(X, y)
```

Transforms data into pairs with balanced labels for ranking
Transforms a n-class ranking problem into a two-class classification
problem. Subclasses implementing particular strategies for choosing
pairs should override this method.
In this method, all pairs are choosen, except for those that have the
same target value. The output is an array of balanced classes, i.e.
there are the same number of -1 as +1

Reference: "Large Margin Rank Boundaries for Ordinal Regression",
R. Herbrich, T. Graepel, K. Obermayer. Authors: Fabian Pedregosa
<fabian@fseoane.net> Alexandre Gramfort <alexandre.gramfort@inria.fr>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` |  | (np.array), shape (n_samples, n_features) The data | *required*
`y` |  | (np.array), shape (n_samples,) or (n_samples, 2) Target labels. If it's a 2D array, the second column represents the grouping of samples, i.e., samples with different groups will not be considered. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`X_trans` |  | (np.array), shape (k, n_feaures) Data as pairs, where k = n_samples * (n_samples-1)) / 2 if grouping values were not passed. If grouping variables exist, then returns values computed for each group.
`y_trans` |  | (np.array), shape (k,) Output class labels, where classes have values {-1, +1} If y was shape (n_samples, 2), then returns (k, 2) with groups on the second dimension.

#### `nltools.stats.trim`

```python
trim(data, cutoff = None)
```

Trim a Polars or pandas DataFrame/Series by replacing outlier values with NaNs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pl.DataFrame, pl.Series, pd.DataFrame, pd.Series) data to trim | *required*
`cutoff` |  | (dict) a dictionary with keys {'std':[low,high]} or     {'quantile':[low,high]} | <code>None</code>

Returns:
    out: (pl.DataFrame, pl.Series) trimmed data (same type as input)

#### `nltools.stats.two_sample_permutation`

```python
two_sample_permutation(data1, data2, n_permute = 5000, tail = 2, n_jobs = -1, return_perms = False, random_state = None)
```

Independent sample permutation test.

.. deprecated:: 0.6.0
    Use ``nltools.algorithms.inference.two_sample_permutation_test`` instead.

This function is a wrapper around `nltools.algorithms.inference.two_sample_permutation_test`
for backward compatibility. The underlying implementation provides optimized CPU parallelization
and optional GPU acceleration.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` |  | (pd.DataFrame, pd.Series, np.array) dataset 1 to permute | *required*
`data2` |  | (pd.DataFrame, pd.Series, np.array) dataset 2 to permute | *required*
`n_permute` |  | (int) number of permutations | <code>5000</code>
`tail` |  | (int) either 1 for one-tail or 2 for two-tailed test (default: 2) | <code>2</code>
`n_jobs` |  | (int) The number of CPUs to use to do the computation.     -1 means all CPUs. | <code>-1</code>
`return_parms` |  | (bool) Return the permutation distribution along with the p-value; default False | *required*

Returns:
    stats: (dict) dictionary of permutation results ['mean','p']

<details class="notes" open markdown="1">
<summary>Notes</summary>

This function uses the optimized inference module implementation which provides:
- 4-8× speedup with CPU parallelization (default)
- 10-100× speedup with GPU acceleration (via backend='torch')
- Identical results to the original implementation

</details>

#### `nltools.stats.u_center`

```python
u_center(mat: np.ndarray) -> np.ndarray
```

U-center a 2d array. U-centering is a bias-corrected form of double-centering.

U-centering corrects for bias that occurs with double-centering as the number
of dimensions increases. The diagonal is explicitly set to zero.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mat` | <code>[ndarray](#ndarray)</code> | 2d numpy array | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`mat` | <code>[ndarray](#ndarray)</code> | u-centered version of input

**Examples:**

```pycon
>>> mat = np.random.randn(5, 5)
>>> result = u_center(mat)
>>> np.allclose(np.diag(result), 0)
True
```

#### `nltools.stats.upsample`

```python
upsample(data, sampling_freq = None, target = None, target_type = 'samples', method = 'linear')
```

Upsample Polars or pandas DataFrame/Series to a new target frequency or number of samples using interpolation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pl.DataFrame, pl.Series, pd.DataFrame, pd.Series) data to upsample   (Note: will drop non-numeric columns from DataFrame) | *required*
`sampling_freq` |  | Sampling frequency of data in hertz | <code>None</code>
`target` |  | (float) upsampling target | <code>None</code>
`target_type` |  | (str) type of target can be [samples,seconds,hz] | <code>'samples'</code>
`method` |  | (str) ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']           where 'zero', 'slinear', 'quadratic' and 'cubic'           refer to a spline interpolation of zeroth, first,           second or third order  (default: linear) | <code>'linear'</code>

Returns:
    upsampled Polars DataFrame or Series (same type as input)

#### `nltools.stats.winsorize`

```python
winsorize(data, cutoff = None, replace_with_cutoff = True)
```

Winsorize a Polars or pandas DataFrame/Series with the largest/lowest value not considered outlier.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pl.DataFrame, pl.Series, pd.DataFrame, pd.Series) data to winsorize | *required*
`cutoff` |  | (dict) a dictionary with keys {'std':[low,high]} or     {'quantile':[low,high]} | <code>None</code>
`replace_with_cutoff` |  | (bool) If True, replace outliers with cutoff.                  If False, replaces outliers with closest                  existing values; (default: False) | <code>True</code>

Returns:
    out: (pl.DataFrame, pl.Series) winsorized data (same type as input)

#### `nltools.stats.zscore`

```python
zscore(df)
```

zscore every column in a pandas dataframe or series.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`df` |  | (pd.DataFrame) Pandas DataFrame instance | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`z_data` |  | (pd.DataFrame) z-scored pandas DataFrame or series instance

