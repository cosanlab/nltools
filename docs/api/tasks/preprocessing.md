---
title: Preprocessing & signal
label: page-tasks-preprocessing
---

Clean timeseries before modelling. Standardize or trim outliers, flag motion spikes, resample to another sampling rate, build cosine drift regressors. Every function takes and returns numpy arrays or DataFrames. The [BrainData](../data/brain_data.md) and [DesignMatrix](../data/design_matrix.md) methods of the same name call them.

**Functions:**

Name | Description
---- | -----------
[`zscore`](#tasks-preprocessing-zscore) | Z-score every column of a Polars or pandas DataFrame/Series.
[`trim`](#tasks-preprocessing-trim) | Trim a Polars DataFrame/Series by replacing outlier values with NaNs.
[`winsorize`](#tasks-preprocessing-winsorize) | Winsorize a Polars DataFrame/Series with the largest/lowest value not considered outlier.
[`find_spikes`](#tasks-preprocessing-find-spikes) | Identify spikes (motion artifacts, intensity outliers) in 4D fMRI data.
[`downsample`](#tasks-preprocessing-downsample) | Downsample a Polars DataFrame/Series to a new target frequency or number of samples using averaging.
[`upsample`](#tasks-preprocessing-upsample) | Upsample a Polars DataFrame/Series to a new target frequency or number of samples using interpolation.
[`make_cosine_basis`](#tasks-preprocessing-make-cosine-basis) | Create basis functions for a discrete cosine transform.
[`calc_bpm`](#tasks-preprocessing-calc-bpm) | Calculate instantaneous BPM from beat to beat interval.

## Functions

(tasks-preprocessing-zscore)=
### `zscore`

```python
zscore(data)
```

Z-score every column of a Polars or pandas DataFrame/Series.

Pandas inputs are converted to Polars; the result is always Polars (a
DataFrame for DataFrame input, a Series for Series input).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>DataFrame \| Series \| DataFrame \| Series</code> | Data to z-score. | *required*

**Returns:**

Type | Description
---- | -----------
<code>DataFrame \| Series</code> | Same shape as the input, each column z-scored     with the sample standard deviation (ddof=1).

(tasks-preprocessing-trim)=
### `trim`

```python
trim(data, cutoff = None)
```

Trim a Polars DataFrame/Series by replacing outlier values with NaNs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>DataFrame \| Series</code> | Data to trim. | *required*
`cutoff` | <code>dict</code> | A dictionary with keys `{'std': [low, high]}` or `{'quantile': [low, high]}`. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>DataFrame \| Series</code> | Trimmed data (outliers replaced with NaN), the     same type as the input.

(tasks-preprocessing-winsorize)=
### `winsorize`

```python
winsorize(data, cutoff = None, replace_with_cutoff = True)
```

Winsorize a Polars DataFrame/Series with the largest/lowest value not considered outlier.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>DataFrame \| Series</code> | Data to winsorize. | *required*
`cutoff` | <code>dict</code> | A dictionary with keys `{'std': [low, high]}` or `{'quantile': [low, high]}`. | <code>None</code>
`replace_with_cutoff` | <code>bool</code> | If True, replace outliers with the cutoff value; if False, replace them with the closest existing values (default: True). | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>DataFrame \| Series</code> | Winsorized data, the same type as the input.

(tasks-preprocessing-find-spikes)=
### `find_spikes`

```python
find_spikes(data, global_spike_cutoff = 3, diff_spike_cutoff = 3, *, TR: float | None = None, sampling_freq: float | None = None)
```

Identify spikes (motion artifacts, intensity outliers) in 4D fMRI data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image</code> | 4D functional data. | *required*
`global_spike_cutoff` | <code>float \| None</code> | Cutoff in standard deviations for spikes in the per-TR global mean signal; None skips this detector. Defaults to 3. | <code>3</code>
`diff_spike_cutoff` | <code>float \| None</code> | Cutoff in standard deviations for spikes in the per-TR mean absolute frame-to-frame difference; None skips this detector. Defaults to 3. | <code>3</code>
`TR` | <code>float \| None</code> | Repetition time in seconds; sets the returned DesignMatrix's `sampling_freq` for downstream `.append()` / `.convolve()`. Pass at most one of `TR` and `sampling_freq`. | <code>None</code>
`sampling_freq` | <code>float \| None</code> | Sampling frequency in Hz (1 / TR). See `TR`. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#page-data-design-matrix)</code> | One indicator column per detected spike TR, named     `.nl_global_spike{n}` / `.nl_diff_spike{n}` in the reserved namespace     for generated columns (see `RESERVED_PREFIX`) and pre-marked as     confounds. Row position is the time axis. A volume flagged by both     detectors yields identical one-hot columns, so only the     `.nl_global_spike*` one is kept. Without `TR` / `sampling_freq` the     result has `sampling_freq=None` and can still be appended to a     DesignMatrix that has one.

(tasks-preprocessing-downsample)=
### `downsample`

```python
downsample(data, *, sampling_freq = None, target = None, target_type = 'samples', method = 'mean')
```

Downsample a Polars DataFrame/Series to a new target frequency or number of samples using averaging.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>DataFrame \| Series</code> | Data to downsample. | *required*
`sampling_freq` | <code>float</code> | Sampling frequency of the data in Hz. | <code>None</code>
`target` | <code>float</code> | Downsampling target. | <code>None</code>
`target_type` | <code>str</code> | Unit of `target`, one of 'samples', 'seconds', or 'hz'. Defaults to 'samples'. | <code>'samples'</code>
`method` | <code>str</code> | Aggregation within each bin, 'mean' or 'median'. Defaults to 'mean'. | <code>'mean'</code>

**Returns:**

Type | Description
---- | -----------
<code>DataFrame \| Series</code> | Downsampled data (same type as input).

(tasks-preprocessing-upsample)=
### `upsample`

```python
upsample(data, *, sampling_freq = None, target = None, target_type = 'samples', method = 'linear')
```

Upsample a Polars DataFrame/Series to a new target frequency or number of samples using interpolation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>DataFrame \| Series</code> | Data to upsample. Non-numeric columns are dropped from a DataFrame. | *required*
`sampling_freq` | <code>float</code> | Sampling frequency of the data in Hz. | <code>None</code>
`target` | <code>float</code> | Upsampling target. | <code>None</code>
`target_type` | <code>str</code> | Unit of `target`, one of 'samples', 'seconds', or 'hz'. | <code>'samples'</code>
`method` | <code>str</code> | Interpolation method, one of 'linear', 'nearest', 'zero', 'slinear', 'quadratic', or 'cubic'; 'zero', 'slinear', 'quadratic' and 'cubic' refer to spline interpolation of zeroth, first, second or third order (default: 'linear'). | <code>'linear'</code>

**Returns:**

Type | Description
---- | -----------
<code>DataFrame \| Series</code> | Upsampled data, the same type as the input.

(tasks-preprocessing-make-cosine-basis)=
### `make_cosine_basis`

```python
make_cosine_basis(nsamples, sampling_freq, filter_length, unit_scale = True, drop = 0)
```

Create basis functions for a discrete cosine transform.

Based on the implementation in ``spm_filter`` and ``spm_dctmtx`` because
scipy DCT can only apply transforms but not return the basis functions. Like
SPM, this does not add a constant (i.e. intercept), but does retain the first
basis (i.e. sigmoidal/linear drift).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`nsamples` | <code>int</code> | Number of observations (e.g. TRs). | *required*
`sampling_freq` | <code>float</code> | Sampling frequency in Hz (i.e. 1 / TR). | *required*
`filter_length` | <code>int</code> | Filter length in seconds. | *required*
`unit_scale` | <code>bool</code> | Scale the basis functions to the range [-1, 1]. Defaults to True. | <code>True</code>
`drop` | <code>int</code> | Number of leading (slowest) bases to drop after the constant is removed; `drop=2` removes the first two. Defaults to 0, which keeps the linear/sigmoidal drift basis that SPM discards. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Basis matrix of shape (nsamples, n_bases).

(tasks-preprocessing-calc-bpm)=
### `calc_bpm`

```python
calc_bpm(beat_interval, sampling_freq)
```

Calculate instantaneous BPM from beat to beat interval.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`beat_interval` | <code>int</code> | Number of samples between beats (typically the R-R interval). | *required*
`sampling_freq` | <code>float</code> | Sampling frequency in Hz. | *required*

**Returns:**

Type | Description
---- | -----------
<code>float</code> | Beats per minute for the time interval.
