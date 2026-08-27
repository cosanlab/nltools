(algorithms-signal-signal)=
## `signal`

Temporal signal processing — resampling, filtering, and basis functions.

**Methods:**

Name | Description
---- | -----------
[`calc_bpm`](#algorithms-signal-calc-bpm) | Calculate instantaneous BPM from beat to beat interval.
[`downsample`](#algorithms-signal-downsample) | Downsample a Polars DataFrame/Series to a new target frequency or number of samples using averaging.
[`make_cosine_basis`](#algorithms-signal-make-cosine-basis) | Create basis functions for a discrete cosine transform.
[`upsample`](#algorithms-signal-upsample) | Upsample a Polars DataFrame/Series to a new target frequency or number of samples using interpolation.



### Methods

(algorithms-signal-calc-bpm)=
#### `calc_bpm`

```python
calc_bpm(beat_interval, sampling_freq)
```

Calculate instantaneous BPM from beat to beat interval.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`beat_interval` |  | (int) number of samples in between each beat             (typically R-R Interval) | *required*
`sampling_freq` |  | (float) sampling frequency in Hz | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`bpm` |  | (float) beats per minute for time interval

(algorithms-signal-downsample)=
#### `downsample`

```python
downsample(data, *, sampling_freq = None, target = None, target_type = 'samples', method = 'mean')
```

Downsample a Polars DataFrame/Series to a new target frequency or number of samples using averaging.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pl.DataFrame, pl.Series) data to downsample | *required*
`sampling_freq` |  | (float) Sampling frequency of data in hertz | <code>None</code>
`target` |  | (float) downsampling target | <code>None</code>
`target_type` |  | type of target can be [samples,seconds,hz] | <code>'samples'</code>
`method` |  | (str) type of downsample method ['mean','median'],     default: mean | <code>'mean'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (pl.DataFrame, pl.Series) downsampled data (same type as input)

(algorithms-signal-make-cosine-basis)=
#### `make_cosine_basis`

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
`nsamples` | <code>[int](#int)</code> | number of observations (e.g. TRs) | *required*
`sampling_freq` | <code>[float](#float)</code> | sampling frequency in hertz (i.e. 1 / TR) | *required*
`filter_length` | <code>[int](#int)</code> | length of filter in seconds | *required*
`unit_scale` | <code>[bool](#bool)</code> | assure that the basis functions are on the normalized range [-1, 1]; default True | <code>True</code>
`drop` | <code>[int](#int)</code> | index of which early/slow bases to drop if any; default is to drop constant (i.e. intercept) like SPM. Unlike SPM, retains first basis (i.e. linear/sigmoidal). Will cumulatively drop bases up to and inclusive of index provided (e.g. 2, drops bases 1 and 2) | <code>0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` | <code>[ndarray](#ndarray)</code> | nsamples x number of basis sets numpy array

(algorithms-signal-upsample)=
#### `upsample`

```python
upsample(data, *, sampling_freq = None, target = None, target_type = 'samples', method = 'linear')
```

Upsample a Polars DataFrame/Series to a new target frequency or number of samples using interpolation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pl.DataFrame, pl.Series) data to upsample   (Note: will drop non-numeric columns from DataFrame) | *required*
`sampling_freq` |  | Sampling frequency of data in hertz | <code>None</code>
`target` |  | (float) upsampling target | <code>None</code>
`target_type` |  | (str) type of target can be [samples,seconds,hz] | <code>'samples'</code>
`method` |  | (str) ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']           where 'zero', 'slinear', 'quadratic' and 'cubic'           refer to a spline interpolation of zeroth, first,           second or third order  (default: linear) | <code>'linear'</code>

Returns:
    upsampled Polars DataFrame or Series (same type as input)

