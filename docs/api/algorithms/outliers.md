(algorithms-outliers-outliers)=
## `outliers`

Outlier detection, robust statistics, and data normalization.

**Methods:**

Name | Description
---- | -----------
[`find_spikes`](#algorithms-outliers-find-spikes) | Identify spikes (motion artifacts, intensity outliers) in 4D fMRI data.
[`trim`](#algorithms-outliers-trim) | Trim a Polars DataFrame/Series by replacing outlier values with NaNs.
[`winsorize`](#algorithms-outliers-winsorize) | Winsorize a Polars DataFrame/Series with the largest/lowest value not considered outlier.
[`zscore`](#algorithms-outliers-zscore) | Z-score every column of a Polars or pandas DataFrame/Series.



### Methods

(algorithms-outliers-find-spikes)=
#### `find_spikes`

```python
find_spikes(data, global_spike_cutoff = 3, diff_spike_cutoff = 3, *, TR: float | None = None, sampling_freq: float | None = None)
```

Identify spikes (motion artifacts, intensity outliers) in 4D fMRI data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | BrainData or nibabel instance | *required*
`global_spike_cutoff` |  | (int, None) cutoff in std-deviations for spikes in the per-TR global signal. None to skip. | <code>3</code>
`diff_spike_cutoff` |  | (int, None) cutoff in std-deviations for spikes in the per-TR mean absolute frame-to-frame difference. None to skip. | <code>3</code>
`TR` | <code>[float](#float) \| None</code> | Repetition time in seconds. Sets the returned DesignMatrix's sampling_freq for downstream `.append(...)` / `.convolve()`. Pass exactly one of `TR` or `sampling_freq`. | <code>None</code>
`sampling_freq` | <code>[float](#float) \| None</code> | Sampling frequency in Hz (= 1/TR). See `TR`. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` |  | one indicator column per detected spike TR, named
 |  | ``.nl_global_spike{n}`` / ``.nl_diff_spike{n}`` in the reserved
 |  | namespace for generated columns (see `RESERVED_PREFIX`), with all
 |  | spike columns pre-marked as confounds. The two detectors run
 |  | independently, so a single bad volume is routinely caught by both;
 |  | those detections are bitwise-identical one-hot columns, and only one
 |  | is kept (the ``.nl_global_spike*`` name, a deterministic tie-break —
 |  | the column values are the same either way). Row position is the time
 |  | axis (no separate `TR` index column — that was a pandas-era
 |  | artifact). When `TR` / `sampling_freq` aren't provided the DM has
 |  | `sampling_freq=None`; you can still `.append()` it onto a DM that
 |  | does have one.

(algorithms-outliers-trim)=
#### `trim`

```python
trim(data, cutoff = None)
```

Trim a Polars DataFrame/Series by replacing outlier values with NaNs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pl.DataFrame, pl.Series) data to trim | *required*
`cutoff` |  | (dict) a dictionary with keys {'std':[low,high]} or     {'quantile':[low,high]} | <code>None</code>

Returns:
    out: (pl.DataFrame, pl.Series) trimmed data (same type as input)

(algorithms-outliers-winsorize)=
#### `winsorize`

```python
winsorize(data, cutoff = None, replace_with_cutoff = True)
```

Winsorize a Polars DataFrame/Series with the largest/lowest value not considered outlier.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pl.DataFrame, pl.Series) data to winsorize | *required*
`cutoff` |  | (dict) a dictionary with keys {'std':[low,high]} or     {'quantile':[low,high]} | <code>None</code>
`replace_with_cutoff` |  | (bool) If True, replace outliers with cutoff.                  If False, replaces outliers with closest                  existing values; (default: True) | <code>True</code>

Returns:
    out: (pl.DataFrame, pl.Series) winsorized data (same type as input)

(algorithms-outliers-zscore)=
#### `zscore`

```python
zscore(data)
```

Z-score every column of a Polars or pandas DataFrame/Series.

Accepts pandas inputs at the boundary for convenience and converts to
Polars internally. Always returns Polars output (DataFrame or Series,
matching the input shape).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | pl.DataFrame, pl.Series, pd.DataFrame, or pd.Series. | *required*

**Returns:**

Type | Description
---- | -----------
 | pl.DataFrame or pl.Series with each column z-scored using sample
 | standard deviation (ddof=1), matching the input shape.

