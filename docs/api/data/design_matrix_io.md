---
title: data.designmatrix.io
---

Provide DesignMatrix I/O and visualization functions.

Standalone functions extracted from DesignMatrix methods.
Each takes a DesignMatrix instance (`dm`) as its first argument.

**Methods:**

Name | Description
---- | -----------
[`events_to_dm`](#data-design-matrix-io-events-to-dm) | Convert a BIDS events table to boxcar regressors aligned to TRs.
[`load_from_file`](#data-design-matrix-io-load-from-file) | Read a TSV/CSV into the frame a DesignMatrix wraps.
[`read_h5`](#data-design-matrix-io-read-h5) | Read a DesignMatrix HDF5 file written by `write_h5`.
[`separator_for_path`](#data-design-matrix-io-separator-for-path) | Return the delimiter a text DesignMatrix file uses, from its extension.
[`to_numpy`](#data-design-matrix-io-to-numpy) | Convert a DesignMatrix to a NumPy array.
[`to_pandas`](#data-design-matrix-io-to-pandas) | Convert DesignMatrix to pandas DataFrame.
[`write`](#data-design-matrix-io-write) | Write DesignMatrix to file.
[`write_h5`](#data-design-matrix-io-write-h5) | Write DesignMatrix to HDF5 file with metadata.

## Methods

(data-design-matrix-io-events-to-dm)=
### `events_to_dm`

```python
events_to_dm(events: pl.DataFrame | pd.DataFrame, *, run_length: int, sampling_freq: float) -> pl.DataFrame
```

Convert a BIDS events table to boxcar regressors aligned to TRs.

Uses `nilearn.glm.first_level.make_first_level_design_matrix` with
`hrf_model=None` to sample events onto the TR grid without HRF
convolution — the caller is expected to call `DesignMatrix.convolve()`
explicitly when convolution is desired. Drops nilearn's auto-added
`constant` column; users add the intercept via `add_poly(0)`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`events` | <code>[DataFrame](#polars.DataFrame) \| [DataFrame](#pandas.DataFrame)</code> | pandas or polars DataFrame with BIDS columns `onset`, `duration`, `trial_type` (required); `modulation` is passed through if present. | *required*
`run_length` | <code>[int](#int)</code> | Number of TRs the run contains. | *required*
`sampling_freq` | <code>[float](#float)</code> | Sampling frequency in Hz (= 1/TR). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#polars.DataFrame)</code> | One column per unique `trial_type`, values in     {0, modulation} indicating where each condition is active.

(data-design-matrix-io-load-from-file)=
### `load_from_file`

```python
load_from_file(path: str | Path, *, run_length: int | str, sampling_freq: float) -> tuple[pl.DataFrame, bool]
```

Read a TSV/CSV into the frame a DesignMatrix wraps.

Dispatches on column inspection:

- `onset` and `duration` both present → BIDS events → boxcar DM via
  `events_to_dm` (unconvolved; caller convolves later).
- otherwise → tabular file (confounds / nuisance regressors) read as-is.

`run_length='infer'` is accepted only for the tabular path; events
files must provide an explicit integer (they have a variable row count
per run, unlike confounds which are 1 row per TR).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`path` | <code>[str](#str) \| [Path](#pathlib.Path)</code> | Path to a `.tsv` or `.csv` file. | *required*
`run_length` | <code>[int](#int) \| [str](#str)</code> | Number of TRs, or `'infer'` for tabular inputs. | *required*
`sampling_freq` | <code>[float](#float)</code> | Sampling frequency in Hz (= 1/TR). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[DataFrame](#polars.DataFrame), [bool](#bool)]</code> | `(frame, is_events)` — `is_events` signals to     the caller that the columns are experimental regressors rather than     nuisance.

(data-design-matrix-io-read-h5)=
### `read_h5`

```python
read_h5(file_name: str | Path) -> tuple[pl.DataFrame, dict]
```

Read a DesignMatrix HDF5 file written by `write_h5`.

Handles both on-disk layouts: the current one (frame as Arrow IPC bytes)
and the pre-reader one written by nltools <= 0.6.0 (a plain float matrix
in ``data`` beside an ``S``-typed ``columns`` dataset). Legacy files may
also carry pre-`.nl_` generated column names (``poly_0``, ``0_poly_0``,
``cosine_1``); those are translated into the reserved namespace at load
time so downstream recognition stays keyed on the prefix alone.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>[str](#str) \| [Path](#pathlib.Path)</code> | Path to the HDF5 file. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[DataFrame](#polars.DataFrame), [dict](#dict)]</code> | `(frame, metadata)`, where metadata holds     ``sampling_freq``, ``convolved``, ``confounds``, ``multi``, and     ``n_rows`` — absent keys meaning the file didn't record them.

(data-design-matrix-io-separator-for-path)=
### `separator_for_path`

```python
separator_for_path(path: str | Path) -> str
```

Return the delimiter a text DesignMatrix file uses, from its extension.

The single source of truth for both `write` and `load_from_file`, so a
file nltools writes is always a file nltools can read back. ``.csv`` means
comma; every other extension means tab, matching the BIDS convention for
``.tsv`` and keeping the historical default for ``.txt`` and friends.

(data-design-matrix-io-to-numpy)=
### `to_numpy`

```python
to_numpy(dm: DesignMatrix) -> np.ndarray
```

Convert a DesignMatrix to a NumPy array.

Returns data columns as 2D numpy array (rows x columns).
Column order is preserved from DataFrame.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | 2D array with shape (n_samples, n_columns)

**Examples:**

```pycon
>>> dm = DesignMatrix({"a": [1, 2, 3], "b": [4, 5, 6]}, sampling_freq=1)
>>> arr = to_numpy(dm)
>>> arr.shape
(3, 2)
```

(data-design-matrix-io-to-pandas)=
### `to_pandas`

```python
to_pandas(dm: DesignMatrix)
```

Convert DesignMatrix to pandas DataFrame.

Uses dict-based conversion to avoid pyarrow dependency. This is slightly
slower (~10-20%) than pyarrow-based conversion but removes the dependency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#pandas.DataFrame)</code> | Pandas DataFrame with same data and column names.

**Examples:**

```pycon
>>> dm = DesignMatrix(np.random.randn(100, 3))
>>> pd_df = to_pandas(dm)
>>> type(pd_df)
<class 'pandas.core.frame.DataFrame'>
```

(data-design-matrix-io-write)=
### `write`

```python
write(dm: DesignMatrix, file_name: str, sep: str | None = None) -> None
```

Write DesignMatrix to file.

Supports TSV, CSV, and HDF5 formats. The format is automatically
determined by file extension.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`file_name` | <code>[str](#str)</code> | Output file path. Use .tsv, .csv, or .h5/.hdf5 extension. | *required*
`sep` | <code>[str](#str) \| None</code> | Column separator for text files. Defaults to the delimiter the extension implies (comma for ``.csv``, tab otherwise), so the file reads back correctly; pass a value to override. Ignored for HDF5. | <code>None</code>

**Examples:**

```pycon
>>> dm = DesignMatrix(np.random.randn(100, 3), sampling_freq=1)
>>> write(dm, "design_matrix.tsv")  # tab separated (BIDS compatible)
>>> write(dm, "design_matrix.csv")  # comma separated
>>> write(dm, "design_matrix.h5")  # HDF5, metadata preserved
```

<details class="note" open markdown="1">
<summary>Note</summary>

TSV format is recommended for BIDS compatibility. Text formats carry
the data only — HDF5 additionally preserves ``sampling_freq``,
``.convolved``, ``.confounds``, ``.multi``, and the row count of a
column-less matrix, so ``DesignMatrix(path)`` restores the object.

</details>

(data-design-matrix-io-write-h5)=
### `write_h5`

```python
write_h5(dm: DesignMatrix, file_name: str) -> None
```

Write DesignMatrix to HDF5 file with metadata.

The frame is stored as Arrow IPC bytes (via the shared
`nltools.io.h5` helpers) so every dtype round-trips exactly — an integer
spike indicator comes back an integer rather than being floated by a
detour through a homogeneous numpy array.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`file_name` | <code>[str](#str)</code> | Output HDF5 file path. | *required*
