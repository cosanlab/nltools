## `io`

DesignMatrix I/O and visualization functions.

Standalone functions extracted from DesignMatrix methods.
Each takes a DesignMatrix instance (`dm`) as its first argument.

**Methods:**

Name | Description
---- | -----------
[`events_to_dm`](#events_to_dm) | Convert a BIDS events table to boxcar regressors aligned to TRs.
[`load_from_file`](#load_from_file) | Read a TSV/CSV into the frame a DesignMatrix wraps.
[`plot_designmatrix`](#plot_designmatrix) | Visualize design matrix as heatmap (SPM-style).
[`to_numpy`](#to_numpy) | Convert DesignMatrix to numpy array.
[`to_pandas`](#to_pandas) | Convert DesignMatrix to pandas DataFrame.
[`write`](#write) | Write DesignMatrix to file.
[`write_h5`](#write_h5) | Write DesignMatrix to HDF5 file with metadata.



### Classes

### Methods

#### `events_to_dm`

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
<code>[DataFrame](#polars.DataFrame)</code> | pl.DataFrame with one column per unique `trial_type`, values in
<code>[DataFrame](#polars.DataFrame)</code> | {0, modulation} indicating where each condition is active.

#### `load_from_file`

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
<code>[DataFrame](#polars.DataFrame)</code> | Tuple of (data frame, is_events) — `is_events` signals to the
<code>[bool](#bool)</code> | caller that the columns are experimental regressors rather than
<code>[tuple](#tuple)[[DataFrame](#polars.DataFrame), [bool](#bool)]</code> | nuisance.

#### `plot_designmatrix`

```python
plot_designmatrix(dm: DesignMatrix, figsize: tuple = (8, 6), *, rescale: bool = True, **kwargs: bool)
```

Visualize design matrix as heatmap (SPM-style).

Creates a heatmap visualization of the design matrix columns.
Uses seaborn + matplotlib under the hood.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`figsize` | <code>[tuple](#tuple)</code> | Figure size (width, height) in inches. Default: (8, 6). | <code>(8, 6)</code>
`rescale` | <code>[bool](#bool)</code> | If True, rescale each column by its L2 norm so columns with different native magnitudes are visually comparable (matches SPM/nilearn convention). Default: True. | <code>True</code>
`**kwargs` |  | Additional keyword arguments passed to seaborn.heatmap(). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure: The figure containing the heatmap.

**Examples:**

```pycon
>>> dm = DesignMatrix(np.random.randn(100, 3), columns=['a', 'b', 'c'])
>>> plot_designmatrix(dm)
```

#### `to_numpy`

```python
to_numpy(dm: DesignMatrix) -> np.ndarray
```

Convert DesignMatrix to numpy array.

Returns data columns as 2D numpy array (rows x columns).
Column order is preserved from DataFrame.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | np.ndarray: 2D array with shape (n_samples, n_columns)

**Examples:**

```pycon
>>> dm = DesignMatrix({"a": [1, 2, 3], "b": [4, 5, 6]}, sampling_freq=1)
>>> arr = to_numpy(dm)
>>> arr.shape
(3, 2)
```

#### `to_pandas`

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
 | pd.DataFrame: Pandas DataFrame with same data and column names.

**Examples:**

```pycon
>>> dm = DesignMatrix(np.random.randn(100, 3))
>>> pd_df = to_pandas(dm)
>>> type(pd_df)
<class 'pandas.core.frame.DataFrame'>
```

#### `write`

```python
write(dm: DesignMatrix, file_name: str, sep: str = '\t') -> None
```

Write DesignMatrix to file.

Supports TSV (default), CSV, and HDF5 formats. The format is
automatically determined by file extension.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`file_name` | <code>[str](#str)</code> | Output file path. Use .tsv, .csv, or .h5/.hdf5 extension. | *required*
`sep` | <code>[str](#str)</code> | Column separator for text files (default: tab for TSV).  Ignored for HDF5 files. | <code>'\t'</code>

**Returns:**

Type | Description
---- | -----------
<code>None</code> | None

**Examples:**

```pycon
>>> dm = DesignMatrix(np.random.randn(100, 3), sampling_freq=1)
>>> write(dm, "design_matrix.tsv")  # TSV format (BIDS compatible)
>>> write(dm, "design_matrix.csv", sep=",")  # CSV format
>>> write(dm, "design_matrix.h5")  # HDF5 format
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

TSV format is recommended for BIDS compatibility.
HDF5 format preserves metadata (sampling_freq, convolved, confounds).

</details>

#### `write_h5`

```python
write_h5(dm: DesignMatrix, file_name: str) -> None
```

Write DesignMatrix to HDF5 file with metadata.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`file_name` | <code>[str](#str)</code> | Output HDF5 file path. | *required*

**Returns:**

Type | Description
---- | -----------
<code>None</code> | None

