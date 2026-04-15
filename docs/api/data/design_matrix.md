## `DesignMatrix`

```python
DesignMatrix(data: Union[pl.DataFrame, pd.DataFrame, np.ndarray, dict, None] = None, *, sampling_freq: Optional[float] = None, columns: Optional[List[str]] = None, convolved: Optional[List[str]] = None, polys: Optional[List[str]] = None)
```

Polars-based design matrix for experimental designs in neuroimaging.

Wraps a Polars DataFrame with neuroimaging-specific metadata and methods.
Uses composition pattern (not subclassing) for clean metadata preservation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>DataFrame, ndarray, dict, or None</code> | Input data. Accepts: - Polars DataFrame (zero-copy) - pandas DataFrame (converted to Polars) - numpy ndarray - dict (keys=columns, values=data) - None (empty initialization) | <code>None</code>
`sampling_freq` | <code>[float](#float)</code> | Sampling frequency in Hz (1/TR for fMRI data) | <code>None</code>
`columns` | <code>list of str</code> | Column names (used with ndarray input) | <code>None</code>
`convolved` | <code>list of str</code> | Names of convolved columns (tracked internally) | <code>None</code>
`polys` | <code>list of str</code> | Names of polynomial columns (tracked internally) | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`sampling_freq`](#sampling_freq) | <code>[float](#float) or None</code> | Sampling frequency in Hz
[`convolved`](#convolved) | <code>list of str</code> | Columns that have been convolved
[`polys`](#polys) | <code>list of str</code> | Polynomial/nuisance columns (intercept, trends, DCT bases)
[`multi`](#multi) | <code>[bool](#bool)</code> | True if created from multi-run concatenation

**Examples:**

```pycon
>>> # Create from numpy array
>>> dm = DesignMatrix(np.zeros((100, 2)), sampling_freq=0.5, columns=['a', 'b'])
```

```pycon
>>> # Add columns
>>> dm['stim'] = [0, 1, 1, 0] * 25
```

```pycon
>>> # Convolve with HRF
>>> dm_conv = dm.convolve()
```

```pycon
>>> # Add polynomial drift terms
>>> dm_conv = dm_conv.add_poly(order=2)
```

```pycon
>>> # Multi-run concatenation (auto-separates polynomials)
>>> dm_run1 = DesignMatrix(...).add_poly(0)
>>> dm_run2 = DesignMatrix(...).add_poly(0)
>>> dm_multi = dm_run1.append(dm_run2, axis=0)  # Creates 0_poly_0, 1_poly_0
```

**Methods:**

Name | Description
---- | -----------
[`add_dct_basis`](#add_dct_basis) | Add discrete cosine transform basis functions (high-pass filter).
[`add_poly`](#add_poly) | Add Legendre polynomial drift terms.
[`append`](#append) | Concatenate design matrices.
[`clean`](#clean) | Remove highly correlated columns.
[`convolve`](#convolve) | Convolve columns with HRF or custom kernel.
[`copy`](#copy) | Create a deep copy of the DesignMatrix.
[`details`](#details) | Return human-readable metadata summary.
[`downsample`](#downsample) | Reduce temporal resolution to target frequency using Polars-native operations.
[`drop`](#drop) | Drop specified columns.
[`fillna`](#fillna) | Fill NaN/null values with specified value.
[`plot`](#plot) | Visualize design matrix as heatmap (SPM-style).
[`replace_data`](#replace_data) | Replace data columns while preserving polynomials and metadata.
[`standardize`](#standardize) | Standardize columns using the specified method.
[`sum`](#sum) | Compute sum along axis.
[`to_numpy`](#to_numpy) | Convert DesignMatrix to numpy array.
[`to_pandas`](#to_pandas) | Convert DesignMatrix to pandas DataFrame.
[`upsample`](#upsample) | Increase temporal resolution to target frequency.
[`vif`](#vif) | Compute variance inflation factor for each column.
[`write`](#write) | Write DesignMatrix to file.
[`zscore`](#zscore) | Z-score standardize columns (mean=0, std=1).

### Methods

#### `add_dct_basis`

```python
add_dct_basis(duration: float = 180, drop: int = 0) -> 'DesignMatrix'
```

Add discrete cosine transform basis functions (high-pass filter).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`duration` | <code>[float](#float)</code> | Filter duration in seconds. Default: 180. | <code>180</code>
`drop` | <code>[int](#int)</code> | Number of low-frequency bases to drop. Default: 0. | <code>0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>'DesignMatrix'</code> | New DesignMatrix with DCT basis columns appended.

#### `add_poly`

```python
add_poly(order: int = 0, include_lower: bool = True) -> 'DesignMatrix'
```

Add Legendre polynomial drift terms.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`order` | <code>[int](#int)</code> | Polynomial order (0=intercept, 1=linear, 2=quadratic, ...). Default: 0. | <code>0</code>
`include_lower` | <code>[bool](#bool)</code> | If True, include all orders from 0 to order. Default: True. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>'DesignMatrix'</code> | New DesignMatrix with polynomial columns appended.

#### `append`

```python
append(dm: Union['DesignMatrix', List['DesignMatrix']], axis: int = 0, keep_separate: bool = True, unique_cols: Optional[List[str]] = None, fill_na: Union[int, float] = 0, verbose: bool = False) -> 'DesignMatrix'
```

Concatenate design matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>DesignMatrix or list of DesignMatrix</code> | Design matrix/matrices to append. | *required*
`axis` | <code>[int](#int)</code> | 0 for row-wise (vertical), 1 for column-wise (horizontal). Default: 0. | <code>0</code>
`keep_separate` | <code>[bool](#bool)</code> | Whether to separate polynomial columns across runs (only applies when axis=0). Default: True. | <code>True</code>
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | <code>None</code>
`fill_na` | <code>[int](#int) or [float](#float)</code> | Value to fill NaN values during vertical concatenation. Default: 0. | <code>0</code>
`verbose` | <code>[bool](#bool)</code> | Print messages about polynomial separation. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>'DesignMatrix'</code> | Concatenated design matrix.

#### `clean`

```python
clean(fill_na: Union[int, float, None] = 0, exclude_polys: bool = False, thresh: float = 0.95, verbose: bool = True) -> 'DesignMatrix'
```

Remove highly correlated columns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fill_na` | <code>int, float, or None</code> | Fill NaN values before checking correlations (default 0) | <code>0</code>
`exclude_polys` | <code>[bool](#bool)</code> | Skip polynomial columns from correlation check | <code>False</code>
`thresh` | <code>[float](#float)</code> | Correlation threshold (drop if abs(r) >= thresh, default 0.95) | <code>0.95</code>
`verbose` | <code>[bool](#bool)</code> | Print dropped column names | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>'DesignMatrix'</code> | Cleaned matrix with highly correlated columns removed

#### `convolve`

```python
convolve(conv_func: Union[str, np.ndarray] = 'hrf', columns: Optional[List[str]] = None) -> 'DesignMatrix'
```

Convolve columns with HRF or custom kernel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`conv_func` | <code>[str](#str) or [ndarray](#ndarray)</code> | 'hrf' for canonical Glover HRF, or custom kernel(s). Can be 1D array (single kernel) or 2D (samples x kernels) | <code>'hrf'</code>
`columns` | <code>list of str</code> | Columns to convolve (default: all non-polynomial columns) | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>'DesignMatrix'</code> | New DesignMatrix with convolved columns

#### `copy`

```python
copy() -> 'DesignMatrix'
```

Create a deep copy of the DesignMatrix.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>'DesignMatrix'</code> | Copy of the current DesignMatrix

#### `details`

```python
details() -> str
```

Return human-readable metadata summary.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` | <code>[str](#str)</code> | Formatted string showing sampling_freq, shape, convolved columns, and polynomial columns

#### `downsample`

```python
downsample(target: float, method: str = 'mean', **kwargs: str) -> 'DesignMatrix'
```

Reduce temporal resolution to target frequency using Polars-native operations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` | <code>[float](#float)</code> | Target sampling frequency in Hz (must be < current sampling_freq) | *required*
`method` | <code>[str](#str)</code> | Aggregation method - 'mean' or 'median' (default: 'mean') | <code>'mean'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>'DesignMatrix'</code> | Downsampled DesignMatrix with updated sampling_freq

#### `drop`

```python
drop(columns: List[str]) -> 'DesignMatrix'
```

Drop specified columns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`columns` | <code>list of str</code> | Column names to remove. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>'DesignMatrix'</code> | New DesignMatrix without the specified columns.

#### `fillna`

```python
fillna(value: Union[int, float]) -> 'DesignMatrix'
```

Fill NaN/null values with specified value.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`value` | <code>[int](#int) or [float](#float)</code> | Value to replace NaN/null entries with. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>'DesignMatrix'</code> | New DesignMatrix with NaN/null values replaced.

#### `plot`

```python
plot(figsize: tuple = (8, 6), **kwargs: tuple)
```

Visualize design matrix as heatmap (SPM-style).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`figsize` | <code>tuple, default=(8, 6)</code> | Figure size (width, height) in inches | <code>(8, 6)</code>
`**kwargs` |  | Additional keyword arguments passed to seaborn.heatmap() | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.axes.Axes: The axes object containing the heatmap

#### `replace_data`

```python
replace_data(data: np.ndarray, column_names: Optional[List[str]] = None) -> 'DesignMatrix'
```

Replace data columns while preserving polynomials and metadata.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#ndarray)</code> | New data array (must match number of rows in current DesignMatrix) | *required*
`column_names` | <code>list of str</code> | Names for new data columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>'DesignMatrix'</code> | New DesignMatrix with replaced data columns, preserved polynomials

#### `standardize`

```python
standardize(method: str = 'zscore', columns: Optional[List[str]] = None) -> 'DesignMatrix'
```

Standardize columns using the specified method.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Standardization method ('zscore' or 'center'). Default: 'zscore'. | <code>'zscore'</code>
`columns` | <code>[Optional](#typing.Optional)[[List](#typing.List)[[str](#str)]]</code> | Columns to standardize. If None, standardize all non-polynomial columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>'DesignMatrix'</code> | New DesignMatrix with standardized columns.

#### `sum`

```python
sum(axis: int = 0) -> pl.Series
```

Compute sum along axis.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>int, default=0</code> | 0: sum down columns, 1: sum across rows. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>[Series](#polars.Series)</code> | pl.Series: Sums along specified axis.

#### `to_numpy`

```python
to_numpy() -> np.ndarray
```

Convert DesignMatrix to numpy array.

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | np.ndarray: 2D array with shape (n_samples, n_columns)

#### `to_pandas`

```python
to_pandas() -> pd.DataFrame
```

Convert DesignMatrix to pandas DataFrame.

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#pandas.DataFrame)</code> | pd.DataFrame: Pandas DataFrame with same data and column names.

#### `upsample`

```python
upsample(target: float, method: str = 'linear', **kwargs: str) -> 'DesignMatrix'
```

Increase temporal resolution to target frequency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` | <code>[float](#float)</code> | Target sampling frequency in Hz (must be > current sampling_freq) | *required*
`method` | <code>[str](#str)</code> | Interpolation method - 'linear' or 'nearest' (default: 'linear') | <code>'linear'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>'DesignMatrix'</code> | Upsampled DesignMatrix with updated sampling_freq

#### `vif`

```python
vif(exclude_polys: bool = True) -> np.ndarray | None
```

Compute variance inflation factor for each column.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`exclude_polys` | <code>[bool](#bool)</code> | Skip polynomial columns. Default: True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray) \| None</code> | np.ndarray: VIF values for each included column. Returns None if the correlation matrix is singular.

#### `write`

```python
write(file_name: str, sep: str = '\t') -> None
```

Write DesignMatrix to file.

Supports TSV (default), CSV, and HDF5 formats. Format is
auto-detected from file extension.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>[str](#str)</code> | Output file path. Use .tsv, .csv, or .h5/.hdf5 extension. | *required*
`sep` | <code>[str](#str)</code> | Column separator for text files (default: tab). | <code>'\t'</code>

#### `zscore`

```python
zscore(columns: Optional[List[str]] = None) -> 'DesignMatrix'
```

Z-score standardize columns (mean=0, std=1).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`columns` | <code>list of str</code> | Columns to standardize. If None, standardize all non-polynomial columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>'DesignMatrix'</code> | New DesignMatrix with standardized columns

