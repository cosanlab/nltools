## `nltools.data.designmatrix`

DesignMatrix - Polars-based design matrix for neuroimaging analysis

Efficient design matrix implementation using Polars for fast DataFrame operations.
Provides HRF convolution, resampling, polynomial regressors, and diagnostic tools.

Uses composition pattern (wrapping pl.DataFrame) for clean metadata preservation.

**Modules:**

Name | Description
---- | -----------
[`append`](#nltools.data.designmatrix.append) | Standalone functions for DesignMatrix concatenation operations.
[`diagnostics`](#nltools.data.designmatrix.diagnostics) | Diagnostic and utility functions for DesignMatrix.
[`io`](#nltools.data.designmatrix.io) | DesignMatrix I/O and visualization functions.
[`regressors`](#nltools.data.designmatrix.regressors) | Standalone regressor functions for DesignMatrix.
[`transforms`](#nltools.data.designmatrix.transforms) | Standalone transform functions for DesignMatrix.
[`utils`](#nltools.data.designmatrix.utils) | Shared helpers for DesignMatrix submodules.

**Classes:**

Name | Description
---- | -----------
[`DesignMatrix`](#nltools.data.designmatrix.DesignMatrix) | Polars-based design matrix for experimental designs in neuroimaging.



### Classes#### `nltools.data.designmatrix.DesignMatrix`

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
[`sampling_freq`](#nltools.data.designmatrix.DesignMatrix.sampling_freq) | <code>[float](#float) or None</code> | Sampling frequency in Hz
[`convolved`](#nltools.data.designmatrix.DesignMatrix.convolved) | <code>list of str</code> | Columns that have been convolved
[`polys`](#nltools.data.designmatrix.DesignMatrix.polys) | <code>list of str</code> | Polynomial/nuisance columns (intercept, trends, DCT bases)
[`multi`](#nltools.data.designmatrix.DesignMatrix.multi) | <code>[bool](#bool)</code> | True if created from multi-run concatenation

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

**Functions:**

Name | Description
---- | -----------
[`add_dct_basis`](#nltools.data.designmatrix.DesignMatrix.add_dct_basis) | Add discrete cosine transform basis functions (high-pass filter).
[`add_poly`](#nltools.data.designmatrix.DesignMatrix.add_poly) | Add Legendre polynomial drift terms.
[`append`](#nltools.data.designmatrix.DesignMatrix.append) | Concatenate design matrices.
[`clean`](#nltools.data.designmatrix.DesignMatrix.clean) | Remove highly correlated columns.
[`convolve`](#nltools.data.designmatrix.DesignMatrix.convolve) | Convolve columns with HRF or custom kernel.
[`copy`](#nltools.data.designmatrix.DesignMatrix.copy) | Create a deep copy of the DesignMatrix.
[`details`](#nltools.data.designmatrix.DesignMatrix.details) | Return human-readable metadata summary.
[`downsample`](#nltools.data.designmatrix.DesignMatrix.downsample) | Reduce temporal resolution to target frequency using Polars-native operations.
[`drop`](#nltools.data.designmatrix.DesignMatrix.drop) | Drop specified columns.
[`fillna`](#nltools.data.designmatrix.DesignMatrix.fillna) | Fill NaN/null values with specified value.
[`heatmap`](#nltools.data.designmatrix.DesignMatrix.heatmap) | Visualize design matrix as heatmap (SPM-style).
[`replace_data`](#nltools.data.designmatrix.DesignMatrix.replace_data) | Replace data columns while preserving polynomials and metadata.
[`reset_index`](#nltools.data.designmatrix.DesignMatrix.reset_index) | Reset index (pandas compatibility method).
[`standardize`](#nltools.data.designmatrix.DesignMatrix.standardize) | Standardize columns using the specified method.
[`sum`](#nltools.data.designmatrix.DesignMatrix.sum) | Compute sum along axis.
[`to_numpy`](#nltools.data.designmatrix.DesignMatrix.to_numpy) | Convert DesignMatrix to numpy array.
[`to_pandas`](#nltools.data.designmatrix.DesignMatrix.to_pandas) | Convert DesignMatrix to pandas DataFrame.
[`upsample`](#nltools.data.designmatrix.DesignMatrix.upsample) | Increase temporal resolution to target frequency.
[`vif`](#nltools.data.designmatrix.DesignMatrix.vif) | Compute variance inflation factor for each column.
[`write`](#nltools.data.designmatrix.DesignMatrix.write) | Write DesignMatrix to file.
[`zscore`](#nltools.data.designmatrix.DesignMatrix.zscore) | Z-score standardize columns (mean=0, std=1).



##### Attributes###### `nltools.data.designmatrix.DesignMatrix.columns`

```python
columns: List[str]
```

Column names of the design matrix as a list of strings.

###### `nltools.data.designmatrix.DesignMatrix.convolved`

```python
convolved = convolved if convolved is not None else []
```

###### `nltools.data.designmatrix.DesignMatrix.is_empty`

```python
is_empty: bool
```

Check if DesignMatrix has no data.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`bool` | <code>[bool](#bool)</code> | True if the design matrix is empty, False otherwise.

###### `nltools.data.designmatrix.DesignMatrix.multi`

```python
multi = False
```

###### `nltools.data.designmatrix.DesignMatrix.polys`

```python
polys = polys if polys is not None else []
```

###### `nltools.data.designmatrix.DesignMatrix.sampling_freq`

```python
sampling_freq = sampling_freq
```

###### `nltools.data.designmatrix.DesignMatrix.shape`

```python
shape: tuple
```

Return (n_rows, n_cols) tuple.



##### Functions###### `nltools.data.designmatrix.DesignMatrix.add_dct_basis`

```python
add_dct_basis(duration: float = 180, drop: int = 0) -> DesignMatrix
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
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with DCT basis columns appended.

###### `nltools.data.designmatrix.DesignMatrix.add_poly`

```python
add_poly(order: int = 0, include_lower: bool = True) -> DesignMatrix
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
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with polynomial columns appended.

###### `nltools.data.designmatrix.DesignMatrix.append`

```python
append(dm: Union[DesignMatrix, List[DesignMatrix]], axis: int = 0, keep_separate: bool = True, unique_cols: Optional[List[str]] = None, fill_na: Union[int, float] = 0, verbose: bool = False) -> DesignMatrix
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
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Concatenated design matrix.

###### `nltools.data.designmatrix.DesignMatrix.clean`

```python
clean(fill_na: Union[int, float, None] = 0, exclude_polys: bool = False, thresh: float = 0.95, verbose: bool = True) -> DesignMatrix
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
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Cleaned matrix with highly correlated columns removed

###### `nltools.data.designmatrix.DesignMatrix.convolve`

```python
convolve(conv_func: Union[str, np.ndarray] = 'hrf', columns: Optional[List[str]] = None) -> DesignMatrix
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
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with convolved columns

###### `nltools.data.designmatrix.DesignMatrix.copy`

```python
copy() -> DesignMatrix
```

Create a deep copy of the DesignMatrix.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Copy of the current DesignMatrix

###### `nltools.data.designmatrix.DesignMatrix.details`

```python
details() -> str
```

Return human-readable metadata summary.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` | <code>[str](#str)</code> | Formatted string showing sampling_freq, shape, convolved columns, and polynomial columns

###### `nltools.data.designmatrix.DesignMatrix.downsample`

```python
downsample(target: float, method: str = 'mean', **kwargs: str) -> DesignMatrix
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
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Downsampled DesignMatrix with updated sampling_freq

###### `nltools.data.designmatrix.DesignMatrix.drop`

```python
drop(columns: List[str]) -> DesignMatrix
```

Drop specified columns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`columns` | <code>list of str</code> | Column names to remove. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix without the specified columns.

###### `nltools.data.designmatrix.DesignMatrix.fillna`

```python
fillna(value: Union[int, float]) -> DesignMatrix
```

Fill NaN/null values with specified value.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`value` | <code>[int](#int) or [float](#float)</code> | Value to replace NaN/null entries with. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with NaN/null values replaced.

###### `nltools.data.designmatrix.DesignMatrix.heatmap`

```python
heatmap(figsize: tuple = (8, 6), **kwargs: tuple)
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

###### `nltools.data.designmatrix.DesignMatrix.replace_data`

```python
replace_data(data: np.ndarray, column_names: Optional[List[str]] = None) -> DesignMatrix
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
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with replaced data columns, preserved polynomials

###### `nltools.data.designmatrix.DesignMatrix.reset_index`

```python
reset_index(drop: bool = True) -> DesignMatrix
```

Reset index (pandas compatibility method).

Polars DataFrames don't have row indexes, so this is a no-op.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`drop` | <code>bool, default=True</code> | Ignored. Kept for API compatibility. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Returns self unchanged

###### `nltools.data.designmatrix.DesignMatrix.standardize`

```python
standardize(method: str = 'zscore', columns: Optional[List[str]] = None) -> DesignMatrix
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
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with standardized columns.

###### `nltools.data.designmatrix.DesignMatrix.sum`

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

###### `nltools.data.designmatrix.DesignMatrix.to_numpy`

```python
to_numpy() -> np.ndarray
```

Convert DesignMatrix to numpy array.

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | np.ndarray: 2D array with shape (n_samples, n_columns)

###### `nltools.data.designmatrix.DesignMatrix.to_pandas`

```python
to_pandas() -> pd.DataFrame
```

Convert DesignMatrix to pandas DataFrame.

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#pandas.DataFrame)</code> | pd.DataFrame: Pandas DataFrame with same data and column names.

###### `nltools.data.designmatrix.DesignMatrix.upsample`

```python
upsample(target: float, method: str = 'linear', **kwargs: str) -> DesignMatrix
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
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Upsampled DesignMatrix with updated sampling_freq

###### `nltools.data.designmatrix.DesignMatrix.vif`

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

###### `nltools.data.designmatrix.DesignMatrix.write`

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

###### `nltools.data.designmatrix.DesignMatrix.zscore`

```python
zscore(columns: Optional[List[str]] = None) -> DesignMatrix
```

Z-score standardize columns (mean=0, std=1).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`columns` | <code>list of str</code> | Columns to standardize. If None, standardize all non-polynomial columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with standardized columns



### Functions

### Modules#### `nltools.data.designmatrix.append`

Standalone functions for DesignMatrix concatenation operations.

These functions implement the append/concatenation logic extracted from
DesignMatrix methods, following the "functional core" pattern.

**Functions:**

Name | Description
---- | -----------
[`append`](#nltools.data.designmatrix.append.append) | Concatenate design matrices.
[`append_horizontal`](#nltools.data.designmatrix.append.append_horizontal) | Horizontal concatenation (axis=1) - add columns from other matrices.
[`append_vertical`](#nltools.data.designmatrix.append.append_vertical) | Vertical concatenation (axis=0) - stack rows, with optional polynomial separation.
[`append_vertical_with_separation`](#nltools.data.designmatrix.append.append_vertical_with_separation) | Vertical concatenation with automatic polynomial separation.
[`get_starting_run_idx`](#nltools.data.designmatrix.append.get_starting_run_idx) | Determine next run index for multi-run appending.
[`identify_columns_to_separate`](#nltools.data.designmatrix.append.identify_columns_to_separate) | Identify which columns need run-specific separation.
[`match_column_pattern`](#nltools.data.designmatrix.append.match_column_pattern) | Match columns against pattern with wildcard support.



##### Classes

##### Functions###### `nltools.data.designmatrix.append.append`

```python
append(dm: DesignMatrix, other: Union[DesignMatrix, List[DesignMatrix]], axis: int = 0, keep_separate: bool = True, unique_cols: Optional[List[str]] = None, fill_na: Union[int, float] = 0, verbose: bool = False) -> DesignMatrix
```

Concatenate design matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | The base design matrix. | *required*
`other` | <code>DesignMatrix or list of DesignMatrix</code> | Design matrix/matrices to append. | *required*
`axis` | <code>[int](#int)</code> | 0 for row-wise (vertical), 1 for column-wise (horizontal). | <code>0</code>
`keep_separate` | <code>[bool](#bool)</code> | Whether to separate polynomial columns across runs (only axis=0). | <code>True</code>
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | <code>None</code>
`fill_na` | <code>[int](#int) or [float](#float)</code> | Value to fill NaN values during vertical concatenation. Default: 0. | <code>0</code>
`verbose` | <code>[bool](#bool)</code> | Print messages about polynomial separation. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Concatenated design matrix.

###### `nltools.data.designmatrix.append.append_horizontal`

```python
append_horizontal(dm: DesignMatrix, to_append: List[DesignMatrix], fill_na: Union[int, float]) -> DesignMatrix
```

Horizontal concatenation (axis=1) - add columns from other matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Base DesignMatrix instance. | *required*
`to_append` | <code>list of DesignMatrix</code> | Matrices whose columns to add. | *required*
`fill_na` | <code>[int](#int) or [float](#float)</code> | Value to fill NaN/null entries with. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with columns from all matrices.

###### `nltools.data.designmatrix.append.append_vertical`

```python
append_vertical(dm: DesignMatrix, to_append: List[DesignMatrix], keep_separate: bool, unique_cols: Optional[List[str]], fill_na: Union[int, float], verbose: bool) -> DesignMatrix
```

Vertical concatenation (axis=0) - stack rows, with optional polynomial separation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Base DesignMatrix instance. | *required*
`to_append` | <code>list of DesignMatrix</code> | Matrices to stack below dm. | *required*
`keep_separate` | <code>[bool](#bool)</code> | Whether to separate polynomial columns across runs. | *required*
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | *required*
`fill_na` | <code>[int](#int) or [float](#float)</code> | Value to fill NaN/null entries with. | *required*
`verbose` | <code>[bool](#bool)</code> | Print messages about polynomial separation. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with rows from all matrices.

###### `nltools.data.designmatrix.append.append_vertical_with_separation`

```python
append_vertical_with_separation(dm: DesignMatrix, to_append: List[DesignMatrix], unique_cols: Optional[List[str]], fill_na: Union[int, float], verbose: bool) -> DesignMatrix
```

Vertical concatenation with automatic polynomial separation.

Creates run-specific columns (e.g., 0_poly_0, 1_poly_0) that are
active only in their respective runs (sparse representation).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Base DesignMatrix instance. | *required*
`to_append` | <code>list of DesignMatrix</code> | Matrices to stack below dm. | *required*
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | *required*
`fill_na` | <code>[int](#int) or [float](#float)</code> | Value to fill NaN/null entries with. | *required*
`verbose` | <code>[bool](#bool)</code> | Print messages about polynomial separation. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Concatenated DesignMatrix with run-separated polynomial columns and multi=True.

###### `nltools.data.designmatrix.append.get_starting_run_idx`

```python
get_starting_run_idx(dm: DesignMatrix) -> int
```

Determine next run index for multi-run appending.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance to inspect. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`int` | <code>[int](#int)</code> | Next run index (0 if not multi-run, max_existing_idx + 1 otherwise).

###### `nltools.data.designmatrix.append.identify_columns_to_separate`

```python
identify_columns_to_separate(dm: DesignMatrix, all_dms: List[DesignMatrix], unique_cols: Optional[List[str]]) -> set
```

Identify which columns need run-specific separation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | The base design matrix (used for context only). | *required*
`all_dms` | <code>list of DesignMatrix</code> | All matrices being concatenated. | *required*
`unique_cols` | <code>list of str</code> | User-specified columns to separate (supports wildcards). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`set` | <code>[set](#set)</code> | Column names that should be separated with run prefixes.

###### `nltools.data.designmatrix.append.match_column_pattern`

```python
match_column_pattern(columns: List[str], pattern: str) -> List[str]
```

Match columns against pattern with wildcard support.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`columns` | <code>list of str</code> | Column names to search. | *required*
`pattern` | <code>[str](#str)</code> | Pattern to match (supports '*' as wildcard). - 'motion*' matches motion_x, motion_y - '*_motion' matches x_motion, y_motion - 'exact' matches only 'exact' | *required*

**Returns:**

Type | Description
---- | -----------
<code>[List](#typing.List)[[str](#str)]</code> | list of str: Column names matching the pattern.

#### `nltools.data.designmatrix.diagnostics`

Diagnostic and utility functions for DesignMatrix.

**Functions:**

Name | Description
---- | -----------
[`clean`](#nltools.data.designmatrix.diagnostics.clean) | Remove highly correlated columns.
[`details`](#nltools.data.designmatrix.diagnostics.details) | Return human-readable metadata summary.
[`vif`](#nltools.data.designmatrix.diagnostics.vif) | Compute variance inflation factor for each column.



##### Classes

##### Functions###### `nltools.data.designmatrix.diagnostics.clean`

```python
clean(dm: DesignMatrix, fill_na: Union[int, float, None] = 0, exclude_polys: bool = False, thresh: float = 0.95, verbose: bool = True) -> DesignMatrix
```

Remove highly correlated columns.

Removes columns with correlation >= threshold. Keeps first instance
of correlated pair, drops duplicates.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`fill_na` | <code>int, float, or None</code> | Fill NaN values before checking correlations. Default: 0. | <code>0</code>
`exclude_polys` | <code>[bool](#bool)</code> | Skip polynomial columns from correlation check. Default: False. | <code>False</code>
`thresh` | <code>[float](#float)</code> | Correlation threshold (drop if abs(r) >= thresh). Default: 0.95. | <code>0.95</code>
`verbose` | <code>[bool](#bool)</code> | Print dropped column names. Default: True. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Cleaned matrix with highly correlated columns removed

###### `nltools.data.designmatrix.diagnostics.details`

```python
details(dm: DesignMatrix) -> str
```

Return human-readable metadata summary.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` | <code>[str](#str)</code> | Formatted string showing sampling_freq, shape, convolved columns, and polynomial columns.

###### `nltools.data.designmatrix.diagnostics.vif`

```python
vif(dm: DesignMatrix, exclude_polys: bool = True) -> np.ndarray | None
```

Compute variance inflation factor for each column.

Uses diagonal elements of inverted correlation matrix
(same method as Matlab and R).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`exclude_polys` | <code>[bool](#bool)</code> | Skip polynomial columns. Default: True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray) \| None</code> | np.ndarray: VIF values for each included column. Returns None if the correlation matrix is singular (perfect collinearity detected).

#### `nltools.data.designmatrix.io`

DesignMatrix I/O and visualization functions.

Standalone functions extracted from DesignMatrix methods.
Each takes a DesignMatrix instance (`dm`) as its first argument.

**Functions:**

Name | Description
---- | -----------
[`heatmap`](#nltools.data.designmatrix.io.heatmap) | Visualize design matrix as heatmap (SPM-style).
[`to_numpy`](#nltools.data.designmatrix.io.to_numpy) | Convert DesignMatrix to numpy array.
[`to_pandas`](#nltools.data.designmatrix.io.to_pandas) | Convert DesignMatrix to pandas DataFrame.
[`write`](#nltools.data.designmatrix.io.write) | Write DesignMatrix to file.
[`write_h5`](#nltools.data.designmatrix.io.write_h5) | Write DesignMatrix to HDF5 file with metadata.



##### Classes

##### Functions###### `nltools.data.designmatrix.io.heatmap`

```python
heatmap(dm: DesignMatrix, figsize: tuple = (8, 6), **kwargs: tuple)
```

Visualize design matrix as heatmap (SPM-style).

Creates a heatmap visualization of the design matrix columns.
Uses seaborn + matplotlib under the hood.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`figsize` | <code>[tuple](#tuple)</code> | Figure size (width, height) in inches. Default: (8, 6). | <code>(8, 6)</code>
`**kwargs` |  | Additional keyword arguments passed to seaborn.heatmap(). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.axes.Axes: The axes object containing the heatmap.

**Examples:**

```pycon
>>> dm = DesignMatrix(np.random.randn(100, 3), columns=['a', 'b', 'c'])
>>> heatmap(dm)
```

###### `nltools.data.designmatrix.io.to_numpy`

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

###### `nltools.data.designmatrix.io.to_pandas`

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

###### `nltools.data.designmatrix.io.write`

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
HDF5 format preserves metadata (sampling_freq, convolved, polys).

</details>

###### `nltools.data.designmatrix.io.write_h5`

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

#### `nltools.data.designmatrix.regressors`

Standalone regressor functions for DesignMatrix.

Each function takes a DesignMatrix as its first argument (`dm`) and returns
a new DesignMatrix with the requested transformation applied.

**Functions:**

Name | Description
---- | -----------
[`add_dct_basis`](#nltools.data.designmatrix.regressors.add_dct_basis) | Add discrete cosine transform basis functions (high-pass filter).
[`add_poly`](#nltools.data.designmatrix.regressors.add_poly) | Add Legendre polynomial drift terms.
[`convolve`](#nltools.data.designmatrix.regressors.convolve) | Convolve columns with HRF or custom kernel.



##### Classes

##### Functions###### `nltools.data.designmatrix.regressors.add_dct_basis`

```python
add_dct_basis(dm: DesignMatrix, duration: float = 180, drop: int = 0) -> DesignMatrix
```

Add discrete cosine transform basis functions (high-pass filter).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix to add DCT basis to. | *required*
`duration` | <code>[float](#float)</code> | Filter duration in seconds. Default: 180. | <code>180</code>
`drop` | <code>[int](#int)</code> | Number of low-frequency bases to drop. Default: 0. | <code>0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with DCT basis columns appended.

###### `nltools.data.designmatrix.regressors.add_poly`

```python
add_poly(dm: DesignMatrix, order: int = 0, include_lower: bool = True) -> DesignMatrix
```

Add Legendre polynomial drift terms.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix to add polynomials to. | *required*
`order` | <code>[int](#int)</code> | Polynomial order (0=intercept, 1=linear, 2=quadratic, ...). Default: 0. | <code>0</code>
`include_lower` | <code>[bool](#bool)</code> | If True, include all orders from 0 to order. Default: True. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with polynomial columns appended.

###### `nltools.data.designmatrix.regressors.convolve`

```python
convolve(dm: DesignMatrix, conv_func: Union[str, np.ndarray] = 'hrf', columns: Optional[List[str]] = None) -> DesignMatrix
```

Convolve columns with HRF or custom kernel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix to convolve. | *required*
`conv_func` | <code>[str](#str) or [ndarray](#ndarray)</code> | 'hrf' for canonical Glover HRF, or custom kernel(s). Can be 1D array (single kernel) or 2D (samples x kernels) | <code>'hrf'</code>
`columns` | <code>list of str</code> | Columns to convolve (default: all non-polynomial columns) | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with convolved columns

**Examples:**

```pycon
>>> # Default HRF convolution
>>> dm_conv = convolve(dm)
```

```pycon
>>> # Custom kernel
>>> kernel = np.array([0.5, 1.0, 0.5])
>>> dm_conv = convolve(dm, conv_func=kernel)
```

```pycon
>>> # Multiple kernels (FIR model)
>>> kernels = np.array([[1.0, 0.5], [0.5, 1.0]]).T  # 2 kernels
>>> dm_conv = convolve(dm, conv_func=kernels)  # Creates col_c0, col_c1
```

#### `nltools.data.designmatrix.transforms`

Standalone transform functions for DesignMatrix.

Each function takes a DesignMatrix instance as the first argument (`dm`)
and returns a new DesignMatrix via `copy_with(dm,...)`.

**Functions:**

Name | Description
---- | -----------
[`downsample`](#nltools.data.designmatrix.transforms.downsample) | Reduce temporal resolution to target frequency using Polars-native operations.
[`standardize`](#nltools.data.designmatrix.transforms.standardize) | Standardize columns using the specified method.
[`upsample`](#nltools.data.designmatrix.transforms.upsample) | Increase temporal resolution to target frequency using Polars-native interpolation.
[`zscore`](#nltools.data.designmatrix.transforms.zscore) | Z-score standardize columns (mean=0, std=1).



##### Classes

##### Functions###### `nltools.data.designmatrix.transforms.downsample`

```python
downsample(dm: DesignMatrix, target: float, **kwargs: float) -> DesignMatrix
```

Reduce temporal resolution to target frequency using Polars-native operations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance to transform. | *required*
`target` | <code>[float](#float)</code> | Target sampling frequency in Hz (must be < current sampling_freq). | *required*
`**kwargs` |  | Additional keyword arguments:<br>- **method** (str): Aggregation method - 'mean' or 'median'.   Default: 'mean'. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Downsampled DesignMatrix with updated sampling_freq.

**Examples:**

```pycon
>>> dm = DesignMatrix({"a": list(range(100))}, sampling_freq=1.0)
>>> dm_down = downsample(dm, target=0.5)  # 1 Hz -> 0.5 Hz (100 -> 50 samples)
```

###### `nltools.data.designmatrix.transforms.standardize`

```python
standardize(dm: DesignMatrix, columns: Optional[List[str]] = None, method: str = 'zscore') -> DesignMatrix
```

Standardize columns using the specified method.

This method provides a consistent API with BrainData and Collection
for data normalization.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance to transform. | *required*
`columns` | <code>[Optional](#typing.Optional)[[List](#typing.List)[[str](#str)]]</code> | Columns to standardize. If None, standardize all non-polynomial columns. | <code>None</code>
`method` | <code>[str](#str)</code> | Standardization method. Options are: - 'zscore': Z-score standardization (mean=0, std=1) [default] - 'center': Mean centering only (mean=0) | <code>'zscore'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with standardized columns.

**Examples:**

```pycon
>>> dm = DesignMatrix(np.random.randn(100, 3))
>>> dm_z = standardize(dm, method='zscore')  # z-score all columns
>>> dm_c = standardize(dm, method='center')  # center only
```

###### `nltools.data.designmatrix.transforms.upsample`

```python
upsample(dm: DesignMatrix, target: float, method: str = 'linear', **kwargs: str) -> DesignMatrix
```

Increase temporal resolution to target frequency using Polars-native interpolation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance to transform. | *required*
`target` | <code>[float](#float)</code> | Target sampling frequency in Hz (must be > current sampling_freq) | *required*
`method` | <code>[str](#str)</code> | Interpolation method - 'linear' or 'nearest' (default: 'linear') | <code>'linear'</code>
`**kwargs` |  | Reserved for future extensions | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Upsampled DesignMatrix with updated sampling_freq.

**Examples:**

```pycon
>>> dm = DesignMatrix({"a": list(range(10))}, sampling_freq=1.0)
>>> dm_up = upsample(dm, target=2.0)  # 1 Hz -> 2 Hz (10 -> 19 samples)
```

###### `nltools.data.designmatrix.transforms.zscore`

```python
zscore(dm: DesignMatrix, columns: Optional[List[str]] = None) -> DesignMatrix
```

Z-score standardize columns (mean=0, std=1).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance to transform. | *required*
`columns` | <code>list of str</code> | Columns to standardize. If None, standardize all non-polynomial columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with standardized columns

#### `nltools.data.designmatrix.utils`

Shared helpers for DesignMatrix submodules.

These are internal utilities used by the facade and submodules — not part of the
public API.

**Functions:**

Name | Description
---- | -----------
[`copy_with`](#nltools.data.designmatrix.utils.copy_with) | Create new DesignMatrix with updated data/metadata.
[`get_data_columns`](#nltools.data.designmatrix.utils.get_data_columns) | Get column names, optionally excluding polynomials.
[`get_metadata`](#nltools.data.designmatrix.utils.get_metadata) | Extract metadata as dict (for copying).



##### Classes

##### Functions###### `nltools.data.designmatrix.utils.copy_with`

```python
copy_with(dm: DesignMatrix, new_df: pl.DataFrame, **metadata_updates: pl.DataFrame) -> DesignMatrix
```

Create new DesignMatrix with updated data/metadata.

This is the core pattern for immutable transformations.
All methods that transform data should use this helper.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Source DesignMatrix whose metadata to copy. | *required*
`new_df` | <code>[DataFrame](#polars.DataFrame)</code> | New underlying data. | *required*
`**metadata_updates` |  | Metadata attributes to override (e.g., convolved=['stim']). | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with updated data and metadata.

###### `nltools.data.designmatrix.utils.get_data_columns`

```python
get_data_columns(dm: DesignMatrix, exclude_polys: bool = True) -> List[str]
```

Get column names, optionally excluding polynomials.

This helper reduces code duplication across methods that need to
distinguish between data columns and polynomial/nuisance columns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`exclude_polys` | <code>bool, default=True</code> | If True, exclude polynomial/nuisance columns from the result. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[List](#typing.List)[[str](#str)]</code> | list of str: Column names (excluding polys if requested).

###### `nltools.data.designmatrix.utils.get_metadata`

```python
get_metadata(dm: DesignMatrix) -> dict
```

Extract metadata as dict (for copying).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys 'sampling_freq', 'convolved', 'polys', 'multi'.

