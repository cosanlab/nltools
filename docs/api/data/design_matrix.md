(data-design-matrix-designmatrix)=
## `DesignMatrix`

```python
DesignMatrix(data: DesignMatrix | pl.DataFrame | pd.DataFrame | np.ndarray | dict | str | Path | None = None, *, sampling_freq: float | None = None, TR: float | None = None, run_length: int | str | None = None, columns: list[str] | None = None, convolved: list[str] | None = None, confounds: list[str] | None = None, hrf_model: str | None = 'glover')
```

Represent experimental designs for neuroimaging with Polars.

This is a Polars-based design matrix for experimental designs in
neuroimaging.

Wraps a Polars DataFrame with neuroimaging-specific metadata and methods.
Uses composition pattern (not subclassing) for clean metadata preservation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>DataFrame, ndarray, dict, str/Path, or None</code> | Input data. Accepts: - Polars DataFrame (zero-copy) - pandas DataFrame (converted to Polars) - numpy ndarray - dict (keys=columns, values=data) - str or Path to a `.tsv`/`.csv` file. BIDS events files   (containing `onset` and `duration` columns) are converted to   boxcar regressors — call ``convolve()`` afterwards if you want   HRF convolution. Any other tabular file is read as-is and is   typically used for confounds. - None (empty initialization) | <code>None</code>
`sampling_freq` | <code>[float](#float)</code> | Sampling frequency in Hz (1/TR for fMRI data). Mutually exclusive with ``TR``. | <code>None</code>
`TR` | <code>[float](#float)</code> | Repetition time in seconds. Convenience for ``sampling_freq = 1/TR``. Mutually exclusive with ``sampling_freq``. | <code>None</code>
`run_length` | <code>[int](#int) or 'infer'</code> | Required when ``data`` is a file path. Number of TRs in the run. Pass ``'infer'`` for tabular/confounds files to accept whatever row count the file has (not valid for events files). | <code>None</code>
`columns` | <code>list of str</code> | Column names (used with ndarray input) | <code>None</code>
`convolved` | <code>list of str</code> | Names of convolved columns (tracked internally) | <code>None</code>
`confounds` | <code>list of str</code> | Names of nuisance/confound columns (intercept, polynomial drift, DCT cosines, motion, …) tracked internally | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`sampling_freq` | <code>[float](#float) or None</code> | Sampling frequency in Hz
`convolved` | <code>list of str</code> | Columns that have been convolved
`confounds` | <code>list of str</code> | Nuisance/confound columns (intercept, polynomial trends, DCT bases, motion, physio, …) — these are skipped by ``.convolve()`` and kept separate per run on multi-run vertical append.
`multi` | <code>[bool](#bool)</code> | True if created from multi-run concatenation

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
>>> # Convolve with HRF — convolved columns get a `_c0` suffix
>>> dm_conv = dm.convolve()  # 'stim' → 'stim_c0'
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
[`add_dct_basis`](#data-design-matrix-add-dct-basis) | Add discrete cosine transform basis functions for high-pass filtering.
[`add_poly`](#data-design-matrix-add-poly) | Add Legendre polynomial drift terms.
[`append`](#data-design-matrix-append) | Concatenate design matrices.
[`clean`](#data-design-matrix-clean) | Remove highly correlated columns.
[`convolve`](#data-design-matrix-convolve) | Convolve columns with an HRF or custom kernel.
[`copy`](#data-design-matrix-copy) | Create a deep copy of the DesignMatrix.
[`corr`](#data-design-matrix-corr) | Calculate column correlations as a similarity ``Adjacency``.
[`downsample`](#data-design-matrix-downsample) | Reduce temporal resolution using Polars-native operations.
[`drop`](#data-design-matrix-drop) | Drop specified columns.
[`fillna`](#data-design-matrix-fillna) | Fill NaN/null values with specified value.
[`plot`](#data-design-matrix-plot) | Visualize the design matrix.
[`replace_data`](#data-design-matrix-replace-data) | Replace data columns while preserving confounds and metadata.
[`standardize`](#data-design-matrix-standardize) | Standardize columns using the specified method.
[`sum`](#data-design-matrix-sum) | Compute the sum along an axis.
[`to_numpy`](#data-design-matrix-to-numpy) | Convert a DesignMatrix to a NumPy array.
[`to_pandas`](#data-design-matrix-to-pandas) | Convert DesignMatrix to pandas DataFrame.
[`upsample`](#data-design-matrix-upsample) | Increase temporal resolution to a target frequency.
[`vif`](#data-design-matrix-vif) | Compute the variance inflation factor for each column.
[`with_columns`](#data-design-matrix-with-columns) | Add or replace columns via Polars expressions.
[`write`](#data-design-matrix-write) | Write DesignMatrix to file.
[`zscore`](#data-design-matrix-zscore) | Z-score standardize columns to mean zero and unit variance.

Passing another ``DesignMatrix`` returns a copy: ``data``,
``sampling_freq``, ``convolved``, ``confounds``, and ``multi`` are
carried over. Any explicit kwarg overrides the inherited value.

When ``data`` is a path to a BIDS events file, the constructor
HRF-convolves the regressors by default (``hrf_model='glover'``,
matching nilearn's ``make_first_level_design_matrix``). The output
columns are suffixed ``_c0`` and ``.convolved`` is populated. Pass
``hrf_model=None`` to load raw boxcar regressors instead — useful
for FIR designs, PPI flows that build interaction terms before
convolution, or pedagogical material that introduces convolution
as a separate step. ``hrf_model`` is silently ignored when ``data``
is anything other than an events file.

### Methods

(data-design-matrix-add-dct-basis)=
#### `add_dct_basis`

```python
add_dct_basis(duration: float = 180, drop: int = 0, *, include_constant: bool = True) -> DesignMatrix
```

Add discrete cosine transform basis functions for high-pass filtering.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`duration` | <code>[float](#float)</code> | Filter duration in seconds. Default: 180. | <code>180</code>
`drop` | <code>[int](#int)</code> | Number of low-frequency bases to drop. Default: 0. | <code>0</code>
`include_constant` | <code>[bool](#bool)</code> | If True, also add a constant/intercept column named ``cosine_0`` (analogous to ``poly_0`` in `add_poly`). The underlying DCT basis drops the constant per SPM convention; set False to match SPM behavior. Default: True. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with DCT basis columns appended.

(data-design-matrix-add-poly)=
#### `add_poly`

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

(data-design-matrix-append)=
#### `append`

```python
append(dm: DesignMatrix | list[DesignMatrix], *, axis: int = 0, keep_separate: bool = True, unique_cols: list[str] | None = None, fill_na: int | float = 0, as_confounds: bool = False, verbose: bool = False) -> DesignMatrix
```

Concatenate design matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>DesignMatrix or list of DesignMatrix</code> | Design matrix/matrices to append. | *required*
`axis` | <code>[int](#int)</code> | 0 for row-wise (vertical), 1 for column-wise (horizontal). Default: 0. | <code>0</code>
`keep_separate` | <code>[bool](#bool)</code> | Whether to separate confound columns across runs (only applies when axis=0). Default: True. | <code>True</code>
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | <code>None</code>
`fill_na` | <code>[int](#int) or [float](#float)</code> | Value to fill NaN values during vertical concatenation. Default: 0. | <code>0</code>
`as_confounds` | <code>[bool](#bool)</code> | Only applies when ``axis=1``. If True, mark all columns from ``dm`` as nuisance/confounds in the result — they get skipped by ``.convolve()`` and separated across runs on later vertical appends. Default: False. | <code>False</code>
`verbose` | <code>[bool](#bool)</code> | Print messages about confound separation. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Concatenated design matrix.

(data-design-matrix-clean)=
#### `clean`

```python
clean(fill_na: int | float | None = 0, exclude_confounds: bool = False, thresh: float = 0.95, verbose: bool = True) -> DesignMatrix
```

Remove highly correlated columns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fill_na` | <code>int, float, or None</code> | Fill NaN values before checking correlations (default 0) | <code>0</code>
`exclude_confounds` | <code>[bool](#bool)</code> | Skip confound/nuisance columns from correlation check | <code>False</code>
`thresh` | <code>[float](#float)</code> | Correlation threshold (drop if abs(r) >= thresh, default 0.95) | <code>0.95</code>
`verbose` | <code>[bool](#bool)</code> | Print dropped column names | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Cleaned matrix with highly correlated columns removed

(data-design-matrix-convolve)=
#### `convolve`

```python
convolve(conv_func: str | np.ndarray = 'hrf', columns: list[str] | None = None) -> DesignMatrix
```

Convolve columns with an HRF or custom kernel.

Convolved columns are always renamed to ``<col>_c{i}`` (where ``i`` is
the kernel index, ``0`` for a single 1-D kernel). The source columns
are dropped, and ``self.convolved`` lists the post-suffix names so
downstream metadata stays in sync with the dataframe.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`conv_func` | <code>[str](#str) or [ndarray](#ndarray)</code> | 'hrf' for canonical Glover HRF, or custom kernel(s). Can be 1D array (single kernel) or 2D (samples x kernels). | <code>'hrf'</code>
`columns` | <code>list of str</code> | Columns to convolve (default: all non-confound columns). | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with convolved columns renamed.

(data-design-matrix-copy)=
#### `copy`

```python
copy() -> DesignMatrix
```

Create a deep copy of the DesignMatrix.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Copy of the current DesignMatrix

(data-design-matrix-corr)=
#### `corr`

```python
corr(*, metric: str = 'pearson', columns: list[str] | None = None)
```

Calculate column correlations as a similarity ``Adjacency``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` | <code>[str](#str)</code> | ``'pearson'`` (default) or ``'spearman'``. | <code>'pearson'</code>
`columns` | <code>list of str</code> | Subset of columns to correlate. Defaults to all columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | Similarity matrix whose ``labels`` are the column names. The unit diagonal is dropped (self-correlation isn't an edge); use ``.plot(method='corr')`` for a heatmap with the diagonal restored.

(data-design-matrix-downsample)=
#### `downsample`

```python
downsample(target: float, method: str = 'mean', **kwargs: str) -> DesignMatrix
```

Reduce temporal resolution using Polars-native operations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` | <code>[float](#float)</code> | Target sampling frequency in Hz (must be < current sampling_freq) | *required*
`method` | <code>[str](#str)</code> | Aggregation method - 'mean' or 'median' (default: 'mean') | <code>'mean'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Downsampled DesignMatrix with updated sampling_freq

(data-design-matrix-drop)=
#### `drop`

```python
drop(columns: list[str]) -> DesignMatrix
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

(data-design-matrix-fillna)=
#### `fillna`

```python
fillna(value: int | float) -> DesignMatrix
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

(data-design-matrix-plot)=
#### `plot`

```python
plot(method: str = 'matrix', *, columns: list[str] | None = None, rescale: bool = True, metric: str = 'pearson', ax: str = None, figsize: tuple | None = None, title: str | None = None, cmap: str | None = None, save: str | None = None, **kwargs: str | None)
```

Visualize the design matrix.

Dispatches over ``method`` (mirroring ``BrainData.plot``):

- ``'matrix'`` (default): SPM-style heatmap (rows=TRs, cols=regressors).
- ``'timeseries'``: overlaid line plot of regressor time courses. Pass
  the same ``ax`` across calls to overlay multiple DesignMatrices
  (e.g. original vs. convolved).
- ``'corr'``: labeled correlation heatmap of the columns (reuses
  `corr`; diagonal restored to 1.0 for display).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ``'matrix'`` | ``'timeseries'`` | ``'corr'``. Default: ``'matrix'``. | <code>'matrix'</code>
`columns` | <code>list of str</code> | Subset of columns to plot. Defaults to all columns. | <code>None</code>
`rescale` | <code>[bool](#bool)</code> | ``'matrix'`` only. Rescale each column by its L2 norm so columns with different native magnitudes are visually comparable (SPM/nilearn convention). Default: True. | <code>True</code>
`metric` | <code>[str](#str)</code> | ``'corr'`` only. ``'pearson'`` (default) or ``'spearman'``. | <code>'pearson'</code>
`ax` | <code>[Axes](#matplotlib.axes.Axes)</code> | Existing axis to draw on; a new figure is created if omitted. | <code>None</code>
`figsize` | <code>[tuple](#tuple)</code> | Figure size; sensible per-method default when omitted. | <code>None</code>
`title` | <code>[str](#str)</code> | Axis title. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Colormap (``'matrix'`` / ``'corr'``). | <code>None</code>
`save` | <code>[str](#str)</code> | Path to save the figure. | <code>None</code>
`**kwargs` |  | Forwarded to the underlying plotter (``seaborn.heatmap`` for ``'matrix'`` / ``'corr'``; ``Axes.plot`` for ``'timeseries'``). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure: The figure containing the plot.

(data-design-matrix-replace-data)=
#### `replace_data`

```python
replace_data(data: np.ndarray, column_names: list[str] | None = None) -> DesignMatrix
```

Replace data columns while preserving confounds and metadata.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#ndarray)</code> | New data array (must match number of rows in current DesignMatrix) | *required*
`column_names` | <code>list of str</code> | Names for new data columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with replaced data columns, preserved confounds

(data-design-matrix-standardize)=
#### `standardize`

```python
standardize(method: str = 'zscore', columns: list[str] | None = None) -> DesignMatrix
```

Standardize columns using the specified method.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Standardization method ('zscore' or 'center'). Default: 'zscore'. | <code>'zscore'</code>
`columns` | <code>[list](#list)[[str](#str)] \| None</code> | Columns to standardize. If None, standardize all non-polynomial columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with standardized columns.

(data-design-matrix-sum)=
#### `sum`

```python
sum(axis: int = 0) -> pl.Series
```

Compute the sum along an axis.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>int, default=0</code> | 0: sum down columns, 1: sum across rows. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>[Series](#polars.Series)</code> | pl.Series: Sums along specified axis.

(data-design-matrix-to-numpy)=
#### `to_numpy`

```python
to_numpy() -> np.ndarray
```

Convert a DesignMatrix to a NumPy array.

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | np.ndarray: 2D array with shape (n_samples, n_columns)

(data-design-matrix-to-pandas)=
#### `to_pandas`

```python
to_pandas() -> pd.DataFrame
```

Convert DesignMatrix to pandas DataFrame.

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#pandas.DataFrame)</code> | pd.DataFrame: Pandas DataFrame with same data and column names.

(data-design-matrix-upsample)=
#### `upsample`

```python
upsample(target: float, method: str = 'linear', **kwargs: str) -> DesignMatrix
```

Increase temporal resolution to a target frequency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` | <code>[float](#float)</code> | Target sampling frequency in Hz (must be > current sampling_freq) | *required*
`method` | <code>[str](#str)</code> | Interpolation method - 'linear' or 'nearest' (default: 'linear') | <code>'linear'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Upsampled DesignMatrix with updated sampling_freq

(data-design-matrix-vif)=
#### `vif`

```python
vif(exclude_confounds: bool = True) -> np.ndarray | None
```

Compute the variance inflation factor for each column.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`exclude_confounds` | <code>[bool](#bool)</code> | Skip confound/nuisance columns. Default: True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray) \| None</code> | np.ndarray: VIF values for each included column. Returns None if the correlation matrix is singular.

(data-design-matrix-with-columns)=
#### `with_columns`

```python
with_columns(*exprs, **named_exprs) -> DesignMatrix
```

Add or replace columns via Polars expressions.

Mirrors `DataFrame.with_columns`. Named kwargs become
named columns; positional ``pl.Expr`` arguments are accepted as-is
(including ``pl.Expr.alias("name")``). Returns a new ``DesignMatrix``
with metadata preserved; new columns are *not* auto-tagged as
convolved or confounds.

For convenience, named-kwarg values that aren't ``pl.Expr`` /
``pl.Series`` are coerced:

- ``int``/``float`` → broadcast scalar via ``pl.lit``
- ``list`` / ``np.ndarray`` → wrapped as ``pl.Series``

**Examples:**

```pycon
>>> dm = dm.with_columns(motor=pl.sum_horizontal(motor_cols)).drop(motor_cols)
>>> dm = dm.with_columns(
...     vmpfc=seed_signal,
...     vmpfc_motor=pl.col("vmpfc") * pl.col("motor_c0"),
... )
```

(data-design-matrix-write)=
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

(data-design-matrix-zscore)=
#### `zscore`

```python
zscore(columns: list[str] | None = None) -> DesignMatrix
```

Z-score standardize columns to mean zero and unit variance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`columns` | <code>list of str</code> | Columns to standardize. If None, standardize all non-polynomial columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with standardized columns

