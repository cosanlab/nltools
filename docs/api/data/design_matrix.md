---
title: DesignMatrix
label: page-data-design-matrix
---

```python
DesignMatrix(data: DesignMatrix | pl.DataFrame | pd.DataFrame | np.ndarray | dict | str | Path | None = None, *, sampling_freq: float | None = None, TR: float | None = None, run_length: int | str | None = None, columns: list[str] | None = None, convolved: list[str] | None = None, confounds: list[str] | None = None, hrf_model: str | None = 'glover', n_rows: int | None = None)
```

Represent an experimental design for neuroimaging as a Polars-backed matrix.

Wraps a Polars DataFrame (one row per timepoint, one column per regressor)
together with the metadata a GLM needs: the sampling frequency, which
columns have been HRF-convolved, and which columns are nuisance/confound
regressors. Transformations return new instances with that metadata
preserved; `DesignMatrix` is composed over the DataFrame rather than
subclassing it. Unknown attributes are forwarded to the underlying
DataFrame, so the Polars API is available directly (``dm.select(...)``,
``dm.filter(...)``, ``dm.slice(...)`` return a `DesignMatrix`). Every eager DataFrame result becomes a new
`DesignMatrix`; Series and builder objects remain native Polars values.
Metadata is retained only when the operation establishes its validity.

`data` accepts a Polars DataFrame (copied), a pandas DataFrame
(converted), a NumPy array (named via `columns`), a dict of columns,
another `DesignMatrix` (copied), ``None`` (empty), or a file path.
A `.tsv`/`.csv` path is read as a BIDS events file when it has `onset`
and `duration` columns — each `trial_type` becomes a boxcar regressor,
HRF-convolved unless ``hrf_model=None`` — and as a plain table otherwise
(typically confounds). A `.h5`/`.hdf5` path written by `write` restores
the data and the metadata (`sampling_freq`, `convolved`, `confounds`,
`multi`), so neither `run_length` nor `sampling_freq` is required;
passing either overrides what the file recorded.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[DesignMatrix](#page-data-design-matrix) \| DataFrame \| DataFrame \| ndarray \| dict \| str \| Path \| None</code> | Input data; see above for how each type is interpreted. | <code>None</code>
`sampling_freq` | <code>float \| None</code> | Sampling frequency in Hz (1/TR for fMRI data). Mutually exclusive with `TR`. | <code>None</code>
`TR` | <code>float \| None</code> | Repetition time in seconds, a convenience for ``sampling_freq = 1/TR``. Mutually exclusive with `sampling_freq`. | <code>None</code>
`run_length` | <code>int \| str \| None</code> | Number of TRs in the run. Required when `data` is a path to a text file. Pass ``'infer'`` for tabular (confounds) files to accept whatever row count the file has; not valid for events files. Not used for `.h5` inputs, which carry their own length. | <code>None</code>
`columns` | <code>list[str] \| None</code> | Column names, used with NumPy input. | <code>None</code>
`convolved` | <code>list[str] \| None</code> | Names of columns that are already HRF-convolved. | <code>None</code>
`confounds` | <code>list[str] \| None</code> | Names of nuisance/confound columns (intercept, polynomial drift, DCT cosines, motion, …). | <code>None</code>
`hrf_model` | <code>str \| None</code> | HRF used to convolve regressors loaded from a BIDS events file. ``'glover'`` (the default, matching nilearn's ``make_first_level_design_matrix``) or ``None`` to keep raw boxcar regressors. Ignored for every other kind of `data`. | <code>'glover'</code>
`n_rows` | <code>int \| None</code> | Number of timepoints for a matrix with no columns (Polars cannot represent "n rows, 0 columns"). Rarely needed directly; set by `find_spikes` and by `append`. | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`data` | <code>DataFrame</code> | The underlying Polars DataFrame.
`sampling_freq` | <code>float \| None</code> | Sampling frequency in Hz.
`convolved` | <code>list[str]</code> | Names of HRF-convolved columns (read-only; managed by `convolve` and `append`).
`confounds` | <code>list[str]</code> | Names of nuisance/confound columns (read-only; managed by `add_poly`, `add_dct_basis`, `append`, and the constructor). Skipped by `convolve` and kept separate per run on multi-run vertical `append`.
`multi` | <code>bool</code> | True if the matrix was created by a multi-run vertical `append`.
`columns` | <code>list[str]</code> | Column names.
`shape` | <code>tuple[int, int]</code> | ``(n_rows, n_cols)``.
`is_empty` | <code>bool</code> | True if the matrix holds no data.

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
[`upsample`](#data-design-matrix-upsample) | Increase temporal resolution to a target frequency.
[`vif`](#data-design-matrix-vif) | Compute the variance inflation factor for each column.
[`with_columns`](#data-design-matrix-with-columns) | Add or replace columns via Polars expressions.
[`write`](#data-design-matrix-write) | Write DesignMatrix to file.
[`zscore`](#data-design-matrix-zscore) | Z-score standardize columns to mean zero and unit variance.

Passing another `DesignMatrix` returns a copy: `data`, `sampling_freq`,
`convolved`, `confounds`, and `multi` are carried over, and any explicit
kwarg overrides the inherited value.

When `data` is a path to a BIDS events file, the regressors are
HRF-convolved by default (``hrf_model='glover'``): output columns are
suffixed ``_c0`` and `convolved` is populated. Pass ``hrf_model=None``
to load raw boxcar regressors instead — useful for FIR designs, PPI
flows that build interaction terms before convolution, or teaching
material that introduces convolution as a separate step.



**Examples:**

```python
# Create from a NumPy array
dm = DesignMatrix(np.zeros((100, 2)), sampling_freq=0.5, columns=["a", "b"])

# Add a column
dm["stim"] = [0, 1, 1, 0] * 25

# Convolve with the HRF — convolved columns get a `_c0` suffix
dm_conv = dm.convolve()  # 'stim' → 'stim_c0'

# Add polynomial drift terms
dm_conv = dm_conv.add_poly(order=2)

# Multi-run concatenation separates drift terms per run
dm_run1 = DesignMatrix(run1_events, sampling_freq=0.5, run_length=100).add_poly(0)
dm_run2 = DesignMatrix(run2_events, sampling_freq=0.5, run_length=100).add_poly(0)
dm_multi = dm_run1.append(dm_run2, axis=0)  # → .nl_r0_poly_0, .nl_r1_poly_0
```

## Methods

(data-design-matrix-add-dct-basis)=
### `add_dct_basis`

```python
add_dct_basis(duration: float = 180, drop: int = 0, *, include_constant: bool = True) -> DesignMatrix
```

Add discrete cosine transform basis functions for high-pass filtering.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`duration` | <code>float</code> | Filter duration in seconds. Default: 180. | <code>180</code>
`drop` | <code>int</code> | Number of low-frequency bases to drop. Default: 0. | <code>0</code>
`include_constant` | <code>bool</code> | If True, also add a constant/intercept column named ``.nl_cosine_0`` (analogous to ``.nl_poly_0`` in `add_poly`). The underlying DCT basis drops the constant per SPM convention; set False to match SPM behavior. Default: True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#page-data-design-matrix)</code> | New DesignMatrix with DCT basis columns appended.

(data-design-matrix-add-poly)=
### `add_poly`

```python
add_poly(order: int = 0, include_lower: bool = True) -> DesignMatrix
```

Add Legendre polynomial drift terms.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`order` | <code>int</code> | Polynomial order (0=intercept, 1=linear, 2=quadratic, ...). Default: 0. | <code>0</code>
`include_lower` | <code>bool</code> | If True, include all orders from 0 to order. Default: True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#page-data-design-matrix)</code> | New DesignMatrix with polynomial columns appended.

(data-design-matrix-append)=
### `append`

```python
append(dm: DesignMatrix | list[DesignMatrix], *, axis: int = 0, keep_separate: bool = True, unique_cols: list[str] | None = None, fill_na: int | float | None = 0, as_confounds: bool = False, progress_bar: bool = False) -> DesignMatrix
```

Concatenate design matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>DesignMatrix or list of DesignMatrix</code> | Design matrix/matrices to append. | *required*
`axis` | <code>int</code> | 0 for row-wise (vertical), 1 for column-wise (horizontal). Default: 0. | <code>0</code>
`keep_separate` | <code>bool</code> | Whether to separate confound columns across runs (only applies when axis=0). Default: True. | <code>True</code>
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | <code>None</code>
`fill_na` | <code>int, float, or None</code> | Value to fill NaN values during vertical concatenation, or None to preserve nulls. Default: 0. | <code>0</code>
`as_confounds` | <code>bool</code> | Only applies when ``axis=1``. If True, mark all columns from ``dm`` as nuisance/confounds in the result — they get skipped by ``.convolve()`` and separated across runs on later vertical appends. Default: False. | <code>False</code>
`progress_bar` | <code>bool</code> | Print messages about confound separation. Default: False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#page-data-design-matrix)</code> | Concatenated design matrix.

(data-design-matrix-clean)=
### `clean`

```python
clean(*, fill_na: int | float | None = 0, exclude_confounds: bool = False, thresh: float = 0.95, progress_bar: bool = False) -> DesignMatrix
```

Remove highly correlated columns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fill_na` | <code>int, float, or None</code> | Fill NaN values before checking correlations (default 0) | <code>0</code>
`exclude_confounds` | <code>bool</code> | Skip confound/nuisance columns from correlation check | <code>False</code>
`thresh` | <code>float</code> | Correlation threshold (drop if abs(r) >= thresh, default 0.95) | <code>0.95</code>
`progress_bar` | <code>bool</code> | Print dropped column names. Default: False | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#page-data-design-matrix)</code> | Cleaned matrix with highly correlated columns removed

(data-design-matrix-convolve)=
### `convolve`

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
`conv_func` | <code>str or ndarray</code> | 'hrf' for canonical Glover HRF, or custom kernel(s). Can be 1D array (single kernel) or 2D (samples x kernels). | <code>'hrf'</code>
`columns` | <code>list of str</code> | Columns to convolve (default: all non-confound columns). | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#page-data-design-matrix)</code> | New DesignMatrix with convolved columns renamed.

(data-design-matrix-copy)=
### `copy`

```python
copy() -> DesignMatrix
```

Create a deep copy of the DesignMatrix.

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#page-data-design-matrix)</code> | Copy of the current DesignMatrix

(data-design-matrix-corr)=
### `corr`

```python
corr(*, metric: str = 'pearson', columns: list[str] | None = None)
```

Calculate column correlations as a similarity ``Adjacency``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` | <code>str</code> | ``'pearson'`` (default) or ``'spearman'``. | <code>'pearson'</code>
`columns` | <code>list of str</code> | Subset of columns to correlate. Defaults to all columns. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[Adjacency](#page-data-adjacency)</code> | Similarity matrix whose ``labels`` are the column names.     The unit diagonal is dropped (self-correlation isn't an edge);     use ``.plot(method='corr')`` for a heatmap with the diagonal     restored.

(data-design-matrix-downsample)=
### `downsample`

```python
downsample(target: float, method: str = 'mean') -> DesignMatrix
```

Reduce temporal resolution using Polars-native operations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` | <code>float</code> | Target sampling frequency in Hz (must be < current sampling_freq) | *required*
`method` | <code>str</code> | Aggregation method - 'mean' or 'median' (default: 'mean') | <code>'mean'</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#page-data-design-matrix)</code> | Downsampled DesignMatrix with updated sampling_freq

(data-design-matrix-drop)=
### `drop`

```python
drop(columns: list[str]) -> DesignMatrix
```

Drop specified columns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`columns` | <code>list of str</code> | Column names to remove. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#page-data-design-matrix)</code> | New DesignMatrix without the specified columns.

(data-design-matrix-fillna)=
### `fillna`

```python
fillna(value: int | float) -> DesignMatrix
```

Fill NaN/null values with specified value.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`value` | <code>int or float</code> | Value to replace NaN/null entries with. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#page-data-design-matrix)</code> | New DesignMatrix with NaN/null values replaced.

(data-design-matrix-plot)=
### `plot`

```python
plot(method: str = 'matrix', *, columns: list[str] | None = None, rescale: bool = True, metric: str = 'pearson', ax: str = None, figsize: tuple | None = None, title: str | None = None, cmap: str | None = None, save: str | None = None, **kwargs: str | None) -> Figure
```

Visualize the design matrix.

Dispatches over `method` (mirroring `BrainData.plot`):

- ``'matrix'`` (default): SPM-style heatmap (rows = TRs, columns = regressors).
- ``'timeseries'``: overlaid line plot of regressor time courses. Pass
  the same `ax` across calls to overlay multiple DesignMatrices
  (e.g. original vs. convolved).
- ``'corr'``: labeled correlation heatmap of the columns (reuses
  `corr`; diagonal restored to 1.0 for display).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>str</code> | One of ``'matrix'``, ``'timeseries'``, or ``'corr'``. Default: ``'matrix'``. | <code>'matrix'</code>
`columns` | <code>list of str</code> | Subset of columns to plot. Defaults to all columns. | <code>None</code>
`rescale` | <code>bool</code> | ``'matrix'`` only. Rescale each column by its L2 norm so columns with different native magnitudes are visually comparable (SPM/nilearn convention). Default: True. | <code>True</code>
`metric` | <code>str</code> | ``'corr'`` only. ``'pearson'`` (default) or ``'spearman'``. | <code>'pearson'</code>
`ax` | <code>Axes</code> | Existing axis to draw on; a new figure is created if omitted. | <code>None</code>
`figsize` | <code>tuple</code> | Figure size; sensible per-method default when omitted. | <code>None</code>
`title` | <code>str</code> | Axis title. | <code>None</code>
`cmap` | <code>str</code> | Colormap (``'matrix'`` / ``'corr'``). | <code>None</code>
`save` | <code>str</code> | Path to save the figure. | <code>None</code>
`**kwargs` | <code>dict</code> | Forwarded to the underlying plotter (``seaborn.heatmap`` for ``'matrix'`` / ``'corr'``; ``matplotlib.axes.Axes.plot`` for ``'timeseries'``). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>Figure</code> | The figure containing the plot.

(data-design-matrix-replace-data)=
### `replace_data`

```python
replace_data(data: np.ndarray, column_names: list[str] | None = None) -> DesignMatrix
```

Replace data columns while preserving confounds and metadata.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>ndarray</code> | New data array (must match number of rows in current DesignMatrix) | *required*
`column_names` | <code>list of str</code> | Names for new data columns. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#page-data-design-matrix)</code> | New DesignMatrix with replaced data columns, preserved confounds

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If row count doesn't match existing data

(data-design-matrix-standardize)=
### `standardize`

```python
standardize(method: str = 'zscore', columns: list[str] | None = None) -> DesignMatrix
```

Standardize columns using the specified method.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>str</code> | ``'zscore'`` (mean 0, std 1) or ``'center'`` (mean 0 only). Default: ``'zscore'``. | <code>'zscore'</code>
`columns` | <code>list[str] \| None</code> | Columns to standardize. If None, standardize all non-confound columns. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#page-data-design-matrix)</code> | New DesignMatrix with standardized columns.

(data-design-matrix-sum)=
### `sum`

```python
sum(axis: int = 0) -> pl.Series
```

Compute the sum along an axis.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>int</code> | 0 to sum down each column, 1 to sum across each row. Default: 0. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>Series</code> | Sums along the specified axis.

(data-design-matrix-to-numpy)=
### `to_numpy`

```python
to_numpy() -> np.ndarray
```

Convert a DesignMatrix to a NumPy array.

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | 2D array with shape (n_samples, n_columns)

(data-design-matrix-upsample)=
### `upsample`

```python
upsample(target: float, method: str = 'linear') -> DesignMatrix
```

Increase temporal resolution to a target frequency.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` | <code>float</code> | Target sampling frequency in Hz (must be > current sampling_freq) | *required*
`method` | <code>str</code> | Interpolation method - 'linear' or 'nearest' (default: 'linear') | <code>'linear'</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#page-data-design-matrix)</code> | Upsampled DesignMatrix with updated sampling_freq

(data-design-matrix-vif)=
### `vif`

```python
vif(exclude_confounds: bool = True) -> np.ndarray | None
```

Compute the variance inflation factor for each column.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`exclude_confounds` | <code>bool</code> | Skip confound/nuisance columns. Default: True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | VIF values for each included column. Returns None if the     correlation matrix is singular.

(data-design-matrix-with-columns)=
### `with_columns`

```python
with_columns(*exprs, **named_exprs) -> DesignMatrix
```

Add or replace columns via Polars expressions.

Mirrors ``pl.DataFrame.with_columns``. Named kwargs become named
columns; positional ``pl.Expr`` arguments are accepted as-is
(including ``pl.Expr.alias("name")``). Returns a new `DesignMatrix`
preserving annotations on untouched columns. Replacing a column clears
its convolution annotation and retains its confound role; new columns
are untagged.

For convenience, named-kwarg values that aren't ``pl.Expr`` /
``pl.Series`` are coerced: an ``int``/``float`` is broadcast as a
scalar via ``pl.lit``, and a ``list`` / ``np.ndarray`` is wrapped as a
``pl.Series``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`*exprs` | <code>Expr</code> | Positional Polars expressions, passed through. | <code>()</code>
`**named_exprs` | <code>Expr \| Series \| ndarray \| list \| int \| float</code> | New columns keyed by name. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#page-data-design-matrix)</code> | New DesignMatrix with the columns added or replaced.

**Examples:**

```python
dm = dm.with_columns(motor=pl.sum_horizontal(motor_cols)).drop(motor_cols)
dm = dm.with_columns(
    vmpfc=seed_signal,
    vmpfc_motor=pl.col("vmpfc") * pl.col("motor_c0"),
)
```

(data-design-matrix-write)=
### `write`

```python
write(file_name: str, sep: str | None = None) -> None
```

Write DesignMatrix to file.

Supports TSV, CSV, and HDF5 formats. Format is auto-detected from the
file extension. Text formats carry the data only; ``.h5`` also
preserves ``sampling_freq``, ``.convolved``, ``.confounds``, and
``.multi``, so ``DesignMatrix(path)`` restores the whole object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>str</code> | Output file path with a `.tsv`, `.csv`, `.h5`, or `.hdf5` extension. | *required*
`sep` | <code>str \| None</code> | Column separator for text files. Defaults to the delimiter the extension implies (comma for `.csv`, tab otherwise); pass a value to override. | <code>None</code>

(data-design-matrix-zscore)=
### `zscore`

```python
zscore(columns: list[str] | None = None) -> DesignMatrix
```

Z-score standardize columns to mean zero and unit variance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`columns` | <code>list of str</code> | Columns to standardize. If None, standardize all non-confound columns. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#page-data-design-matrix)</code> | New DesignMatrix with standardized columns
