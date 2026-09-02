---
title: data.designmatrix.regressors
label: data-design-matrix-regressors
---

Provide standalone regressor functions for DesignMatrix.

Each function takes a DesignMatrix as its first argument (`dm`) and returns
a new DesignMatrix with the requested transformation applied.

**Functions:**

Name | Description
---- | -----------
[`add_dct_basis`](#data-design-matrix-regressors-add-dct-basis) | Add discrete cosine transform basis functions for high-pass filtering.
[`add_poly`](#data-design-matrix-regressors-add-poly) | Add Legendre polynomial drift terms.
[`convolve`](#data-design-matrix-regressors-convolve) | Convolve columns with an HRF or custom kernel.

## Functions

(data-design-matrix-regressors-add-dct-basis)=
### `add_dct_basis`

```python
add_dct_basis(dm: DesignMatrix, *, duration: float = 180, drop: int = 0, include_constant: bool = True) -> DesignMatrix
```

Add discrete cosine transform basis functions for high-pass filtering.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#data-design-matrix)</code> | DesignMatrix to add DCT basis to. | *required*
`duration` | <code>float</code> | Filter duration in seconds. Default: 180. | <code>180</code>
`drop` | <code>int</code> | Number of low-frequency bases to drop. Default: 0. | <code>0</code>
`include_constant` | <code>bool</code> | If True, also add a constant/intercept column named ``.nl_cosine_0`` (analogous to ``.nl_poly_0`` in `add_poly`). The underlying DCT basis drops the constant per SPM convention; set False to match SPM behavior. Default: True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#data-design-matrix)</code> | New DesignMatrix with DCT basis columns appended, named     ``.nl_cosine_{i}`` in the reserved namespace (see `RESERVED_PREFIX`).

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If sampling_freq is not set, or if the design already carries run-separated drift terms from a previous multi-run append.

(data-design-matrix-regressors-add-poly)=
### `add_poly`

```python
add_poly(dm: DesignMatrix, order: int = 0, include_lower: bool = True) -> DesignMatrix
```

Add Legendre polynomial drift terms.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#data-design-matrix)</code> | DesignMatrix to add polynomials to. | *required*
`order` | <code>int</code> | Polynomial order (0=intercept, 1=linear, 2=quadratic, ...). Default: 0. | <code>0</code>
`include_lower` | <code>bool</code> | If True, include all orders from 0 to order. Default: True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#data-design-matrix)</code> | New DesignMatrix with polynomial columns appended, named     ``.nl_poly_{order}`` in the reserved namespace (see `RESERVED_PREFIX`).

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If order < 0, or if the design already carries run-separated drift terms from a previous multi-run append.

(data-design-matrix-regressors-convolve)=
### `convolve`

```python
convolve(dm: DesignMatrix, conv_func: str | np.ndarray = 'hrf', columns: list[str] | None = None) -> DesignMatrix
```

Convolve columns with an HRF or custom kernel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#data-design-matrix)</code> | DesignMatrix to convolve. | *required*
`conv_func` | <code>str or ndarray</code> | 'hrf' for canonical Glover HRF, or custom kernel(s). Can be 1D array (single kernel) or 2D (samples x kernels) | <code>'hrf'</code>
`columns` | <code>list of str</code> | Columns to convolve (default: all non-confound columns) | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#data-design-matrix)</code> | New DesignMatrix with convolved columns

**Examples:**

```pycon
>>> # Default HRF convolution → produces 'stim_c0'
>>> dm_conv = convolve(dm)
```

```pycon
>>> # Custom 1-D kernel → produces 'stim_c0'
>>> kernel = np.array([0.5, 1.0, 0.5])
>>> dm_conv = convolve(dm, conv_func=kernel)
```

```pycon
>>> # Multiple kernels (FIR model) → produces 'stim_c0', 'stim_c1'
>>> kernels = np.array([[1.0, 0.5], [0.5, 1.0]]).T  # 2 kernels
>>> dm_conv = convolve(dm, conv_func=kernels)
```

<details class="note" open markdown="1">
<summary>Note</summary>

Convolved columns are always renamed to ``<col>_c{i}``; the source
column is dropped. ``dm.convolved`` records the post-suffix names
(the columns that actually exist in the returned dataframe), so
downstream metadata propagation through ``.append()`` stays in
sync with the dataframe.

</details>
