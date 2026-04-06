## `regressors`

Standalone regressor functions for DesignMatrix.

Each function takes a DesignMatrix as its first argument (`dm`) and returns
a new DesignMatrix with the requested transformation applied.

**Methods:**

Name | Description
---- | -----------
[`add_dct_basis`](#add_dct_basis) | Add discrete cosine transform basis functions (high-pass filter).
[`add_poly`](#add_poly) | Add Legendre polynomial drift terms.
[`convolve`](#convolve) | Convolve columns with HRF or custom kernel.



### Classes

### Methods

#### `add_dct_basis`

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

#### `add_poly`

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

#### `convolve`

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

