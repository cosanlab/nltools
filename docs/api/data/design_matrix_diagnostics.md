## `nltools.data.designmatrix.diagnostics`

Diagnostic and utility functions for DesignMatrix.

**Functions:**

Name | Description
---- | -----------
[`clean`](#nltools.data.designmatrix.diagnostics.clean) | Remove highly correlated columns.
[`details`](#nltools.data.designmatrix.diagnostics.details) | Return human-readable metadata summary.
[`vif`](#nltools.data.designmatrix.diagnostics.vif) | Compute variance inflation factor for each column.



### Classes

### Functions#### `nltools.data.designmatrix.diagnostics.clean`

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

#### `nltools.data.designmatrix.diagnostics.details`

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

#### `nltools.data.designmatrix.diagnostics.vif`

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

