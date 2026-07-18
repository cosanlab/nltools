(data-design-matrix-diagnostics-diagnostics)=
## `diagnostics`

Diagnostic and utility functions for DesignMatrix.

**Methods:**

Name | Description
---- | -----------
[`clean`](#data-design-matrix-diagnostics-clean) | Remove highly correlated columns.
[`corr`](#data-design-matrix-diagnostics-corr) | Correlation between DesignMatrix columns as an Adjacency.
[`vif`](#data-design-matrix-diagnostics-vif) | Compute the variance inflation factor for each column.



### Classes

### Methods

(data-design-matrix-diagnostics-clean)=
#### `clean`

```python
clean(dm: DesignMatrix, *, fill_na: int | float | None = 0, exclude_confounds: bool = False, thresh: float = 0.95, progress_bar: bool = False) -> DesignMatrix
```

Remove highly correlated columns.

Removes columns with correlation >= threshold. Keeps first instance
of correlated pair, drops duplicates.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`fill_na` | <code>int, float, or None</code> | Fill NaN values before checking correlations. Default: 0. | <code>0</code>
`exclude_confounds` | <code>[bool](#bool)</code> | Skip nuisance/confound columns from correlation check. Default: False. | <code>False</code>
`thresh` | <code>[float](#float)</code> | Correlation threshold (drop if abs(r) >= thresh). Default: 0.95. | <code>0.95</code>
`progress_bar` | <code>[bool](#bool)</code> | Print dropped column names. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Cleaned matrix with highly correlated columns removed

(data-design-matrix-diagnostics-corr)=
#### `corr`

```python
corr(dm: DesignMatrix, *, metric: str = 'pearson', columns: list[str] | None = None) -> Adjacency
```

Correlation between DesignMatrix columns as an Adjacency.

Returns the column-by-column correlation matrix wrapped in an nltools
``Adjacency`` (``matrix_type='similarity'``) so it composes with the rest
of the similarity-matrix tooling (``.plot()``, MDS, etc.). The Adjacency
stores only the off-diagonal entries — self-correlation isn't a meaningful
edge — so the unit diagonal is implicit; ``DesignMatrix.plot(method='corr')``
restores it for display.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`metric` | <code>[str](#str)</code> | ``'pearson'`` (default) or ``'spearman'``. Spearman is computed as Pearson on column ranks. | <code>'pearson'</code>
`columns` | <code>list of str</code> | Subset of columns to correlate. Defaults to all columns. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` | <code>[Adjacency](#nltools.data.Adjacency)</code> | Similarity matrix whose ``labels`` are the included column names.

<details class="note" open markdown="1">
<summary>Note</summary>

Constant columns (e.g. the ``poly_0`` intercept) have zero variance and
yield NaN correlations.

</details>

(data-design-matrix-diagnostics-vif)=
#### `vif`

```python
vif(dm: DesignMatrix, exclude_confounds: bool = True) -> np.ndarray | None
```

Compute the variance inflation factor for each column.

Uses diagonal elements of inverted correlation matrix
(same method as Matlab and R).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | DesignMatrix instance. | *required*
`exclude_confounds` | <code>[bool](#bool)</code> | Skip nuisance/confound columns. Default: True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray) \| None</code> | np.ndarray: VIF values for each included column. Returns None if the correlation matrix is singular (perfect collinearity detected).

