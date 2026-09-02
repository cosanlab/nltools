---
title: data.designmatrix.diagnostics
label: page-data-design-matrix-diagnostics
---

Collinearity diagnostics for DesignMatrix: column correlations, VIF, and cleanup.

**Functions:**

Name | Description
---- | -----------
[`clean`](#data-design-matrix-diagnostics-clean) | Remove highly correlated columns.
[`corr`](#data-design-matrix-diagnostics-corr) | Correlation between DesignMatrix columns as an Adjacency.
[`vif`](#data-design-matrix-diagnostics-vif) | Compute the variance inflation factor for each column.

## Functions

(data-design-matrix-diagnostics-clean)=
### `clean`

```python
clean(dm: DesignMatrix, *, fill_na: int | float | None = 0, exclude_confounds: bool = False, thresh: float = 0.95, progress_bar: bool = False) -> DesignMatrix
```

Remove highly correlated columns.

Removes columns with correlation >= threshold. Keeps first instance
of correlated pair, drops duplicates.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#page-data-design-matrix)</code> | DesignMatrix instance. | *required*
`fill_na` | <code>int \| float \| None</code> | Fill NaN values before checking correlations. Default: 0. | <code>0</code>
`exclude_confounds` | <code>bool</code> | Skip nuisance/confound columns from correlation check. Default: False. | <code>False</code>
`thresh` | <code>float</code> | Correlation threshold (drop if abs(r) >= thresh). Default: 0.95. | <code>0.95</code>
`progress_bar` | <code>bool</code> | Print dropped column names. Default: False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#page-data-design-matrix)</code> | Cleaned matrix with highly correlated columns removed

(data-design-matrix-diagnostics-corr)=
### `corr`

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
`dm` | <code>[DesignMatrix](#page-data-design-matrix)</code> | DesignMatrix instance. | *required*
`metric` | <code>str</code> | ``'pearson'`` (default) or ``'spearman'``. Spearman is computed as Pearson on column ranks. | <code>'pearson'</code>
`columns` | <code>list[str] \| None</code> | Subset of columns to correlate. Defaults to all columns. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[Adjacency](#page-data-adjacency)</code> | Similarity matrix whose ``labels`` are the included column     names.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If ``metric`` is unknown or fewer than 2 columns remain.

<details class="note" open markdown="1">
<summary>Note</summary>

Constant columns (e.g. the ``.nl_poly_0`` intercept) have zero variance and
yield NaN correlations.

</details>

(data-design-matrix-diagnostics-vif)=
### `vif`

```python
vif(dm: DesignMatrix, exclude_confounds: bool = True) -> np.ndarray | None
```

Compute the variance inflation factor for each column.

Uses diagonal elements of inverted correlation matrix
(same method as Matlab and R).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#page-data-design-matrix)</code> | DesignMatrix instance. | *required*
`exclude_confounds` | <code>bool</code> | Skip nuisance/confound columns. Default: True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray \| None</code> | VIF values for each included column, or None if the     correlation matrix is singular (perfect collinearity detected).

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If the DesignMatrix has only 1 column.
