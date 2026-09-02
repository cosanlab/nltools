---
title: data.designmatrix.append
label: page-data-design-matrix-append
---

```python
append(dm: DesignMatrix, other: DesignMatrix, *, axis: int = 0, keep_separate: bool = True, unique_cols: list[str] | None = None, fill_na: int | float | None = 0, as_confounds: bool = False, progress_bar: bool = False) -> DesignMatrix
```

Concatenate design matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#page-data-design-matrix)</code> | The base design matrix. | *required*
`other` | <code>[DesignMatrix](#page-data-design-matrix) \| DataFrame \| DataFrame \| list</code> | Matrix or matrices to append. For ``axis=1`` (horizontal), also accepts a pandas or polars DataFrame (or list thereof); the new columns are treated as nuisance regressors (tracked in `confounds` on the result). For ``axis=0`` (vertical), all items must be `DesignMatrix`. | *required*
`axis` | <code>int</code> | 0 for row-wise (vertical), 1 for column-wise (horizontal). | <code>0</code>
`keep_separate` | <code>bool</code> | Whether to separate confound columns across runs (only ``axis=0``). | <code>True</code>
`unique_cols` | <code>list[str] \| None</code> | Additional columns to keep separated (supports ``*`` wildcards). | <code>None</code>
`fill_na` | <code>int, float, or None</code> | Value to fill NaN/null entries introduced by the concatenation. Pass ``None`` to preserve nulls. Default: 0. | <code>0</code>
`as_confounds` | <code>bool</code> | Only applies to ``axis=1``. When True, all columns contributed by ``other`` are tracked as nuisance regressors in the result's ``.confounds`` — so they're skipped by ``.convolve()`` and kept separate across runs in later vertical appends. Useful when ``other`` is a pre-built DesignMatrix of confounds that hasn't already marked its columns. Default: False. | <code>False</code>
`progress_bar` | <code>bool</code> | Print messages about confound separation. Default: False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#page-data-design-matrix)</code> | Concatenated design matrix.

**Raises:**

Type | Description
---- | -----------
<code>TypeError</code> | If items to append are not DesignMatrix (or, for ``axis=1``, a DesignMatrix / pandas DataFrame / polars DataFrame).
<code>ValueError</code> | If sampling frequencies do not match, axis is invalid, a non-multi base is combined with a multi-run DM, or shared columns have mismatched dtypes.
