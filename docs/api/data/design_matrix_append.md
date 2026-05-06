## `append`

Standalone functions for DesignMatrix concatenation operations.

These functions implement the append/concatenation logic extracted from
DesignMatrix methods, following the "functional core" pattern.

**Methods:**

Name | Description
---- | -----------
[`append`](#append) | Concatenate design matrices.
[`append_horizontal`](#append_horizontal) | Horizontal concatenation (axis=1) - add columns from other matrices.
[`append_vertical`](#append_vertical) | Vertical concatenation (axis=0) - stack rows, with optional confound separation.
[`append_vertical_with_separation`](#append_vertical_with_separation) | Vertical concatenation with automatic confound separation.
[`get_starting_run_idx`](#get_starting_run_idx) | Determine next run index for multi-run appending.
[`identify_columns_to_separate`](#identify_columns_to_separate) | Identify which columns need run-specific separation.
[`match_column_pattern`](#match_column_pattern) | Match columns against pattern with wildcard support.



### Classes

### Methods

#### `append`

```python
append(dm: DesignMatrix, other: DesignMatrix, axis: int = 0, keep_separate: bool = True, unique_cols: list[str] | None = None, fill_na: int | float | None = 0, as_confounds: bool = False, verbose: bool = False) -> DesignMatrix
```

Concatenate design matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | The base design matrix. | *required*
`other` | <code>DesignMatrix, DataFrame, or list</code> | Matrix/matrices to append. For ``axis=1`` (horizontal), also accepts a pandas or polars DataFrame (or list thereof); the new columns are treated as nuisance regressors (tracked in ``.confounds`` on the result). For ``axis=0`` (vertical), all items must be ``DesignMatrix``. | *required*
`axis` | <code>[int](#int)</code> | 0 for row-wise (vertical), 1 for column-wise (horizontal). | <code>0</code>
`keep_separate` | <code>[bool](#bool)</code> | Whether to separate confound columns across runs (only axis=0). | <code>True</code>
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | <code>None</code>
`fill_na` | <code>int, float, or None</code> | Value to fill NaN/null entries introduced by the concatenation. Pass ``None`` to preserve nulls. Default: 0. | <code>0</code>
`as_confounds` | <code>[bool](#bool)</code> | Only applies to ``axis=1``. When True, all columns contributed by ``other`` are tracked as nuisance regressors in the result's ``.confounds`` — so they're skipped by ``.convolve()`` and kept separate across runs in later vertical appends. Useful when ``other`` is a pre-built DesignMatrix of confounds that hasn't already marked its columns. Default: False. | <code>False</code>
`verbose` | <code>[bool](#bool)</code> | Print messages about confound separation. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Concatenated design matrix.

#### `append_horizontal`

```python
append_horizontal(dm: DesignMatrix, to_append: list[DesignMatrix], fill_na: int | float | None, as_confounds: bool = False) -> DesignMatrix
```

Horizontal concatenation (axis=1) - add columns from other matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Base DesignMatrix instance. | *required*
`to_append` | <code>list of DesignMatrix</code> | Matrices whose columns to add. | *required*
`fill_na` | <code>int, float, or None</code> | Value to fill NaN/null entries with. Pass ``None`` to preserve nulls. | *required*
`as_confounds` | <code>[bool](#bool)</code> | If True, mark all columns contributed by ``to_append`` as nuisance/confounds in the result. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with columns from all matrices.

#### `append_vertical`

```python
append_vertical(dm: DesignMatrix, to_append: list[DesignMatrix], keep_separate: bool, unique_cols: list[str] | None, fill_na: int | float | None, verbose: bool) -> DesignMatrix
```

Vertical concatenation (axis=0) - stack rows, with optional confound separation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Base DesignMatrix instance. | *required*
`to_append` | <code>list of DesignMatrix</code> | Matrices to stack below dm. | *required*
`keep_separate` | <code>[bool](#bool)</code> | Whether to separate confound columns across runs. | *required*
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | *required*
`fill_na` | <code>int, float, or None</code> | Value to fill NaN/null entries with. Pass ``None`` to preserve nulls. | *required*
`verbose` | <code>[bool](#bool)</code> | Print messages about confound separation. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | New DesignMatrix with rows from all matrices.

#### `append_vertical_with_separation`

```python
append_vertical_with_separation(dm: DesignMatrix, to_append: list[DesignMatrix], unique_cols: list[str] | None, fill_na: int | float | None, verbose: bool) -> DesignMatrix
```

Vertical concatenation with automatic confound separation.

Creates run-specific columns (e.g., 0_poly_0, 1_poly_0) that are
active only in their respective runs (sparse representation).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dm` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Base DesignMatrix instance. | *required*
`to_append` | <code>list of DesignMatrix</code> | Matrices to stack below dm. | *required*
`unique_cols` | <code>list of str</code> | Additional columns to keep separated (supports wildcards). | *required*
`fill_na` | <code>int, float, or None</code> | Value to fill NaN/null entries with. Pass ``None`` to preserve nulls. | *required*
`verbose` | <code>[bool](#bool)</code> | Print messages about confound separation. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` | <code>[DesignMatrix](#nltools.data.designmatrix.DesignMatrix)</code> | Concatenated DesignMatrix with run-separated confound columns and multi=True.

#### `get_starting_run_idx`

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

#### `identify_columns_to_separate`

```python
identify_columns_to_separate(dm: DesignMatrix, all_dms: list[DesignMatrix], unique_cols: list[str] | None) -> set
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

#### `match_column_pattern`

```python
match_column_pattern(columns: list[str], pattern: str) -> list[str]
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
<code>[list](#list)[[str](#str)]</code> | list of str: Column names matching the pattern.

