## `append`

Standalone functions for DesignMatrix concatenation operations.

These functions implement the append/concatenation logic extracted from
DesignMatrix methods, following the "functional core" pattern.

**Methods:**

Name | Description
---- | -----------
[`append`](#append) | Concatenate design matrices.
[`append_horizontal`](#append_horizontal) | Horizontal concatenation (axis=1) - add columns from other matrices.
[`append_vertical`](#append_vertical) | Vertical concatenation (axis=0) - stack rows, with optional polynomial separation.
[`append_vertical_with_separation`](#append_vertical_with_separation) | Vertical concatenation with automatic polynomial separation.
[`get_starting_run_idx`](#get_starting_run_idx) | Determine next run index for multi-run appending.
[`identify_columns_to_separate`](#identify_columns_to_separate) | Identify which columns need run-specific separation.
[`match_column_pattern`](#match_column_pattern) | Match columns against pattern with wildcard support.



### Classes

### Methods

#### `append`

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

#### `append_horizontal`

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

#### `append_vertical`

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

#### `append_vertical_with_separation`

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

#### `match_column_pattern`

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

