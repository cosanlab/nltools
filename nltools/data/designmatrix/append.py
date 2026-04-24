"""
Standalone functions for DesignMatrix concatenation operations.

These functions implement the append/concatenation logic extracted from
DesignMatrix methods, following the "functional core" pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from .utils import copy_with

if TYPE_CHECKING:
    from nltools.data.designmatrix import DesignMatrix


def _check_dtype_compatibility(dfs: list[pl.DataFrame]) -> None:
    """Raise a clear ValueError if shared columns across frames have mismatched dtypes.

    Polars' native error (``SchemaError: type Float64 is incompatible with
    expected type Int64``) doesn't name the offending column, so we check
    ahead of time and produce an actionable message.
    """
    if len(dfs) < 2:
        return
    base_schema = dict(dfs[0].schema)
    for idx, df in enumerate(dfs[1:], start=1):
        for col, dtype in df.schema.items():
            if col in base_schema and base_schema[col] != dtype:
                raise ValueError(
                    f"Column {col!r} has mismatched dtype {base_schema[col]} "
                    f"in dm[0] vs {dtype} in dm[{idx}]. Cast one side with "
                    f".with_columns(pl.col({col!r}).cast(...)) to align dtypes "
                    f"before appending."
                )


def append(
    dm: DesignMatrix,
    other: DesignMatrix | list[DesignMatrix],
    axis: int = 0,
    keep_separate: bool = True,
    unique_cols: list[str] | None = None,
    fill_na: int | float | None = 0,
    verbose: bool = False,
) -> DesignMatrix:
    """
    Concatenate design matrices.

    Args:
        dm (DesignMatrix): The base design matrix.
        other (DesignMatrix or list of DesignMatrix): Design matrix/matrices to append.
        axis (int): 0 for row-wise (vertical), 1 for column-wise (horizontal).
        keep_separate (bool): Whether to separate polynomial columns across runs (only axis=0).
        unique_cols (list of str, optional): Additional columns to keep separated (supports wildcards).
        fill_na (int, float, or None): Value to fill NaN/null entries introduced
            by the concatenation. Pass ``None`` to preserve nulls. Default: 0.
        verbose (bool): Print messages about polynomial separation. Default: False.

    Returns:
        DesignMatrix: Concatenated design matrix.

    Raises:
        TypeError: If items to append are not DesignMatrix instances.
        ValueError: If sampling frequencies do not match, axis is invalid,
            a non-multi base is combined with a multi-run DM, or shared
            columns have mismatched dtypes.
    """
    from nltools.data.designmatrix import DesignMatrix

    # Normalize to list
    to_append = [other] if not isinstance(other, list) else other

    # Validate all are DesignMatrix with same sampling_freq
    if not all(isinstance(elem, DesignMatrix) for elem in to_append):
        raise TypeError("All items to append must be DesignMatrix objects")
    if not all(elem.sampling_freq == dm.sampling_freq for elem in to_append):
        raise ValueError("All Design Matrices must have the same sampling frequency!")

    if axis == 1:
        return append_horizontal(dm, to_append, fill_na)
    if axis == 0:
        # Refuse the silent-collision case: a non-multi base with any multi
        # DM in to_append would re-index the base as run 0 and collide with
        # the appended DM's existing 0_* columns.
        if not dm.multi and any(elem.multi for elem in to_append):
            raise ValueError(
                "Cannot append a multi-run DesignMatrix to a non-multi base: "
                "the base would be re-indexed as run 0 and collide with the "
                "appended matrix's existing run-prefixed columns. Either start "
                "from the multi-run DM and append the single-run DM to it, or "
                "rebuild the multi-run DM from its constituent single-run DMs "
                "in the desired order."
            )
        return append_vertical(
            dm, to_append, keep_separate, unique_cols, fill_na, verbose
        )
    raise ValueError("axis must be 0 (vertical) or 1 (horizontal)")


def append_horizontal(
    dm: DesignMatrix,
    to_append: list[DesignMatrix],
    fill_na: int | float | None,
) -> DesignMatrix:
    """
    Horizontal concatenation (axis=1) - add columns from other matrices.

    Args:
        dm: Base DesignMatrix instance.
        to_append (list of DesignMatrix): Matrices whose columns to add.
        fill_na (int, float, or None): Value to fill NaN/null entries with.
            Pass ``None`` to preserve nulls.

    Returns:
        DesignMatrix: New DesignMatrix with columns from all matrices.

    Raises:
        ValueError: If matrices have different row counts.
    """
    # Check all have same number of rows
    if not all(elem.shape[0] == dm.shape[0] for elem in to_append):
        raise ValueError("All Design Matrices must have the same number of rows!")

    # Polars refuses duplicate column names on horizontal concat. Detect up
    # front and surface an actionable error instead of the cryptic polars one.
    all_columns = set(dm.columns)
    for elem in to_append:
        dupes = all_columns.intersection(elem.columns)
        if dupes:
            raise ValueError(
                f"Duplicate column names on horizontal append: {sorted(dupes)}. "
                f"Rename the conflicting columns on one side before appending."
            )
        all_columns.update(elem.columns)

    # Use Polars hstack to concatenate DataFrames horizontally
    dfs_to_stack = [dm.data] + [elem.data for elem in to_append]
    new_df = pl.concat(dfs_to_stack, how="horizontal")

    # Fill NaN if requested
    if fill_na is not None:
        new_df = new_df.fill_null(fill_na)

    # Merge polys + convolved metadata across all matrices, dedup in order.
    all_polys = _merge_ordered([dm.polys, *(e.polys for e in to_append)])
    all_convolved = _merge_ordered([dm.convolved, *(e.convolved for e in to_append)])

    return copy_with(dm, new_df, polys=all_polys, convolved=all_convolved)


def _merge_ordered(lists: list[list[str]]) -> list[str]:
    """Concatenate lists, preserving first-seen order, skipping duplicates."""
    seen: set[str] = set()
    out: list[str] = []
    for lst in lists:
        for item in lst:
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def append_vertical(
    dm: DesignMatrix,
    to_append: list[DesignMatrix],
    keep_separate: bool,
    unique_cols: list[str] | None,
    fill_na: int | float | None,
    verbose: bool,
) -> DesignMatrix:
    """
    Vertical concatenation (axis=0) - stack rows, with optional polynomial separation.

    Args:
        dm: Base DesignMatrix instance.
        to_append (list of DesignMatrix): Matrices to stack below dm.
        keep_separate (bool): Whether to separate polynomial columns across runs.
        unique_cols (list of str, optional): Additional columns to keep separated
            (supports wildcards).
        fill_na (int, float, or None): Value to fill NaN/null entries with.
            Pass ``None`` to preserve nulls.
        verbose (bool): Print messages about polynomial separation.

    Returns:
        DesignMatrix: New DesignMatrix with rows from all matrices.
    """
    all_dms = [dm, *to_append]

    # Simple case: keep_separate=False - just stack rows
    if not keep_separate:
        dfs_to_stack = [d.data for d in all_dms]
        _check_dtype_compatibility(dfs_to_stack)
        new_df = pl.concat(dfs_to_stack, how="diagonal")

        # Fill NaN if requested
        if fill_na is not None:
            new_df = new_df.fill_null(fill_na)

        # Merge polys + convolved across matrices
        all_polys = _merge_ordered([d.polys for d in all_dms])
        all_convolved = _merge_ordered([d.convolved for d in all_dms])

        return copy_with(dm, new_df, polys=all_polys, convolved=all_convolved)

    # Complex case: keep_separate=True - separate polynomial columns across runs
    return append_vertical_with_separation(dm, to_append, unique_cols, fill_na, verbose)


def match_column_pattern(columns: list[str], pattern: str) -> list[str]:
    """
    Match columns against pattern with wildcard support.

    Args:
        columns (list of str): Column names to search.
        pattern (str): Pattern to match (supports '*' as wildcard).
            - 'motion*' matches motion_x, motion_y
            - '*_motion' matches x_motion, y_motion
            - 'exact' matches only 'exact'

    Returns:
        list of str: Column names matching the pattern.
    """
    if pattern.endswith("*"):
        prefix = pattern[:-1]
        return [c for c in columns if c.startswith(prefix)]
    if pattern.startswith("*"):
        suffix = pattern[1:]
        return [c for c in columns if c.endswith(suffix)]
    return [c for c in columns if c == pattern]


def get_starting_run_idx(dm: DesignMatrix) -> int:
    """
    Determine next run index for multi-run appending.

    Args:
        dm: DesignMatrix instance to inspect.

    Returns:
        int: Next run index (0 if not multi-run, max_existing_idx + 1 otherwise).
    """
    if not dm.multi:
        return 0

    # Find max run index from column names like "0_poly_0", "1_motion_x"
    max_idx = -1
    for col in dm.columns:
        if "_" in col:
            first_part = col.split("_")[0]
            if first_part.isdigit():
                idx = int(first_part)
                max_idx = max(max_idx, idx)

    return max_idx + 1 if max_idx >= 0 else 0


def identify_columns_to_separate(
    dm: DesignMatrix,
    all_dms: list[DesignMatrix],
    unique_cols: list[str] | None,
) -> set:
    """
    Identify which columns need run-specific separation.

    Args:
        dm (DesignMatrix): The base design matrix (used for context only).
        all_dms (list of DesignMatrix): All matrices being concatenated.
        unique_cols (list of str, optional): User-specified columns to separate (supports wildcards).

    Returns:
        set: Column names that should be separated with run prefixes.
    """
    cols_to_sep = set()

    # Add polynomial columns from non-multi DMs only
    # (Multi-run DMs already have separated polynomials)
    for d in all_dms:
        if d.polys and not d.multi:
            cols_to_sep.update(d.polys)

    # Add unique_cols with wildcard matching
    if unique_cols:
        # Collect all column names across all DMs
        all_column_names = set()
        for d in all_dms:
            all_column_names.update(d.columns)

        # Match each pattern
        for pattern in unique_cols:
            matched = match_column_pattern(list(all_column_names), pattern)
            cols_to_sep.update(matched)

    return cols_to_sep


def append_vertical_with_separation(
    dm: DesignMatrix,
    to_append: list[DesignMatrix],
    unique_cols: list[str] | None,
    fill_na: int | float | None,
    verbose: bool,
) -> DesignMatrix:
    """
    Vertical concatenation with automatic polynomial separation.

    Creates run-specific columns (e.g., 0_poly_0, 1_poly_0) that are
    active only in their respective runs (sparse representation).

    Args:
        dm: Base DesignMatrix instance.
        to_append (list of DesignMatrix): Matrices to stack below dm.
        unique_cols (list of str, optional): Additional columns to keep separated
            (supports wildcards).
        fill_na (int, float, or None): Value to fill NaN/null entries with.
            Pass ``None`` to preserve nulls.
        verbose (bool): Print messages about polynomial separation.

    Returns:
        DesignMatrix: Concatenated DesignMatrix with run-separated polynomial columns
            and multi=True.
    """
    # Handle two cases differently:
    # 1. Self is NOT multi: process all DMs with sequential numbering
    # 2. Self IS multi: keep self unchanged, only process to_append DMs

    if not dm.multi:
        # Case 1: Standard multi-run creation
        all_dms = [dm, *to_append]
        cols_to_sep = identify_columns_to_separate(dm, all_dms, unique_cols)

        if verbose and cols_to_sep:
            print(f"Separating columns across runs: {sorted(cols_to_sep)}")

        processed_dfs = []
        all_new_polys: list[str] = []
        all_new_convolved: list[str] = []

        for i, d in enumerate(all_dms):
            rename_map = {col: f"{i}_{col}" for col in d.columns if col in cols_to_sep}
            processed_df = d.data.rename(rename_map) if rename_map else d.data
            processed_dfs.append(processed_df)

            for poly in d.polys:
                all_new_polys.append(rename_map.get(poly, poly))
            for conv in d.convolved:
                renamed = rename_map.get(conv, conv)
                if renamed not in all_new_convolved:
                    all_new_convolved.append(renamed)

    else:
        # Case 2: Appending to existing multi-run DM
        start_idx = get_starting_run_idx(dm)
        cols_to_sep = identify_columns_to_separate(dm, to_append, unique_cols)

        if verbose and cols_to_sep:
            print(f"Separating columns across runs: {sorted(cols_to_sep)}")

        processed_dfs = [dm.data]
        all_new_polys = list(dm.polys)
        all_new_convolved = list(dm.convolved)

        for i, d in enumerate(to_append):
            run_idx = start_idx + i
            rename_map = {
                col: f"{run_idx}_{col}" for col in d.columns if col in cols_to_sep
            }
            processed_df = d.data.rename(rename_map) if rename_map else d.data
            processed_dfs.append(processed_df)

            for poly in d.polys:
                all_new_polys.append(rename_map.get(poly, poly))
            for conv in d.convolved:
                renamed = rename_map.get(conv, conv)
                if renamed not in all_new_convolved:
                    all_new_convolved.append(renamed)

    # Validate dtype compatibility for any overlapping column names after renames
    _check_dtype_compatibility(processed_dfs)

    # Concatenate with diagonal (auto-fills missing columns with null)
    result_df = pl.concat(processed_dfs, how="diagonal")

    # Fill nulls with fill_na value unless caller asked to preserve nulls
    if fill_na is not None:
        result_df = result_df.fill_null(fill_na)

    return copy_with(
        dm,
        result_df,
        polys=all_new_polys,
        convolved=all_new_convolved,
        multi=True,
    )
