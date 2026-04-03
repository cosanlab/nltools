"""
Standalone functions for DesignMatrix concatenation operations.

These functions implement the append/concatenation logic extracted from
DesignMatrix methods, following the "functional core" pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

import polars as pl

if TYPE_CHECKING:
    from nltools.data.design_matrix import DesignMatrix


def append(
    dm: DesignMatrix,
    other: Union[DesignMatrix, List[DesignMatrix]],
    axis: int = 0,
    keep_separate: bool = True,
    unique_cols: Optional[List[str]] = None,
    fill_na: Union[int, float] = 0,
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
        fill_na (int or float): Value to fill NaN values during vertical
            concatenation. Default: 0.
        verbose (bool): Print messages about polynomial separation. Default: False.

    Returns:
        DesignMatrix: Concatenated design matrix.

    Raises:
        TypeError: If items to append are not DesignMatrix instances.
        ValueError: If sampling frequencies do not match or axis is invalid.
    """
    from nltools.data.design_matrix import DesignMatrix

    # Normalize to list
    to_append = [other] if not isinstance(other, list) else other

    # Validate all are DesignMatrix with same sampling_freq
    if not all(isinstance(elem, DesignMatrix) for elem in to_append):
        raise TypeError("All items to append must be DesignMatrix objects")
    if not all(elem.sampling_freq == dm.sampling_freq for elem in to_append):
        raise ValueError("All Design Matrices must have the same sampling frequency!")

    if axis == 1:
        return append_horizontal(dm, to_append, fill_na)
    elif axis == 0:
        return append_vertical(
            dm, to_append, keep_separate, unique_cols, fill_na, verbose
        )
    else:
        raise ValueError("axis must be 0 (vertical) or 1 (horizontal)")


def append_horizontal(
    dm: DesignMatrix,
    to_append: List[DesignMatrix],
    fill_na: Union[int, float],
) -> DesignMatrix:
    """
    Horizontal concatenation (axis=1) - add columns from other matrices.

    Args:
        dm: Base DesignMatrix instance.
        to_append (list of DesignMatrix): Matrices whose columns to add.
        fill_na (int or float): Value to fill NaN/null entries with.

    Returns:
        DesignMatrix: New DesignMatrix with columns from all matrices.

    Raises:
        ValueError: If matrices have different row counts.
    """
    # Check all have same number of rows
    if not all(elem.shape[0] == dm.shape[0] for elem in to_append):
        raise ValueError("All Design Matrices must have the same number of rows!")

    # Warn about duplicate column names
    all_columns = set(dm.columns)
    for elem in to_append:
        if not all_columns.isdisjoint(elem.columns):
            print("Duplicate column names detected. Will be repeated.")
        all_columns.update(elem.columns)

    # Use Polars hstack to concatenate DataFrames horizontally
    dfs_to_stack = [dm._df] + [elem._df for elem in to_append]
    new_df = pl.concat(dfs_to_stack, how="horizontal")

    # Fill NaN if requested
    if fill_na is not None:
        new_df = new_df.fill_null(fill_na)

    # Combine polys metadata from all matrices
    all_polys = dm.polys.copy() if dm.polys else []
    for elem in to_append:
        if elem.polys:
            all_polys.extend(elem.polys)

    return dm._copy_with(new_df, polys=all_polys)


def append_vertical(
    dm: DesignMatrix,
    to_append: List[DesignMatrix],
    keep_separate: bool,
    unique_cols: Optional[List[str]],
    fill_na: Union[int, float],
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
        fill_na (int or float): Value to fill NaN/null entries with.
        verbose (bool): Print messages about polynomial separation.

    Returns:
        DesignMatrix: New DesignMatrix with rows from all matrices.
    """
    # Simple case: keep_separate=False - just stack rows
    if not keep_separate:
        dfs_to_stack = [dm._df] + [elem._df for elem in to_append]
        new_df = pl.concat(dfs_to_stack, how="vertical")

        # Fill NaN if requested
        if fill_na is not None:
            new_df = new_df.fill_null(fill_na)

        # Combine polys metadata (no separation)
        all_polys = dm.polys.copy() if dm.polys else []
        for elem in to_append:
            if elem.polys:
                # Add new polys we don't already have
                for p in elem.polys:
                    if p not in all_polys:
                        all_polys.append(p)

        return dm._copy_with(new_df, polys=all_polys)

    # Complex case: keep_separate=True - separate polynomial columns across runs
    return append_vertical_with_separation(dm, to_append, unique_cols, fill_na, verbose)


def match_column_pattern(columns: List[str], pattern: str) -> List[str]:
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
    elif pattern.startswith("*"):
        suffix = pattern[1:]
        return [c for c in columns if c.endswith(suffix)]
    else:
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
    all_dms: List[DesignMatrix],
    unique_cols: Optional[List[str]],
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
    to_append: List[DesignMatrix],
    unique_cols: Optional[List[str]],
    fill_na: Union[int, float],
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
        fill_na (int or float): Value to fill NaN/null entries with.
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
        all_dms = [dm] + to_append
        cols_to_sep = identify_columns_to_separate(dm, all_dms, unique_cols)

        if verbose and cols_to_sep:
            print(f"Separating columns across runs: {sorted(cols_to_sep)}")

        processed_dfs = []
        all_new_polys = []

        for i, d in enumerate(all_dms):
            # Build rename mapping for separated columns
            rename_map = {col: f"{i}_{col}" for col in d.columns if col in cols_to_sep}

            # Rename and collect
            processed_df = d._df.rename(rename_map) if rename_map else d._df
            processed_dfs.append(processed_df)

            # Track renamed polys
            for poly in d.polys:
                if poly in rename_map:
                    all_new_polys.append(rename_map[poly])

    else:
        # Case 2: Appending to existing multi-run DM
        start_idx = get_starting_run_idx(dm)
        cols_to_sep = identify_columns_to_separate(dm, to_append, unique_cols)

        if verbose and cols_to_sep:
            print(f"Separating columns across runs: {sorted(cols_to_sep)}")

        # Keep self's DataFrame unchanged
        processed_dfs = [dm._df]
        all_new_polys = dm.polys.copy() if dm.polys else []

        # Process only the DMs being appended
        for i, d in enumerate(to_append):
            run_idx = start_idx + i

            # Build rename mapping for separated columns
            rename_map = {
                col: f"{run_idx}_{col}" for col in d.columns if col in cols_to_sep
            }

            # Rename and collect
            processed_df = d._df.rename(rename_map) if rename_map else d._df
            processed_dfs.append(processed_df)

            # Track renamed polys
            for poly in d.polys:
                if poly in rename_map:
                    all_new_polys.append(rename_map[poly])

    # Concatenate with diagonal (auto-fills missing columns with null)
    result_df = pl.concat(processed_dfs, how="diagonal")

    # Fill nulls with fill_na value (creates sparse separation)
    result_df = result_df.fill_null(fill_na)

    # Return with updated metadata
    return dm._copy_with(result_df, polys=all_new_polys, multi=True)
