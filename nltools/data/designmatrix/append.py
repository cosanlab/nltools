"""Concatenate DesignMatrix objects horizontally or across runs.

`append` dispatches to `append_horizontal` (add columns) or `append_vertical`
(stack runs). Vertical appends can keep confound columns separate per run by
renaming them into the reserved ``.nl_r{run}_`` namespace, so each run gets
its own intercept and drift terms.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl

from .utils import (
    RESERVED_PREFIX,
    copy_with,
    is_reserved_name,
    run_separated_name,
)

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
    # Accumulate the first dtype (and defining frame index) seen for each column
    # across ALL frames, so a mismatch between two later frames is caught even
    # when the column is absent from dfs[0].
    seen: dict[str, tuple[pl.DataType, int]] = {}
    for idx, df in enumerate(dfs):
        for col, dtype in df.schema.items():
            if col not in seen:
                seen[col] = (dtype, idx)
                continue
            first_dtype, first_idx = seen[col]
            if first_dtype != dtype:
                raise ValueError(
                    f"Column {col!r} has mismatched dtype {first_dtype} "
                    f"in dm[{first_idx}] vs {dtype} in dm[{idx}]. Cast one side "
                    f"with .with_columns(pl.col({col!r}).cast(...)) to align "
                    f"dtypes before appending."
                )


def _coerce_horizontal_input(x, sampling_freq):
    """Coerce a horizontal-append input into a DesignMatrix.

    ``append(axis=1)`` accepts DesignMatrix or Polars DataFrame. Raw-frame inputs are wrapped into a DesignMatrix whose new
    columns are tracked as nuisance (``.confounds``) — that way a subsequent
    multi-run vertical append keeps them separated per run, which matches the
    usual use of this path (motion / physio / compcor confounds).

    Args:
        x (DesignMatrix | pl.DataFrame): Input to coerce.
        sampling_freq (float | None): Base DM's sampling frequency (inherited
            by the wrapped DM).

    Returns:
        DesignMatrix: The coerced input.

    Raises:
        TypeError: If ``x`` is not a DesignMatrix or supported DataFrame.
        ValueError: If a raw frame's columns intrude on the reserved namespace.
    """
    from nltools.data.designmatrix import DesignMatrix

    if isinstance(x, DesignMatrix):
        return x
    if isinstance(x, pl.DataFrame):
        # Columns arriving as a raw frame are user-authored by definition, so
        # the reserved namespace is off limits: letting them in would make a
        # user column indistinguishable from one nltools generated.
        reserved = sorted(c for c in x.columns if is_reserved_name(c))
        if reserved:
            raise ValueError(
                f"Column names starting with {RESERVED_PREFIX!r} are reserved for "
                f"regressors nltools generates (polynomials, cosine bases, spikes, "
                f"run-separated columns): {reserved}. Rename them before appending."
            )
        # Build once, then re-wrap so we can pass `confounds=` via the
        # constructor (the public attribute is read-only).
        tmp = DesignMatrix(x, sampling_freq=sampling_freq)
        return DesignMatrix(
            tmp.data, sampling_freq=sampling_freq, confounds=list(tmp.columns)
        )
    raise TypeError(
        f"append(axis=1) expects DesignMatrix, polars DataFrame; got {type(x).__name__}"
    )


def append(
    dm: DesignMatrix,
    other,
    *,
    axis: int = 0,
    keep_separate: bool = True,
    unique_cols: list[str] | None = None,
    fill_na: int | float | None = 0,
    as_confounds: bool = False,
    progress_bar: bool = False,
) -> DesignMatrix:
    """Concatenate design matrices.

    Args:
        dm (DesignMatrix): The base design matrix.
        other (DesignMatrix | pl.DataFrame | list): Matrix or
            matrices to append. For ``axis=1`` (horizontal), also accepts a
            polars DataFrame (or list thereof); the new columns are
            treated as nuisance regressors (tracked in `confounds` on the
            result). For ``axis=0`` (vertical), all items must be `DesignMatrix`.
        axis (int): 0 for row-wise (vertical), 1 for column-wise (horizontal).
        keep_separate (bool): Whether to separate confound columns across runs
            (only ``axis=0``).
        unique_cols (list[str] | None): Additional columns to keep separated
            (supports ``*`` wildcards).
        fill_na (int, float, or None): Value to fill NaN/null entries introduced
            by the concatenation. Pass ``None`` to preserve nulls. Default: 0.
        as_confounds (bool): Only applies to ``axis=1``. When True, all columns
            contributed by ``other`` are tracked as nuisance regressors in
            the result's ``.confounds`` — so they're skipped by ``.convolve()``
            and kept separate across runs in later vertical appends. Useful
            when ``other`` is a pre-built DesignMatrix of confounds that
            hasn't already marked its columns. Default: False.
        progress_bar (bool): Print messages about confound separation. Default: False.

    Returns:
        DesignMatrix: Concatenated design matrix.

    Raises:
        TypeError: If items to append are not DesignMatrix (or, for ``axis=1``,
            a DesignMatrix / polars DataFrame).
        ValueError: If sampling frequencies do not match, axis is invalid,
            a non-multi base is combined with a multi-run DM, or shared
            columns have mismatched dtypes.
    """
    from nltools.data.designmatrix import DesignMatrix

    # Normalize to list
    to_append = [other] if not isinstance(other, list) else list(other)

    # Horizontal append additionally accepts raw DataFrames — convert them
    # to DesignMatrix first so the rest of the validation and merge path is
    # unchanged.
    if axis == 1:
        to_append = [_coerce_horizontal_input(e, dm.sampling_freq) for e in to_append]

    # Validate all are DesignMatrix with same sampling_freq
    if not all(isinstance(elem, DesignMatrix) for elem in to_append):
        raise TypeError(
            "All items to append must be DesignMatrix objects "
            "(axis=1 also accepts polars DataFrames)"
        )
    if axis not in (0, 1):
        raise ValueError("axis must be 0 (vertical) or 1 (horizontal)")

    # Only the fully default empty constructor is an untimed identity.
    def is_identity(matrix):
        return (
            matrix.shape == (0, 0)
            and matrix._n_rows is None
            and matrix.sampling_freq is None
        )

    to_append = [elem for elem in to_append if not is_identity(elem)]
    if is_identity(dm) and to_append:
        dm, *to_append = to_append
    if not to_append:
        return dm.copy()
    if not all(elem.sampling_freq == dm.sampling_freq for elem in to_append):
        raise ValueError("All Design Matrices must have the same sampling frequency!")

    if axis == 1:
        return append_horizontal(dm, to_append, fill_na, as_confounds=as_confounds)
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
            dm,
            to_append,
            keep_separate,
            unique_cols,
            fill_na,
            progress_bar=progress_bar,
        )
    raise ValueError("axis must be 0 (vertical) or 1 (horizontal)")


def append_horizontal(
    dm: DesignMatrix,
    to_append: list[DesignMatrix],
    fill_na: int | float | None,
    as_confounds: bool = False,
) -> DesignMatrix:
    """Concatenate matrices horizontally by adding columns.

    Args:
        dm (DesignMatrix): Base DesignMatrix instance.
        to_append (list[DesignMatrix]): Matrices whose columns to add.
        fill_na (int, float, or None): Value to fill NaN/null entries with.
            Pass ``None`` to preserve nulls.
        as_confounds (bool): If True, mark all columns contributed by
            ``to_append`` as nuisance/confounds in the result.

    Returns:
        DesignMatrix: New DesignMatrix with columns from all matrices.

    Raises:
        ValueError: If matrices have different row counts, share column
            names, or an appended column duplicates an existing column's
            values under a different name (a design with straight duplicate
            columns is rank deficient by construction, so the model over it
            is not computable — refuse at assembly time rather than decide
            on the user's behalf which copy to keep).
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

    # Straight duplicate VALUES under different names are just as degenerate
    # as duplicate names: the design becomes rank deficient by construction.
    # Only duplication introduced by this append is checked — the base's
    # pre-existing state is the user's business, not this operation's.

    # Heights were validated above, so 'horizontal_extend' (the stable name
    # polars >= 1.42.1 gives the classic horizontal concat) never pads.
    dfs_to_stack = [dm.data] + [elem.data for elem in to_append]
    new_df = pl.concat(dfs_to_stack, how="horizontal_extend")

    # Fill NaN if requested
    if fill_na is not None:
        new_df = new_df.fill_null(fill_na).fill_nan(fill_na)
    _check_duplicate_values(new_df, dm.columns)

    # Merge confounds + convolved metadata across all matrices, dedup in order.
    confound_lists = [dm.confounds, *(e.confounds for e in to_append)]
    if as_confounds:
        # Promote all columns from to_append to nuisance/confounds
        confound_lists.extend(e.columns for e in to_append)
    all_confounds = _merge_ordered(confound_lists)
    all_convolved = _merge_ordered([dm.convolved, *(e.convolved for e in to_append)])

    return copy_with(dm, new_df, confounds=all_confounds, convolved=all_convolved)


def _check_duplicate_values(frame: pl.DataFrame, base_columns: list[str]) -> None:
    """Reject new equal numeric columns after filling, preserving exact values."""
    null = object()
    nan = object()
    seen = {}
    for col in frame.columns:
        series = frame[col]
        if not series.dtype.is_numeric() and series.dtype != pl.Null:
            continue
        # Python numeric equality/hash compares integer and float values exactly,
        # including integers beyond Float64 precision; null and NaN stay distinct.
        key = tuple(
            null
            if value is None
            else nan
            if isinstance(value, float) and math.isnan(value)
            else value
            for value in series
        )
        if key in seen and col not in base_columns:
            raise ValueError(
                f"Column {col!r} duplicates column {seen[key]!r}: identical values under different names."
            )
        seen.setdefault(key, col)


def _stack_frames(frames: list[pl.DataFrame], heights: list[int]) -> pl.DataFrame:
    """Include recorded rows from column-less inputs in a vertical stack."""
    schema = {}
    for frame in frames:
        schema.update(frame.schema)
    if not schema:
        return pl.DataFrame()
    populated = [
        frame
        if frame.width
        else pl.DataFrame(
            [
                pl.Series(name, [None] * height, dtype=dtype)
                for name, dtype in schema.items()
            ]
        )
        for frame, height in zip(frames, heights)
    ]
    return pl.concat(populated, how="diagonal")


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
    *,
    progress_bar: bool,
) -> DesignMatrix:
    """Concatenate matrices vertically with optional confound separation.

    Args:
        dm (DesignMatrix): Base DesignMatrix instance.
        to_append (list[DesignMatrix]): Matrices to stack below `dm`.
        keep_separate (bool): Whether to separate confound columns across runs.
        unique_cols (list[str] | None): Additional columns to keep separated
            (supports ``*`` wildcards).
        fill_na (int, float, or None): Value to fill NaN/null entries with.
            Pass ``None`` to preserve nulls.
        progress_bar (bool): Print messages about confound separation.

    Returns:
        DesignMatrix: New DesignMatrix with rows from all matrices.
    """
    all_dms = [dm, *to_append]

    # Simple case: keep_separate=False - just stack rows
    if not keep_separate:
        dfs_to_stack = [d.data for d in all_dms]
        _check_dtype_compatibility(dfs_to_stack)
        new_df = _stack_frames(dfs_to_stack, [d.shape[0] for d in all_dms])

        # Fill NaN if requested
        if fill_na is not None:
            new_df = new_df.fill_null(fill_na).fill_nan(fill_na)

        # Merge confounds + convolved across matrices
        all_confounds = _merge_ordered([d.confounds for d in all_dms])
        all_convolved = _merge_ordered([d.convolved for d in all_dms])

        return copy_with(
            dm,
            new_df,
            confounds=all_confounds,
            convolved=all_convolved,
            n_rows=sum(d.shape[0] for d in all_dms),
        )

    # Complex case: keep_separate=True - separate confound columns across runs
    return append_vertical_with_separation(
        dm, to_append, unique_cols, fill_na, progress_bar=progress_bar
    )


def match_column_pattern(columns: list[str], pattern: str) -> list[str]:
    """Match columns against a pattern with wildcard support.

    Args:
        columns (list[str]): Column names to search.
        pattern (str): Pattern to match, with ``*`` as a leading or trailing
            wildcard: ``'motion*'`` matches ``motion_x`` and ``motion_y``,
            ``'*_motion'`` matches ``x_motion`` and ``y_motion``, and
            ``'exact'`` matches only ``exact``.

    Returns:
        list[str]: Column names matching the pattern.
    """
    if pattern.endswith("*"):
        prefix = pattern[:-1]
        return [c for c in columns if c.startswith(prefix)]
    if pattern.startswith("*"):
        suffix = pattern[1:]
        return [c for c in columns if c.endswith(suffix)]
    return [c for c in columns if c == pattern]


def identify_columns_to_separate(
    dm: DesignMatrix,
    all_dms: list[DesignMatrix],
    unique_cols: list[str] | None,
) -> set:
    """Identify columns that need run-specific separation.

    Args:
        dm (DesignMatrix): The base design matrix (used for context only).
        all_dms (list[DesignMatrix]): All matrices being concatenated.
        unique_cols (list[str] | None): User-specified columns to separate
            (supports ``*`` wildcards).

    Returns:
        set: Column names that should be separated with run prefixes.
    """
    cols_to_sep = set()

    # Add confound columns from non-multi DMs only
    # (Multi-run DMs already have separated confounds)
    for d in all_dms:
        if d.confounds and not d.multi:
            cols_to_sep.update(d.confounds)

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
    *,
    progress_bar: bool,
) -> DesignMatrix:
    """Concatenate vertically with automatic confound separation.

    Creates run-specific columns (e.g. ``.nl_r0_poly_0``, ``.nl_r1_poly_0``)
    that are active only in their respective runs (sparse representation).

    Args:
        dm (DesignMatrix): Base DesignMatrix instance.
        to_append (list[DesignMatrix]): Matrices to stack below `dm`.
        unique_cols (list[str] | None): Additional columns to keep separated
            (supports ``*`` wildcards).
        fill_na (int, float, or None): Value to fill NaN/null entries with.
            Pass ``None`` to preserve nulls.
        progress_bar (bool): Print messages about confound separation.

    Returns:
        DesignMatrix: Concatenated DesignMatrix with run-separated confound columns
            and multi=True.
    """
    all_dms = [dm, *to_append]
    cols_to_sep = identify_columns_to_separate(dm, all_dms, unique_cols)
    if progress_bar and cols_to_sep:
        print(f"Separating columns across runs: {sorted(cols_to_sep)}")

    processed_dfs = []
    all_new_confounds: list[str] = []
    all_new_convolved: list[str] = []
    next_run = 0
    for d in all_dms:
        rename_map = {}
        if d.shape[0] > 0:
            if d.multi:
                # Preassembled inputs already carry their accepted identities.
                next_run = max(next_run, d._run_count)
            else:
                rename_map = {
                    col: run_separated_name(next_run, col)
                    for col in d.columns
                    if col in cols_to_sep
                }
                next_run += 1
            all_new_confounds.extend(rename_map.get(c, c) for c in d.confounds)
            all_new_convolved.extend(rename_map.get(c, c) for c in d.convolved)
        # Even a zero-row input contributes its schema and dtype obligations.
        processed_dfs.append(d.data.rename(rename_map) if rename_map else d.data)

    _check_dtype_compatibility(processed_dfs)
    result_df = _stack_frames(processed_dfs, [d.shape[0] for d in all_dms])
    if fill_na is not None:
        result_df = result_df.fill_null(fill_na).fill_nan(fill_na)
    return copy_with(
        dm,
        result_df,
        confounds=_merge_ordered([all_new_confounds]),
        convolved=_merge_ordered([all_new_convolved]),
        multi=True,
        n_rows=sum(d.shape[0] for d in all_dms),
        run_count=next_run,
    )
