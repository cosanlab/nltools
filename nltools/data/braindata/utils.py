"""Shared helpers for BrainData submodules.

These are internal utilities used by the facade and submodules — not part of the
public API.
"""

import gc
import os
from contextlib import contextmanager

import numpy as np
from numpy.typing import ArrayLike

from ..ownership import _copy_graph


@contextmanager
def _coalesced_gc():
    """Collapse nilearn's forced per-copy `gc.collect()` calls into one per operation.

    nilearn runs a full `gc.collect()` after every masked-array copy it makes; a
    masking-heavy operation — a GLM fit that re-validates the same mask and
    builds several result maps — fires dozens. With torch/nilearn/sklearn
    resident each sweep costs ~0.1s, so the storm dominates the wall-clock of
    otherwise-trivial numerical work.

    This no-ops the interim collects and runs a single real collect on exit,
    so peak memory stays bounded to one operation's worth of cyclic garbage
    (nilearn's collect is a peak-memory optimization, not a correctness
    requirement — suppressing it only defers reclamation). Opt out with
    `NLTOOLS_NO_GC_COALESCE=1`.

    Because `@contextmanager` results double as decorators, this can also be
    used as `@_coalesced_gc()` on an operation-boundary method.

    Nesting is safe: each frame restores whatever it saved, so only the
    outermost frame restores the real `gc.collect` and runs the final sweep;
    inner frames' exit-time collect is a no-op.

    Caveat: this swaps a process-global builtin. It is safe under the default
    loky (process) worker backend — each worker has its own `gc`. Under a
    *threading* backend there is a brief window where a concurrent thread sees
    the no-op collect; `NLTOOLS_NO_GC_COALESCE=1` is the escape hatch there.
    """
    if os.environ.get("NLTOOLS_NO_GC_COALESCE"):
        yield
        return
    saved = gc.collect  # may already be the no-op if we're nested
    gc.collect = lambda *a, **k: 0
    try:
        yield
    finally:
        gc.collect = saved  # only the outermost frame restores the real collect
        gc.collect()  # no-op if still nested; one real sweep at the top


def _resolve_threshold(value: float | str | None, data: ArrayLike) -> float | None:
    """Resolve a threshold spec — a number or a percentile string — to a float.

    The single source of truth for what `"98%"` means across the library
    (`BrainData.threshold` and `BrainData.iplot` both route through it).
    Numbers and None pass through unchanged. A percentile string is resolved
    against the **finite nonzero** values of `data`: on a masked stat map
    most voxels are exactly zero (absence of data), and including them drags
    every percentile toward zero.

    Args:
        value: A number (returned as-is), None (returned as-is), or a string
            like `"98%"`.
        data: Array-like the percentile is computed over. Callers choose the
            frame of reference — e.g. `iplot` passes magnitudes
            (`np.abs(data)`) because its window is a magnitude window, while
            `threshold` passes signed values.

    Returns:
        float | None: The resolved threshold.

    Raises:
        ValueError: If `value` is a string without a trailing `%`.
    """
    if value is None or not isinstance(value, str):
        return value
    if not value.endswith("%"):
        raise ValueError(
            f"string threshold must be a percentile like '98%', got {value!r}"
        )
    pct = float(value[:-1])
    vals = np.asarray(data, dtype=float).ravel()
    vals = vals[np.isfinite(vals)]
    vals = vals[vals != 0]
    if vals.size == 0:
        return 0.0
    return float(np.percentile(vals, pct))


def _is_default(value, default):
    """Report whether a `fit` or `predict` option still holds its signature default.

    The check rejects non-default *values*, not the act of passing a keyword:
    an option explicitly given its own default is indistinguishable from an
    untouched one and is treated as untouched. Array-like options
    (`ridge_alpha`, `ridge_dirichlet_concentration`) make a bare `!=` return an
    array, so equality is compared elementwise, and a sequence given as a list
    matches a tuple default.

    Args:
        value: The supplied option value.
        default: The signature default.

    Returns:
        bool: True when the option still holds its default value.
    """
    if value is default:
        return True
    if isinstance(value, bool) != isinstance(default, bool):
        # `0` is not `False`: a flag given an integer was supplied deliberately.
        return False
    if np.ndim(value) != np.ndim(default):
        return False
    return bool(np.array_equal(value, default))


def _check_brain_data(data, mask=None):
    """Return *data* as a BrainData, coercing Niimg-like inputs if needed.

    If *data* is already a BrainData, the optional *mask* is applied via
    `BrainData.apply_mask`.  Otherwise *data* is passed through
    `BrainData`, which dispatches on type (file path, list of paths,
    URL, h5, ``nib.Nifti1Image``, numpy array).  Unsupported types raise
    ``TypeError`` from
    `_validate_data_type`.
    """
    from . import BrainData

    if isinstance(data, BrainData):
        if mask is not None:
            data = data.apply_mask(mask)
        return data
    return BrainData(data, mask=mask)


def _check_brain_data_is_single(data):
    """Logical test if BrainData instance is a single image.

    Args:
        data (BrainData | Nifti1Image | str): Data to test; non-BrainData
            inputs are coerced first.

    Returns:
        bool: True if the data holds a single image.
    """
    data = _check_brain_data(data)
    return len(data.shape) <= 1


#: Every attribute a fit may attach to a BrainData. The enumeration is
#: exhaustive: clearing fitted state deletes exactly these names, with no
#: predicates and no special cases. `predict` attaches nothing, so it
#: contributes no names.
_FIT_STATE_ATTRIBUTES = (
    "model_",
    "ridge_weights",
    "ridge_fitted_values",
    "ridge_r2",
    "glm_betas",
    "glm_residual",
    "glm_predicted",
    "glm_r2",
)


def _clear_fit_state(bd):
    """Remove state invalidated by changing a BrainData object's data."""
    for name in _FIT_STATE_ATTRIBUTES:
        if hasattr(bd, name):
            delattr(bd, name)


def _copy_for_fit(source):
    """Copy retained state without traversing obsolete fitted attributes."""
    return _copy_graph(source, exclude=_FIT_STATE_ATTRIBUTES)


def _row_values(data, X, Y):
    """Validate the complete replacement row state before graph construction."""
    from ..validation import _validate_frame

    data = np.asarray(data)
    count = 0 if data.size == 0 else (1 if data.ndim == 1 else data.shape[0])
    X, Y = _validate_frame(X, frame_type="X"), _validate_frame(Y, frame_type="Y")
    for name, frame in (("X", X), ("Y", Y)):
        if not frame.is_empty() and frame.height != count:
            raise ValueError(
                f"{name} has {frame.height} rows but result data has {count} rows"
            )
    return {"data": data, "_X": X, "_Y": Y}


def _result_from_rows(source, data, *, X, Y):
    """Construct independently owned data with complete replacement row metadata."""
    return _copy_graph(
        source, exclude=_FIT_STATE_ATTRIBUTES, replacements=_row_values(data, X, Y)
    )


def _result_from_array(source, data, *, rows):
    """Construct a result with an explicit observation-row policy."""
    if rows not in ("preserve", "clear"):
        raise ValueError("rows must be 'preserve' or 'clear'")
    return _result_from_rows(
        source,
        data,
        X=source.X if rows == "preserve" else None,
        Y=source.Y if rows == "preserve" else None,
    )


def _result_from_selection(source, index):
    """Apply one observation selection to data and both metadata frames."""
    if not isinstance(index, (int, np.integer, slice)):
        index = np.asarray(index).flatten()
    data = source.data[index, :]
    if not isinstance(index, slice) and data.ndim == 2 and data.shape[0] == 1:
        data = data[0]
    return _result_from_rows(
        source,
        data,
        X=_polars_row_select(source.X, index),
        Y=_polars_row_select(source.Y, index),
    )


def _result_with_mask(source, data, mask, *, rows):
    """Install independent replacement spatial state for a changed voxel axis."""
    from .io import _initialize_mask

    data = np.asarray(data)
    if data.size and data.shape[-1] != int(np.count_nonzero(mask.get_fdata() > 0)):
        raise ValueError("Result voxel axis must match replacement mask support")
    if rows not in ("preserve", "clear"):
        raise ValueError("rows must be 'preserve' or 'clear'")
    values = _row_values(
        data,
        source.X if rows == "preserve" else None,
        source.Y if rows == "preserve" else None,
    )
    values.update({"mask": mask, "masker": None, "_labels": None})
    result = _copy_graph(source, exclude=_FIT_STATE_ATTRIBUTES, replacements=values)
    _initialize_mask(result, result.mask)
    return result


def _perform_arithmetic(
    bd, other, operation, operation_name, inplace=False, reverse=False
):
    """Perform an arithmetic operation with validation.

    Args:
        bd (BrainData): Left operand unless ``reverse`` is True.
        other (float | BrainData | np.ndarray): The other operand.
        operation (Callable): NumPy ufunc (e.g. ``np.add``, ``np.subtract``).
        operation_name (str): Human-readable name for error messages.
        inplace (bool): If True, mutate ``bd`` in place.
        reverse (bool): If True, reverse operand order (for ``__rsub__`` etc.).

    Returns:
        BrainData: Result of the operation.
    """
    from .validation import _validate_arithmetic_operand, _validate_brain_data_shapes

    operand_type = _validate_arithmetic_operand(other, operation_name)

    if operand_type == "scalar":
        if reverse:
            result_data = operation(other, bd.data)
        else:
            result_data = operation(bd.data, other)
    elif operand_type == "brain_data":
        _validate_brain_data_shapes(bd, other, operation_name)
        if reverse:
            result_data = operation(other.data, bd.data)
        else:
            result_data = operation(bd.data, other.data)
    elif operand_type == "array":
        if len(other) != len(bd):
            raise ValueError(
                f"Vector {operation_name} requires that the length of the vector "
                f"({len(other)}) match the number of images ({len(bd)})"
            )
        result_data = np.dot(bd.data.T, other).T

    if inplace:
        _clear_fit_state(bd)
        bd.data = result_data
        if operand_type == "array":
            bd.X = None
            bd.Y = None
        return bd
    return _result_from_array(
        bd, result_data, rows="clear" if operand_type == "array" else "preserve"
    )


def _apply_func(bd, stat_func, axis=0):
    """Apply a statistical function to BrainData's ``.data`` attribute.

    If *axis* is 0, returns a BrainData with the statistic computed across
    samples (e.g. within a voxel over time).  If *axis* is 1, returns a numpy
    array with the statistic computed across features (e.g. across voxels
    within a single time-point).

    Args:
        bd (BrainData): Data to reduce.
        stat_func (Callable): Accepts an array and an ``axis`` kwarg.
        axis (int): ``0`` = across images, ``1`` = within images.

    Returns:
        float | np.ndarray | BrainData: The reduced result; type depends on
            whether the input is a single image and on ``axis``.
    """
    if _check_brain_data_is_single(bd):
        return stat_func(bd.data)

    if axis == 1:
        return stat_func(bd.data, axis=1)
    if axis == 0:
        return _result_from_array(bd, stat_func(bd.data, axis=0), rows="clear")
    raise ValueError("axis must be 0 or 1")


def _polars_row_select(df, index):
    """Row-select a polars DataFrame by int / slice / int-array index.

    Polars has no ``.iloc`` — this helper normalizes the three index
    shapes BrainData's ``__getitem__`` hands it (pandas parity).
    """
    import polars as pl

    if df.is_empty():
        return df
    if isinstance(index, (int, np.integer)):
        return df.slice(int(index), 1)
    if isinstance(index, slice):
        return df[index]
    idx = np.asarray(index).flatten()
    if idx.dtype == bool:
        return df.filter(pl.Series(idx))
    return df[idx.tolist()]
