"""Shared helpers for BrainData submodules.

These are internal utilities used by the facade and submodules — not part of the
public API.
"""

from copy import deepcopy

import numpy as np


def check_brain_data(data, mask=None):
    """Return *data* as a BrainData, coercing Niimg-like inputs if needed.

    If *data* is already a BrainData, the optional *mask* is applied via
    `BrainData.apply_mask`.  Otherwise *data* is passed through
    `BrainData`, which dispatches on type (file path, list of paths,
    URL, h5, ``nib.Nifti1Image``, numpy array).  Unsupported types raise
    ``TypeError`` from
    `validate_data_type`.
    """
    from . import BrainData

    if isinstance(data, BrainData):
        if mask is not None:
            data = data.apply_mask(mask)
        return data
    return BrainData(data, mask=mask)


def check_brain_data_is_single(data):
    """Logical test if BrainData instance is a single image.

    Args:
        data (BrainData | Nifti1Image | str): Data to test; non-BrainData
            inputs are coerced first.

    Returns:
        bool: True if the data holds a single image.
    """
    data = check_brain_data(data)
    return len(data.shape) <= 1


_PREDICTION_STATE_ATTRIBUTES = (
    "predict_predictions",
    "predict_scores",
    "predict_mean_score",
    "predict_std_score",
    "predict_cv_folds",
    "predict_roi_labels",
    "predict_accuracy_map",
    "predict_weight_map",
    "predict_fold_weight_maps",
    "predict_estimator",
)

#: Every attribute a fit may attach to a BrainData. The enumeration is
#: exhaustive: clearing fitted state deletes exactly these names, with no
#: predicates and no special cases.
_FIT_STATE_ATTRIBUTES = (
    "model_",
    "X_",
    "ridge_weights",
    "ridge_fitted_values",
    "ridge_scores",
    "glm_betas",
    "glm_residual",
    "glm_predicted",
    "glm_r2",
    *_PREDICTION_STATE_ATTRIBUTES,
)


def _clear_prediction_state(bd):
    """Remove results attached by a previous in-place decoding call."""
    for name in _PREDICTION_STATE_ATTRIBUTES:
        if hasattr(bd, name):
            delattr(bd, name)


def _clear_fit_state(bd):
    """Remove state invalidated by changing a BrainData object's data."""
    for name in _FIT_STATE_ATTRIBUTES:
        if hasattr(bd, name):
            delattr(bd, name)


def _copy_graph(source, *, memo=None, exclude=(), replacements=None):
    """Copy one retained object graph, preserving its internal aliases."""
    if memo is None:
        memo = {}
    if id(source) in memo:
        return memo[id(source)]
    new = type(source).__new__(type(source))
    memo[id(source)] = new
    values = {
        key: value for key, value in source.__dict__.items() if key not in exclude
    }
    if replacements is not None:
        values.update(replacements)
    _copy_object_frames(values, memo)
    for key, value in values.items():
        setattr(new, key, deepcopy(value, memo))
    return new


def _copy_object_frames(values, memo):
    """Prepare Polars Object cells for deepcopy without sharing Python objects."""
    import polars as pl

    frames = []
    seen = set()

    def discover(value):
        if id(value) in seen or id(value) in memo:
            return
        seen.add(id(value))
        if isinstance(value, pl.DataFrame):
            frames.append(value)
            for series in value:
                if series.dtype == pl.Object:
                    for cell in series:
                        discover(cell)
        elif isinstance(value, dict):
            for key, item in value.items():
                discover(key)
                discover(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                discover(item)

    discover(values)
    # Register all frames first, including frames referred to by Object cells.
    # The common memo preserves cycles and cell aliases across metadata frames.
    for frame in frames:
        memo[id(frame)] = frame.clone()
    for frame in frames:
        for index, series in enumerate(frame):
            if series.dtype == pl.Object:
                memo[id(frame)].replace_column(
                    index,
                    pl.Series(
                        series.name,
                        [deepcopy(cell, memo) for cell in series],
                        dtype=pl.Object,
                    ),
                )


def _copy_complete(source, memo=None):
    """Return a complete independently owned snapshot."""
    return _copy_graph(source, memo=memo)


def _copy_for_fit(source):
    """Copy retained state without traversing obsolete fitted attributes."""
    return _copy_graph(source, exclude=_FIT_STATE_ATTRIBUTES)


def _row_values(data, X, Y):
    """Validate the complete replacement row state before graph construction."""
    from .validation import validate_frame

    data = np.asarray(data)
    count = 0 if data.size == 0 else (1 if data.ndim == 1 else data.shape[0])
    X, Y = validate_frame(X, frame_type="X"), validate_frame(Y, frame_type="Y")
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
    from .io import initialize_mask

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
    initialize_mask(result, result.mask)
    return result


def perform_arithmetic(
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
    from .validation import validate_arithmetic_operand, validate_brain_data_shapes

    operand_type = validate_arithmetic_operand(other, operation_name)

    if operand_type == "scalar":
        if reverse:
            result_data = operation(other, bd.data)
        else:
            result_data = operation(bd.data, other)
    elif operand_type == "brain_data":
        validate_brain_data_shapes(bd, other, operation_name)
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


def apply_func(bd, stat_func, axis=0):
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
    if check_brain_data_is_single(bd):
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
