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
    "predict_permutation_scores",
    "predict_permutation_pvalue",
)

_FIT_STATE_ATTRIBUTES = (
    "model_",
    "X_",
    "design_matrix",
    "cv_results_",
    "ridge_weights",
    "ridge_fitted_values",
    "ridge_scores",
    "glm_betas",
    "glm_t",
    "glm_p",
    "glm_se",
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
        if name == "design_matrix":
            bd.design_matrix = None
        elif hasattr(bd, name):
            delattr(bd, name)


def _copy_without_fit_state(bd, *, copy_data=True, independent=False):
    """Create a result-shaped BrainData without derived model state.

    By default, the data buffer and mutable metadata are copied while the mask
    and masker are shared. Callers that immediately replace ``data`` can opt
    out of that copy. With ``independent=True``, all retained state is
    deep-copied. Every mode omits fitted/prediction state and resets
    ``design_matrix`` to ``None``.

    Args:
        bd (BrainData): Instance to copy.
        copy_data (bool): Copy the data buffer. Default True.
        independent (bool): Deep-copy retained structure and data. Default False.

    Returns:
        BrainData: Clean derived instance.
    """
    from . import BrainData

    new = BrainData.__new__(BrainData)
    memo = {id(bd): new}

    for key, value in bd.__dict__.items():
        if key in _FIT_STATE_ATTRIBUTES:
            if key == "design_matrix":
                new.design_matrix = None
            continue
        if key == "data":
            setattr(
                new,
                key,
                deepcopy(value, memo) if independent or copy_data else value,
            )
        elif independent:
            setattr(new, key, deepcopy(value, memo))
        elif key in ("mask", "masker"):
            setattr(new, key, value)
        elif key in ("_X", "_Y"):
            import polars as pl

            setattr(
                new, key, value.clone() if isinstance(value, pl.DataFrame) else value
            )
        else:
            setattr(new, key, deepcopy(value, memo))

    if not hasattr(new, "design_matrix"):
        new.design_matrix = None

    return new


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

    new = bd if inplace else _copy_without_fit_state(bd, copy_data=False)
    if inplace:
        _clear_fit_state(new)
    new.data = result_data
    return new


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
        import polars as pl

        out = _copy_without_fit_state(bd, copy_data=False)
        out.data = stat_func(bd.data, axis=0)
        out.X = pl.DataFrame()
        out.Y = pl.DataFrame()
        return out
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
