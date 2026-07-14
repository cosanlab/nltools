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
    URL, h5, ``nib.Nifti1Image``).  Unsupported types raise ``TypeError`` from
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
        data: brain data

    Returns:
        (bool)
    """
    data = check_brain_data(data)
    return len(data.shape) <= 1


def shallow_copy(bd):
    """Create a shallow copy of a BrainData for efficient method chaining.

    Creates a new BrainData instance that shares immutable objects (mask)
    but copies mutable attributes.  The data array is NOT copied — callers
    should handle data copying as needed.

    Args:
        bd: BrainData instance to copy.

    Returns:
        BrainData: New instance with shared/copied attributes.
    """
    from . import BrainData

    new = BrainData.__new__(BrainData)

    for key, value in bd.__dict__.items():
        if key == "data":
            new.data = bd.data  # reference only
        elif key in ("mask", "masker"):
            setattr(new, key, value)
        elif key in ("_X", "_Y"):
            import polars as pl

            setattr(
                new, key, value.clone() if isinstance(value, pl.DataFrame) else value
            )
        elif key == "design_matrix":
            if value is not None:
                if hasattr(value, "copy"):
                    setattr(new, key, value.copy())
                else:
                    setattr(new, key, deepcopy(value))
            else:
                setattr(new, key, None)
        elif key.startswith(("glm_", "ridge_")):
            setattr(new, key, value)
        elif key in ("model_", "X_", "cv_results_"):
            pass  # fitted model state — don't propagate
        else:
            setattr(new, key, deepcopy(value))

    return new


def perform_arithmetic(
    bd, other, operation, operation_name, inplace=False, reverse=False
):
    """Perform an arithmetic operation with validation.

    Args:
        bd: BrainData instance (left operand unless *reverse* is True).
        other: The other operand (scalar, BrainData, or array).
        operation: Numpy ufunc (e.g. ``np.add``, ``np.subtract``).
        operation_name: Human-readable name for error messages.
        inplace: If True, mutate *bd* in place.
        reverse: If True, reverse operand order (for ``__rsub__`` etc.).

    Returns:
        BrainData: Result of the operation.
    """
    from .validation import validate_arithmetic_operand, validate_brain_data_shapes

    new = bd if inplace else shallow_copy(bd)
    operand_type = validate_arithmetic_operand(other, operation_name)

    if operand_type == "scalar":
        if reverse:
            new.data = operation(other, bd.data)
        else:
            new.data = operation(bd.data, other)
    elif operand_type == "brain_data":
        validate_brain_data_shapes(bd, other, operation_name)
        if reverse:
            new.data = operation(other.data, bd.data)
        else:
            new.data = operation(bd.data, other.data)
    elif operand_type == "array":
        if len(other) != len(bd):
            raise ValueError(
                f"Vector {operation_name} requires that the length of the vector "
                f"({len(other)}) match the number of images ({len(bd)})"
            )
        new.data = np.dot(bd.data.T, other).T

    return new


def apply_func(bd, stat_func, axis=0):
    """Apply a statistical function to BrainData's ``.data`` attribute.

    If *axis* is 0, returns a BrainData with the statistic computed across
    samples (e.g. within a voxel over time).  If *axis* is 1, returns a numpy
    array with the statistic computed across features (e.g. across voxels
    within a single time-point).

    Args:
        bd: BrainData instance.
        stat_func: Callable accepting an array and an ``axis`` kwarg.
        axis: 0 = across images, 1 = within images.

    Returns:
        float | np.ndarray | BrainData
    """
    if check_brain_data_is_single(bd):
        return stat_func(bd.data)

    if axis == 1:
        return stat_func(bd.data, axis=1)
    if axis == 0:
        import polars as pl

        out = shallow_copy(bd)
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
