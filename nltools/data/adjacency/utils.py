"""Shared helpers for Adjacency submodules.

These are internal utilities used by the facade and submodules — not part of the
public API.
"""

from copy import deepcopy

import numpy as np
import polars as pl


def perform_arithmetic(adj, y, op, op_name, reverse=False):
    """Perform arithmetic operation with validation.

    Args:
        adj: Adjacency instance (left operand unless *reverse* is True).
        y: Operand (scalar or Adjacency).
        op: Callable that performs the operation on arrays.
        op_name: Name of operation for error messages.
        reverse: If True, reverse operand order (y op adj).

    Returns:
        Adjacency: New instance with result.
    """
    new = deepcopy(adj)
    if isinstance(y, (int, np.integer, float, np.floating)):
        if reverse:
            new.data = op(y, new.data)
        else:
            new.data = op(new.data, y)
    else:
        # Import here to avoid circular import at module level
        from . import Adjacency

        if isinstance(y, Adjacency):
            from .state import validate_compatible

            validate_compatible(adj, y, labels=True)
            if adj.shape != y.shape:
                raise ValueError(
                    "Both Adjacency() instances need to be the same shape."
                )
            if reverse:
                new.data = op(y.data, new.data)
            else:
                new.data = op(new.data, y.data)
        else:
            raise ValueError(f"Can only {op_name} int, float, or Adjacency")
    return new


def apply_stat(adj, func, axis=0):
    """Apply a statistical function along an axis.

    Args:
        adj: Adjacency instance.
        func: Numpy function to apply (e.g., np.nanmean).
        axis: Axis along which to apply function. 0 for across matrices,
              1 for across upper triangle elements.

    Returns:
        float | Adjacency | np.ndarray: A float for a single matrix; an Adjacency
            when `axis=0` with multiple matrices; an array when `axis=1` with
            multiple matrices.
    """
    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    if adj.is_single_matrix:
        return func(adj.data)
    from .state import common_labels, result

    if axis == 0:
        return result(
            adj, func(adj.data, axis=axis), labels=common_labels(adj), Y=pl.DataFrame()
        )
    if axis == 1:
        return func(adj.data, axis=axis)
    raise ValueError(f"axis must be 0 or 1, got {axis}")
