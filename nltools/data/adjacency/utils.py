"""Shared helpers for Adjacency submodules.

These are internal utilities used by the facade and submodules — not part of the
public API.
"""

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial.distance import squareform


def test_is_single_matrix(data):
    """Check whether data represents a single matrix (1-D vector).

    Args:
        data: numpy array of adjacency data.

    Returns:
        bool: True if data is 1-D (single matrix in vector form).
    """
    return len(data.shape) == 1


def import_single_data(data, matrix_type=None):
    """Import and validate a single adjacency data matrix.

    Handles file paths (CSV), DataFrames, and numpy arrays. Determines
    symmetry and matrix type when not provided.

    Args:
        data: File path (str/Path), DataFrame, or numpy array.
        matrix_type: Optional explicit matrix type ('distance', 'similarity',
            'directed', or their '_flat' variants).

    Returns:
        tuple: (data, issymmetric, matrix_type, is_single_matrix)
    """
    if isinstance(data, (str, Path)):
        if os.path.isfile(data):
            data = pl.read_csv(str(data)).to_numpy()
        else:
            raise ValueError("Make sure you have specified a valid file path.")

    # Accept pandas DataFrame at the boundary for user convenience
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            data = data.values
    except ImportError:
        pass
    if isinstance(data, pl.DataFrame):
        data = data.to_numpy()

    if matrix_type is not None:
        if matrix_type.lower() == "distance_flat":
            matrix_type = "distance"
            data = np.asarray(data)
            issymmetric = True
            is_single_matrix = test_is_single_matrix(data)
        elif matrix_type.lower() == "similarity_flat":
            matrix_type = "similarity"
            data = np.asarray(data)
            issymmetric = True
            is_single_matrix = test_is_single_matrix(data)
        elif matrix_type.lower() == "directed_flat":
            matrix_type = "directed"
            # Collapse a single flat vector stored as an (n_features, 1) column
            # (e.g. a long-format CSV) to 1-D, but leave a genuine 2-D stack of
            # directed flat vectors, (n_matrices, n_features), intact.
            data = np.squeeze(np.asarray(data))
            issymmetric = False
            is_single_matrix = test_is_single_matrix(data)
        elif matrix_type.lower() in ["distance", "similarity", "directed"]:
            data = np.asarray(data)
            if data.shape[0] != data.shape[1]:
                raise ValueError("Data matrix must be square")
            matrix_type = matrix_type.lower()
            if matrix_type in ["distance", "similarity"]:
                issymmetric = True
                data = data[np.triu_indices(data.shape[0], k=1)]
            else:
                issymmetric = False
                data = data.flatten()
            is_single_matrix = True
    else:
        if len(data.shape) == 1:  # Single Vector
            try:
                data = squareform(data)
            except ValueError:
                print(
                    "Data is not flattened upper triangle from "
                    "similarity/distance matrix or flattened directed "
                    "matrix."
                )
            is_single_matrix = True
        elif data.shape[0] == data.shape[1]:  # Square Matrix
            is_single_matrix = True
        else:  # Rectangular Matrix
            data_all = deepcopy(data)
            try:
                data = squareform(data_all[0, :])
            except ValueError:
                print(
                    "Data is not flattened upper triangle from multiple "
                    "similarity/distance matrices or flattened directed "
                    "matrices."
                )
            is_single_matrix = False

        # Test if matrix is symmetrical
        if np.all(
            data[np.triu_indices(data.shape[0], k=1)]
            == data.T[np.triu_indices(data.shape[0], k=1)]
        ):
            issymmetric = True
        else:
            issymmetric = False

        # Determine matrix type
        if issymmetric:
            if np.sum(np.diag(data)) == 0:
                matrix_type = "distance"
            elif np.sum(np.diag(data)) == data.shape[0]:
                matrix_type = "similarity"
            data = data[np.triu_indices(data.shape[0], k=1)]
        else:
            matrix_type = "directed"
            data = data.flatten()

        if not is_single_matrix:
            data = data_all

    return (data, issymmetric, matrix_type, is_single_matrix)


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
        float if single matrix, Adjacency if axis=0 with multiple matrices,
        np.array if axis=1 with multiple matrices.
    """
    if adj.is_single_matrix:
        return func(adj.data)
    # Import here to avoid circular import at module level
    from . import Adjacency

    if axis == 0:
        return Adjacency(
            data=func(adj.data, axis=axis),
            matrix_type=adj.matrix_type + "_flat",
        )
    if axis == 1:
        return func(adj.data, axis=axis)
    raise ValueError(f"axis must be 0 or 1, got {axis}")
