"""Shared shape manipulation utilities for algorithms module.

This module provides common shape manipulation functions to reduce code
duplication and ensure consistent behavior across the algorithms module.

Key functions:
    - extract_triangle_elements: Extract upper/lower triangle from matrices
    - permute_matrix_symmetric: Apply symmetric permutation (key for matrix tests)
    - ensure_2d: Ensure arrays are 2D (adds dimension if needed)

Usage:
    These utilities are used throughout the algorithms module for consistent
    shape handling and matrix operations.

    Example:
        >>> from nltools.algorithms.shape_utils import extract_triangle_elements
        >>> matrix = np.arange(16).reshape(4, 4)
        >>> upper = extract_triangle_elements(matrix, triangle='upper')
"""

import numpy as np


def extract_triangle_elements(
    matrix: np.ndarray,
    triangle: str = "upper",
    include_diag: bool = False,
) -> np.ndarray:
    """Extract triangle elements from square matrix.

    Args:
        matrix: Square matrix (n×n)
        triangle: Which triangle ['upper'|'lower'|'full']
        include_diag: Include diagonal (only for 'full')

    Returns:
        Extracted elements as 1D array

    Examples:
        >>> matrix = np.arange(16).reshape(4, 4)
        >>> extract_triangle_elements(matrix, triangle='upper')
        array([ 1,  2,  3,  6,  7, 11])
    """
    if triangle == "upper":
        return matrix[np.triu_indices(matrix.shape[0], k=1)]
    if triangle == "lower":
        return matrix[np.tril_indices(matrix.shape[0], k=-1)]
    if triangle == "full":
        if include_diag:
            return matrix.ravel()
        # Concatenate upper and lower triangles (exclude diagonal)
        upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
        lower = matrix[np.tril_indices(matrix.shape[0], k=-1)]
        return np.concatenate([upper, lower])
    raise ValueError(f"triangle must be 'upper', 'lower', or 'full', got {triangle}")


def permute_matrix_symmetric(
    matrix: np.ndarray,
    permutation: np.ndarray,
) -> np.ndarray:
    """Apply symmetric row+column permutation to square matrix.

    This is the KEY operation for matrix permutation tests. It reorders
    both rows AND columns together, preserving matrix structure while
    destroying correlation between matrices.

    Args:
        matrix: Square matrix (n×n)
        permutation: Permutation indices (length n)

    Returns:
        Symmetrically permuted matrix (n×n)

    Examples:
        >>> matrix = np.arange(9).reshape(3, 3)
        >>> perm = np.array([2, 0, 1])  # Rotate indices
        >>> permute_matrix_symmetric(matrix, perm)
        array([[8, 6, 7],
               [2, 0, 1],
               [5, 3, 4]])
    """
    return matrix[permutation][:, permutation]


def ensure_2d(array: np.ndarray, name: str = "array") -> np.ndarray:
    """Ensure array is 2D, adding dimension if needed.

    Args:
        array: Input array (1D or 2D)
        name: Name for error messages

    Returns:
        2D array (shape [n, 1] if input was 1D)

    Examples:
        >>> x = np.array([1, 2, 3])
        >>> ensure_2d(x).shape
        (3, 1)
        >>> x2d = np.array([[1, 2], [3, 4]])
        >>> ensure_2d(x2d).shape
        (2, 2)
    """
    if array.ndim == 1:
        return array[:, np.newaxis]
    if array.ndim == 2:
        return array
    raise ValueError(
        f"{name} must be 1D or 2D, got shape {array.shape} ({array.ndim}D)"
    )
