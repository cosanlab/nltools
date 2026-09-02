"""Square-matrix helpers: triangle extraction and symmetric permutation.

`extract_triangle_elements` pulls the upper or lower triangle of a matrix into a
vector; `permute_matrix_symmetric` reorders rows and columns together, the
operation at the heart of the matrix permutation tests.

Examples:
    ```python
    import numpy as np
    from nltools.algorithms.shape_utils import extract_triangle_elements

    matrix = np.arange(16).reshape(4, 4)
    upper = extract_triangle_elements(matrix, triangle='upper')
    ```
"""

import numpy as np


def extract_triangle_elements(
    matrix: np.ndarray,
    triangle: str = "upper",
    include_diag: bool = False,
) -> np.ndarray:
    """Extract the off-diagonal triangle of a square matrix as a vector.

    Args:
        matrix (np.ndarray): Square matrix, shape (n, n).
        triangle (str): 'upper', 'lower', or 'full' (upper then lower). Defaults to 'upper'.
        include_diag (bool): With `triangle='full'`, return every element
            (`matrix.ravel()`) instead of excluding the diagonal. Defaults to False.

    Returns:
        np.ndarray: The selected elements as a 1D array.

    Examples:
        ```python
        matrix = np.arange(16).reshape(4, 4)
        extract_triangle_elements(matrix, triangle="upper")
        # → array([ 1,  2,  3,  6,  7, 11])
        ```
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
    """Permute the rows and columns of a square matrix together.

    Reordering both axes with the same permutation relabels the items while
    preserving the matrix's internal structure, which is what breaks the
    correspondence between two matrices in a matrix permutation test.

    Args:
        matrix (np.ndarray): Square matrix, shape (n, n).
        permutation (np.ndarray): Permutation of `range(n)`.

    Returns:
        np.ndarray: The permuted matrix, shape (n, n).

    Examples:
        ```python
        matrix = np.arange(9).reshape(3, 3)
        perm = np.array([2, 0, 1])
        permute_matrix_symmetric(matrix, perm)
        # → array([[8, 6, 7],
        #          [2, 0, 1],
        #          [5, 3, 4]])
        ```
    """
    return matrix[permutation][:, permutation]
