"""
Matrix permutation test implementations (Mantel test).

This module provides CPU-parallel implementations of matrix permutation tests
for testing correlation between two square matrices.
"""

import numpy as np
from typing import Optional
from scipy.stats import pearsonr, spearmanr, kendalltau

from .utils import _compute_pvalue

# Maximum integer for random seed generation
MAX_INT = np.iinfo(np.int32).max


def _extract_matrix_elements(
    matrix: np.ndarray,
    how: str = "upper",
    include_diag: bool = False,
) -> np.ndarray:
    """
    Extract elements from square matrix for correlation computation.

    Args:
        matrix (np.ndarray): Square matrix (n×n)
        how (str): Which elements to extract ['upper'|'lower'|'full']
            - 'upper': Upper triangle (default, assumes symmetric)
            - 'lower': Lower triangle
            - 'full': All elements (see include_diag)
        include_diag (bool): Whether to include diagonal (only applies if how='full')

    Returns:
        np.ndarray: 1D array of extracted elements

    Examples:
        >>> matrix = np.arange(16).reshape(4, 4)
        >>> _extract_matrix_elements(matrix, how='upper')
        array([ 1,  2,  3,  6,  7, 11])
    """
    if how == "upper":
        return matrix[np.triu_indices(matrix.shape[0], k=1)]
    elif how == "lower":
        return matrix[np.tril_indices(matrix.shape[0], k=-1)]
    elif how == "full":
        if include_diag:
            return matrix.ravel()
        else:
            # Concatenate upper and lower triangles (exclude diagonal)
            upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
            lower = matrix[np.tril_indices(matrix.shape[0], k=-1)]
            return np.concatenate([upper, lower])
    else:
        raise ValueError(f"how must be 'upper', 'lower', or 'full', got {how}")


def _permute_matrix_symmetric(
    matrix: np.ndarray,
    permutation: np.ndarray,
) -> np.ndarray:
    """
    Apply symmetric row+column permutation to square matrix.

    This is the KEY operation for matrix permutation tests. It reorders
    both rows AND columns together, preserving matrix structure while
    destroying correlation between matrices.

    Args:
        matrix (np.ndarray): Square matrix (n×n)
        permutation (np.ndarray): Permutation indices (length n)

    Returns:
        np.ndarray: Symmetrically permuted matrix (n×n)

    Examples:
        >>> matrix = np.arange(9).reshape(3, 3)
        >>> perm = np.array([2, 0, 1])  # Rotate indices
        >>> _permute_matrix_symmetric(matrix, perm)
        array([[8, 6, 7],
               [2, 0, 1],
               [5, 3, 4]])
    """
    return matrix[permutation][:, permutation]


def _compute_matrix_correlation(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    how: str = "upper",
    include_diag: bool = False,
    metric: str = "pearson",
) -> float:
    """
    Compute correlation between elements of two matrices.

    Args:
        matrix1 (np.ndarray): First square matrix (n×n)
        matrix2 (np.ndarray): Second square matrix (n×n)
        how (str): Element extraction mode (passed to _extract_matrix_elements)
        include_diag (bool): Include diagonal (passed to _extract_matrix_elements)
        metric (str): Correlation type ['pearson'|'spearman'|'kendall']

    Returns:
        float: Correlation coefficient

    Examples:
        >>> m1 = np.eye(3)
        >>> m2 = np.eye(3)
        >>> _compute_matrix_correlation(m1, m2, metric='pearson')
        1.0
    """
    # Extract elements from both matrices
    elements1 = _extract_matrix_elements(matrix1, how=how, include_diag=include_diag)
    elements2 = _extract_matrix_elements(matrix2, how=how, include_diag=include_diag)

    # Compute correlation
    if metric == "pearson":
        r, _ = pearsonr(elements1, elements2)
    elif metric == "spearman":
        r, _ = spearmanr(elements1, elements2)
    elif metric == "kendall":
        r, _ = kendalltau(elements1, elements2)
    else:
        raise ValueError(
            f"metric must be 'pearson', 'spearman', or 'kendall', got {metric}"
        )

    return r


def _matrix_permutation_cpu_parallel(
    data1: np.ndarray,
    data2: np.ndarray,
    n_permute: int,
    metric: str,
    how: str,
    include_diag: bool,
    tail: int,
    return_null: bool,
    n_jobs: int,
    random_state: Optional[int],
) -> dict:
    """
    Matrix permutation test using CPU parallelization with joblib.

    Pre-generates seeds deterministically, then parallelizes permutation
    computation. This ensures perfect reproducibility across runs.

    Args:
        data1 (np.ndarray): First square matrix (n×n)
        data2 (np.ndarray): Second square matrix (n×n)
        n_permute (int): Number of permutations
        metric (str): Correlation metric
        how (str): Element extraction mode
        include_diag (bool): Include diagonal
        tail (int): Test type (1 or 2)
        return_null (bool): Whether to return null distribution
        n_jobs (int): Number of parallel jobs (-1 = all cores)
        random_state (int, optional): Random seed for reproducibility

    Returns:
        dict: Dictionary with 'correlation', 'p', 'backend', and optionally 'null_dist'

    Notes:
        - Pre-generates seeds (matches established pattern from two_sample.py)
        - Parallelizes computation, not RNG (ensures determinism)
        - Progress bar shows permutation completion
        - Typical speedup: 4-8× on 8-core machines
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm

    # Validate inputs
    assert data1.shape == data2.shape, "Matrices must have same shape"
    assert data1.shape[0] == data1.shape[1], "Matrices must be square"

    # Pre-generate seeds (deterministic)
    rng = np.random.RandomState(random_state)
    seeds = rng.randint(MAX_INT, size=n_permute)

    # Compute observed correlation
    obs_corr = _compute_matrix_correlation(
        data1, data2, how=how, include_diag=include_diag, metric=metric
    )

    # Define worker function
    def _compute_one_perm(seed):
        """Compute correlation for one permutation."""
        perm_rng = np.random.RandomState(seed)
        perm = perm_rng.permutation(data1.shape[0])
        permuted_matrix = _permute_matrix_symmetric(data1, perm)
        return _compute_matrix_correlation(
            permuted_matrix, data2, how=how, include_diag=include_diag, metric=metric
        )

    # Execute in parallel with progress bar
    null_dist = Parallel(n_jobs=n_jobs)(
        delayed(_compute_one_perm)(seeds[i])
        for i in tqdm(range(n_permute), desc="Matrix permutation", unit="perm")
    )
    null_dist = np.array(null_dist)

    # Compute p-value
    p_value = _compute_pvalue(obs_corr, null_dist, tail=tail)
    # _compute_pvalue returns array, extract scalar for single correlation
    if isinstance(p_value, np.ndarray):
        p_value = float(p_value[0])

    # Determine backend name based on n_jobs
    if n_jobs == -1:
        import multiprocessing

        n_cores = multiprocessing.cpu_count()
        backend_name = f"cpu-parallel-{n_cores}"
    else:
        backend_name = f"cpu-parallel-{n_jobs}"

    # Build result
    result = {
        "correlation": obs_corr,
        "p": p_value,
        "backend": backend_name,
    }

    if return_null:
        result["null_dist"] = null_dist

    return result


def matrix_permutation_test(
    data1: np.ndarray,
    data2: np.ndarray,
    n_permute: int = 5000,
    metric: str = "pearson",
    how: str = "upper",
    include_diag: bool = False,
    tail: int = 2,
    n_jobs: int = -1,
    return_null: bool = False,
    random_state: Optional[int] = None,
) -> dict:
    """Matrix permutation test (Mantel test) for correlating two square matrices.

    Tests whether the correlation between elements of two matrices is significant
    by permuting rows and columns of one matrix symmetrically while keeping the
    other fixed.

    **Statistical Method**:
    For each permutation, create random permutation `perm`, then apply:
    `matrix1[perm][:, perm]`. This preserves matrix structure while destroying
    correlation. Count how often permuted correlation is as extreme as observed.

    **Assumptions**:
    - Matrices are square and same size
    - Under H₀, row/column ordering is exchangeable
    - Symmetric permutation preserves matrix properties (e.g., symmetry)

    Args:
        data1 (np.ndarray): First square matrix (n×n)
        data2 (np.ndarray): Second square matrix (n×n)
        n_permute (int): Number of permutations (default: 5000)
        metric (str): Correlation metric ['pearson'|'spearman'|'kendall'] (default: 'pearson')
        how (str): Which elements to compare ['upper'|'lower'|'full'] (default: 'upper')
            - 'upper': Upper triangle only (assumes symmetric matrices)
            - 'lower': Lower triangle only
            - 'full': All elements (see include_diag)
        include_diag (bool): Include diagonal elements (only applies if how='full') (default: False)
        tail (int): One-tailed (1) or two-tailed (2) test (default: 2)
        n_jobs (int): Number of parallel workers, -1 = all cores (default: -1)
        return_null (bool): Return null distribution (default: False)
        random_state (int, optional): Random seed for reproducibility

    Returns:
        dict: Dictionary with keys:
            - 'correlation' (float): Observed correlation coefficient
            - 'p' (float): P-value using Phipson-Smyth correction
            - 'backend' (str): Backend used (e.g., 'cpu-parallel-8')
            - 'null_dist' (np.ndarray): Null distribution (if return_null=True)

    References:
        Chen, G. et al. (2016). Untangling the relatedness among correlations,
        part I: nonparametric approaches to inter-subject correlation analysis
        at the group level. NeuroImage, 142, 248-259.

        Mantel, N. (1967). The detection of disease clustering and a generalized
        regression approach. Cancer Research, 27(2), 209-220.

    Examples:
        >>> import numpy as np
        >>> from nltools.algorithms.inference import matrix_permutation_test
        >>>
        >>> # Create two correlated similarity matrices
        >>> np.random.seed(42)
        >>> n = 50
        >>> true_pattern = np.random.randn(n)
        >>> data1 = np.corrcoef(true_pattern + np.random.randn(n) * 0.1)
        >>> data2 = np.corrcoef(true_pattern + np.random.randn(n) * 0.1)
        >>>
        >>> # Test if matrices are correlated
        >>> result = matrix_permutation_test(data1, data2, n_permute=1000)
        >>> print(f"Correlation: {result['correlation']:.3f}, p = {result['p']:.4f}")
    """
    # Input validation
    if not isinstance(data1, np.ndarray) or not isinstance(data2, np.ndarray):
        raise TypeError("data1 and data2 must be numpy arrays")

    if data1.shape != data2.shape:
        raise ValueError(
            f"Matrices must have same shape, got {data1.shape} and {data2.shape}"
        )

    if data1.shape[0] != data1.shape[1]:
        raise ValueError(f"Matrices must be square, got shape {data1.shape}")

    if metric not in ["pearson", "spearman", "kendall"]:
        raise ValueError(
            f"metric must be 'pearson', 'spearman', or 'kendall', got {metric}"
        )

    if how not in ["upper", "lower", "full"]:
        raise ValueError(f"how must be 'upper', 'lower', or 'full', got {how}")

    if tail not in [1, 2]:
        raise ValueError(f"tail must be 1 or 2, got {tail}")

    # Route to CPU-parallel implementation
    return _matrix_permutation_cpu_parallel(
        data1=data1,
        data2=data2,
        n_permute=n_permute,
        metric=metric,
        how=how,
        include_diag=include_diag,
        tail=tail,
        return_null=return_null,
        n_jobs=n_jobs,
        random_state=random_state,
    )
