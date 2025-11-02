"""
Matrix permutation test implementations (Mantel test).

This module provides CPU-parallel implementations of matrix permutation tests
for testing correlation between two square matrices, as well as matrix utility
functions for distance correlation and matrix centering operations.
"""

import numpy as np
from typing import Optional
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.spatial.distance import squareform, pdist
from scipy.stats import t as t_dist

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


def _compute_cross_correlation(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """
    Compute cross-correlation between columns (features) of two matrices.

    This function computes the correlation between each column of matrix1
    with each column of matrix2. Useful for computing connectivity matrices
    such as in intersubject functional connectivity (ISFC).

    Args:
        matrix1 (np.ndarray): First matrix with shape (n_observations, n_features1)
        matrix2 (np.ndarray): Second matrix with shape (n_observations, n_features2)

    Returns:
        np.ndarray: Cross-correlation matrix with shape (n_features1, n_features2)
            where element [i, j] is the correlation between matrix1[:, i] and matrix2[:, j]

    Examples:
        >>> matrix1 = np.random.randn(100, 5)  # 100 observations, 5 features
        >>> matrix2 = np.random.randn(100, 3)  # 100 observations, 3 features
        >>> corr = _compute_cross_correlation(matrix1, matrix2)
        >>> corr.shape
        (5, 3)

    Notes:
        Uses np.corrcoef for efficient computation. The result is extracted
        from the full correlation matrix by selecting the cross-correlation block.
    """
    if matrix1.shape[0] != matrix2.shape[0]:
        raise ValueError(
            f"Matrices must have same number of rows (observations), "
            f"got {matrix1.shape[0]} and {matrix2.shape[0]}"
        )

    # Compute full correlation matrix of concatenated matrices
    # Shape: (n_features1 + n_features2, n_features1 + n_features2)
    full_corr = np.corrcoef(matrix1.T, matrix2.T)

    # Extract cross-correlation block: correlations between matrix1 columns and matrix2 columns
    # This is the top-right block of the full correlation matrix
    # Block [0:n_features1, n_features1:] gives correlations between matrix1 columns and matrix2 columns
    cross_corr = full_corr[: matrix1.shape[1], matrix1.shape[1] :]

    return cross_corr


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

    # Build result
    result = {
        "correlation": obs_corr,
        "p": p_value,
        "parallel": "cpu",
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
    parallel: Optional[str] = "cpu",
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
        parallel (str, optional): Parallelization method (default: 'cpu')
            - None: Single-threaded NumPy (for debugging/small problems)
            - 'cpu': CPU parallelization via joblib (default, 4-8× speedup)
        n_jobs (int): Number of parallel workers, -1 = all cores (default: -1)
            Only used when parallel='cpu'
        return_null (bool): Return null distribution (default: False)
        random_state (int, optional): Random seed for reproducibility

    Returns:
        dict: Dictionary with keys:
            - 'correlation' (float): Observed correlation coefficient
            - 'p' (float): P-value using Phipson-Smyth correction
            - 'parallel' (str): Parallelization method used ('cpu' or None)
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

    # Validate parallel parameter
    if parallel not in [None, "cpu"]:
        raise ValueError(
            f"parallel must be None or 'cpu', got {parallel!r}. "
            "GPU support not yet implemented for matrix permutation tests."
        )

    # Decide execution mode based on parallel parameter
    if parallel == "cpu":
        # CPU parallelization mode
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
    else:
        # Single-threaded NumPy mode
        rng = np.random.RandomState(random_state)
        seeds = rng.randint(MAX_INT, size=n_permute)

        # Compute observed correlation
        obs_corr = _compute_matrix_correlation(
            data1, data2, how=how, include_diag=include_diag, metric=metric
        )

        # Generate null distribution
        null_dist = []
        for seed in seeds:
            perm_rng = np.random.RandomState(seed)
            perm = perm_rng.permutation(data1.shape[0])
            permuted_matrix = _permute_matrix_symmetric(data1, perm)
            corr = _compute_matrix_correlation(
                permuted_matrix,
                data2,
                how=how,
                include_diag=include_diag,
                metric=metric,
            )
            null_dist.append(corr)

        null_dist = np.array(null_dist)

        # Compute p-value
        p_value = _compute_pvalue(obs_corr, null_dist, tail=tail)
        if isinstance(p_value, np.ndarray):
            p_value = float(p_value[0])

        result = {
            "correlation": obs_corr,
            "p": p_value,
            "parallel": None,
        }

        if return_null:
            result["null_dist"] = null_dist

        return result


# ============================================================================
# Matrix Utility Functions (moved from nltools.stats)
# ============================================================================


def double_center(mat: np.ndarray) -> np.ndarray:
    """Double center a 2d array.

    Double-centering subtracts row means, column means, and adds the grand mean.
    This centers both rows and columns around zero.

    Args:
        mat (ndarray): 2d numpy array

    Returns:
        mat (ndarray): double-centered version of input

    Raises:
        ValueError: If input is not 2D

    Examples:
        >>> mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        >>> result = double_center(mat)
        >>> np.allclose(result.mean(axis=0), 0)
        True
        >>> np.allclose(result.mean(axis=1), 0)
        True
    """
    if len(mat.shape) != 2:
        raise ValueError("Array should be 2d")

    # keepdims ensures that row/column means are not incorrectly broadcast during subtraction
    row_mean = mat.mean(axis=0, keepdims=True)
    col_mean = mat.mean(axis=1, keepdims=True)
    grand_mean = mat.mean()
    return mat - row_mean - col_mean + grand_mean


def u_center(mat: np.ndarray) -> np.ndarray:
    """U-center a 2d array. U-centering is a bias-corrected form of double-centering.

    U-centering corrects for bias that occurs with double-centering as the number
    of dimensions increases. The diagonal is explicitly set to zero.

    Args:
        mat (ndarray): 2d numpy array

    Returns:
        mat (ndarray): u-centered version of input

    Raises:
        ValueError: If input is not 2D

    Examples:
        >>> mat = np.random.randn(5, 5)
        >>> result = u_center(mat)
        >>> np.allclose(np.diag(result), 0)
        True
    """
    if len(mat.shape) != 2:
        raise ValueError("Array should be 2d")

    dim = mat.shape[0]
    u_mu = mat.sum() / ((dim - 1) * (dim - 2))
    sum_cols = mat.sum(axis=0, keepdims=True)
    sum_rows = mat.sum(axis=1, keepdims=True)
    u_mu_cols = np.ones((dim, 1)).dot(sum_cols / (dim - 2))
    u_mu_rows = (sum_rows / (dim - 2)).dot(np.ones((1, dim)))
    out = np.copy(mat)
    # Do one operation at a time, to improve broadcasting memory usage.
    out -= u_mu_rows
    out -= u_mu_cols
    out += u_mu
    # The diagonal is zero
    out[np.eye(dim, dtype=bool)] = 0
    return out


def distance_correlation(
    x: np.ndarray,
    y: np.ndarray,
    bias_corrected: bool = True,
    ttest: bool = False,
) -> dict:
    """
    Compute the distance correlation between 2 arrays to test for multivariate dependence (linear or non-linear).

    Arrays must match on their first dimension. It's almost always preferable to compute the bias_corrected
    version which can also optionally perform a ttest. This ttest operates on a statistic thats ~dcorr^2
    and will be also returned.

    Explanation:
    Distance correlation involves computing the normalized covariance of two centered euclidean distance
    matrices. Each distance matrix is the euclidean distance between rows (if x or y are 2d) or scalars
    (if x or y are 1d). Each matrix is centered prior to computing the covariance either using double-centering
    or u-centering, which corrects for bias as the number of dimensions increases. U-centering is almost always
    preferred in all cases. It also permits inference of the normalized covariance between each distance matrix
    using a one-tailed directional t-test. (Szekely & Rizzo, 2013). While distance correlation is normally
    bounded between 0 and 1, u-centering can produce negative estimates, which are never significant.

    Validated against the dcor and dcor.ttest functions in the 'energy' R package and the
    dcor.distance_correlation, dcor.udistance_correlation_sqr, and dcor.independence.distance_correlation_t_test
    functions in the dcor Python package.

    Args:
        x (ndarray): 1d or 2d numpy array of observations by features
        y (ndarray): 1d or 2d numpy array of observations by features
        bias_corrected (bool): if false use double-centering which produces a biased-estimate that converges
            to 1 as the number of dimensions increase. Otherwise used u-centering to correct this bias.
            **Note** this must be True if ttest=True; default True
        ttest (bool): perform a ttest using the bias_corrected distance correlation; default False

    Returns:
        results (dict): dictionary of results (correlation, t, p, and df.) Optionally, covariance,
            x variance, and y variance

    Raises:
        ValueError: If arrays are not 1d or 2d, or if ttest=True and bias_corrected=False

    Examples:
        >>> import numpy as np
        >>> x = np.random.randn(20, 3)
        >>> y = x + np.random.randn(20, 3) * 0.1  # Strongly correlated
        >>> result = distance_correlation(x, y, bias_corrected=True)
        >>> 'dcorr' in result
        True
        >>> 0 <= result['dcorr'] <= 1
        True
    """
    if len(x.shape) > 2 or len(y.shape) > 2:
        raise ValueError("Both arrays must be 1d or 2d")

    if (not bias_corrected) and ttest:
        raise ValueError("bias_corrected must be true to perform ttest!")

    # 1 compute euclidean distances between pairs of value in each array
    if len(x.shape) == 1:
        _x = x[:, np.newaxis]
    else:
        _x = x
    if len(y.shape) == 1:
        _y = y[:, np.newaxis]
    else:
        _y = y

    x_dist = squareform(pdist(_x))
    y_dist = squareform(pdist(_y))

    # 2 center each matrix
    if bias_corrected:
        # U-centering
        x_dist_cent = u_center(x_dist)
        y_dist_cent = u_center(y_dist)
        # Compute covariances using N*(N-3) in denominator
        adjusted_n = _x.shape[0] * (_x.shape[0] - 3)
        xy = np.multiply(x_dist_cent, y_dist_cent).sum() / adjusted_n
        xx = np.multiply(x_dist_cent, x_dist_cent).sum() / adjusted_n
        yy = np.multiply(y_dist_cent, y_dist_cent).sum() / adjusted_n
    else:
        # double-centering
        x_dist_cent = double_center(x_dist)
        y_dist_cent = double_center(y_dist)
        # Compute covariances using N^2 in denominator
        xy = np.multiply(x_dist_cent, y_dist_cent).mean()
        xx = np.multiply(x_dist_cent, x_dist_cent).mean()
        yy = np.multiply(y_dist_cent, y_dist_cent).mean()

    # 3 Normalize to get correlation
    denom = np.sqrt(xx * yy)
    dcor = xy / denom
    out = {}

    if dcor < 0:
        # This will only apply in the bias_corrected case as values can be < 0
        out["dcorr"] = 0
    else:
        out["dcorr"] = np.sqrt(dcor)
    if bias_corrected:
        out["dcorr_squared"] = dcor
    if ttest:
        dof = (adjusted_n / 2) - 1
        t = np.sqrt(dof) * (dcor / np.sqrt(1 - dcor**2))
        p = 1 - t_dist.cdf(t, dof)
        out["t"] = t
        out["p"] = p
        out["df"] = dof

    return out
