"""Permutation tests and dependence measures for square matrices.

`matrix_permutation_test` is the Mantel test: it asks whether two square matrices
(e.g. two representational dissimilarity matrices, or a brain and a model
similarity matrix) are correlated, building the null by permuting the rows and
columns of one matrix together. `distance_correlation` measures multivariate
dependence (linear or not) between two arrays, with `_double_center` and
`_u_center` as the centering steps it is built on. Permutations run on joblib
workers; `n_jobs` sets how many, and a given `random_state` gives the same
result at any worker count.

`_extract_triangle_elements` pulls the upper or lower triangle of a matrix into a
vector; `_permute_matrix_symmetric` reorders rows and columns together, the
operation at the heart of the matrix permutation tests.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.spatial.distance import squareform, pdist
from scipy.stats import t as t_dist

from .utils import _maybe_tqdm
from .validation import (
    _validate_how_parameter,
    _validate_metric_parameter,
    _validate_same_shape,
    _validate_square_matrix,
)
from ..validation import _compute_pvalue, _validate_tail_parameter


# Maximum integer for random seed generation
MAX_INT = np.iinfo(np.int32).max


def _extract_triangle_elements(
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
        _extract_triangle_elements(matrix, triangle="upper")
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


def _permute_matrix_symmetric(
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
        _permute_matrix_symmetric(matrix, perm)
        # → array([[8, 6, 7],
        #          [2, 0, 1],
        #          [5, 3, 4]])
        ```
    """
    return matrix[permutation][:, permutation]


def _extract_matrix_elements(
    matrix: np.ndarray,
    how: str = "upper",
    include_diag: bool = False,
) -> np.ndarray:
    """Extract elements from a square matrix (wrapper for `_extract_triangle_elements`).

    Args:
        matrix (np.ndarray): Square matrix (n×n).
        how (str): Which elements to extract, one of 'upper', 'lower', or 'full'.
        include_diag (bool): Include the diagonal (only for 'full').

    Returns:
        np.ndarray: 1D array of extracted elements.
    """
    return _extract_triangle_elements(matrix, triangle=how, include_diag=include_diag)


def _compute_matrix_correlation(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    how: str = "upper",
    include_diag: bool = False,
    metric: str = "pearson",
) -> float:
    """Compute the correlation between the elements of two square matrices.

    Args:
        matrix1 (np.ndarray): First square matrix (n×n).
        matrix2 (np.ndarray): Second square matrix (n×n).
        how (str): Which elements to compare, one of 'upper', 'lower', or 'full'.
        include_diag (bool): Include the diagonal (only for `how='full'`).
        metric (str): Correlation type, one of 'pearson', 'spearman', or 'kendall'.

    Returns:
        float: Correlation coefficient.

    Examples:
        ```python
        m1 = np.eye(3)
        m2 = np.eye(3)
        _compute_matrix_correlation(m1, m2, metric="pearson")  # → 1.0
        ```
    """
    # Extract elements from both matrices
    elements1 = _extract_matrix_elements(matrix1, how=how, include_diag=include_diag)
    elements2 = _extract_matrix_elements(matrix2, how=how, include_diag=include_diag)

    # Compute correlation
    _validate_metric_parameter(
        metric, ["pearson", "spearman", "kendall"], name="metric"
    )
    if metric == "pearson":
        r, _ = pearsonr(elements1, elements2)
    elif metric == "spearman":
        r, _ = spearmanr(elements1, elements2)
    elif metric == "kendall":
        r, _ = kendalltau(elements1, elements2)

    return r


def _compute_cross_correlation(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """Compute cross-correlation between columns (features) of two matrices.

    This function computes the correlation between each column of matrix1
    with each column of matrix2. Useful for computing connectivity matrices
    such as in intersubject functional connectivity (ISFC).

    Args:
        matrix1 (np.ndarray): First matrix, shape (n_observations, n_features1).
        matrix2 (np.ndarray): Second matrix, shape (n_observations, n_features2).

    Returns:
        np.ndarray: Cross-correlation matrix, shape (n_features1, n_features2),
            where element [i, j] is the correlation between `matrix1[:, i]` and
            `matrix2[:, j]`.

    Examples:
        ```python
        matrix1 = np.random.randn(100, 5)  # 100 observations, 5 features
        matrix2 = np.random.randn(100, 3)  # 100 observations, 3 features
        corr = _compute_cross_correlation(matrix1, matrix2)
        corr.shape  # → (5, 3)
        ```

    Note:
        Computed as the off-diagonal block of `np.corrcoef` over the concatenated
        columns.
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
    *,
    n_permute: int,
    metric: str,
    how: str,
    include_diag: bool,
    tail: int,
    return_null: bool,
    n_jobs: int,
    random_state: int | None,
    progress_bar: bool = False,
) -> dict:
    """Matrix permutation test parallelized across CPU workers with joblib.

    Seeds are pre-generated from `random_state` and one permutation runs per seed,
    so results are identical regardless of `n_jobs`. Typical speedup is 4-8× on an
    8-core machine.

    Args:
        data1 (np.ndarray): First square matrix (n×n).
        data2 (np.ndarray): Second square matrix (n×n).
        n_permute (int): Number of permutations.
        metric (str): Correlation metric, one of 'pearson', 'spearman', or 'kendall'.
        how (str): Which elements to compare, one of 'upper', 'lower', or 'full'.
        include_diag (bool): Include the diagonal (only for `how='full'`).
        tail (int | str): `2` or `'two'` for two-tailed; `1` or `'one'` for one-tailed.
        return_null (bool): Whether to return the null distribution.
        n_jobs (int): Number of parallel workers (-1 = all cores).
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Show a progress bar over permutations.

    Returns:
        dict: Keys 'correlation' (float), 'p' (float), and 'null_dist'
            (np.ndarray) when `return_null=True`.
    """
    from joblib import Parallel, delayed

    # Validate inputs
    _validate_same_shape(data1, data2, name1="data1", name2="data2")
    _validate_square_matrix(data1, name="data1")
    _validate_square_matrix(data2, name="data2")

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
        for i in _maybe_tqdm(
            range(n_permute),
            progress_bar=progress_bar,
            desc="Matrix permutation",
            unit="perm",
        )
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
    }

    if return_null:
        result["null_dist"] = null_dist

    return result


def matrix_permutation_test(
    data1: np.ndarray,
    data2: np.ndarray,
    *,
    n_permute: int = 5000,
    metric: str = "pearson",
    how: str = "upper",
    include_diag: bool = False,
    tail: int | str = 2,
    return_null: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
    progress_bar: bool = False,
) -> dict:
    """Matrix permutation test (Mantel test) for correlating two square matrices.

    Tests whether the correlation between the elements of two matrices is
    significant by permuting the rows and columns of one matrix together
    (`data1[perm][:, perm]`) while keeping the other fixed. Each permutation
    preserves the matrix's structure (including symmetry) but destroys its
    relationship to `data2`; the p-value is the fraction of permuted correlations
    at least as extreme as the observed one. Assumes both matrices are square and
    the same size, and that row/column ordering is exchangeable under the null.

    Args:
        data1 (np.ndarray): First square matrix (n×n).
        data2 (np.ndarray): Second square matrix (n×n).
        n_permute (int): Number of permutations. Defaults to 5000.
        metric (str): Correlation metric, one of 'pearson', 'spearman', or
            'kendall'. Defaults to 'pearson'.
        how (str): Which elements to compare: 'upper' (upper triangle; assumes
            symmetric matrices), 'lower' (lower triangle), or 'full' (all elements,
            see `include_diag`). Defaults to 'upper'.
        include_diag (bool): Include diagonal elements (only when `how='full'`).
            Defaults to False.
        tail (int | str): `2` or `'two'` for a two-tailed test (r != 0); `1` or
            `'one'` for a one-tailed test of r > 0 (negate one matrix for the other
            direction). Defaults to 2.
        return_null (bool): Also return the null distribution. Defaults to False.
        n_jobs (int): Number of joblib workers, -1 = all cores. Defaults to -1.
            Results are identical at every worker count.
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Show a progress bar over permutations. Defaults to False.

    Returns:
        dict: Keys 'correlation' (float, observed correlation), 'p' (float,
            Phipson-Smyth corrected p-value), and 'null_dist' (np.ndarray) when
            `return_null=True`.

    References:
        Chen, G. et al. (2016). Untangling the relatedness among correlations,
        part I: nonparametric approaches to inter-subject correlation analysis
        at the group level. NeuroImage, 142, 248-259.

        Mantel, N. (1967). The detection of disease clustering and a generalized
        regression approach. Cancer Research, 27(2), 209-220.

    Examples:
        ```python
        import numpy as np
        from nltools.algorithms.inference import matrix_permutation_test

        # Two 20×20 similarity matrices sharing a common pattern
        rng = np.random.default_rng(42)
        pattern = rng.standard_normal((20, 10))
        data1 = np.corrcoef(pattern + rng.standard_normal((20, 10)) * 0.5)
        data2 = np.corrcoef(pattern + rng.standard_normal((20, 10)) * 0.5)

        result = matrix_permutation_test(data1, data2, n_permute=1000)
        print(f"Correlation: {result['correlation']:.3f}, p = {result['p']:.4f}")
        ```
    """
    # Input validation
    if not isinstance(data1, np.ndarray) or not isinstance(data2, np.ndarray):
        raise TypeError("data1 and data2 must be numpy arrays")

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    _validate_same_shape(data1, data2, name1="data1", name2="data2")
    _validate_square_matrix(data1, name="data1")
    _validate_square_matrix(data2, name="data2")
    _validate_metric_parameter(
        metric, ["pearson", "spearman", "kendall"], name="metric"
    )
    _validate_how_parameter(how)
    _validate_tail_parameter(tail)

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
        progress_bar=progress_bar,
    )


# ============================================================================
# Matrix Utility Functions (moved from nltools.algorithms)
# ============================================================================


def _double_center(mat: np.ndarray) -> np.ndarray:
    """Double center a 2d array.

    Double-centering subtracts row means, column means, and adds the grand mean.
    This centers both rows and columns around zero.

    Args:
        mat (np.ndarray): 2d numpy array.

    Returns:
        np.ndarray: Double-centered version of the input.

    Raises:
        ValueError: If input is not 2D.

    Examples:
        ```python
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        result = _double_center(mat)
        np.allclose(result.mean(axis=0), 0)  # → True
        np.allclose(result.mean(axis=1), 0)  # → True
        ```
    """
    if len(mat.shape) != 2:
        raise ValueError("Array should be 2d")

    # keepdims ensures that row/column means are not incorrectly broadcast during subtraction
    row_mean = mat.mean(axis=0, keepdims=True)
    col_mean = mat.mean(axis=1, keepdims=True)
    grand_mean = mat.mean()
    return mat - row_mean - col_mean + grand_mean


def _u_center(mat: np.ndarray) -> np.ndarray:
    """U-center a 2d array.

    U-centering is a bias-corrected form of double-centering: it corrects for the
    bias that grows with the number of dimensions under plain double-centering.
    The diagonal is explicitly set to zero.

    Args:
        mat (np.ndarray): 2d numpy array.

    Returns:
        np.ndarray: U-centered version of the input.

    Raises:
        ValueError: If input is not 2D.

    Examples:
        ```python
        mat = np.random.randn(5, 5)
        result = _u_center(mat)
        np.allclose(np.diag(result), 0)  # → True
        ```
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
    """Compute the distance correlation between two arrays to test for multivariate dependence.

    Distance correlation detects linear and non-linear dependence. The arrays must
    match on their first dimension. Prefer the bias-corrected version (the default),
    which can also perform a t-test; that test operates on a statistic that is
    approximately the squared distance correlation, which is also returned.

    Distance correlation is the normalized covariance of two centered Euclidean
    distance matrices. Each distance matrix holds the distances between rows (if x
    or y is 2d) or scalars (if 1d). Each matrix is centered before the covariance
    is computed, either by double-centering or by U-centering, which corrects the
    bias that grows with the number of dimensions. U-centering is almost always
    preferable and also permits a one-tailed directional t-test on the normalized
    covariance (Szekely & Rizzo, 2013). Distance correlation is normally bounded
    between 0 and 1, but U-centering can produce negative estimates, which are
    never significant.

    Validated against `dcor` and `dcor.ttest` in the R package *energy* and
    `dcor.distance_correlation`, `dcor.u_distance_correlation_sqr`, and
    `dcor.independence.distance_correlation_t_test` in the Python package *dcor*.

    Args:
        x (np.ndarray): 1d or 2d array of observations by features.
        y (np.ndarray): 1d or 2d array of observations by features.
        bias_corrected (bool): If True, U-center the distance matrices; if False,
            double-center them, which gives a biased estimate that converges to 1
            as the number of dimensions grows. Must be True when `ttest=True`.
            Defaults to True.
        ttest (bool): Perform a t-test on the bias-corrected distance correlation.
            Defaults to False.

    Returns:
        dict: Key 'dcorr' (float, distance correlation); with `bias_corrected=True`
            also 'dcorr_squared' (float, the U-centered statistic, which can be
            negative); with `ttest=True` also 't', 'p', and 'df'.

    Raises:
        ValueError: If arrays are not 1d or 2d, or if `ttest=True` and
            `bias_corrected=False`.

    Examples:
        ```python
        import numpy as np

        x = np.random.randn(20, 3)
        y = x + np.random.randn(20, 3) * 0.1  # strongly dependent
        result = distance_correlation(x, y, bias_corrected=True)
        "dcorr" in result  # → True
        0 <= result["dcorr"] <= 1  # → True
        ```
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
        x_dist_cent = _u_center(x_dist)
        y_dist_cent = _u_center(y_dist)
        # Compute covariances using N*(N-3) in denominator
        adjusted_n = _x.shape[0] * (_x.shape[0] - 3)
        xy = np.multiply(x_dist_cent, y_dist_cent).sum() / adjusted_n
        xx = np.multiply(x_dist_cent, x_dist_cent).sum() / adjusted_n
        yy = np.multiply(y_dist_cent, y_dist_cent).sum() / adjusted_n
    else:
        # double-centering
        x_dist_cent = _double_center(x_dist)
        y_dist_cent = _double_center(y_dist)
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
