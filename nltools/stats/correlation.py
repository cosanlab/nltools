"""Similarity metrics and correlation."""

import itertools

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, t as t_dist

__all__ = [
    "compute_multivariate_similarity",
    "compute_similarity",
    "fisher_r_to_z",
    "fisher_z_to_r",
    "transform_pairwise",
]


def fisher_r_to_z(r):
    """Use Fisher transformation to convert correlation to z score.

    Args:
        r: correlation coefficient(s)

    Returns:
        z: Fisher z-transformed correlation(s)

    Note:
        Clips r values to (-1, 1) range to avoid invalid arctanh inputs
    """
    # Clip r to valid range for arctanh to avoid invalid value warnings
    with np.errstate(invalid="ignore"):
        return np.arctanh(r)


def fisher_z_to_r(z):
    """Use Fisher transformation to convert correlation to z score."""
    return np.tanh(z)


def transform_pairwise(X, y):
    """Transform data into pairs with balanced labels for ranking.

    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.
    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1

    Reference: "Large Margin Rank Boundaries for Ordinal Regression",
    R. Herbrich, T. Graepel, K. Obermayer. Authors: Fabian Pedregosa
    <fabian@fseoane.net> Alexandre Gramfort <alexandre.gramfort@inria.fr>

    Args:
        X: (np.array), shape (n_samples, n_features)
            The data
        y: (np.array), shape (n_samples,) or (n_samples, 2)
            Target labels. If it's a 2D array, the second column represents
            the grouping of samples, i.e., samples with different groups will
            not be considered.

    Returns:
        X_trans: (np.array), shape (k, n_feaures)
            Data as pairs, where k = n_samples * (n_samples-1)) / 2 if grouping
            values were not passed. If grouping variables exist, then returns
            values computed for each group.
        y_trans: (np.array), shape (k,)
            Output class labels, where classes have values {-1, +1}
            If y was shape (n_samples, 2), then returns (k, 2) with groups on
            the second dimension.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    y_ndim = y.ndim
    if y_ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]

    # Pre-allocate lists (more efficient than repeated appends)
    X_new = []
    y_new = []
    y_group = []

    # Use itertools.combinations (necessary for pairwise combinations)
    # Optimize by using pre-allocated lists and vectorized sign operation
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        sign_val = np.sign(y[i, 0] - y[j, 0])
        # output balanced classes
        if sign_val != (-1) ** k:
            sign_val = -sign_val
            X_new[-1] = -X_new[-1]
        y_new.append(sign_val)
        y_group.append(y[i, 1])

    # Convert to arrays efficiently
    if len(X_new) == 0:
        if y_ndim == 1:
            return np.array([]).reshape(0, X.shape[1]), np.array([])
        return np.array([]).reshape(0, X.shape[1]), np.array([]).reshape(0, 2)

    X_trans = np.array(X_new)
    if y_ndim == 1:
        return X_trans, np.array(y_new)
    return X_trans, np.column_stack([np.array(y_new), np.array(y_group)])


def compute_similarity(data1, data2, method="correlation"):
    """Compute similarity between two data arrays.

    This is the functional core implementation for similarity computation.
    Used by BrainData.similarity() to delegate computation to the functional core.

    Args:
        data1 (np.ndarray): First data array, shape (n_samples1, n_features)
        data2 (np.ndarray): Second data array, shape (n_samples2, n_features)
        method (str): Type of similarity metric
            - 'correlation' or 'pearson': Pearson correlation
            - 'spearman' or 'rank_correlation': Spearman rank correlation
            - 'dot_product': Dot product
            - 'cosine': Cosine similarity

    Returns:
        np.ndarray: Similarity matrix or vector
            - If data1.shape[0] == 1 and data2.shape[0] == 1: scalar
            - If data1.shape[0] == 1 or data2.shape[0] == 1: 1D array
            - Otherwise: 2D array shape (n_samples1, n_samples2)

    Examples:
        >>> data1 = np.random.randn(10, 100)
        >>> data2 = np.random.randn(5, 100)
        >>> sim = compute_similarity(data1, data2, method='correlation')
        >>> sim.shape
        (10, 5)
    """
    # Ensure 2D arrays
    data1 = np.atleast_2d(data1)
    data2 = np.atleast_2d(data2)

    if method == "dot_product":
        # Vectorized dot product
        if data2.shape[0] == 1:
            out = np.dot(data1, data2.T).squeeze()
        else:
            out = np.dot(data1, data2.T)
    elif method in ["pearson", "correlation"]:
        # Use np.corrcoef (BLAS-optimized) for Pearson correlation
        stacked = np.vstack([data1, data2])
        corr_matrix = np.corrcoef(stacked)
        n_data1 = data1.shape[0]
        n_data2 = data2.shape[0]
        # Extract correlations between data1 rows and data2 rows
        out = corr_matrix[:n_data1, n_data1 : n_data1 + n_data2]
        out = out.squeeze()
    elif method in ["spearman", "rank_correlation"]:
        # Spearman correlation: rank-transform then use np.corrcoef
        data1_ranked = np.apply_along_axis(rankdata, axis=1, arr=data1)
        data2_ranked = np.apply_along_axis(rankdata, axis=1, arr=data2)
        stacked = np.vstack([data1_ranked, data2_ranked])
        corr_matrix = np.corrcoef(stacked)
        n_data1 = data1.shape[0]
        n_data2 = data2.shape[0]
        out = corr_matrix[:n_data1, n_data1 : n_data1 + n_data2]
        out = out.squeeze()
    elif method == "cosine":
        # Use cdist with cosine metric, then convert distance to similarity
        out = cdist(data1, data2, metric="cosine").squeeze()
        out = 1 - out  # Convert distance to similarity
    else:
        raise ValueError(
            f"method must be one of ['correlation', 'pearson', 'spearman', "
            f"'rank_correlation', 'dot_product', 'cosine'], got '{method}'"
        )

    return out


def compute_multivariate_similarity(y, X, method="ols"):
    """Compute multivariate similarity via OLS regression.

    This is the functional core implementation for multivariate similarity computation.
    Used by BrainData.multivariate_similarity() to delegate computation to the functional core.

    Predicts spatial distribution of y from linear combination of X columns.
    Computes OLS regression statistics including beta coefficients, t-statistics,
    p-values, and residuals.

    Args:
        y (np.ndarray): Target data, shape (n_features,) - single image
        X (np.ndarray): Predictor data, shape (n_features, n_predictors) where first column
            should be intercept (ones) if intercept is desired. If X does not include intercept,
            an intercept will be added automatically.
        method (str): Regression method (currently only 'ols' supported)

    Returns:
        dict: Dictionary with keys:
            - 'beta': Regression coefficients including intercept, shape (n_predictors+1,)
            - 't': t-statistics, shape (n_predictors+1,)
            - 'p': p-values, shape (n_predictors+1,)
            - 'df': Degrees of freedom (int)
            - 'sigma': Residual standard deviation (float)
            - 'residual': Residuals, shape (n_features,)

    Examples:
        >>> y = np.random.randn(100)
        >>> X = np.random.randn(100, 5)
        >>> result = compute_multivariate_similarity(y, X, method='ols')
        >>> 'beta' in result
        True
        >>> result['beta'].shape
        (6,)  # 5 predictors + intercept
    """
    if method != "ols":
        raise NotImplementedError(f"method '{method}' not implemented")

    # Ensure y is 1D
    y = np.atleast_1d(y)
    if y.ndim > 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")

    # Ensure X is 2D: (n_features, n_predictors)
    X = np.atleast_2d(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")

    # Check if X needs to be transposed (handle both orientations)
    if X.shape[0] == y.shape[0]:
        # X is (n_features, n_predictors) - correct orientation
        pass
    elif X.shape[1] == y.shape[0]:
        # X is (n_predictors, n_features) - transpose needed
        X = X.T
    else:
        raise ValueError(
            f"X must have shape (n_features, n_predictors) or (n_predictors, n_features), "
            f"where n_features={y.shape[0]}, got shape {X.shape}"
        )

    # Add intercept (first column)
    X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X])

    # OLS regression: b = (X'X)^(-1) X'y
    b = np.dot(np.linalg.pinv(X_with_intercept), y)
    res = y - np.dot(X_with_intercept, b)
    # Unbiased estimator of residual standard error: sqrt(RSS / df)
    # This is correct for both intercept and intercept-free models
    # See GH #287 for details on why np.std(res, ddof=p) is biased
    n, p = X_with_intercept.shape
    sigma = np.sqrt(np.dot(res, res) / (n - p))

    # Compute standard errors
    XtX_inv = np.linalg.inv(np.dot(X_with_intercept.T, X_with_intercept))
    se_diag = np.sqrt(np.diagonal(XtX_inv))
    stderr = se_diag * sigma

    # t-statistics
    t_out = b / stderr

    # Degrees of freedom
    df = X_with_intercept.shape[0] - X_with_intercept.shape[1]

    # p-values (two-tailed)
    p = 2 * (1 - t_dist.cdf(np.abs(t_out), df))

    return {
        "beta": b,
        "t": t_out,
        "p": p,
        "df": df,
        "sigma": sigma,
        "residual": res,
    }
