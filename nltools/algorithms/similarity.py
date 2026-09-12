"""Similarity metrics and correlation."""

import itertools

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from .regression import regress

__all__ = [
    "compute_multivariate_similarity",
    "compute_similarity",
    "fisher_r_to_z",
    "fisher_z_to_r",
    "transform_pairwise",
]


def fisher_r_to_z(r):
    """Convert correlation coefficients to Fisher z values.

    Args:
        r (float | np.ndarray): Correlation coefficient(s).

    Returns:
        np.ndarray: Fisher z-transformed correlation(s).
    """
    with np.errstate(invalid="ignore"):
        return np.arctanh(r)


def fisher_z_to_r(z):
    """Convert Fisher z back to a correlation coefficient.

    Args:
        z (float | np.ndarray): Fisher z value(s).

    Returns:
        np.ndarray: Correlation coefficient(s).
    """
    return np.tanh(z)


# Adapted from the scikit-learn RankSVM example by Fabian Pedregosa and
# Alexandre Gramfort (BSD licensed).
def transform_pairwise(X, y):
    """Transform data into pairwise differences with balanced labels for ranking.

    Turns an n-class ranking problem into a two-class classification problem:
    every pair of samples with different target values becomes one difference
    row, and signs are flipped so that the -1 and +1 classes are balanced.

    Reference: Herbrich, R., Graepel, T., & Obermayer, K. "Large Margin Rank
    Boundaries for Ordinal Regression".

    Args:
        X (np.ndarray): Data, shape (n_samples, n_features).
        y (np.ndarray): Target labels, shape (n_samples,) or (n_samples, 2). A
            second column groups the samples; pairs from different groups are skipped.

    Returns:
        tuple[np.ndarray, np.ndarray]: `(X_trans, y_trans)`. `X_trans` has shape
            (k, n_features) with one row per retained pair (k is at most
            n_samples * (n_samples - 1) / 2; pairs are formed within groups when
            given). `y_trans` holds the labels in {-1, +1}, shape (k,), or (k, 2)
            with the group in the second column when `y` had two columns.
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


def compute_similarity(data1, data2, metric="correlation"):
    """Compute row-wise similarity between two data arrays.

    The array engine behind `BrainData.similarity`.

    Args:
        data1 (np.ndarray): First data array, shape (n_samples1, n_features).
        data2 (np.ndarray): Second data array, shape (n_samples2, n_features).
        metric (str): 'correlation' (or 'pearson'), 'spearman' (or
            'rank_correlation'), 'dot_product', or 'cosine'. Defaults to 'correlation'.

    Returns:
        np.ndarray: Similarities of shape (n_samples1, n_samples2), squeezed: a 1D
            array when either input has a single row, a scalar when both do.

    Examples:
        ```python
        data1 = np.random.randn(10, 100)
        data2 = np.random.randn(5, 100)
        sim = compute_similarity(data1, data2, metric="correlation")
        sim.shape  # → (10, 5)
        ```
    """
    # Ensure 2D arrays
    data1 = np.atleast_2d(data1)
    data2 = np.atleast_2d(data2)

    if metric == "dot_product":
        # Vectorized dot product
        if data2.shape[0] == 1:
            out = np.dot(data1, data2.T).squeeze()
        else:
            out = np.dot(data1, data2.T)
    elif metric in ["pearson", "correlation"]:
        # Use np.corrcoef (BLAS-optimized) for Pearson correlation
        stacked = np.vstack([data1, data2])
        corr_matrix = np.corrcoef(stacked)
        n_data1 = data1.shape[0]
        n_data2 = data2.shape[0]
        # Extract correlations between data1 rows and data2 rows
        out = corr_matrix[:n_data1, n_data1 : n_data1 + n_data2]
        out = out.squeeze()
    elif metric in ["spearman", "rank_correlation"]:
        # Spearman correlation: rank-transform then use np.corrcoef
        data1_ranked = np.apply_along_axis(rankdata, axis=1, arr=data1)
        data2_ranked = np.apply_along_axis(rankdata, axis=1, arr=data2)
        stacked = np.vstack([data1_ranked, data2_ranked])
        corr_matrix = np.corrcoef(stacked)
        n_data1 = data1.shape[0]
        n_data2 = data2.shape[0]
        out = corr_matrix[:n_data1, n_data1 : n_data1 + n_data2]
        out = out.squeeze()
    elif metric == "cosine":
        # Use cdist with cosine metric, then convert distance to similarity
        out = cdist(data1, data2, metric="cosine").squeeze()
        out = 1 - out  # Convert distance to similarity
    else:
        raise ValueError(
            f"metric must be one of ['correlation', 'pearson', 'spearman', "
            f"'rank_correlation', 'dot_product', 'cosine'], got '{metric}'"
        )

    return out


def compute_multivariate_similarity(y, X, tail=2):
    """Compute multivariate similarity by regressing one pattern on several.

    The array engine behind `BrainData.multivariate_similarity`: predicts the
    spatial pattern `y` from a linear combination of the columns of `X` and
    returns the OLS coefficients, t-statistics, p-values, and residuals.

    Args:
        y (np.ndarray): Target pattern, shape (n_features,).
        X (np.ndarray): Predictor patterns, shape (n_features, n_predictors) (the
            transpose is accepted). An intercept column is always prepended, so
            do not include one.
        tail (int): 2 for two-sided p-values, 1 for an upper-tail test. Defaults to 2.

    Returns:
        dict: Keys 'beta' (coefficients, intercept first, shape (n_predictors + 1,)),
            't' (t-statistics, same shape), 'p' (p-values, same shape), 'df'
            (residual degrees of freedom), 'sigma' (residual standard deviation),
            and 'residual' (residuals, shape (n_features,)).

    Examples:
        ```python
        y = np.random.randn(100)
        X = np.random.randn(100, 5)
        result = compute_multivariate_similarity(y, X)
        result["beta"].shape  # → (6,)  5 predictors + intercept
        ```
    """
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

    b, _, t_out, p, _, res = regress(X_with_intercept, y, tail=tail)

    n, p_cols = X_with_intercept.shape
    df = n - p_cols
    # Unbiased estimator of residual standard error: sqrt(RSS / df); correct for
    # both intercept and intercept-free models. See GH #287.
    sigma = float(np.sqrt(np.dot(res, res) / df))

    return {
        "beta": b,
        "t": t_out,
        "p": p,
        "df": df,
        "sigma": sigma,
        "residual": res,
    }
