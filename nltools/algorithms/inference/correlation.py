"""Permutation test for the correlation between two variables.

`correlation_permutation_test` asks whether the Pearson, Spearman, or Kendall
correlation between two arrays differs from zero, building the null distribution
by shuffling one array's observations. It assumes observations are independent;
for autocorrelated time series use `timeseries_correlation_permutation_test`.
Multi-feature inputs (2D arrays) test each column pair independently.
Permutations run on joblib workers; `n_jobs` sets how many, and a given
`random_state` gives the same result at any worker count.
"""

import numpy as np
from collections.abc import Callable
from scipy.stats import rankdata, kendalltau
from sklearn.utils import check_random_state

from .utils import EPSILON, maybe_tqdm
from ..validation import _compute_pvalue, validate_tail_parameter


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray | float:
    """Compute Pearson correlation coefficient(s).

    Args:
        x (np.ndarray): Data array, shape (n_samples,) or (n_permute, n_samples).
        y (np.ndarray): Data array, shape (n_samples,).

    Returns:
        float | np.ndarray: A scalar if `x` is 1D, else one correlation per row
            of `x`, shape (n_permute,).

    Note:
        Centers the data before computing, and vectorizes across the rows of a
        2D `x`.
    """
    # Handle dimensions
    if x.ndim == 1:
        x = x[np.newaxis, :]  # (1, n_samples)
        squeeze_output = True
    else:
        squeeze_output = False

    # Center data
    x_centered = x - x.mean(axis=1, keepdims=True)  # (n_permute, n_samples)
    y_centered = y - y.mean()  # (n_samples,)

    # Compute correlation
    numerator = (x_centered @ y_centered) / x.shape[1]
    denominator = x_centered.std(axis=1, ddof=0) * y_centered.std(ddof=0)

    # Handle division by zero (constant data) using EPSILON
    correlations = numerator / (denominator + EPSILON)

    if squeeze_output:
        return float(correlations[0])
    return correlations


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray | float:
    """Compute Spearman rank correlation coefficient(s).

    Spearman correlation is the Pearson correlation of the rank-transformed data.
    It measures monotonic (not necessarily linear) relationships.

    Args:
        x (np.ndarray): Data array, shape (n_samples,) or (n_permute, n_samples).
        y (np.ndarray): Data array, shape (n_samples,).

    Returns:
        float | np.ndarray: A scalar if `x` is 1D, else one correlation per row
            of `x`, shape (n_permute,).

    Note:
        Ranks with `scipy.stats.rankdata` (ties get their average rank), then
        applies the Pearson correlation to the ranks.
    """
    # Handle dimensions
    if x.ndim == 1:
        x = x[np.newaxis, :]  # (1, n_samples)
        squeeze_output = True
    else:
        squeeze_output = False

    n_permute, n_samples = x.shape

    # Rank-transform data (average method for tied ranks)
    # For vectorized case, rank each permutation separately
    x_ranked = np.empty_like(x)
    for i in range(n_permute):
        x_ranked[i] = rankdata(x[i], method="average")

    y_ranked = rankdata(y, method="average")

    # Apply Pearson correlation to ranks
    # Center ranked data
    x_centered = x_ranked - x_ranked.mean(axis=1, keepdims=True)
    y_centered = y_ranked - y_ranked.mean()

    # Compute correlation
    numerator = (x_centered @ y_centered) / n_samples
    denominator = x_centered.std(axis=1, ddof=0) * y_centered.std(ddof=0)

    # Handle division by zero (constant data - all tied ranks) using EPSILON
    correlations = numerator / (denominator + EPSILON)

    if squeeze_output:
        return float(correlations[0])
    return correlations


def _kendall_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray | float:
    """Compute Kendall rank correlation coefficient(s).

    Kendall tau correlation measures ordinal association based on concordant
    and discordant pairs. More robust than Spearman for small samples or
    data with many tied ranks.

    Args:
        x (np.ndarray): Data array, shape (n_samples,) or (n_permute, n_samples).
        y (np.ndarray): Data array, shape (n_samples,).

    Returns:
        float | np.ndarray: A scalar if `x` is 1D, else one correlation per row
            of `x`, shape (n_permute,). A NaN tau (constant input) is returned
            as 0.0.

    Note:
        Calls `scipy.stats.kendalltau` once per row; O(n²) per call, so slower
        than Pearson or Spearman.
    """
    # Handle dimensions
    if x.ndim == 1:
        # Single correlation
        tau, _ = kendalltau(x, y)
        return float(tau) if not np.isnan(tau) else 0.0
    # Vectorized: compute each permutation separately
    n_permute = x.shape[0]
    correlations = np.empty(n_permute)
    for i in range(n_permute):
        tau, _ = kendalltau(x[i], y)
        correlations[i] = tau if not np.isnan(tau) else 0.0
    return correlations


def _select_corr_func(
    metric: str,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray | float]:
    """Return the correlation function for a metric name."""
    if metric == "pearson":
        return _pearson_correlation
    if metric == "spearman":
        return _spearman_correlation
    if metric == "kendall":
        return _kendall_correlation
    raise NotImplementedError(f"Metric '{metric}' not yet implemented")


def _correlation_permutation_cpu_parallel(
    data1: np.ndarray,
    data2: np.ndarray,
    *,
    n_permute: int,
    metric: str,
    tail: int,
    return_null: bool,
    n_jobs: int,
    random_state: int | None,
    single_feature: bool = False,
    progress_bar: bool = False,
) -> dict:
    """Correlation permutation test parallelized across CPU workers with joblib.

    Each worker handles one permutation: shuffle `data1`, correlate with `data2`.
    Seeds are pre-generated from `random_state`, so results do not depend on
    `n_jobs`.

    Args:
        data1 (np.ndarray): Data to permute, shape (n_samples, n_features).
        data2 (np.ndarray): Data to correlate with, shape (n_samples, n_features).
        n_permute (int): Number of permutations.
        metric (str): Correlation metric, one of 'pearson', 'spearman', or 'kendall'.
        tail (int | str): `2` or `'two'` for two-tailed; `1` or `'one'` for one-tailed.
        return_null (bool): Whether to return the null distribution.
        n_jobs (int): Number of parallel workers (-1 = all cores).
        random_state (int | None): Random seed for reproducibility.
        single_feature (bool): Whether the caller passed 1D inputs (results are
            returned as scalars).
        progress_bar (bool): Show a progress bar over permutations.

    Returns:
        dict: Same keys as `correlation_permutation_test`.
    """
    from joblib import Parallel, delayed

    # Setup random state and generate seeds for workers
    rng = check_random_state(random_state)
    MAX_INT = 2**31 - 1
    seeds = rng.randint(MAX_INT, size=n_permute)

    # Get dimensions (data already reshaped by caller)
    n_samples, n_features = data1.shape

    # Select correlation function
    corr_func = _select_corr_func(metric)

    # Compute observed correlation
    if n_features == 1:
        obs_corr = corr_func(data1[:, 0], data2[:, 0])
        obs_corr = np.array([obs_corr])
    else:
        obs_corr = np.array(
            [corr_func(data1[:, i], data2[:, i]) for i in range(n_features)]
        )

    # Define worker function (each processes ONE permutation)
    def _compute_one_perm(seed):
        """Compute correlation for one permutation."""
        perm_rng = np.random.RandomState(seed)
        # Permute data1 indices
        indices = perm_rng.permutation(n_samples)
        perm_data1 = data1[indices]

        # Compute correlation for each feature
        if n_features == 1:
            return corr_func(perm_data1[:, 0], data2[:, 0])
        return np.array(
            [corr_func(perm_data1[:, i], data2[:, i]) for i in range(n_features)]
        )

    # Execute in parallel with progress bar
    null_dist = Parallel(n_jobs=n_jobs)(
        delayed(_compute_one_perm)(seeds[i])
        for i in maybe_tqdm(
            range(n_permute),
            progress_bar=progress_bar,
            desc="CPU parallel perms",
            unit="perm",
        )
    )
    null_dist = np.array(null_dist)  # Shape: (n_permute, n_features)

    # Compute p-values
    p_values = _compute_pvalue(obs_corr, null_dist, tail=tail)

    # Return to original shape
    if single_feature:
        obs_corr = obs_corr.item() if hasattr(obs_corr, "item") else float(obs_corr[0])
        p_values = p_values.item() if hasattr(p_values, "item") else float(p_values[0])

    # Build result
    result = {
        "correlation": obs_corr,
        "p": p_values,
    }

    if return_null:
        if single_feature:
            null_dist = null_dist.squeeze()
        result["null_dist"] = null_dist

    return result


def correlation_permutation_test(
    data1: np.ndarray,
    data2: np.ndarray,
    *,
    n_permute: int = 5000,
    metric: str = "pearson",
    tail: int | str = 2,
    return_null: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
    progress_bar: bool = False,
) -> dict:
    """Permutation test for whether the correlation between two arrays differs from zero.

    Builds the null distribution by randomly permuting the observations of `data1`
    and re-correlating with `data2`. Assumes observations are independent (i.i.d.);
    for autocorrelated time series use `timeseries_correlation_permutation_test`,
    whose `'circle_shift'` and `'phase_randomize'` methods preserve temporal
    structure. With 2D inputs each column of `data1` is tested against the
    matching column of `data2`, independently.

    Args:
        data1 (np.ndarray): Data to permute, shape (n_samples,) for a single
            feature or (n_samples, n_features) for several.
        data2 (np.ndarray): Data to correlate with, same shape as `data1`.
        n_permute (int): Number of permutations. Defaults to 5000.
        metric (str): 'pearson' (linear), 'spearman' (rank-based, monotonic), or
            'kendall' (tau-b, ordinal association, tie-corrected). Defaults to
            'pearson'.
        tail (int | str): `2` or `'two'` for a two-tailed test (r != 0); `1` or
            `'one'` for a one-tailed test of r > 0 (negate one variable for the
            other direction; the fixed direction keeps multiple-comparison
            correction valid). Defaults to 2.
        return_null (bool): Also return the full null distribution. Defaults to
            False.
        n_jobs (int): Number of joblib workers, -1 = all cores. Defaults to -1.
            Results are identical at every worker count.
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Show a progress bar over permutations. Defaults to
            False.

    Returns:
        dict: Keys 'correlation' (float, or np.ndarray of shape (n_features,) for
            2D inputs: the observed correlation), 'p' (float or np.ndarray, the
            matching p-values), and 'null_dist' (np.ndarray of shape
            (n_permute,) or (n_permute, n_features)) when `return_null=True`.

    Examples:
        ```python
        import numpy as np
        from nltools.algorithms import correlation_permutation_test

        # Single feature
        x = np.random.randn(100)
        y = x + np.random.randn(100) * 0.5
        result = correlation_permutation_test(x, y, n_permute=5000)
        result["correlation"]  # → 0.85 (approximately)
        result["p"]  # → 0.0002

        # Multi-feature: each column pair tested independently
        data1 = np.random.randn(100, 10)
        data2 = data1 + np.random.randn(100, 10) * 0.3
        result = correlation_permutation_test(data1, data2, n_permute=5000)
        result["correlation"].shape  # → (10,)
        result["p"].shape  # → (10,)
        ```

    Note:
        Kendall's tau is O(n²) in the number of samples, so it is markedly
        slower than Pearson or Spearman for large samples.
    """
    # Input validation
    data1 = np.asarray(data1, dtype=np.float64)
    data2 = np.asarray(data2, dtype=np.float64)

    if data1.ndim not in [1, 2]:
        raise ValueError(f"data1 must be 1D or 2D, got shape {data1.shape}")
    if data2.ndim not in [1, 2]:
        raise ValueError(f"data2 must be 1D or 2D, got shape {data2.shape}")
    tail = validate_tail_parameter(tail)
    if metric not in ["pearson", "spearman", "kendall"]:
        raise ValueError(
            f"metric must be 'pearson', 'spearman', or 'kendall', got '{metric}'"
        )

    # Handle shape
    single_feature = data1.ndim == 1 and data2.ndim == 1
    if data1.ndim == 1:
        data1 = data1[:, np.newaxis]
    if data2.ndim == 1:
        data2 = data2[:, np.newaxis]

    # Check dimensions match
    if data1.shape != data2.shape:
        raise ValueError(
            f"data1 and data2 must have same shape, got {data1.shape} and {data2.shape}"
        )

    return _correlation_permutation_cpu_parallel(
        data1,
        data2,
        n_permute=n_permute,
        metric=metric,
        tail=tail,
        return_null=return_null,
        n_jobs=n_jobs,
        random_state=random_state,
        single_feature=single_feature,
        progress_bar=progress_bar,
    )
