"""Provide standalone modeling and inference functions for Adjacency matrices.

Each function takes an Adjacency instance as its first argument (`adj`).
"""

import numpy as np


def bootstrap(
    adj,
    stat,
    *,
    n_samples=5000,
    save_boots=False,
    percentiles=(2.5, 97.5),
    n_jobs=-1,
    random_state=None,
):
    """Bootstrap statistics using efficient online algorithms.

    Uses memory-efficient bootstrap infrastructure with CPU parallelization.
    Supports simple aggregation statistics (mean, std, median, sum, min, max).

    Args:
        adj: (Adjacency) Adjacency instance containing multiple matrices
        stat: (str) Statistic to bootstrap. Options:
            - Simple stats: 'mean', 'median', 'std', 'sum', 'min', 'max'
        n_samples: (int) Number of bootstrap iterations. Default: 5000
        save_boots: (bool) If True, store all bootstrap samples (memory intensive).
                   Default: False
        percentiles: (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5)
        n_jobs: (int) Number of CPU cores for parallelization. -1 means all CPUs.
        random_state: (int, optional) Random seed for reproducibility

    Returns:
        dict: Dictionary with keys: 'Z', 'p', 'mean', 'std', 'ci_lower', 'ci_upper'
              (all Adjacency objects). If save_boots=True, also includes 'samples'.

    Examples:
        >>> # Simple aggregation
        >>> boot = bootstrap(adj, stat='mean', n_samples=1000)
        >>> assert 'mean' in boot
        >>> assert isinstance(boot['mean'], Adjacency)
    """
    from nltools.algorithms.inference.bootstrap import (
        _bootstrap_simple_cpu_parallel,
    )

    # Validate stat parameter
    SIMPLE_STATS = ["mean", "median", "std", "sum", "min", "max"]
    if stat not in SIMPLE_STATS:
        raise ValueError(
            f"Unsupported stat '{stat}'. Supported simple stats: {SIMPLE_STATS}."
        )

    # Get data as numpy array
    # Adjacency.data shape: (n_matrices, n_features)
    data = adj.data  # Shape: (n_samples, n_features)

    # Route to bootstrap function
    result = _bootstrap_simple_cpu_parallel(
        data,
        method=stat,
        n_samples=n_samples,
        save_boots=save_boots,
        n_jobs=n_jobs,
        random_state=random_state,
        percentiles=percentiles,
    )

    # Convert result to Adjacency format
    return convert_bootstrap_results_to_adjacency(adj, result, save_boots=save_boots)


def convert_bootstrap_results_to_adjacency(adj, result, save_boots=False):
    """Convert bootstrap results dictionary to Adjacency format.

    Helper function to convert numpy arrays from bootstrap functions into
    Adjacency objects.

    Args:
        adj: (Adjacency) Adjacency instance (used for matrix_type metadata)
        result: (dict) Result dictionary from bootstrap function with keys:
                'mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper', and optionally 'samples'
        save_boots: (bool) If True, include 'samples' key in output

    Returns:
        dict: Dictionary with Adjacency objects for each statistic
    """
    from nltools.data.adjacency import Adjacency

    out = {}
    for key in ["mean", "std", "Z", "p", "ci_lower", "ci_upper"]:
        if key in result:
            # Convert numpy array to Adjacency
            # Result shape: (n_features,) for aggregated stats
            adj_data = result[key]
            if adj_data.ndim == 0:
                # Scalar - convert to 1D array
                adj_data = np.array([adj_data])
            elif adj_data.ndim == 1:
                # Already 1D - reshape to (1, n_features) for Adjacency
                adj_data = adj_data.reshape(1, -1)
            # adj_data is now (1, n_features)

            out[key] = Adjacency(
                data=adj_data,
                matrix_type=adj.matrix_type + "_flat",
            )

    if save_boots and "samples" in result:
        # Samples shape: (n_samples, n_features)
        out["samples"] = result["samples"]

    return out


def regress(adj, X, method="ols"):
    """Run a regression on an adjacency instance.
    You can decompose an adjacency instance with another adjacency instance.
    You can also decompose each pixel by passing a design_matrix instance.

    Args:
        adj: (Adjacency) Adjacency instance
        X: Design matrix can be an Adjacency or DesignMatrix instance
        method: type of regression (default: ols) - only 'ols' is currently supported

    Returns:
        stats: (dict) dictionary of stats outputs.
    """
    from nltools.data.adjacency import Adjacency
    from nltools.data.designmatrix import DesignMatrix
    from scipy.stats import t as t_dist

    if method != "ols":
        raise ValueError(
            "Only 'ols' method is currently supported for Adjacency.regress()"
        )

    stats = {}
    if isinstance(X, Adjacency):
        if X.n_nodes != adj.n_nodes:
            raise ValueError("Adjacency instances must be the same size.")
        # Convert to numpy arrays for regression
        X_data = X.data.T
        Y_data = adj.data

        # Ensure Y is 2D
        if len(Y_data.shape) == 1:
            Y_data = Y_data[:, np.newaxis]

        # OLS regression: b = (X'X)^-1 X'Y
        # X_data shape: (n_features, n_regressors)
        # Y_data shape: (n_features, 1)
        # b shape: (n_regressors, 1)
        b = np.dot(np.linalg.pinv(X_data), Y_data)
        res = Y_data - np.dot(X_data, b)

        # Unbiased estimator of residual standard error: sqrt(RSS / df)
        # This is correct for both intercept and intercept-free models
        # See GH #287 for details on why np.std(res, ddof=p) is biased
        n, p = X_data.shape
        sigma = np.sqrt(np.sum(res**2, axis=0) / (n - p))
        if sigma.ndim == 0:
            sigma = sigma[np.newaxis]

        stderr = (
            np.sqrt(np.diag(np.linalg.pinv(np.dot(X_data.T, X_data))))[:, np.newaxis]
            * sigma[np.newaxis, :]
        )

        # t-statistics
        t = np.zeros_like(b)
        t[stderr > 1.0e-6] = b[stderr > 1.0e-6] / stderr[stderr > 1.0e-6]

        # p-values
        df = np.array([X_data.shape[0] - X_data.shape[1]] * t.shape[1])
        p = 2 * (1 - t_dist.cdf(np.abs(t), df))

        # Create Adjacency objects for each stat
        # For Adjacency X, b has shape (n_regressors, 1), so we need to reshape
        # to match Adjacency data format which expects (n_matrices, n_features)
        stats["beta"] = adj.copy()
        stats["sigma"] = adj.copy()
        stats["t"] = adj.copy()
        stats["p"] = adj.copy()
        stats["df"] = adj.copy()
        stats["residual"] = adj.copy()

        # Assign data - ensure 2D shape for Adjacency compatibility
        b_flat = b.squeeze()
        if b_flat.ndim == 0:
            b_flat = np.array([b_flat])
        stats["beta"].data = b_flat
        stats["sigma"].data = (
            stderr.squeeze().T if stderr.shape[0] > 1 else stderr.squeeze()
        )
        stats["t"].data = t.squeeze().T if t.shape[0] > 1 else t.squeeze()
        stats["p"].data = p.squeeze().T if p.shape[0] > 1 else p.squeeze()
        stats["df"].data = df.squeeze()
        stats["residual"].data = res.squeeze().T if res.shape[1] == 1 else res.squeeze()

    elif isinstance(X, DesignMatrix):
        if X.shape[0] != len(adj):
            raise ValueError(
                "Design matrix must have same number of observations as Adjacency"
            )
        # Convert Polars DesignMatrix to numpy
        X_data = X.to_numpy()
        Y_data = adj.data

        # Ensure Y is 2D
        if len(Y_data.shape) == 1:
            Y_data = Y_data[:, np.newaxis]

        # OLS regression: b = (X'X)^-1 X'Y
        b = np.dot(np.linalg.pinv(X_data), Y_data)
        res = Y_data - np.dot(X_data, b)

        # Unbiased estimator of residual standard error: sqrt(RSS / df)
        # This is correct for both intercept and intercept-free models
        # See GH #287 for details on why np.std(res, ddof=p) is biased
        n, p = X_data.shape
        sigma = np.sqrt(np.sum(res**2, axis=0) / (n - p))
        if sigma.ndim == 0:
            sigma = sigma[np.newaxis]

        stderr = (
            np.sqrt(np.diag(np.linalg.pinv(np.dot(X_data.T, X_data))))[:, np.newaxis]
            * sigma[np.newaxis, :]
        )

        # t-statistics
        t = np.zeros_like(b)
        t[stderr > 1.0e-6] = b[stderr > 1.0e-6] / stderr[stderr > 1.0e-6]

        # p-values
        df = np.array([X_data.shape[0] - X_data.shape[1]] * t.shape[1])
        p = 2 * (1 - t_dist.cdf(np.abs(t), df))

        stats["beta"], stats["sigma"], stats["t"] = [adj.copy() for _ in range(3)]
        stats["p"], stats["df"], stats["residual"] = [adj.copy() for _ in range(3)]

        # Assign data - ensure proper shape for DesignMatrix case
        # For DesignMatrix, b has shape (n_regressors, n_features)
        # We need to reshape to (n_features, n_regressors) to match Adjacency format
        # where each row is a matrix (feature) and columns are regressors
        # But since we only have one regressor, we need (n_features,) shape
        # to match the original Adjacency data format
        if b.shape[0] == 1:
            # Single regressor case: b is (1, n_features), transpose to (n_features,)
            # Result is a single matrix of coefficients
            for key in ["beta", "sigma", "t", "p", "df", "residual"]:
                stats[key].is_single_matrix = True
            stats["beta"].data = b.squeeze()
            stats["sigma"].data = stderr.squeeze()
            stats["t"].data = t.squeeze()
            stats["p"].data = p.squeeze()
            stats["df"].data = df.squeeze() if df.ndim > 0 else df
            stats["residual"].data = res.squeeze()
        else:
            # Multiple regressors: b is (n_regressors, n_features), transpose to (n_features, n_regressors)
            stats["beta"].data = b.T
            stats["sigma"].data = stderr.T
            stats["t"].data = t.T
            stats["p"].data = p.T
            stats["df"].data = df
            stats["residual"].data = res.T
    else:
        raise ValueError("X must be a DesignMatrix or Adjacency Instance.")

    return stats


def social_relations_model(adj, summarize_results=True, nan_replace=True):
    """Estimate the social relations model from a matrix for a round-robin design.

    X_{ij} = m + \\alpha_i + \\beta_j + g_{ij} + \\epsilon_{ijl}

    where X_{ij} is the score for person i rating person j, m is the group mean,
    \\alpha_i  is person i's actor effect, \\beta_j is person j's partner effect, g_{ij}
    is the relationship  effect and \\epsilon_{ijl} is the error in measure l  for actor i and partner j.

    This model is primarily concerned with partioning the variance of the various effects.

    Code is based on implementation presented in Chapter 8 of Kenny, Kashy, & Cook (2006).
    Tests replicate examples  presented in the book. Note, that this method assumes that
    actor scores are rows (lower triangle), while partner scores are columnns (upper triangle).
    The minimal sample size to estimate these effects is 4.

    Model Assumptions:
     - Social interactions are exclusively dyadic
     - People are randomly sampled from population
     - No order effects
     - The effects combine additively and relationships are linear

    In the future we might update the formulas and standard errors based on
    Bond and Lashley, 1996

    Args:
        adj: (Adjacency) can be a single matrix or many matrices for each group
        summarize_results: (bool) will provide a formatted summary of model results
        nan_replace: (bool) will replace nan values with row and column means

    Returns:
        estimated effects: (pd.Series/pd.DataFrame) All of the effects estimated using SRM
    """
    import pandas as pd

    from nltools.data.adjacency import Adjacency
    from scipy.spatial.distance import squareform
    import scipy.stats as scipy_stats

    def mean_square_between(x1, x2=None, df="standard"):
        """Calculate between-dyad variance."""

        if df == "standard":
            n = len(x1)
            df = n - 1
        elif df == "relationship":
            n = len(squareform(x1))
            df = ((n - 1) * (n - 2) / 2) - 1
        else:
            raise ValueError("df can only be ['standard', 'relationship']")
        if x2 is not None:
            return (
                2 * np.nansum((((x1 + x2) / 2) - np.nanmean((x1 + x2) / 2)) ** 2) / df
            )
        return np.nansum((x1 - np.nanmean(x1)) ** 2) / df

    def mean_square_within(x1, x2, df="standard"):
        """Calculate within-dyad variance."""

        if df == "standard":
            n = len(x1)
            df = n
        elif df == "relationship":
            n = len(squareform(x1))
            df = (n - 1) * (n - 2) / 2
        else:
            raise ValueError("df can only be ['standard', 'relationship']")
        return np.nansum((x1 - x2) ** 2) / (2 * df)

    def estimate_person_effect(n, x1_mean, x2_mean, grand_mean):
        """Calculate actor, partner, and relationship effects."""
        return (
            ((n - 1) ** 2 / (n * (n - 2))) * x1_mean
            + ((n - 1) / (n * (n - 2))) * x2_mean
            - ((n - 1) / (n - 2)) * grand_mean
        )

    def estimate_person_variance(x, ms_b, ms_w):
        """Calculate variance for a specific dyad member, such as actor or partner."""
        n = len(x)
        return mean_square_between(x) - (ms_b / (2 * (n - 2))) - (ms_w / (2 * n))

    def estimate_srm(data):
        """Estimate a Social Relations Model from a single matrix."""

        if not data.is_single_matrix:
            raise ValueError(
                "This function only operates on single matrix Adjacency instances."
            )

        n = data.n_nodes
        if n < 4:
            raise ValueError(
                "The Social Relations Model cannot be estimated when sample size is less than 4."
            )
        grand_mean = data.mean()
        dat = data.squareform().copy()
        np.fill_diagonal(dat, np.nan)
        actor_mean = np.nanmean(dat, axis=1)
        partner_mean = np.nanmean(dat, axis=0)

        a = estimate_person_effect(
            n, actor_mean, partner_mean, grand_mean
        )  # Actor effects
        b = estimate_person_effect(
            n, partner_mean, actor_mean, grand_mean
        )  # Partner effects

        # Relationship effects
        g = np.ones(dat.shape) * np.nan
        for i in range(n):
            for j in range(n):
                if i != j:
                    g[i, j] = dat[i, j] - a[i] - b[j] - grand_mean

        # Estimate Variance
        x1 = g[np.tril_indices(n, k=-1)]
        x2 = g[np.triu_indices(n, k=1)]
        ms_b = mean_square_between(x1, x2, df="relationship")
        ms_w = mean_square_within(x1, x2, df="relationship")
        actor_variance = estimate_person_variance(a, ms_b, ms_w)
        partner_variance = estimate_person_variance(b, ms_b, ms_w)
        relationship_variance = (ms_b + ms_w) / 2
        dyadic_reciprocity_covariance = (ms_b - ms_w) / 2
        dyadic_reciprocity_correlation = (ms_b - ms_w) / (ms_b + ms_w)
        actor_partner_covariance = (
            (np.sum(a * b) / (n - 1)) - (ms_b / (2 * (n - 2))) + (ms_w / (2 * n))
        )
        actor_partner_correlation = actor_partner_covariance / (
            np.sqrt(actor_variance * partner_variance)
        )
        actor_reliability = actor_variance / (
            actor_variance
            + (relationship_variance / (n - 1))
            - (dyadic_reciprocity_covariance / ((n - 1) ** 2))
        )
        partner_reliability = partner_variance / (
            partner_variance
            + (relationship_variance / (n - 1))
            - (dyadic_reciprocity_covariance / ((n - 1) ** 2))
        )
        adjusted_dyadic_reciprocity_correlation = actor_partner_correlation * np.sqrt(
            actor_reliability * partner_reliability
        )
        total_variance = actor_variance + partner_variance + relationship_variance

        return pd.Series(
            {
                "grand_mean": grand_mean,
                "actor_effect": a,
                "partner_effect": b,
                "relationship_effect": g,
                "actor_variance": actor_variance,
                "partner_variance": partner_variance,
                "relationship_variance": relationship_variance,
                "actor_partner_covariance": actor_partner_covariance,
                "actor_partner_correlation": actor_partner_correlation,
                "dyadic_reciprocity_covariance": dyadic_reciprocity_covariance,
                "dyadic_reciprocity_correlation": dyadic_reciprocity_correlation,
                "adjusted_dyadic_reciprocity_correlation": adjusted_dyadic_reciprocity_correlation,
                "actor_reliability": actor_reliability,
                "partner_reliability": partner_reliability,
                "total_variance": total_variance,
            }
        )

    def summarize_srm_results(results):
        """Summarize Social Relations Model results."""

        def estimate_srm_stats(results, var_name, tailed=1):
            """Compute mean estimate, standard error, t-statistic, and p-value for an SRM variance component.

            Args:
                results: DataFrame of SRM results across groups, or Series for a single group.
                var_name: Name of the variance component column to summarize.
                tailed: Number of tails for the t-test (1 or 2).

            Returns:
                Tuple of (estimate, standardized, se, t, p).
            """
            estimate = results[var_name].mean()
            standardized = (results[var_name] / results["total_variance"]).mean()
            se = results[var_name].std() / np.sqrt(len(results[var_name]))
            with np.errstate(invalid="ignore", divide="ignore"):
                t = estimate / se
            if tailed == 1:
                p = 1 - scipy_stats.t.cdf(t, len(results[var_name]) - 1)
            elif tailed == 2:
                p = 2 * (1 - scipy_stats.t.cdf(t, len(results[var_name]) - 1))
            else:
                raise ValueError("tailed can only be [1,2]")
            return (estimate, standardized, se, t, p)

        def print_srm_stats(results, var_name, tailed=1):
            """Print a formatted summary row for an SRM variance component across multiple groups.

            Args:
                results: DataFrame of SRM results across groups.
                var_name: Name of the variance component column to print.
                tailed: Number of tails for the t-test (1 or 2).
            """
            estimate, standardized, se, t, p = estimate_srm_stats(
                results, var_name, tailed
            )
            print(
                f"{var_name:<40} {estimate:^10.2f}{standardized:^10.2f} {se:^10.2f} {t:^10.2f} {p:^10.4f}"
            )

        def print_single_group_srm_stats(results, var_name):
            """Print a formatted summary row for an SRM variance component for a single group.

            Inference statistics (se, t, p) are printed as NaN since they require multiple groups.

            Args:
                results: Series of SRM results for a single group.
                var_name: Name of the variance component to print.
            """
            estimate = results[var_name].mean()
            standardized = (results[var_name] / results["total_variance"]).mean()
            print(
                f"{var_name:<40} {estimate:^10.2f}{standardized:^10.2f} {np.nan:^10.2f} {np.nan:^10.2f} {np.nan:^10.4f}"
            )

        def print_srm_covariances(results, var_name):
            """Print a formatted summary row for an SRM covariance component across multiple groups.

            Uses the covariance estimate for inference and correlation as the standardized effect size.

            Args:
                results: DataFrame of SRM results across groups.
                var_name: Name of the covariance component (without '_covariance' or '_correlation' suffix).
            """
            estimate, _, se, t, p = estimate_srm_stats(
                results, f"{var_name}_covariance", tailed=2
            )
            standardized = results[f"{var_name}_correlation"].mean()
            print(
                f"{var_name:<40} {estimate:^10.2f}{standardized:^10.2f} {se:^10.2f} {t:^10.2f} {p:^10.4f}"
            )

        def print_single_srm_covariances(results, var_name):
            """Print a formatted summary row for an SRM covariance component for a single group.

            Inference statistics (se, t, p) are printed as NaN since they require multiple groups.

            Args:
                results: Series of SRM results for a single group.
                var_name: Name of the covariance component (without '_covariance' or '_correlation' suffix).
            """
            estimate = results[f"{var_name}_covariance"].mean()
            standardized = results[f"{var_name}_correlation"].mean()
            print(
                f"{var_name:<40} {estimate:^10.2f}{standardized:^10.2f} {np.nan:^10.2f} {np.nan:^10.2f} {np.nan:^10.4f}"
            )

        if isinstance(results, pd.Series):
            n_groups = 1
            group_size = results["actor_effect"].shape[0]
        elif isinstance(results, pd.DataFrame):
            n_groups = len(results)
            group_size = np.mean([x.shape for x in results["actor_effect"]])

        print("Social Relations Model: Results")
        print("\n")
        print(f"Number of Groups: {n_groups:<20}")
        print(f"Average Group Size: {group_size:<20}")
        print("\n")
        print(
            f"{'':<40} {'Estimate':<10} {'Standardized':<10} {'se':<10} {'t':<10} {'p':<10}"
        )
        if isinstance(results, pd.Series):
            print_single_group_srm_stats(results, "actor_variance")
            print_single_group_srm_stats(results, "partner_variance")
            print_single_group_srm_stats(results, "relationship_variance")
            print_single_srm_covariances(results, "actor_partner")
            print_single_srm_covariances(results, "dyadic_reciprocity")
        elif isinstance(results, pd.DataFrame):
            print_srm_stats(results, "actor_variance")
            print_srm_stats(results, "partner_variance")
            print_srm_stats(results, "relationship_variance")
            print_srm_covariances(results, "actor_partner")
            print_srm_covariances(results, "dyadic_reciprocity")
        print("\n")
        print(f"{'Actor Reliability':<20} {results['actor_reliability'].mean():^20.2f}")
        print(
            f"{'Partner Reliability':<20} {results['partner_reliability'].mean():^20.2f}"
        )
        print("\n")

    def replace_missing(data):
        """Replace missing data with row and column means and return missing coordinates."""

        def fix_missing(data):
            """Replace NaN off-diagonal entries with the mean of their row and column.

            Args:
                data: Adjacency matrix with possible NaN values.

            Returns:
                Tuple of (Adjacency with NaNs replaced, (row, col) coordinates of replaced values).
            """
            X = data.squareform().copy()
            x, y = np.where(np.isnan(X))
            for i, j in zip(x, y):
                if i != j:
                    X[i, j] = (np.nanmean(X[i, :]) + np.nanmean(X[:, j])) / 2
            X = Adjacency(X, matrix_type=data.matrix_type)
            return (X, (x, y))

        if data.is_single_matrix:
            X, coord = fix_missing(data)
        else:
            X = []
            coord = []
            for d in data:
                m, c = fix_missing(d)
                X.append(m)
                coord.append(c)
            X = Adjacency(X)
        return (X, coord)

    if nan_replace:
        data, _ = replace_missing(adj)
    else:
        data = adj.copy()

    if adj.is_single_matrix:
        results = estimate_srm(data)
    else:
        results = pd.DataFrame([estimate_srm(x) for x in data])

    if summarize_results:
        summarize_srm_results(results)

    return results


def generate_permutations(adj, n_permute, random_state=None):
    """Generate permuted versions of an Adjacency instance lazily.

    This is useful for iterative comparisons.

    Args:
        adj: (Adjacency) Adjacency instance
        n_permute (int): number of permutations
        random_state (int or np.random.RandomState, optional): random seed for reproducibility. Defaults to None.

    Examples:
        >>> for perm in generate_permutations(adj, 1000):
        >>>     out = neural_distance_mat.similarity(perm)
        >>>     ...

    Yields:
        Adjacency: permuted version of adj
    """
    from nltools.data.adjacency import Adjacency
    from sklearn.utils import check_random_state

    random_state = check_random_state(random_state)

    for _ in range(n_permute):
        # Get squareform as numpy array (no pandas conversion needed)
        dat = adj.squareform()
        # Generate random permutation indices
        permuted_idx = random_state.choice(
            dat.shape[0], size=dat.shape[0], replace=False
        )
        # Permute rows and columns using numpy advanced indexing (faster than pandas)
        dat = dat[np.ix_(permuted_idx, permuted_idx)]
        yield Adjacency(dat)
