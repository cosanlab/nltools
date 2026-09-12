"""Provide standalone modeling and inference functions for Adjacency matrices.

Each function takes an Adjacency instance as its first argument (`adj`).
"""

import numpy as np


def bootstrap(
    adj,
    statistic,
    *,
    n_samples=5000,
    confidence_level=0.95,
    return_samples=False,
    n_jobs=-1,
    random_state=None,
    progress_bar=False,
):
    """Bootstrap an aggregate statistic across a stack of matrices.

    Resamples matrices with replacement and aggregates the replicates as they
    complete, so what the run holds is the retained tail — about
    `(1 - confidence_level)` of the replicates per edge — plus one dispatch
    window, rather than all `n_samples` matrices.

    Args:
        adj (Adjacency): Adjacency instance containing multiple matrices.
        statistic (str): Statistic to bootstrap: `'mean'`, `'median'`, `'std'`,
            `'sum'`, `'min'`, or `'max'` — each the corresponding NumPy
            reduction over matrices, with `'std'` at `ddof=0`.
        n_samples (int): Number of bootstrap replicates, at least two. Default
            5000.
        confidence_level (float): Confidence level of the reported interval,
            strictly between zero and one. Default 0.95.
        return_samples (bool): Retain and return every replicate. Default
            False.
        n_jobs (int): CPU worker ceiling. -1 (default) means all cores.
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): If True, show a progress bar. Default False.

    Returns:
        BootstrapResult: `estimate`, `standard_error`, `ci_lower` and
            `ci_upper` as single-matrix `Adjacency` objects, plus `samples` as
            a NumPy array with the bootstrap axis first when
            `return_samples=True`.

    Raises:
        ValueError: If `statistic` is unknown, an argument is out of range, or
            the retained output cannot fit the measured memory budget.

    Examples:
        ```python
        boot = bootstrap(adj, "mean", n_samples=1000)
        boot.estimate  # → Adjacency
        ```
    """
    from nltools.algorithms.inference.bootstrap import (
        _bootstrap_simple_cpu_parallel,
    )

    SIMPLE_STATS = ["mean", "median", "std", "sum", "min", "max"]
    if statistic not in SIMPLE_STATS:
        raise ValueError(
            f"Unsupported statistic '{statistic}'. "
            f"Supported basic statistics: {SIMPLE_STATS}."
        )

    # Adjacency.data shape: (n_matrices, n_edges)
    result = _bootstrap_simple_cpu_parallel(
        adj.data,
        method=statistic,
        n_samples=n_samples,
        confidence_level=confidence_level,
        return_samples=return_samples,
        n_jobs=n_jobs,
        random_state=random_state,
        progress_bar=progress_bar,
    )

    return convert_bootstrap_results_to_adjacency(adj, result)


def convert_bootstrap_results_to_adjacency(adj, result):
    """Wrap an engine's arrays as a `BootstrapResult` of single-matrix `Adjacency`.

    Args:
        adj (Adjacency): Instance supplying matrix kind and node labels.
        result (dict): Engine output with `'estimate'`, `'standard_error'`,
            `'ci_lower'`, `'ci_upper'`, and optionally `'samples'`.

    Returns:
        BootstrapResult: The four summaries as `Adjacency`, and the retained
            replicates as a NumPy array when present.
    """
    import polars as pl

    from nltools.data.results import BootstrapResult

    from .state import common_labels, result as adjacency_result

    labels = common_labels(adj)

    def _map(values):
        return adjacency_result(
            adj,
            np.asarray(values).reshape(-1),
            labels=labels,
            Y=pl.DataFrame(),
        )

    return BootstrapResult(
        estimate=_map(result["estimate"]),
        standard_error=_map(result["standard_error"]),
        ci_lower=_map(result["ci_lower"]),
        ci_upper=_map(result["ci_upper"]),
        samples=result.get("samples"),
    )


def regress(adj, X, method="ols", tail=2):
    """Run a regression on an adjacency instance.

    Pass an `Adjacency` as `X` to decompose `adj` with other matrices, or a
    `DesignMatrix` to regress each cell across a stack of matrices.

    Args:
        adj (Adjacency): Adjacency instance.
        X (Adjacency | DesignMatrix): Design matrix.
        method (str): Type of regression; only `'ols'` is currently supported.
        tail (int | str): `2`/`'two'` for two-tailed (default); `1`/`'one'` for
            one-tailed (beta > 0; negate a regressor for the other direction).

    Returns:
        dict: Coefficient fields `beta`, `sigma` (coefficient standard error),
            `t`, and `p` are predictor Adjacency maps for DesignMatrix input and
            native predictor arrays/scalars for Adjacency input. `df` is scalar;
            `residual` is an Adjacency retaining the response shape and metadata.
    """
    import polars as pl
    from scipy.stats import t as t_dist
    from nltools.data.adjacency import Adjacency
    from nltools.data.designmatrix import DesignMatrix
    from nltools.algorithms.inference.validation import validate_tail_parameter
    from .state import common_labels, result, validate_compatible

    tail_internal = validate_tail_parameter(tail)
    if method != "ols":
        raise ValueError(
            "Only 'ols' method is currently supported for Adjacency.regress()"
        )
    if isinstance(X, Adjacency):
        if not adj.is_single_matrix:
            raise ValueError("Adjacency predictors require a single response matrix.")
        validate_compatible(adj, X)
        response_labels = common_labels(adj)
        predictor_labels = common_labels(X)
        if response_labels != predictor_labels or X.labels and not predictor_labels:
            raise ValueError("Predictor and response node ordering must match.")
        design = np.atleast_2d(X.data).T
        response = adj.data[:, None]
    elif isinstance(X, DesignMatrix):
        if X.shape[0] != len(adj):
            raise ValueError(
                "Design matrix must have same number of observations as Adjacency"
            )
        design = X.to_numpy()
        response = np.atleast_2d(adj.data)
    else:
        raise ValueError("X must be a DesignMatrix or Adjacency Instance.")

    beta = np.linalg.pinv(design) @ response
    residual = response - design @ beta
    df = design.shape[0] - design.shape[1]
    # Retain the RSS-based scale, including intercept-free models (GH #287).
    residual_scale = np.sqrt(np.sum(residual**2, axis=0) / df)
    stderr = (
        np.sqrt(np.diag(np.linalg.pinv(design.T @ design)))[:, None] * residual_scale
    )
    t = np.zeros_like(beta)
    np.divide(beta, stderr, out=t, where=stderr > 1.0e-6)
    p = (
        1 - t_dist.cdf(t, df)
        if tail_internal == "upper"
        else 2 * (1 - t_dist.cdf(np.abs(t), df))
    )
    stats = {"df": df}
    for key, values in [("beta", beta), ("sigma", stderr), ("t", t), ("p", p)]:
        if isinstance(X, Adjacency):
            stats[key] = values[:, 0].copy() if len(values) > 1 else values[0, 0].item()
        else:
            stats[key] = result(
                adj,
                values[0] if len(values) == 1 else values,
                labels=common_labels(adj),
                Y=pl.DataFrame(),
            )
    residual_values = (
        residual[:, 0]
        if isinstance(X, Adjacency)
        else residual[0]
        if adj.is_single_matrix
        else residual
    )
    stats["residual"] = result(adj, residual_values, labels=adj.labels, Y=adj.Y)
    return stats


def social_relations_model(adj, summarize_results=True, nan_replace=True):
    """Estimate the social relations model from a matrix for a round-robin design.

    $$X_{ij} = m + \\alpha_i + \\beta_j + g_{ij} + \\epsilon_{ijl}$$

    where $X_{ij}$ is the score for person i rating person j, $m$ is the group mean,
    $\\alpha_i$ is person i's actor effect, $\\beta_j$ is person j's partner effect, $g_{ij}$
    is the relationship effect and $\\epsilon_{ijl}$ is the error in measure l for actor i and partner j.

    This model is primarily concerned with partitioning the variance of the various
    effects. The implementation follows Chapter 8 of Kenny, Kashy, & Cook (2006) and
    the tests replicate the book's examples. Actor scores are rows (lower triangle)
    and partner scores are columns (upper triangle). The minimal sample size to
    estimate these effects is 4.

    **Model assumptions:** social interactions are exclusively dyadic; people are
    randomly sampled from the population; there are no order effects; the effects
    combine additively and relationships are linear.

    Args:
        adj (Adjacency): A single matrix, or one matrix per group.
        summarize_results (bool): If True, print a formatted summary of model results.
        nan_replace (bool): If True, replace NaN values with row and column means.

    Returns:
        pd.Series | pd.DataFrame: All of the effects estimated using SRM (a Series
            for a single matrix, a DataFrame with one row per matrix otherwise).

    References:
        Kenny, D. A., Kashy, D. A., & Cook, W. L. (2006). *Dyadic data analysis*.
        Guilford Press.
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
        adj (Adjacency): Adjacency instance.
        n_permute (int): Number of permutations.
        random_state (int | np.random.RandomState, optional): Random seed for
            reproducibility. Defaults to None.

    Yields:
        Adjacency: Permuted version of `adj`.

    Examples:
        ```python
        for perm in generate_permutations(adj, 1000):
            out = neural_distance_mat.similarity(perm)
        ```
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
