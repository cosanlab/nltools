"""One-sample permutation test (sign flipping).

Tests whether a mean differs from zero by randomly flipping the sign of each
observation — the permutation analogue of a one-sample t-test. Permutations run
on joblib workers; `n_jobs` sets how many, and a given `random_state` gives the
same result at any worker count.
"""

import numpy as np
from scipy.stats import ttest_1samp

from .utils import (
    _generate_sign_flips,
    _compute_pvalue,
    _signed_z_from_p,
    maybe_tqdm,
)
from .validation import (
    validate_tail_parameter,
    validate_array_shape_range,
)


def _one_sample_permutation_cpu_parallel(
    data: np.ndarray,
    *,
    n_permute: int,
    tail: int,
    return_null: bool,
    n_jobs: int,
    random_state: int | None,
    single_feature: bool = False,
    progress_bar: bool = False,
) -> dict:
    """One-sample permutation test parallelized across CPU cores with joblib.

    Pre-generates every sign flip deterministically (`n_permute × n_samples`
    bytes — negligible) and parallelizes only the computation, so p-values are
    reproducible regardless of worker count. Typical speedup is 4-8× on an
    8-core machine.

    Args:
        data (np.ndarray): Data to test, shape `(n_samples, n_features)`.
        n_permute (int): Number of permutations.
        tail (int | str): `2` or `'two'` for two-tailed; `1` or `'one'` for
            one-tailed.
        return_null (bool): Whether to return the null distribution.
        n_jobs (int): Number of parallel jobs (-1 = all cores).
        random_state (int | None): Random seed for reproducibility.
        single_feature (bool): Whether the caller passed 1D data (results are
            returned as scalars).
        progress_bar (bool): Whether to display a tqdm progress bar.

    Returns:
        dict: Same format as `one_sample_permutation_test`.
    """
    from joblib import Parallel, delayed

    # Get dimensions (data is already reshaped by caller)
    n_samples, n_features = data.shape

    # Compute observed statistic
    obs_stat = np.mean(data, axis=0)

    # Pre-generate ALL sign-flips (matches stats.py pattern exactly)
    sign_flips = _generate_sign_flips(n_permute, n_samples, random_state=random_state)

    # Define worker function (each processes ONE permutation with pre-computed signs)
    def _compute_one_perm(signs):
        """Compute statistic for one sign-flip permutation (signs pre-computed)."""
        perm_data = data * signs[:, np.newaxis]
        return np.mean(perm_data, axis=0)

    # Execute in parallel with progress bar
    null_dist = Parallel(n_jobs=n_jobs)(
        delayed(_compute_one_perm)(sign_flips[i])
        for i in maybe_tqdm(
            range(n_permute),
            progress_bar=progress_bar,
            desc="CPU parallel perms",
            unit="perm",
        )
    )
    null_dist = np.array(null_dist)  # Shape: (n_permute, n_features)

    # Compute p-values
    p_values = _compute_pvalue(obs_stat, null_dist, tail=tail)

    # Return to original shape
    if single_feature:
        obs_stat = float(obs_stat[0])
        p_values = float(p_values[0])

    # Build result
    result = {
        "mean": obs_stat,
        "p": p_values,
    }

    if return_null:
        if single_feature:
            null_dist = null_dist.squeeze()
        result["null_dist"] = null_dist

    return result


def one_sample_permutation_test(
    data: np.ndarray,
    *,
    n_permute: int = 5000,
    tail: int | str = 2,
    return_null: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
    progress_bar: bool = False,
) -> dict:
    """One-sample permutation test using sign flipping.

    Tests whether the mean of `data` differs from zero by randomly flipping the
    sign of each observation — the permutation analogue of a one-sample t-test.
    Multi-feature (voxel-wise) data tests each column independently against
    the same permutations.

    Assumes errors are distributed symmetrically around zero. For strongly
    skewed data, prefer bootstrap resampling.

    Args:
        data (np.ndarray): Data to test, shape `(n_samples,)` for a single
            feature or `(n_samples, n_features)` for voxel-wise data.
        n_permute (int): Number of permutations. Defaults to 5000.
        tail (int | str): `2` or `'two'` (default) for a two-tailed test
            (mean != 0); `1` or `'one'` for a one-tailed test of mean > 0
            (negate the data for the other direction — the fixed direction
            keeps multiple-comparison correction valid).
        return_null (bool): If True, include the full null distribution in the
            result. Defaults to False.
        n_jobs (int): Number of joblib workers. Defaults to -1 (all cores).
            Results are identical at every worker count.
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Whether to display a progress bar. Defaults to False.

    Returns:
        dict: Keys `'mean'` (float or np.ndarray, observed mean(s)), `'p'`
            (float or np.ndarray, p-value(s)), and — when `return_null=True` —
            `'null_dist'` (np.ndarray, shape `(n_permute,)` or
            `(n_permute, n_features)`).

    Examples:
        ```python
        # Single feature
        data = np.random.randn(30)
        result = one_sample_permutation_test(data, n_permute=5000)
        result["p"]  # → 0.23

        # Voxel-wise test
        data = np.random.randn(30, 10000)  # 30 subjects, 10K voxels
        result = one_sample_permutation_test(data, n_permute=5000)
        result["mean"].shape  # → (10000,)
        result["p"].shape  # → (10000,)
        ```
    """
    # Input validation
    data = np.asarray(data, dtype=np.float64)
    validate_array_shape_range(data, 1, 2, name="data")
    validate_tail_parameter(tail)

    # Handle shape
    single_feature = data.ndim == 1
    if single_feature:
        data = data[:, np.newaxis]  # (n_samples, 1)

    return _one_sample_permutation_cpu_parallel(
        data,
        n_permute=n_permute,
        tail=tail,
        return_null=return_null,
        n_jobs=n_jobs,
        random_state=random_state,
        single_feature=single_feature,
        progress_bar=progress_bar,
    )


def _one_sample_statistics(
    data: np.ndarray,
    *,
    popmean: float = 0.0,
    permutation: bool = False,
    n_permute: int = 5000,
    tail: int | str = 2,
    return_null: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
    progress_bar: bool = False,
) -> dict:
    """Compute the shared one-sample t-test statistics for a 2-D feature matrix.

    The single implementation behind `BrainData.ttest` and `Adjacency.ttest`
    (see `docs/development/specs/ttest.md`). It works on plain arrays and
    returns plain arrays; each facade wraps them in its own result type.

    `t` is always the observed SciPy statistic against `popmean`, on both the
    parametric and the permutation path. Only `p` changes. The parametric path
    uses SciPy's p-value for the requested tail; the permutation path uses the
    empirical sign-flip p-value from `data - popmean`, computed once for the
    whole matrix by `one_sample_permutation_test`. The permutation null
    therefore holds centered *means*, not t-statistics.

    Args:
        data (np.ndarray): Observations to test, shape `(n_obs, n_features)`.
        popmean (float): Population mean to test against. Defaults to 0.0.
        permutation (bool): If True, take p from a sign-flip permutation test
            instead of the parametric test. Defaults to False.
        n_permute (int): Number of permutations, used only when
            `permutation=True`. Defaults to 5000.
        tail (int | str): `2` or `'two'` (default) for a two-tailed test;
            `1` or `'one'` for a one-tailed test of mean > `popmean`.
        return_null (bool): If True, also return the permutation null. Has no
            effect on the parametric path, which computes no null. Defaults to
            False.
        n_jobs (int): CPU cores for the permutation engine. Defaults to -1.
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Whether to display a progress bar. Defaults to False.

    Returns:
        dict: `'mean'` (sample mean minus `popmean`), `'t'`, `'z'`, and `'p'`,
            each an `(n_features,)` array. With `permutation=True` and
            `return_null=True` the dict also holds `'null_dist'`, an
            `(n_permute, n_features)` array of centered means whose feature
            axis is never squeezed. No returned array aliases `data` or any
            other returned array.

    Raises:
        ValueError: If `data` is not 2-D or holds fewer than two observations.
    """
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(
            f"data must be 2-D with shape (n_obs, n_features); got {values.shape}."
        )
    if values.shape[0] < 2:
        raise ValueError(
            "A one-sample t-test requires at least two observations to estimate "
            "the variance."
        )

    tail_internal = validate_tail_parameter(tail)
    # 'one' is the test's positive direction: mean > popmean.
    alternative = "two-sided" if tail_internal == "two" else "greater"
    t_values, p_parametric = ttest_1samp(
        values, popmean, axis=0, alternative=alternative
    )
    t_values = np.asarray(t_values, dtype=np.float64)
    mean_values = values.mean(axis=0) - popmean
    null_dist = None

    if permutation:
        # Sign flipping tests symmetry around zero, so the engine must see the
        # popmean-referenced data — flipping raw values would silently test
        # mean != 0 instead of mean != popmean.
        engine = one_sample_permutation_test(
            values - popmean,
            n_permute=n_permute,
            tail=tail,
            return_null=return_null,
            n_jobs=n_jobs,
            random_state=random_state,
            progress_bar=progress_bar,
        )
        p_values = np.asarray(engine["p"], dtype=np.float64)
        # The engine's mean of the centered data IS mean(data) - popmean; keep
        # it so the reported effect and p come from the same numbers.
        mean_values = np.asarray(engine["mean"], dtype=np.float64)
        if return_null:
            null_dist = np.asarray(engine["null_dist"], dtype=np.float64)
    else:
        p_values = np.asarray(p_parametric, dtype=np.float64)

    results = {
        "mean": mean_values,
        "t": t_values,
        "z": _signed_z_from_p(t_values, p_values, tail_internal),
        "p": p_values,
    }
    if null_dist is not None:
        results["null_dist"] = null_dist
    return results
