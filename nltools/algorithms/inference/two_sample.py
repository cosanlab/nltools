"""Two-sample permutation test (group-label shuffling).

Tests whether two independent groups differ in mean by randomly reassigning
observations to groups — the permutation analogue of an independent-samples
t-test. Permutations run on joblib workers; `n_jobs` sets how many, and a
given `random_state` gives the same result at any worker count.
"""

import numpy as np

from .utils import maybe_tqdm
from .validation import validate_array_shape_range
from ..validation import _compute_pvalue, validate_tail_parameter
from .random import generate_seeds


def _two_sample_permutation_cpu_parallel(
    data1: np.ndarray,
    data2: np.ndarray,
    *,
    n_permute: int,
    tail: int,
    return_null: bool,
    n_jobs: int,
    random_state: int | None,
    single_feature: bool = False,
    progress_bar: bool = False,
) -> dict:
    """Two-sample permutation test parallelized across CPU cores with joblib.

    Each worker shuffles the group labels for one permutation (from its own
    pre-drawn seed) and computes the mean difference, so results are
    reproducible regardless of worker count.

    Args:
        data1 (np.ndarray): Group 1 data, shape `(n_samples1, n_features)`.
        data2 (np.ndarray): Group 2 data, shape `(n_samples2, n_features)`.
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
        dict: Same format as `two_sample_permutation_test`.
    """
    from joblib import Parallel, delayed

    # Setup random state and generate seeds for workers
    seeds = generate_seeds(n_permute, random_state=random_state)

    # Get dimensions (data already reshaped by caller)
    n1, n_features = data1.shape
    n2 = data2.shape[0]
    n_total = n1 + n2

    # Compute observed mean difference
    obs_diff = np.nanmean(data1, axis=0) - np.nanmean(data2, axis=0)

    # Concatenate data for permutation
    combined = np.vstack([data1, data2])  # (n_total, n_features)

    # Define worker function (each processes ONE permutation)
    def _compute_one_perm(seed):
        """Compute mean difference for one group permutation."""
        perm_rng = np.random.RandomState(seed)
        # Randomly shuffle indices
        indices = perm_rng.permutation(n_total)
        # Split into two groups
        group1_indices = indices[:n1]
        group2_indices = indices[n1:]
        # Compute mean difference
        mean1 = np.nanmean(combined[group1_indices], axis=0)
        mean2 = np.nanmean(combined[group2_indices], axis=0)
        return mean1 - mean2

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
    p_values = _compute_pvalue(obs_diff, null_dist, tail=tail)

    # Return to original shape
    if single_feature:
        obs_diff = obs_diff.item() if hasattr(obs_diff, "item") else float(obs_diff[0])
        p_values = p_values.item() if hasattr(p_values, "item") else float(p_values[0])

    # Build result
    result = {
        "mean_diff": obs_diff,
        "p": p_values,
    }

    if return_null:
        if single_feature:
            null_dist = null_dist.squeeze()
        result["null_dist"] = null_dist

    return result


def two_sample_permutation_test(
    data1: np.ndarray,
    data2: np.ndarray,
    *,
    n_permute: int = 5000,
    tail: int | str = 2,
    return_null: bool = False,
    n_jobs: int = -1,
    random_state: int | None = None,
    progress_bar: bool = False,
) -> dict:
    """Two-sample permutation test using group-label shuffling.

    Tests whether two independent groups have different means by randomly
    reassigning observations to groups — the permutation analogue of an
    independent-samples t-test. Group sizes may differ. Multi-feature
    (voxel-wise) data tests each column independently against the same
    permutations.

    Assumes exchangeability under the null (group assignment is arbitrary):
    independent samples from similarly shaped distributions. NaN observations
    are dropped from the observed and every permuted mean (`np.nanmean`),
    feature by feature.

    Args:
        data1 (np.ndarray): Group 1 data, shape `(n_samples1,)` for a single
            feature or `(n_samples1, n_features)` for voxel-wise data. May
            contain NaN observations.
        data2 (np.ndarray): Group 2 data, shape `(n_samples2,)` or
            `(n_samples2, n_features)`; must have the same number of features
            as `data1`. May contain NaN observations.
        n_permute (int): Number of permutations. Defaults to 5000.
        tail (int | str): `2` or `'two'` (default) for a two-tailed test
            (mean1 != mean2); `1` or `'one'` for a one-tailed test of
            mean1 > mean2 (swap the groups for the other direction — the fixed
            direction keeps multiple-comparison correction valid).
        return_null (bool): If True, include the full null distribution in the
            result. Defaults to False.
        n_jobs (int): Number of joblib workers. Defaults to -1 (all cores).
            Results are identical at every worker count.
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Whether to display a progress bar. Defaults to False.

    Returns:
        dict: Keys `'mean_diff'` (float or np.ndarray, observed
            `mean(data1) - mean(data2)`), `'p'` (float or np.ndarray,
            p-value(s)), and — when `return_null=True` — `'null_dist'`
            (np.ndarray, shape `(n_permute,)` or `(n_permute, n_features)`).

    Examples:
        ```python
        # Single feature
        data1 = np.random.randn(20)  # Group 1: 20 subjects
        data2 = np.random.randn(25)  # Group 2: 25 subjects
        result = two_sample_permutation_test(data1, data2, n_permute=5000)
        result["p"]  # → 0.45

        # Voxel-wise test
        data1 = np.random.randn(20, 10000)  # 20 subjects, 10K voxels
        data2 = np.random.randn(25, 10000)  # 25 subjects, 10K voxels
        result = two_sample_permutation_test(data1, data2, n_permute=5000)
        result["mean_diff"].shape  # → (10000,)
        result["p"].shape  # → (10000,)
        ```
    """
    # Input validation
    data1 = np.asarray(data1, dtype=np.float64)
    data2 = np.asarray(data2, dtype=np.float64)

    validate_array_shape_range(data1, 1, 2, name="data1")
    validate_array_shape_range(data2, 1, 2, name="data2")
    validate_tail_parameter(tail)

    # Handle shape
    single_feature = data1.ndim == 1 and data2.ndim == 1
    if data1.ndim == 1:
        data1 = data1[:, np.newaxis]
    if data2.ndim == 1:
        data2 = data2[:, np.newaxis]

    # Check feature dimensions match
    if data1.shape[1] != data2.shape[1]:
        raise ValueError(
            f"data1 and data2 must have same number of features, "
            f"got {data1.shape[1]} and {data2.shape[1]}"
        )

    return _two_sample_permutation_cpu_parallel(
        data1,
        data2,
        n_permute=n_permute,
        tail=tail,
        return_null=return_null,
        n_jobs=n_jobs,
        random_state=random_state,
        single_feature=single_feature,
        progress_bar=progress_bar,
    )
