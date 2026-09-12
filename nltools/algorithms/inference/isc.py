"""Intersubject correlation (ISC) with permutation and bootstrap inference.

Computes leave-one-out and pairwise ISC and tests it with the subject-wise
bootstrap of Chen et al. (2016) or surrogate time series (circular shift,
phase randomization), plus a two-group ISC difference test. Resamples run on
joblib workers; `n_jobs` sets how many, and a given `random_state` gives the
same result at any worker count. Pairwise correlations are stored in condensed
(upper-triangle) form.

Leave-one-out and pairwise ISC are monotonically related but statistically
different: leave-one-out is O(n_subjects) and gives an unbiased subject-level
estimate; pairwise captures the full correlation structure but is
O(n_subjects²).

References:
    Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
    Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
    correlations, part I: nonparametric approaches to inter-subject
    correlation analysis at the group level. NeuroImage, 142, 248-259.
"""

import numpy as np
from typing import Literal, Any
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise_distances

from .utils import EPSILON, maybe_tqdm
from ..validation import _compute_pvalue, validate_tail_parameter


# ============================================================================
# Phase 1: Leave-One-Out (LOO) ISC Computation
# ============================================================================


def _compute_loo_isc(data):
    """Compute leave-one-out intersubject correlation.

    For each subject, correlates their data with the mean of all other
    subjects. This provides an unbiased estimate of subject-level ISC
    and is computationally efficient (O(n_subjects) vs O(n_subjects²)).

    Args:
        data (np.ndarray): Shape `(n_observations, n_subjects)` for a single
            feature or `(n_observations, n_subjects, n_voxels)` for voxel-wise
            data.

    Returns:
        np.ndarray: Leave-one-out ISC values, shape `(n_subjects,)` for a single
            feature or `(n_subjects, n_voxels)` for voxel-wise data.

    Raises:
        ValueError: If `data` is neither 2-D nor 3-D.

    Examples:
        ```python
        data = np.random.randn(100, 10)  # 100 timepoints, 10 subjects
        _compute_loo_isc(data).shape  # → (10,)
        ```
    """
    if data.ndim == 2:
        # Single feature: (n_observations, n_subjects)
        n_obs, n_subjects = data.shape
        loo_values = np.zeros(n_subjects)

        for i in range(n_subjects):
            # Mean of all subjects except i
            others_mean = data[:, np.arange(n_subjects) != i].mean(axis=1)
            # Correlation between subject i and others' mean
            loo_values[i] = np.corrcoef(data[:, i], others_mean)[0, 1]

        return loo_values

    if data.ndim == 3:
        # Voxel-wise: (n_observations, n_subjects, n_voxels)
        n_obs, n_subjects, n_voxels = data.shape
        loo_values = np.zeros((n_subjects, n_voxels))

        for v in range(n_voxels):
            voxel_data = data[:, :, v]
            for i in range(n_subjects):
                others_mean = voxel_data[:, np.arange(n_subjects) != i].mean(axis=1)
                loo_values[i, v] = np.corrcoef(voxel_data[:, i], others_mean)[0, 1]

        return loo_values

    raise ValueError(f"data must be 2D or 3D, got shape {data.shape}")


def _compute_pairwise_isc(data, metric="correlation"):
    """Compute pairwise intersubject correlation (condensed form).

    Computes all n×(n-1)/2 pairwise correlations between subjects and
    stores in condensed upper-triangle format for memory efficiency.

    `'correlation'`, `'spearman'` (rank-transform then `np.corrcoef`),
    `'cosine'` (normalized dot products), and `'euclidean'` (vectorized
    squared distances) take fast vectorized paths; any other metric falls back
    to the slower `sklearn.metrics.pairwise_distances`.

    Args:
        data (np.ndarray): Shape `(n_observations, n_subjects)` for a single
            feature or `(n_observations, n_subjects, n_voxels)` for voxel-wise
            data.
        metric (str): `'correlation'` (Pearson, default), `'spearman'`,
            `'cosine'`, `'euclidean'` (1 - distance), or any metric accepted by
            `sklearn.metrics.pairwise_distances`.

    Returns:
        np.ndarray: Pairwise similarities in condensed upper-triangle form,
            shape `(n_pairs,)` for a single feature (`n_pairs = n*(n-1)/2`) or
            `(n_pairs, n_voxels)` for voxel-wise data.

    Examples:
        ```python
        data = np.random.randn(100, 5)  # 5 subjects
        _compute_pairwise_isc(data).shape  # → (10,)  (5*4/2 pairs)
        ```
    """
    if metric == "correlation":
        # Fast path: use np.corrcoef (optimized C implementation)
        if data.ndim == 2:
            # Single feature: (n_observations, n_subjects)
            corr_matrix = np.corrcoef(data.T)
            return squareform(corr_matrix, checks=False)
        if data.ndim == 3:
            # Voxel-wise: (n_observations, n_subjects, n_voxels)
            n_obs, n_subjects, n_voxels = data.shape
            n_pairs = n_subjects * (n_subjects - 1) // 2
            pairwise_all = np.zeros((n_pairs, n_voxels))
            for v in range(n_voxels):
                corr_matrix = np.corrcoef(data[:, :, v].T)
                pairwise_all[:, v] = squareform(corr_matrix, checks=False)
            return pairwise_all
        raise ValueError(f"data must be 2D or 3D, got shape {data.shape}")
    if metric == "spearman":
        # Spearman correlation: rank-transform then use fast np.corrcoef path
        # Spearman = Pearson correlation of rank-transformed data
        if data.ndim == 2:
            # Single feature: (n_observations, n_subjects)
            # Rank-transform each subject's time series
            data_ranked = np.array(
                [rankdata(data[:, i], method="average") for i in range(data.shape[1])]
            ).T
            corr_matrix = np.corrcoef(data_ranked.T)
            return squareform(corr_matrix, checks=False)
        if data.ndim == 3:
            # Voxel-wise: (n_observations, n_subjects, n_voxels)
            n_obs, n_subjects, n_voxels = data.shape
            n_pairs = n_subjects * (n_subjects - 1) // 2
            pairwise_all = np.zeros((n_pairs, n_voxels))

            # Rank-transform data per voxel, then use fast corrcoef path
            for v in range(n_voxels):
                # Rank-transform each subject's time series for this voxel
                data_ranked = np.array(
                    [
                        rankdata(data[:, s, v], method="average")
                        for s in range(n_subjects)
                    ]
                ).T
                corr_matrix = np.corrcoef(data_ranked.T)
                pairwise_all[:, v] = squareform(corr_matrix, checks=False)
            return pairwise_all
        raise ValueError(f"data must be 2D or 3D, got shape {data.shape}")
    if metric == "cosine":
        # Cosine similarity: normalized dot products
        # Cosine similarity = dot(a, b) / (||a|| * ||b||)
        # Optimized: normalize vectors, then compute dot product matrix
        if data.ndim == 2:
            # Single feature: (n_observations, n_subjects)
            # Normalize each subject's time series
            norms = np.linalg.norm(data, axis=0, keepdims=True)
            data_norm = data / (norms + EPSILON)  # Avoid division by zero

            # Compute cosine similarity matrix: data_norm.T @ data_norm
            sim_matrix = data_norm.T @ data_norm
            return squareform(sim_matrix, checks=False)
        if data.ndim == 3:
            # Voxel-wise: (n_observations, n_subjects, n_voxels)
            n_obs, n_subjects, n_voxels = data.shape
            n_pairs = n_subjects * (n_subjects - 1) // 2
            pairwise_all = np.zeros((n_pairs, n_voxels))

            # Normalize and compute cosine similarity per voxel
            for v in range(n_voxels):
                # Normalize each subject's time series for this voxel
                norms = np.linalg.norm(data[:, :, v], axis=0, keepdims=True)
                data_norm = data[:, :, v] / (norms + EPSILON)

                # Compute cosine similarity matrix
                sim_matrix = data_norm.T @ data_norm
                pairwise_all[:, v] = squareform(sim_matrix, checks=False)
            return pairwise_all
        raise ValueError(f"data must be 2D or 3D, got shape {data.shape}")
    if metric == "euclidean":
        # Euclidean distance: optimized using squared-distance formula
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*<a, b>
        # Then: distance = sqrt(squared_distance), similarity = 1 - distance
        if data.ndim == 2:
            # Single feature: (n_observations, n_subjects)
            # Compute squared norms for each subject
            norms_sq = np.sum(data**2, axis=0)  # (n_subjects,)

            # Compute dot products: data.T @ data
            dot_products = data.T @ data  # (n_subjects, n_subjects)

            # Compute squared distances: norms_sq[i] + norms_sq[j] - 2*dot_products[i, j]
            # Broadcasting: (n_subjects, 1) + (1, n_subjects) - 2*dot_products
            distances_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dot_products

            # Compute distances (handle numerical errors with max(0))
            distances = np.sqrt(np.maximum(distances_sq, 0))

            # Convert to similarity: similarity = 1 - distance
            sim_matrix = 1 - distances
            return squareform(sim_matrix, checks=False)
        if data.ndim == 3:
            # Voxel-wise: (n_observations, n_subjects, n_voxels)
            n_obs, n_subjects, n_voxels = data.shape
            n_pairs = n_subjects * (n_subjects - 1) // 2
            pairwise_all = np.zeros((n_pairs, n_voxels))

            # Compute euclidean similarity per voxel
            for v in range(n_voxels):
                # Compute squared norms for each subject
                norms_sq = np.sum(data[:, :, v] ** 2, axis=0)  # (n_subjects,)

                # Compute dot products
                dot_products = (
                    data[:, :, v].T @ data[:, :, v]
                )  # (n_subjects, n_subjects)

                # Compute squared distances
                distances_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dot_products

                # Compute distances
                distances = np.sqrt(np.maximum(distances_sq, 0))

                # Convert to similarity
                sim_matrix = 1 - distances
                pairwise_all[:, v] = squareform(sim_matrix, checks=False)
            return pairwise_all
        raise ValueError(f"data must be 2D or 3D, got shape {data.shape}")
    # General path: use pairwise_distances for other metrics
    # Convert distance to similarity: similarity = 1 - distance
    if data.ndim == 2:
        # Single feature: (n_observations, n_subjects)
        dist_matrix = pairwise_distances(data.T, metric=metric)
        sim_matrix = 1 - dist_matrix
        return squareform(sim_matrix, checks=False)
    if data.ndim == 3:
        # Voxel-wise: (n_observations, n_subjects, n_voxels)
        n_obs, n_subjects, n_voxels = data.shape
        n_pairs = n_subjects * (n_subjects - 1) // 2
        pairwise_all = np.zeros((n_pairs, n_voxels))
        for v in range(n_voxels):
            dist_matrix = pairwise_distances(data[:, :, v].T, metric=metric)
            sim_matrix = 1 - dist_matrix
            pairwise_all[:, v] = squareform(sim_matrix, checks=False)
        return pairwise_all
    raise ValueError(f"data must be 2D or 3D, got shape {data.shape}")


def _compute_isc_group_difference(
    group1,
    group2,
    summary="median",
    summary_statistic="pairwise",
    metric="correlation",
):
    """Compute ISC difference between two groups.

    Computes intersubject correlation for each group separately, then takes
    the difference (group1 ISC - group2 ISC). Supports both pairwise and
    leave-one-out ISC computation methods.

    Args:
        group1 (np.ndarray): First group, shape `(n_observations, n_subjects1)`
            for a single feature or `(n_observations, n_subjects1, n_voxels)`
            for voxel-wise data.
        group2 (np.ndarray): Second group, shape `(n_observations, n_subjects2)`
            or `(n_observations, n_subjects2, n_voxels)`.
        summary (str): How ISC values are aggregated: `'median'` (default,
            robust to outliers) or `'mean'` (Fisher z-transformed mean).
        summary_statistic (str): `'pairwise'` (default; summarize all pairwise
            correlations) or `'leave-one-out'` (correlate each subject with
            the mean of the others).
        metric (str): Similarity metric for pairwise ISC. Defaults to
            `'correlation'`.

    Returns:
        np.ndarray: `group1 ISC - group2 ISC`, shape `()` for a single feature
            or `(n_voxels,)` for voxel-wise data.

    Examples:
        ```python
        group1 = np.random.randn(100, 5)  # 5 subjects
        group2 = np.random.randn(100, 5)
        _compute_isc_group_difference(group1, group2).shape  # → ()

        # Voxel-wise
        group1_voxels = np.random.randn(100, 5, 1000)
        group2_voxels = np.random.randn(100, 5, 1000)
        _compute_isc_group_difference(group1_voxels, group2_voxels).shape  # → (1000,)
        ```
    """
    # Input validation
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    if group1.shape[0] != group2.shape[0]:
        raise ValueError(
            "group1 and group2 must have the same number of observations. "
            f"Got group1.shape[0]={group1.shape[0]}, group2.shape[0]={group2.shape[0]}"
        )

    if group1.ndim != group2.ndim:
        raise ValueError(
            "group1 and group2 must have the same number of dimensions. "
            f"Got group1.ndim={group1.ndim}, group2.ndim={group2.ndim}"
        )

    if group1.ndim not in [2, 3]:
        raise ValueError(
            f"group1 and group2 must be 2D or 3D, got shapes {group1.shape}, {group2.shape}"
        )

    if summary not in ["median", "mean"]:
        raise ValueError(f"summary must be 'median' or 'mean', got {summary}")

    if summary_statistic not in ["pairwise", "leave-one-out"]:
        raise ValueError(
            f"summary_statistic must be 'pairwise' or 'leave-one-out', got {summary_statistic}"
        )

    # Compute ISC for each group
    if summary_statistic == "pairwise":
        # Pairwise ISC: compute condensed correlation matrices
        isc1_values = _compute_pairwise_isc(group1, metric=metric)
        isc2_values = _compute_pairwise_isc(group2, metric=metric)

        # Handle single feature vs voxel-wise
        if isc1_values.ndim == 1:
            # Single feature: (n_pairs,)
            axis = None
        else:
            # Voxel-wise: (n_pairs, n_voxels)
            axis = 0

        # Compute summary statistic
        if summary == "median":
            isc1 = np.nanmedian(isc1_values, axis=axis)
            isc2 = np.nanmedian(isc2_values, axis=axis)
        elif summary == "mean":
            # Fisher z-transform
            z1 = np.arctanh(np.clip(isc1_values, -0.9999, 0.9999))
            z2 = np.arctanh(np.clip(isc2_values, -0.9999, 0.9999))
            isc1 = np.tanh(np.nanmean(z1, axis=axis))
            isc2 = np.tanh(np.nanmean(z2, axis=axis))

    else:  # leave-one-out
        # LOO ISC: compute LOO values for each subject
        loo1_values = _compute_loo_isc(group1)
        loo2_values = _compute_loo_isc(group2)

        # Handle single feature vs voxel-wise
        if loo1_values.ndim == 1:
            # Single feature: (n_subjects,)
            axis = 0
        else:
            # Voxel-wise: (n_subjects, n_voxels)
            axis = 0

        # Compute summary statistic
        if summary == "median":
            isc1 = np.median(loo1_values, axis=axis)
            isc2 = np.median(loo2_values, axis=axis)
        elif summary == "mean":
            # Fisher z-transform
            z1 = np.arctanh(np.clip(loo1_values, -0.9999, 0.9999))
            z2 = np.arctanh(np.clip(loo2_values, -0.9999, 0.9999))
            isc1 = np.tanh(np.mean(z1, axis=axis))
            isc2 = np.tanh(np.mean(z2, axis=axis))

    # Compute difference
    isc_diff = isc1 - isc2

    return isc_diff


# ============================================================================
# Phase 2.6: ISC Group Permutation (Subject-wise Permutation)
# ============================================================================


def _permute_isc_group_numpy(
    group1,
    group2,
    summary="median",
    summary_statistic="pairwise",
    random_state=None,
    metric="correlation",
):
    """Single permutation: permute group labels and compute ISC difference.

    Implements the subject-wise permutation method from Chen et al. (2016).
    Combines the two groups, permutes group labels, then computes ISC difference
    for the permuted groups.

    Args:
        group1 (np.ndarray): First group, shape `(n_observations, n_subjects1)`
            or `(n_observations, n_subjects1, n_voxels)`.
        group2 (np.ndarray): Second group, shape `(n_observations, n_subjects2)`
            or `(n_observations, n_subjects2, n_voxels)`.
        summary (str): `'median'` (default) or `'mean'`.
        summary_statistic (str): `'pairwise'` (default) or `'leave-one-out'`.
        random_state (int | np.random.RandomState | None): Random state for
            reproducibility.
        metric (str): Similarity metric for pairwise ISC. Defaults to
            `'correlation'`.

    Returns:
        np.ndarray: Permuted ISC difference, shape `()` or `(n_voxels,)`.
    """
    from sklearn.utils import check_random_state

    rng = check_random_state(random_state)

    # Combine groups
    combined = np.concatenate([group1, group2], axis=1)
    n_subjects1 = group1.shape[1]

    # Create group labels
    n_subjects_total = combined.shape[1]
    group_labels = np.array([1] * n_subjects1 + [2] * (n_subjects_total - n_subjects1))

    # Permute group labels
    permuted_labels = rng.permutation(group_labels)

    # Split back into groups based on permuted labels
    group1_id, group2_id = 1, 2
    group1_perm = combined[:, permuted_labels == group1_id]
    group2_perm = combined[:, permuted_labels == group2_id]

    # Compute ISC difference for permuted groups
    isc_diff = _compute_isc_group_difference(
        group1_perm,
        group2_perm,
        summary=summary,
        summary_statistic=summary_statistic,
        metric=metric,
    )

    return isc_diff


def _permute_isc_group_cpu_parallel(
    group1,
    group2,
    *,
    n_permute=5000,
    summary="median",
    summary_statistic="pairwise",
    n_jobs=-1,
    random_state=None,
    progress_bar=False,
    metric="correlation",
    max_memory_gb=None,
):
    """CPU-parallel permutation for ISC group difference.

    Efficiently parallelizes permutation resampling across CPU cores using joblib.
    Uses deterministic seed generation for reproducibility.
    Automatically limits workers based on available memory if n_jobs=-1.

    Args:
        group1 (np.ndarray): First group, shape `(n_observations, n_subjects1)`
            or `(n_observations, n_subjects1, n_voxels)`.
        group2 (np.ndarray): Second group, shape `(n_observations, n_subjects2)`
            or `(n_observations, n_subjects2, n_voxels)`.
        n_permute (int): Number of permutations. Defaults to 5000.
        summary (str): `'median'` (default) or `'mean'`.
        summary_statistic (str): `'pairwise'` (default) or `'leave-one-out'`.
        n_jobs (int): CPU cores; -1 (default) picks the worker count from
            available memory.
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Show a progress bar. Defaults to False.
        metric (str): Similarity metric for pairwise ISC. Defaults to
            `'correlation'`.
        max_memory_gb (float | None): Memory budget in GB for the `n_jobs=-1`
            auto-detection; None measures the machine.

    Returns:
        np.ndarray: Permuted ISC differences, shape `(n_permute,)` for a single
            feature or `(n_permute, n_voxels)` for voxel-wise data.
    """
    from joblib import Parallel, delayed
    from nltools.algorithms.backends import _auto_n_jobs_cpu, _estimate_data_size_mb

    # Auto-detect optimal n_jobs based on memory if n_jobs=-1
    # Estimate memory for combined groups
    if n_jobs == -1:
        combined_size_mb = _estimate_data_size_mb(group1) + _estimate_data_size_mb(
            group2
        )
        n_jobs = _auto_n_jobs_cpu(
            data_size_mb=combined_size_mb,
            n_permute=n_permute,
            max_memory_gb=max_memory_gb,
        )

    rng = check_random_state(random_state)
    MAX_INT = 2**31 - 1
    seeds = rng.randint(MAX_INT, size=n_permute)

    # Parallelize
    iterator = maybe_tqdm(
        range(n_permute), progress_bar=progress_bar, desc="Permute ISC Group"
    )

    permutations = Parallel(n_jobs=n_jobs)(
        delayed(_permute_isc_group_numpy)(
            group1,
            group2,
            summary=summary,
            summary_statistic=summary_statistic,
            random_state=np.random.RandomState(seeds[i]),
            metric=metric,
        )
        for i in iterator
    )

    return np.array(permutations)


# ============================================================================
# Phase 2.7: ISC Group Bootstrap (Subject-wise Bootstrap)
# ============================================================================


def _bootstrap_isc_group_numpy(
    group1,
    group2,
    observed_diff,
    summary="median",
    summary_statistic="pairwise",
    exclude_self_corr=True,
    random_state=None,
    metric="correlation",
):
    """Single bootstrap: resample subjects within each group and compute ISC difference.

    Implements the subject-wise bootstrap method from Chen et al. (2016).
    Bootstraps each group independently, then computes ISC difference.
    Centers by subtracting observed difference: (boot1 - boot2) - observed_diff.

    Args:
        group1 (np.ndarray): First group, shape `(n_observations, n_subjects1)`
            or `(n_observations, n_subjects1, n_voxels)`.
        group2 (np.ndarray): Second group, shape `(n_observations, n_subjects2)`
            or `(n_observations, n_subjects2, n_voxels)`.
        observed_diff (float | np.ndarray): Observed ISC difference, subtracted
            to center the draw.
        summary (str): `'median'` (default) or `'mean'`.
        summary_statistic (str): `'pairwise'` (default) or `'leave-one-out'`.
        exclude_self_corr (bool): Mask the perfect correlations a duplicated
            subject produces (pairwise only). Defaults to True.
        random_state (int | np.random.RandomState | None): Random state for
            reproducibility.
        metric (str): Similarity metric for pairwise ISC. Defaults to
            `'correlation'`.

    Returns:
        np.ndarray: Centered bootstrap difference,
            `(boot1 - boot2) - observed_diff`, shape `()` or `(n_voxels,)`.
    """
    from sklearn.utils import check_random_state

    rng = check_random_state(random_state)

    if summary_statistic == "pairwise":
        # Pairwise bootstrap: resample subjects, recompute pairwise ISC
        n_subjects1 = group1.shape[1]
        n_subjects2 = group2.shape[1]

        # Bootstrap subjects for each group
        boot_indices1 = rng.choice(n_subjects1, size=n_subjects1, replace=True)
        boot_indices2 = rng.choice(n_subjects2, size=n_subjects2, replace=True)

        # Resample data
        if group1.ndim == 2:
            group1_boot = group1[:, boot_indices1]
            group2_boot = group2[:, boot_indices2]
        else:
            group1_boot = group1[:, boot_indices1, :]
            group2_boot = group2[:, boot_indices2, :]

        # Compute pairwise ISC for bootstrapped groups
        pairwise1_boot = _compute_pairwise_isc(group1_boot, metric=metric)
        pairwise2_boot = _compute_pairwise_isc(group2_boot, metric=metric)

        # Handle exclude_self_corr: mask perfect correlations from duplicate subjects
        if exclude_self_corr:
            # Mask correlations >= 0.99999 (perfect correlations from duplicates)
            pairwise1_boot = np.where(
                np.abs(pairwise1_boot) >= 0.99999, np.nan, pairwise1_boot
            )
            pairwise2_boot = np.where(
                np.abs(pairwise2_boot) >= 0.99999, np.nan, pairwise2_boot
            )

        # Handle single feature vs voxel-wise
        if pairwise1_boot.ndim == 1:
            axis = None
        else:
            axis = 0

        # Compute summary statistic
        if summary == "median":
            isc1_boot = np.nanmedian(pairwise1_boot, axis=axis)
            isc2_boot = np.nanmedian(pairwise2_boot, axis=axis)
        elif summary == "mean":
            z1 = np.arctanh(np.clip(pairwise1_boot, -0.9999, 0.9999))
            z2 = np.arctanh(np.clip(pairwise2_boot, -0.9999, 0.9999))
            isc1_boot = np.tanh(np.nanmean(z1, axis=axis))
            isc2_boot = np.tanh(np.nanmean(z2, axis=axis))

    else:  # leave-one-out
        # LOO bootstrap: resample pre-computed LOO values
        loo1_values = _compute_loo_isc(group1)
        loo2_values = _compute_loo_isc(group2)

        # Bootstrap LOO values
        isc1_boot = _bootstrap_loo_numpy(loo1_values, summary=summary, random_state=rng)
        # Use different seed for group2 to ensure independence
        rng2 = check_random_state(rng.randint(0, 2**31 - 1))
        isc2_boot = _bootstrap_loo_numpy(
            loo2_values, summary=summary, random_state=rng2
        )

    # Compute difference and center
    boot_diff = (isc1_boot - isc2_boot) - observed_diff

    return boot_diff


def _bootstrap_isc_group_cpu_parallel(
    group1,
    group2,
    observed_diff,
    *,
    n_permute=5000,
    summary="median",
    summary_statistic="pairwise",
    exclude_self_corr=True,
    n_jobs=-1,
    random_state=None,
    progress_bar=False,
    metric="correlation",
    max_memory_gb=None,
):
    """CPU-parallel bootstrap for ISC group difference.

    Efficiently parallelizes bootstrap resampling across CPU cores using joblib.
    Uses deterministic seed generation for reproducibility.
    Automatically limits workers based on available memory if n_jobs=-1.

    Args:
        group1 (np.ndarray): First group, shape `(n_observations, n_subjects1)`
            or `(n_observations, n_subjects1, n_voxels)`.
        group2 (np.ndarray): Second group, shape `(n_observations, n_subjects2)`
            or `(n_observations, n_subjects2, n_voxels)`.
        observed_diff (float | np.ndarray): Observed ISC difference, subtracted
            to center each draw.
        n_permute (int): Number of bootstrap iterations. Defaults to 5000.
        summary (str): `'median'` (default) or `'mean'`.
        summary_statistic (str): `'pairwise'` (default) or `'leave-one-out'`.
        exclude_self_corr (bool): Mask the perfect correlations a duplicated
            subject produces (pairwise only). Defaults to True.
        n_jobs (int): CPU cores; -1 (default) picks the worker count from
            available memory.
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Show a progress bar. Defaults to False.
        metric (str): Similarity metric for pairwise ISC. Defaults to
            `'correlation'`.
        max_memory_gb (float | None): Memory budget in GB for the `n_jobs=-1`
            auto-detection; None measures the machine.

    Returns:
        np.ndarray: Centered bootstrap differences, shape `(n_permute,)` for a
            single feature or `(n_permute, n_voxels)` for voxel-wise data.
    """
    from joblib import Parallel, delayed
    from nltools.algorithms.backends import _auto_n_jobs_cpu, _estimate_data_size_mb

    # Auto-detect optimal n_jobs based on memory if n_jobs=-1
    # Estimate memory for combined groups
    if n_jobs == -1:
        combined_size_mb = _estimate_data_size_mb(group1) + _estimate_data_size_mb(
            group2
        )
        n_jobs = _auto_n_jobs_cpu(
            data_size_mb=combined_size_mb,
            n_permute=n_permute,
            max_memory_gb=max_memory_gb,
        )

    rng = check_random_state(random_state)
    MAX_INT = 2**31 - 1
    seeds = rng.randint(MAX_INT, size=n_permute)

    # Parallelize
    iterator = maybe_tqdm(
        range(n_permute), progress_bar=progress_bar, desc="Bootstrap ISC Group"
    )

    bootstraps = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_isc_group_numpy)(
            group1,
            group2,
            observed_diff=observed_diff,
            summary=summary,
            summary_statistic=summary_statistic,
            exclude_self_corr=exclude_self_corr,
            random_state=np.random.RandomState(seeds[i]),
            metric=metric,
        )
        for i in iterator
    )

    return np.array(bootstraps)


# ============================================================================
# Phase 2.8: Main ISC Group Permutation Test Function
# ============================================================================


def isc_group_permutation_test(
    group1: np.ndarray,
    group2: np.ndarray,
    *,
    n_permute: int = 5000,
    summary: Literal["median", "mean"] = "median",
    method: Literal["permute", "bootstrap"] = "permute",
    summary_statistic: Literal["leave-one-out", "pairwise"] = "pairwise",
    ci_percentile: float = 95,
    tail: int | str = 2,
    n_jobs: int = -1,
    random_state: int | None = None,
    return_null: bool = False,
    progress_bar: bool = False,
    exclude_self_corr: bool = True,
    metric: str = "correlation",
) -> dict[str, Any]:
    """Test the difference in intersubject correlation between two groups.

    Computes ISC within each group, takes `group1 - group2`, and builds a null
    distribution by either subject-wise permutation (pool the subjects and
    reshuffle the group labels — the Chen et al. 2016 recommendation) or
    subject-wise bootstrap (resample subjects within each group; the bootstrap
    draws are centered on the observed difference before the p-value is
    computed). The confidence interval brackets the observed difference for
    `method='bootstrap'` and describes the null spread for `method='permute'`.

    Args:
        group1 (np.ndarray): First group, shape `(n_observations, n_subjects1)`
            for a single feature or `(n_observations, n_subjects1, n_voxels)`
            for voxel-wise data.
        group2 (np.ndarray): Second group, shape `(n_observations, n_subjects2)`
            or `(n_observations, n_subjects2, n_voxels)`; `n_observations` must
            match `group1`.
        n_permute (int): Number of permutations or bootstrap draws. Defaults to
            5000.
        summary (str): How ISC values are aggregated: `'median'` (default,
            robust to outliers) or `'mean'` (Fisher z-transformed mean).
        method (str): `'permute'` (default; pool subjects and permute labels) or
            `'bootstrap'` (resample subjects within each group).
        summary_statistic (str): `'pairwise'` (default; summarize all pairwise
            correlations) or `'leave-one-out'` (correlate each subject with the
            mean of the others).
        ci_percentile (float): Confidence-interval width in percent (95 gives a
            95% CI). Defaults to 95.
        tail (int | str): `2` or `'two'` (default) for a two-tailed p-value;
            `1` or `'one'` for one-tailed (group1 > group2).
        n_jobs (int): Number of joblib workers for the resamples. -1 (default)
            picks the worker count from available memory. Results are identical
            at every worker count.
        random_state (int | None): Random seed for reproducibility.
        return_null (bool): If True, include the null distribution in the
            result. Defaults to False.
        progress_bar (bool): Show a progress bar over the resamples. Defaults to
            False.
        exclude_self_corr (bool): In the bootstrap, mask the perfect
            correlations a duplicated subject produces (pairwise only).
            Defaults to True.
        metric (str): Similarity metric for pairwise ISC; any metric accepted by
            `sklearn.metrics.pairwise_distances`. Ignored for
            `summary_statistic='leave-one-out'`. Defaults to `'correlation'`.

    Returns:
        dict: Keys `'isc_group_difference'` (float or np.ndarray, observed
            difference), `'p'` (float or np.ndarray, p-value with the
            `(count + 1) / (n + 1)` correction), `'ci'` (tuple
            `(lower, upper)`), and — when `return_null=True` — `'null_dist'`
            (np.ndarray).

    Examples:
        ```python
        # Single-feature comparison
        group1 = np.random.randn(100, 10)  # 10 subjects
        group2 = np.random.randn(100, 10)
        result = isc_group_permutation_test(group1, group2, n_permute=1000)
        result["isc_group_difference"], result["p"]

        # Voxel-wise comparison
        group1_voxels = np.random.randn(100, 10, 5000)  # 5K voxels
        group2_voxels = np.random.randn(100, 10, 5000)
        result = isc_group_permutation_test(
            group1_voxels,
            group2_voxels,
            summary_statistic="leave-one-out",
            n_permute=5000,
        )
        (result["p"] < 0.05).sum()  # → number of significant voxels
        ```

    References:
        Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
        Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
        correlations, part I: nonparametric approaches to inter-subject
        correlation analysis at the group level. NeuroImage, 142, 248-259.
    """
    # Input validation
    validate_tail_parameter(tail)
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)

    if group1.shape[0] != group2.shape[0]:
        raise ValueError(
            "group1 and group2 must have the same number of observations. "
            f"Got group1.shape[0]={group1.shape[0]}, group2.shape[0]={group2.shape[0]}"
        )

    if group1.ndim != group2.ndim:
        raise ValueError(
            "group1 and group2 must have the same number of dimensions. "
            f"Got group1.ndim={group1.ndim}, group2.ndim={group2.ndim}"
        )

    if group1.ndim not in [2, 3]:
        raise ValueError(
            f"group1 and group2 must be 2D or 3D, got shapes {group1.shape}, {group2.shape}"
        )

    if summary not in ["median", "mean"]:
        raise ValueError(f"summary must be 'median' or 'mean', got {summary}")

    if method not in ["permute", "bootstrap"]:
        raise ValueError(f"method must be 'permute' or 'bootstrap', got {method}")

    if summary_statistic not in ["pairwise", "leave-one-out"]:
        raise ValueError(
            f"summary_statistic must be 'pairwise' or 'leave-one-out', got {summary_statistic}"
        )

    # Phase 1: Compute observed ISC difference
    observed_diff = _compute_isc_group_difference(
        group1,
        group2,
        summary=summary,
        summary_statistic=summary_statistic,
        metric=metric,
    )

    # Phase 2: Bootstrap/Permutation (run n_permute times)
    if method == "permute":
        null_dist = _permute_isc_group_cpu_parallel(
            group1,
            group2,
            n_permute=n_permute,
            summary=summary,
            summary_statistic=summary_statistic,
            n_jobs=n_jobs,
            random_state=random_state,
            progress_bar=progress_bar,
            metric=metric,
        )
    else:  # bootstrap
        null_dist = _bootstrap_isc_group_cpu_parallel(
            group1,
            group2,
            observed_diff=observed_diff,
            n_permute=n_permute,
            summary=summary,
            summary_statistic=summary_statistic,
            exclude_self_corr=exclude_self_corr,
            n_jobs=n_jobs,
            random_state=random_state,
            progress_bar=progress_bar,
            metric=metric,
            max_memory_gb=None,  # Auto-detect
        )

    # Handle NaN values (from exclude_self_corr masking)
    # For single feature: remove all NaN values
    # For voxel-wise: keep NaN per voxel (they represent valid bootstrap samples)
    if null_dist.ndim == 1:
        # Single feature: filter out NaN values
        null_dist = null_dist[~np.isnan(null_dist)]
    # For voxel-wise, keep NaN values (they're handled by nanpercentile)

    # Phase 3: Compute p-value and confidence interval
    # Handle scalar vs array observed_diff
    if isinstance(observed_diff, np.ndarray) and observed_diff.ndim > 0:
        # Voxel-wise: (n_voxels,)
        if null_dist.ndim == 1:
            # This shouldn't happen - voxel-wise should produce 2D null_dist
            raise ValueError("Voxel-wise data should produce 2D null_dist")
        # null_dist shape: (n_permute, n_voxels)
        p_values = _compute_pvalue(observed_diff, null_dist, tail=tail)
    else:
        # Single feature: scalar observed_diff
        if null_dist.ndim == 1:
            # null_dist shape: (n_permute,)
            p_values = _compute_pvalue(
                np.array([observed_diff]), null_dist.reshape(-1, 1), tail=tail
            )[0]
        else:
            # Shouldn't happen for single feature
            raise ValueError("Single feature should produce 1D null_dist")

    # Compute confidence intervals.
    # For method='bootstrap' the draws in null_dist are CENTERED
    # (boot_estimate - observed_diff) so that the permutation-style p-value tests
    # against H0: difference == 0. A confidence interval, however, must bracket
    # the ESTIMATE, so we re-add observed_diff to recover the uncentered
    # bootstrap distribution before taking percentiles (matching
    # isc_permutation_test). For method='permute' the null is a label-permutation
    # band around zero, not a bootstrap of the estimate; its percentiles describe
    # the null spread and are left uncentered.
    ci_source = null_dist + observed_diff if method == "bootstrap" else null_dist
    if ci_source.ndim == 1:
        ci_lower = np.percentile(ci_source, (100 - ci_percentile) / 2)
        ci_upper = np.percentile(ci_source, ci_percentile + (100 - ci_percentile) / 2)
    else:
        # Voxel-wise: compute CI per voxel
        ci_lower = np.nanpercentile(ci_source, (100 - ci_percentile) / 2, axis=0)
        ci_upper = np.nanpercentile(
            ci_source, ci_percentile + (100 - ci_percentile) / 2, axis=0
        )

    # Build result dictionary
    result = {
        "isc_group_difference": observed_diff,
        "p": p_values,
        "ci": (ci_lower, ci_upper),
    }

    if return_null:
        result["null_dist"] = null_dist

    return result


# ============================================================================
# Phase 3: Leave-One-Out Bootstrap
# ============================================================================


def _bootstrap_loo_numpy(loo_values, summary="median", random_state=None):
    """Bootstrap LOO ISC by resampling subjects.

    Implements the subject-wise bootstrap method from Chen et al. (2016).
    Resamples the pre-computed LOO values (not raw data) for efficiency.

    Args:
        loo_values (np.ndarray): Pre-computed LOO values, shape `(n_subjects,)`
            for a single feature or `(n_subjects, n_voxels)` for voxel-wise data.
        summary (str): `'median'` (default) or `'mean'` (Fisher z-transformed:
            arctanh → mean → tanh).
        random_state (int | np.random.RandomState | None): Random state for
            reproducibility.

    Returns:
        np.ndarray: Bootstrap summary statistic, shape `()` or `(n_voxels,)`.
    """
    rng = check_random_state(random_state)
    n_subjects = loo_values.shape[0]

    # Sample subjects with replacement
    indices = rng.choice(n_subjects, size=n_subjects, replace=True)

    # Resample LOO values
    if loo_values.ndim == 1:
        boot_values = loo_values[indices]
    else:
        # Voxel-wise: (n_subjects, n_voxels)
        boot_values = loo_values[indices, :]

    # Compute summary statistic
    if summary == "median":
        return np.median(boot_values, axis=0)
    if summary == "mean":
        # Fisher z-transform for unbiased mean
        z = np.arctanh(np.clip(boot_values, -0.9999, 0.9999))
        return np.tanh(np.mean(z, axis=0))
    raise ValueError(f"summary must be 'median' or 'mean', got {summary}")


def _bootstrap_loo_cpu_parallel(
    loo_values,
    *,
    n_permute=5000,
    summary="median",
    n_jobs=-1,
    random_state=None,
    progress_bar=False,
    max_memory_gb=None,
):
    """CPU-parallel LOO bootstrap using joblib.

    Efficiently parallelizes bootstrap resampling across CPU cores.
    Uses deterministic seed generation for reproducibility.
    Automatically limits workers based on available memory if n_jobs=-1.

    Args:
        loo_values (np.ndarray): Pre-computed LOO values, shape `(n_subjects,)`
            or `(n_subjects, n_voxels)`.
        n_permute (int): Number of bootstrap iterations. Defaults to 5000.
        summary (str): `'median'` (default) or `'mean'`.
        n_jobs (int): CPU cores; -1 (default) picks the worker count from
            available memory.
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Show a progress bar. Defaults to False.
        max_memory_gb (float | None): Memory budget in GB for the `n_jobs=-1`
            auto-detection; None measures the machine.

    Returns:
        np.ndarray: Bootstrap distribution, shape `(n_permute,)` or
            `(n_permute, n_voxels)`.
    """
    from joblib import Parallel, delayed
    from nltools.algorithms.backends import _auto_n_jobs_cpu, _estimate_data_size_mb

    # Auto-detect optimal n_jobs based on memory if n_jobs=-1
    if n_jobs == -1:
        data_size_mb = _estimate_data_size_mb(loo_values)
        n_jobs = _auto_n_jobs_cpu(
            data_size_mb=data_size_mb,
            n_permute=n_permute,
            max_memory_gb=max_memory_gb,
        )

    # Pre-generate seeds for deterministic parallelization
    rng = check_random_state(random_state)
    seeds = rng.randint(0, 2**31 - 1, size=n_permute)

    # Parallelize with independent RandomState per permutation
    iterator = maybe_tqdm(
        range(n_permute), progress_bar=progress_bar, desc="Bootstrap LOO"
    )

    bootstraps = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_loo_numpy)(
            loo_values,
            summary=summary,
            random_state=np.random.RandomState(seeds[i]),
        )
        for i in iterator
    )

    return np.array(bootstraps)


# ============================================================================
# Phase 4: Pairwise Bootstrap
# ============================================================================


def _bootstrap_pairwise_numpy(
    pairwise_condensed,
    summary="median",
    bootstrap_subjects=None,
    n_subjects=None,
    random_state=None,
    exclude_self_corr=True,
):
    """Bootstrap pairwise ISC by subject-wise matrix indexing.

    Implements the correct bootstrap procedure for correlation matrices
    (Chen et al. 2016): resample subjects, extract submatrix, mask
    same-subject pairs (self-correlations from duplicates).

    A subject drawn twice correlates perfectly with itself; with
    `exclude_self_corr=True` those entries are masked as NaN before the summary,
    as Chen et al. (2016) recommend.

    Args:
        pairwise_condensed (np.ndarray): Pre-computed pairwise correlations in
            condensed form, shape `(n_pairs,)` for a single feature or
            `(n_pairs, n_voxels)` for voxel-wise data.
        summary (str): `'median'` (default) or `'mean'`.
        bootstrap_subjects (np.ndarray | None): Pre-drawn subject indices (for
            testing); drawn from `random_state` when None.
        n_subjects (int | None): Number of subjects; required when
            `bootstrap_subjects` is None.
        random_state (int | np.random.RandomState | None): Random state for
            sampling.
        exclude_self_corr (bool): Mask self-correlations as NaN. Defaults to
            True.

    Returns:
        np.ndarray: Bootstrap summary statistic, shape `()` or `(n_voxels,)`.
    """
    if bootstrap_subjects is None:
        if n_subjects is None:
            raise ValueError("Must provide either bootstrap_subjects or n_subjects")
        rng = check_random_state(random_state)
        bootstrap_subjects = rng.choice(n_subjects, size=n_subjects, replace=True)
    else:
        n_subjects = len(bootstrap_subjects)

    # Handle single feature vs voxel-wise
    if pairwise_condensed.ndim == 1:
        # Single feature
        # Reconstruct correlation matrix from condensed form
        corr_matrix = squareform(pairwise_condensed, force="tomatrix")
        np.fill_diagonal(corr_matrix, 1.0)

        # Index by bootstrap subjects (symmetric: rows and columns)
        boot_matrix = corr_matrix[bootstrap_subjects, :][:, bootstrap_subjects]

        # Mask self-correlations if requested
        if exclude_self_corr:
            boot_matrix[boot_matrix >= 0.99999] = np.nan

        # Extract upper triangle (excluding diagonal)
        boot_condensed = squareform(boot_matrix, checks=False)

    else:
        # Voxel-wise: (n_pairs, n_voxels)
        n_pairs, n_voxels = pairwise_condensed.shape

        # Vectorized approach: process all voxels at once using matrix operations
        # Instead of looping over voxels, we can vectorize the squareform operations
        # by building all matrices at once and using advanced indexing

        # Build all correlation matrices at once: (n_subjects, n_subjects, n_voxels)
        # This is more memory-intensive but much faster
        corr_matrices = np.zeros(
            (n_subjects, n_subjects, n_voxels), dtype=pairwise_condensed.dtype
        )
        for v in range(n_voxels):
            corr_matrix = squareform(pairwise_condensed[:, v], force="tomatrix")
            np.fill_diagonal(corr_matrix, 1.0)
            corr_matrices[:, :, v] = corr_matrix

        # Index by bootstrap subjects for all voxels at once
        # Shape: (n_subjects, n_subjects, n_voxels)
        boot_matrices = corr_matrices[bootstrap_subjects, :, :][
            :, bootstrap_subjects, :
        ]

        # Mask self-correlations if requested (vectorized across all voxels)
        if exclude_self_corr:
            boot_matrices[boot_matrices >= 0.99999] = np.nan

        # Extract upper triangle for all voxels
        boot_condensed = np.zeros((n_pairs, n_voxels), dtype=pairwise_condensed.dtype)
        for v in range(n_voxels):
            boot_condensed[:, v] = squareform(boot_matrices[:, :, v], checks=False)

    # Compute summary (ignoring NaNs from masked pairs)
    axis = 0 if boot_condensed.ndim > 1 else None

    if summary == "median":
        return np.nanmedian(boot_condensed, axis=axis)
    if summary == "mean":
        # Fisher z-transform
        z = np.arctanh(np.clip(boot_condensed, -0.9999, 0.9999))
        return np.tanh(np.nanmean(z, axis=axis))
    raise ValueError(f"summary must be 'median' or 'mean', got {summary}")


def _bootstrap_pairwise_cpu_parallel(
    pairwise_condensed,
    *,
    n_permute=5000,
    n_subjects=None,
    summary="median",
    n_jobs=-1,
    random_state=None,
    progress_bar=False,
    exclude_self_corr=True,
    max_memory_gb=None,
):
    """CPU-parallel pairwise bootstrap using joblib.

    Same pattern as LOO bootstrap, but operates on pairwise correlation
    matrices with subject-wise indexing.
    Automatically limits workers based on available memory if n_jobs=-1.

    Args:
        pairwise_condensed (np.ndarray): Pre-computed pairwise correlations,
            shape `(n_pairs,)` or `(n_pairs, n_voxels)`.
        n_permute (int): Number of bootstrap iterations. Defaults to 5000.
        n_subjects (int): Number of subjects in the original data.
        summary (str): `'median'` (default) or `'mean'`.
        n_jobs (int): CPU cores; -1 (default) picks the worker count from
            available memory.
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Show a progress bar. Defaults to False.
        exclude_self_corr (bool): Mask self-correlations as NaN. Defaults to
            True.
        max_memory_gb (float | None): Memory budget in GB for the `n_jobs=-1`
            auto-detection; None measures the machine.

    Returns:
        np.ndarray: Bootstrap distribution, shape `(n_permute,)` or
            `(n_permute, n_voxels)`.
    """
    from joblib import Parallel, delayed
    from nltools.algorithms.backends import _auto_n_jobs_cpu, _estimate_data_size_mb

    if n_subjects is None:
        raise ValueError("n_subjects is required for pairwise bootstrap")

    # Auto-detect optimal n_jobs based on memory if n_jobs=-1
    if n_jobs == -1:
        data_size_mb = _estimate_data_size_mb(pairwise_condensed)
        n_jobs = _auto_n_jobs_cpu(
            data_size_mb=data_size_mb,
            n_permute=n_permute,
            max_memory_gb=max_memory_gb,
        )

    # Pre-generate seeds
    rng = check_random_state(random_state)
    seeds = rng.randint(0, 2**31 - 1, size=n_permute)

    # Parallelize
    iterator = maybe_tqdm(
        range(n_permute), progress_bar=progress_bar, desc="Bootstrap Pairwise"
    )

    bootstraps = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_pairwise_numpy)(
            pairwise_condensed,
            summary=summary,
            n_subjects=n_subjects,
            random_state=np.random.RandomState(seeds[i]),
            exclude_self_corr=exclude_self_corr,
        )
        for i in iterator
    )

    return np.array(bootstraps)


def isc_permutation_test(
    # Required
    data: np.ndarray,
    *,
    # Optional algorithm parameters
    n_permute: int = 5000,
    summary: Literal["median", "mean"] = "median",
    summary_statistic: Literal["leave-one-out", "pairwise"] = "pairwise",
    method: Literal["bootstrap", "circle_shift", "phase_randomize"] = "bootstrap",
    ci_percentile: float = 95,
    tail: int | str = 2,
    return_null: bool = False,
    progress_bar: bool = False,
    exclude_self_corr: bool = True,
    metric: str = "correlation",
    # Backend parameters (grouped)
    n_jobs: int = -1,
    # Random state (last)
    random_state: int | None = None,
) -> dict[str, Any]:
    """Compute intersubject correlation with bootstrap or permutation inference.

    Summarizes how similarly subjects respond over time — either leave-one-out
    (each subject against the mean of the others; O(n_subjects), unbiased) or
    pairwise (all subject pairs; O(n_subjects²), full correlation structure).
    The two are monotonically but non-linearly related and statistically
    different (Chen et al. 2016, Figure 3). The null distribution comes from a
    subject-wise bootstrap (centered on the observed ISC, so the p-value tests
    H0: ISC = 0) or from surrogate time series that preserve each subject's
    autocorrelation (circular shift) or power spectrum (phase randomization).

    Args:
        data (np.ndarray): Shape `(n_observations, n_subjects)` for a single
            feature or `(n_observations, n_subjects, n_voxels)` for voxel-wise
            ISC.
        n_permute (int): Number of bootstrap draws or permutations. Defaults to
            5000.
        summary (str): How ISC values are aggregated: `'median'` (default,
            robust to outliers) or `'mean'` (Fisher z-transformed mean).
        summary_statistic (str): `'pairwise'` (default) or `'leave-one-out'`.
        method (str): `'bootstrap'` (default; subject-wise bootstrap, Chen et
            al. 2016), `'circle_shift'` (circular time-series shift), or
            `'phase_randomize'` (FFT phase randomization).
        ci_percentile (float): Confidence-interval width in percent (95 gives a
            95% CI). Defaults to 95.
        tail (int | str): `2` or `'two'` (default) for a two-tailed p-value;
            `1` or `'one'` for one-tailed (ISC > 0).
        return_null (bool): If True, include the null distribution in the
            result. Defaults to False.
        progress_bar (bool): Show a progress bar over the resamples. Defaults to
            False.
        exclude_self_corr (bool): In the pairwise bootstrap, mask the perfect
            correlations a duplicated subject produces as NaN. Defaults to True.
        metric (str): Similarity metric for pairwise ISC; any metric accepted by
            `sklearn.metrics.pairwise_distances` (`'correlation'`,
            `'spearman'`, `'cosine'`, and `'euclidean'` take fast paths). Ignored
            for `summary_statistic='leave-one-out'`. Defaults to
            `'correlation'`.
        n_jobs (int): Number of joblib workers for the resamples. -1 (default)
            picks the worker count from available memory. Results are identical
            at every worker count.
        random_state (int | None): Random seed for reproducibility.

    Returns:
        dict: Keys `'isc'` (float or np.ndarray, observed ISC), `'p'` (float or
            np.ndarray, p-value with the `(count + 1) / (n + 1)` correction),
            `'ci'` (tuple `(lower, upper)` percentiles of the resamples), and
            — when `return_null=True` — `'null_dist'` (np.ndarray).

    Examples:
        ```python
        # Single-feature ISC
        data = np.random.randn(100, 10)  # 100 timepoints, 10 subjects
        result = isc_permutation_test(data, n_permute=1000)
        result["isc"], result["p"]

        # Voxel-wise leave-one-out ISC
        data_voxels = np.random.randn(100, 50, 5000)  # 5K voxels
        result = isc_permutation_test(
            data_voxels,
            summary_statistic="leave-one-out",
            n_permute=5000,
        )
        (result["p"] < 0.05).sum()  # → number of significant voxels

        # Leave-one-out vs pairwise
        result_loo = isc_permutation_test(data, summary_statistic="leave-one-out")
        result_pair = isc_permutation_test(data, summary_statistic="pairwise")
        ```

    References:
        Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
        Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
        correlations, part I: nonparametric approaches to inter-subject
        correlation analysis at the group level. NeuroImage, 142, 248-259.
    """
    # Input validation
    validate_tail_parameter(tail)
    data = np.asarray(data)
    if data.ndim not in [2, 3]:
        raise ValueError(f"data must be 2D or 3D, got shape {data.shape}")

    if summary_statistic not in ["leave-one-out", "pairwise"]:
        raise ValueError(
            f"summary_statistic must be 'leave-one-out' or 'pairwise', "
            f"got {summary_statistic}"
        )

    if method not in ["bootstrap", "circle_shift", "phase_randomize"]:
        raise ValueError(
            f"method must be 'bootstrap', 'circle_shift', or 'phase_randomize', "
            f"got {method}"
        )

    # Phase 1: Compute ISC (run once)
    if summary_statistic == "leave-one-out":
        # Compute leave-one-out values
        loo_values = _compute_loo_isc(data)

        # Compute observed summary statistic
        if summary == "median":
            observed_isc = np.median(loo_values, axis=0)
        elif summary == "mean":
            z = np.arctanh(np.clip(loo_values, -0.9999, 0.9999))
            observed_isc = np.tanh(np.mean(z, axis=0))
        else:
            raise ValueError(f"summary must be 'median' or 'mean', got {summary}")

    else:  # pairwise
        # Compute pairwise correlation matrix (condensed form)
        pairwise_condensed = _compute_pairwise_isc(data, metric=metric)
        n_subjects = data.shape[1]

        # Compute observed summary statistic
        if summary == "median":
            observed_isc = np.nanmedian(pairwise_condensed, axis=0)
        elif summary == "mean":
            z = np.arctanh(np.clip(pairwise_condensed, -0.9999, 0.9999))
            observed_isc = np.tanh(np.nanmean(z, axis=0))
        else:
            raise ValueError(f"summary must be 'median' or 'mean', got {summary}")

    # Phase 2: Bootstrap/permutation (run n_permute times)
    if method == "bootstrap":
        if summary_statistic == "leave-one-out":
            # LOO bootstrap: resample pre-computed values
            bootstraps = _bootstrap_loo_cpu_parallel(
                loo_values,
                n_permute=n_permute,
                summary=summary,
                n_jobs=n_jobs,
                random_state=random_state,
                progress_bar=progress_bar,
                max_memory_gb=None,  # Auto-detect
            )
        else:  # pairwise
            # Pairwise bootstrap: subject-wise matrix indexing
            bootstraps = _bootstrap_pairwise_cpu_parallel(
                pairwise_condensed,
                n_permute=n_permute,
                n_subjects=n_subjects,
                summary=summary,
                n_jobs=n_jobs,
                random_state=random_state,
                progress_bar=progress_bar,
                exclude_self_corr=exclude_self_corr,
                max_memory_gb=None,  # Auto-detect
            )

        # Center bootstrap distribution by subtracting observed (Chen et al. 2016)
        null_distribution = bootstraps - observed_isc

    elif method == "circle_shift":
        # Import timeseries utilities
        from .timeseries import circle_shift

        # Permute data and recompute ISC
        rng = check_random_state(random_state)
        seeds = rng.randint(0, 2**31 - 1, size=n_permute)

        bootstraps = []
        for i in range(n_permute):
            # Circle shift the data
            # For 3D data (n_obs, n_subjects, n_voxels), apply per subject
            if data.ndim == 3:
                perm_rng = np.random.RandomState(seeds[i])
                data_permuted = np.empty_like(data)
                for subj in range(data.shape[1]):
                    data_permuted[:, subj, :] = circle_shift(
                        data[:, subj, :], random_state=perm_rng
                    )
            else:
                data_permuted = circle_shift(
                    data, random_state=np.random.RandomState(seeds[i])
                )

            # Recompute ISC
            if summary_statistic == "leave-one-out":
                loo_perm = _compute_loo_isc(data_permuted)
                if summary == "median":
                    isc_perm = np.median(loo_perm, axis=0)
                else:
                    z = np.arctanh(np.clip(loo_perm, -0.9999, 0.9999))
                    isc_perm = np.tanh(np.mean(z, axis=0))
            else:  # pairwise
                pair_perm = _compute_pairwise_isc(data_permuted, metric=metric)
                if summary == "median":
                    isc_perm = np.nanmedian(pair_perm, axis=0)
                else:
                    z = np.arctanh(np.clip(pair_perm, -0.9999, 0.9999))
                    isc_perm = np.tanh(np.nanmean(z, axis=0))

            bootstraps.append(isc_perm)

        bootstraps = np.array(bootstraps)
        null_distribution = bootstraps  # Already centered for permutation methods

    elif method == "phase_randomize":
        # Import timeseries utilities
        from .timeseries import phase_randomize

        # Similar to circle_shift but with phase randomization
        rng = check_random_state(random_state)
        seeds = rng.randint(0, 2**31 - 1, size=n_permute)

        bootstraps = []
        for i in range(n_permute):
            # Phase randomize the data
            # For 3D data (n_obs, n_subjects, n_voxels), apply per subject
            if data.ndim == 3:
                perm_rng = np.random.RandomState(seeds[i])
                data_permuted = np.empty_like(data)
                for subj in range(data.shape[1]):
                    data_permuted[:, subj, :] = phase_randomize(
                        data[:, subj, :], random_state=perm_rng
                    )
            else:
                data_permuted = phase_randomize(
                    data, random_state=np.random.RandomState(seeds[i])
                )

            # Recompute ISC
            if summary_statistic == "leave-one-out":
                loo_perm = _compute_loo_isc(data_permuted)
                if summary == "median":
                    isc_perm = np.median(loo_perm, axis=0)
                else:
                    z = np.arctanh(np.clip(loo_perm, -0.9999, 0.9999))
                    isc_perm = np.tanh(np.mean(z, axis=0))
            else:  # pairwise
                pair_perm = _compute_pairwise_isc(data_permuted, metric=metric)
                if summary == "median":
                    isc_perm = np.nanmedian(pair_perm, axis=0)
                else:
                    z = np.arctanh(np.clip(pair_perm, -0.9999, 0.9999))
                    isc_perm = np.tanh(np.nanmean(z, axis=0))

            bootstraps.append(isc_perm)

        bootstraps = np.array(bootstraps)
        null_distribution = bootstraps

    # Compute p-value (Phipson-Smyth correction)
    # NOTE: _compute_pvalue signature is (obs_stat, null_dist, tail)
    p_value = _compute_pvalue(observed_isc, null_distribution, tail=tail)

    # Compute confidence interval
    ci_lower = (100 - ci_percentile) / 2
    ci_upper = ci_percentile + ci_lower

    if observed_isc.ndim == 0 or observed_isc.shape == ():
        # Single value
        ci = (np.percentile(bootstraps, ci_lower), np.percentile(bootstraps, ci_upper))
    else:
        # Per-voxel
        ci = (
            np.percentile(bootstraps, ci_lower, axis=0),
            np.percentile(bootstraps, ci_upper, axis=0),
        )

    # Build result dictionary
    result = {
        "isc": observed_isc,
        "p": p_value,
        "ci": ci,
    }

    if return_null:
        result["null_dist"] = null_distribution

    return result
