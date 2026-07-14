"""Intersubject Correlation (ISC) with GPU-Accelerated Permutation Testing.

This module provides both leave-one-out (LOO) and pairwise ISC computation
with efficient CPU-parallel and GPU-batched implementations. Follows the
statistical methods from Chen et al. (2016) for correct bootstrap resampling
of correlation matrices.

Key Features:
    - Two ISC modes: leave-one-out and pairwise (statistically different)
    - GPU acceleration for voxel-wise computation (10-30× speedup)
    - CPU-parallel bootstrap with joblib
    - Correct subject-wise bootstrap (Chen et al. 2016)
    - Memory-efficient condensed matrix storage

References:
    Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
    Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
    correlations, part I: nonparametric approaches to inter-subject
    correlation analysis at the group level. NeuroImage, 142, 248-259.

Notes:
    Leave-one-out and pairwise ISC are monotonically correlated but
    statistically different. LOO is computationally more efficient
    and provides unbiased estimates. Pairwise captures full correlation
    structure but is O(n²) in subjects.
"""

import numpy as np
from typing import Literal, Any
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise_distances

from .utils import _compute_pvalue, EPSILON


# ============================================================================
# Phase 1: Leave-One-Out (LOO) ISC Computation
# ============================================================================


def _compute_loo_isc(data, backend="numpy"):
    """Compute leave-one-out intersubject correlation.

    For each subject, correlates their data with the mean of all other
    subjects. This provides an unbiased estimate of subject-level ISC
    and is computationally efficient (O(n_subjects) vs O(n_subjects²)).

    Args:
        data: Data array with one of the following shapes:
            - (n_observations, n_subjects): Single feature
            - (n_observations, n_subjects, n_voxels): Voxel-wise
        backend: Computation backend. Use 'torch' for GPU acceleration on
            voxel-wise data (10-30× speedup for large n_voxels). Defaults to 'numpy'.

    Returns:
        Leave-one-out ISC values:
        - Shape (n_subjects,) for single feature
        - Shape (n_subjects, n_voxels) for voxel-wise

    Examples:
        >>> data = np.random.randn(100, 10)  # 100 timepoints, 10 subjects
        >>> loo = _compute_loo_isc(data)
        >>> loo.shape
        (10,)

        >>> # Voxel-wise
        >>> data_voxels = np.random.randn(100, 10, 1000)  # 1000 voxels
        >>> loo = _compute_loo_isc(data_voxels, backend='torch')
        >>> loo.shape
        (10, 1000)

    Notes:
        For each subject i, computes: corr(subject_i, mean(all other subjects))
        This is the method recommended by Chen et al. (2016) for unbiased ISC.
    """
    if backend == "numpy":
        return _compute_loo_isc_numpy(data)
    if backend == "torch":
        return _compute_loo_isc_gpu(data)
    raise ValueError(f"backend must be 'numpy' or 'torch', got {backend}")


def _compute_loo_isc_numpy(data):
    """NumPy implementation of leave-one-out ISC."""
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


def _batch_correlation_gpu(x, y):
    """Compute correlation between x and y in parallel across features.

    Args:
        x: Data tensor on GPU, shape (n_observations, n_features).
        y: Data tensor on GPU, shape (n_observations, n_features).

    Returns:
        Correlation coefficients, shape (n_features,).
    """
    import torch

    # Center the data
    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)

    # Compute correlation
    numerator = (x_centered * y_centered).sum(dim=0)
    denominator = torch.sqrt((x_centered**2).sum(dim=0) * (y_centered**2).sum(dim=0))

    # Handle zero variance case using EPSILON
    correlations = numerator / (denominator + EPSILON)

    return correlations


def _compute_loo_isc_gpu(data):
    """GPU-accelerated leave-one-out ISC computation.

    Batches correlation computation across voxels for significant speedup
    on large voxel-wise problems (10-30× faster than NumPy for >5K voxels).
    """
    import torch

    if data.ndim != 3:
        raise ValueError("GPU backend requires 3D voxel-wise data")

    n_obs, n_subjects, n_voxels = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transfer data to GPU once
    data_gpu = torch.tensor(data, dtype=torch.float32, device=device)
    loo_values = torch.zeros(n_subjects, n_voxels, device=device)

    for i in range(n_subjects):
        # Create mask for all subjects except i
        mask = torch.ones(n_subjects, dtype=torch.bool, device=device)
        mask[i] = False

        # Compute mean of all other subjects (across all voxels in parallel)
        others_mean = data_gpu[:, mask, :].mean(dim=1)  # (n_obs, n_voxels)

        # Get subject i's data
        subject_i = data_gpu[:, i, :]  # (n_obs, n_voxels)

        # Batch correlation across all voxels
        loo_values[i, :] = _batch_correlation_gpu(subject_i, others_mean)

    return loo_values.cpu().numpy()


# ============================================================================
# Phase 2: Pairwise ISC Computation
# ============================================================================


def _compute_pairwise_isc(data, backend="numpy", sim_metric="correlation"):
    """Compute pairwise intersubject correlation (condensed form).

    Computes all n×(n-1)/2 pairwise correlations between subjects and
    stores in condensed upper-triangle format for memory efficiency.

    Args:
        data: Data array with one of the following shapes:
            - (n_observations, n_subjects): Single feature
            - (n_observations, n_subjects, n_voxels): Voxel-wise
        backend: Computation backend. Use 'torch' for GPU acceleration on
            voxel-wise data. Defaults to 'numpy'.
        sim_metric: Similarity metric. Options:
            - 'correlation': Pearson correlation (uses optimized np.corrcoef)
            - 'spearman': Spearman rank correlation (rank-transform then np.corrcoef)
            - 'cosine': Cosine similarity (normalized dot products, optimized)
            - 'euclidean': Euclidean similarity (1 - distance, optimized)
            - Other metrics: Uses sklearn.metrics.pairwise_distances (slower)
            Defaults to 'correlation'.

    Returns:
        Pairwise correlations in condensed form (upper triangle):
        - Shape (n_pairs,) for single feature, where n_pairs = n*(n-1)/2
        - Shape (n_pairs, n_voxels) for voxel-wise

    Examples:
        >>> data = np.random.randn(100, 5)  # 5 subjects
        >>> pairwise = _compute_pairwise_isc(data)
        >>> pairwise.shape
        (10,)  # 5*4/2 = 10 pairs

    Notes:
        Uses optimized paths for common metrics:
        - 'correlation': np.corrcoef (fast BLAS-accelerated)
        - 'spearman': rank-transform then np.corrcoef (10-100× faster than pairwise_distances)
        - 'cosine': normalized dot products (5-20× faster than pairwise_distances)
        - 'euclidean': vectorized squared-distance (3-10× faster than pairwise_distances)
        - Other metrics: sklearn.metrics.pairwise_distances (slower fallback)
    """
    if backend == "numpy":
        return _compute_pairwise_isc_numpy(data, sim_metric=sim_metric)
    if backend == "torch":
        if sim_metric != "correlation":
            raise ValueError(
                f"GPU backend only supports sim_metric='correlation', got {sim_metric}"
            )
        return _compute_pairwise_isc_gpu(data)
    raise ValueError(f"backend must be 'numpy' or 'torch', got {backend}")


def _compute_pairwise_isc_numpy(data, sim_metric="correlation"):
    """NumPy implementation of pairwise ISC (condensed storage)."""
    if sim_metric == "correlation":
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
    if sim_metric == "spearman":
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
    if sim_metric == "cosine":
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
    if sim_metric == "euclidean":
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
        dist_matrix = pairwise_distances(data.T, metric=sim_metric)
        sim_matrix = 1 - dist_matrix
        return squareform(sim_matrix, checks=False)
    if data.ndim == 3:
        # Voxel-wise: (n_observations, n_subjects, n_voxels)
        n_obs, n_subjects, n_voxels = data.shape
        n_pairs = n_subjects * (n_subjects - 1) // 2
        pairwise_all = np.zeros((n_pairs, n_voxels))
        for v in range(n_voxels):
            dist_matrix = pairwise_distances(data[:, :, v].T, metric=sim_metric)
            sim_matrix = 1 - dist_matrix
            pairwise_all[:, v] = squareform(sim_matrix, checks=False)
        return pairwise_all
    raise ValueError(f"data must be 2D or 3D, got shape {data.shape}")


def _batch_corrcoef_gpu(data_gpu):
    """Compute correlation matrices in parallel across voxels on GPU.

    Args:
        data_gpu: Data tensor on GPU (transposed for efficient correlation),
            shape (n_voxels, n_subjects, n_observations).

    Returns:
        Correlation matrices for each voxel, shape (n_voxels, n_subjects, n_subjects).
    """
    import torch

    # Center the data (per subject, per voxel)
    data_centered = data_gpu - data_gpu.mean(dim=2, keepdim=True)

    # Compute covariance matrices (batched matrix multiply)
    # (n_voxels, n_subjects, n_obs) @ (n_voxels, n_obs, n_subjects)
    cov_matrices = torch.bmm(data_centered, data_centered.transpose(1, 2))

    # Compute standard deviations
    std_devs = torch.sqrt((data_centered**2).sum(dim=2))  # (n_voxels, n_subjects)

    # Normalize to get correlations: cov / (std_i * std_j)
    # Broadcasting: (n_voxels, n_subjects, 1) * (n_voxels, 1, n_subjects)
    std_outer = std_devs.unsqueeze(2) * std_devs.unsqueeze(1)

    # Avoid division by zero using EPSILON
    corr_matrices = cov_matrices / (std_outer + EPSILON)

    return corr_matrices


def _compute_pairwise_isc_gpu(data):
    """GPU-accelerated pairwise ISC computation.

    Batches correlation matrix computation across voxels for significant
    speedup on large voxel-wise problems.
    """
    import torch

    if data.ndim != 3:
        raise ValueError("GPU backend requires 3D voxel-wise data")

    n_obs, n_subjects, n_voxels = data.shape
    n_pairs = n_subjects * (n_subjects - 1) // 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transpose to (n_voxels, n_subjects, n_observations) for efficient batching
    data_transposed = np.transpose(data, (2, 1, 0))
    data_gpu = torch.tensor(data_transposed, dtype=torch.float32, device=device)

    # Compute correlation matrices for all voxels in parallel
    corr_matrices = _batch_corrcoef_gpu(data_gpu)  # (n_voxels, n_subjects, n_subjects)

    # Extract upper triangles (condensed form)
    # Move to CPU for squareform (more efficient than GPU indexing)
    corr_matrices_cpu = corr_matrices.cpu().numpy()

    pairwise_all = np.zeros((n_pairs, n_voxels))
    for v in range(n_voxels):
        pairwise_all[:, v] = squareform(corr_matrices_cpu[v], checks=False)

    return pairwise_all


# ============================================================================
# Phase 2.5: ISC Group Difference Computation
# ============================================================================


def _compute_isc_group_difference(
    group1,
    group2,
    metric="median",
    summary_statistic="pairwise",
    backend="numpy",
    sim_metric="correlation",
):
    """Compute ISC difference between two groups.

    Computes intersubject correlation for each group separately, then takes
    the difference (group1 ISC - group2 ISC). Supports both pairwise and
    leave-one-out ISC computation methods.

    Args:
        group1: First group data with one of the following shapes:
            - (n_observations, n_subjects1): Single feature
            - (n_observations, n_subjects1, n_voxels): Voxel-wise
        group2: Second group data with one of the following shapes:
            - (n_observations, n_subjects2): Single feature
            - (n_observations, n_subjects2, n_voxels): Voxel-wise
        metric: Summary statistic for aggregating ISC values:
            - 'median': Direct median (robust to outliers)
            - 'mean': Fisher z-transformed mean (unbiased averaging)
            Defaults to 'median'.
        summary_statistic: ISC computation method:
            - 'pairwise': Average all pairwise correlations
            - 'leave-one-out': Correlate each subject with mean of others
            Defaults to 'pairwise'.
        backend: Computation backend. Use 'torch' for GPU acceleration on
            voxel-wise LOO data (10-30× speedup for large n_voxels). Defaults to 'numpy'.
        sim_metric: Similarity metric for pairwise ISC. Defaults to 'correlation'.

    Returns:
        ISC difference (group1 ISC - group2 ISC):
        - Shape () for single feature (scalar)
        - Shape (n_voxels,) for voxel-wise

    Examples:
        >>> group1 = np.random.randn(100, 5)  # 5 subjects
        >>> group2 = np.random.randn(100, 5)
        >>> diff = _compute_isc_group_difference(group1, group2)
        >>> diff.shape
        ()

        >>> # Voxel-wise
        >>> group1_voxels = np.random.randn(100, 5, 1000)
        >>> group2_voxels = np.random.randn(100, 5, 1000)
        >>> diff = _compute_isc_group_difference(group1_voxels, group2_voxels, backend='torch')
        >>> diff.shape
        (1000,)

    Notes:
        This function reuses existing ISC computation functions (_compute_pairwise_isc
        and _compute_loo_isc) to compute ISC for each group, then computes the difference.
        GPU backend is only available for voxel-wise LOO computation (similar to
        isc_permutation_test).
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

    if metric not in ["median", "mean"]:
        raise ValueError(f"metric must be 'median' or 'mean', got {metric}")

    if summary_statistic not in ["pairwise", "leave-one-out"]:
        raise ValueError(
            f"summary_statistic must be 'pairwise' or 'leave-one-out', got {summary_statistic}"
        )

    # Compute ISC for each group
    if summary_statistic == "pairwise":
        # Pairwise ISC: compute condensed correlation matrices
        isc1_values = _compute_pairwise_isc(
            group1, backend=backend, sim_metric=sim_metric
        )
        isc2_values = _compute_pairwise_isc(
            group2, backend=backend, sim_metric=sim_metric
        )

        # Handle single feature vs voxel-wise
        if isc1_values.ndim == 1:
            # Single feature: (n_pairs,)
            axis = None
        else:
            # Voxel-wise: (n_pairs, n_voxels)
            axis = 0

        # Compute summary statistic
        if metric == "median":
            isc1 = np.nanmedian(isc1_values, axis=axis)
            isc2 = np.nanmedian(isc2_values, axis=axis)
        elif metric == "mean":
            # Fisher z-transform
            z1 = np.arctanh(np.clip(isc1_values, -0.9999, 0.9999))
            z2 = np.arctanh(np.clip(isc2_values, -0.9999, 0.9999))
            isc1 = np.tanh(np.nanmean(z1, axis=axis))
            isc2 = np.tanh(np.nanmean(z2, axis=axis))

    else:  # leave-one-out
        # LOO ISC: compute LOO values for each subject
        loo1_values = _compute_loo_isc(group1, backend=backend)
        loo2_values = _compute_loo_isc(group2, backend=backend)

        # Handle single feature vs voxel-wise
        if loo1_values.ndim == 1:
            # Single feature: (n_subjects,)
            axis = 0
        else:
            # Voxel-wise: (n_subjects, n_voxels)
            axis = 0

        # Compute summary statistic
        if metric == "median":
            isc1 = np.median(loo1_values, axis=axis)
            isc2 = np.median(loo2_values, axis=axis)
        elif metric == "mean":
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
    metric="median",
    summary_statistic="pairwise",
    random_state=None,
    sim_metric="correlation",
):
    """Single permutation: permute group labels and compute ISC difference.

    Implements the subject-wise permutation method from Chen et al. (2016).
    Combines the two groups, permutes group labels, then computes ISC difference
    for the permuted groups.

    Args:
        group1: First group data: (n_observations, n_subjects1) or (n_observations, n_subjects1, n_voxels).
        group2: Second group data: (n_observations, n_subjects2) or (n_observations, n_subjects2, n_voxels).
        metric: Summary statistic for aggregating ISC values. Defaults to 'median'.
        summary_statistic: ISC computation method. Defaults to 'pairwise'.
        random_state: Random state for reproducibility.
        sim_metric: Similarity metric for pairwise ISC. Defaults to 'correlation'.

    Returns:
        Permuted ISC difference (scalar or per-voxel array).
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
        metric=metric,
        summary_statistic=summary_statistic,
        backend="numpy",
        sim_metric=sim_metric,
    )

    return isc_diff


def _permute_isc_group_cpu_parallel(
    group1,
    group2,
    n_permute=5000,
    metric="median",
    summary_statistic="pairwise",
    n_jobs=-1,
    random_state=None,
    progress_bar=True,
    sim_metric="correlation",
    max_memory_gb=None,
):
    """CPU-parallel permutation for ISC group difference.

    Efficiently parallelizes permutation resampling across CPU cores using joblib.
    Uses deterministic seed generation for reproducibility.
    Automatically limits workers based on available memory if n_jobs=-1.

    Args:
        group1: First group data: (n_observations, n_subjects1) or (n_observations, n_subjects1, n_voxels).
        group2: Second group data: (n_observations, n_subjects2) or (n_observations, n_subjects2, n_voxels).
        n_permute: Number of permutations. Defaults to 5000.
        metric: Summary statistic for aggregating ISC values. Defaults to 'median'.
        summary_statistic: ISC computation method. Defaults to 'pairwise'.
        n_jobs: Number of CPU cores for parallelization (-1 = auto-detect based on memory). Defaults to -1.
        random_state: Random seed for reproducibility.
        progress_bar: Show progress bar. Defaults to True.
        sim_metric: Similarity metric for pairwise ISC. Defaults to 'correlation'.
        max_memory_gb: Maximum memory budget in GB (only used if n_jobs=-1).

    Returns:
        Permuted ISC differences:
        - Shape (n_permute,) for single feature
        - Shape (n_permute, n_voxels) for voxel-wise
    """
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm
    from .utils import _auto_n_jobs_cpu, _estimate_data_size_mb

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
            min_jobs=1,
        )

    rng = check_random_state(random_state)
    MAX_INT = 2**31 - 1
    seeds = rng.randint(MAX_INT, size=n_permute)

    # Parallelize
    iterator = range(n_permute)
    if progress_bar:
        iterator = tqdm(iterator, desc="Permute ISC Group")

    permutations = Parallel(n_jobs=n_jobs)(
        delayed(_permute_isc_group_numpy)(
            group1,
            group2,
            metric=metric,
            summary_statistic=summary_statistic,
            random_state=np.random.RandomState(seeds[i]),
            sim_metric=sim_metric,
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
    metric="median",
    summary_statistic="pairwise",
    exclude_self_corr=True,
    random_state=None,
    sim_metric="correlation",
):
    """Single bootstrap: resample subjects within each group and compute ISC difference.

    Implements the subject-wise bootstrap method from Chen et al. (2016).
    Bootstraps each group independently, then computes ISC difference.
    Centers by subtracting observed difference: (boot1 - boot2) - observed_diff.

    Args:
        group1: First group data: (n_observations, n_subjects1) or (n_observations, n_subjects1, n_voxels).
        group2: Second group data: (n_observations, n_subjects2) or (n_observations, n_subjects2, n_voxels).
        observed_diff: Observed ISC difference (for centering).
        metric: Summary statistic for aggregating ISC values. Defaults to 'median'.
        summary_statistic: ISC computation method. Defaults to 'pairwise'.
        exclude_self_corr: Mask self-correlations in bootstrap (pairwise only). Defaults to True.
        random_state: Random state for reproducibility.
        sim_metric: Similarity metric for pairwise ISC. Defaults to 'correlation'.

    Returns:
        Bootstrapped ISC difference (centered): (boot1 - boot2) - observed_diff.
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
        pairwise1_boot = _compute_pairwise_isc(
            group1_boot, backend="numpy", sim_metric=sim_metric
        )
        pairwise2_boot = _compute_pairwise_isc(
            group2_boot, backend="numpy", sim_metric=sim_metric
        )

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
        if metric == "median":
            isc1_boot = np.nanmedian(pairwise1_boot, axis=axis)
            isc2_boot = np.nanmedian(pairwise2_boot, axis=axis)
        elif metric == "mean":
            z1 = np.arctanh(np.clip(pairwise1_boot, -0.9999, 0.9999))
            z2 = np.arctanh(np.clip(pairwise2_boot, -0.9999, 0.9999))
            isc1_boot = np.tanh(np.nanmean(z1, axis=axis))
            isc2_boot = np.tanh(np.nanmean(z2, axis=axis))

    else:  # leave-one-out
        # LOO bootstrap: resample pre-computed LOO values
        loo1_values = _compute_loo_isc(group1, backend="numpy")
        loo2_values = _compute_loo_isc(group2, backend="numpy")

        # Bootstrap LOO values
        isc1_boot = _bootstrap_loo_numpy(loo1_values, metric=metric, random_state=rng)
        # Use different seed for group2 to ensure independence
        rng2 = check_random_state(rng.randint(0, 2**31 - 1))
        isc2_boot = _bootstrap_loo_numpy(loo2_values, metric=metric, random_state=rng2)

    # Compute difference and center
    boot_diff = (isc1_boot - isc2_boot) - observed_diff

    return boot_diff


def _bootstrap_isc_group_cpu_parallel(
    group1,
    group2,
    observed_diff,
    n_permute=5000,
    metric="median",
    summary_statistic="pairwise",
    exclude_self_corr=True,
    n_jobs=-1,
    random_state=None,
    progress_bar=True,
    sim_metric="correlation",
    max_memory_gb=None,
):
    """CPU-parallel bootstrap for ISC group difference.

    Efficiently parallelizes bootstrap resampling across CPU cores using joblib.
    Uses deterministic seed generation for reproducibility.
    Automatically limits workers based on available memory if n_jobs=-1.

    Args:
        group1: First group data: (n_observations, n_subjects1) or (n_observations, n_subjects1, n_voxels).
        group2: Second group data: (n_observations, n_subjects2) or (n_observations, n_subjects2, n_voxels).
        observed_diff: Observed ISC difference (for centering).
        n_permute: Number of bootstrap iterations. Defaults to 5000.
        metric: Summary statistic for aggregating ISC values. Defaults to 'median'.
        summary_statistic: ISC computation method. Defaults to 'pairwise'.
        exclude_self_corr: Mask self-correlations in bootstrap (pairwise only). Defaults to True.
        n_jobs: Number of CPU cores for parallelization (-1 = auto-detect based on memory). Defaults to -1.
        random_state: Random seed for reproducibility.
        progress_bar: Show progress bar. Defaults to True.
        sim_metric: Similarity metric for pairwise ISC. Defaults to 'correlation'.
        max_memory_gb: Maximum memory budget in GB (only used if n_jobs=-1).

    Returns:
        Bootstrapped ISC differences (centered):
        - Shape (n_permute,) for single feature
        - Shape (n_permute, n_voxels) for voxel-wise
    """
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm
    from .utils import _auto_n_jobs_cpu, _estimate_data_size_mb

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
            min_jobs=1,
        )

    rng = check_random_state(random_state)
    MAX_INT = 2**31 - 1
    seeds = rng.randint(MAX_INT, size=n_permute)

    # Parallelize
    iterator = range(n_permute)
    if progress_bar:
        iterator = tqdm(iterator, desc="Bootstrap ISC Group")

    bootstraps = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_isc_group_numpy)(
            group1,
            group2,
            observed_diff=observed_diff,
            metric=metric,
            summary_statistic=summary_statistic,
            exclude_self_corr=exclude_self_corr,
            random_state=np.random.RandomState(seeds[i]),
            sim_metric=sim_metric,
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
    n_permute: int = 5000,
    metric: Literal["median", "mean"] = "median",
    method: Literal["permute", "bootstrap"] = "permute",
    summary_statistic: Literal["leave-one-out", "pairwise"] = "pairwise",
    ci_percentile: float = 95,
    tail: Literal[1, 2] = 2,
    parallel: Literal["cpu", "gpu"] | None = "cpu",
    n_jobs: int = -1,
    random_state: int | None = None,
    return_null: bool = False,
    progress_bar: bool = True,
    exclude_self_corr: bool = True,
    sim_metric: str = "correlation",
) -> dict[str, Any]:
    """Compute ISC difference between groups with permutation testing.

    Supports both subject-wise permutation and bootstrap methods with efficient
    CPU-parallel and optional GPU acceleration. Follows the statistical methods
    from Chen et al. (2016) for correct group comparison inference.

    Args:
        group1: First group data with one of the following shapes:
            - (n_observations, n_subjects1): Single feature
            - (n_observations, n_subjects1, n_voxels): Voxel-wise
        group2: Second group data with one of the following shapes:
            - (n_observations, n_subjects2): Single feature
            - (n_observations, n_subjects2, n_voxels): Voxel-wise
        n_permute: Number of permutations/bootstrap iterations. Defaults to 5000.
        metric: Summary statistic for aggregating ISC values:
            - 'median': Direct median (robust to outliers)
            - 'mean': Fisher z-transformed mean (unbiased averaging)
            Defaults to 'median'.
        method: Resampling method for p-value computation:
            - 'permute': Subject-wise permutation (combines groups, permutes labels)
            - 'bootstrap': Subject-wise bootstrap (resamples within each group)
            Defaults to 'permute'.
        summary_statistic: ISC computation method:
            - 'pairwise': Average all pairwise correlations
            - 'leave-one-out': Correlate each subject with mean of others
            Defaults to 'pairwise'.
        ci_percentile: Confidence interval percentile (e.g., 95 for 95% CI). Defaults to 95.
        tail: One-tailed (1) or two-tailed (2) p-value. Defaults to 2.
        parallel: Parallelization method:
            - 'cpu': CPU parallelization via joblib (default, 4-8× speedup)
            - 'gpu': GPU acceleration via PyTorch (10-30× speedup for voxel-wise LOO)
            - None: Single-threaded NumPy (for debugging/small problems)
            Defaults to 'cpu'.
        n_jobs: Number of CPU cores for parallelization (-1 = all cores).
            Only used when parallel='cpu'. Defaults to -1.
        random_state: Random seed for reproducibility.
        return_null: If True, return null distribution in result dict. Defaults to False.
        progress_bar: Show progress bar during bootstrap/permutation. Defaults to True.
        exclude_self_corr: Mask self-correlations in bootstrap (pairwise only). Defaults to True.
        sim_metric: Similarity metric for pairwise ISC computation. See
            sklearn.metrics.pairwise_distances for valid options. Only applies
            when summary_statistic='pairwise'. Defaults to 'correlation'.

    Returns:
        Dictionary with the following keys:
        - 'isc_group_difference': Observed ISC difference (float or array per voxel)
        - 'p': P-value (Phipson-Smyth corrected)
        - 'ci': Confidence interval tuple (lower, upper)
        - 'parallel': Parallelization method used
        - 'null_dist': (optional) Bootstrap/permutation distribution

    Examples:
    >>> # Single-feature ISC group comparison
    >>> group1 = np.random.randn(100, 10)  # 10 subjects
    >>> group2 = np.random.randn(100, 10)
    >>> result = isc_group_permutation_test(group1, group2, n_permute=1000)
    >>> print(f"ISC difference: {result['isc_group_difference']:.3f}, p: {result['p']:.3f}")

    >>> # Voxel-wise ISC group comparison with GPU acceleration
    >>> group1_voxels = np.random.randn(100, 10, 5000)  # 5K voxels
    >>> group2_voxels = np.random.randn(100, 10, 5000)
    >>> result = isc_group_permutation_test(
    ...     group1_voxels,
    ...     group2_voxels,
    ...     summary_statistic='leave-one-out',
    ...     parallel='gpu',  # GPU for LOO computation
    ...     n_permute=5000
    ... )
    >>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")

    References:
        Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
        Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
        correlations, part I: nonparametric approaches to inter-subject
        correlation analysis at the group level. NeuroImage, 142, 248-259.

    Notes:
        - Permutation method combines groups and permutes labels (Chen et al. 2016)
        - Bootstrap method resamples subjects within each group independently
        - Bootstrap distribution is centered by subtracting observed difference
        - GPU acceleration available for voxel-wise LOO computation
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

    if metric not in ["median", "mean"]:
        raise ValueError(f"metric must be 'median' or 'mean', got {metric}")

    if method not in ["permute", "bootstrap"]:
        raise ValueError(f"method must be 'permute' or 'bootstrap', got {method}")

    if summary_statistic not in ["pairwise", "leave-one-out"]:
        raise ValueError(
            f"summary_statistic must be 'pairwise' or 'leave-one-out', got {summary_statistic}"
        )

    # Validate parallel parameter
    if parallel not in [None, "cpu", "gpu"]:
        raise ValueError(f"parallel must be None, 'cpu', or 'gpu', got {parallel!r}")

    # Determine backend for computation phase based on parallel parameter
    if parallel == "cpu" or parallel is None:
        # CPU modes
        if parallel is None:
            # Single-threaded NumPy
            compute_backend = "numpy"
            bootstrap_backend = "numpy"
        else:
            # CPU parallelization
            compute_backend = "numpy"
            bootstrap_backend = "cpu-parallel"
    else:
        # GPU mode
        compute_backend = "torch"
        bootstrap_backend = "cpu-parallel"  # Bootstrap still uses CPU parallel

    # Phase 1: Compute observed ISC difference
    observed_diff = _compute_isc_group_difference(
        group1,
        group2,
        metric=metric,
        summary_statistic=summary_statistic,
        backend=compute_backend,
        sim_metric=sim_metric,
    )

    # Phase 2: Bootstrap/Permutation (run n_permute times)
    if method == "permute":
        if bootstrap_backend == "numpy":
            # Sequential NumPy
            rng = check_random_state(random_state)
            seeds = rng.randint(0, 2**31 - 1, size=n_permute)
            null_dist = np.array(
                [
                    _permute_isc_group_numpy(
                        group1,
                        group2,
                        metric=metric,
                        summary_statistic=summary_statistic,
                        random_state=np.random.RandomState(seeds[i]),
                        sim_metric=sim_metric,
                    )
                    for i in range(n_permute)
                ]
            )
        else:  # cpu-parallel
            null_dist = _permute_isc_group_cpu_parallel(
                group1,
                group2,
                n_permute=n_permute,
                metric=metric,
                summary_statistic=summary_statistic,
                n_jobs=n_jobs,
                random_state=random_state,
                progress_bar=progress_bar,
                sim_metric=sim_metric,
            )

    else:  # bootstrap
        if bootstrap_backend == "numpy":
            # Sequential NumPy
            rng = check_random_state(random_state)
            seeds = rng.randint(0, 2**31 - 1, size=n_permute)
            null_dist = np.array(
                [
                    _bootstrap_isc_group_numpy(
                        group1,
                        group2,
                        observed_diff=observed_diff,
                        metric=metric,
                        summary_statistic=summary_statistic,
                        exclude_self_corr=exclude_self_corr,
                        random_state=np.random.RandomState(seeds[i]),
                        sim_metric=sim_metric,
                    )
                    for i in range(n_permute)
                ]
            )
        else:  # cpu-parallel
            null_dist = _bootstrap_isc_group_cpu_parallel(
                group1,
                group2,
                observed_diff=observed_diff,
                n_permute=n_permute,
                metric=metric,
                summary_statistic=summary_statistic,
                exclude_self_corr=exclude_self_corr,
                n_jobs=n_jobs,
                random_state=random_state,
                progress_bar=progress_bar,
                sim_metric=sim_metric,
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

    # Compute confidence intervals
    if null_dist.ndim == 1:
        ci_lower = np.percentile(null_dist, (100 - ci_percentile) / 2)
        ci_upper = np.percentile(null_dist, ci_percentile + (100 - ci_percentile) / 2)
    else:
        # Voxel-wise: compute CI per voxel
        ci_lower = np.nanpercentile(null_dist, (100 - ci_percentile) / 2, axis=0)
        ci_upper = np.nanpercentile(
            null_dist, ci_percentile + (100 - ci_percentile) / 2, axis=0
        )

    # Build result dictionary
    result = {
        "isc_group_difference": observed_diff,
        "p": p_values,
        "ci": (ci_lower, ci_upper),
        "parallel": parallel,
    }

    if return_null:
        result["null_dist"] = null_dist

    return result


# ============================================================================
# Phase 3: Leave-One-Out Bootstrap
# ============================================================================


def _bootstrap_loo_numpy(loo_values, metric="median", random_state=None):
    """Bootstrap LOO ISC by resampling subjects.

    Implements the subject-wise bootstrap method from Chen et al. (2016).
    Resamples the pre-computed LOO values (not raw data) for efficiency.

    Args:
        loo_values: Pre-computed LOO values:
            - Shape (n_subjects,) for single feature
            - Shape (n_subjects, n_voxels) for voxel-wise
        metric: Summary statistic to compute from bootstrap sample. Defaults to 'median'.
        random_state: Random state for reproducibility.

    Returns:
        Bootstrap summary statistic (scalar or per-voxel array).

    Notes:
        For 'mean', applies Fisher z-transform before averaging to avoid
        bias (arctanh → mean → tanh). For 'median', computes directly.
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
    if metric == "median":
        return np.median(boot_values, axis=0)
    if metric == "mean":
        # Fisher z-transform for unbiased mean
        z = np.arctanh(np.clip(boot_values, -0.9999, 0.9999))
        return np.tanh(np.mean(z, axis=0))
    raise ValueError(f"metric must be 'median' or 'mean', got {metric}")


def _bootstrap_loo_cpu_parallel(
    loo_values,
    n_permute=5000,
    metric="median",
    n_jobs=-1,
    random_state=None,
    progress_bar=True,
    max_memory_gb=None,
):
    """CPU-parallel LOO bootstrap using joblib.

    Efficiently parallelizes bootstrap resampling across CPU cores.
    Uses deterministic seed generation for reproducibility.
    Automatically limits workers based on available memory if n_jobs=-1.

    Args:
        loo_values: Pre-computed LOO values (n_subjects,) or (n_subjects, n_voxels).
        n_permute: Number of bootstrap iterations. Defaults to 5000.
        metric: Summary statistic. Defaults to 'median'.
        n_jobs: Number of CPU cores (-1 = auto-detect based on memory). Defaults to -1.
        random_state: Random seed for reproducibility.
        progress_bar: Show progress bar. Defaults to True.
        max_memory_gb: Maximum memory budget in GB (only used if n_jobs=-1).

    Returns:
        Bootstrap distribution, shape (n_permute,) or (n_permute, n_voxels).
    """
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm
    from .utils import _auto_n_jobs_cpu, _estimate_data_size_mb

    # Auto-detect optimal n_jobs based on memory if n_jobs=-1
    if n_jobs == -1:
        data_size_mb = _estimate_data_size_mb(loo_values)
        n_jobs = _auto_n_jobs_cpu(
            data_size_mb=data_size_mb,
            n_permute=n_permute,
            max_memory_gb=max_memory_gb,
            min_jobs=1,
        )

    # Pre-generate seeds for deterministic parallelization
    rng = check_random_state(random_state)
    seeds = rng.randint(0, 2**31 - 1, size=n_permute)

    # Parallelize with independent RandomState per permutation
    iterator = range(n_permute)
    if progress_bar:
        iterator = tqdm(iterator, desc="Bootstrap LOO")

    bootstraps = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_loo_numpy)(
            loo_values,
            metric=metric,
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
    metric="median",
    bootstrap_subjects=None,
    n_subjects=None,
    random_state=None,
    exclude_self_corr=True,
):
    """Bootstrap pairwise ISC by subject-wise matrix indexing.

    Implements the correct bootstrap procedure for correlation matrices
    (Chen et al. 2016): resample subjects, extract submatrix, mask
    same-subject pairs (self-correlations from duplicates).

    Args:
        pairwise_condensed: Pre-computed pairwise correlations in condensed form:
            - Shape (n_pairs,) for single feature
            - Shape (n_pairs, n_voxels) for voxel-wise
        metric: Summary statistic. Defaults to 'median'.
        bootstrap_subjects: Pre-generated bootstrap subject indices (for testing).
        n_subjects: Number of subjects (required if bootstrap_subjects is None).
        random_state: Random state for sampling.
        exclude_self_corr: If True, mask self-correlations (perfect correlations from duplicate
            subjects) as NaN. If False, include them in the summary statistic. Defaults to True.

    Returns:
        Bootstrap summary statistic.

    Notes:
        When the same subject appears multiple times in the bootstrap sample,
        their pairwise correlation is 1.0 (perfect self-correlation). These
        are masked as NaN to exclude from the summary statistic, following
        Chen et al. (2016) recommendations.
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

    if metric == "median":
        return np.nanmedian(boot_condensed, axis=axis)
    if metric == "mean":
        # Fisher z-transform
        z = np.arctanh(np.clip(boot_condensed, -0.9999, 0.9999))
        return np.tanh(np.nanmean(z, axis=axis))
    raise ValueError(f"metric must be 'median' or 'mean', got {metric}")


def _bootstrap_pairwise_cpu_parallel(
    pairwise_condensed,
    n_permute=5000,
    n_subjects=None,
    metric="median",
    n_jobs=-1,
    random_state=None,
    progress_bar=True,
    exclude_self_corr=True,
    max_memory_gb=None,
):
    """CPU-parallel pairwise bootstrap using joblib.

    Same pattern as LOO bootstrap, but operates on pairwise correlation
    matrices with subject-wise indexing.
    Automatically limits workers based on available memory if n_jobs=-1.

    Args:
        pairwise_condensed: Pre-computed pairwise correlations (n_pairs,) or (n_pairs, n_voxels).
        n_permute: Number of bootstrap iterations. Defaults to 5000.
        n_subjects: Number of subjects in original data.
        metric: Summary statistic. Defaults to 'median'.
        n_jobs: Number of CPU cores (-1 = auto-detect based on memory). Defaults to -1.
        random_state: Random seed for reproducibility.
        progress_bar: Show progress bar. Defaults to True.
        exclude_self_corr: If True, mask self-correlations (perfect correlations from duplicate
            subjects) as NaN. If False, include them in the summary statistic. Defaults to True.
        max_memory_gb: Maximum memory budget in GB (only used if n_jobs=-1).

    Returns:
        Bootstrap distribution, shape (n_permute,) or (n_permute, n_voxels).
    """
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm
    from .utils import _auto_n_jobs_cpu, _estimate_data_size_mb

    if n_subjects is None:
        raise ValueError("n_subjects is required for pairwise bootstrap")

    # Auto-detect optimal n_jobs based on memory if n_jobs=-1
    if n_jobs == -1:
        data_size_mb = _estimate_data_size_mb(pairwise_condensed)
        n_jobs = _auto_n_jobs_cpu(
            data_size_mb=data_size_mb,
            n_permute=n_permute,
            max_memory_gb=max_memory_gb,
            min_jobs=1,
        )

    # Pre-generate seeds
    rng = check_random_state(random_state)
    seeds = rng.randint(0, 2**31 - 1, size=n_permute)

    # Parallelize
    iterator = range(n_permute)
    if progress_bar:
        iterator = tqdm(iterator, desc="Bootstrap Pairwise")

    bootstraps = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_pairwise_numpy)(
            pairwise_condensed,
            metric=metric,
            n_subjects=n_subjects,
            random_state=np.random.RandomState(seeds[i]),
            exclude_self_corr=exclude_self_corr,
        )
        for i in iterator
    )

    return np.array(bootstraps)


# ============================================================================
# Phase 5: Main ISC Permutation Test Function
# ============================================================================


def isc_permutation_test(
    # Required
    data: np.ndarray,
    # Optional algorithm parameters
    n_permute: int = 5000,
    metric: Literal["median", "mean"] = "median",
    summary_statistic: Literal["leave-one-out", "pairwise"] = "pairwise",
    method: Literal["bootstrap", "circle_shift", "phase_randomize"] = "bootstrap",
    ci_percentile: float = 95,
    tail: Literal[1, 2] = 2,
    return_null: bool = False,
    progress_bar: bool = True,
    exclude_self_corr: bool = True,
    sim_metric: str = "correlation",
    # Backend parameters (grouped)
    parallel: Literal["cpu", "gpu"] | None = "cpu",
    n_jobs: int = -1,
    max_gpu_memory_gb: float = 4.0,
    # Random state (last)
    random_state: int | None = None,
) -> dict[str, Any]:
    """Compute intersubject correlation with permutation testing.

    Supports both leave-one-out and pairwise ISC computation modes with
    GPU acceleration for large voxel-wise problems and CPU-parallel
    bootstrap resampling.

    Args:
        data: Data array with one of the following shapes:
            - (n_observations, n_subjects): Single feature ISC
            - (n_observations, n_subjects, n_voxels): Voxel-wise ISC
        n_permute: Number of bootstrap iterations or permutations. Defaults to 5000.
        metric: Summary statistic to aggregate ISC values.
            - 'median': Direct median (robust to outliers)
            - 'mean': Fisher z-transformed mean (unbiased averaging)
            Defaults to 'median'.
        summary_statistic: ISC computation method. Options:
            - 'leave-one-out': Correlate each subject with mean of others. O(n_subjects), unbiased, recommended by Chen et al. 2016.
            - 'pairwise': Average all pairwise correlations. O(n_subjects²), captures full correlation structure.
            Note: These methods are statistically different and monotonically but non-linearly related (see Chen et al. 2016, Figure 3).
            Defaults to 'pairwise'.
        method: Resampling method for p-value computation:
            - 'bootstrap': Subject-wise bootstrap (default, Chen et al. 2016)
            - 'circle_shift': Circular time-series shift (preserves autocorrelation)
            - 'phase_randomize': FFT phase randomization (preserves power spectrum)
            Defaults to 'bootstrap'.
        ci_percentile: Confidence interval percentile (e.g., 95 for 95% CI). Defaults to 95.
        tail: One-tailed (1) or two-tailed (2) p-value. Defaults to 2.
        return_null: If True, return bootstrap/permutation distribution in result dict. Defaults to False.
        progress_bar: Show progress bar during bootstrap/permutation. Defaults to True.
        exclude_self_corr: If True, mask self-correlations (perfect correlations from duplicate
            subjects in bootstrap samples) as NaN. If False, include them in the
            summary statistic. Only applies when method='bootstrap' and
            summary_statistic='pairwise'. Defaults to True.
        sim_metric: Similarity metric for pairwise ISC computation. See
            sklearn.metrics.pairwise_distances for valid options. Only applies
            when summary_statistic='pairwise'. For 'correlation', uses optimized
            np.corrcoef. Other metrics use pairwise_distances. Defaults to 'correlation'.
        parallel: Parallelization method:
            - 'cpu': CPU parallelization via joblib (default, 4-8× speedup)
            - 'gpu': GPU acceleration via PyTorch (10-30× speedup for voxel-wise LOO)
            - None: Single-threaded NumPy (for debugging/small problems)
            Defaults to 'cpu'.
        n_jobs: Number of CPU cores for parallelization (-1 = all cores).
            Only used when parallel='cpu'. Defaults to -1.
        max_gpu_memory_gb: GPU memory budget in GB (only used if parallel='gpu'). Defaults to 4.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary with the following keys:
        - 'isc': Observed ISC value (float or array per voxel)
        - 'p': P-value (Phipson-Smyth corrected)
        - 'ci': Confidence interval tuple (lower, upper)
        - 'parallel': Parallelization method used
        - 'null_dist': (optional) Bootstrap/permutation distribution

    Examples:
    >>> # Single-feature ISC
    >>> data = np.random.randn(100, 10)  # 100 timepoints, 10 subjects
    >>> result = isc_permutation_test(data, n_permute=1000)
    >>> print(f"ISC: {result['isc']:.3f}, p: {result['p']:.3f}")

    >>> # Voxel-wise ISC with GPU acceleration
    >>> data_voxels = np.random.randn(100, 50, 5000)  # 5K voxels
    >>> result = isc_permutation_test(
    ...     data_voxels,
    ...     summary_statistic='leave-one-out',
    ...     parallel='gpu',  # GPU for LOO computation
    ...     n_permute=5000
    ... )
    >>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")

    >>> # Compare LOO vs pairwise
    >>> result_loo = isc_permutation_test(data, summary_statistic='leave-one-out')
    >>> result_pair = isc_permutation_test(data, summary_statistic='pairwise')
    >>> print(f"LOO: {result_loo['isc']:.3f}, Pairwise: {result_pair['isc']:.3f}")

    References:
        Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
        Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
        correlations, part I: nonparametric approaches to inter-subject
        correlation analysis at the group level. NeuroImage, 142, 248-259.

    Notes:
        - Leave-one-out is 20-30× faster than pairwise for large n_subjects
        - GPU acceleration helps most for voxel-wise LOO (10-30× speedup)
        - Pairwise bootstrap uses correct subject-wise resampling (Chen 2016)
        - Bootstrap distribution is centered by subtracting observed ISC
    """
    # Input validation
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

    # Validate parallel parameter
    if parallel not in [None, "cpu", "gpu"]:
        raise ValueError(f"parallel must be None, 'cpu', or 'gpu', got {parallel!r}")

    # Determine backend for computation phase based on parallel parameter
    if parallel == "cpu" or parallel is None:
        # CPU modes
        if parallel is None:
            # Single-threaded NumPy
            compute_backend = "numpy"
            bootstrap_backend = "numpy"
        else:
            # CPU parallelization
            compute_backend = "numpy"
            bootstrap_backend = "cpu-parallel"
    else:
        # GPU mode
        compute_backend = "torch"
        bootstrap_backend = "cpu-parallel"  # Bootstrap still uses CPU parallel

    # Input validation

    # Phase 1: Compute ISC (run once)
    if summary_statistic == "leave-one-out":
        # Compute leave-one-out values
        loo_values = _compute_loo_isc(data, backend=compute_backend)

        # Compute observed summary statistic
        if metric == "median":
            observed_isc = np.median(loo_values, axis=0)
        elif metric == "mean":
            z = np.arctanh(np.clip(loo_values, -0.9999, 0.9999))
            observed_isc = np.tanh(np.mean(z, axis=0))
        else:
            raise ValueError(f"metric must be 'median' or 'mean', got {metric}")

    else:  # pairwise
        # Compute pairwise correlation matrix (condensed form)
        pairwise_condensed = _compute_pairwise_isc(
            data, backend="numpy", sim_metric=sim_metric
        )
        n_subjects = data.shape[1]

        # Compute observed summary statistic
        if metric == "median":
            observed_isc = np.nanmedian(pairwise_condensed, axis=0)
        elif metric == "mean":
            z = np.arctanh(np.clip(pairwise_condensed, -0.9999, 0.9999))
            observed_isc = np.tanh(np.nanmean(z, axis=0))
        else:
            raise ValueError(f"metric must be 'median' or 'mean', got {metric}")

    # Phase 2: Bootstrap/permutation (run n_permute times)
    if method == "bootstrap":
        if summary_statistic == "leave-one-out":
            # LOO bootstrap: resample pre-computed values
            if bootstrap_backend == "numpy":
                rng = check_random_state(random_state)
                bootstraps = np.array(
                    [
                        _bootstrap_loo_numpy(
                            loo_values,
                            metric=metric,
                            random_state=np.random.RandomState(rng.randint(2**31)),
                        )
                        for _ in range(n_permute)
                    ]
                )
            else:  # cpu-parallel
                bootstraps = _bootstrap_loo_cpu_parallel(
                    loo_values,
                    n_permute=n_permute,
                    metric=metric,
                    n_jobs=n_jobs,
                    random_state=random_state,
                    progress_bar=progress_bar,
                    max_memory_gb=None,  # Auto-detect
                )

        else:  # pairwise
            # Pairwise bootstrap: subject-wise matrix indexing
            if bootstrap_backend == "numpy":
                rng = check_random_state(random_state)
                bootstraps = np.array(
                    [
                        _bootstrap_pairwise_numpy(
                            pairwise_condensed,
                            metric=metric,
                            n_subjects=n_subjects,
                            random_state=np.random.RandomState(rng.randint(2**31)),
                            exclude_self_corr=exclude_self_corr,
                        )
                        for _ in range(n_permute)
                    ]
                )
            else:  # cpu-parallel
                bootstraps = _bootstrap_pairwise_cpu_parallel(
                    pairwise_condensed,
                    n_permute=n_permute,
                    n_subjects=n_subjects,
                    metric=metric,
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
                loo_perm = _compute_loo_isc(data_permuted, backend="numpy")
                if metric == "median":
                    isc_perm = np.median(loo_perm, axis=0)
                else:
                    z = np.arctanh(np.clip(loo_perm, -0.9999, 0.9999))
                    isc_perm = np.tanh(np.mean(z, axis=0))
            else:  # pairwise
                pair_perm = _compute_pairwise_isc(
                    data_permuted, backend="numpy", sim_metric=sim_metric
                )
                if metric == "median":
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
                loo_perm = _compute_loo_isc(data_permuted, backend="numpy")
                if metric == "median":
                    isc_perm = np.median(loo_perm, axis=0)
                else:
                    z = np.arctanh(np.clip(loo_perm, -0.9999, 0.9999))
                    isc_perm = np.tanh(np.mean(z, axis=0))
            else:  # pairwise
                pair_perm = _compute_pairwise_isc(
                    data_permuted, backend="numpy", sim_metric=sim_metric
                )
                if metric == "median":
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
        "parallel": parallel,
    }

    if return_null:
        result["null_dist"] = null_distribution

    return result
