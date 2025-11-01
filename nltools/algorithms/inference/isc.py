"""
Intersubject Correlation (ISC) with GPU-Accelerated Permutation Testing.

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
from scipy.spatial.distance import squareform
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise_distances

from .utils import _compute_pvalue, EPSILON


# ============================================================================
# Phase 1: Leave-One-Out (LOO) ISC Computation
# ============================================================================


def _compute_loo_isc(data, backend="numpy"):
    """
    Compute leave-one-out intersubject correlation.

    For each subject, correlates their data with the mean of all other
    subjects. This provides an unbiased estimate of subject-level ISC
    and is computationally efficient (O(n_subjects) vs O(n_subjects²)).

    Parameters
    ----------
    data : ndarray
        Data array with one of the following shapes:
        - (n_observations, n_subjects): Single feature
        - (n_observations, n_subjects, n_voxels): Voxel-wise
    backend : {'numpy', 'torch'}, default='numpy'
        Computation backend. Use 'torch' for GPU acceleration on
        voxel-wise data (10-30× speedup for large n_voxels).

    Returns
    -------
    loo_values : ndarray
        Leave-one-out ISC values:
        - Shape (n_subjects,) for single feature
        - Shape (n_subjects, n_voxels) for voxel-wise

    Examples
    --------
    >>> data = np.random.randn(100, 10)  # 100 timepoints, 10 subjects
    >>> loo = _compute_loo_isc(data)
    >>> loo.shape
    (10,)

    >>> # Voxel-wise
    >>> data_voxels = np.random.randn(100, 10, 1000)  # 1000 voxels
    >>> loo = _compute_loo_isc(data_voxels, backend='torch')
    >>> loo.shape
    (10, 1000)

    Notes
    -----
    For each subject i, computes: corr(subject_i, mean(all other subjects))
    This is the method recommended by Chen et al. (2016) for unbiased ISC.
    """
    if backend == "numpy":
        return _compute_loo_isc_numpy(data)
    elif backend == "torch":
        return _compute_loo_isc_gpu(data)
    else:
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

    elif data.ndim == 3:
        # Voxel-wise: (n_observations, n_subjects, n_voxels)
        n_obs, n_subjects, n_voxels = data.shape
        loo_values = np.zeros((n_subjects, n_voxels))

        for v in range(n_voxels):
            voxel_data = data[:, :, v]
            for i in range(n_subjects):
                others_mean = voxel_data[:, np.arange(n_subjects) != i].mean(axis=1)
                loo_values[i, v] = np.corrcoef(voxel_data[:, i], others_mean)[0, 1]

        return loo_values

    else:
        raise ValueError(f"data must be 2D or 3D, got shape {data.shape}")


def _batch_correlation_gpu(x, y):
    """
    Compute correlation between x and y in parallel across features.

    Parameters
    ----------
    x, y : torch.Tensor, shape (n_observations, n_features)
        Data tensors on GPU

    Returns
    -------
    correlations : torch.Tensor, shape (n_features,)
        Correlation coefficients
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
    """
    GPU-accelerated leave-one-out ISC computation.

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
    """
    Compute pairwise intersubject correlation (condensed form).

    Computes all n×(n-1)/2 pairwise correlations between subjects and
    stores in condensed upper-triangle format for memory efficiency.

    Parameters
    ----------
    data : ndarray
        Data array with one of the following shapes:
        - (n_observations, n_subjects): Single feature
        - (n_observations, n_subjects, n_voxels): Voxel-wise
    backend : {'numpy', 'torch'}, default='numpy'
        Computation backend. Use 'torch' for GPU acceleration on
        voxel-wise data.
    sim_metric : str, default='correlation'
        Similarity metric. See sklearn.metrics.pairwise_distances for valid
        options. 'correlation' uses optimized np.corrcoef. Other metrics
        use pairwise_distances.

    Returns
    -------
    pairwise_condensed : ndarray
        Pairwise correlations in condensed form (upper triangle):
        - Shape (n_pairs,) for single feature, where n_pairs = n*(n-1)/2
        - Shape (n_pairs, n_voxels) for voxel-wise

    Examples
    --------
    >>> data = np.random.randn(100, 5)  # 5 subjects
    >>> pairwise = _compute_pairwise_isc(data)
    >>> pairwise.shape
    (10,)  # 5*4/2 = 10 pairs

    Notes
    -----
    Uses np.corrcoef for efficient correlation matrix computation when
    sim_metric='correlation', otherwise uses pairwise_distances.
    """
    if backend == "numpy":
        return _compute_pairwise_isc_numpy(data, sim_metric=sim_metric)
    elif backend == "torch":
        if sim_metric != "correlation":
            raise ValueError(
                f"GPU backend only supports sim_metric='correlation', got {sim_metric}"
            )
        return _compute_pairwise_isc_gpu(data)
    else:
        raise ValueError(f"backend must be 'numpy' or 'torch', got {backend}")


def _compute_pairwise_isc_numpy(data, sim_metric="correlation"):
    """NumPy implementation of pairwise ISC (condensed storage)."""
    if sim_metric == "correlation":
        # Fast path: use np.corrcoef (optimized C implementation)
        if data.ndim == 2:
            # Single feature: (n_observations, n_subjects)
            corr_matrix = np.corrcoef(data.T)
            return squareform(corr_matrix, checks=False)
        elif data.ndim == 3:
            # Voxel-wise: (n_observations, n_subjects, n_voxels)
            n_obs, n_subjects, n_voxels = data.shape
            n_pairs = n_subjects * (n_subjects - 1) // 2
            pairwise_all = np.zeros((n_pairs, n_voxels))
            for v in range(n_voxels):
                corr_matrix = np.corrcoef(data[:, :, v].T)
                pairwise_all[:, v] = squareform(corr_matrix, checks=False)
            return pairwise_all
        else:
            raise ValueError(f"data must be 2D or 3D, got shape {data.shape}")
    else:
        # General path: use pairwise_distances for other metrics
        # Convert distance to similarity: similarity = 1 - distance
        if data.ndim == 2:
            # Single feature: (n_observations, n_subjects)
            dist_matrix = pairwise_distances(data.T, metric=sim_metric)
            sim_matrix = 1 - dist_matrix
            return squareform(sim_matrix, checks=False)
        elif data.ndim == 3:
            # Voxel-wise: (n_observations, n_subjects, n_voxels)
            n_obs, n_subjects, n_voxels = data.shape
            n_pairs = n_subjects * (n_subjects - 1) // 2
            pairwise_all = np.zeros((n_pairs, n_voxels))
            for v in range(n_voxels):
                dist_matrix = pairwise_distances(data[:, :, v].T, metric=sim_metric)
                sim_matrix = 1 - dist_matrix
                pairwise_all[:, v] = squareform(sim_matrix, checks=False)
            return pairwise_all
        else:
            raise ValueError(f"data must be 2D or 3D, got shape {data.shape}")


def _batch_corrcoef_gpu(data_gpu):
    """
    Compute correlation matrices in parallel across voxels on GPU.

    Parameters
    ----------
    data_gpu : torch.Tensor, shape (n_voxels, n_subjects, n_observations)
        Data tensor on GPU (transposed for efficient correlation)

    Returns
    -------
    corr_matrices : torch.Tensor, shape (n_voxels, n_subjects, n_subjects)
        Correlation matrices for each voxel
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
    """
    GPU-accelerated pairwise ISC computation.

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
# Phase 3: Leave-One-Out Bootstrap
# ============================================================================


def _bootstrap_loo_numpy(loo_values, metric="median", random_state=None):
    """
    Bootstrap LOO ISC by resampling subjects.

    Implements the subject-wise bootstrap method from Chen et al. (2016).
    Resamples the pre-computed LOO values (not raw data) for efficiency.

    Parameters
    ----------
    loo_values : ndarray
        Pre-computed LOO values:
        - Shape (n_subjects,) for single feature
        - Shape (n_subjects, n_voxels) for voxel-wise
    metric : {'median', 'mean'}, default='median'
        Summary statistic to compute from bootstrap sample
    random_state : int or RandomState, optional
        Random state for reproducibility

    Returns
    -------
    summary : float or ndarray
        Bootstrap summary statistic (scalar or per-voxel array)

    Notes
    -----
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
    elif metric == "mean":
        # Fisher z-transform for unbiased mean
        z = np.arctanh(np.clip(boot_values, -0.9999, 0.9999))
        return np.tanh(np.mean(z, axis=0))
    else:
        raise ValueError(f"metric must be 'median' or 'mean', got {metric}")


def _bootstrap_loo_cpu_parallel(
    loo_values,
    n_permute=5000,
    metric="median",
    n_jobs=-1,
    random_state=None,
    progress_bar=True,
):
    """
    CPU-parallel LOO bootstrap using joblib.

    Efficiently parallelizes bootstrap resampling across CPU cores.
    Uses deterministic seed generation for reproducibility.

    Parameters
    ----------
    loo_values : ndarray
        Pre-computed LOO values (n_subjects,) or (n_subjects, n_voxels)
    n_permute : int, default=5000
        Number of bootstrap iterations
    metric : {'median', 'mean'}, default='median'
        Summary statistic
    n_jobs : int, default=-1
        Number of CPU cores (-1 = all cores)
    random_state : int or RandomState, optional
        Random seed for reproducibility
    progress_bar : bool, default=True
        Show progress bar

    Returns
    -------
    bootstraps : ndarray, shape (n_permute,) or (n_permute, n_voxels)
        Bootstrap distribution
    """
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm

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
    """
    Bootstrap pairwise ISC by subject-wise matrix indexing.

    Implements the correct bootstrap procedure for correlation matrices
    (Chen et al. 2016): resample subjects, extract submatrix, mask
    same-subject pairs (self-correlations from duplicates).

    Parameters
    ----------
    pairwise_condensed : ndarray
        Pre-computed pairwise correlations in condensed form:
        - Shape (n_pairs,) for single feature
        - Shape (n_pairs, n_voxels) for voxel-wise
    metric : {'median', 'mean'}, default='median'
        Summary statistic
    bootstrap_subjects : ndarray, optional
        Pre-generated bootstrap subject indices (for testing)
    n_subjects : int, optional
        Number of subjects (required if bootstrap_subjects is None)
    random_state : int or RandomState, optional
        Random state for sampling
    exclude_self_corr : bool, default=True
        If True, mask self-correlations (perfect correlations from duplicate
        subjects) as NaN. If False, include them in the summary statistic.

    Returns
    -------
    summary : float or ndarray
        Bootstrap summary statistic

    Notes
    -----
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
        boot_condensed_list = []

        for v in range(n_voxels):
            # Process each voxel independently
            corr_matrix = squareform(pairwise_condensed[:, v], force="tomatrix")
            np.fill_diagonal(corr_matrix, 1.0)

            # Index by bootstrap subjects
            boot_matrix = corr_matrix[bootstrap_subjects, :][:, bootstrap_subjects]

            # Mask same-subject pairs if requested
            if exclude_self_corr:
                boot_matrix[boot_matrix >= 0.99999] = np.nan

            # Extract triangle
            boot_condensed_list.append(squareform(boot_matrix, checks=False))

        boot_condensed = np.column_stack(boot_condensed_list)

    # Compute summary (ignoring NaNs from masked pairs)
    axis = 0 if boot_condensed.ndim > 1 else None

    if metric == "median":
        return np.nanmedian(boot_condensed, axis=axis)
    elif metric == "mean":
        # Fisher z-transform
        z = np.arctanh(np.clip(boot_condensed, -0.9999, 0.9999))
        return np.tanh(np.nanmean(z, axis=axis))
    else:
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
):
    """
    CPU-parallel pairwise bootstrap using joblib.

    Same pattern as LOO bootstrap, but operates on pairwise correlation
    matrices with subject-wise indexing.

    Parameters
    ----------
    pairwise_condensed : ndarray
        Pre-computed pairwise correlations (n_pairs,) or (n_pairs, n_voxels)
    n_permute : int, default=5000
        Number of bootstrap iterations
    n_subjects : int
        Number of subjects in original data
    metric : {'median', 'mean'}, default='median'
        Summary statistic
    n_jobs : int, default=-1
        Number of CPU cores
    random_state : int or RandomState, optional
        Random seed
    progress_bar : bool, default=True
        Show progress bar
    exclude_self_corr : bool, default=True
        If True, mask self-correlations (perfect correlations from duplicate
        subjects) as NaN. If False, include them in the summary statistic.

    Returns
    -------
    bootstraps : ndarray, shape (n_permute,) or (n_permute, n_voxels)
        Bootstrap distribution
    """
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm

    if n_subjects is None:
        raise ValueError("n_subjects is required for pairwise bootstrap")

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
    data,
    n_permute=5000,
    metric="median",
    summary_statistic="pairwise",
    method="bootstrap",
    ci_percentile=95,
    tail=2,
    backend=None,
    n_jobs=-1,
    max_memory_gb=4,
    random_state=None,
    return_null=False,
    progress_bar=True,
    exclude_self_corr=True,
    sim_metric="correlation",
):
    """
    Compute intersubject correlation with permutation testing.

    Supports both leave-one-out and pairwise ISC computation modes with
    GPU acceleration for large voxel-wise problems and CPU-parallel
    bootstrap resampling.

    Parameters
    ----------
    data : ndarray
        Data array with one of the following shapes:
        - (n_observations, n_subjects): Single feature ISC
        - (n_observations, n_subjects, n_voxels): Voxel-wise ISC
    n_permute : int, default=5000
        Number of bootstrap iterations or permutations
    metric : {'median', 'mean'}, default='median'
        Summary statistic to aggregate ISC values.
        - 'median': Direct median (robust to outliers)
        - 'mean': Fisher z-transformed mean (unbiased averaging)
    summary_statistic : {'leave-one-out', 'pairwise'}, default='leave-one-out'
        ISC computation method. Options:
          - 'leave-one-out': Correlate each subject with mean of others. O(n_subjects), unbiased, recommended by Chen et al. 2016.
          - 'pairwise': Average all pairwise correlations. O(n_subjects²), captures full correlation structure.
        Note: These methods are statistically different and monotonically but non-linearly related (see Chen et al. 2016, Figure 3).
    method : {'bootstrap', 'circle_shift', 'phase_randomize'}, default='bootstrap'
        Resampling method for p-value computation:
        - 'bootstrap': Subject-wise bootstrap (default, Chen et al. 2016)
        - 'circle_shift': Circular time-series shift (preserves autocorrelation)
        - 'phase_randomize': FFT phase randomization (preserves power spectrum)
    ci_percentile : float, default=95
        Confidence interval percentile (e.g., 95 for 95% CI)
    tail : {1, 2}, default=2
        One-tailed (1) or two-tailed (2) p-value
    backend : {'numpy', 'cpu-parallel', 'torch', 'auto'}, optional
        Computation backend:
        - 'numpy': Single-threaded (simple, portable)
        - 'cpu-parallel': Joblib parallelization (default, 4-8× speedup)
        - 'torch': GPU acceleration (10-30× speedup for voxel-wise LOO)
        - 'auto': Automatically select based on problem size
        Default: 'cpu-parallel' for bootstrap, 'numpy' for computation
    n_jobs : int, default=-1
        Number of CPU cores for parallelization (-1 = all cores)
    max_memory_gb : float, default=4
        GPU memory budget (only used if backend='torch')
    random_state : int or RandomState, optional
        Random seed for reproducibility
    return_null : bool, default=False
        If True, return bootstrap/permutation distribution in result dict
    progress_bar : bool, default=True
        Show progress bar during bootstrap/permutation
    exclude_self_corr : bool, default=True
        If True, mask self-correlations (perfect correlations from duplicate
        subjects in bootstrap samples) as NaN. If False, include them in the
        summary statistic. Only applies when method='bootstrap' and
        summary_statistic='pairwise'.
    sim_metric : str, default='correlation'
        Similarity metric for pairwise ISC computation. See
        sklearn.metrics.pairwise_distances for valid options. Only applies
        when summary_statistic='pairwise'. For 'correlation', uses optimized
        np.corrcoef. Other metrics use pairwise_distances.

    Returns
    -------
    result : dict
        Dictionary with the following keys:
        - 'isc': Observed ISC value (float or array per voxel)
        - 'p': P-value (Phipson-Smyth corrected)
        - 'ci': Confidence interval tuple (lower, upper)
        - 'null_distribution': (optional) Bootstrap/permutation distribution

    Examples
    --------
    >>> # Single-feature ISC
    >>> data = np.random.randn(100, 10)  # 100 timepoints, 10 subjects
    >>> result = isc_permutation_test(data, n_permute=1000)
    >>> print(f"ISC: {result['isc']:.3f}, p: {result['p']:.3f}")

    >>> # Voxel-wise ISC with GPU acceleration
    >>> data_voxels = np.random.randn(100, 50, 5000)  # 5K voxels
    >>> result = isc_permutation_test(
    ...     data_voxels,
    ...     summary_statistic='leave-one-out',
    ...     backend='torch',  # GPU for LOO computation
    ...     n_permute=5000
    ... )
    >>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")

    >>> # Compare LOO vs pairwise
    >>> result_loo = isc_permutation_test(data, summary_statistic='leave-one-out')
    >>> result_pair = isc_permutation_test(data, summary_statistic='pairwise')
    >>> print(f"LOO: {result_loo['isc']:.3f}, Pairwise: {result_pair['isc']:.3f}")

    References
    ----------
    Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
    Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
    correlations, part I: nonparametric approaches to inter-subject
    correlation analysis at the group level. NeuroImage, 142, 248-259.

    Notes
    -----
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

    # Validate sim_metric (only used for pairwise)
    if summary_statistic == "pairwise" and sim_metric != "correlation":
        # Warn if using non-correlation metric with GPU backend
        if backend == "torch":
            import warnings

            warnings.warn(
                f"sim_metric='{sim_metric}' not supported with GPU backend. "
                "Falling back to CPU. Use sim_metric='correlation' for GPU acceleration.",
                UserWarning,
            )

    # Determine backend for computation phase
    if backend is None:
        # Default: numpy for computation, cpu-parallel for bootstrap
        compute_backend = "numpy"
        bootstrap_backend = "cpu-parallel"
    elif backend == "auto":
        # Auto-select based on problem size
        is_voxelwise = data.ndim == 3
        n_voxels = data.shape[2] if is_voxelwise else 1

        # Use GPU for large voxel-wise LOO problems
        if is_voxelwise and n_voxels > 5000 and summary_statistic == "leave-one-out":
            try:
                import torch

                compute_backend = "torch" if torch.cuda.is_available() else "numpy"
            except ImportError:
                compute_backend = "numpy"
        else:
            compute_backend = "numpy"

        bootstrap_backend = "cpu-parallel"
    else:
        compute_backend = backend
        bootstrap_backend = "cpu-parallel" if backend != "numpy" else "numpy"

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
    }

    if return_null:
        result["null_distribution"] = bootstraps

    return result
