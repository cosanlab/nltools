"""
Correlation permutation test implementations.

This module provides CPU-parallel and GPU-batched implementations
of correlation permutation tests for assessing statistical significance
of correlations.
"""

import numpy as np
from typing import Union, Optional
from sklearn.utils import check_random_state
from scipy.stats import rankdata, kendalltau

from nltools.backends import Backend, auto_select_backend
from .utils import _compute_pvalue, _auto_batch_size


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation coefficient(s).

    Args:
        x (np.ndarray): Data array, shape (n_samples,) or (n_permute, n_samples)
        y (np.ndarray): Data array, shape (n_samples,)

    Returns:
        np.ndarray: Correlation coefficient(s)
            - scalar if x is 1D
            - shape (n_permute,) if x is 2D

    Notes:
        Uses centered data for numerical stability.
        Handles broadcasting for vectorized computation.
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

    # Handle division by zero (constant data)
    correlations = np.where(denominator != 0, numerator / denominator, 0.0)

    if squeeze_output:
        return float(correlations[0])
    return correlations


def _spearman_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute Spearman rank correlation coefficient(s).

    Spearman correlation is the Pearson correlation of the rank-transformed data.
    It measures monotonic (not necessarily linear) relationships.

    Args:
        x (np.ndarray): Data array, shape (n_samples,) or (n_permute, n_samples)
        y (np.ndarray): Data array, shape (n_samples,)

    Returns:
        np.ndarray: Spearman correlation coefficient(s)
            - scalar if x is 1D
            - shape (n_permute,) if x is 2D

    Notes:
        Uses scipy.stats.rankdata for rank transformation, then applies
        Pearson correlation to ranks. Handles tied ranks appropriately.
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
        x_ranked[i] = rankdata(x[i], method='average')

    y_ranked = rankdata(y, method='average')

    # Apply Pearson correlation to ranks
    # Center ranked data
    x_centered = x_ranked - x_ranked.mean(axis=1, keepdims=True)
    y_centered = y_ranked - y_ranked.mean()

    # Compute correlation
    numerator = (x_centered @ y_centered) / n_samples
    denominator = x_centered.std(axis=1, ddof=0) * y_centered.std(ddof=0)

    # Handle division by zero (constant data - all tied ranks)
    correlations = np.where(denominator != 0, numerator / denominator, 0.0)

    if squeeze_output:
        return float(correlations[0])
    return correlations


def _kendall_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute Kendall rank correlation coefficient(s).

    Kendall tau correlation measures ordinal association based on concordant
    and discordant pairs. More robust than Spearman for small samples or
    data with many tied ranks.

    Args:
        x (np.ndarray): Data array, shape (n_samples,) or (n_permute, n_samples)
        y (np.ndarray): Data array, shape (n_samples,)

    Returns:
        np.ndarray: Kendall correlation coefficient(s)
            - scalar if x is 1D
            - shape (n_permute,) if x is 2D

    Notes:
        Uses scipy.stats.kendalltau for each correlation computation.
        This is O(n^2) complexity, slower than Pearson/Spearman.
        For vectorized case, computes each correlation separately.
    """
    # Handle dimensions
    if x.ndim == 1:
        # Single correlation
        tau, _ = kendalltau(x, y)
        return float(tau) if not np.isnan(tau) else 0.0
    else:
        # Vectorized: compute each permutation separately
        n_permute = x.shape[0]
        correlations = np.empty(n_permute)
        for i in range(n_permute):
            tau, _ = kendalltau(x[i], y)
            correlations[i] = tau if not np.isnan(tau) else 0.0
        return correlations


def _correlation_permutation_cpu_parallel(
    data1: np.ndarray,
    data2: np.ndarray,
    n_permute: int,
    metric: str,
    tail: int,
    return_null: bool,
    n_jobs: int,
    random_state: Optional[int],
    single_feature: bool = False,
) -> dict:
    """
    Correlation permutation test using CPU parallelization with joblib.

    Memory-efficient implementation that processes one permutation per worker.
    Randomly shuffles data1 and computes correlation with data2.

    Args:
        data1 (np.ndarray): Data to permute, shape (n_samples, n_features)
        data2 (np.ndarray): Data to correlate with, shape (n_samples, n_features)
        n_permute (int): Number of permutations
        metric (str): Correlation metric ('pearson', 'spearman', 'kendall')
        tail (int): Test type (1 or 2)
        return_null (bool): Whether to return null distribution
        n_jobs (int): Number of parallel jobs (-1 = all cores)
        random_state (int, optional): Random seed for reproducibility
        single_feature (bool): Whether data is single feature

    Returns:
        dict: Same format as main function, with 'backend' indicating CPU parallel
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm

    # Setup random state and generate seeds for workers
    rng = check_random_state(random_state)
    MAX_INT = 2**31 - 1
    seeds = rng.randint(MAX_INT, size=n_permute)

    # Get dimensions (data already reshaped by caller)
    n_samples, n_features = data1.shape

    # Select correlation function
    if metric == 'pearson':
        corr_func = _pearson_correlation
    elif metric == 'spearman':
        corr_func = _spearman_correlation
    elif metric == 'kendall':
        corr_func = _kendall_correlation
    else:
        raise NotImplementedError(f"Metric '{metric}' not yet implemented")

    # Compute observed correlation
    if n_features == 1:
        obs_corr = corr_func(data1[:, 0], data2[:, 0])
        obs_corr = np.array([obs_corr])
    else:
        obs_corr = np.array([
            corr_func(data1[:, i], data2[:, i])
            for i in range(n_features)
        ])

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
        else:
            return np.array([
                corr_func(perm_data1[:, i], data2[:, i])
                for i in range(n_features)
            ])

    # Execute in parallel with progress bar
    null_dist = Parallel(n_jobs=n_jobs)(
        delayed(_compute_one_perm)(seeds[i])
        for i in tqdm(range(n_permute), desc="CPU parallel perms", unit="perm")
    )
    null_dist = np.array(null_dist)  # Shape: (n_permute, n_features)

    # Compute p-values
    p_values = _compute_pvalue(obs_corr, null_dist, tail=tail)

    # Return to original shape
    if single_feature:
        obs_corr = float(obs_corr[0])
        p_values = float(p_values[0])

    # Determine backend name
    if n_jobs == -1:
        import multiprocessing
        n_cores = multiprocessing.cpu_count()
        backend_name = f"cpu-parallel-{n_cores}"
    else:
        backend_name = f"cpu-parallel-{n_jobs}"

    # Build result
    result = {
        "correlation": obs_corr,
        "p": p_values,
        "backend": backend_name,
    }

    if return_null:
        if single_feature:
            null_dist = null_dist.squeeze()
        result["null_dist"] = null_dist

    return result


def _correlation_permutation_gpu_batched(
    data1: np.ndarray,
    data2: np.ndarray,
    n_permute: int,
    metric: str,
    tail: int,
    return_null: bool,
    backend: Backend,
    max_gpu_memory_gb: float,
    random_state,
    single_feature: bool = False,
) -> dict:
    """
    Correlation permutation test using GPU with automatic batching.

    Processes permutations in batches to avoid GPU OOM. Transfers data once
    and reuses across batches for efficiency.

    Args:
        data1 (np.ndarray): Data to permute, shape (n_samples, n_features)
        data2 (np.ndarray): Data to correlate with, shape (n_samples, n_features)
        n_permute (int): Number of permutations
        metric (str): Correlation metric ('pearson')
        tail (int): Test type (1 or 2)
        return_null (bool): Whether to return null distribution
        backend (Backend): Backend instance (must be PyTorch)
        max_gpu_memory_gb (float): Maximum GPU memory budget
        random_state: Random state instance
        single_feature (bool): Whether data is single feature

    Returns:
        dict: Same format as main function, with 'backend' indicating GPU device
    """
    import torch
    from tqdm import tqdm

    n_samples, n_features = data1.shape

    # Convert to float32 for GPU efficiency
    data1 = data1.astype(np.float32)
    data2 = data2.astype(np.float32)

    # Select correlation function
    if metric == 'pearson':
        corr_func = _pearson_correlation
    elif metric == 'spearman':
        corr_func = _spearman_correlation
    elif metric == 'kendall':
        corr_func = _kendall_correlation
    else:
        raise NotImplementedError(f"Metric '{metric}' not yet implemented")

    # Compute observed correlation
    if n_features == 1:
        obs_corr = corr_func(data1[:, 0], data2[:, 0])
        obs_corr = np.array([obs_corr])
    else:
        obs_corr = np.array([
            corr_func(data1[:, i], data2[:, i])
            for i in range(n_features)
        ])

    # Determine batch size based on memory budget
    # Memory bottleneck: permuted indices and correlation computation
    batch_size, n_batches = _auto_batch_size(
        n_permute, n_samples, n_features, max_memory_gb=max_gpu_memory_gb
    )

    # Transfer data to device once
    data1_device = backend.to_device(data1)
    data2_device = backend.to_device(data2)

    # Accumulate null distribution across batches
    null_dist_list = []

    # Process permutations in batches with progress bar
    pbar = tqdm(
        total=n_permute,
        desc="GPU permutation batches",
        unit="perm",
        disable=n_batches == 1,
    )

    for batch_idx in range(n_batches):
        # Determine current batch size
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_permute)
        current_batch_size = end_idx - start_idx

        # Generate permutation indices for this batch
        batch_indices = np.array([
            random_state.permutation(n_samples) for _ in range(current_batch_size)
        ])

        # Transfer to device
        batch_indices_device = backend.to_device(batch_indices)
        if backend.name.startswith("torch"):
            batch_indices_device = batch_indices_device.long()

        # Compute correlations for this batch
        batch_corrs = []
        for feat_idx in range(n_features):
            # Get data for this feature
            feat_data1 = data1_device[:, feat_idx]  # (n_samples,)
            feat_data2 = data2_device[:, feat_idx]  # (n_samples,)

            # Permute data1 for all permutations in batch
            # batch_indices_device: (current_batch_size, n_samples)
            perm_data1 = feat_data1[batch_indices_device]  # (current_batch_size, n_samples)

            # Compute correlations
            # Center data
            perm_data1_centered = perm_data1 - torch.mean(perm_data1, dim=1, keepdim=True)
            feat_data2_centered = feat_data2 - torch.mean(feat_data2)

            # Correlation
            numerator = (perm_data1_centered @ feat_data2_centered) / n_samples
            denominator = torch.std(perm_data1_centered, dim=1) * torch.std(feat_data2)

            # Handle division by zero
            correlations = torch.where(
                denominator != 0,
                numerator / denominator,
                torch.zeros_like(numerator)
            )

            batch_corrs.append(correlations)

        # Stack correlations: (current_batch_size, n_features)
        batch_corrs = torch.stack(batch_corrs, dim=1)
        batch_corrs = backend.to_numpy(batch_corrs)

        null_dist_list.append(batch_corrs)

        # Update progress bar
        pbar.update(current_batch_size)

        # Free batch memory
        del batch_indices_device, batch_corrs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar.close()

    # Combine batches: (n_permute, n_features)
    null_dist = np.vstack(null_dist_list)

    # Compute p-values
    p_values = _compute_pvalue(obs_corr, null_dist, tail=tail)

    # Return to original shape
    if single_feature:
        obs_corr = float(obs_corr[0])
        p_values = float(p_values[0])

    # Build result
    result = {
        "correlation": obs_corr,
        "p": p_values,
        "backend": backend.name,
    }

    if return_null:
        if single_feature:
            null_dist = null_dist.squeeze()
        result["null_dist"] = null_dist

    return result


def correlation_permutation_test(
    data1: np.ndarray,
    data2: np.ndarray,
    n_permute: int = 5000,
    metric: str = 'pearson',
    tail: int = 2,
    return_null: bool = False,
    backend: Optional[Union[Backend, str]] = None,
    n_jobs: int = -1,
    max_gpu_memory_gb: float = 4.0,
    random_state: Optional[int] = None,
) -> dict:
    """
    Correlation permutation test.

    Tests whether the correlation between data1 and data2 is significantly
    different from zero by randomly permuting data1 and computing correlations.

    When backend='torch', uses GPU acceleration with automatic batching.
    When backend=None (default), uses efficient CPU parallelization via joblib.

    Args:
        data1 (np.ndarray): Data to permute
            - shape (n_samples,) for single feature
            - shape (n_samples, n_features) for multi-feature
        data2 (np.ndarray): Data to correlate with
            - shape (n_samples,) for single feature
            - shape (n_samples, n_features) for multi-feature
        n_permute (int): Number of permutations (default: 5000)
        metric (str): Correlation metric (default: 'pearson')
            - 'pearson': Pearson correlation (linear relationships)
            - 'spearman': Spearman rank correlation (monotonic relationships)
            - 'kendall': Kendall tau rank correlation (ordinal association, robust to ties)
        tail (int): Test type (default: 2)
            - 1: One-tailed test
            - 2: Two-tailed test
        return_null (bool): If True, return full null distribution (default: False)
        backend (Backend or str, optional): Backend for computation
            - 'numpy': CPU-only (single-threaded)
            - 'torch': GPU acceleration with automatic batching
            - 'auto': Automatically select based on problem size
            - None: Uses efficient CPU parallelization (n_jobs)
        n_jobs (int): Number of CPU cores for parallelization (default: -1 = all cores)
            Only used when backend=None (CPU parallelization mode)
        max_gpu_memory_gb (float): Maximum GPU memory to use in GB (default: 4.0)
            Controls automatic batching to prevent OOM errors. Only used with
            backend='torch' or 'auto'.
        random_state (int, optional): Random seed for reproducibility

    Returns:
        dict: Dictionary with keys:
            - 'correlation' (float or np.ndarray): Observed correlation(s)
            - 'p' (float or np.ndarray): P-value(s)
            - 'null_dist' (np.ndarray): Null distribution (if return_null=True)
            - 'backend' (str): Backend used for computation

    Examples:
        >>> # Single feature (1D arrays)
        >>> x = np.random.randn(100)
        >>> y = x + np.random.randn(100) * 0.5  # Correlated
        >>> result = correlation_permutation_test(x, y, n_permute=5000)
        >>> result['correlation']
        0.85
        >>> result['p']
        0.001

        >>> # Multi-feature (2D arrays)
        >>> data1 = np.random.randn(100, 10)  # 100 samples, 10 features
        >>> data2 = data1 + np.random.randn(100, 10) * 0.3  # Correlated
        >>> result = correlation_permutation_test(data1, data2, n_permute=5000)
        >>> result['correlation'].shape
        (10,)
        >>> result['p'].shape
        (10,)

        >>> # GPU acceleration
        >>> result = correlation_permutation_test(data1, data2, n_permute=5000, backend='torch')

    Notes:
        - Default (backend=None): CPU parallelization with joblib (4-8× speedup)
        - GPU backend (torch): Fastest for large problems with automatic batching (Pearson only)
        - NumPy backend: Single-threaded, use for small problems or debugging
        - For multi-feature data, each feature pair tested independently
        - Spearman/Kendall: CPU-parallel and NumPy backends only (GPU not yet implemented)
        - Kendall is O(n^2) complexity, slower than Pearson/Spearman for large samples
    """
    # Input validation
    data1 = np.asarray(data1, dtype=np.float64)
    data2 = np.asarray(data2, dtype=np.float64)

    if data1.ndim not in [1, 2]:
        raise ValueError(f"data1 must be 1D or 2D, got shape {data1.shape}")
    if data2.ndim not in [1, 2]:
        raise ValueError(f"data2 must be 1D or 2D, got shape {data2.shape}")
    if tail not in [1, 2]:
        raise ValueError(f"tail must be 1 or 2, got {tail}")
    if metric not in ['pearson', 'spearman', 'kendall']:
        raise ValueError(f"metric must be 'pearson', 'spearman', or 'kendall', got '{metric}'")

    # Handle shape
    single_feature = data1.ndim == 1 and data2.ndim == 1
    if data1.ndim == 1:
        data1 = data1[:, np.newaxis]
    if data2.ndim == 1:
        data2 = data2[:, np.newaxis]

    # Check dimensions match
    if data1.shape != data2.shape:
        raise ValueError(
            f"data1 and data2 must have same shape, "
            f"got {data1.shape} and {data2.shape}"
        )

    n_samples, n_features = data1.shape

    # Decide execution mode: GPU backend vs CPU parallelization
    use_cpu_parallel = backend is None

    if use_cpu_parallel:
        # CPU parallelization mode (memory-efficient fallback)
        return _correlation_permutation_cpu_parallel(
            data1, data2, n_permute, metric, tail, return_null,
            n_jobs, random_state, single_feature
        )
    else:
        # GPU/Backend mode
        if isinstance(backend, str):
            if backend == "auto":
                # Auto-select based on problem size and GPU availability
                backend = auto_select_backend(n_samples, n_features, cv=n_permute // 1000)
            else:
                backend = Backend(backend)

    # GPU backend doesn't support Spearman/Kendall yet (ranking on GPU is complex)
    if not use_cpu_parallel and metric in ['spearman', 'kendall']:
        raise NotImplementedError(
            f"{metric.capitalize()} correlation on GPU backend not yet implemented. "
            "Use backend=None for CPU-parallel or backend='numpy' for sequential."
        )

    # Setup random state
    rng = check_random_state(random_state)

    # Compute null distribution based on backend
    if backend.name == "numpy":
        # NumPy: Sequential processing (simple, memory-efficient)

        # Select correlation function
        if metric == 'pearson':
            corr_func = _pearson_correlation
        elif metric == 'spearman':
            corr_func = _spearman_correlation
        else:
            raise NotImplementedError(f"Metric '{metric}' not yet implemented")

        # Compute observed correlation
        if n_features == 1:
            obs_corr = corr_func(data1[:, 0], data2[:, 0])
            obs_corr = np.array([obs_corr])
        else:
            obs_corr = np.array([
                corr_func(data1[:, i], data2[:, i])
                for i in range(n_features)
            ])

        # Generate null distribution
        null_dist = []
        for _ in range(n_permute):
            # Permute data1 indices
            indices = rng.permutation(n_samples)
            perm_data1 = data1[indices]

            # Compute correlation for each feature
            if n_features == 1:
                corr = corr_func(perm_data1[:, 0], data2[:, 0])
            else:
                corr = np.array([
                    corr_func(perm_data1[:, i], data2[:, i])
                    for i in range(n_features)
                ])
            null_dist.append(corr)

        null_dist = np.array(null_dist)  # (n_permute, n_features)

        # Compute p-values
        p_values = _compute_pvalue(obs_corr, null_dist, tail=tail)

        # Return to original shape
        if single_feature:
            obs_corr = float(obs_corr[0])
            p_values = float(p_values[0])

        # Build result dict
        result = {
            "correlation": obs_corr,
            "p": p_values,
            "backend": backend.name,
        }

        if return_null:
            if single_feature:
                null_dist = null_dist.squeeze()
            result["null_dist"] = null_dist

        return result
    else:
        # PyTorch: GPU with automatic batching
        return _correlation_permutation_gpu_batched(
            data1, data2, n_permute, metric, tail, return_null,
            backend, max_gpu_memory_gb, rng, single_feature
        )
