"""Correlation permutation test implementations.

This module provides CPU-parallel and GPU-batched implementations
of correlation permutation tests for assessing statistical significance
of correlations.
"""

import warnings
import numpy as np
from typing import TYPE_CHECKING
from sklearn.utils import check_random_state
from scipy.stats import rankdata, kendalltau

from nltools.algorithms.backends import Backend
from .utils import _compute_pvalue, _auto_batch_size, EPSILON

if TYPE_CHECKING:
    import torch


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray | float:
    """Compute Pearson correlation coefficient(s).

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
    random_state: int | None,
    single_feature: bool = False,
) -> dict:
    """Correlation permutation test using CPU parallelization with joblib.

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
    if metric == "pearson":
        corr_func = _pearson_correlation
    elif metric == "spearman":
        corr_func = _spearman_correlation
    elif metric == "kendall":
        corr_func = _kendall_correlation
    else:
        raise NotImplementedError(f"Metric '{metric}' not yet implemented")

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
        for i in tqdm(range(n_permute), desc="CPU parallel perms", unit="perm")
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
        "parallel": "cpu",
    }

    if return_null:
        if single_feature:
            null_dist = null_dist.squeeze()
        result["null_dist"] = null_dist

    return result


def _rank_transform_gpu(
    data: "torch.Tensor", dim: int = -1, method: str = "average"
) -> "torch.Tensor":
    """GPU-accelerated rank transformation using PyTorch.

    Computes ranks using the "average" method for tied ranks, matching
    scipy.stats.rankdata behavior. This is used for Spearman correlation.

    Args:
        data: Input tensor, shape (..., n_samples, ...)
        dim: Dimension along which to rank (default: last dimension)
        method: Ranking method (currently only "average" supported)

    Returns:
        Ranked tensor with same shape as input, dtype float32

    Notes:
        Uses torch.argsort twice to compute ranks:
        1. First argsort: get indices that sort the data
        2. Second argsort: get ranks (indices that sort the sorted indices)
        Then averages ranks for tied values by grouping consecutive duplicates.
    """
    import torch

    if method != "average":
        raise NotImplementedError(f"Rank method '{method}' not yet implemented on GPU")

    # Save original shape and device
    original_shape = data.shape
    device = data.device

    # Move ranking dimension to last position
    ndim = len(original_shape)
    if dim < 0:
        dim = ndim + dim
    dims = list(range(ndim))
    dims[dim], dims[-1] = dims[-1], dims[dim]
    data_reordered = data.permute(*dims)  # (..., n_samples)

    # Flatten all but last dimension
    n_samples = data_reordered.shape[-1]
    data_flat = data_reordered.reshape(-1, n_samples)  # (n_batches, n_samples)
    n_batches = data_flat.shape[0]

    # Compute ranks for each batch
    ranked_flat = torch.zeros_like(data_flat, dtype=torch.float32)
    for i in range(n_batches):
        batch_data = data_flat[i]  # (n_samples,)

        # Sort data to handle ties
        sorted_data, sorted_indices = torch.sort(batch_data, stable=True)

        # Initial ranks: 1, 2, 3, ..., n_samples
        ranks = torch.arange(1, n_samples + 1, dtype=torch.float32, device=device)

        # Handle tied ranks: average method
        # Find consecutive duplicates and assign average rank
        if n_samples > 1:
            # Compare adjacent sorted values
            diff = torch.diff(sorted_data)
            ties = torch.cat([torch.tensor([False], device=device), diff == 0])

            if ties.any():
                # Group consecutive ties - simpler approach
                # Find where ties start and end
                tie_groups = []
                i = 0
                while i < n_samples:
                    if ties[i]:
                        # Found start of tie group
                        start = i
                        # Find end of tie group
                        while i < n_samples and ties[i]:
                            i += 1
                        end = i + 1 if i < n_samples else n_samples
                        tie_groups.append((start, end))
                    else:
                        i += 1

                # Assign average rank to each tie group
                for start, end in tie_groups:
                    avg_rank = ranks[start:end].mean()
                    ranks[start:end] = avg_rank

        # Map ranks back to original order
        ranked_flat[i, sorted_indices] = ranks

    # Reshape back to reordered shape
    ranked_reordered = ranked_flat.reshape(*data_reordered.shape)

    # Restore original dimension order
    ranked = ranked_reordered.permute(*dims)

    return ranked


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
    """Correlation permutation test using GPU with automatic batching.

    Processes permutations in batches to avoid GPU OOM. Transfers data once
    and reuses across batches for efficiency.

    Args:
        data1 (np.ndarray): Data to permute, shape (n_samples, n_features)
        data2 (np.ndarray): Data to correlate with, shape (n_samples, n_features)
        n_permute (int): Number of permutations
        metric (str): Correlation metric ('pearson', 'spearman', or 'kendall')
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

    # Kendall is dispatched to the CPU parallel path upstream (see
    # correlation_permutation_test). Guard here so any direct internal caller
    # gets a clear error instead of hitting an incomplete GPU path.
    if metric not in ("pearson", "spearman"):
        raise NotImplementedError(
            f"_correlation_permutation_gpu_batched does not implement metric={metric!r}. "
            "Call correlation_permutation_test, which routes non-GPU metrics to CPU."
        )

    # Convert to float32 for GPU efficiency
    data1 = data1.astype(np.float32)
    data2 = data2.astype(np.float32)

    # Determine batch size based on memory budget
    # Memory bottleneck: permuted indices and correlation computation
    batch_size, n_batches = _auto_batch_size(
        n_permute, n_samples, n_features, max_memory_gb=max_gpu_memory_gb
    )

    # Transfer data to device once (for GPU-accelerated observed correlation)
    data1_device = backend.to_device(data1)
    data2_device = backend.to_device(data2)

    # Pre-rank data2 for Spearman (only needs to be done once)
    if metric == "spearman":
        data2_ranked_device = _rank_transform_gpu(data2_device, dim=0)
    else:
        data2_ranked_device = None

    # Compute observed correlation on GPU for efficiency
    if metric == "pearson":
        # Use GPU vectorized Pearson correlation
        data1_centered = data1_device - torch.mean(data1_device, dim=0, keepdim=True)
        data2_centered = data2_device - torch.mean(data2_device, dim=0, keepdim=True)
        numerator = torch.sum(data1_centered * data2_centered, dim=0) / n_samples
        denominator = torch.std(data1_device, dim=0, unbiased=False) * torch.std(
            data2_device, dim=0, unbiased=False
        )
        obs_corr = backend.to_numpy(numerator / (denominator + EPSILON))
    elif metric == "spearman":
        # Use GPU rank transformation + Pearson correlation
        data1_ranked = _rank_transform_gpu(data1_device, dim=0)
        data2_ranked = _rank_transform_gpu(data2_device, dim=0)
        # Pearson correlation on ranks
        data1_ranked_centered = data1_ranked - torch.mean(
            data1_ranked, dim=0, keepdim=True
        )
        data2_ranked_centered = data2_ranked - torch.mean(
            data2_ranked, dim=0, keepdim=True
        )
        numerator = (
            torch.sum(data1_ranked_centered * data2_ranked_centered, dim=0) / n_samples
        )
        denominator = torch.std(data1_ranked, dim=0, unbiased=False) * torch.std(
            data2_ranked, dim=0, unbiased=False
        )
        obs_corr = backend.to_numpy(numerator / (denominator + EPSILON))

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

        # Pre-generate seeds for this batch (memory-efficient, deterministic)
        # Matches CPU-parallel pattern: independent RandomState per permutation
        MAX_INT = 2**31 - 1
        batch_seeds = random_state.randint(MAX_INT, size=current_batch_size)

        # Generate permutation indices using independent RNG per permutation
        batch_indices = np.array(
            [
                np.random.RandomState(batch_seeds[i]).permutation(n_samples)
                for i in range(current_batch_size)
            ]
        )

        # Transfer to device
        batch_indices_device = backend.to_device(batch_indices)
        if backend.name.startswith("torch"):
            batch_indices_device = batch_indices_device.long()

        # Vectorized correlation computation for all features simultaneously
        if metric == "pearson":
            # Use advanced indexing to permute all features at once
            # batch_indices_device: (current_batch_size, n_samples)
            # data1_device: (n_samples, n_features)
            # Use advanced indexing: perm_data1[b, s, f] = data1[batch_indices[b, s], f]
            # Result shape: (current_batch_size, n_samples, n_features)
            # Create batch indices for advanced indexing
            batch_idx = torch.arange(
                current_batch_size, device=batch_indices_device.device
            )[:, None]  # (current_batch_size, 1)
            perm_data1 = data1_device[
                batch_indices_device
            ]  # (current_batch_size, n_samples, n_features)

            # Center data for all features simultaneously
            # perm_data1: (current_batch_size, n_samples, n_features)
            perm_data1_centered = perm_data1 - torch.mean(
                perm_data1, dim=1, keepdim=True
            )  # (current_batch_size, n_samples, n_features)
            data2_centered = data2_device - torch.mean(
                data2_device, dim=0, keepdim=True
            )  # (n_samples, n_features)

            # Compute correlations for all features at once using einsum or bmm
            # numerator: sum over samples dimension
            # (current_batch_size, n_samples, n_features) @ (n_samples, n_features)
            # Use einsum: 'bsf,sf->bf' (batch, sample, feature)
            numerator = (
                torch.einsum("bsf,sf->bf", perm_data1_centered, data2_centered)
                / n_samples
            )

            # Compute denominators for all features
            std_perm = torch.std(
                perm_data1_centered, dim=1, unbiased=False
            )  # (current_batch_size, n_features)
            std_data2 = torch.std(data2_device, dim=0, unbiased=False)  # (n_features,)

            # Broadcasting: (current_batch_size, n_features) * (1, n_features)
            denominator = std_perm * std_data2.unsqueeze(0)

            # Handle division by zero using EPSILON
            batch_corrs = numerator / (
                denominator + EPSILON
            )  # (current_batch_size, n_features)
            batch_corrs = backend.to_numpy(batch_corrs)
        elif metric == "spearman":
            # Vectorized Spearman correlation using GPU rank transformation
            # Permute data1 for all features simultaneously using advanced indexing
            perm_data1 = data1_device[
                batch_indices_device
            ]  # (current_batch_size, n_samples, n_features)

            # Rank-transform permuted data1 and use pre-ranked data2
            # Rank along sample dimension (dim=1 for perm_data1)
            perm_data1_ranked = _rank_transform_gpu(perm_data1, dim=1)
            # data2_ranked was pre-computed once, expand to match batch dimension
            data2_ranked_batch = data2_ranked_device.unsqueeze(0).expand(
                current_batch_size, -1, -1
            )  # (current_batch_size, n_samples, n_features)

            # Center ranked data
            perm_data1_ranked_centered = perm_data1_ranked - torch.mean(
                perm_data1_ranked, dim=1, keepdim=True
            )  # (current_batch_size, n_samples, n_features)
            data2_ranked_centered = data2_ranked_batch - torch.mean(
                data2_ranked_batch, dim=1, keepdim=True
            )  # (current_batch_size, n_samples, n_features)

            # Compute Spearman correlations (Pearson on ranks) for all features
            numerator = (
                torch.einsum(
                    "bsf,bsf->bf", perm_data1_ranked_centered, data2_ranked_centered
                )
                / n_samples
            )

            # Compute denominators
            std_perm = torch.std(
                perm_data1_ranked, dim=1, unbiased=False
            )  # (current_batch_size, n_features)
            std_data2 = torch.std(
                data2_ranked_batch, dim=1, unbiased=False
            )  # (current_batch_size, n_features)

            # Broadcasting
            denominator = std_perm * std_data2

            # Handle division by zero using EPSILON
            batch_corrs = numerator / (
                denominator + EPSILON
            )  # (current_batch_size, n_features)
            batch_corrs = backend.to_numpy(batch_corrs)
        else:
            # Kendall is routed to the CPU parallel path by correlation_permutation_test
            # before ever reaching this function. Keep this as a defensive guard so
            # future direct callers get a clear error instead of a silent slow path.
            raise NotImplementedError(
                f"_correlation_permutation_gpu_batched does not implement metric={metric!r}. "
                "Call correlation_permutation_test, which routes non-GPU metrics to CPU."
            )

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
        obs_corr = obs_corr.item() if hasattr(obs_corr, "item") else float(obs_corr[0])
        p_values = p_values.item() if hasattr(p_values, "item") else float(p_values[0])

    # Build result
    result = {
        "correlation": obs_corr,
        "p": p_values,
        "parallel": "gpu",
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
    metric: str = "pearson",
    tail: int | str = 2,
    return_null: bool = False,
    parallel: str | None = "cpu",
    n_jobs: int = -1,
    max_gpu_memory_gb: float = 4.0,
    random_state: int | None = None,
) -> dict:
    """Correlation permutation test.

    Tests whether the correlation between data1 and data2 is significantly
    different from zero by randomly permuting data1 and computing correlations.

    Assumption: Observations are independent (i.i.d.). For autocorrelated time
    series, use timeseries_correlation_permutation_test with circle_shift or
    phase_randomize methods instead.

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
        tail (int | str): Test type (default: 2)
            - 'two' or 2: Two-tailed test (r != 0)
            - 'upper' or 1: One-tailed upper (r > 0, positive correlation)
            - 'lower' or -1: One-tailed lower (r < 0, negative correlation)
            For MCP correction (FDR), use 'upper' or 'lower' for consistent direction.
        return_null (bool): If True, return full null distribution (default: False)
        parallel (str, optional): Parallelization method (default: 'cpu')
            - None: Single-threaded NumPy (for debugging/small problems)
            - 'cpu': CPU parallelization via joblib (default, 4-8× speedup)
            - 'gpu': GPU acceleration via PyTorch (fastest for large problems)
        n_jobs (int): Number of CPU cores for parallelization (default: -1 = all cores)
            Only used when parallel='cpu'
        max_gpu_memory_gb (float): Maximum GPU memory to use in GB (default: 4.0)
            Controls automatic batching to prevent OOM errors. Only used with
            parallel='gpu'. Larger values allow more permutations per batch but
            risk OOM on smaller GPUs.
        random_state (int, optional): Random seed for reproducibility

    Returns:
        dict: Dictionary with keys:
            - 'correlation' (float or np.ndarray): Observed correlation(s)
            - 'p' (float or np.ndarray): P-value(s)
            - 'null_dist' (np.ndarray): Null distribution (if return_null=True)
            - 'parallel' (str): Parallelization method used

    Examples:
        >>> # Single feature (default CPU parallelization)
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
        >>> result = correlation_permutation_test(data1, data2, n_permute=5000, parallel='gpu')

    Notes:
        - Default (parallel='cpu'): CPU parallelization with joblib (4-8× speedup)
        - GPU parallelization ('gpu'): Fastest for large problems with automatic batching
            - Pearson correlation: Fully vectorized across all features (5-20× speedup for multi-feature)
            - Spearman/Kendall: Only supported with parallel='cpu' or parallel=None (GPU not yet implemented)
        - Single-threaded (parallel=None): Use for small problems or debugging
        - For multi-feature data, each feature pair tested independently
        - Kendall is O(n^2) complexity, slower than Pearson/Spearman for large samples
    """
    # Validate parallel parameter
    if parallel not in [None, "cpu", "gpu"]:
        raise ValueError(f"parallel must be None, 'cpu', or 'gpu', got {parallel!r}")

    # Input validation
    data1 = np.asarray(data1, dtype=np.float64)
    data2 = np.asarray(data2, dtype=np.float64)

    if data1.ndim not in [1, 2]:
        raise ValueError(f"data1 must be 1D or 2D, got shape {data1.shape}")
    if data2.ndim not in [1, 2]:
        raise ValueError(f"data2 must be 1D or 2D, got shape {data2.shape}")
    if tail not in [1, 2]:
        raise ValueError(f"tail must be 1 or 2, got {tail}")
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

    n_samples, n_features = data1.shape

    # GPU path supports Pearson and Spearman only. For Kendall, fall through
    # to the CPU-parallel path so user code with parallel='gpu' still works.
    # True GPU Kendall tau-b kernel tracked in EJO-453.
    if parallel == "gpu" and metric == "kendall":
        warnings.warn(
            "Kendall correlation is not implemented on GPU; falling back to "
            "parallel='cpu'. Use parallel='cpu' explicitly to silence this warning.",
            UserWarning,
            stacklevel=2,
        )
        parallel = "cpu"

    # Decide execution mode based on parallel parameter
    if parallel == "cpu" or parallel is None:
        # CPU modes
        if parallel is None:
            # Single-threaded NumPy
            rng = check_random_state(random_state)

            # Select correlation function
            if metric == "pearson":
                corr_func = _pearson_correlation
            elif metric == "spearman":
                corr_func = _spearman_correlation
            elif metric == "kendall":
                corr_func = _kendall_correlation
            else:
                raise NotImplementedError(f"Metric '{metric}' not yet implemented")

            # Compute observed correlation
            if n_features == 1:
                obs_corr = corr_func(data1[:, 0], data2[:, 0])
                obs_corr = np.array([obs_corr])
            else:
                obs_corr = np.array(
                    [corr_func(data1[:, i], data2[:, i]) for i in range(n_features)]
                )

            # Pre-generate seeds for deterministic permutations
            MAX_INT = 2**31 - 1
            seeds = rng.randint(MAX_INT, size=n_permute)

            # Generate null distribution
            null_dist = []
            for i in range(n_permute):
                perm_rng = np.random.RandomState(seeds[i])
                indices = perm_rng.permutation(n_samples)
                perm_data1 = data1[indices]

                # Compute correlation for each feature
                if n_features == 1:
                    corr = corr_func(perm_data1[:, 0], data2[:, 0])
                else:
                    corr = np.array(
                        [
                            corr_func(perm_data1[:, i], data2[:, i])
                            for i in range(n_features)
                        ]
                    )
                null_dist.append(corr)

            null_dist = np.array(null_dist)
            p_values = _compute_pvalue(obs_corr, null_dist, tail=tail)

            if single_feature:
                obs_corr = (
                    obs_corr.item() if hasattr(obs_corr, "item") else float(obs_corr[0])
                )
                p_values = (
                    p_values.item() if hasattr(p_values, "item") else float(p_values[0])
                )

            result = {
                "correlation": obs_corr,
                "p": p_values,
                "parallel": None,
            }

            if return_null:
                if single_feature:
                    null_dist = null_dist.squeeze()
                result["null_dist"] = null_dist

            return result
        # CPU parallelization mode
        return _correlation_permutation_cpu_parallel(
            data1,
            data2,
            n_permute,
            metric,
            tail,
            return_null,
            n_jobs,
            random_state,
            single_feature,
        )
    # GPU mode
    backend_obj = Backend("torch")
    rng = check_random_state(random_state)
    return _correlation_permutation_gpu_batched(
        data1,
        data2,
        n_permute,
        metric,
        tail,
        return_null,
        backend_obj,
        max_gpu_memory_gb,
        rng,
        single_feature,
    )
