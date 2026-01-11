"""
Utility functions for permutation testing.

This module contains shared helper functions used across different
permutation test implementations.
"""

import numpy as np
from typing import Optional
from .._random import generate_sign_flips as _generate_sign_flips_from_random
from .._validation import validate_tail_parameter


# ============================================================================
# Numerical Stability Constants
# ============================================================================

# Small constant added to denominators to prevent division by zero
# Value: 1e-10 is standard in scientific computing for float64 precision
# - Well above machine epsilon (2.22e-16 for float64)
# - Small enough not to affect correlation values
# - Also safe for float32 GPU operations (machine epsilon 1.19e-07)
# - Matches established practice in neuroimaging libraries
EPSILON = 1e-10


# Re-export from shared random utilities for backward compatibility
_generate_sign_flips = _generate_sign_flips_from_random


def _compute_pvalue(
    obs_stat: np.ndarray,
    null_dist: np.ndarray,
    tail: int = 2,
) -> np.ndarray:
    """
    Calculate p-values from observed statistic and null distribution.

    Computes the proportion of null distribution values as extreme or more
    extreme than the observed statistic, using the correction factor approach
    from nltools.stats._calc_pvalue.

    Args:
        obs_stat (np.ndarray): Observed statistic(s)
            - shape () for scalar
            - shape (n_features,) for multi-feature
        null_dist (np.ndarray): Null distribution from permutations
            - shape (n_permute,) for single feature
            - shape (n_permute, n_features) for multi-feature
        tail (int): Test type
            - 1: One-tailed test (obs > null)
            - 2: Two-tailed test (|obs| > |null|)

    Returns:
        np.ndarray: P-value(s) with same shape as obs_stat

    Examples:
        >>> obs_stat = np.array([2.5])
        >>> null_dist = np.random.randn(1000, 1)
        >>> p = _compute_pvalue(obs_stat, null_dist, tail=2)
        >>> 0 < p <= 1
        True

    Notes:
        - Uses correction factor: (count + 1) / (n_permute + 1)
        - This prevents p-value = 0 and accounts for observed value
        - Minimum p-value is 1/(n_permute + 1)
        - For two-tailed tests, uses absolute values
    """
    validate_tail_parameter(tail)

    # Ensure inputs are numpy arrays (handles Python float/int scalars)
    obs_stat = np.asarray(obs_stat)
    null_dist = np.asarray(null_dist)

    # Handle shape differences
    if null_dist.ndim == 1:
        null_dist = null_dist[:, np.newaxis]
    if obs_stat.ndim == 0:
        obs_stat = obs_stat.reshape(1)
    elif obs_stat.ndim == 1:
        obs_stat = obs_stat.reshape(1, -1)

    n_permute = null_dist.shape[0]
    denom = float(n_permute) + 1.0

    if tail == 1:
        # One-tailed: count how many null >= observed (if obs >= 0) or null <= observed (if obs < 0)
        # This matches nltools.stats._calc_pvalue behavior exactly
        # Apply correction: +1 to numerator
        # Use np.where to handle both positive and negative observed statistics
        mask_positive = obs_stat >= 0
        numer = np.where(
            mask_positive,
            np.sum(null_dist >= obs_stat, axis=0) + 1.0,
            np.sum(null_dist <= obs_stat, axis=0) + 1.0,
        )
    else:
        # Two-tailed: count how many |null| >= |observed|
        # Apply correction: +1 to numerator
        numer = np.sum(np.abs(null_dist) >= np.abs(obs_stat), axis=0) + 1.0

    p_values = numer / denom

    return p_values


# Re-export from shared random utilities for backward compatibility


def _auto_batch_size(
    n_permute: int,
    n_samples: int,
    n_features: int,
    max_memory_gb: float = 4.0,
) -> tuple[int, int]:
    """
    Automatically determine batch size to avoid GPU OOM.

    Calculates how many permutations can be processed simultaneously
    without exceeding the memory budget. The bottleneck is the
    data_perm tensor: (batch_size, n_samples, n_features).

    Args:
        n_permute (int): Total number of permutations to compute
        n_samples (int): Number of samples in dataset
        n_features (int): Number of features/voxels
        max_memory_gb (float): Maximum GPU memory to use in GB (default: 4.0)

    Returns:
        tuple[int, int]: (batch_size, n_batches)
            - batch_size: Number of permutations per batch
            - n_batches: Total number of batches needed

    Examples:
        >>> # Small problem: All permutations fit in one batch
        >>> batch_size, n_batches = _auto_batch_size(1000, 30, 1000, max_memory_gb=4.0)
        >>> n_batches
        1

        >>> # Large problem: Need multiple batches
        >>> batch_size, n_batches = _auto_batch_size(10000, 30, 50000, max_memory_gb=4.0)
        >>> n_batches > 1
        True

    Notes:
        - Uses float32 (4 bytes per element) for memory calculation
        - Minimum batch size is 100 permutations
        - Maximum batch size is n_permute (all at once)
        - Conservative estimates ensure we stay under budget
    """
    bytes_per_element = 4  # float32

    # Memory per permutation (bottleneck: data_perm tensor)
    # Shape: (1, n_samples, n_features)
    memory_per_perm = n_samples * n_features * bytes_per_element

    # How many permutations fit in memory budget?
    max_memory_bytes = max_memory_gb * 1e9
    batch_size = int(max_memory_bytes / memory_per_perm)

    # Clamp to reasonable range
    batch_size = max(100, min(batch_size, n_permute))  # At least 100, at most all

    # Calculate number of batches needed
    n_batches = int(np.ceil(n_permute / batch_size))

    return batch_size, n_batches


def _auto_n_jobs_cpu(
    data_size_mb: float,
    n_permute: int,
    max_memory_gb: float = 8.0,
    min_jobs: int = 1,
    max_jobs: Optional[int] = None,
) -> int:
    """
    Automatically determine optimal number of CPU workers to avoid memory exhaustion.

    Calculates how many parallel workers can safely process permutations given
    available memory. Each worker process needs to serialize (pickle) data,
    which typically requires 2-4× the original data size in memory.

    Args:
        data_size_mb (float): Size of data array in MB (float32: 4 bytes per element)
        n_permute (int): Number of permutations to compute
        max_memory_gb (float): Maximum memory budget in GB (default: 8.0)
            Conservative default leaves headroom for OS and other processes
        min_jobs (int): Minimum number of workers (default: 1)
        max_jobs (int, optional): Maximum number of workers (default: None = all cores)

    Returns:
        int: Optimal number of workers (n_jobs parameter for joblib.Parallel)

    Examples:
        >>> # Small data: Use all cores
        >>> n_jobs = _auto_n_jobs_cpu(1.0, 5000, max_memory_gb=8.0)
        >>> n_jobs >= 4  # Should use multiple cores
        True

        >>> # Large data: Limit workers
        >>> n_jobs = _auto_n_jobs_cpu(100.0, 5000, max_memory_gb=8.0)
        >>> n_jobs < 8  # Should limit workers
        True

    Notes:
        - Accounts for joblib serialization overhead (3× multiplier)
        - Leaves 50% headroom for OS and other processes
        - Minimum 1 worker, maximum all available cores (unless max_jobs specified)
        - Uses available RAM if max_memory_gb is None
    """
    import multiprocessing

    # Get system limits
    if max_jobs is None:
        max_jobs = multiprocessing.cpu_count()

    # Calculate memory budget (leave 50% headroom for OS and other processes)
    available_memory_gb = max_memory_gb
    if available_memory_gb is None:
        try:
            import psutil

            mem = psutil.virtual_memory()
            available_memory_gb = (mem.available / 1024**3) * 0.5  # 50% headroom
        except ImportError:
            # Fallback: assume 8 GB available
            available_memory_gb = 8.0

    available_memory_mb = available_memory_gb * 1024

    # Memory per worker: data serialization overhead (3× is conservative for pickle)
    # Plus small overhead for result arrays (n_permute results per worker)
    serialization_factor = 3.0
    result_overhead_mb = (n_permute * 4 / 1024**2) * 0.1  # ~10% overhead estimate
    memory_per_worker_mb = data_size_mb * serialization_factor + result_overhead_mb

    # How many workers can fit in memory budget?
    if memory_per_worker_mb <= 0:
        return min_jobs

    max_workers_by_memory = int(available_memory_mb / memory_per_worker_mb)
    max_workers_by_memory = max(min_jobs, min(max_workers_by_memory, max_jobs))

    # Use at least min_jobs, but don't exceed memory budget
    optimal_n_jobs = max(min_jobs, min(max_workers_by_memory, max_jobs))

    return optimal_n_jobs


def _verify_n_jobs_memory_constraint(
    requested_n_jobs: int,
    data_size_mb: float,
    n_permute: int,
    max_memory_gb: float = 8.0,
    min_jobs: int = 1,
    warn_threshold: float = 0.2,
) -> int:
    """
    Verify memory constraint for explicitly requested n_jobs.

    Ensures that requested number of workers doesn't exceed memory budget.
    If memory constraint is violated, reduces n_jobs and optionally warns.

    Args:
        requested_n_jobs (int): User-requested number of workers
        data_size_mb (float): Size of data per worker in MB
        n_permute (int): Number of tasks to process
        max_memory_gb (float): Maximum memory budget in GB (default: 8.0)
        min_jobs (int): Minimum number of workers (default: 1)
        warn_threshold (float): Warn if reduction exceeds this fraction (default: 0.2)

    Returns:
        int: Verified number of workers (may be reduced from requested)

    Examples:
        >>> # Memory allows requested workers
        >>> n_jobs = _verify_n_jobs_memory_constraint(
        ...     requested_n_jobs=4,
        ...     data_size_mb=1.0,
        ...     n_permute=1000,
        ...     max_memory_gb=8.0
        ... )
        >>> n_jobs
        4

        >>> # Memory constraint reduces workers
        >>> n_jobs = _verify_n_jobs_memory_constraint(
        ...     requested_n_jobs=8,
        ...     data_size_mb=2.0,  # Large data
        ...     n_permute=1000,
        ...     max_memory_gb=8.0
        ... )
        >>> n_jobs < 8  # Should be reduced
        True

    Notes:
        - Always respects memory constraints to prevent OOM
        - Warns if reduction is significant (>20% by default)
        - Never reduces below min_jobs
        - Uses same memory calculation as _auto_n_jobs_cpu for consistency
    """
    import multiprocessing
    import warnings

    # Get CPU count limit
    max_jobs_by_cpu = multiprocessing.cpu_count()

    # Calculate memory budget (same as _auto_n_jobs_cpu)
    available_memory_gb = max_memory_gb
    if available_memory_gb is None:
        try:
            import psutil

            mem = psutil.virtual_memory()
            available_memory_gb = (mem.available / 1024**3) * 0.5  # 50% headroom
        except ImportError:
            available_memory_gb = 8.0

    available_memory_mb = available_memory_gb * 1024

    # Calculate memory per worker (same as _auto_n_jobs_cpu)
    serialization_factor = 3.0
    result_overhead_mb = (n_permute * 4 / 1024**2) * 0.1
    memory_per_worker_mb = data_size_mb * serialization_factor + result_overhead_mb

    # Calculate maximum workers allowed by memory
    if memory_per_worker_mb <= 0:
        max_workers_by_memory = max_jobs_by_cpu
    else:
        max_workers_by_memory = int(available_memory_mb / memory_per_worker_mb)
        max_workers_by_memory = max(
            min_jobs, min(max_workers_by_memory, max_jobs_by_cpu)
        )

    # Determine final n_jobs
    # Priority: memory constraint > CPU limit > user request
    final_n_jobs = min(requested_n_jobs, max_workers_by_memory)
    final_n_jobs = max(min_jobs, final_n_jobs)  # Never below min_jobs

    # Warn if significant reduction occurred
    if final_n_jobs < requested_n_jobs:
        reduction_fraction = (requested_n_jobs - final_n_jobs) / requested_n_jobs
        if reduction_fraction >= warn_threshold:
            warnings.warn(
                f"Requested n_jobs={requested_n_jobs} exceeds memory limit "
                f"({max_memory_gb:.1f} GB). Reducing to {final_n_jobs} workers "
                f"to prevent out-of-memory errors. "
                f"Estimated memory per worker: {memory_per_worker_mb:.2f} MB.",
                UserWarning,
                stacklevel=3,
            )

    return final_n_jobs


def _estimate_data_size_mb(data: np.ndarray) -> float:
    """
    Estimate memory size of data array in MB.

    Accounts for numpy array overhead and dtype.

    Args:
        data (np.ndarray): Data array

    Returns:
        float: Estimated size in MB
    """
    if data.size == 0:
        return 0.0

    # Base size: elements × bytes per element
    bytes_per_element = data.dtype.itemsize
    base_size_bytes = data.size * bytes_per_element

    # Add numpy array overhead (typically ~100 bytes)
    overhead_bytes = 100

    total_size_mb = (base_size_bytes + overhead_bytes) / 1024**2

    return total_size_mb
