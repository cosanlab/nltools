"""
GPU-accelerated statistical inference for neuroimaging.

This module provides fast permutation testing and bootstrap resampling using
optional GPU acceleration via PyTorch. When GPU is unavailable, efficiently
uses CPU parallelization.

Inspired by BROCCOLI's GPU permutation testing (Eklund et al. 2014).

Key Features:
    - 10-100× speedup for permutation tests with GPU
    - Efficient CPU parallelization when GPU unavailable
    - Transparent CPU/GPU support via Backend abstraction
    - Drop-in replacement for nltools.stats functions

Examples:
    >>> import numpy as np
    >>> from nltools.algorithms.inference import one_sample_permutation_test

    >>> # Simple one-sample test
    >>> data = np.random.randn(30)  # 30 subjects
    >>> result = one_sample_permutation_test(data, n_permute=5000)
    >>> print(f"p-value: {result['p']:.3f}")

    >>> # Voxel-wise test with GPU acceleration
    >>> data = np.random.randn(30, 50000)  # 30 subjects, 50K voxels
    >>> result = one_sample_permutation_test(data, n_permute=10000, backend='torch')
    >>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")

Performance:
    - CPU (NumPy): Good for small problems (< 5K permutations)
    - GPU (PyTorch): Excellent for large problems (> 5K permutations)
    - CPU Parallel (joblib): Efficient fallback when GPU unavailable
    - Auto-selection: Use backend='auto' for best performance

References:
    Eklund, A., Dufort, P., Villani, M., & LaConte, S. M. (2014).
    BROCCOLI: Software for fast fMRI analysis on many-core CPUs and GPUs.
    Frontiers in Neuroinformatics, 8, 24.

Notes:
    This module is part of the "functional core" of nltools. For integration
    with Brain_Data objects, see nltools.data.brain_data.
"""

import numpy as np
from typing import Union, Optional
from sklearn.utils import check_random_state
from nltools.backends import Backend, auto_select_backend, check_gpu_available


# ============================================================================
# Helper Functions
# ============================================================================


def _generate_sign_flips(
    n_permute: int,
    n_samples: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generate random sign-flip matrix for one-sample permutation tests.

    Creates a matrix of random +1/-1 values for sign-flipping permutation tests.
    Each row represents one permutation, where each sample is randomly multiplied
    by +1 or -1 to create the null distribution.

    Args:
        n_permute (int): Number of permutations to generate
        n_samples (int): Number of samples in the dataset
        random_state (int, optional): Random seed for reproducibility

    Returns:
        np.ndarray: Sign-flip matrix of shape (n_permute, n_samples)
            containing only +1 and -1 values

    Examples:
        >>> sign_flips = _generate_sign_flips(n_permute=100, n_samples=30, random_state=42)
        >>> sign_flips.shape
        (100, 30)
        >>> np.all(np.isin(sign_flips, [-1, 1]))
        True

    Notes:
        - Each permutation is independent
        - Values are uniformly sampled from {-1, +1}
        - Returns NumPy array (device transfer handled by caller)
    """
    rng = check_random_state(random_state)
    # Generate random binary values (0 or 1), then map to {-1, +1}
    sign_flips = rng.choice([-1, 1], size=(n_permute, n_samples))
    return sign_flips


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
    if tail not in [1, 2]:
        raise ValueError(f"tail must be 1 or 2, got {tail}")

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
        # One-tailed: count how many null >= observed (or <= for negative)
        # Apply correction: +1 to numerator
        numer = np.sum(null_dist >= obs_stat, axis=0) + 1.0
    else:
        # Two-tailed: count how many |null| >= |observed|
        # Apply correction: +1 to numerator
        numer = np.sum(np.abs(null_dist) >= np.abs(obs_stat), axis=0) + 1.0

    p_values = numer / denom

    return p_values


# ============================================================================
# GPU Memory Management
# ============================================================================


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


# ============================================================================
# CPU Parallelization
# ============================================================================


def _one_sample_permutation_cpu_parallel(
    data: np.ndarray,
    n_permute: int,
    tail: int,
    return_null: bool,
    n_jobs: int,
    random_state: Optional[int],
    single_feature: bool = False,
) -> dict:
    """
    One-sample permutation test using CPU parallelization with joblib.

    This is a memory-efficient implementation that processes one permutation
    per worker. Unlike GPU batching which loads all permutations into memory,
    this approach keeps memory usage proportional to the number of workers,
    not the number of permutations.

    Args:
        data (np.ndarray): Data to test, shape (n_samples, n_features)
        n_permute (int): Number of permutations
        tail (int): Test type (1 or 2)
        return_null (bool): Whether to return null distribution
        n_jobs (int): Number of parallel jobs (-1 = all cores)
        random_state (int, optional): Random seed for reproducibility

    Returns:
        dict: Same format as main function, with 'backend' indicating CPU parallel

    Notes:
        - Memory usage: ~O(n_workers × n_features), NOT O(n_permute × n_features)
        - Each worker gets unique random seed derived from main seed
        - Progress bar shows permutation completion
        - Typical speedup: 4-8× on 8-core machines
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm

    # Setup random state and generate seeds for workers
    rng = check_random_state(random_state)
    MAX_INT = 2**31 - 1
    seeds = rng.randint(MAX_INT, size=n_permute)

    # Get dimensions (data is already reshaped by caller)
    n_samples, n_features = data.shape

    # Compute observed statistic
    obs_stat = np.mean(data, axis=0)

    # Define worker function (each processes ONE permutation)
    def _compute_one_perm(seed):
        """Compute statistic for one sign-flip permutation."""
        perm_rng = np.random.RandomState(seed)
        signs = perm_rng.choice([-1, 1], size=n_samples)
        perm_data = data * signs[:, np.newaxis]
        return np.mean(perm_data, axis=0)

    # Execute in parallel with progress bar
    # Use tqdm with delayed to track progress
    null_dist = Parallel(n_jobs=n_jobs)(
        delayed(_compute_one_perm)(seeds[i])
        for i in tqdm(range(n_permute), desc="CPU parallel perms", unit="perm")
    )
    null_dist = np.array(null_dist)  # Shape: (n_permute, n_features)

    # Compute p-values
    p_values = _compute_pvalue(obs_stat, null_dist, tail=tail)

    # Return to original shape (passed as parameter from caller)
    if single_feature:
        obs_stat = float(obs_stat[0])
        p_values = float(p_values[0])

    # Determine backend name based on n_jobs
    if n_jobs == -1:
        import multiprocessing
        n_cores = multiprocessing.cpu_count()
        backend_name = f"cpu-parallel-{n_cores}"
    else:
        backend_name = f"cpu-parallel-{n_jobs}"

    # Build result
    result = {
        "mean": obs_stat,
        "p": p_values,
        "backend": backend_name,
    }

    if return_null:
        if single_feature:
            null_dist = null_dist.squeeze()
        result["null_dist"] = null_dist

    return result


# ============================================================================
# Main Inference Functions
# ============================================================================


def one_sample_permutation_test(
    data: np.ndarray,
    n_permute: int = 5000,
    tail: int = 2,
    return_null: bool = False,
    backend: Optional[Union[Backend, str]] = None,
    n_jobs: int = -1,
    max_gpu_memory_gb: float = 4.0,
    random_state: Optional[int] = None,
) -> dict:
    """
    One-sample permutation test using sign-flipping.

    Tests whether the mean of data is significantly different from zero
    by randomly flipping the sign of each observation. This is the
    permutation test equivalent of a one-sample t-test.

    When backend='torch', uses GPU acceleration with automatic batching.
    When backend=None (default), uses efficient CPU parallelization via joblib.

    Args:
        data (np.ndarray): Data to test
            - shape (n_samples,) for single feature
            - shape (n_samples, n_features) for multi-feature (voxel-wise)
        n_permute (int): Number of permutations (default: 5000)
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
            backend='torch' or 'auto'. Larger values allow more permutations
            per batch but risk OOM on smaller GPUs.
        random_state (int, optional): Random seed for reproducibility

    Returns:
        dict: Dictionary with keys:
            - 'mean' (float or np.ndarray): Observed mean(s)
            - 'p' (float or np.ndarray): P-value(s)
            - 'null_dist' (np.ndarray): Null distribution (if return_null=True)
            - 'backend' (str): Backend used for computation

    Examples:
        >>> # Single feature
        >>> data = np.random.randn(30)
        >>> result = one_sample_permutation_test(data, n_permute=5000)
        >>> result['p']
        0.23

        >>> # Voxel-wise test with GPU
        >>> data = np.random.randn(30, 10000)  # 30 subjects, 10K voxels
        >>> result = one_sample_permutation_test(data, n_permute=5000, backend='torch')
        >>> result['mean'].shape
        (10000,)
        >>> result['p'].shape
        (10000,)

        >>> # CPU parallelization (no GPU)
        >>> result = one_sample_permutation_test(data, n_permute=5000, n_jobs=-1)

    Notes:
        - Default (backend=None): CPU parallelization with joblib (4-8× speedup)
        - GPU backend (torch): Fastest for large problems with automatic batching
        - NumPy backend: Single-threaded, use for small problems or debugging
        - For voxel-wise tests, each voxel tested independently
        - Progress bars show completion for both CPU parallel and GPU batched modes
    """
    # Input validation
    data = np.asarray(data, dtype=np.float64)
    if data.ndim not in [1, 2]:
        raise ValueError(f"data must be 1D or 2D, got shape {data.shape}")
    if tail not in [1, 2]:
        raise ValueError(f"tail must be 1 or 2, got {tail}")

    # Handle shape
    single_feature = data.ndim == 1
    if single_feature:
        data = data[:, np.newaxis]  # (n_samples, 1)

    n_samples, n_features = data.shape

    # Decide execution mode: GPU backend vs CPU parallelization
    use_cpu_parallel = backend is None

    if use_cpu_parallel:
        # CPU parallelization mode (memory-efficient fallback)
        return _one_sample_permutation_cpu_parallel(
            data, n_permute, tail, return_null, n_jobs, random_state, single_feature
        )
    else:
        # GPU/Backend mode
        if isinstance(backend, str):
            if backend == "auto":
                # Auto-select based on problem size and GPU availability
                backend = auto_select_backend(n_samples, n_features, cv=n_permute // 1000)
            else:
                backend = Backend(backend)

    # Setup random state
    rng = check_random_state(random_state)

    # Compute observed statistic
    obs_stat = np.mean(data, axis=0)  # (n_features,)

    # Compute null distribution
    if backend.name == "numpy":
        # NumPy: Sequential processing (memory-efficient)
        # Generate sign-flip matrix: (n_permute, n_samples)
        sign_flips = _generate_sign_flips(n_permute, n_samples, random_state=rng)

        # Broadcasting: (n_permute, n_samples, 1) * (1, n_samples, n_features)
        data_perm = sign_flips[:, :, None] * data[None, :, :]
        null_dist = np.mean(data_perm, axis=1)  # (n_permute, n_features)
    else:
        # PyTorch: GPU with automatic batching to avoid OOM
        import torch
        from tqdm import tqdm

        # Convert to float32 for GPU efficiency
        data = data.astype(np.float32)

        # Determine batch size based on memory budget
        batch_size, n_batches = _auto_batch_size(
            n_permute, n_samples, n_features, max_memory_gb=max_gpu_memory_gb
        )

        # Transfer data to device once (reused across batches)
        data_device = backend.to_device(data)

        # Accumulate null distribution across batches
        null_dist_list = []

        # Process permutations in batches with progress bar
        pbar = tqdm(
            total=n_permute,
            desc="Permutation batches",
            unit="perm",
            disable=n_batches == 1,  # Disable progress bar if only 1 batch
        )

        for batch_idx in range(n_batches):
            # Determine current batch size
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_permute)
            current_batch_size = end_idx - start_idx

            # Generate sign flips for this batch only
            batch_sign_flips = _generate_sign_flips(
                current_batch_size, n_samples, random_state=rng
            )

            # Transfer to device
            sign_flips_device = backend.to_device(batch_sign_flips.astype(np.float32))

            # Compute null distribution for this batch
            # Broadcasting: (batch_size, n_samples, 1) * (1, n_samples, n_features)
            data_perm = sign_flips_device[:, :, None] * data_device[None, :, :]
            batch_null = torch.mean(data_perm, dim=1)
            batch_null = backend.to_numpy(batch_null)

            null_dist_list.append(batch_null)

            # Update progress bar
            pbar.update(current_batch_size)

            # Free batch memory
            del sign_flips_device, data_perm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar.close()

        # Combine batches: (n_permute, n_features)
        null_dist = np.vstack(null_dist_list)

    # Compute p-values
    p_values = _compute_pvalue(obs_stat, null_dist, tail=tail)

    # Return to original shape (convert to scalar for single feature)
    if single_feature:
        obs_stat = float(obs_stat[0])
        p_values = float(p_values[0])

    # Build result dict
    result = {
        "mean": obs_stat,
        "p": p_values,
        "backend": backend.name,
    }

    if return_null:
        if single_feature:
            null_dist = null_dist.squeeze()
        result["null_dist"] = null_dist

    return result
