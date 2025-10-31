"""
One-sample permutation test implementations.

This module provides CPU-parallel and GPU-batched implementations
of the one-sample permutation test (sign-flipping test).
"""

import numpy as np
from typing import Union, Optional
from sklearn.utils import check_random_state

from nltools.backends import Backend, auto_select_backend
from .utils import _generate_sign_flips, _compute_pvalue, _auto_batch_size


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

    Pre-generates all sign-flips deterministically (matching stats.py pattern),
    then parallelizes only the computation. This ensures perfect reproducibility
    and backward compatibility with nltools.stats.one_sample_permutation.

    Args:
        data (np.ndarray): Data to test, shape (n_samples, n_features)
        n_permute (int): Number of permutations
        tail (int): Test type (1 or 2)
        return_null (bool): Whether to return null distribution
        n_jobs (int): Number of parallel jobs (-1 = all cores)
        random_state (int, optional): Random seed for reproducibility
        single_feature (bool): Whether data is single feature

    Returns:
        dict: Same format as main function, with 'backend' indicating CPU parallel

    Notes:
        - Pre-generates sign-flip matrix (matches stats.py for exact p-values)
        - Memory usage: n_permute × n_samples × 1 byte (negligible)
        - Parallelizes computation, not RNG (ensures determinism)
        - Progress bar shows permutation completion
        - Typical speedup: 4-8× on 8-core machines
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm

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
        for i in tqdm(range(n_permute), desc="CPU parallel perms", unit="perm")
    )
    null_dist = np.array(null_dist)  # Shape: (n_permute, n_features)

    # Compute p-values
    p_values = _compute_pvalue(obs_stat, null_dist, tail=tail)

    # Return to original shape
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


def _one_sample_permutation_gpu_batched(
    data: np.ndarray,
    n_permute: int,
    tail: int,
    return_null: bool,
    backend: Backend,
    max_gpu_memory_gb: float,
    random_state,
    single_feature: bool = False,
) -> dict:
    """
    One-sample permutation test using GPU with automatic batching.

    Processes permutations in batches to avoid GPU OOM. Transfers data once
    and reuses across batches for efficiency.

    Args:
        data (np.ndarray): Data to test, shape (n_samples, n_features)
        n_permute (int): Number of permutations
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

    n_samples, n_features = data.shape

    # Convert to float32 for GPU efficiency
    data = data.astype(np.float32)

    # Compute observed statistic
    obs_stat = np.mean(data, axis=0)

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
        desc="GPU permutation batches",
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
            current_batch_size, n_samples, random_state=random_state
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

    # Return to original shape
    if single_feature:
        obs_stat = float(obs_stat[0])
        p_values = float(p_values[0])

    # Build result
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

    Assumption: Symmetric error distribution around zero. For highly skewed
    distributions, consider alternative methods (e.g., bootstrap resampling).

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
                backend = auto_select_backend(
                    n_samples, n_features, cv=n_permute // 1000
                )
            else:
                backend = Backend(backend)

    # Setup random state
    rng = check_random_state(random_state)

    # Compute null distribution based on backend
    if backend.name == "numpy":
        # NumPy: Sequential processing (simple, memory-efficient)

        # Compute observed statistic
        obs_stat = np.mean(data, axis=0)

        # Generate sign-flip matrix: (n_permute, n_samples)
        sign_flips = _generate_sign_flips(n_permute, n_samples, random_state=rng)

        # Broadcasting: (n_permute, n_samples, 1) * (1, n_samples, n_features)
        data_perm = sign_flips[:, :, None] * data[None, :, :]
        null_dist = np.mean(data_perm, axis=1)  # (n_permute, n_features)

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
    else:
        # PyTorch: GPU with automatic batching
        return _one_sample_permutation_gpu_batched(
            data,
            n_permute,
            tail,
            return_null,
            backend,
            max_gpu_memory_gb,
            rng,
            single_feature,
        )
