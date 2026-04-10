"""
One-sample permutation test implementations.

This module provides CPU-parallel and GPU-batched implementations
of the one-sample permutation test (sign-flipping test).
"""

import numpy as np
from typing import Optional
from sklearn.utils import check_random_state

from nltools.algorithms.backends import Backend
from .utils import _generate_sign_flips, _compute_pvalue, _auto_batch_size
from .validation import (
    validate_tail_parameter,
    validate_parallel_parameter,
    validate_array_shape_range,
)


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

    # Build result
    result = {
        "mean": obs_stat,
        "p": p_values,
        "parallel": "cpu",
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
        "parallel": "gpu",
    }

    if return_null:
        if single_feature:
            null_dist = null_dist.squeeze()
        result["null_dist"] = null_dist

    return result


def one_sample_permutation_test(
    data: np.ndarray,
    n_permute: int = 5000,
    tail: int | str = 2,
    return_null: bool = False,
    parallel: Optional[str] = "cpu",
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

    Args:
        data (np.ndarray): Data to test
            - shape (n_samples,) for single feature
            - shape (n_samples, n_features) for multi-feature (voxel-wise)
        n_permute (int): Number of permutations (default: 5000)
        tail (int | str): Test type (default: 2)
            - 'two' or 2: Two-tailed test (mean != 0)
            - 'upper' or 1: One-tailed upper (mean > 0)
            - 'lower' or -1: One-tailed lower (mean < 0)
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
            - 'mean' (float or np.ndarray): Observed mean(s)
            - 'p' (float or np.ndarray): P-value(s)
            - 'null_dist' (np.ndarray): Null distribution (if return_null=True)
            - 'parallel' (str): Parallelization method used

    Examples:
        >>> # Single feature (default CPU parallelization)
        >>> data = np.random.randn(30)
        >>> result = one_sample_permutation_test(data, n_permute=5000)
        >>> result['p']
        0.23

        >>> # Voxel-wise test with GPU
        >>> data = np.random.randn(30, 10000)  # 30 subjects, 10K voxels
        >>> result = one_sample_permutation_test(data, n_permute=5000, parallel='gpu')
        >>> result['mean'].shape
        (10000,)
        >>> result['p'].shape
        (10000,)

        >>> # Single-threaded (for debugging)
        >>> result = one_sample_permutation_test(data, n_permute=5000, parallel=None)

    Notes:
        - Default (parallel='cpu'): CPU parallelization with joblib (4-8× speedup)
        - GPU parallelization ('gpu'): Fastest for large problems with automatic batching
        - Single-threaded (parallel=None): Use for small problems or debugging
        - For voxel-wise tests, each voxel tested independently
        - Progress bars show completion for both CPU parallel and GPU batched modes
    """
    # Input validation
    data = np.asarray(data, dtype=np.float64)
    validate_array_shape_range(data, 1, 2, name="data")
    validate_tail_parameter(tail)
    validate_parallel_parameter(parallel)

    # Handle shape
    single_feature = data.ndim == 1
    if single_feature:
        data = data[:, np.newaxis]  # (n_samples, 1)

    n_samples, n_features = data.shape

    # Decide execution mode based on parallel parameter
    if parallel == "cpu" or parallel is None:
        # CPU modes
        if parallel is None:
            # Single-threaded NumPy
            rng = check_random_state(random_state)
            obs_stat = np.mean(data, axis=0)
            sign_flips = _generate_sign_flips(n_permute, n_samples, random_state=rng)
            data_perm = sign_flips[:, :, None] * data[None, :, :]
            null_dist = np.mean(data_perm, axis=1)
            p_values = _compute_pvalue(obs_stat, null_dist, tail=tail)

            if single_feature:
                obs_stat = float(obs_stat[0])
                p_values = float(p_values[0])

            result = {
                "mean": obs_stat,
                "p": p_values,
                "parallel": None,
            }

            if return_null:
                if single_feature:
                    null_dist = null_dist.squeeze()
                result["null_dist"] = null_dist

            return result
        else:
            # CPU parallelization mode
            return _one_sample_permutation_cpu_parallel(
                data, n_permute, tail, return_null, n_jobs, random_state, single_feature
            )
    else:
        # GPU mode
        backend_obj = Backend("torch")
        rng = check_random_state(random_state)
        return _one_sample_permutation_gpu_batched(
            data,
            n_permute,
            tail,
            return_null,
            backend_obj,
            max_gpu_memory_gb,
            rng,
            single_feature,
        )
