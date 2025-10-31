"""
Two-sample permutation test implementations.

This module provides CPU-parallel and GPU-batched implementations
of the two-sample permutation test (group permutation test).
"""

import numpy as np
from typing import Union, Optional
from sklearn.utils import check_random_state

from nltools.backends import Backend, auto_select_backend
from .utils import _compute_pvalue, _auto_batch_size


def _two_sample_permutation_cpu_parallel(
    data1: np.ndarray,
    data2: np.ndarray,
    n_permute: int,
    tail: int,
    return_null: bool,
    n_jobs: int,
    random_state: Optional[int],
    single_feature: bool = False,
) -> dict:
    """
    Two-sample permutation test using CPU parallelization with joblib.

    Memory-efficient implementation that processes one permutation per worker.
    Randomly shuffles group labels and computes mean difference.

    Args:
        data1 (np.ndarray): Group 1 data, shape (n_samples1, n_features)
        data2 (np.ndarray): Group 2 data, shape (n_samples2, n_features)
        n_permute (int): Number of permutations
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
    n1, n_features = data1.shape
    n2 = data2.shape[0]
    n_total = n1 + n2

    # Compute observed mean difference
    obs_diff = np.mean(data1, axis=0) - np.mean(data2, axis=0)

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
        mean1 = np.mean(combined[group1_indices], axis=0)
        mean2 = np.mean(combined[group2_indices], axis=0)
        return mean1 - mean2

    # Execute in parallel with progress bar
    null_dist = Parallel(n_jobs=n_jobs)(
        delayed(_compute_one_perm)(seeds[i])
        for i in tqdm(range(n_permute), desc="CPU parallel perms", unit="perm")
    )
    null_dist = np.array(null_dist)  # Shape: (n_permute, n_features)

    # Compute p-values
    p_values = _compute_pvalue(obs_diff, null_dist, tail=tail)

    # Return to original shape
    if single_feature:
        obs_diff = float(obs_diff[0])
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
        "mean_diff": obs_diff,
        "p": p_values,
        "backend": backend_name,
    }

    if return_null:
        if single_feature:
            null_dist = null_dist.squeeze()
        result["null_dist"] = null_dist

    return result


def _two_sample_permutation_gpu_batched(
    data1: np.ndarray,
    data2: np.ndarray,
    n_permute: int,
    tail: int,
    return_null: bool,
    backend: Backend,
    max_gpu_memory_gb: float,
    random_state,
    single_feature: bool = False,
) -> dict:
    """
    Two-sample permutation test using GPU with automatic batching.

    Processes permutations in batches to avoid GPU OOM. Transfers data once
    and reuses across batches for efficiency.

    Args:
        data1 (np.ndarray): Group 1 data, shape (n_samples1, n_features)
        data2 (np.ndarray): Group 2 data, shape (n_samples2, n_features)
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

    n1, n_features = data1.shape
    n2 = data2.shape[0]
    n_total = n1 + n2

    # Convert to float32 for GPU efficiency
    data1 = data1.astype(np.float32)
    data2 = data2.astype(np.float32)

    # Compute observed mean difference
    obs_diff = np.mean(data1, axis=0) - np.mean(data2, axis=0)

    # Concatenate data for permutation
    combined = np.vstack([data1, data2])  # (n_total, n_features)

    # Determine batch size based on memory budget
    batch_size, n_batches = _auto_batch_size(
        n_permute, n_total, n_features, max_memory_gb=max_gpu_memory_gb
    )

    # Transfer data to device once
    combined_device = backend.to_device(combined)

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
        # Shape: (current_batch_size, n_total)
        batch_indices = np.array(
            [
                np.random.RandomState(batch_seeds[i]).permutation(n_total)
                for i in range(current_batch_size)
            ]
        )

        # Transfer to device and ensure indices are long type
        batch_indices_device = backend.to_device(batch_indices)
        if backend.name.startswith("torch"):
            batch_indices_device = batch_indices_device.long()

        # Compute mean differences for this batch
        # For each permutation, index into combined data
        batch_null = []
        for i in range(current_batch_size):
            indices = batch_indices_device[i]
            group1_indices = indices[:n1]
            group2_indices = indices[n1:]

            mean1 = torch.mean(combined_device[group1_indices], dim=0)
            mean2 = torch.mean(combined_device[group2_indices], dim=0)
            batch_null.append(mean1 - mean2)

        batch_null = torch.stack(batch_null)  # (current_batch_size, n_features)
        batch_null = backend.to_numpy(batch_null)

        null_dist_list.append(batch_null)

        # Update progress bar
        pbar.update(current_batch_size)

        # Free batch memory
        del batch_indices_device, batch_null
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pbar.close()

    # Combine batches: (n_permute, n_features)
    null_dist = np.vstack(null_dist_list)

    # Compute p-values
    p_values = _compute_pvalue(obs_diff, null_dist, tail=tail)

    # Return to original shape
    if single_feature:
        obs_diff = float(obs_diff[0])
        p_values = float(p_values[0])

    # Build result
    result = {
        "mean_diff": obs_diff,
        "p": p_values,
        "backend": backend.name,
    }

    if return_null:
        if single_feature:
            null_dist = null_dist.squeeze()
        result["null_dist"] = null_dist

    return result


def two_sample_permutation_test(
    data1: np.ndarray,
    data2: np.ndarray,
    n_permute: int = 5000,
    tail: int = 2,
    return_null: bool = False,
    backend: Optional[Union[Backend, str]] = None,
    n_jobs: int = -1,
    max_gpu_memory_gb: float = 4.0,
    random_state: Optional[int] = None,
) -> dict:
    """
    Two-sample permutation test using group label shuffling.

    Tests whether two independent groups have different means by randomly
    permuting group labels. This is the permutation test equivalent of an
    independent samples t-test.

    Assumption: Exchangeability under the null hypothesis (group assignments
    are arbitrary). Valid for independent samples from similar distributions.

    When backend='torch', uses GPU acceleration with automatic batching.
    When backend=None (default), uses efficient CPU parallelization via joblib.

    Args:
        data1 (np.ndarray): Group 1 data
            - shape (n_samples1,) for single feature
            - shape (n_samples1, n_features) for multi-feature (voxel-wise)
        data2 (np.ndarray): Group 2 data
            - shape (n_samples2,) for single feature
            - shape (n_samples2, n_features) for multi-feature (voxel-wise)
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
            backend='torch' or 'auto'.
        random_state (int, optional): Random seed for reproducibility

    Returns:
        dict: Dictionary with keys:
            - 'mean_diff' (float or np.ndarray): Observed mean difference (data1 - data2)
            - 'p' (float or np.ndarray): P-value(s)
            - 'null_dist' (np.ndarray): Null distribution (if return_null=True)
            - 'backend' (str): Backend used for computation

    Examples:
        >>> # Single feature
        >>> data1 = np.random.randn(20)  # Group 1: 20 subjects
        >>> data2 = np.random.randn(25)  # Group 2: 25 subjects
        >>> result = two_sample_permutation_test(data1, data2, n_permute=5000)
        >>> result['p']
        0.45

        >>> # Voxel-wise test with GPU
        >>> data1 = np.random.randn(20, 10000)  # 20 subjects, 10K voxels
        >>> data2 = np.random.randn(25, 10000)  # 25 subjects, 10K voxels
        >>> result = two_sample_permutation_test(data1, data2, n_permute=5000, backend='torch')
        >>> result['mean_diff'].shape
        (10000,)
        >>> result['p'].shape
        (10000,)

        >>> # CPU parallelization (no GPU)
        >>> result = two_sample_permutation_test(data1, data2, n_permute=5000, n_jobs=-1)

    Notes:
        - Default (backend=None): CPU parallelization with joblib (4-8× speedup)
        - GPU backend (torch): Fastest for large problems with automatic batching
        - NumPy backend: Single-threaded, use for small problems or debugging
        - For voxel-wise tests, each voxel tested independently
        - Group sizes can be unequal
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

    n1, n_features = data1.shape
    n2 = data2.shape[0]
    n_total = n1 + n2

    # Decide execution mode: GPU backend vs CPU parallelization
    use_cpu_parallel = backend is None

    if use_cpu_parallel:
        # CPU parallelization mode (memory-efficient fallback)
        return _two_sample_permutation_cpu_parallel(
            data1,
            data2,
            n_permute,
            tail,
            return_null,
            n_jobs,
            random_state,
            single_feature,
        )
    else:
        # GPU/Backend mode
        if isinstance(backend, str):
            if backend == "auto":
                # Auto-select based on problem size and GPU availability
                backend = auto_select_backend(n_total, n_features, cv=n_permute // 1000)
            else:
                backend = Backend(backend)

    # Setup random state
    rng = check_random_state(random_state)

    # Compute null distribution based on backend
    if backend.name == "numpy":
        # NumPy: Sequential processing (simple, memory-efficient)
        # Uses same deterministic RNG pattern as CPU-parallel and GPU

        # Compute observed mean difference
        obs_diff = np.mean(data1, axis=0) - np.mean(data2, axis=0)

        # Concatenate data for permutation
        combined = np.vstack([data1, data2])  # (n_total, n_features)

        # Pre-generate seeds for deterministic permutations
        # Matches CPU-parallel and GPU pattern: independent RandomState per permutation
        MAX_INT = 2**31 - 1
        seeds = rng.randint(MAX_INT, size=n_permute)

        # Generate null distribution
        null_dist = []
        for i in range(n_permute):
            # Use independent RandomState for this permutation (matches CPU-parallel)
            perm_rng = np.random.RandomState(seeds[i])
            indices = perm_rng.permutation(n_total)
            # Split into two groups
            group1_indices = indices[:n1]
            group2_indices = indices[n1:]
            # Compute mean difference
            mean1 = np.mean(combined[group1_indices], axis=0)
            mean2 = np.mean(combined[group2_indices], axis=0)
            null_dist.append(mean1 - mean2)

        null_dist = np.array(null_dist)  # (n_permute, n_features)

        # Compute p-values
        p_values = _compute_pvalue(obs_diff, null_dist, tail=tail)

        # Return to original shape (convert to scalar for single feature)
        if single_feature:
            obs_diff = float(obs_diff[0])
            p_values = float(p_values[0])

        # Build result dict
        result = {
            "mean_diff": obs_diff,
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
        return _two_sample_permutation_gpu_batched(
            data1,
            data2,
            n_permute,
            tail,
            return_null,
            backend,
            max_gpu_memory_gb,
            rng,
            single_feature,
        )
