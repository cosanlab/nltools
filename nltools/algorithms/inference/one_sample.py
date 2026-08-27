"""One-sample permutation test implementations.

This module provides CPU-parallel and GPU-batched implementations
of the one-sample permutation test (sign-flipping test).
"""

import numpy as np
from sklearn.utils import check_random_state

from nltools.algorithms.backends import Backend
from .utils import (
    _generate_sign_flips,
    _compute_pvalue,
    _auto_batch_size,
    maybe_tqdm,
    make_progress_bar,
)
from .validation import (
    validate_tail_parameter,
    validate_device_parameter,
    validate_array_shape_range,
)


def _one_sample_permutation_cpu_parallel(
    data: np.ndarray,
    *,
    n_permute: int,
    tail: int,
    return_null: bool,
    n_jobs: int,
    random_state: int | None,
    single_feature: bool = False,
    progress_bar: bool = False,
) -> dict:
    """One-sample permutation test using CPU parallelization with joblib.

    Pre-generates all sign-flips deterministically (matching stats.py pattern),
    then parallelizes only the computation. This ensures perfect reproducibility
    and backward compatibility with nltools.algorithms.one_sample_permutation.

    Args:
        data (np.ndarray): Data to test, shape (n_samples, n_features)
        n_permute (int): Number of permutations
        tail (int | str): Test type (2|'two' or 1|'one')
        return_null (bool): Whether to return null distribution
        n_jobs (int): Number of parallel jobs (-1 = all cores)
        random_state (int, optional): Random seed for reproducibility
        single_feature (bool): Whether data is single feature
        progress_bar (bool): Whether to display a tqdm progress bar

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
        for i in maybe_tqdm(
            range(n_permute),
            progress_bar=progress_bar,
            desc="CPU parallel perms",
            unit="perm",
        )
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
        "device": "cpu",
    }

    if return_null:
        if single_feature:
            null_dist = null_dist.squeeze()
        result["null_dist"] = null_dist

    return result


def _one_sample_permutation_gpu_batched(
    data: np.ndarray,
    *,
    n_permute: int,
    tail: int,
    return_null: bool,
    backend: Backend,
    max_gpu_memory_gb: float,
    random_state,
    single_feature: bool = False,
    progress_bar: bool = False,
) -> dict:
    """One-sample permutation test using GPU with automatic batching.

    Processes permutations in batches to avoid GPU OOM. Transfers data once
    and reuses across batches for efficiency.

    Args:
        data (np.ndarray): Data to test, shape (n_samples, n_features)
        n_permute (int): Number of permutations
        tail (int | str): Test type (2|'two' or 1|'one')
        return_null (bool): Whether to return null distribution
        backend (Backend): Backend instance (must be PyTorch)
        max_gpu_memory_gb (float): Maximum GPU memory budget
        random_state: Random state instance
        single_feature (bool): Whether data is single feature
        progress_bar (bool): Whether to display a tqdm progress bar

    Returns:
        dict: Same format as main function, with 'backend' indicating GPU device
    """
    import torch

    from nltools.algorithms.backends import compute_oom_safe

    n_samples, n_features = data.shape

    # Convert to float32 for GPU efficiency
    data = data.astype(np.float32)

    # Compute observed statistic
    obs_stat = np.mean(data, axis=0)

    # Determine batch size based on memory budget
    batch_size, n_batches = _auto_batch_size(
        n_permute,
        n_samples,
        n_features,
        max_memory_gb=max_gpu_memory_gb,
        backend=backend,
    )

    # Transfer data to device once (reused across batches)
    data_device = backend.to_device(data)

    def _compute_batch(batch_sign_flips: np.ndarray) -> np.ndarray:
        """Device compute for one (sub-)batch of pre-drawn sign flips."""
        sign_flips_device = backend.to_device(batch_sign_flips.astype(np.float32))
        # Broadcasting: (batch_size, n_samples, 1) * (1, n_samples, n_features)
        data_perm = sign_flips_device[:, :, None] * data_device[None, :, :]
        batch_null = backend.to_numpy(torch.mean(data_perm, dim=1))
        del sign_flips_device, data_perm
        return batch_null

    # Accumulate null distribution across batches
    null_dist_list = []

    # Process permutations in batches with progress bar
    pbar = make_progress_bar(
        progress_bar=progress_bar and n_batches > 1,  # pointless bar if 1 batch
        total=n_permute,
        desc="GPU permutation batches",
        unit="perm",
    )

    for batch_idx in range(n_batches):
        # Determine current batch size
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_permute)
        current_batch_size = end_idx - start_idx

        # Generate sign flips for this batch only. RNG draws stay outside the
        # OOM-retried compute, so recovery reuses these exact flips.
        batch_sign_flips = _generate_sign_flips(
            current_batch_size, n_samples, random_state=random_state
        )

        null_dist_list.append(compute_oom_safe(_compute_batch, batch_sign_flips))

        # Update progress bar
        pbar.update(current_batch_size)

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
        "device": "gpu",
    }

    if return_null:
        if single_feature:
            null_dist = null_dist.squeeze()
        result["null_dist"] = null_dist

    return result


def one_sample_permutation_test(
    data: np.ndarray,
    *,
    n_permute: int = 5000,
    tail: int | str = 2,
    return_null: bool = False,
    device: str | None = "cpu",
    n_jobs: int = -1,
    max_gpu_memory_gb: float | None = None,
    random_state: int | None = None,
    progress_bar: bool = False,
) -> dict:
    """One-sample permutation test using sign-flipping.

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
        tail (int | str): Test type — 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction)
            - 'two' or 2: Two-tailed test (mean != 0)
            - 2 | 'two': Two-tailed test (mean != 0)
            - 1 | 'one': One-tailed (mean > 0; negate the data for the other
              direction). The fixed direction keeps MCP correction valid.
        return_null (bool): If True, return full null distribution (default: False)
        device (str, optional): Parallelization method (default: 'cpu')
            - None: Single-threaded NumPy (for debugging/small problems)
            - 'cpu': CPU parallelization via joblib (default, 4-8× speedup)
            - 'gpu': GPU acceleration via PyTorch (fastest for large problems)
        n_jobs (int): Number of CPU cores for parallelization (default: -1 = all cores)
            Only used when device='cpu'
        max_gpu_memory_gb (float, optional): Explicit GPU memory budget in GB.
            None (default) measures the device's available memory.
            Controls automatic batching to prevent OOM errors. Only used with
            device='gpu'. Larger values allow more permutations per batch but
            risk OOM on smaller GPUs.
        random_state (int, optional): Random seed for reproducibility
        progress_bar (bool): Whether to display a progress bar (default: False)

    Returns:
        dict: Dictionary with keys:
            - 'mean' (float or np.ndarray): Observed mean(s)
            - 'p' (float or np.ndarray): P-value(s)
            - 'null_dist' (np.ndarray): Null distribution (if return_null=True)
            - 'device' (str): Parallelization method used

    Examples:
        >>> # Single feature (default CPU parallelization)
        >>> data = np.random.randn(30)
        >>> result = one_sample_permutation_test(data, n_permute=5000)
        >>> result['p']
        0.23

        >>> # Voxel-wise test with GPU
        >>> data = np.random.randn(30, 10000)  # 30 subjects, 10K voxels
        >>> result = one_sample_permutation_test(data, n_permute=5000, device='gpu')
        >>> result['mean'].shape
        (10000,)
        >>> result['p'].shape
        (10000,)

        >>> # Single-threaded (for debugging)
        >>> result = one_sample_permutation_test(data, n_permute=5000, device=None)

    Notes:
        - Default (device='cpu'): CPU parallelization with joblib (4-8× speedup)
        - GPU parallelization ('gpu'): Fastest for large problems with automatic batching
        - Single-threaded (device=None): Use for small problems or debugging
        - For voxel-wise tests, each voxel tested independently
        - Progress bars show completion for both CPU parallel and GPU batched modes
    """
    # Input validation
    data = np.asarray(data, dtype=np.float64)
    validate_array_shape_range(data, 1, 2, name="data")
    validate_tail_parameter(tail)
    validate_device_parameter(device)

    # Handle shape
    single_feature = data.ndim == 1
    if single_feature:
        data = data[:, np.newaxis]  # (n_samples, 1)

    n_samples, n_features = data.shape

    # Decide execution mode based on device parameter
    if device == "cpu" or device is None:
        # CPU modes
        if device is None:
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
                "device": None,
            }

            if return_null:
                if single_feature:
                    null_dist = null_dist.squeeze()
                result["null_dist"] = null_dist

            return result
        # CPU parallelization mode
        return _one_sample_permutation_cpu_parallel(
            data,
            n_permute=n_permute,
            tail=tail,
            return_null=return_null,
            n_jobs=n_jobs,
            random_state=random_state,
            single_feature=single_feature,
            progress_bar=progress_bar,
        )
    # GPU mode
    backend_obj = Backend("torch")
    rng = check_random_state(random_state)
    return _one_sample_permutation_gpu_batched(
        data,
        n_permute=n_permute,
        tail=tail,
        return_null=return_null,
        backend=backend_obj,
        max_gpu_memory_gb=max_gpu_memory_gb,
        random_state=rng,
        single_feature=single_feature,
        progress_bar=progress_bar,
    )
