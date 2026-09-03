"""Two-sample permutation test (group-label shuffling).

Tests whether two independent groups differ in mean by randomly reassigning
observations to groups — the permutation analogue of an independent-samples
t-test. `device=` picks the execution path (single-threaded numpy, joblib
across `n_jobs` cores, or batched PyTorch on the GPU).
"""

import numpy as np
from sklearn.utils import check_random_state

from nltools.algorithms.backends import Backend, resolve_backend
from .utils import (
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
from ..random import generate_seeds


def _two_sample_permutation_cpu_parallel(
    data1: np.ndarray,
    data2: np.ndarray,
    *,
    n_permute: int,
    tail: int,
    return_null: bool,
    n_jobs: int,
    random_state: int | None,
    single_feature: bool = False,
    progress_bar: bool = False,
) -> dict:
    """Two-sample permutation test parallelized across CPU cores with joblib.

    Each worker shuffles the group labels for one permutation (from its own
    pre-drawn seed) and computes the mean difference, so results are
    reproducible regardless of worker count.

    Args:
        data1 (np.ndarray): Group 1 data, shape `(n_samples1, n_features)`.
        data2 (np.ndarray): Group 2 data, shape `(n_samples2, n_features)`.
        n_permute (int): Number of permutations.
        tail (int | str): `2` or `'two'` for two-tailed; `1` or `'one'` for
            one-tailed.
        return_null (bool): Whether to return the null distribution.
        n_jobs (int): Number of parallel jobs (-1 = all cores).
        random_state (int | None): Random seed for reproducibility.
        single_feature (bool): Whether the caller passed 1D data (results are
            returned as scalars).
        progress_bar (bool): Whether to display a tqdm progress bar.

    Returns:
        dict: Same format as `two_sample_permutation_test`, with `'device'`
            set to `'cpu'`.
    """
    from joblib import Parallel, delayed

    # Setup random state and generate seeds for workers
    seeds = generate_seeds(n_permute, random_state=random_state)

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
        for i in maybe_tqdm(
            range(n_permute),
            progress_bar=progress_bar,
            desc="CPU parallel perms",
            unit="perm",
        )
    )
    null_dist = np.array(null_dist)  # Shape: (n_permute, n_features)

    # Compute p-values
    p_values = _compute_pvalue(obs_diff, null_dist, tail=tail)

    # Return to original shape
    if single_feature:
        obs_diff = obs_diff.item() if hasattr(obs_diff, "item") else float(obs_diff[0])
        p_values = p_values.item() if hasattr(p_values, "item") else float(p_values[0])

    # Build result
    result = {
        "mean_diff": obs_diff,
        "p": p_values,
        "device": "cpu",
    }

    if return_null:
        if single_feature:
            null_dist = null_dist.squeeze()
        result["null_dist"] = null_dist

    return result


def _two_sample_permutation_gpu_batched(
    data1: np.ndarray,
    data2: np.ndarray,
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
    """Two-sample permutation test on the GPU with automatic batching.

    Processes permutations in memory-budgeted batches to avoid GPU OOM; the
    stacked data is transferred once and reused across batches.

    Args:
        data1 (np.ndarray): Group 1 data, shape `(n_samples1, n_features)`.
        data2 (np.ndarray): Group 2 data, shape `(n_samples2, n_features)`.
        n_permute (int): Number of permutations.
        tail (int | str): `2` or `'two'` for two-tailed; `1` or `'one'` for
            one-tailed.
        return_null (bool): Whether to return the null distribution.
        backend (Backend): Resolved PyTorch backend.
        max_gpu_memory_gb (float | None): GPU memory budget; None measures the
            device.
        random_state (np.random.RandomState): Random state instance.
        single_feature (bool): Whether the caller passed 1D data (results are
            returned as scalars).
        progress_bar (bool): Whether to display a tqdm progress bar.

    Returns:
        dict: Same format as `two_sample_permutation_test`, with `'device'`
            set to `'gpu'`.
    """
    import torch

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

    from nltools.algorithms.backends import compute_oom_safe

    # Determine batch size based on memory budget
    batch_size, n_batches = _auto_batch_size(
        n_permute,
        n_total,
        n_features,
        max_memory_gb=max_gpu_memory_gb,
        backend=backend,
    )

    # Transfer data to device once
    combined_device = backend.to_device(combined)

    def _compute_batch(batch_indices: np.ndarray) -> np.ndarray:
        """Device compute for one (sub-)batch of pre-drawn permutations."""
        batch_indices_device = backend.to_device(batch_indices)
        if backend.name.startswith("torch"):
            batch_indices_device = batch_indices_device.long()

        # For each permutation, index into combined data
        batch_null = []
        for i in range(len(batch_indices)):
            indices = batch_indices_device[i]
            group1_indices = indices[:n1]
            group2_indices = indices[n1:]

            mean1 = torch.mean(combined_device[group1_indices], dim=0)
            mean2 = torch.mean(combined_device[group2_indices], dim=0)
            batch_null.append(mean1 - mean2)

        result = backend.to_numpy(torch.stack(batch_null))
        del batch_indices_device, batch_null
        return result

    # Accumulate null distribution across batches
    null_dist_list = []

    # Process permutations in batches with progress bar
    pbar = make_progress_bar(
        progress_bar=progress_bar,
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

        # Generate permutation indices using independent RNG per permutation.
        # Shape: (current_batch_size, n_total). RNG draws stay outside the
        # OOM-retried compute, so recovery reuses these exact permutations.
        batch_indices = np.array(
            [
                np.random.RandomState(batch_seeds[i]).permutation(n_total)
                for i in range(current_batch_size)
            ]
        )

        null_dist_list.append(compute_oom_safe(_compute_batch, batch_indices))

        # Update progress bar
        pbar.update(current_batch_size)

    pbar.close()

    # Combine batches: (n_permute, n_features)
    null_dist = np.vstack(null_dist_list)

    # Compute p-values
    p_values = _compute_pvalue(obs_diff, null_dist, tail=tail)

    # Return to original shape
    if single_feature:
        obs_diff = obs_diff.item() if hasattr(obs_diff, "item") else float(obs_diff[0])
        p_values = p_values.item() if hasattr(p_values, "item") else float(p_values[0])

    # Build result
    result = {
        "mean_diff": obs_diff,
        "p": p_values,
        "device": "gpu",
    }

    if return_null:
        if single_feature:
            null_dist = null_dist.squeeze()
        result["null_dist"] = null_dist

    return result


def two_sample_permutation_test(
    data1: np.ndarray,
    data2: np.ndarray,
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
    """Two-sample permutation test using group-label shuffling.

    Tests whether two independent groups have different means by randomly
    reassigning observations to groups — the permutation analogue of an
    independent-samples t-test. Group sizes may differ. Multi-feature
    (voxel-wise) data tests each column independently against the same
    permutations.

    Assumes exchangeability under the null (group assignment is arbitrary):
    independent samples from similarly shaped distributions.

    Args:
        data1 (np.ndarray): Group 1 data, shape `(n_samples1,)` for a single
            feature or `(n_samples1, n_features)` for voxel-wise data.
        data2 (np.ndarray): Group 2 data, shape `(n_samples2,)` or
            `(n_samples2, n_features)`; must have the same number of features
            as `data1`.
        n_permute (int): Number of permutations. Defaults to 5000.
        tail (int | str): `2` or `'two'` (default) for a two-tailed test
            (mean1 != mean2); `1` or `'one'` for a one-tailed test of
            mean1 > mean2 (swap the groups for the other direction — the fixed
            direction keeps multiple-comparison correction valid).
        return_null (bool): If True, include the full null distribution in the
            result. Defaults to False.
        device (str | None): Execution path. `'cpu'` (default) parallelizes
            with joblib across `n_jobs` cores (4-8× speedup); `'gpu'` batches
            permutations through PyTorch (fastest for large problems); None
            runs single-threaded numpy (small problems, debugging).
        n_jobs (int): CPU cores for `device='cpu'`. Defaults to -1 (all cores).
        max_gpu_memory_gb (float | None): GPU memory budget in GB for
            `device='gpu'`; controls automatic batching. None (default)
            measures the device's available memory.
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Whether to display a progress bar. Defaults to False.

    Returns:
        dict: Keys `'mean_diff'` (float or np.ndarray, observed
            `mean(data1) - mean(data2)`), `'p'` (float or np.ndarray,
            p-value(s)), `'device'` (the execution path used), and — when
            `return_null=True` — `'null_dist'` (np.ndarray, shape
            `(n_permute,)` or `(n_permute, n_features)`).

    Examples:
        ```python
        # Single feature (default CPU parallelization)
        data1 = np.random.randn(20)  # Group 1: 20 subjects
        data2 = np.random.randn(25)  # Group 2: 25 subjects
        result = two_sample_permutation_test(data1, data2, n_permute=5000)
        result["p"]  # → 0.45

        # Voxel-wise test on the GPU
        data1 = np.random.randn(20, 10000)  # 20 subjects, 10K voxels
        data2 = np.random.randn(25, 10000)  # 25 subjects, 10K voxels
        result = two_sample_permutation_test(data1, data2, n_permute=5000, device="gpu")
        result["mean_diff"].shape  # → (10000,)
        result["p"].shape  # → (10000,)

        # Single-threaded (for debugging)
        result = two_sample_permutation_test(data1, data2, n_permute=5000, device=None)
        ```
    """
    # Validate device parameter
    validate_device_parameter(device)

    # Input validation
    data1 = np.asarray(data1, dtype=np.float64)
    data2 = np.asarray(data2, dtype=np.float64)

    validate_array_shape_range(data1, 1, 2, name="data1")
    validate_array_shape_range(data2, 1, 2, name="data2")
    validate_tail_parameter(tail)

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

    # Decide execution mode based on device parameter
    if device == "cpu" or device is None:
        # CPU modes
        if device is None:
            # Single-threaded NumPy
            rng = check_random_state(random_state)
            obs_diff = np.mean(data1, axis=0) - np.mean(data2, axis=0)
            combined = np.vstack([data1, data2])
            MAX_INT = 2**31 - 1
            seeds = rng.randint(MAX_INT, size=n_permute)

            null_dist = []
            for i in range(n_permute):
                perm_rng = np.random.RandomState(seeds[i])
                indices = perm_rng.permutation(n_total)
                group1_indices = indices[:n1]
                group2_indices = indices[n1:]
                mean1 = np.mean(combined[group1_indices], axis=0)
                mean2 = np.mean(combined[group2_indices], axis=0)
                null_dist.append(mean1 - mean2)

            null_dist = np.array(null_dist)
            p_values = _compute_pvalue(obs_diff, null_dist, tail=tail)

            if single_feature:
                obs_diff = (
                    obs_diff.item() if hasattr(obs_diff, "item") else float(obs_diff[0])
                )
                p_values = (
                    p_values.item() if hasattr(p_values, "item") else float(p_values[0])
                )

            result = {
                "mean_diff": obs_diff,
                "p": p_values,
                "device": None,
            }

            if return_null:
                if single_feature:
                    null_dist = null_dist.squeeze()
                result["null_dist"] = null_dist

            return result
        # CPU parallelization mode
        return _two_sample_permutation_cpu_parallel(
            data1,
            data2,
            n_permute=n_permute,
            tail=tail,
            return_null=return_null,
            n_jobs=n_jobs,
            random_state=random_state,
            single_feature=single_feature,
            progress_bar=progress_bar,
        )
    # GPU mode
    backend_obj = resolve_backend("gpu")
    rng = check_random_state(random_state)
    return _two_sample_permutation_gpu_batched(
        data1,
        data2,
        n_permute=n_permute,
        tail=tail,
        return_null=return_null,
        backend=backend_obj,
        max_gpu_memory_gb=max_gpu_memory_gb,
        random_state=rng,
        single_feature=single_feature,
        progress_bar=progress_bar,
    )
