"""Permutation test for the correlation between two variables.

`correlation_permutation_test` asks whether the Pearson, Spearman, or Kendall
correlation between two arrays differs from zero, building the null distribution
by shuffling one array's observations. It assumes observations are independent;
for autocorrelated time series use `timeseries_correlation_permutation_test`.
Multi-feature inputs (2D arrays) test each column pair independently. `device`
selects the execution path: `'cpu'` (default) spreads permutations across
`n_jobs` workers, `'gpu'` vectorizes them in memory-bounded batches, `None` runs
single-threaded.
"""

import numpy as np
from collections.abc import Callable
from typing import TYPE_CHECKING
from sklearn.utils import check_random_state
from scipy.stats import rankdata, kendalltau

from nltools.algorithms.backends import Backend
from .utils import (
    _compute_pvalue,
    _auto_batch_size,
    EPSILON,
    maybe_tqdm,
    make_progress_bar,
)
from .validation import validate_device_parameter, validate_tail_parameter

if TYPE_CHECKING:
    import torch


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray | float:
    """Compute Pearson correlation coefficient(s).

    Args:
        x (np.ndarray): Data array, shape (n_samples,) or (n_permute, n_samples).
        y (np.ndarray): Data array, shape (n_samples,).

    Returns:
        float | np.ndarray: A scalar if `x` is 1D, else one correlation per row
            of `x`, shape (n_permute,).

    Note:
        Centers the data before computing, and vectorizes across the rows of a
        2D `x`.
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
        x (np.ndarray): Data array, shape (n_samples,) or (n_permute, n_samples).
        y (np.ndarray): Data array, shape (n_samples,).

    Returns:
        float | np.ndarray: A scalar if `x` is 1D, else one correlation per row
            of `x`, shape (n_permute,).

    Note:
        Ranks with `scipy.stats.rankdata` (ties get their average rank), then
        applies the Pearson correlation to the ranks.
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
        x (np.ndarray): Data array, shape (n_samples,) or (n_permute, n_samples).
        y (np.ndarray): Data array, shape (n_samples,).

    Returns:
        float | np.ndarray: A scalar if `x` is 1D, else one correlation per row
            of `x`, shape (n_permute,). A NaN tau (constant input) is returned
            as 0.0.

    Note:
        Calls `scipy.stats.kendalltau` once per row; O(n²) per call, so slower
        than Pearson or Spearman.
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


def _select_corr_func(
    metric: str,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray | float]:
    """Return the correlation function for a metric name."""
    if metric == "pearson":
        return _pearson_correlation
    if metric == "spearman":
        return _spearman_correlation
    if metric == "kendall":
        return _kendall_correlation
    raise NotImplementedError(f"Metric '{metric}' not yet implemented")


def _correlation_permutation_cpu_parallel(
    data1: np.ndarray,
    data2: np.ndarray,
    *,
    n_permute: int,
    metric: str,
    tail: int,
    return_null: bool,
    n_jobs: int,
    random_state: int | None,
    single_feature: bool = False,
    progress_bar: bool = False,
) -> dict:
    """Correlation permutation test parallelized across CPU workers with joblib.

    Each worker handles one permutation: shuffle `data1`, correlate with `data2`.
    Seeds are pre-generated from `random_state`, so results do not depend on
    `n_jobs`.

    Args:
        data1 (np.ndarray): Data to permute, shape (n_samples, n_features).
        data2 (np.ndarray): Data to correlate with, shape (n_samples, n_features).
        n_permute (int): Number of permutations.
        metric (str): Correlation metric, one of 'pearson', 'spearman', or 'kendall'.
        tail (int | str): `2` or `'two'` for two-tailed; `1` or `'one'` for one-tailed.
        return_null (bool): Whether to return the null distribution.
        n_jobs (int): Number of parallel workers (-1 = all cores).
        random_state (int | None): Random seed for reproducibility.
        single_feature (bool): Whether the caller passed 1D inputs (results are
            returned as scalars).
        progress_bar (bool): Show a progress bar over permutations.

    Returns:
        dict: Same keys as `correlation_permutation_test`, with `'device'` set to
            `'cpu'`.
    """
    from joblib import Parallel, delayed

    # Setup random state and generate seeds for workers
    rng = check_random_state(random_state)
    MAX_INT = 2**31 - 1
    seeds = rng.randint(MAX_INT, size=n_permute)

    # Get dimensions (data already reshaped by caller)
    n_samples, n_features = data1.shape

    # Select correlation function
    corr_func = _select_corr_func(metric)

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
        for i in maybe_tqdm(
            range(n_permute),
            progress_bar=progress_bar,
            desc="CPU parallel perms",
            unit="perm",
        )
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
        "device": "cpu",
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

    Ties receive their average rank, matching `scipy.stats.rankdata`. Used for
    Spearman correlation.

    Args:
        data (torch.Tensor): Input tensor, shape (..., n_samples, ...).
        dim (int): Dimension along which to rank. Defaults to the last dimension.
        method (str): Ranking method; only "average" is supported.

    Returns:
        torch.Tensor: Ranks with the same shape as the input, dtype float32.

    Note:
        Each slice along `dim` is sorted (stably); runs of equal sorted values
        share the mean of their positional ranks, and the ranks are scattered
        back to the original order.
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
    for row in range(n_batches):
        batch_data = data_flat[row]  # (n_samples,)

        # Sort data to handle ties
        sorted_data, sorted_indices = torch.sort(batch_data, stable=True)

        # Initial ranks: 1, 2, 3, ..., n_samples
        ranks = torch.arange(1, n_samples + 1, dtype=torch.float32, device=device)

        # Handle tied ranks: average method (matches scipy.stats.rankdata).
        # Sorted equal values form runs; each run's ranks become its mean rank.
        if n_samples > 1:
            # ties[j] is True when sorted_data[j] == sorted_data[j - 1], so a
            # run of equal values starts at each False position.
            diff = torch.diff(sorted_data)
            ties = torch.cat([torch.tensor([False], device=device), diff == 0])

            if ties.any():
                run_starts = torch.nonzero(~ties).flatten().tolist()
                run_bounds = run_starts + [n_samples]
                for start, end in zip(run_bounds[:-1], run_bounds[1:]):
                    if end - start > 1:
                        ranks[start:end] = ranks[start:end].mean()

        # Map ranks back to original order
        ranked_flat[row, sorted_indices] = ranks

    # Reshape back to reordered shape
    ranked_reordered = ranked_flat.reshape(*data_reordered.shape)

    # Restore original dimension order
    ranked = ranked_reordered.permute(*dims)

    return ranked


def _correlation_permutation_gpu_batched(
    data1: np.ndarray,
    data2: np.ndarray,
    *,
    n_permute: int,
    metric: str,
    tail: int,
    return_null: bool,
    backend: Backend,
    max_gpu_memory_gb: float,
    random_state,
    single_feature: bool = False,
    progress_bar: bool = False,
) -> dict:
    """Correlation permutation test on the GPU with automatic batching.

    Permutations run in memory-bounded batches to avoid OOM; the data is
    transferred once and reused across batches.

    Args:
        data1 (np.ndarray): Data to permute, shape (n_samples, n_features).
        data2 (np.ndarray): Data to correlate with, shape (n_samples, n_features).
        n_permute (int): Number of permutations.
        metric (str): Correlation metric, one of 'pearson', 'spearman', or 'kendall'.
        tail (int | str): `2` or `'two'` for two-tailed; `1` or `'one'` for one-tailed.
        return_null (bool): Whether to return the null distribution.
        backend (Backend): Backend instance (must be PyTorch).
        max_gpu_memory_gb (float | None): GPU memory budget in GB; None measures
            the device.
        random_state (np.random.RandomState): Random state used to draw the
            per-permutation seeds.
        single_feature (bool): Whether the caller passed 1D inputs (results are
            returned as scalars).
        progress_bar (bool): Show a progress bar over batches.

    Returns:
        dict: Same keys as `correlation_permutation_test`, with `'device'` set to
            `'gpu'`.
    """
    import torch

    n_samples, n_features = data1.shape

    if metric not in ("pearson", "spearman", "kendall"):
        raise NotImplementedError(
            f"_correlation_permutation_gpu_batched does not implement metric={metric!r}."
        )

    # Convert to float32 for GPU efficiency
    data1 = data1.astype(np.float32)
    data2 = data2.astype(np.float32)

    from nltools.algorithms.backends import (
        auto_batch_size,
        compute_oom_safe,
        device_memory_budget,
    )

    # Determine batch size based on memory budget
    if metric == "kendall":
        # Kendall's working set is the gathered pairwise-sign tensor
        # (batch, n, n, f) float32 plus its product scratch — n² per feature,
        # not n like the other metrics.
        budget_gb = device_memory_budget(
            backend, max_gpu_memory_gb=max_gpu_memory_gb, cap_for_batching=True
        )
        batch_size, n_batches = auto_batch_size(
            n_permute,
            n_samples * n_samples * n_features * 4,
            budget_gb=budget_gb,
            overhead=2.0,
        )
    else:
        # Memory bottleneck: permuted indices and correlation computation
        batch_size, n_batches = _auto_batch_size(
            n_permute,
            n_samples,
            n_features,
            max_memory_gb=max_gpu_memory_gb,
            backend=backend,
        )

    # Transfer data to device once (for GPU-accelerated observed correlation)
    data1_device = backend.to_device(data1)
    data2_device = backend.to_device(data2)

    # Pre-rank data2 for Spearman (only needs to be done once)
    if metric == "spearman":
        data2_ranked_device = _rank_transform_gpu(data2_device, dim=0)
    else:
        data2_ranked_device = None

    # Pre-compute pairwise sign tensors for Kendall (once; permutations only
    # reindex them). sx[i, j, f] = sign(x[i, f] - x[j, f]).
    if metric == "kendall":
        sx = torch.sign(data1_device.unsqueeze(1) - data1_device.unsqueeze(0))
        sy = torch.sign(data2_device.unsqueeze(1) - data2_device.unsqueeze(0))
        # Tau-b tie correction: denom = sqrt((n0 - n1) (n0 - n2)) with
        # n0 = n(n-1)/2 and n1/n2 the tied-pair counts, which equal the
        # off-diagonal zeros of the sign matrices halved. The tie structure —
        # and therefore the denominator — is permutation-invariant.
        n0 = n_samples * (n_samples - 1) / 2.0
        n1 = ((sx == 0).sum(dim=(0, 1)).float() - n_samples) / 2.0
        n2 = ((sy == 0).sum(dim=(0, 1)).float() - n_samples) / 2.0
        kendall_denom = torch.sqrt((n0 - n1) * (n0 - n2))  # (n_features,)

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
    elif metric == "kendall":
        # Full-matrix sum double-counts each i<j pair, hence / 2. A zero
        # denominator (constant column) maps to tau = 0.0, matching the CPU
        # path's NaN -> 0.0 convention.
        num_obs = torch.einsum("ijf,ijf->f", sx, sy) / 2.0
        obs_corr = backend.to_numpy(
            torch.where(
                kendall_denom > 0, num_obs / kendall_denom, torch.zeros_like(num_obs)
            )
        )

    def _compute_batch(batch_indices: np.ndarray) -> np.ndarray:
        """Device compute for one (sub-)batch of pre-drawn permutations."""
        current_batch_size = len(batch_indices)

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
        else:  # kendall (validated at function entry)
            # Permuting x permutes the pairwise sign matrix on both axes:
            # sign(x[p(i)] - x[p(j)]) = sx[p(i), p(j)] — gather instead of
            # recompute. bi: (current_batch_size, n_samples) long.
            bi = batch_indices_device
            sx_p = sx[bi[:, :, None], bi[:, None, :]]  # (b, n, n, f)
            num = torch.einsum("bijf,ijf->bf", sx_p, sy) / 2.0
            batch_corrs = torch.where(
                kendall_denom > 0, num / kendall_denom, torch.zeros_like(num)
            )
            batch_corrs = backend.to_numpy(batch_corrs)
            del sx_p

        del batch_indices_device
        return batch_corrs

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

        # Pre-generate seeds for this batch (memory-efficient, deterministic).
        # Matches CPU-parallel pattern: independent RandomState per permutation.
        # RNG draws stay outside the OOM-retried compute, so recovery reuses
        # these exact permutations.
        MAX_INT = 2**31 - 1
        batch_seeds = random_state.randint(MAX_INT, size=current_batch_size)

        # Generate permutation indices using independent RNG per permutation
        batch_indices = np.array(
            [
                np.random.RandomState(batch_seeds[i]).permutation(n_samples)
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
    p_values = _compute_pvalue(obs_corr, null_dist, tail=tail)

    # Return to original shape
    if single_feature:
        obs_corr = obs_corr.item() if hasattr(obs_corr, "item") else float(obs_corr[0])
        p_values = p_values.item() if hasattr(p_values, "item") else float(p_values[0])

    # Build result
    result = {
        "correlation": obs_corr,
        "p": p_values,
        "device": "gpu",
    }

    if return_null:
        if single_feature:
            null_dist = null_dist.squeeze()
        result["null_dist"] = null_dist

    return result


def correlation_permutation_test(
    data1: np.ndarray,
    data2: np.ndarray,
    *,
    n_permute: int = 5000,
    metric: str = "pearson",
    tail: int | str = 2,
    return_null: bool = False,
    device: str | None = "cpu",
    n_jobs: int = -1,
    max_gpu_memory_gb: float | None = None,
    random_state: int | None = None,
    progress_bar: bool = False,
) -> dict:
    """Permutation test for whether the correlation between two arrays differs from zero.

    Builds the null distribution by randomly permuting the observations of `data1`
    and re-correlating with `data2`. Assumes observations are independent (i.i.d.);
    for autocorrelated time series use `timeseries_correlation_permutation_test`,
    whose `'circle_shift'` and `'phase_randomize'` methods preserve temporal
    structure. With 2D inputs each column of `data1` is tested against the
    matching column of `data2`, independently.

    Args:
        data1 (np.ndarray): Data to permute, shape (n_samples,) for a single
            feature or (n_samples, n_features) for several.
        data2 (np.ndarray): Data to correlate with, same shape as `data1`.
        n_permute (int): Number of permutations. Defaults to 5000.
        metric (str): 'pearson' (linear), 'spearman' (rank-based, monotonic), or
            'kendall' (tau-b, ordinal association, tie-corrected). Defaults to
            'pearson'.
        tail (int | str): `2` or `'two'` for a two-tailed test (r != 0); `1` or
            `'one'` for a one-tailed test of r > 0 (negate one variable for the
            other direction; the fixed direction keeps multiple-comparison
            correction valid). Defaults to 2.
        return_null (bool): Also return the full null distribution. Defaults to
            False.
        device (str | None): Execution path. `'cpu'` parallelizes permutations
            across `n_jobs` joblib workers (4-8× speedup); `'gpu'` vectorizes them
            with PyTorch in memory-bounded batches (fastest for large problems;
            Pearson and Spearman run 5-20× faster on multi-feature data, Kendall
            needs O(n²) memory per permutation so its batches are smaller);
            `None` runs single-threaded NumPy (for debugging or small problems).
            Defaults to 'cpu'.
        n_jobs (int): Number of CPU workers, -1 = all cores; only used when
            `device='cpu'`. Defaults to -1.
        max_gpu_memory_gb (float | None): GPU memory budget in GB that sizes the
            permutation batches; only used when `device='gpu'`. None (default)
            measures the device's available memory. Larger values fit more
            permutations per batch but risk OOM on smaller GPUs.
        random_state (int | None): Random seed for reproducibility.
        progress_bar (bool): Show a progress bar over permutations. Defaults to
            False.

    Returns:
        dict: Keys 'correlation' (float, or np.ndarray of shape (n_features,) for
            2D inputs: the observed correlation), 'p' (float or np.ndarray, the
            matching p-values), 'device' (the execution path used: `'cpu'`,
            `'gpu'`, or `None`), and 'null_dist' (np.ndarray of shape
            (n_permute,) or (n_permute, n_features)) when `return_null=True`.

    Examples:
        ```python
        import numpy as np
        from nltools.algorithms import correlation_permutation_test

        # Single feature (default: CPU parallel)
        x = np.random.randn(100)
        y = x + np.random.randn(100) * 0.5
        result = correlation_permutation_test(x, y, n_permute=5000)
        result["correlation"]  # → 0.85 (approximately)
        result["p"]  # → 0.0002

        # Multi-feature: each column pair tested independently
        data1 = np.random.randn(100, 10)
        data2 = data1 + np.random.randn(100, 10) * 0.3
        result = correlation_permutation_test(data1, data2, n_permute=5000)
        result["correlation"].shape  # → (10,)
        result["p"].shape  # → (10,)

        # GPU acceleration
        result = correlation_permutation_test(data1, data2, n_permute=5000, device="gpu")
        ```

    Note:
        Kendall's tau is O(n²) in the number of samples on every path, so it is
        markedly slower than Pearson or Spearman for large samples.
    """
    validate_device_parameter(device)

    # Input validation
    data1 = np.asarray(data1, dtype=np.float64)
    data2 = np.asarray(data2, dtype=np.float64)

    if data1.ndim not in [1, 2]:
        raise ValueError(f"data1 must be 1D or 2D, got shape {data1.shape}")
    if data2.ndim not in [1, 2]:
        raise ValueError(f"data2 must be 1D or 2D, got shape {data2.shape}")
    tail = validate_tail_parameter(tail)
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

    # Decide execution mode based on device parameter
    if device == "cpu" or device is None:
        # CPU modes
        if device is None:
            # Single-threaded NumPy
            rng = check_random_state(random_state)

            # Select correlation function
            corr_func = _select_corr_func(metric)

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
                "device": None,
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
            n_permute=n_permute,
            metric=metric,
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
    return _correlation_permutation_gpu_batched(
        data1,
        data2,
        n_permute=n_permute,
        metric=metric,
        tail=tail,
        return_null=return_null,
        backend=backend_obj,
        max_gpu_memory_gb=max_gpu_memory_gb,
        random_state=rng,
        single_feature=single_feature,
        progress_bar=progress_bar,
    )
