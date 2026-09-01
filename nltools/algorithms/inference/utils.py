"""Utility functions for permutation testing.

This module contains shared helper functions used across different
permutation test implementations.
"""

import numpy as np
from ...utils import _NullProgressBar, make_progress_bar, maybe_tqdm  # noqa: F401
from ..random import generate_sign_flips as _generate_sign_flips_from_random


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


def _normalize_tail_internal(tail: int | str) -> str:
    """Normalize a tail value to the internal 'two'/'upper'/'lower' form.

    Accepts BOTH the public v0.6.0 vocabulary (2|'two', 1|'one') and the
    internal directional forms ('upper', 'lower', -1) that forced-tail call
    sites use directly. Public entry points must validate with the strict
    `validate_tail_parameter` first — this permissive form exists only so
    `_compute_pvalue` can serve both layers.
    """
    if tail == 2 or tail == "two":
        return "two"
    if tail == 1 or tail == "one" or tail == "upper":
        return "upper"
    if tail == -1 or tail == "lower":
        return "lower"
    raise ValueError(
        f"tail must be 2|'two', 1|'one' (or internal 'upper'/'lower'/-1), got {tail!r}"
    )


def _signed_z_from_p(t_like_arr, p_arr, tail_internal: str = "two") -> np.ndarray:
    """Compute a signed z-score map from a p-value map.

    The single z-from-p conversion shared by `BrainData.ttest`,
    `BrainCollection.ttest`/`ttest2`, and the GLM-bundle contrast reader —
    the clipping policy below must live in exactly one place.

    Two-tailed p: ``|z| = norm.isf(p/2)`` so that p=0.05 → |z|≈1.96, matching
    nilearn's ``output_type='z_score'`` convention, with the sign copied from
    the accompanying statistic. One-tailed (upper) p: ``z = norm.isf(p)`` —
    a one-sided p already encodes direction, so no sign copy is needed.

    p is clipped to the open interval (0, 1) so z stays finite in BOTH
    directions: the lower bound is the smallest positive normal float64
    (z ≈ +37.7), the upper bound the largest float64 below 1 (z ≈ -8.3).
    The bounds look asymmetric because float64 resolves p near 0 far more
    finely than near 1; both sit beyond any meaningful statistical
    resolution. Without the upper clip, ``p == 1.0`` — reachable on the
    one-tailed path, e.g. ``t.sf(-45, 29) == 1.0`` — maps to ``-inf`` and
    poisons downstream percentiles and plotting.

    Args:
        t_like_arr: Statistic supplying the sign (t map or similar).
        p_arr: P-value map (two-tailed, or one-tailed upper).
        tail_internal: ``'two'`` (default) or ``'upper'``.

    Returns:
        np.ndarray: Signed z map, finite everywhere.
    """
    from scipy.stats import norm

    p_clipped = np.clip(
        np.asarray(p_arr, dtype=np.float64),
        np.finfo(np.float64).tiny,
        np.nextafter(1.0, 0.0),
    )
    if tail_internal == "upper":
        return norm.isf(p_clipped)
    z_abs = norm.isf(p_clipped / 2.0)
    return np.sign(np.asarray(t_like_arr)) * z_abs


def _compute_pvalue(
    obs_stat: np.ndarray,
    null_dist: np.ndarray,
    tail: int | str = 2,
) -> np.ndarray:
    """Calculate p-values from observed statistic and null distribution.

    Computes the proportion of null distribution values as extreme or more
    extreme than the observed statistic, using the correction factor approach
    the original nltools.algorithms._calc_pvalue.

    Args:
        obs_stat (np.ndarray): Observed statistic(s)
            - shape () for scalar
            - shape (n_features,) for multi-feature
        null_dist (np.ndarray): Null distribution from permutations
            - shape (n_permute,) for single feature
            - shape (n_permute, n_features) for multi-feature
        tail (int | str): Test type
            - 'two' or 2: Two-tailed test (|obs| > |null|)
            - 'one' or 1 (or internal 'upper'): One-tailed upper (null >= obs,
              for positive effects)
            - internal 'lower' or -1: One-tailed lower (null <= obs) — reserved
              for forced-tail call sites; not part of the public vocabulary

    Returns:
        np.ndarray: P-value(s) with same shape as obs_stat

    Examples:
        >>> obs_stat = np.array([2.5])
        >>> null_dist = np.random.randn(1000, 1)
        >>> p = _compute_pvalue(obs_stat, null_dist, tail='two')
        >>> 0 < p <= 1
        True

        # For multiple comparisons, use explicit direction:
        >>> p_upper = _compute_pvalue(obs_stat, null_dist, tail='upper')
        >>> p_lower = _compute_pvalue(obs_stat, null_dist, tail='lower')

    Notes:
        - Uses correction factor: (count + 1) / (n_permute + 1)
        - This prevents p-value = 0 and accounts for observed value
        - Minimum p-value is 1/(n_permute + 1)
        - For two-tailed tests, uses absolute values
        - The fixed per-tail direction keeps MCP correction (FDR, Bonferroni)
          valid across tests. See GH #315.
    """
    tail_normalized = _normalize_tail_internal(tail)

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

    if tail_normalized == "upper":
        # One-tailed upper: count how many null >= observed
        # Tests for positive effects (H1: statistic > 0)
        numer = np.sum(null_dist >= obs_stat, axis=0) + 1.0
    elif tail_normalized == "lower":
        # One-tailed lower: count how many null <= observed
        # Tests for negative effects (H1: statistic < 0)
        numer = np.sum(null_dist <= obs_stat, axis=0) + 1.0
    else:  # tail_normalized == "two"
        # Two-tailed: count how many |null| >= |observed|
        numer = np.sum(np.abs(null_dist) >= np.abs(obs_stat), axis=0) + 1.0

    p_values = numer / denom

    return p_values


# Re-export from shared random utilities for backward compatibility


def _auto_batch_size(
    n_permute: int,
    n_samples: int,
    n_features: int,
    max_memory_gb: float | None = None,
    backend=None,
) -> tuple[int, int]:
    """Determine the GPU permutation batch size for a memory budget.

    Thin adapter over the core layer in `nltools.algorithms.backends`:
    supplies the permutation-test working-set estimate (the ``data_perm``
    tensor, ``(batch_size, n_samples, n_features)`` float32) and the
    100-permutation dispatch floor; the budget/clamp policy lives in
    `auto_batch_size`.

    Args:
        n_permute (int): Total number of permutations to compute
        n_samples (int): Number of samples in dataset
        n_features (int): Number of features/voxels
        max_memory_gb (float, optional): Explicit memory budget in GB. None
            (default) measures the device via `device_memory_budget`.
        backend: Resolved `Backend` the work runs on (used only to measure
            the budget when ``max_memory_gb`` is None).

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
    """
    from nltools.algorithms.backends import auto_batch_size, device_memory_budget

    budget_gb = device_memory_budget(
        backend, max_gpu_memory_gb=max_memory_gb, cap_for_batching=True
    )
    bytes_per_perm = n_samples * n_features * 4  # float32 data_perm row
    return auto_batch_size(
        n_permute, bytes_per_perm, budget_gb=budget_gb, min_batch=100
    )
