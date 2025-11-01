"""
Utility functions for permutation testing.

This module contains shared helper functions used across different
permutation test implementations.
"""

import numpy as np
from typing import Optional
from sklearn.utils import check_random_state


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

    This implementation matches the RNG pattern from nltools.stats.one_sample_permutation
    for exact backward compatibility: each permutation gets an independent RandomState
    derived from a unique seed.

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
        - Each permutation uses independent RandomState for stats.py compatibility
        - Values are uniformly sampled from {+1, -1} (matching stats.py order)
        - Returns NumPy array (device transfer handled by caller)
        - Memory cost: n_permute × n_samples × 1 byte (negligible for typical use)
    """
    rng = check_random_state(random_state)

    # Generate unique seed for each permutation (matches stats.py pattern)
    MAX_INT = 2**31 - 1
    seeds = rng.randint(MAX_INT, size=n_permute)

    # Generate sign-flips using independent RNG per permutation
    # This matches stats._permute_sign behavior exactly
    sign_flips = np.array(
        [
            np.random.RandomState(seeds[i]).choice([1, -1], n_samples)
            for i in range(n_permute)
        ]
    )

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


def _generate_bootstrap_indices(
    n_samples: int,
    n_bootstrap: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generate bootstrap indices deterministically for resampling.

    Uses the same pattern as permutation tests: pre-generate seeds for
    reproducible parallelization.

    Parameters
    ----------
    n_samples : int
        Number of samples in original dataset
    n_bootstrap : int
        Number of bootstrap iterations
    random_state : int, optional
        Random seed for reproducibility

    Returns
    -------
    np.ndarray
        Bootstrap indices with shape (n_bootstrap, n_samples)
        Each row contains indices sampled with replacement from [0, n_samples)

    Examples
    --------
    >>> indices = _generate_bootstrap_indices(100, 1000, random_state=42)
    >>> indices.shape
    (1000, 100)
    >>> indices[0]  # First bootstrap sample indices
    array([23, 45, 23, 67, ...])  # Some repeated (sampling with replacement)
    """
    rng = check_random_state(random_state)
    MAX_INT = 2**31 - 1
    seeds = rng.randint(MAX_INT, size=n_bootstrap)

    # Each bootstrap gets independent RandomState
    indices = np.array(
        [
            np.random.RandomState(seeds[i]).choice(n_samples, n_samples, replace=True)
            for i in range(n_bootstrap)
        ]
    )

    return indices


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
