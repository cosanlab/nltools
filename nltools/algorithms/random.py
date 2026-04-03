"""Shared random state utilities for algorithms module.

This module provides common random state handling to ensure consistent
random number generation across the algorithms module.

Key features:
    - Deterministic parallelization: Pre-generates seeds for reproducible parallel execution
    - Consistent RNG patterns: Matches stats.py patterns for backward compatibility
    - Thread-safe design: Each parallel worker gets independent RandomState

Usage:
    These utilities are used in bootstrap and permutation tests to ensure
    deterministic behavior when using parallel processing.

    Example:
        >>> from nltools.algorithms.random import generate_seeds
        >>> seeds = generate_seeds(100, random_state=42)
        >>> # Use seeds in parallel workers for deterministic results
"""

from typing import Optional
import numpy as np
from sklearn.utils import check_random_state


def get_random_state(random_state: Optional[int] = None):
    """Get RandomState instance from seed.

    Args:
        random_state: Random seed (int, RandomState, or None)

    Returns:
        RandomState instance

    Note:
        Uses sklearn.utils.check_random_state for consistency
    """
    return check_random_state(random_state)


def generate_seeds(n_permute: int, random_state: Optional[int] = None) -> np.ndarray:
    """Generate random seeds for deterministic parallelization.

    Pre-generates unique seeds for each permutation/bootstrap iteration
    to ensure deterministic behavior across parallel workers.

    Args:
        n_permute: Number of permutations/bootstrap iterations
        random_state: Random seed for reproducibility

    Returns:
        Array of seeds with shape (n_permute,)

    Examples:
        >>> seeds = generate_seeds(100, random_state=42)
        >>> seeds.shape
        (100,)
        >>> isinstance(seeds[0], (int, np.integer))
        True
    """
    rng = check_random_state(random_state)
    MAX_INT = 2**31 - 1
    seeds = rng.randint(MAX_INT, size=n_permute)
    return seeds


def generate_sign_flips(
    n_permute: int,
    n_samples: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Generate random sign-flip matrix for one-sample permutation tests.

    Creates a matrix of random +1/-1 values for sign-flipping permutation tests.
    Each row represents one permutation, where each sample is randomly multiplied
    by +1 or -1 to create the null distribution.

    This implementation matches the RNG pattern from nltools.stats.one_sample_permutation
    for exact backward compatibility: each permutation gets an independent RandomState
    derived from a unique seed.

    Args:
        n_permute: Number of permutations to generate
        n_samples: Number of samples in the dataset
        random_state: Random seed for reproducibility

    Returns:
        Sign-flip matrix of shape (n_permute, n_samples)
            containing only +1 and -1 values

    Examples:
        >>> sign_flips = generate_sign_flips(n_permute=100, n_samples=30, random_state=42)
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
    seeds = generate_seeds(n_permute, random_state=random_state)

    # Generate sign-flips using independent RNG per permutation
    # This matches stats._permute_sign behavior exactly
    sign_flips = np.array(
        [
            np.random.RandomState(seeds[i]).choice([1, -1], n_samples)
            for i in range(n_permute)
        ]
    )

    return sign_flips


def generate_bootstrap_indices(
    n_samples: int,
    n_bootstrap: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Generate bootstrap indices deterministically for resampling.

    Uses the same pattern as permutation tests: pre-generate seeds for
    reproducible parallelization.

    Args:
        n_samples: Number of samples in original dataset.
        n_bootstrap: Number of bootstrap iterations.
        random_state: Random seed for reproducibility. Defaults to None.

    Returns:
        Bootstrap indices with shape (n_bootstrap, n_samples).
            Each row contains indices sampled with replacement from [0, n_samples).

    Examples:
        >>> indices = generate_bootstrap_indices(100, 1000, random_state=42)
        >>> indices.shape
        (1000, 100)
        >>> indices[0]  # First bootstrap sample indices
        array([23, 45, 23, 67, ...])  # Some repeated (sampling with replacement)

    Notes:
        - Uses same seed generation pattern as permutation tests for consistency
        - Each bootstrap iteration gets independent RandomState for reproducibility
        - Sampling is with replacement (some indices may repeat)
    """
    seeds = generate_seeds(n_bootstrap, random_state=random_state)

    # Each bootstrap gets independent RandomState
    indices = np.array(
        [
            np.random.RandomState(seeds[i]).choice(n_samples, n_samples, replace=True)
            for i in range(n_bootstrap)
        ]
    )

    return indices
