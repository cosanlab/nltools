"""Random-state utilities for deterministic parallel resampling.

The bootstrap and permutation tests pre-generate one seed per iteration from a
single `random_state`, then give each parallel worker its own `RandomState`
built from its seed. Results are therefore identical for any `n_jobs` and
across CPU and GPU execution.

Examples:
    ```python
    from nltools.algorithms.inference.random import _generate_seeds

    seeds = _generate_seeds(100, random_state=42)
    # Hand one seed to each parallel worker for deterministic results
    ```
"""

import numpy as np
from sklearn.utils import check_random_state


def _generate_seeds(n_permute: int, random_state: int | None = None) -> np.ndarray:
    """Generate one random seed per permutation or bootstrap iteration.

    Args:
        n_permute (int): Number of iterations to seed.
        random_state (int | None): Seed for the seed generator. Defaults to None.

    Returns:
        np.ndarray: Integer seeds, shape (n_permute,).

    Examples:
        ```python
        seeds = _generate_seeds(100, random_state=42)
        seeds.shape  # (100,)
        isinstance(seeds[0], (int, np.integer))  # True
        ```
    """
    rng = check_random_state(random_state)
    MAX_INT = 2**31 - 1
    seeds = rng.randint(MAX_INT, size=n_permute)
    return seeds


def _generate_sign_flips(
    n_permute: int,
    n_samples: int,
    random_state: int | None = None,
) -> np.ndarray:
    """Generate the random sign-flip matrix for one-sample permutation tests.

    Each row is one permutation: every sample is multiplied by +1 or -1 to build
    the null distribution. Each permutation draws from an independent
    `RandomState` seeded by `_generate_seeds`, so the matrix is reproducible for
    any degree of parallelism.

    Args:
        n_permute (int): Number of permutations.
        n_samples (int): Number of samples in the dataset.
        random_state (int | None): Seed for reproducibility. Defaults to None.

    Returns:
        np.ndarray: Matrix of +1 and -1 values, shape (n_permute, n_samples). A
            NumPy array; callers move it to the GPU when needed.

    Examples:
        ```python
        sign_flips = _generate_sign_flips(n_permute=100, n_samples=30, random_state=42)
        sign_flips.shape  # → (100, 30)
        np.all(np.isin(sign_flips, [-1, 1]))  # → True
        ```
    """
    seeds = _generate_seeds(n_permute, random_state=random_state)

    # Generate sign-flips using independent RNG per permutation
    # This matches stats._permute_sign behavior exactly
    sign_flips = np.array(
        [
            np.random.RandomState(seeds[i]).choice([1, -1], n_samples)
            for i in range(n_permute)
        ]
    )

    return sign_flips


def _generate_bootstrap_indices(
    n_samples: int,
    n_bootstrap: int,
    random_state: int | None = None,
) -> np.ndarray:
    """Generate bootstrap resampling indices deterministically.

    Each bootstrap draw uses an independent `RandomState` seeded by
    `_generate_seeds`, the same scheme as the permutation tests.

    Args:
        n_samples (int): Number of samples in the original dataset.
        n_bootstrap (int): Number of bootstrap iterations.
        random_state (int | None): Seed for reproducibility. Defaults to None.

    Returns:
        np.ndarray: Indices sampled with replacement from `[0, n_samples)`,
            shape (n_bootstrap, n_samples); repeats within a row are expected.

    Examples:
        ```python
        indices = _generate_bootstrap_indices(100, 1000, random_state=42)
        indices.shape  # → (1000, 100)
        indices[0]  # → array([23, 45, 23, 67, ...])  one bootstrap sample
        ```
    """
    seeds = _generate_seeds(n_bootstrap, random_state=random_state)

    # Each bootstrap gets independent RandomState
    indices = np.array(
        [
            np.random.RandomState(seeds[i]).choice(n_samples, n_samples, replace=True)
            for i in range(n_bootstrap)
        ]
    )

    return indices
