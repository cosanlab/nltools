"""GPU-accelerated statistical inference for neuroimaging.

This module provides fast permutation testing and bootstrap resampling using
optional GPU acceleration via PyTorch. When GPU is unavailable, efficiently
uses CPU parallelization.

Inspired by BROCCOLI's GPU permutation testing (Eklund et al. 2014).

Key Features:
    - 10-100× speedup for permutation tests with GPU
    - Efficient CPU parallelization when GPU unavailable
    - Transparent CPU/GPU support via Backend abstraction
    - Drop-in replacement for nltools.stats functions

Examples:
    >>> import numpy as np
    >>> from nltools.algorithms.inference import one_sample_permutation_test

    >>> # Simple one-sample test
    >>> data = np.random.randn(30)  # 30 subjects
    >>> result = one_sample_permutation_test(data, n_permute=5000)
    >>> print(f"p-value: {result['p']:.3f}")

    >>> # Voxel-wise test with GPU acceleration
    >>> data = np.random.randn(30, 50000)  # 30 subjects, 50K voxels
    >>> result = one_sample_permutation_test(data, n_permute=10000, parallel='gpu')
    >>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")

Performance:
    - CPU (NumPy): Good for small problems (< 5K permutations)
    - GPU (PyTorch): Excellent for large problems (> 5K permutations)
    - CPU Parallel (joblib): Efficient fallback when GPU unavailable
    - Select with parallel='cpu' | 'gpu' | None (no 'auto' selector)

References:
    Eklund, A., Dufort, P., Villani, M., & LaConte, S. M. (2014).
    BROCCOLI: Software for fast fMRI analysis on many-core CPUs and GPUs.
    Frontiers in Neuroinformatics, 8, 24.

Notes:
    This module is part of the "functional core" of nltools. For integration
    with BrainData objects, see nltools.data.brain_data.
"""

# Import public API functions
from .one_sample import one_sample_permutation_test
from .two_sample import two_sample_permutation_test
from .correlation import correlation_permutation_test
from .timeseries import (
    circle_shift,
    phase_randomize,
    timeseries_correlation_permutation_test,
)
from .matrix import (
    matrix_permutation_test,
    double_center,
    u_center,
    distance_correlation,
)
from .isc import isc_permutation_test, isc_group_permutation_test

# Import utility functions (for testing and internal use)
from .utils import _generate_sign_flips, _compute_pvalue, _auto_batch_size

# Import bootstrap utilities
from .bootstrap import OnlineBootstrapStats

# Define public exports
__all__ = [
    "OnlineBootstrapStats",
    "_auto_batch_size",
    "_compute_pvalue",
    # Private functions (exported for testing)
    "_generate_sign_flips",
    "circle_shift",
    "correlation_permutation_test",
    "distance_correlation",
    "double_center",
    "isc_group_permutation_test",
    "isc_permutation_test",
    "matrix_permutation_test",
    "one_sample_permutation_test",
    "phase_randomize",
    "timeseries_correlation_permutation_test",
    "two_sample_permutation_test",
    "u_center",
]
