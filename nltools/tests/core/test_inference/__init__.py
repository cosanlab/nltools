"""
Shared constants, fixtures, and utilities for inference module tests.

This module provides:
- Test tolerance constants (TOLERANCE_EXACT, TOLERANCE_STATS_PVALUE, etc.)
- Shared fixtures (sample_data, backends) - moved to conftest.py
- Common imports used across test files
"""

import pytest
import numpy as np

from nltools.backends import check_gpu_available

# ============================================================================
# Test Constants - DO NOT MODIFY without updating docstring above
# ============================================================================

# Tolerance for backend consistency (NumPy vs PyTorch with same seed)
# These should be EXACT matches (same algorithm, only precision differs)
TOLERANCE_EXACT = 1e-5

# Tolerance for deterministic values when comparing to stats.py
# (mean, correlation, etc. - these are computed identically)
TOLERANCE_STATS_DETERMINISTIC = 1e-5

# Tolerance for P-values when comparing to stats.py
# One-sample: 0.000% error (uses identical _generate_sign_flips pattern)
# Two-sample/Correlation: ~1-2% error (prioritizes cross-backend determinism over stats.py exact match)
# Trade-off: Cross-backend consistency (0.000%) > backward compatibility (~1-2%)
TOLERANCE_STATS_PVALUE = 0.02  # 2% relative error acceptable

# One-tailed tests: Same tolerance as two-tailed
# One-sample achieves 0.000%, two-sample ~1-2% (same patterns as above)
TOLERANCE_STATS_PVALUE_ONE_TAILED = 0.02  # 2% relative error acceptable

# Special case: Time-series methods have higher variance vs stats.py
# Root cause: Same as two-sample/correlation (independent RandomState vs shared RNG state)
# circle_shift: ~32% actual variance (shift amounts determined by RNG sequence)
# phase_randomize: ~3% actual variance (FFT operations more numerically stable)
# Both implementations are fully deterministic (same seed → identical results)
TOLERANCE_STATS_PVALUE_CIRCLE_SHIFT = (
    0.4  # 40% relative error (accommodates ~32% actual)
)

# phase_randomize: Lower variance due to FFT numerical stability
TOLERANCE_STATS_PVALUE_PHASE_RANDOMIZE = (
    0.05  # 5% relative error (accommodates ~3% actual)
)

# Tolerance for GPU vs CPU comparisons (float32 vs float64)
TOLERANCE_GPU_VALUE = 1e-3  # 0.1% error for computed values
TOLERANCE_GPU_PVALUE = 5e-3  # 0.5% error for P-values (more FP error)

# Number of permutations for different test types
N_PERMUTE_BACKEND = 100  # Fast checks for backend consistency
N_PERMUTE_STATS_COMPARISON = 1000  # Stable comparison with stats.py
