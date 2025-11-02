"""Ridge regression algorithms and utilities.

This package contains ridge regression implementations and backend abstractions.

Features:
- Backend abstraction (NumPy, PyTorch CPU, PyTorch GPU)
- Cross-validation with per-target or global alpha selection
- Memory-efficient batching for large-scale problems
- GPU acceleration (10-100× speedup on large datasets)
- Banded ridge for multiple feature spaces

Quick Start:
    >>> from nltools.algorithms.ridge import solve_ridge_cv, set_backend
    >>> set_backend("auto")  # Use GPU if available
    >>> X = np.random.randn(100, 50)
    >>> Y = np.random.randn(100, 10)
    >>> alphas, coefs, scores = solve_ridge_cv(X, Y, alphas=[0.1, 1.0, 10.0])
"""

# Backend management
from .backends import (
    set_backend,
    get_backend,
    ALL_BACKENDS,
)

# Core solvers (new GPU-enabled API)
from .solvers import (
    solve_ridge_cv,
    solve_banded_ridge_cv,
)

# Utilities (internal but useful)
from .utils import (
    _decompose_ridge,
    _r2_score,
    generate_dirichlet_samples,
)

# Backward compatibility (legacy API)
from ._core import (
    ridge_svd,
    ridge_cv,
)

__all__ = [
    # Backend management
    "set_backend",
    "get_backend",
    "ALL_BACKENDS",
    # New solvers (GPU-enabled)
    "solve_ridge_cv",
    "solve_banded_ridge_cv",
    # Legacy solvers (backward compatible)
    "ridge_svd",
    "ridge_cv",
    # Utilities (advanced usage)
    "_decompose_ridge",
    "_r2_score",
    "generate_dirichlet_samples",
]
