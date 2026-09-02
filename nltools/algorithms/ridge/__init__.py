"""Ridge regression algorithms and utilities.

Ridge regression solvers with cross-validated alpha selection (per target or
global), memory-efficient batching for large problems, optional GPU acceleration
(roughly 10-100× faster on large datasets), and banded ridge for multiple
feature spaces. `solve_ridge_cv` and `solve_banded_ridge_cv` are the main entry
points; `ridge_svd` and `ridge_cv` are simpler single-alpha and CV solvers.
`BrainData.fit(model='ridge')` wraps these for brain data.

Examples:
    ```python
    import numpy as np
    from nltools.algorithms.ridge import solve_ridge_cv

    X = np.random.randn(100, 50)
    Y = np.random.randn(100, 10)
    result = solve_ridge_cv(X, Y, alphas=[0.1, 1.0, 10.0])
    ```
"""

# Core solvers (new GPU-enabled API)
from .solvers import (
    solve_ridge_cv,
    solve_banded_ridge_cv,
    cross_val_predict_ridge,
)

# Utilities (internal but useful)
from .utils import (
    _decompose_ridge,
    _r2_score,
    generate_dirichlet_samples,
)

# Backward compatibility (legacy API)
from .core import (
    ridge_svd,
    ridge_cv,
)

__all__ = [
    "_decompose_ridge",
    "_r2_score",
    "cross_val_predict_ridge",
    "generate_dirichlet_samples",
    "ridge_cv",
    "ridge_svd",
    "solve_banded_ridge_cv",
    "solve_ridge_cv",
]
