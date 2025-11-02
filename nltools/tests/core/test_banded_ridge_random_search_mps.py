"""Test MPS compatibility for random search banded ridge."""

import numpy as np
import pytest
from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv


@pytest.mark.skipif(
    not hasattr(__import__("torch"), "backends")
    or not hasattr(__import__("torch").backends, "mps")
    or not __import__("torch").backends.mps.is_available(),
    reason="MPS backend not available",
)
def test_solve_banded_ridge_cv_mps():
    """Test random search banded ridge with MPS backend."""
    np.random.seed(42)

    n_samples = 50
    n_features1 = 20
    n_features2 = 15
    n_targets = 5

    X1 = np.random.randn(n_samples, n_features1).astype(np.float32)
    X2 = np.random.randn(n_samples, n_features2).astype(np.float32)
    Y = np.random.randn(n_samples, n_targets).astype(np.float32)

    # Test with torch backend (will use MPS if available)
    result = solve_banded_ridge_cv(
        [X1, X2],
        Y,
        n_iter=3,  # Small for speed
        alphas=[0.1, 1.0],
        cv=3,
        parallel="cpu",  # Use CPU parallelization (torch backend handles MPS internally)
        return_weights=True,
        progress_bar=False,
        random_state=42,
    )
    
    deltas = result['deltas']
    coefs = result['coefs']
    cv_scores = result['cv_scores']

    # Check outputs
    assert deltas.shape == (2, n_targets)
    assert coefs.shape == (n_features1 + n_features2, n_targets)
    assert cv_scores.shape == (3, n_targets)

    # Check that results are finite
    assert np.all(np.isfinite(deltas))
    assert np.all(np.isfinite(coefs))
    assert np.all(np.isfinite(cv_scores))
