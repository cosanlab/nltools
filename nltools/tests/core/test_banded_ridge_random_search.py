"""Tests for banded ridge regression random search.

Tests the Himalaya-style random search implementation for banded ridge regression.
"""

import numpy as np
import pytest
from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv
from nltools.algorithms.ridge.utils import generate_dirichlet_samples


def test_generate_dirichlet_samples():
    """Test Dirichlet sampling utility."""
    gammas = generate_dirichlet_samples(
        10, 3, concentration=[0.1, 1.0], random_state=42
    )

    assert gammas.shape == (10, 3)
    assert np.allclose(gammas.sum(axis=1), 1.0)
    assert np.all(gammas >= 0)
    assert np.all(gammas <= 1)


def test_solve_banded_ridge_cv_basic():
    """Test basic functionality of random search banded ridge."""
    np.random.seed(42)

    # Generate synthetic data
    n_samples = 50
    n_features1 = 20
    n_features2 = 15
    n_targets = 5

    X1 = np.random.randn(n_samples, n_features1).astype(np.float32)
    X2 = np.random.randn(n_samples, n_features2).astype(np.float32)
    Y = np.random.randn(n_samples, n_targets).astype(np.float32)

    # Run with small n_iter for speed
    deltas, coefs, cv_scores = solve_banded_ridge_cv(
        [X1, X2],
        Y,
        n_iter=5,
        alphas=[0.1, 1.0, 10.0],
        cv=3,
        return_weights=True,
        progress_bar=False,
        random_state=42,
    )

    # Check outputs
    assert deltas.shape == (2, n_targets)  # 2 feature spaces
    assert coefs.shape == (n_features1 + n_features2, n_targets)
    assert cv_scores.shape == (5, n_targets)  # 5 iterations

    # Check that deltas are reasonable
    assert np.all(np.isfinite(deltas))

    # Check that coefficients are reasonable
    assert np.all(np.isfinite(coefs))


def test_solve_banded_ridge_cv_with_provided_gammas():
    """Test random search with provided gamma weights."""
    np.random.seed(42)

    n_samples = 50
    n_features1 = 20
    n_features2 = 15
    n_targets = 5

    X1 = np.random.randn(n_samples, n_features1).astype(np.float32)
    X2 = np.random.randn(n_samples, n_features2).astype(np.float32)
    Y = np.random.randn(n_samples, n_targets).astype(np.float32)

    # Provide custom gammas
    gammas = np.array(
        [
            [0.5, 0.5],  # Equal weights
            [0.8, 0.2],  # Favor first space
            [0.2, 0.8],  # Favor second space
        ]
    )

    deltas, coefs, cv_scores = solve_banded_ridge_cv(
        [X1, X2],
        Y,
        n_iter=gammas,
        alphas=[0.1, 1.0],
        cv=3,
        return_weights=True,
        progress_bar=False,
    )

    assert deltas.shape == (2, n_targets)
    assert coefs.shape == (n_features1 + n_features2, n_targets)
    assert cv_scores.shape == (3, n_targets)  # 3 provided gammas


def test_solve_banded_ridge_cv_no_weights():
    """Test random search without returning weights."""
    np.random.seed(42)

    n_samples = 50
    n_features1 = 20
    n_features2 = 15
    n_targets = 5

    X1 = np.random.randn(n_samples, n_features1).astype(np.float32)
    X2 = np.random.randn(n_samples, n_features2).astype(np.float32)
    Y = np.random.randn(n_samples, n_targets).astype(np.float32)

    deltas, coefs, cv_scores = solve_banded_ridge_cv(
        [X1, X2],
        Y,
        n_iter=5,
        alphas=[0.1, 1.0],
        cv=3,
        return_weights=False,
        progress_bar=False,
        random_state=42,
    )

    assert deltas.shape == (2, n_targets)
    assert coefs is None
    assert cv_scores.shape == (5, n_targets)


def test_solve_banded_ridge_cv_single_space():
    """Test random search with single feature space (should work like regular ridge)."""
    np.random.seed(42)

    n_samples = 50
    n_features = 20
    n_targets = 5

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    Y = np.random.randn(n_samples, n_targets).astype(np.float32)

    deltas, coefs, cv_scores = solve_banded_ridge_cv(
        [X],
        Y,
        n_iter=5,
        alphas=[0.1, 1.0],
        cv=3,
        return_weights=True,
        progress_bar=False,
        random_state=42,
    )

    assert deltas.shape == (1, n_targets)  # 1 feature space
    assert coefs.shape == (n_features, n_targets)
    assert cv_scores.shape == (5, n_targets)


def test_solve_banded_ridge_cv_with_intercept():
    """Test random search with intercept fitting."""
    np.random.seed(42)

    n_samples = 50
    n_features1 = 20
    n_features2 = 15
    n_targets = 5

    X1 = np.random.randn(n_samples, n_features1).astype(np.float32)
    X2 = np.random.randn(n_samples, n_features2).astype(np.float32)
    Y = np.random.randn(n_samples, n_targets).astype(np.float32)

    deltas, coefs, cv_scores, intercept = solve_banded_ridge_cv(
        [X1, X2],
        Y,
        n_iter=5,
        alphas=[0.1, 1.0],
        cv=3,
        fit_intercept=True,
        return_weights=True,
        progress_bar=False,
        random_state=42,
    )

    assert deltas.shape == (2, n_targets)
    assert coefs.shape == (n_features1 + n_features2, n_targets)
    assert intercept.shape == (n_targets,)
    assert np.all(np.isfinite(intercept))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
