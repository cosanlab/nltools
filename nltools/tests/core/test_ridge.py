"""
Test ridge regression algorithms.

Part of functional core - tests SVD-based ridge regression inspired by himalaya.
Following model-spec.md Phase 2 implementation.
"""

import numpy as np
import pytest
from nltools.algorithms.ridge import ridge_svd
from nltools.backends import Backend


# ============================================================================
# Helper Functions
# ============================================================================


def _torch_available():
    """Check if PyTorch is installed"""
    try:
        import torch
        return True
    except ImportError:
        return False


# ============================================================================
# Ridge SVD Solver
# ============================================================================


def test_ridge_svd_single_target():
    """Ridge SVD should solve single-target regression"""
    np.random.seed(42)
    n_samples, n_features = 100, 50

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    alpha = 1.0

    beta = ridge_svd(X, y, alpha=alpha)

    # Check shape
    assert beta.shape == (n_features,)

    # Verify it reduces to OLS when alpha≈0
    beta_ols = ridge_svd(X, y, alpha=1e-10)
    beta_expected = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(beta_ols, beta_expected, rtol=1e-3)


def test_ridge_svd_multi_target():
    """Ridge SVD should handle multiple targets"""
    np.random.seed(42)
    n_samples, n_features, n_targets = 100, 50, 5

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    Y = np.random.randn(n_samples, n_targets).astype(np.float32)
    alpha = 1.0

    beta = ridge_svd(X, Y, alpha=alpha)

    # Check shape
    assert beta.shape == (n_features, n_targets)

    # Each column should solve the corresponding target
    for i in range(n_targets):
        beta_single = ridge_svd(X, Y[:, i], alpha=alpha)
        np.testing.assert_allclose(beta[:, i], beta_single, rtol=1e-4)


def test_ridge_vs_sklearn():
    """Ridge SVD should match sklearn Ridge"""
    from sklearn.linear_model import Ridge

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    alpha = 1.0

    # Our implementation
    beta_ours = ridge_svd(X, y, alpha=alpha)

    # sklearn
    ridge_sklearn = Ridge(alpha=alpha, fit_intercept=False, solver='svd')
    ridge_sklearn.fit(X, y)
    beta_sklearn = ridge_sklearn.coef_

    np.testing.assert_allclose(beta_ours, beta_sklearn, rtol=1e-4)


def test_ridge_regularization_effect():
    """Higher alpha should shrink coefficients"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    beta_small = ridge_svd(X, y, alpha=0.1)
    beta_large = ridge_svd(X, y, alpha=10.0)

    # Higher alpha should give smaller coefficients
    assert np.linalg.norm(beta_large) < np.linalg.norm(beta_small)


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_ridge_cpu_gpu_equivalence():
    """CPU and GPU should give same results"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    alpha = 1.0

    # CPU
    backend_cpu = Backend('numpy')
    beta_cpu = ridge_svd(X, y, alpha=alpha, backend=backend_cpu)

    # GPU
    backend_gpu = Backend('torch')
    beta_gpu = ridge_svd(X, y, alpha=alpha, backend=backend_gpu)

    np.testing.assert_allclose(beta_gpu, beta_cpu, rtol=1e-4)


# ============================================================================
# Ridge Cross-Validation
# ============================================================================


# ============================================================================
# Performance & Large Datasets
# ============================================================================


# ============================================================================
# Edge Cases
# ============================================================================
