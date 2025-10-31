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
    import importlib.util

    return importlib.util.find_spec("torch") is not None


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
    ridge_sklearn = Ridge(alpha=alpha, fit_intercept=False, solver="svd")
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
    backend_cpu = Backend("numpy")
    beta_cpu = ridge_svd(X, y, alpha=alpha, backend=backend_cpu)

    # GPU
    backend_gpu = Backend("torch")
    beta_gpu = ridge_svd(X, y, alpha=alpha, backend=backend_gpu)

    np.testing.assert_allclose(beta_gpu, beta_cpu, rtol=1e-4)


# ============================================================================
# Ridge Cross-Validation
# ============================================================================


def test_ridge_cv_basic():
    """Ridge CV should select alpha and return results"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    result = ridge_cv(X, y, alphas=[0.1, 1.0, 10.0], cv=3, backend="numpy")

    # Check result structure
    assert "alpha" in result
    assert "coef" in result
    assert "cv_scores" in result
    assert "backend" in result

    # Check selected alpha
    assert result["alpha"] in [0.1, 1.0, 10.0]

    # Check coefficients shape
    assert result["coef"].shape == (50,)

    # Check CV scores shape: (n_folds, n_alphas, n_targets)
    assert result["cv_scores"].shape == (3, 3, 1)


def test_ridge_cv_multi_target():
    """Ridge CV should handle multiple targets"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    Y = np.random.randn(100, 5).astype(np.float32)

    result = ridge_cv(X, Y, alphas=[0.1, 1.0, 10.0], cv=3, backend="numpy")

    # Check coefficients shape
    assert result["coef"].shape == (50, 5)

    # Check CV scores shape
    assert result["cv_scores"].shape == (3, 3, 5)


def test_ridge_cv_default_alphas():
    """Ridge CV should use default alphas if not provided"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    result = ridge_cv(X, y, cv=3, backend="numpy")

    # Should have selected some alpha
    assert result["alpha"] > 0
    assert result["coef"].shape == (50,)


def test_ridge_cv_reproducibility():
    """Ridge CV should give reproducible results with same seed"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    result1 = ridge_cv(X, y, alphas=[0.1, 1.0], cv=3, backend="numpy")

    np.random.seed(42)
    X2 = np.random.randn(100, 50).astype(np.float32)
    y2 = np.random.randn(100).astype(np.float32)
    result2 = ridge_cv(X2, y2, alphas=[0.1, 1.0], cv=3, backend="numpy")

    assert result1["alpha"] == result2["alpha"]
    np.testing.assert_allclose(result1["coef"], result2["coef"], rtol=1e-5)


@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_ridge_cv_cpu_gpu_equivalence():
    """CPU and GPU CV should give same results"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    alphas = [0.1, 1.0, 10.0]

    result_cpu = ridge_cv(X, y, alphas=alphas, cv=3, backend="numpy")
    result_gpu = ridge_cv(X, y, alphas=alphas, cv=3, backend="torch")

    assert result_cpu["alpha"] == result_gpu["alpha"]
    np.testing.assert_allclose(result_cpu["coef"], result_gpu["coef"], rtol=1e-4)


# ============================================================================
# Performance & Large Datasets
# ============================================================================


def test_large_dataset_completion():
    """Ridge CV should complete on neuroimaging-sized datasets"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)
    # Neuroimaging-sized problem
    X = np.random.randn(300, 10000).astype(np.float32)
    y = np.random.randn(300).astype(np.float32)

    result = ridge_cv(X, y, alphas=[0.1, 1.0, 10.0], cv=3, backend="auto")

    assert result["coef"].shape == (10000,)
    assert result["alpha"] > 0


@pytest.mark.tier2
def test_auto_backend_selection():
    """Auto backend should select appropriately based on problem size"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)

    # Small problem
    X_small = np.random.randn(100, 1000).astype(np.float32)
    y_small = np.random.randn(100).astype(np.float32)
    result_small = ridge_cv(X_small, y_small, cv=3, backend="auto")
    assert result_small["coef"].shape == (1000,)

    # Large problem
    X_large = np.random.randn(300, 50000).astype(np.float32)
    y_large = np.random.randn(300).astype(np.float32)
    result_large = ridge_cv(X_large, y_large, alphas=[1.0, 10.0], cv=3, backend="auto")
    assert result_large["coef"].shape == (50000,)
    assert result_large["backend"] in ["numpy", "torch-cpu", "torch-cuda", "torch-mps"]


# ============================================================================
# Edge Cases
# ============================================================================


def test_single_alpha():
    """Should work with single alpha value"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    result = ridge_cv(X, y, alphas=[1.0], cv=3, backend="numpy")
    assert result["alpha"] == 1.0


def test_perfect_fit_case():
    """Should handle perfect fit scenarios"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    beta_true = np.random.randn(50).astype(np.float32)
    y = X @ beta_true  # Perfect linear relationship

    result = ridge_cv(X, y, alphas=[1e-6, 0.1, 1.0], cv=3, backend="numpy")

    # Should prefer small alpha for perfect fit
    assert result["alpha"] <= 0.1


def test_noisy_data():
    """Should handle noisy data appropriately"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    beta_true = np.random.randn(50).astype(np.float32)
    y = X @ beta_true + 0.5 * np.random.randn(100).astype(np.float32)

    result = ridge_cv(X, y, alphas=[0.01, 0.1, 1.0, 10.0], cv=3, backend="numpy")

    # Should select some regularization
    assert 0.01 <= result["alpha"] <= 10.0
