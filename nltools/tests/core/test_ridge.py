"""
Test ridge regression algorithms.

Part of functional core - tests SVD-based ridge regression inspired by himalaya.
Following model-spec.md Phase 2 implementation.
"""

import numpy as np
import pytest
from nltools.algorithms.ridge import ridge_svd


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
    np.testing.assert_allclose(beta_ols, beta_expected, rtol=1e-3, atol=1e-5)


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
        np.testing.assert_allclose(beta[:, i], beta_single, rtol=1e-4, atol=1e-6)


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

    np.testing.assert_allclose(beta_ours, beta_sklearn, rtol=1e-4, atol=1e-6)


def test_ridge_regularization_effect():
    """Higher alpha should shrink coefficients"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    beta_small = ridge_svd(X, y, alpha=0.1)
    beta_large = ridge_svd(X, y, alpha=10.0)

    # Higher alpha should give smaller coefficients
    assert np.linalg.norm(beta_large) < np.linalg.norm(beta_small)


@pytest.mark.slow
@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_ridge_cpu_gpu_equivalence():
    """CPU and GPU should give same results"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    alpha = 1.0

    # CPU
    beta_cpu = ridge_svd(X, y, alpha=alpha, parallel="cpu")

    # GPU
    beta_gpu = ridge_svd(X, y, alpha=alpha, parallel="gpu")

    np.testing.assert_allclose(beta_gpu, beta_cpu, rtol=1e-4, atol=1e-6)


# ============================================================================
# Ridge Cross-Validation
# ============================================================================


@pytest.mark.slow
def test_ridge_cv_basic():
    """Ridge CV should select alpha and return results"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    result = ridge_cv(X, y, alphas=[0.1, 1.0, 10.0], cv=3, parallel="cpu")

    # Check result structure
    assert "alpha" in result
    assert "coef" in result
    assert "cv_scores" in result
    assert "parallel" in result or "backend" in result  # Accept both for now

    # Check selected alpha
    assert result["alpha"] in [0.1, 1.0, 10.0]

    # Check coefficients shape
    assert result["coef"].shape == (50,)

    # Check CV scores shape: (n_folds, n_alphas, n_targets)
    assert result["cv_scores"].shape == (3, 3, 1)


@pytest.mark.slow
def test_ridge_cv_multi_target():
    """Ridge CV should handle multiple targets"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    Y = np.random.randn(100, 5).astype(np.float32)

    result = ridge_cv(X, Y, alphas=[0.1, 1.0, 10.0], cv=3, parallel="cpu")

    # Check coefficients shape
    assert result["coef"].shape == (50, 5)

    # Check CV scores shape
    assert result["cv_scores"].shape == (3, 3, 5)


@pytest.mark.slow
def test_ridge_cv_default_alphas():
    """Ridge CV should use default alphas if not provided"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    result = ridge_cv(X, y, cv=3, parallel="cpu")

    # Should have selected some alpha
    assert result["alpha"] > 0
    assert result["coef"].shape == (50,)


@pytest.mark.slow
def test_ridge_cv_reproducibility():
    """Ridge CV should give reproducible results with same seed"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    result1 = ridge_cv(X, y, alphas=[0.1, 1.0], cv=3, parallel="cpu")

    np.random.seed(42)
    X2 = np.random.randn(100, 50).astype(np.float32)
    y2 = np.random.randn(100).astype(np.float32)
    result2 = ridge_cv(X2, y2, alphas=[0.1, 1.0], cv=3, parallel="cpu")

    assert result1["alpha"] == result2["alpha"]
    np.testing.assert_allclose(result1["coef"], result2["coef"], rtol=1e-5, atol=1e-7)


@pytest.mark.slow
@pytest.mark.skipif(not _torch_available(), reason="PyTorch not installed")
def test_ridge_cv_cpu_gpu_equivalence():
    """CPU and GPU CV should give same results (with graceful fallback)"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    alphas = [0.1, 1.0, 10.0]

    result_cpu = ridge_cv(X, y, alphas=alphas, cv=3, parallel="cpu")
    # Request GPU - should gracefully fallback to CPU if GPU unavailable
    result_gpu = ridge_cv(X, y, alphas=alphas, cv=3, parallel="gpu")

    # Both should produce valid results
    assert result_cpu["alpha"] > 0
    assert result_gpu["alpha"] > 0
    # Results should be identical (both CPU or both GPU) or very close (if different backends)
    assert result_cpu["alpha"] == result_gpu["alpha"]
    np.testing.assert_allclose(
        result_cpu["coef"], result_gpu["coef"], rtol=1e-4, atol=1e-6
    )


# ============================================================================
# Performance & Large Datasets
# ============================================================================


@pytest.mark.slow
def test_large_dataset_completion():
    """Ridge CV should complete on neuroimaging-sized datasets"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)
    # Neuroimaging-sized problem
    X = np.random.randn(300, 10000).astype(np.float32)
    y = np.random.randn(300).astype(np.float32)

    result = ridge_cv(X, y, alphas=[0.1, 1.0, 10.0], cv=3, parallel="cpu")

    assert result["coef"].shape == (10000,)
    assert result["alpha"] > 0


@pytest.mark.slow
def test_backend_selection():
    """Backend selection should work correctly"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)

    # Small problem - use CPU
    X_small = np.random.randn(100, 1000).astype(np.float32)
    y_small = np.random.randn(100).astype(np.float32)
    result_small = ridge_cv(X_small, y_small, cv=3, parallel="cpu")
    assert result_small["coef"].shape == (1000,)

    # Large problem - try GPU (will fallback to CPU if unavailable)
    X_large = np.random.randn(300, 50000).astype(np.float32)
    y_large = np.random.randn(300).astype(np.float32)
    result_large = ridge_cv(X_large, y_large, alphas=[1.0, 10.0], cv=3, parallel="gpu")
    assert result_large["coef"].shape == (50000,)
    # Backend info may be in "parallel" or "backend" key
    assert "parallel" in result_large or "backend" in result_large


# ============================================================================
# Edge Cases
# ============================================================================


@pytest.mark.slow
def test_single_alpha():
    """Should work with single alpha value"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    result = ridge_cv(X, y, alphas=[1.0], cv=3, parallel="cpu")
    assert result["alpha"] == 1.0


@pytest.mark.slow
def test_perfect_fit_case():
    """Should handle perfect fit scenarios"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    beta_true = np.random.randn(50).astype(np.float32)
    y = X @ beta_true  # Perfect linear relationship

    result = ridge_cv(X, y, alphas=[1e-6, 0.1, 1.0], cv=3, parallel="cpu")

    # Should prefer small alpha for perfect fit
    assert result["alpha"] <= 0.1


@pytest.mark.slow
def test_noisy_data():
    """Should handle noisy data appropriately"""
    from nltools.algorithms.ridge import ridge_cv

    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    beta_true = np.random.randn(50).astype(np.float32)
    y = X @ beta_true + 0.5 * np.random.randn(100).astype(np.float32)

    result = ridge_cv(X, y, alphas=[0.01, 0.1, 1.0, 10.0], cv=3, parallel="cpu")

    # Should select some regularization
    assert 0.01 <= result["alpha"] <= 10.0


# ============================================================================
# Statistical Correctness Tests (Tier 1 - No Parallel/GPU)
# ============================================================================


def test_ridge_coefficients_converge_to_true_values():
    """Test that Ridge coefficients converge to true values when known relationship exists."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    alpha = 1.0

    # Create known relationship: y = X @ true_beta + noise
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    true_beta = np.random.randn(n_features).astype(np.float32)
    noise = np.random.randn(n_samples).astype(np.float32) * 0.1
    y = X @ true_beta + noise

    # Fit Ridge regression
    beta_ridge = ridge_svd(X, y, alpha=alpha, parallel=None)

    # With low noise and moderate regularization, Ridge should recover true coefficients
    # Tolerance: rtol=0.15 (15% acceptable for moderate regularization)
    np.testing.assert_allclose(beta_ridge, true_beta, rtol=0.15, atol=0.2)


def test_ridge_regularization_effect_statistical():
    """Test that higher alpha produces smaller coefficient magnitudes (statistical correctness)."""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    # Test multiple alpha values
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    coefficient_norms = []

    for alpha in alphas:
        beta = ridge_svd(X, y, alpha=alpha, parallel=None)
        coefficient_norms.append(np.linalg.norm(beta))

    # Verify monotonic decrease: higher alpha → smaller coefficients
    for i in range(len(coefficient_norms) - 1):
        assert coefficient_norms[i] >= coefficient_norms[i + 1], (
            f"Higher alpha should produce smaller coefficients. "
            f"alpha={alphas[i]}: norm={coefficient_norms[i]:.6f}, "
            f"alpha={alphas[i + 1]}: norm={coefficient_norms[i + 1]:.6f}"
        )


def test_ridge_converges_to_ols_when_alpha_near_zero():
    """Test that Ridge converges to OLS when alpha → 0 (statistical correctness)."""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)

    # OLS solution (using pseudo-inverse)
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]

    # Ridge with very small alpha (should approximate OLS)
    beta_ridge_small = ridge_svd(X, y, alpha=1e-8, parallel=None)

    # Should be very close to OLS (within numerical precision)
    np.testing.assert_allclose(beta_ridge_small, beta_ols, rtol=1e-4, atol=1e-4)


# ============================================================================
# ridge_cv: splitter forwarding + fit_intercept
# ============================================================================


class TestRidgeCvSplitter:
    """ridge_cv must honor an arbitrary sklearn splitter, not just int folds."""

    def _data(self, n=200, p=10, intercept=0.0, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p)).astype(np.float32)
        coef = rng.standard_normal(p).astype(np.float32)
        y = (X @ coef + intercept + 0.1 * rng.standard_normal(n)).astype(np.float32)
        return X, y

    def test_int_cv_still_works(self):
        from nltools.algorithms.ridge import ridge_cv

        X, y = self._data()
        result = ridge_cv(X, y, alphas=np.array([0.1, 1.0, 10.0]), cv=3)
        assert "alpha" in result
        assert result["cv_scores"].shape == (3, 3, 1)

    def test_splitter_object_honored(self):
        """Different splitters should produce different per-fold scores."""
        from sklearn.model_selection import KFold
        from nltools.algorithms.ridge import ridge_cv

        X, y = self._data()
        alphas = np.array([0.1, 1.0, 10.0])
        contig = ridge_cv(X, y, alphas=alphas, cv=KFold(5, shuffle=False))
        shuffled = ridge_cv(
            X, y, alphas=alphas, cv=KFold(5, shuffle=True, random_state=0)
        )
        assert contig["cv_scores"].shape == shuffled["cv_scores"].shape == (5, 3, 1)
        # Per-fold scores should differ — same data, different splits.
        assert not np.allclose(contig["cv_scores"], shuffled["cv_scores"])

    def test_fit_intercept_recovers_offset(self):
        from nltools.algorithms.ridge import ridge_cv

        X, y = self._data(intercept=42.0)
        result = ridge_cv(
            X, y, alphas=np.array([0.01, 0.1, 1.0]), cv=5, fit_intercept=True
        )
        assert "intercept" in result
        assert abs(result["intercept"] - 42.0) < 0.5

    def test_generator_cv_rejected(self):
        """A consumed-once generator can't be re-iterated for alpha sweeps."""
        from sklearn.model_selection import KFold
        from nltools.algorithms.ridge import ridge_cv

        X, y = self._data()
        with pytest.raises(TypeError, match="generator"):
            ridge_cv(X, y, alphas=np.array([0.1, 1.0]), cv=KFold(5).split(X))
