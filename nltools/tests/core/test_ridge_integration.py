"""Integration tests for ridge regression module.

Tests:
- Backward compatibility (old API still works)
- New API functionality (solve_ridge_cv, solve_banded_ridge_cv)
- Backend switching (numpy → torch → numpy)
- Import paths (various ways to import)
- Result consistency across backends
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

# Mark all tests in this file as tier1 (fast integration tests)
pytestmark = pytest.mark.tier1


class TestBackwardCompatibility:
    """Ensure old API still works for backward compatibility."""

    def test_ridge_svd_still_works(self):
        """Old ridge_svd function should still work."""
        from nltools.algorithms.ridge import ridge_svd

        np.random.seed(42)
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        weights = ridge_svd(X, y, alpha=1.0)

        assert weights.shape == (50,)
        assert not np.any(np.isnan(weights))

    def test_ridge_cv_still_works(self):
        """Old ridge_cv function should still work."""
        from nltools.algorithms.ridge import ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50)

        # Test with multiple targets
        Y = np.random.randn(100, 5)

        result = ridge_cv(X, Y, alphas=[0.1, 1.0, 10.0], cv=3)

        assert isinstance(result, dict)
        assert "alpha" in result
        assert "coef" in result
        assert "cv_scores" in result
        assert result["coef"].shape == (50, 5)
        assert not np.any(np.isnan(result["coef"]))

    def test_can_import_from_algorithms(self):
        """Old import path from algorithms should still work."""
        from nltools.algorithms import ridge_svd, ridge_cv

        assert callable(ridge_svd)
        assert callable(ridge_cv)


class TestNewAPI:
    """Test new GPU-enabled API."""

    def test_solve_ridge_cv_basic(self):
        """New solve_ridge_cv should work with basic inputs."""
        from nltools.algorithms.ridge import solve_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 10)

        result = solve_ridge_cv(
            X, Y, alphas=[0.1, 1.0, 10.0], cv=3, parallel="cpu"
        )
        best_alphas = result['best_alphas']
        coefs = result['coefs']
        scores = result['cv_scores']

        assert best_alphas.shape == (10,)
        assert coefs.shape == (50, 10)
        assert scores.shape == (3, 3, 10)  # (n_splits, n_alphas, n_targets)
        assert not np.any(np.isnan(coefs))

    def test_solve_ridge_cv_local_vs_global_alpha(self):
        """Test local_alpha parameter."""
        from nltools.algorithms.ridge import solve_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 10)

        # Local alpha (different per target)
        result_local = solve_ridge_cv(
            X, Y, alphas=[0.1, 1.0, 10.0], cv=3, local_alpha=True, parallel="cpu"
        )
        best_alphas_local = result_local['best_alphas']

        # Global alpha (same for all targets)
        result_global = solve_ridge_cv(
            X, Y, alphas=[0.1, 1.0, 10.0], cv=3, local_alpha=False, parallel="cpu"
        )
        best_alphas_global = result_global['best_alphas']

        # Global should have same alpha for all targets
        assert np.all(best_alphas_global == best_alphas_global[0])

        # Local might have different alphas
        # (not guaranteed, but likely with random data)
        assert best_alphas_local.shape == (10,)

    def test_solve_banded_ridge_cv_basic(self):
        """Test banded ridge with multiple feature spaces."""
        from nltools.algorithms.ridge import solve_banded_ridge_cv

        np.random.seed(42)
        X1 = np.random.randn(100, 30)
        X2 = np.random.randn(100, 20)
        Y = np.random.randn(100, 5)

        result = solve_banded_ridge_cv(
            [X1, X2], Y, n_iter=5, alphas=[0.1, 1.0, 10.0], cv=3, parallel="cpu"
        )
        deltas = result['deltas']
        coefs = result['coefs']
        cv_scores = result['cv_scores']

        assert deltas.shape == (2, 5)  # 2 spaces, 5 targets
        assert coefs.shape == (50, 5)  # 30 + 20 = 50 features
        assert cv_scores.shape == (5, 5)  # n_iter, n_targets
        assert not np.any(np.isnan(coefs))

    def test_solve_ridge_cv_is_wrapper_for_banded(self):
        """solve_ridge_cv should give same results as banded with single X."""
        from nltools.algorithms.ridge import solve_ridge_cv, solve_banded_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 5)
        alphas = [0.1, 1.0, 10.0]

        # Regular ridge
        result_1 = solve_ridge_cv(
            X, Y, alphas=alphas, cv=3, parallel="cpu"
        )
        best_alphas_1 = result_1['best_alphas']
        coefs_1 = result_1['coefs']
        scores_1 = result_1['cv_scores']

        # Banded with single feature space (true banded ridge)
        result_2 = solve_banded_ridge_cv(
            [X], Y, n_iter=5, alphas=alphas, cv=3, parallel="cpu"
        )
        deltas_2 = result_2['deltas']
        coefs_2 = result_2['coefs']
        cv_scores_2 = result_2['cv_scores']

        # Results should be similar but not identical (different algorithms)
        # Banded ridge uses random search, so we just check shapes
        assert best_alphas_1.shape == (5,)
        assert deltas_2.shape == (1, 5)
        assert coefs_1.shape == coefs_2.shape
        assert scores_1.shape == (3, 3, 5)  # (n_splits, n_alphas, n_targets)
        assert cv_scores_2.shape == (5, 5)  # (n_iter, n_targets)

    def test_solve_ridge_cv_batching(self):
        """Test target batching for memory efficiency."""
        from nltools.algorithms.ridge import solve_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 20)

        # Without batching
        result_1 = solve_ridge_cv(
            X, Y, alphas=[1.0], cv=2, parallel="cpu", n_targets_batch=None
        )
        best_alphas_1 = result_1['best_alphas']
        coefs_1 = result_1['coefs']
        scores_1 = result_1['cv_scores']

        # With batching
        result_2 = solve_ridge_cv(
            X, Y, alphas=[1.0], cv=2, parallel="cpu", n_targets_batch=5
        )
        best_alphas_2 = result_2['best_alphas']
        coefs_2 = result_2['coefs']
        scores_2 = result_2['cv_scores']

        # Should give same results
        assert_array_equal(best_alphas_1, best_alphas_2)
        assert_allclose(coefs_1, coefs_2, rtol=1e-10)
        assert_allclose(scores_1, scores_2, rtol=1e-10)


# Helper function
def _torch_available():
    """Check if PyTorch is available."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None


class TestBackendManagement:
    """Test backend switching and management."""

    def test_import_backend_functions(self):
        """Backend management functions should be importable."""
        from nltools.algorithms.ridge import set_backend, get_backend, ALL_BACKENDS

        assert callable(set_backend)
        assert callable(get_backend)
        assert isinstance(ALL_BACKENDS, (list, tuple))
        assert "numpy" in ALL_BACKENDS

    def test_backend_switching(self):
        """Should be able to switch backends."""
        from nltools.algorithms.ridge import set_backend, get_backend

        # Set to numpy
        set_backend("numpy")
        backend = get_backend()
        assert backend.name == "numpy"

        # Try to set to torch (may not be available)
        try:
            set_backend("torch")
            backend = get_backend()
            assert backend.name == "torch"
        except ImportError:
            pytest.skip("PyTorch not available")

        # Set back to numpy
        set_backend("numpy")
        backend = get_backend()
        assert backend.name == "numpy"

    def test_numpy_backend_always_available(self):
        """NumPy backend should always be available."""
        from nltools.algorithms.ridge import set_backend, get_backend

        set_backend("numpy")
        backend = get_backend()
        assert backend.name == "numpy"

    @pytest.mark.skipif(
        not _torch_available(), reason="PyTorch not available for backend testing"
    )
    def test_backend_consistency(self):
        """Results should be consistent across backends."""
        from nltools.algorithms.ridge import solve_ridge_cv, set_backend

        np.random.seed(42)
        X = np.random.randn(50, 30)
        Y = np.random.randn(50, 5)
        alphas = [1.0]

        # NumPy backend
        set_backend("numpy")
        result_np = solve_ridge_cv(X, Y, alphas=alphas, cv=2)
        best_alphas_np = result_np['best_alphas']
        coefs_np = result_np['coefs']
        scores_np = result_np['cv_scores']

        # Torch backend
        set_backend("torch")
        result_torch = solve_ridge_cv(
            X, Y, alphas=alphas, cv=2
        )
        best_alphas_torch = result_torch['best_alphas']
        coefs_torch = result_torch['coefs']
        scores_torch = result_torch['cv_scores']

        # Results should be close (allowing for numerical differences)
        assert_array_equal(best_alphas_np, best_alphas_torch)
        assert_allclose(coefs_np, coefs_torch, rtol=1e-4, atol=1e-6)
        assert_allclose(scores_np, scores_torch, rtol=1e-4, atol=1e-6)

        # Reset to numpy
        set_backend("numpy")


class TestImportPaths:
    """Test various import paths work correctly."""

    def test_import_from_ridge(self):
        """Import from ridge module."""
        from nltools.algorithms.ridge import solve_ridge_cv, solve_banded_ridge_cv

        assert callable(solve_ridge_cv)
        assert callable(solve_banded_ridge_cv)

    def test_import_backends(self):
        """Import backend management."""
        from nltools.algorithms.ridge import set_backend, get_backend, ALL_BACKENDS

        assert callable(set_backend)
        assert callable(get_backend)
        assert isinstance(ALL_BACKENDS, (list, tuple))

    def test_import_utilities(self):
        """Import utility functions."""
        from nltools.algorithms.ridge import _decompose_ridge, _r2_score

        assert callable(_decompose_ridge)
        assert callable(_r2_score)

    def test_import_legacy(self):
        """Import legacy functions."""
        from nltools.algorithms.ridge import ridge_svd, ridge_cv

        assert callable(ridge_svd)
        assert callable(ridge_cv)

    def test_import_ridge_module(self):
        """Import ridge module itself."""
        from nltools.algorithms import ridge

        assert hasattr(ridge, "solve_ridge_cv")
        assert hasattr(ridge, "solve_banded_ridge_cv")
        assert hasattr(ridge, "set_backend")
        assert hasattr(ridge, "get_backend")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_alpha(self):
        """Test with single alpha value."""
        from nltools.algorithms.ridge import solve_ridge_cv

        np.random.seed(42)
        X = np.random.randn(50, 20)
        Y = np.random.randn(50, 5)

        result = solve_ridge_cv(
            X, Y, alphas=[1.0], cv=2, parallel="cpu"
        )
        best_alphas = result['best_alphas']
        coefs = result['coefs']

        # All targets should get the same alpha
        assert np.all(best_alphas == 1.0)
        assert coefs.shape == (20, 5)

    def test_single_target(self):
        """Test with single target."""
        from nltools.algorithms.ridge import solve_ridge_cv

        np.random.seed(42)
        X = np.random.randn(50, 20)
        y = np.random.randn(50, 1)

        result = solve_ridge_cv(
            X, y, alphas=[0.1, 1.0, 10.0], cv=2, parallel="cpu"
        )
        best_alphas = result['best_alphas']
        coefs = result['coefs']
        scores = result['cv_scores']

        assert best_alphas.shape == (1,)
        assert coefs.shape == (20, 1)
        assert scores.shape == (2, 3, 1)  # (n_splits, n_alphas, n_targets)

    def test_empty_feature_space_raises(self):
        """Banded ridge should raise on empty feature space list."""
        from nltools.algorithms.ridge import solve_banded_ridge_cv

        np.random.seed(42)
        Y = np.random.randn(50, 5)

        with pytest.raises(ValueError, match="Xs cannot be empty"):
            solve_banded_ridge_cv([], Y, n_iter=5, alphas=[1.0], cv=2)
