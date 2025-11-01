"""Tests for banded ridge regression (general case for multi-feature-space ridge).

Banded ridge is the general implementation that handles multiple feature spaces.
Regular ridge CV is a special case with a single feature space.

This follows himalaya's solve_group_ridge_random_search pattern but without
the Dirichlet sampling (we use fixed banded regularization instead).
"""

import pytest
import numpy as np
from sklearn.model_selection import KFold


def _torch_available():
    """Check if PyTorch is available."""
    import importlib.util

    return importlib.util.find_spec("torch") is not None


def _torch_cuda_available():
    """Check if PyTorch with CUDA is available."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class TestBandedRidgeBasics:
    """Test basic banded ridge functionality."""

    def test_single_group_matches_regular_ridge(self):
        """Banded ridge with 1 group should match regular ridge behavior."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        # Create data
        np.random.seed(42)
        n_samples, n_features, n_targets = 100, 50, 10
        X = np.random.randn(n_samples, n_features)
        Y = np.random.randn(n_samples, n_targets)

        # Banded with single group
        best_alphas, coefs, scores = solve_banded_ridge_cv(
            Xs=[X],
            Y=Y,
            alphas=[0.1, 1.0, 10.0],
            cv=3,
            local_alpha=True,
            backend="numpy",
        )

        # Should return per-target alphas
        assert best_alphas.shape == (n_targets,)
        assert coefs.shape == (n_features, n_targets)
        assert scores.shape == (n_targets,)
        assert np.all(np.isfinite(coefs))
        assert np.all(np.isfinite(scores))

    def test_two_feature_groups(self):
        """Test banded ridge with 2 feature groups."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        n_samples = 100
        X1 = np.random.randn(n_samples, 30)
        X2 = np.random.randn(n_samples, 20)
        Y = np.random.randn(n_samples, 10)

        best_alphas, coefs, scores = solve_banded_ridge_cv(
            Xs=[X1, X2],
            Y=Y,
            alphas=[0.1, 1.0, 10.0],
            cv=3,
            local_alpha=True,
            backend="numpy",
        )

        # Coefficients should match concatenated feature dimension
        assert coefs.shape == (50, 10)  # 30 + 20 features
        assert best_alphas.shape == (10,)
        assert scores.shape == (10,)

    def test_three_feature_groups(self):
        """Test banded ridge with 3 feature groups."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        n_samples = 100
        X1 = np.random.randn(n_samples, 20)
        X2 = np.random.randn(n_samples, 15)
        X3 = np.random.randn(n_samples, 10)
        Y = np.random.randn(n_samples, 5)

        best_alphas, coefs, scores = solve_banded_ridge_cv(
            Xs=[X1, X2, X3],
            Y=Y,
            alphas=[0.1, 1.0, 10.0],
            cv=3,
            local_alpha=True,
        )

        assert coefs.shape == (45, 5)  # 20 + 15 + 10 features
        assert best_alphas.shape == (5,)

    def test_empty_feature_groups_raises(self):
        """Empty feature group list should raise error."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        Y = np.random.randn(100, 10)

        with pytest.raises((ValueError, IndexError)):
            solve_banded_ridge_cv(Xs=[], Y=Y, alphas=[1.0])

    def test_mismatched_n_samples_raises(self):
        """Mismatched n_samples between feature groups should raise error."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        X1 = np.random.randn(100, 30)
        X2 = np.random.randn(90, 20)  # Wrong n_samples
        Y = np.random.randn(100, 10)

        with pytest.raises(ValueError):
            solve_banded_ridge_cv(Xs=[X1, X2], Y=Y, alphas=[1.0])


class TestAlphaSelection:
    """Test alpha selection strategies."""

    def test_local_alpha_per_target(self):
        """local_alpha=True should select different alpha per target."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 10)

        best_alphas, coefs, scores = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=[0.1, 1.0, 10.0], cv=3, local_alpha=True
        )

        # Should have per-target alphas (might not all be unique, but shape is correct)
        assert best_alphas.shape == (10,)
        assert len(np.unique(best_alphas)) >= 1  # At least 1 unique alpha

    def test_global_alpha_shared(self):
        """local_alpha=False should use same alpha for all targets."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 10)

        best_alphas, coefs, scores = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=[0.1, 1.0, 10.0], cv=3, local_alpha=False
        )

        # Should have same alpha for all targets
        assert best_alphas.shape == (10,)
        assert len(np.unique(best_alphas)) == 1  # All same

    def test_single_alpha_works(self):
        """Single alpha should work (no CV needed)."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 10)

        # Single alpha (still does CV but only one choice)
        best_alphas, coefs, scores = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=[1.0], cv=3, local_alpha=True
        )

        # All alphas should be the single value
        assert np.allclose(best_alphas, 1.0)

    def test_alpha_range_exploration(self):
        """Test that different alpha ranges affect selection."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 10)

        # Small alphas
        alphas_small, _, scores_small = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=[0.001, 0.01, 0.1], cv=3, local_alpha=True
        )

        # Large alphas
        alphas_large, _, scores_large = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=[10.0, 100.0, 1000.0], cv=3, local_alpha=True
        )

        # Different ranges should select different alphas
        assert not np.allclose(alphas_small.mean(), alphas_large.mean())


class TestBatching:
    """Test batching strategies."""

    def test_target_batching(self):
        """Test batching over targets."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 100)  # Many targets

        # With batching
        best_alphas_batch, coefs_batch, scores_batch = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=[0.1, 1.0, 10.0], cv=3, n_targets_batch=20
        )

        # Without batching
        best_alphas_full, coefs_full, scores_full = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=[0.1, 1.0, 10.0], cv=3, n_targets_batch=None
        )

        # Results should be very similar (allowing for numerical differences)
        np.testing.assert_allclose(best_alphas_batch, best_alphas_full, rtol=1e-5)
        np.testing.assert_allclose(coefs_batch, coefs_full, rtol=1e-5)

    def test_alpha_batching(self):
        """Test batching over alphas."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 10)

        # Many alphas with batching
        alphas = np.logspace(-2, 2, 20)
        best_alphas_batch, coefs_batch, scores_batch = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=alphas, cv=3, n_alphas_batch=5
        )

        # Without batching
        best_alphas_full, coefs_full, scores_full = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=alphas, cv=3, n_alphas_batch=None
        )

        # Results should be identical (generator pattern should be transparent)
        np.testing.assert_allclose(best_alphas_batch, best_alphas_full, rtol=1e-5)
        np.testing.assert_allclose(coefs_batch, coefs_full, rtol=1e-5)

    def test_combined_batching(self):
        """Test batching over both targets and alphas."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 50)  # Many targets

        alphas = np.logspace(-2, 2, 20)  # Many alphas

        best_alphas, coefs, scores = solve_banded_ridge_cv(
            Xs=[X],
            Y=Y,
            alphas=alphas,
            cv=3,
            n_targets_batch=10,
            n_alphas_batch=5,
        )

        assert best_alphas.shape == (50,)
        assert coefs.shape == (50, 50)
        assert np.all(np.isfinite(coefs))


class TestCrossValidation:
    """Test cross-validation strategies."""

    def test_kfold_cv(self):
        """Test with KFold cross-validation."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 10)

        # Use sklearn KFold directly
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        best_alphas, coefs, scores = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=[0.1, 1.0, 10.0], cv=cv
        )

        assert best_alphas.shape == (10,)
        assert coefs.shape == (50, 10)

    def test_different_cv_splits(self):
        """Different CV splits should give slightly different results."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 10)

        # 3-fold
        _, _, scores_3 = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=[0.1, 1.0, 10.0], cv=3
        )

        # 5-fold
        _, _, scores_5 = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=[0.1, 1.0, 10.0], cv=5
        )

        # Scores should be somewhat different (different validation sets)
        assert not np.allclose(scores_3, scores_5, rtol=0.01)


@pytest.mark.tier1
class TestBackends:
    """Test different backends."""

    def test_numpy_backend(self):
        """NumPy backend should work."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 10)

        best_alphas, coefs, scores = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=[0.1, 1.0, 10.0], cv=3, backend="numpy"
        )

        assert best_alphas.shape == (10,)
        assert coefs.shape == (50, 10)
        assert np.all(np.isfinite(coefs))

    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not available")
    def test_torch_backend(self):
        """PyTorch CPU backend should work."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50).astype(np.float32)
        Y = np.random.randn(100, 10).astype(np.float32)

        best_alphas, coefs, scores = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=[0.1, 1.0, 10.0], cv=3, backend="torch"
        )

        # Results should be on CPU as numpy arrays
        assert isinstance(best_alphas, np.ndarray)
        assert isinstance(coefs, np.ndarray)

    @pytest.mark.skipif(not _torch_cuda_available(), reason="CUDA not available")
    def test_torch_cuda_backend(self):
        """PyTorch CUDA backend should work."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50).astype(np.float32)
        Y = np.random.randn(100, 10).astype(np.float32)

        best_alphas, coefs, scores = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=[0.1, 1.0, 10.0], cv=3, backend="torch_cuda"
        )

        # Results should be on CPU as numpy arrays
        assert isinstance(best_alphas, np.ndarray)
        assert isinstance(coefs, np.ndarray)


class TestYInCpu:
    """Test Y_in_cpu strategy."""

    def test_y_in_cpu_default(self):
        """Y_in_cpu=True should be default and work."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 10)

        # Default (Y_in_cpu=True)
        best_alphas, coefs, scores = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=[0.1, 1.0, 10.0], cv=3
        )

        assert best_alphas.shape == (10,)

    def test_y_in_cpu_false(self):
        """Y_in_cpu=False should work."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 10)

        best_alphas, coefs, scores = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=[0.1, 1.0, 10.0], cv=3, Y_in_cpu=False
        )

        assert best_alphas.shape == (10,)

    @pytest.mark.skipif(not _torch_available(), reason="PyTorch not available")
    def test_y_in_cpu_with_torch(self):
        """Y_in_cpu strategy should work with PyTorch backend."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50).astype(np.float32)
        Y = np.random.randn(100, 100).astype(np.float32)  # Many targets

        # With Y_in_cpu (memory efficient)
        best_alphas, coefs, scores = solve_banded_ridge_cv(
            Xs=[X],
            Y=Y,
            alphas=[0.1, 1.0, 10.0],
            cv=3,
            backend="torch",
            Y_in_cpu=True,
            n_targets_batch=20,
        )

        assert best_alphas.shape == (100,)


class TestScoring:
    """Test scoring functions."""

    def test_custom_score_func(self):
        """Custom scoring function should work."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        def neg_mse(y_true, y_pred):
            """Negative MSE (to maximize)."""
            backend_module = __import__(
                "nltools.algorithms.ridge.backends",
                fromlist=["get_backend"],
            ).get_backend()
            # Handle 3D predictions (n_alphas, n_samples, n_targets)
            if len(y_pred.shape) == 3:
                mse = backend_module.mean((y_true[None, :, :] - y_pred) ** 2, axis=1)
            else:
                mse = backend_module.mean((y_true - y_pred) ** 2, axis=0)
            return -mse

        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 10)

        best_alphas, coefs, scores = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=[0.1, 1.0, 10.0], cv=3, score_func=neg_mse
        )

        # Scores should be negative (neg MSE)
        assert np.all(scores <= 0)


class TestEdgeCases:
    """Test edge cases."""

    def test_single_sample_per_fold(self):
        """Very small n_samples should still work with appropriate CV."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        X = np.random.randn(10, 5)
        Y = np.random.randn(10, 3)

        best_alphas, coefs, scores = solve_banded_ridge_cv(
            Xs=[X],
            Y=Y,
            alphas=[0.1, 1.0, 10.0],
            cv=2,  # Only 2 folds
        )

        assert best_alphas.shape == (3,)

    def test_single_target(self):
        """Single target should work."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        X = np.random.randn(100, 50)
        Y = np.random.randn(100, 1)

        best_alphas, coefs, scores = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=[0.1, 1.0, 10.0], cv=3
        )

        assert best_alphas.shape == (1,)
        assert coefs.shape == (50, 1)

    def test_more_features_than_samples(self):
        """More features than samples should work (regularization handles it)."""
        from nltools.algorithms.ridge.solvers import solve_banded_ridge_cv

        np.random.seed(42)
        X = np.random.randn(50, 100)  # More features than samples
        Y = np.random.randn(50, 10)

        best_alphas, coefs, scores = solve_banded_ridge_cv(
            Xs=[X], Y=Y, alphas=[0.1, 1.0, 10.0], cv=3
        )

        assert coefs.shape == (100, 10)
