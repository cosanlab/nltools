"""Tests for Ridge regression model: core fit/predict, cross-validation, and backends."""

import numpy as np
import pytest

from nltools.models import Ridge

from .conftest import torch_available

pytestmark = pytest.mark.slow


class TestRidgeCore:
    """Instantiation, fit, predict, and basic properties."""

    def test_instantiation(self):
        """Ridge should instantiate with alpha parameter."""
        model = Ridge(alpha=1.0)
        assert model.alpha == 1.0
        assert not model.is_fitted_

    def test_default_alpha(self):
        """Ridge should default to alpha=1.0."""
        model = Ridge()
        assert model.alpha == 1.0

    def test_single_target_properties(self, fitted_ridge_single):
        """Fitted single-target Ridge should have correct coef_ shape and predictions."""
        model, data = fitted_ridge_single

        assert model.is_fitted_
        assert model.coef_.shape == (50,)
        assert model.progress_bar is False

        y_pred = model.predict(data["X_test"])
        assert y_pred.shape == (20,)
        assert not np.isnan(y_pred).any()
        assert not np.allclose(y_pred, 0)

    def test_multi_target_properties(self, ridge_multi_target_data):
        """Fitted multi-target Ridge should have 2D coef_ and predictions."""
        model = Ridge(alpha=1.0)
        model.fit(ridge_multi_target_data["X"], ridge_multi_target_data["Y"])

        assert model.coef_.shape == (50, 5)

        Y_pred = model.predict(ridge_multi_target_data["X_test"])
        assert Y_pred.shape == (20, 5)

    def test_predict_without_fit(self):
        """Ridge should raise error if predict called before fit."""
        model = Ridge(alpha=1.0)
        X_test = np.random.randn(20, 50)

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X_test)

    def test_vs_sklearn(self):
        """Ridge should match sklearn Ridge results."""
        from sklearn.linear_model import Ridge as SklearnRidge

        np.random.seed(42)
        X = np.random.randn(100, 50).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        alpha = 1.0

        model_ours = Ridge(alpha=alpha)
        model_ours.fit(X, y)
        pred_ours = model_ours.predict(X)

        model_sklearn = SklearnRidge(alpha=alpha, fit_intercept=False, solver="svd")
        model_sklearn.fit(X, y)
        pred_sklearn = model_sklearn.predict(X)

        np.testing.assert_allclose(pred_ours, pred_sklearn, rtol=1e-4)
        np.testing.assert_allclose(model_ours.coef_, model_sklearn.coef_, rtol=1e-4)


class TestRidgeFeatureSpaces:
    """Auto-detection of single vs multiple feature spaces."""

    def test_auto_detects_single_space(self):
        """Array input should use single-space ridge."""
        np.random.seed(42)
        X = np.random.randn(100, 50).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        model = Ridge(alpha="auto", cv=3)
        model.fit(X, y)

        assert isinstance(model.alpha_, float)
        assert model.deltas_ is None

    def test_auto_detects_multiple_spaces(self):
        """List input should use banded ridge with feature space weights."""
        np.random.seed(42)
        X1 = np.random.randn(100, 30).astype(np.float32)
        X2 = np.random.randn(100, 20).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        model = Ridge(alpha="auto", cv=3, n_iter=5, random_state=42)
        model.fit([X1, X2], y)

        assert model.deltas_ is not None
        assert model.deltas_.shape == (2, 1)
        assert model.alpha_ is None
        assert model.progress_bar is False

    def test_multiple_spaces_requires_cv(self):
        """Multiple feature spaces require alpha='auto' with CV."""
        np.random.seed(42)
        X1 = np.random.randn(100, 30).astype(np.float32)
        X2 = np.random.randn(100, 20).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        model = Ridge(alpha=1.0)
        with pytest.raises(ValueError, match="Banded ridge requires"):
            model.fit([X1, X2], y)

        # progress_bar=True should work with banded ridge
        model_with_pb = Ridge(alpha="auto", cv=3, n_iter=3, progress_bar=True)
        assert model_with_pb.progress_bar is True
        model_with_pb.fit([X1, X2], y)
        assert model_with_pb.is_fitted_


class TestRidgeCV:
    """Cross-validation: alpha selection, reproducibility, custom alphas."""

    def test_cv_instantiation(self):
        """Ridge with cv should instantiate properly."""
        model = Ridge(alpha="auto", cv=5)
        assert model.alpha == "auto"
        assert model.cv == 5

    def test_cv_properties(self, fitted_ridge_cv):
        """CV should select a positive alpha and produce cv_scores_."""
        model, data = fitted_ridge_cv

        assert isinstance(model.alpha_, float)
        assert model.alpha_ > 0
        assert model.cv_scores_.shape[0] == 3

        # Reproducibility
        model2 = Ridge(alpha="auto", cv=3)
        model2.fit(data["X"], data["y"])
        assert model.alpha_ == model2.alpha_
        np.testing.assert_allclose(model.coef_, model2.coef_, rtol=1e-5)

    def test_cv_custom_alphas(self, ridge_single_target_data):
        """Ridge should accept and select from custom alpha range."""
        alphas = [0.1, 1.0, 10.0]
        model = Ridge(alpha="auto", cv=3, alphas=alphas)
        model.fit(ridge_single_target_data["X"], ridge_single_target_data["y"])

        assert model.alpha_ in alphas

    def test_cv_multi_target(self, ridge_multi_target_data):
        """Ridge CV should work with multiple targets."""
        model = Ridge(alpha="auto", cv=3)
        model.fit(ridge_multi_target_data["X"], ridge_multi_target_data["Y"])

        assert model.coef_.shape == (50, 5)
        assert model.cv_scores_.shape[2] == 5


class TestRidgeBackend:
    """Backend selection: numpy, torch, auto."""

    def test_numpy_backend(self):
        """Ridge should work with NumPy backend."""
        np.random.seed(42)
        X = np.random.randn(100, 50).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        model = Ridge(alpha=1.0, backend="numpy")
        model.fit(X, y)
        y_pred = model.predict(X)

        assert y_pred.shape == (100,)
        assert model.backend_.name == "numpy"

    @pytest.mark.skipif(not torch_available(), reason="PyTorch not installed")
    def test_torch_backend(self):
        """Ridge should work with PyTorch backend."""
        np.random.seed(42)
        X = np.random.randn(100, 50).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        model = Ridge(alpha=1.0, backend="torch")
        model.fit(X, y)
        y_pred = model.predict(X)

        assert y_pred.shape == (100,)
        assert model.backend_.name.startswith("torch")

    @pytest.mark.skipif(not torch_available(), reason="PyTorch not installed")
    def test_cpu_gpu_equivalence(self):
        """Ridge should give same results on CPU and GPU."""
        np.random.seed(42)
        X = np.random.randn(100, 50).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        model_cpu = Ridge(alpha=1.0, backend="numpy")
        model_cpu.fit(X, y)
        pred_cpu = model_cpu.predict(X)

        model_gpu = Ridge(alpha=1.0, backend="torch")
        model_gpu.fit(X, y)
        pred_gpu = model_gpu.predict(X)

        np.testing.assert_allclose(pred_gpu, pred_cpu, rtol=1e-3, atol=1e-6)

    def test_auto_backend(self):
        """Ridge should select backend automatically."""
        np.random.seed(42)
        X = np.random.randn(100, 50).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        model = Ridge(alpha=1.0, backend="auto")
        model.fit(X, y)
        y_pred = model.predict(X)

        assert y_pred.shape == (100,)
        assert model.backend_.name in ["numpy", "torch-cpu", "torch-cuda", "torch-mps"]


class TestRidgeFitIntercept:
    """fit_intercept must work on both fixed-α and CV paths."""

    def _offset_data(self, intercept=100.0, n=200, p=10, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p))
        coef = rng.standard_normal(p)
        y = X @ coef + intercept + 0.1 * rng.standard_normal(n)
        return X, y, coef, intercept

    def test_fixed_alpha_recovers_intercept(self):
        X, y, _, true_intercept = self._offset_data()
        m = Ridge(alpha=0.1, fit_intercept=True).fit(X, y)
        assert isinstance(m.intercept_, float)
        assert abs(m.intercept_ - true_intercept) < 0.5
        assert m.score(X, y) > 0.95

    def test_fixed_alpha_no_intercept_kept_for_compat(self):
        """fit_intercept=False stores zero — predict() works either way."""
        X, y, _, _ = self._offset_data(intercept=0.0)
        m = Ridge(alpha=0.1, fit_intercept=False).fit(X, y)
        assert m.intercept_ == 0.0
        assert m.score(X, y) > 0.95

    def test_fixed_alpha_intercept_silently_failed_before(self):
        """Regression: previously fit_intercept was silently ignored on the
        fixed-α path, producing catastrophically negative R² when y had a
        non-zero mean. Now should be ~1.0 on this clean data."""
        X, y, _, _ = self._offset_data(intercept=100.0)
        m = Ridge(alpha=0.1, fit_intercept=True).fit(X, y)
        assert m.score(X, y) > 0.99  # would be very negative without the fix

    def test_multi_target_intercept_per_column(self):
        X, y, _, _ = self._offset_data(intercept=100.0)
        Y = np.column_stack([y, y * 2 + 50])  # second target has intercept ≈ 250
        m = Ridge(alpha=0.1, fit_intercept=True).fit(X, Y)
        assert m.intercept_.shape == (2,)
        assert abs(m.intercept_[0] - 100.0) < 0.5
        assert abs(m.intercept_[1] - 250.0) < 1.0

    def test_cv_path_recovers_intercept(self):
        X, y, _, true_intercept = self._offset_data()
        m = Ridge(alpha="auto", cv=5, alphas=[0.01, 0.1, 1, 10], fit_intercept=True)
        m.fit(X, y)
        assert isinstance(m.intercept_, float)
        assert abs(m.intercept_ - true_intercept) < 0.5
        assert m.score(X, y) > 0.95

    def test_predict_includes_intercept(self):
        X, y, _, true_intercept = self._offset_data(intercept=50.0)
        m = Ridge(alpha=0.1, fit_intercept=True).fit(X, y)
        # Predictions should land near y, not near zero.
        assert abs(m.predict(X).mean() - y.mean()) < 1.0

    def test_cv_path_intercept_comes_from_solver(self):
        """CV-path Ridge.fit() must pull intercept from the solver's return,
        not recompute it post-hoc on the original data. The two are
        algebraically identical, but the contract is that the solver owns
        intercept calculation (parity with banded path)."""
        X, y, _, _ = self._offset_data(intercept=100.0, n=120, p=8)
        Y = np.column_stack([y, y * 0.5 + 20.0])  # multi-target
        m = Ridge(
            alpha="auto", cv=4, alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True
        ).fit(X, Y)
        # Per-target intercept array (solver returns shape (n_targets,))
        assert m.intercept_.shape == (2,)
        # alpha_ must be per-target now (local_alpha=True default)
        assert isinstance(m.alpha_, np.ndarray)
        assert m.alpha_.shape == (2,)
