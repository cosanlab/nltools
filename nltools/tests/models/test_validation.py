"""Tests for the shared estimator input validation helpers and their use by `Ridge`."""

import numpy as np
import pytest

from nltools.models import Ridge
from nltools.models.validation import _check_is_fitted, _validate_X, _validate_X_y


class Estimator:
    """Minimal stand-in for an estimator that uses the shared helpers."""

    def __init__(self):
        self.is_fitted_ = False


class TestCheckIsFitted:
    """`_check_is_fitted` gates use of an unfitted estimator."""

    def test_unfitted_raises(self):
        """An unfitted estimator raises `ValueError` naming its class."""
        with pytest.raises(ValueError, match="Estimator instance is not fitted yet"):
            _check_is_fitted(Estimator())

    def test_fitted_passes(self):
        """A fitted estimator passes silently."""
        model = Estimator()
        model.is_fitted_ = True

        assert _check_is_fitted(model) is None


class TestValidateX:
    """`_validate_X` enforces a 2-D feature matrix and the fitted feature count."""

    def test_accepts_2d_and_returns_array(self):
        """A 2-D input is converted to an ndarray and returned."""
        X = _validate_X(Estimator(), [[1.0, 2.0], [3.0, 4.0]])

        assert isinstance(X, np.ndarray)
        assert X.shape == (2, 2)

    @pytest.mark.parametrize("shape", [(100,), (10, 20, 30)])
    def test_rejects_non_2d(self, shape):
        """X must be 2-D."""
        with pytest.raises(ValueError, match="2D array"):
            _validate_X(Estimator(), np.zeros(shape))

    def test_reset_false_rejects_wrong_feature_count(self):
        """With `reset=False`, the feature count must match the fitted count."""
        model = Estimator()
        model.n_features_in_ = 50

        with pytest.raises(ValueError, match="features"):
            _validate_X(model, np.zeros((20, 40)), reset=False)

        assert _validate_X(model, np.zeros((20, 50)), reset=False).shape == (20, 50)

    def test_reset_true_ignores_fitted_feature_count(self):
        """With `reset=True`, a different feature count is a new fit, not an error."""
        model = Estimator()
        model.n_features_in_ = 50

        assert _validate_X(model, np.zeros((20, 40))).shape == (20, 40)


class TestValidateXy:
    """`_validate_X_y` enforces the target rank and matching sample counts."""

    def test_accepts_1d_and_2d_targets(self):
        """1-D and 2-D targets with matching samples are returned as arrays."""
        X = np.zeros((100, 50))

        _, y_1d = _validate_X_y(Estimator(), X, np.zeros(100))
        _, y_2d = _validate_X_y(Estimator(), X, np.zeros((100, 5)))

        assert y_1d.shape == (100,)
        assert y_2d.shape == (100, 5)

    def test_rejects_3d_y(self):
        """y must be 1-D or 2-D."""
        with pytest.raises(ValueError, match="1D or 2D"):
            _validate_X_y(Estimator(), np.zeros((10, 5)), np.zeros((10, 2, 2)))

    def test_rejects_mismatched_sample_counts(self):
        """X and y must have the same number of samples."""
        with pytest.raises(ValueError, match="inconsistent number of samples"):
            _validate_X_y(Estimator(), np.zeros((100, 50)), np.zeros(90))


class TestRidgeUsesTheHelpers:
    """`Ridge` keeps the validation behavior it had while inheriting `BaseModel`."""

    @pytest.fixture()
    def data(self):
        rng = np.random.default_rng(0)
        return rng.standard_normal((30, 4)), rng.standard_normal(30)

    def test_fit_rejects_non_2d_X(self, data):
        """Fitting a 1-D X raises."""
        _, y = data

        with pytest.raises(ValueError, match="2D array"):
            Ridge().fit(np.zeros(30), y)

    def test_score_rejects_mismatched_y(self, data):
        """Scoring X and y with different sample counts raises."""
        X, y = data
        model = Ridge().fit(X, y)

        with pytest.raises(ValueError, match="inconsistent number of samples"):
            model.score(X, np.zeros(29))

    def test_predict_before_fit_raises(self, data):
        """Predicting before fit raises, naming `Ridge`."""
        X, _ = data

        with pytest.raises(ValueError, match="Ridge instance is not fitted yet"):
            Ridge().predict(X)

    def test_predict_rejects_wrong_feature_count(self, data):
        """Predicting with a different feature count than training raises."""
        X, y = data
        model = Ridge().fit(X, y)

        with pytest.raises(ValueError, match="was fitted with 4 features"):
            model.predict(np.zeros((5, 3)))
