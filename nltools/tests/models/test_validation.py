"""Tests for the shared `_check_is_fitted` helper and `Ridge` input validation."""

import numpy as np
import pytest

from nltools.models import Ridge
from nltools.models.validation import _check_is_fitted


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


class TestRidgeInputValidation:
    """`Ridge` validates its own feature matrices and targets."""

    @pytest.fixture()
    def data(self):
        rng = np.random.default_rng(0)
        return rng.standard_normal((30, 4)), rng.standard_normal(30)

    def test_fit_rejects_non_2d_X(self, data):
        """Fitting a 1-D X raises."""
        _, y = data

        with pytest.raises(ValueError, match="2D feature matrix"):
            Ridge().fit(np.zeros(30), y)

    def test_score_rejects_mismatched_y(self, data):
        """Scoring X and y with different sample counts raises."""
        X, y = data
        model = Ridge().fit(X, y)

        with pytest.raises(ValueError, match="inconsistent sample counts"):
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
