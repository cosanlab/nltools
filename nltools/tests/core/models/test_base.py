"""Tests for BaseModel abstract interface, shared functionality, and input validation."""

import numpy as np
import pytest

from nltools.models import BaseModel

pytestmark = pytest.mark.slow


class TestBaseModelInterface:
    """Abstract interface enforcement: fit, predict, score must be implemented."""

    def test_is_abstract(self):
        """BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            BaseModel()

    def test_requires_fit(self):
        """Subclass missing fit() cannot be instantiated."""

        class Incomplete(BaseModel):
            def predict(self, X):
                pass

            def score(self, X, y):
                pass

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()

    def test_requires_predict(self):
        """Subclass missing predict() cannot be instantiated."""

        class Incomplete(BaseModel):
            def fit(self, X, y):
                pass

            def score(self, X, y):
                pass

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()

    def test_requires_score(self):
        """Subclass missing score() cannot be instantiated."""

        class Incomplete(BaseModel):
            def fit(self, X, y):
                pass

            def predict(self, X):
                pass

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()

    def test_concrete_subclass(self):
        """Concrete subclass with all methods should instantiate."""

        class Concrete(BaseModel):
            def fit(self, X, y):
                return self

            def predict(self, X):
                return np.zeros(X.shape[0])

            def score(self, X, y):
                return 0.0

        model = Concrete()
        assert isinstance(model, BaseModel)


class TestBaseModelState:
    """Fit/predict lifecycle: return self, track state, store shapes."""

    @pytest.fixture()
    def concrete_model(self):
        """A minimal concrete BaseModel subclass."""

        class Concrete(BaseModel):
            def fit(self, X, y):
                super().fit(X, y)
                return self

            def predict(self, X):
                self._check_is_fitted()
                X = self._validate_X(X, reset=False)
                return np.zeros(X.shape[0])

            def score(self, X, y):
                return 0.0

        return Concrete()

    def test_fit_returns_self(self, concrete_model):
        """fit() should return self for method chaining."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        result = concrete_model.fit(X, y)
        assert result is concrete_model

    def test_tracks_fitted_state(self, concrete_model):
        """BaseModel should track whether fit() has been called."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="not fitted"):
            concrete_model.predict(X)

        concrete_model.fit(X, y)
        result = concrete_model.predict(X)
        assert result.shape == (100,)

    def test_stores_training_shape(self, concrete_model):
        """BaseModel should store n_features_in_ and n_samples_ from training."""
        X = np.random.randn(100, 50)
        y = np.random.randn(100)
        concrete_model.fit(X, y)

        assert concrete_model.n_features_in_ == 50
        assert concrete_model.n_samples_ == 100


class TestBaseModelValidation:
    """Input validation: X shape, y shape, feature count at predict time."""

    @pytest.fixture()
    def validating_model(self):
        """A concrete BaseModel that validates inputs."""

        class Validating(BaseModel):
            def fit(self, X, y):
                X, y = self._validate_X_y(X, y)
                super().fit(X, y)
                return self

            def predict(self, X):
                self._check_is_fitted()
                X = self._validate_X(X, reset=False)
                return np.zeros(X.shape[0])

            def score(self, X, y):
                return 0.0

        return Validating()

    def test_rejects_non_2d_X(self, validating_model):
        """X must be a 2D array."""
        y = np.random.randn(100)

        with pytest.raises(ValueError, match="2D array"):
            validating_model.fit(np.random.randn(100), y)

        with pytest.raises(ValueError, match="2D array"):
            validating_model.fit(np.random.randn(10, 20, 30), y)

        # 2D should work
        validating_model.fit(np.random.randn(100, 50), y)

    def test_rejects_mismatched_y(self, validating_model):
        """y sample count must match X."""
        X = np.random.randn(100, 50)

        with pytest.raises(ValueError, match="samples"):
            validating_model.fit(X, np.random.randn(90))

        # Correct shapes should work
        validating_model.fit(X, np.random.randn(100))
        validating_model.fit(X, np.random.randn(100, 5))

    def test_rejects_wrong_feature_count_at_predict(self, validating_model):
        """predict() should reject X with different feature count than training."""
        validating_model.fit(np.random.randn(100, 50), np.random.randn(100))

        validating_model.predict(np.random.randn(20, 50))  # correct

        with pytest.raises(ValueError, match="features"):
            validating_model.predict(np.random.randn(20, 40))
