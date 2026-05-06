import numpy as np
import pytest

from nltools.data import BrainData


class TestBrainDataPrediction:
    def test_predict_mvpa_whole_brain(self, sim_brain_data):
        """Test predict(y=...) performs MVPA decoding."""
        # Create binary classification problem
        n_samples = sim_brain_data.shape[0]
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))

        # Run whole-brain MVPA
        accuracy = sim_brain_data.predict(
            y=y, method="whole_brain", cv=3, progress_bar=False
        )

        # Should return BrainData with single accuracy value
        assert isinstance(accuracy, BrainData)
        assert accuracy.shape[0] == 1
        # Accuracy should be between 0 and 1
        assert 0 <= accuracy.data.flatten()[0] <= 1

    def test_predict_mvpa_cannot_specify_both_x_and_y(self, sim_brain_data):
        """Test that specifying both X and y raises error."""
        X = np.random.randn(len(sim_brain_data), 5)
        y = np.array([0, 1] * (len(sim_brain_data) // 2))

        with pytest.raises(ValueError, match="Cannot specify both X and y"):
            sim_brain_data.predict(X=X, y=y)

    def test_predict_mvpa_invalid_method(self, sim_brain_data):
        """Test invalid method raises error."""
        y = np.array([0, 1] * (len(sim_brain_data) // 2))

        with pytest.raises(ValueError, match="Invalid method"):
            sim_brain_data.predict(y=y, method="invalid_method")

    def test_predict_mvpa_custom_estimator(self, sim_brain_data):
        """Test custom sklearn estimator works."""
        from sklearn.linear_model import LogisticRegression

        n_samples = sim_brain_data.shape[0]
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))

        accuracy = sim_brain_data.predict(
            y=y,
            method="whole_brain",
            estimator=LogisticRegression(max_iter=1000),
            cv=3,
            progress_bar=False,
        )

        assert isinstance(accuracy, BrainData)
        assert 0 <= accuracy.data.flatten()[0] <= 1

    def test_predict_multi(self, minimal_brain_data):
        """Deprecated .predict_multi() raises NotImplementedError pointing
        at the future Model class (per migration guide)."""
        with pytest.raises(
            NotImplementedError, match="predict_multi.*deprecated.*Model class"
        ):
            minimal_brain_data.predict_multi(algorithm="svm")
