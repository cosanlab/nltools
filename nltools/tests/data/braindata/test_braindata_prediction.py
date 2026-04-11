import numpy as np
import polars as pl
import pytest

from nltools.data import BrainData
from nltools.data.simulator import Simulator


class TestBrainDataPrediction:
    def test_predict_mvpa_whole_brain(self, sim_brain_data):
        """Test predict(y=...) performs MVPA decoding."""
        # Create binary classification problem
        n_samples = sim_brain_data.shape[0]
        y = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))

        # Run whole-brain MVPA
        accuracy = sim_brain_data.predict(
            y=y, method="whole_brain", cv=3, show_progress=False
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
            show_progress=False,
        )

        assert isinstance(accuracy, BrainData)
        assert 0 <= accuracy.data.flatten()[0] <= 1

    @pytest.mark.slow
    def test_predict_multi(self):
        """Test that deprecated predict_multi method raises NotImplementedError."""
        # Need to set up minimal data for the test
        sim = Simulator()
        dat = sim.create_data([0, 1], sigma=1, reps=5, output_dir=".")
        y = pl.read_csv("y.csv", has_header=False)
        dat = BrainData("data.nii.gz", Y=y)

        with pytest.raises(
            NotImplementedError, match="predict_multi.*deprecated.*Model class"
        ):
            dat.predict_multi(algorithm="svm")
