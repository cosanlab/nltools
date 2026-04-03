import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from nltools.data import BrainData


class TestBrainDataModeling:
    def test_regress_removed(self, sim_brain_data):
        """Verify regress() has been removed with clear migration path."""
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data.Y)),
                "X1": np.array(sim_brain_data.Y).flatten(),
            },
            index=None,
        )

        # Should raise NotImplementedError with migration message
        with pytest.raises(
            NotImplementedError,
            match="regress.*has been removed.*Use fit.*model='glm'",
        ):
            sim_brain_data.regress(design_matrix)

    def test_compute_contrasts_error_not_fitted(self, minimal_brain_data):
        """Test error when compute_contrasts() called before fit()."""
        # Should raise RuntimeError if fit() not called first
        with pytest.raises(RuntimeError, match="Must run .fit"):
            minimal_brain_data.compute_contrasts([1, -1, 0])

    @pytest.mark.slow
    def test_compute_contrasts_numeric_vector(self, minimal_brain_data):
        """Test numeric contrast vector (unique nltools API)."""
        # Set up and run regression using fit()
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condA": np.random.randn(len(minimal_brain_data)),
                "condB": np.random.randn(len(minimal_brain_data)),
            }
        )

        minimal_brain_data.fit(model="glm", X=design_matrix)

        # Compute contrast: A - B (unique nltools logic)
        contrast = minimal_brain_data.compute_contrasts([0, 1, -1])

        # Test nltools-specific API contract
        assert isinstance(contrast, BrainData)
        assert contrast.shape == (1, minimal_brain_data.shape[1])

    @pytest.mark.slow
    def test_compute_contrasts_string_parsing(self, minimal_brain_data):
        """Test string parsing (unique nltools feature)."""
        # Set up and run regression using fit()
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condA": np.random.randn(len(minimal_brain_data)),
                "condB": np.random.randn(len(minimal_brain_data)),
            }
        )

        minimal_brain_data.fit(model="glm", X=design_matrix)

        # Test string parsing (unique nltools feature)
        contrast = minimal_brain_data.compute_contrasts("condA - condB")

        assert isinstance(contrast, BrainData)
        assert contrast.shape == (1, minimal_brain_data.shape[1])

    @pytest.mark.slow
    def test_compute_contrasts_multiple_dict(self, minimal_brain_data):
        """Test multiple contrasts via dict (unique nltools API)."""
        # Set up and run regression using fit()
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condA": np.random.randn(len(minimal_brain_data)),
                "condB": np.random.randn(len(minimal_brain_data)),
            }
        )

        minimal_brain_data.fit(model="glm", X=design_matrix)

        # Test dict of contrasts (unique nltools API)
        contrasts = {"A_vs_B": "condA - condB", "avg_effect": [0, 0.5, 0.5]}
        results = minimal_brain_data.compute_contrasts(contrasts)

        # Should return dict of BrainData objects
        assert isinstance(results, dict)
        assert "A_vs_B" in results
        assert "avg_effect" in results
        assert isinstance(results["A_vs_B"], BrainData)
        assert isinstance(results["avg_effect"], BrainData)

    @pytest.mark.slow
    def test_compute_contrasts_invalid_length(self, minimal_brain_data):
        """Test error for invalid contrast vector length (nltools validation)."""
        # Set up and run regression with 3 regressors using fit()
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condA": np.random.randn(len(minimal_brain_data)),
                "condB": np.random.randn(len(minimal_brain_data)),
            }
        )

        minimal_brain_data.fit(model="glm", X=design_matrix)

        # Provide wrong length contrast (2 instead of 3)
        with pytest.raises(ValueError, match="Contrast vector length.*must match"):
            minimal_brain_data.compute_contrasts([1, -1])

    # ==================== Unified fit/predict API ====================

    def test_fit_predict_ridge_workflow(self, sim_brain_data):
        """Test complete Ridge fit/predict workflow."""
        from nltools.data import BrainData
        from nltools.models import Ridge

        # Fit Ridge model
        X_train = np.random.randn(len(sim_brain_data), 10)
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X_train)

        # Check model stored
        assert hasattr(sim_brain_data, "model_")
        assert isinstance(sim_brain_data.model_, Ridge)
        assert sim_brain_data.model_.is_fitted_

        # Check attributes set
        assert hasattr(sim_brain_data, "ridge_weights")
        assert hasattr(sim_brain_data, "ridge_fitted_values")
        assert hasattr(sim_brain_data, "ridge_scores")

        # Predict on new data
        X_test = np.random.randn(20, 10)  # Different n_samples
        predictions = sim_brain_data.predict(X=X_test)

        # Check predictions
        assert isinstance(predictions, BrainData)
        assert predictions.shape == (20, sim_brain_data.shape[1])

        # Predict on training data (X=None)
        train_predictions = sim_brain_data.predict()
        assert train_predictions.shape == sim_brain_data.shape

    @pytest.mark.slow
    def test_fit_predict_glm_workflow(self, sim_brain_data):
        """Test complete GLM fit/predict workflow."""
        from nltools.models import Glm

        # Fit GLM model
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data)),
                "X1": np.random.randn(len(sim_brain_data)),
            }
        )
        sim_brain_data.fit(model="glm", noise_model="ols", X=design_matrix)

        # Check model stored
        assert hasattr(sim_brain_data, "model_")
        assert isinstance(sim_brain_data.model_, Glm)

        # Check GLM attributes set
        assert hasattr(sim_brain_data, "glm_betas")
        assert hasattr(sim_brain_data, "glm_t")

        # Predict on training data (fitted values)
        # Note: GLM doesn't support prediction with new design matrices yet
        predictions = sim_brain_data.predict()

        # Check predictions match training data shape
        assert predictions.shape == sim_brain_data.shape

    def test_fit_uses_brain_data_as_target(self, sim_brain_data):
        """Test fit() always uses self.data as y target."""
        X = np.random.randn(len(sim_brain_data), 10)

        # Fit Ridge
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X)

        # Model should be fitted to (X, sim_brain_data.data)
        # Check by predicting and comparing shapes
        predictions = sim_brain_data.predict(X=X)
        assert predictions.shape == sim_brain_data.shape

    @pytest.mark.slow
    def test_fit_passes_kwargs_to_model(self, sim_brain_data):
        """Test fit() passes additional kwargs to model constructor."""
        X = np.random.randn(len(sim_brain_data), 10)

        # Ridge with backend kwarg
        sim_brain_data.fit(model="ridge", alpha=1.0, backend="numpy", X=X)
        assert sim_brain_data.model_.backend == "numpy"

        # GLM with noise_model kwarg
        design_matrix = pd.DataFrame({"Intercept": np.ones(len(sim_brain_data))})
        sim_brain_data.fit(model="glm", noise_model="ar1", X=design_matrix)
        assert sim_brain_data.model_.noise_model == "ar1"

    def test_predict_requires_fitted_model(self, sim_brain_data):
        """Test predict() raises error if fit() not called first."""
        # Get a fresh copy (fixture may be contaminated by previous tests)
        bd = sim_brain_data.copy()

        # Explicitly remove model attributes to test the error case
        # (copy shares model_ from fitted instances due to pickle handling)
        for attr in ["model_", "X_"]:
            if hasattr(bd, attr):
                delattr(bd, attr)

        with pytest.raises(ValueError, match="Must call fit"):
            bd.predict()

    def test_predict_validates_X_dimensions(self, sim_brain_data):
        """Test predict() validates X has correct n_features."""
        # Fit with 10 features
        X_train = np.random.randn(len(sim_brain_data), 10)
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X_train)

        # Try to predict with 5 features - should fail
        X_wrong = np.random.randn(15, 5)
        with pytest.raises(ValueError, match="features"):
            sim_brain_data.predict(X=X_wrong)

    def test_ridge_weights_structure(self, sim_brain_data):
        """Test Ridge weights stored correctly as BrainData."""
        from nltools.data import BrainData

        X = np.random.randn(len(sim_brain_data), 10)
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X)

        # Weights should be BrainData
        assert isinstance(sim_brain_data.ridge_weights, BrainData)

        # Shape: (n_features, n_voxels)
        assert sim_brain_data.ridge_weights.shape == (10, sim_brain_data.shape[1])

        # Should have same mask
        assert sim_brain_data.ridge_weights.mask is sim_brain_data.mask

    # ==================== Fit inplace parameter tests ====================

    def test_fit_inplace_true_backward_compatible(self, sim_brain_data):
        """Test inplace=True preserves backward compatibility."""
        import numpy as np

        X_train = np.random.randn(len(sim_brain_data), 10)

        # Fit with inplace=True (default)
        result = sim_brain_data.fit(model="ridge", alpha=1.0, X=X_train, inplace=True)

        # Should return self
        assert result is sim_brain_data

        # Should have mutated attributes
        assert hasattr(sim_brain_data, "ridge_weights")
        assert hasattr(sim_brain_data, "ridge_fitted_values")
        assert hasattr(sim_brain_data, "ridge_scores")
        assert hasattr(sim_brain_data, "model_")
        assert hasattr(sim_brain_data, "X_")

    def test_fit_inplace_false_returns_fit_dataclass_ridge(self, sim_brain_data):
        """Test inplace=False returns Fit dataclass for Ridge."""
        from nltools.data.fitresults import Fit
        import numpy as np

        # Use a fresh copy to avoid contamination from previous tests
        brain = sim_brain_data.copy()
        # Clean up any existing fit attributes that might have been copied
        for attr in [
            "ridge_weights",
            "ridge_fitted_values",
            "ridge_scores",
            "glm_betas",
            "glm_t",
            "glm_p",
            "glm_se",
            "glm_residual",
            "glm_predicted",
            "glm_r2",
            "cv_results_",
            "model_",
            "X_",
        ]:
            if hasattr(brain, attr):
                delattr(brain, attr)

        X_train = np.random.randn(len(brain), 10)
        original_data = brain.data.copy()

        # Fit with inplace=False
        fit = brain.fit(model="ridge", alpha=1.0, X=X_train, inplace=False)

        # Should return Fit dataclass
        assert isinstance(fit, Fit)

        # Should have correct fields
        assert "fitted_values" in fit.available()
        assert "weights" in fit.available()
        assert "scores" in fit.available()

        # Check shapes
        assert fit.fitted_values.shape == brain.shape
        assert fit.weights.shape == (10, brain.shape[1])
        assert fit.scores.shape == (brain.shape[1],)

        # BrainData should not have result attributes
        assert not hasattr(brain, "ridge_weights")
        assert not hasattr(brain, "ridge_fitted_values")
        assert not hasattr(brain, "ridge_scores")

        # But should have model_ and X_ for predict()
        assert hasattr(brain, "model_")
        assert hasattr(brain, "X_")

        # Data should be unchanged
        np.testing.assert_array_equal(brain.data, original_data)

    @pytest.mark.slow
    def test_fit_inplace_false_returns_fit_dataclass_ridge_cv(self, sim_brain_data):
        """Test inplace=False returns Fit dataclass with CV results for Ridge."""
        from nltools.data.fitresults import Fit
        import numpy as np

        # Use a fresh copy to avoid contamination
        brain = sim_brain_data.copy()
        X_train = np.random.randn(len(brain), 10)

        # Fit with CV and inplace=False
        fit = brain.fit(model="ridge", alpha=1.0, X=X_train, cv=3, inplace=False)

        # Should return Fit dataclass
        assert isinstance(fit, Fit)

        # Should have CV fields
        assert "cv_scores" in fit.available()
        assert "cv_mean_score" in fit.available()
        assert "cv_predictions" in fit.available()
        assert "cv_folds" in fit.available()

        # Check shapes
        assert fit.cv_scores.shape == (3, brain.shape[1])
        assert fit.cv_mean_score.shape == (brain.shape[1],)
        assert fit.cv_predictions.shape == brain.shape
        assert fit.cv_folds.shape == (len(brain),)

        # BrainData should not have cv_results_
        assert not hasattr(brain, "cv_results_")

    @pytest.mark.slow
    def test_fit_inplace_false_returns_fit_dataclass_glm(self, sim_brain_data):
        """Test inplace=False returns Fit dataclass for GLM."""
        from nltools.data.fitresults import Fit
        import numpy as np
        import pandas as pd

        # Use a fresh copy to avoid contamination
        brain = sim_brain_data.copy()
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(brain)),
                "X1": np.random.randn(len(brain)),
            }
        )
        original_data = brain.data.copy()

        # Fit with inplace=False
        fit = brain.fit(model="glm", noise_model="ols", X=design_matrix, inplace=False)

        # Should return Fit dataclass
        assert isinstance(fit, Fit)

        # Should have GLM fields
        assert "fitted_values" in fit.available()
        assert "betas" in fit.available()
        assert "t_stats" in fit.available()
        assert "p_values" in fit.available()
        assert "se" in fit.available()
        assert "residuals" in fit.available()
        assert "r2" in fit.available()

        # Check shapes
        assert fit.fitted_values.shape == brain.shape
        assert fit.betas.shape == (2, brain.shape[1])  # 2 regressors
        assert fit.t_stats.shape == (2, brain.shape[1])
        assert fit.p_values.shape == (2, brain.shape[1])
        assert fit.se.shape == (2, brain.shape[1])
        assert fit.residuals.shape == brain.shape
        assert fit.r2.shape == (brain.shape[1],)

        # BrainData should not have GLM result attributes
        assert not hasattr(brain, "glm_betas")
        assert not hasattr(brain, "glm_t")
        assert not hasattr(brain, "glm_p")

        # But should have model_ and design_matrix for predict() and compute_contrasts()
        assert hasattr(brain, "model_")
        assert hasattr(brain, "design_matrix")

        # Data should be unchanged
        np.testing.assert_array_equal(brain.data, original_data)

    def test_fit_inplace_false_allows_predict(self, sim_brain_data):
        """Test that inplace=False still allows predict() to work."""
        import numpy as np

        X_train = np.random.randn(len(sim_brain_data), 10)

        # Fit with inplace=False
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X_train, inplace=False)

        # Should be able to predict (model_ and X_ stored)
        X_test = np.random.randn(20, 10)
        predictions = sim_brain_data.predict(X=X_test)

        assert predictions.shape == (20, sim_brain_data.shape[1])

    def test_fit_inplace_false_serialization(self, sim_brain_data):
        """Test Fit dataclass serialization roundtrip."""
        from nltools.data.fitresults import Fit
        import numpy as np
        import os

        X_train = np.random.randn(len(sim_brain_data), 10)

        # Fit with inplace=False
        fit = sim_brain_data.fit(model="ridge", alpha=1.0, X=X_train, inplace=False)

        # Serialize
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **fit.asdict())

            # Load
            loaded = np.load(f.name)
            fit_reconstructed = Fit(**{k: loaded[k] for k in loaded.files})

            # Clean up
            os.unlink(f.name)

        # Check fields match
        np.testing.assert_array_equal(
            fit.fitted_values, fit_reconstructed.fitted_values
        )
        np.testing.assert_array_equal(fit.weights, fit_reconstructed.weights)
        np.testing.assert_array_equal(fit.scores, fit_reconstructed.scores)

    def test_fit_inplace_default_is_true(self, sim_brain_data):
        """Test that inplace defaults to True for backward compatibility."""
        import numpy as np

        X_train = np.random.randn(len(sim_brain_data), 10)

        # Fit without specifying inplace (should default to True)
        result = sim_brain_data.fit(model="ridge", alpha=1.0, X=X_train)

        # Should return self and mutate attributes
        assert result is sim_brain_data
        assert hasattr(sim_brain_data, "ridge_weights")
        # Verify progress_bar parameter exists on model (defaults to verbose=False)
        assert hasattr(sim_brain_data.model_, "progress_bar")
        assert sim_brain_data.model_.progress_bar is False

    @pytest.mark.slow
    def test_glm_fit_numerical_correctness(self, sim_brain_data):
        """Test fit(model='glm') produces numerically correct results."""

        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data)),
                "X1": np.random.randn(len(sim_brain_data)),
            }
        )

        # Fit GLM model
        sim_brain_data.fit(model="glm", noise_model="ols", X=design_matrix)

        # Check betas are reasonable (not NaN, not all zeros)
        assert not np.isnan(sim_brain_data.glm_betas.data).any()
        assert not np.allclose(sim_brain_data.glm_betas.data, 0)

        # Check t-statistics are reasonable
        assert not np.isnan(sim_brain_data.glm_t.data).any()
        # Verify progress_bar parameter exists and defaults to False
        assert hasattr(sim_brain_data.model_, "progress_bar")
        assert sim_brain_data.model_.progress_bar is False

    @pytest.mark.slow
    def test_glm_fit_suppresses_drift_model_warning(self, sim_brain_data):
        """Test fit(model='glm') suppresses drift_model warning when design matrices are supplied"""
        import warnings

        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(sim_brain_data)),
                "X1": np.random.randn(len(sim_brain_data)),
            }
        )

        # Capture warnings during fit
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Capture all warnings
            # Fit GLM with drift_model set (would trigger warning without suppression)
            sim_brain_data.fit(
                model="glm", noise_model="ols", X=design_matrix, drift_model="cosine"
            )

            # Check that drift_model warning is NOT present
            drift_warnings = [
                warn
                for warn in w
                if "drift_model" in str(warn.message).lower()
                and "will be ignored" in str(warn.message).lower()
            ]
            assert len(drift_warnings) == 0, (
                f"Expected no drift_model warnings, but got {len(drift_warnings)}: "
                f"{[str(w.message) for w in drift_warnings]}"
            )

        # Verify model was fitted successfully
        assert hasattr(sim_brain_data, "model_")
        assert sim_brain_data.model_.is_fitted_
        # Verify progress_bar parameter exists and defaults to False
        assert hasattr(sim_brain_data.model_, "progress_bar")
        assert sim_brain_data.model_.progress_bar is False

        # Test with progress_bar=True to verify it's respected
        sim_brain_data.fit(
            model="glm", noise_model="ols", X=design_matrix, progress_bar=True
        )
        assert sim_brain_data.model_.progress_bar is True
        assert sim_brain_data.model_.is_fitted_

    def test_fit_validates_model_name(self, sim_brain_data):
        """Test fit() raises error for unknown model names."""
        X = np.random.randn(len(sim_brain_data), 10)

        with pytest.raises(ValueError, match="Unknown model"):
            sim_brain_data.fit(model="unknown_model", X=X)

    def test_fit_validates_X_shape(self, sim_brain_data):
        """Test fit() validates X has correct n_samples."""
        # X has wrong number of samples
        X_wrong = np.random.randn(len(sim_brain_data) + 5, 10)

        with pytest.raises(ValueError, match="number of samples"):
            sim_brain_data.fit(model="ridge", alpha=1.0, X=X_wrong)

    def test_fit_scale_default_true(self, sim_brain_data):
        """Test fit() applies scaling by default (scale=True)."""
        X = np.random.randn(len(sim_brain_data), 10)
        original_mean = sim_brain_data.data.mean()

        # Fit with default scale=True
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X)

        # Data should be scaled (mean should be ~100 after grand-mean scaling)
        assert sim_brain_data.data.mean() != original_mean
        # After scaling, mean should be close to scale_value (100)
        np.testing.assert_allclose(sim_brain_data.data.mean(), 100.0, rtol=0.1)

    def test_fit_scale_false_preserves_data(self, sim_brain_data):
        """Test fit() with scale=False preserves original data values."""
        X = np.random.randn(len(sim_brain_data), 10)
        original_data = sim_brain_data.data.copy()

        # Fit with scale=False
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X, scale=False)

        # Data should be unchanged
        np.testing.assert_allclose(sim_brain_data.data, original_data)

    def test_fit_scale_value_custom(self, sim_brain_data):
        """Test fit() respects custom scale_value."""
        X = np.random.randn(len(sim_brain_data), 10)

        # Fit with custom scale_value
        sim_brain_data.fit(
            model="ridge", alpha=1.0, X=X, scale=True, scale_value=1000.0
        )

        # Mean should be close to custom scale_value
        np.testing.assert_allclose(sim_brain_data.data.mean(), 1000.0, rtol=0.1)

    def test_fit_scale_inplace_false(self, sim_brain_data):
        """Test fit() with scale=True and inplace=False doesn't modify original."""
        from nltools.data.fitresults import Fit

        X = np.random.randn(len(sim_brain_data), 10)
        original_data = sim_brain_data.data.copy()

        # Fit with inplace=False and scale=True
        result = sim_brain_data.fit(
            model="ridge", alpha=1.0, X=X, inplace=False, scale=True
        )

        # Should return Fit dataclass
        assert isinstance(result, Fit)

        # Original data should be unchanged (scaling was applied to copy)
        np.testing.assert_allclose(sim_brain_data.data, original_data)

    def test_predict_with_no_X_uses_training_data(self, sim_brain_data):
        """Test predict() with no X returns predictions on training data."""
        X_train = np.random.randn(len(sim_brain_data), 10)
        sim_brain_data.fit(model="ridge", alpha=1.0, X=X_train)

        # Predict with explicit X
        predictions_explicit = sim_brain_data.predict(X=X_train)

        # Predict with no X (should use training data)
        predictions_implicit = sim_brain_data.predict()

        # Should be identical
        np.testing.assert_allclose(predictions_explicit.data, predictions_implicit.data)

        # Should match training data shape
        assert predictions_implicit.shape == sim_brain_data.shape

    # ==================== Ridge CV Tests ====================

    def test_fit_ridge_cv_basic_integer(self, small_brain_data_for_cv):
        """Test fit() with cv=3 returns cross-validated scores for fixed alpha."""
        brain_data, X = small_brain_data_for_cv

        # Fit with CV and fixed alpha
        brain_data.fit(model="ridge", alpha=1.0, cv=3, X=X)

        # CV results should exist
        assert hasattr(brain_data, "cv_results_")
        assert isinstance(brain_data.cv_results_, dict)

        # Check expected keys
        assert "scores" in brain_data.cv_results_
        assert "mean_score" in brain_data.cv_results_
        assert "predictions" in brain_data.cv_results_
        assert "folds" in brain_data.cv_results_

        # Check shapes
        cv_scores = brain_data.cv_results_["scores"]
        assert cv_scores.shape == (3, 5)  # (n_folds=3, n_voxels=5)

        mean_score = brain_data.cv_results_["mean_score"]
        assert mean_score.shape == (5,)  # Per-voxel mean

        # Check fold indices
        folds = brain_data.cv_results_["folds"]
        assert len(folds) == 24  # n_samples
        assert set(folds) == {0, 1, 2}  # Fold IDs

        # Regular fit attributes should still exist
        assert hasattr(brain_data, "ridge_weights")
        assert hasattr(brain_data, "ridge_fitted_values")

    def test_fit_ridge_cv_sklearn_splitter(self, small_brain_data_for_cv):
        """Test fit() accepts sklearn CV splitter objects."""

        brain_data, X = small_brain_data_for_cv

        # Create CV splitter
        cv_splitter = KFold(n_splits=3, shuffle=True, random_state=42)

        # Fit with CV splitter
        brain_data.fit(model="ridge", alpha=1.0, cv=cv_splitter, X=X)

        # CV results should exist with same structure
        assert hasattr(brain_data, "cv_results_")
        assert brain_data.cv_results_["scores"].shape == (3, 5)

        # Test reproducibility - fit again with same random_state
        brain_data2, X2 = small_brain_data_for_cv
        cv_splitter2 = KFold(n_splits=3, shuffle=True, random_state=42)
        brain_data2.fit(model="ridge", alpha=1.0, cv=cv_splitter2, X=X2)

        # Should get identical results
        np.testing.assert_allclose(
            brain_data.cv_results_["mean_score"], brain_data2.cv_results_["mean_score"]
        )

    def test_fit_ridge_cv_predictions(self, small_brain_data_for_cv):
        """Test CV predictions are out-of-fold and stored as BrainData."""
        brain_data, X = small_brain_data_for_cv

        # Fit with CV
        brain_data.fit(model="ridge", alpha=1.0, cv=3, X=X)

        # Check predictions structure
        cv_preds = brain_data.cv_results_["predictions"]
        assert isinstance(cv_preds, BrainData)
        assert cv_preds.shape == (24, 5)  # (n_samples, n_voxels)

        # CV predictions should differ from full model predictions
        # (out-of-fold vs. in-sample)
        full_preds = brain_data.ridge_fitted_values
        assert not np.allclose(cv_preds.data, full_preds.data)

        # Sanity checks on R² values
        # Note: Out-of-sample R² can be negative (model worse than mean)
        cv_r2 = np.mean(brain_data.cv_results_["mean_score"])
        full_r2 = np.mean(brain_data.ridge_scores.data)

        # Just check both are finite and reasonable (not NaN/Inf)
        assert np.isfinite(cv_r2)
        assert np.isfinite(full_r2)
        # Full R² should generally be non-negative (in-sample)
        assert full_r2 >= -0.1  # Allow small numerical errors

    def test_fit_ridge_cv_auto_alpha_selection(self, small_brain_data_for_cv):
        """Test cv='auto' triggers alpha selection."""
        brain_data, X = small_brain_data_for_cv

        # Fit with cv='auto' (implies alpha='auto')
        alphas = [0.1, 1.0, 10.0]  # Small grid for speed
        brain_data.fit(model="ridge", cv="auto", alphas=alphas, X=X)

        # CV results should exist
        assert hasattr(brain_data, "cv_results_")

        # Alpha selection results
        assert "best_alpha" in brain_data.cv_results_
        assert "alpha_scores" in brain_data.cv_results_

        # Best alpha should be one of the tested alphas
        best_alpha = brain_data.cv_results_["best_alpha"]
        assert best_alpha in alphas

        # Alpha scores shape: (n_folds, n_alphas, n_voxels)
        alpha_scores = brain_data.cv_results_["alpha_scores"]
        assert alpha_scores.shape == (
            5,
            3,
            5,
        )  # (5 folds default for 'auto', 3 alphas, 5 voxels)

        # Model should be fitted with best_alpha
        assert brain_data.model_.alpha == best_alpha

    def test_fit_ridge_cv_integer_with_alpha_auto(self, small_brain_data_for_cv):
        """Test cv=int with alpha='auto' performs both alpha selection and CV scoring."""
        brain_data, X = small_brain_data_for_cv

        # Fit with explicit alpha selection + CV
        alphas = [0.1, 1.0, 10.0]
        brain_data.fit(model="ridge", alpha="auto", cv=3, alphas=alphas, X=X)

        # Should have both alpha selection and CV scoring results
        assert "best_alpha" in brain_data.cv_results_
        assert "alpha_scores" in brain_data.cv_results_
        assert "scores" in brain_data.cv_results_
        assert "mean_score" in brain_data.cv_results_

        # Alpha scores: (n_folds=3, n_alphas=3, n_voxels=5)
        assert brain_data.cv_results_["alpha_scores"].shape == (3, 3, 5)

        # CV scores computed with best alpha: (n_folds=3, n_voxels=5)
        assert brain_data.cv_results_["scores"].shape == (3, 5)

        # Best alpha selected
        assert brain_data.cv_results_["best_alpha"] in alphas

    def test_fit_ridge_no_cv_backward_compat(self, small_brain_data_for_cv):
        """Test fit() without cv parameter doesn't create cv_results_ (backward compat)."""
        brain_data, X = small_brain_data_for_cv

        # Fit without CV (existing behavior)
        brain_data.fit(model="ridge", alpha=1.0, X=X)

        # CV results should NOT exist
        assert not hasattr(brain_data, "cv_results_")

        # Regular attributes should exist
        assert hasattr(brain_data, "ridge_weights")
        assert hasattr(brain_data, "ridge_fitted_values")
        assert hasattr(brain_data, "ridge_scores")

    def test_fit_ridge_cv_invalid_parameter(self, small_brain_data_for_cv):
        """Test fit() raises errors for invalid cv parameters."""
        brain_data, X = small_brain_data_for_cv

        # Invalid cv type
        with pytest.raises((TypeError, ValueError)):
            brain_data.fit(model="ridge", alpha=1.0, cv="invalid", X=X)

        # Negative cv
        with pytest.raises(ValueError):
            brain_data.fit(model="ridge", alpha=1.0, cv=-1, X=X)

        # Zero cv
        with pytest.raises(ValueError):
            brain_data.fit(model="ridge", alpha=1.0, cv=0, X=X)

    def test_fit_ridge_cv_with_insufficient_samples(self, tiny_brain_data_for_cv):
        """Test fit() raises error when cv folds > n_samples."""
        brain_data, X = tiny_brain_data_for_cv  # Only 6 samples

        # Try 10-fold CV with 6 samples
        with pytest.raises(ValueError, match="Cannot have number of splits.*greater"):
            brain_data.fit(model="ridge", alpha=1.0, cv=10, X=X)

    def test_fit_ridge_cv_predict_consistency(self, small_brain_data_for_cv):
        """Test predict() returns full model predictions, not CV predictions."""
        brain_data, X = small_brain_data_for_cv

        # Fit with CV
        brain_data.fit(model="ridge", alpha=1.0, cv=3, X=X)

        # Call predict() on training data
        train_predictions = brain_data.predict(X=X)

        # Should match full model predictions (ridge_fitted_values)
        np.testing.assert_allclose(
            train_predictions.data, brain_data.ridge_fitted_values.data
        )

        # Should NOT match CV predictions (out-of-fold)
        assert not np.allclose(
            train_predictions.data, brain_data.cv_results_["predictions"].data
        )

    def test_fit_ridge_cv_stores_all_expected_keys(self, small_brain_data_for_cv):
        """Test cv_results_ dict contains all expected keys and types."""
        brain_data, X = small_brain_data_for_cv

        # Fit with alpha selection
        alphas = [0.1, 1.0, 10.0]
        brain_data.fit(model="ridge", alpha="auto", cv=3, alphas=alphas, X=X)

        # Check all expected keys exist
        expected_keys = {
            "scores",
            "mean_score",
            "predictions",
            "folds",
            "best_alpha",
            "alpha_scores",
        }
        assert set(brain_data.cv_results_.keys()) == expected_keys

        # Check types
        assert isinstance(brain_data.cv_results_["scores"], np.ndarray)
        assert isinstance(brain_data.cv_results_["mean_score"], np.ndarray)
        assert isinstance(brain_data.cv_results_["predictions"], BrainData)
        assert isinstance(brain_data.cv_results_["folds"], np.ndarray)
        assert isinstance(brain_data.cv_results_["best_alpha"], (int, float))
        assert isinstance(brain_data.cv_results_["alpha_scores"], np.ndarray)
