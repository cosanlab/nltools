import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from nltools.data import BrainData, DesignMatrix


def _brain_data_from_array(values):
    """Build a BrainData whose masked data holds `values`, shape (n_obs, n_voxels).

    Images are passed as a list so a one-voxel stack keeps its voxel axis (the
    constructor squeezes singleton axes out of array and 4-D inputs).
    """
    import nibabel as nib

    values = np.asarray(values, dtype=float)
    n_samples, n_voxels = values.shape
    spatial_shape = (n_voxels, 1, 1)
    affine = np.eye(4)
    images = [
        nib.Nifti1Image(values[t].reshape(spatial_shape), affine)
        for t in range(n_samples)
    ]
    return BrainData(
        images,
        mask=nib.Nifti1Image(np.ones(spatial_shape, dtype=np.float32), affine),
    )


#: The Himalaya slice (q6at) retired `alphas=`, `alpha="auto"`, `fit_intercept=`,
#: `local_alpha=`, and the second cross-validation pass behind `cv_results_`. Aligning
#: the `BrainData` facade to the new estimator is Kata e5y6.
e5y6_pending = pytest.mark.xfail(reason="e5y6: facade alignment pending", strict=True)


class TestBrainDataModeling:
    # ==================== Unified fit/predict API ====================

    def test_fit_predict_ridge_workflow(self, minimal_brain_data):
        """Test complete Ridge fit/predict workflow."""
        from nltools.models import Ridge

        X_train = np.random.randn(len(minimal_brain_data), 10)
        minimal_brain_data.fit(model="ridge", alpha=1.0, X=X_train)

        # Check model stored
        assert hasattr(minimal_brain_data, "model_")
        assert isinstance(minimal_brain_data.model_, Ridge)
        assert minimal_brain_data.model_.is_fitted_

        # Check attributes set
        assert hasattr(minimal_brain_data, "ridge_weights")
        assert hasattr(minimal_brain_data, "ridge_fitted_values")
        assert hasattr(minimal_brain_data, "ridge_scores")

        # Predict on new data
        X_test = np.random.randn(20, 10)
        predictions = minimal_brain_data.predict(X=X_test)
        assert isinstance(predictions, BrainData)
        assert predictions.shape == (20, minimal_brain_data.shape[1])

        # Predict on training data (X=None) uses self.data as target
        train_predictions = minimal_brain_data.predict()
        assert train_predictions.shape == minimal_brain_data.shape

    @pytest.mark.slow
    def test_fit_predict_glm_workflow(self, minimal_brain_data):
        """Test complete GLM fit/predict workflow."""
        from nltools.data import DesignMatrix
        from nltools.models import Glm

        design_matrix = DesignMatrix(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "X1": np.random.randn(len(minimal_brain_data)),
            }
        )
        minimal_brain_data.fit(model="glm", glm_noise_model="ols", X=design_matrix)

        assert hasattr(minimal_brain_data, "model_")
        assert isinstance(minimal_brain_data.model_, Glm)
        assert hasattr(minimal_brain_data, "glm_betas")

        predictions = minimal_brain_data.predict()
        assert predictions.shape == minimal_brain_data.shape

    @pytest.mark.slow
    def test_fit_forwards_model_options_to_the_estimator(self, minimal_brain_data):
        """Prefixed and ridge options reach their estimator's constructor."""
        from nltools.data import DesignMatrix

        X = np.random.randn(len(minimal_brain_data), 10)

        minimal_brain_data.fit(model="ridge", alpha=1.0, device="cpu", X=X)
        assert minimal_brain_data.model_.device == "cpu"

        design_matrix = DesignMatrix(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "X1": np.random.randn(len(minimal_brain_data)),
            }
        )
        minimal_brain_data.fit(model="glm", glm_noise_model="ar1", X=design_matrix)
        assert minimal_brain_data.model_.noise_model == "ar1"

    def test_fit_ridge_rejects_backend_kwarg(self, minimal_brain_data):
        """The retired `backend=`/`parallel=` device aliases are rejected at the facade."""
        X = np.random.randn(len(minimal_brain_data), 10)
        with pytest.raises(TypeError, match="device="):
            minimal_brain_data.fit(model="ridge", alpha=1.0, backend="numpy", X=X)
        with pytest.raises(TypeError, match="device="):
            minimal_brain_data.fit(model="ridge", alpha=1.0, parallel="cpu", X=X)

    def test_predict_requires_fitted_model(self, minimal_brain_data):
        """Test predict() raises error if fit() not called first."""
        bd = minimal_brain_data.copy()
        for attr in ["model_", "X_"]:
            if hasattr(bd, attr):
                delattr(bd, attr)

        with pytest.raises(ValueError, match="Must call fit"):
            bd.predict()

    def test_predict_validates_X_dimensions(self, minimal_brain_data):
        """Test predict() validates X has correct n_features."""
        X_train = np.random.randn(len(minimal_brain_data), 10)
        minimal_brain_data.fit(model="ridge", alpha=1.0, X=X_train)

        X_wrong = np.random.randn(15, 5)
        with pytest.raises(ValueError, match="features"):
            minimal_brain_data.predict(X=X_wrong)

    def test_ridge_weights_structure(self, minimal_brain_data):
        """Test Ridge weights stored correctly as BrainData."""
        X = np.random.randn(len(minimal_brain_data), 10)
        minimal_brain_data.fit(model="ridge", alpha=1.0, X=X)

        assert isinstance(minimal_brain_data.ridge_weights, BrainData)
        assert minimal_brain_data.ridge_weights.shape == (
            10,
            minimal_brain_data.shape[1],
        )
        assert minimal_brain_data.ridge_weights.mask is not minimal_brain_data.mask

    # ==================== Fit inplace parameter tests ====================

    def test_fit_inplace_default_true(self, minimal_brain_data):
        """Test inplace=True (default) preserves backward compatibility."""
        X_train = np.random.randn(len(minimal_brain_data), 10)

        # Default (inplace=True)
        result = minimal_brain_data.fit(model="ridge", alpha=1.0, X=X_train)
        assert result is minimal_brain_data
        assert hasattr(minimal_brain_data, "ridge_weights")
        assert hasattr(minimal_brain_data, "ridge_fitted_values")
        assert hasattr(minimal_brain_data, "ridge_scores")
        assert hasattr(minimal_brain_data, "model_")
        assert hasattr(minimal_brain_data, "X_")
        assert minimal_brain_data.model_.progress_bar is False

    def test_fit_inplace_false_returns_independent_fitted_brain_data(
        self, minimal_brain_data
    ):
        """A non-inplace ridge fit owns its complete fitted state."""
        brain = minimal_brain_data.copy()
        for attr in [
            "ridge_weights",
            "ridge_fitted_values",
            "ridge_scores",
            "glm_betas",
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

        fitted = brain.fit(model="ridge", alpha=1.0, X=X_train, inplace=False)

        assert isinstance(fitted, BrainData)
        assert fitted is not brain
        assert fitted.ridge_fitted_values.shape == brain.shape
        assert fitted.ridge_weights.shape == (10, brain.shape[1])
        assert fitted.ridge_scores.shape == (1, brain.shape[1])
        assert not hasattr(brain, "ridge_weights")
        assert not hasattr(brain, "model_")
        assert not hasattr(brain, "X_")
        np.testing.assert_array_equal(brain.data, original_data)

        fitted.data[0, 0] = 123.0
        fitted.X_[0, 0] = 456.0
        fitted.ridge_weights.data[0, 0] = 789.0
        assert brain.data[0, 0] != 123.0
        assert X_train[0, 0] != 456.0
        assert fitted.model_.coef_[0, 0] != 789.0
        assert not hasattr(fitted.ridge_weights, "model_")

    @e5y6_pending
    @pytest.mark.slow
    def test_fit_inplace_false_returns_brain_data_with_ridge_cv(
        self, minimal_brain_data
    ):
        """The returned BrainData carries ridge cross-validation state."""
        brain = minimal_brain_data.copy()
        X_train = np.random.randn(len(brain), 10)

        fitted = brain.fit(model="ridge", alpha=1.0, X=X_train, cv=3, inplace=False)

        assert isinstance(fitted, BrainData)
        assert fitted.cv_results_["scores"].shape == (3, brain.shape[1])
        assert fitted.cv_results_["mean_score"].shape == (brain.shape[1],)
        assert fitted.cv_results_["predictions"].shape == brain.shape
        assert fitted.cv_results_["folds"].shape == (len(brain),)
        assert not hasattr(brain, "cv_results_")

    @pytest.mark.slow
    def test_fit_inplace_false_returns_brain_data_with_glm(self, minimal_brain_data):
        """The returned BrainData carries a complete independent GLM fit."""
        brain = minimal_brain_data.copy()
        design_matrix = DesignMatrix(
            {
                "Intercept": np.ones(len(brain)),
                "X1": np.random.randn(len(brain)),
            }
        )
        original_data = brain.data.copy()

        fitted = brain.fit(
            model="glm", glm_noise_model="ols", X=design_matrix, inplace=False
        )

        assert isinstance(fitted, BrainData)
        assert fitted.glm_predicted.shape == brain.shape
        assert fitted.glm_betas.shape == (2, brain.shape[1])
        assert hasattr(fitted, "glm_residual")
        assert hasattr(fitted, "glm_r2")
        assert not hasattr(fitted.glm_predicted, "model_")
        assert not hasattr(brain, "glm_betas")
        assert not hasattr(brain, "model_")
        np.testing.assert_array_equal(brain.data, original_data)

    def test_fit_inplace_false_result_allows_predict(self, minimal_brain_data):
        """Prediction belongs to the returned fitted copy, not the original."""
        X_train = np.random.randn(len(minimal_brain_data), 10)
        fitted = minimal_brain_data.fit(
            model="ridge", alpha=1.0, X=X_train, inplace=False
        )

        X_test = np.random.randn(20, 10)
        predictions = fitted.predict(X=X_test)
        assert predictions.shape == (20, minimal_brain_data.shape[1])

    @e5y6_pending
    def test_refit_without_cv_clears_prior_cv_results(self, minimal_brain_data):
        X = np.random.randn(len(minimal_brain_data), 4)
        minimal_brain_data.fit(model="ridge", X=X, alpha=1.0, cv=3)
        assert hasattr(minimal_brain_data, "cv_results_")

        minimal_brain_data.fit(model="ridge", X=X, alpha=1.0)

        assert not hasattr(minimal_brain_data, "cv_results_")

    @e5y6_pending
    def test_copy_owns_nested_cv_state(self, minimal_brain_data):
        X = np.random.randn(len(minimal_brain_data), 4)
        minimal_brain_data.fit(model="ridge", X=X, alpha=1.0, cv=3)

        copied = minimal_brain_data.copy()
        copied.cv_results_["scores"][0, 0] = 91.0
        copied.cv_results_["predictions"].data[0, 0] = 92.0

        assert minimal_brain_data.cv_results_["scores"][0, 0] != 91.0
        assert minimal_brain_data.cv_results_["predictions"].data[0, 0] != 92.0

    @pytest.mark.slow
    def test_refit_replaces_prior_model_state(self, minimal_brain_data):
        """Refitting across model types cannot leave mixed result state."""
        ridge_X = np.random.randn(len(minimal_brain_data), 3)
        minimal_brain_data.fit(model="ridge", X=ridge_X, alpha=1.0)

        glm_X = DesignMatrix(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condition": np.random.randn(len(minimal_brain_data)),
            }
        )
        minimal_brain_data.fit(model="glm", X=glm_X, glm_noise_model="ols")

        assert hasattr(minimal_brain_data, "glm_betas")
        assert not hasattr(minimal_brain_data, "ridge_weights")
        assert not hasattr(minimal_brain_data, "ridge_fitted_values")
        assert not hasattr(minimal_brain_data, "ridge_scores")

        minimal_brain_data.fit(model="ridge", X=ridge_X, alpha=1.0)

        assert hasattr(minimal_brain_data, "ridge_weights")
        assert not hasattr(minimal_brain_data, "glm_betas")
        assert not hasattr(minimal_brain_data, "glm_predicted")

    @pytest.mark.slow
    def test_non_inplace_refit_preserves_fitted_source(self, minimal_brain_data):
        ridge_X = np.random.randn(len(minimal_brain_data), 3)
        minimal_brain_data.fit(model="ridge", X=ridge_X, alpha=1.0)
        original_weights = minimal_brain_data.ridge_weights.data.copy()

        glm_X = DesignMatrix(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condition": np.random.randn(len(minimal_brain_data)),
            }
        )
        fitted_glm = minimal_brain_data.fit(
            model="glm", X=glm_X, glm_noise_model="ols", inplace=False
        )

        assert hasattr(fitted_glm, "glm_betas")
        assert not hasattr(fitted_glm, "ridge_weights")
        np.testing.assert_array_equal(
            minimal_brain_data.ridge_weights.data, original_weights
        )
        assert not hasattr(minimal_brain_data, "glm_betas")

    @pytest.mark.slow
    def test_fitted_glm_copy_is_usable_and_independent(self, minimal_brain_data):
        design = DesignMatrix(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condition": np.random.randn(len(minimal_brain_data)),
            }
        )
        minimal_brain_data.fit(model="glm", X=design, glm_noise_model="ols")
        original_coefficient = minimal_brain_data.model_.coef_[0, 0]
        minimal_brain_data.glm_betas.data[0, 0] = 95.0
        assert minimal_brain_data.model_.coef_[0, 0] == original_coefficient

        copied = minimal_brain_data.copy()
        copied_contrast = copied.compute_contrasts("condition")
        copied.glm_betas.data[0, 0] = 93.0
        copied.model_.coef_[0, 0] = 94.0

        assert isinstance(copied_contrast, BrainData)
        assert minimal_brain_data.glm_betas.data[0, 0] != 93.0
        assert minimal_brain_data.model_.coef_[0, 0] != 94.0

    @pytest.mark.slow
    def test_glm_fit_numerical_correctness(self, minimal_brain_data):
        """Test fit(model='glm') produces numerically correct results."""
        design_matrix = DesignMatrix(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "X1": np.random.randn(len(minimal_brain_data)),
            }
        )

        minimal_brain_data.fit(model="glm", glm_noise_model="ols", X=design_matrix)

        assert not np.isnan(minimal_brain_data.glm_betas.data).any()
        assert not np.allclose(minimal_brain_data.glm_betas.data, 0)
        assert not np.isnan(minimal_brain_data.glm_r2.data).any()

    def test_fit_validates_model_name(self, minimal_brain_data):
        """Test fit() raises error for unknown model names."""
        X = np.random.randn(len(minimal_brain_data), 10)
        with pytest.raises(TypeError, match="supported models are"):
            minimal_brain_data.fit(model="unknown_model", X=X)

    def test_fit_validates_X_shape(self, minimal_brain_data):
        """Test fit() validates X has correct n_samples."""
        X_wrong = np.random.randn(len(minimal_brain_data) + 5, 10)
        with pytest.raises(ValueError, match="number of samples"):
            minimal_brain_data.fit(model="ridge", alpha=1.0, X=X_wrong)

    @e5y6_pending
    def test_ridge_intercept_with_centering_warns(self, minimal_brain_data):
        """Ridge fit_intercept=True is redundant when the data is centered by
        standardization/scaling — warn loudly."""
        X = np.random.randn(len(minimal_brain_data), 10)
        bd = minimal_brain_data.copy()
        bd.data = bd.data + 100.0
        with pytest.warns(
            UserWarning, match="intercept.*redundant|redundant.*intercept"
        ):
            bd.fit(model="ridge", alpha=1.0, X=X, fit_intercept=True)

    @e5y6_pending
    def test_ridge_intercept_no_centering_ok(self, minimal_brain_data):
        """fit_intercept=True is fine (no warning) when no centering is applied —
        that is exactly the raw-offset case intercepts exist for."""
        import warnings

        X = np.random.randn(len(minimal_brain_data), 10)
        bd = minimal_brain_data.copy()
        bd.data = bd.data + 100.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bd.fit(model="ridge", alpha=1.0, X=X, fit_intercept=True)
        assert not any("intercept" in str(wi.message).lower() for wi in w)

    def test_predict_with_no_X_uses_training_data(self, minimal_brain_data):
        """Test predict() with no X returns predictions on training data."""
        X_train = np.random.randn(len(minimal_brain_data), 10)
        minimal_brain_data.fit(model="ridge", alpha=1.0, X=X_train)

        predictions_explicit = minimal_brain_data.predict(X=X_train)
        predictions_implicit = minimal_brain_data.predict()

        np.testing.assert_allclose(predictions_explicit.data, predictions_implicit.data)
        assert predictions_implicit.shape == minimal_brain_data.shape

    # ==================== Ridge CV Tests ====================

    @e5y6_pending
    def test_fit_ridge_cv_basic(self, small_brain_data_for_cv):
        """Test fit() with cv=int and sklearn splitter returns cross-validated scores."""
        brain_data, X = small_brain_data_for_cv

        # Test with integer
        brain_data.fit(model="ridge", alpha=1.0, cv=3, X=X)

        assert hasattr(brain_data, "cv_results_")
        assert isinstance(brain_data.cv_results_, dict)
        assert "scores" in brain_data.cv_results_
        assert "mean_score" in brain_data.cv_results_
        assert "predictions" in brain_data.cv_results_
        assert "folds" in brain_data.cv_results_

        cv_scores = brain_data.cv_results_["scores"]
        assert cv_scores.shape == (3, 5)  # (n_folds=3, n_voxels=5)
        assert brain_data.cv_results_["mean_score"].shape == (5,)
        assert set(brain_data.cv_results_["folds"]) == {0, 1, 2}
        assert hasattr(brain_data, "ridge_weights")

        # Test with sklearn splitter (reproducibility)
        brain_data2, X2 = small_brain_data_for_cv
        cv_splitter = KFold(n_splits=3, shuffle=True, random_state=42)
        brain_data2.fit(model="ridge", alpha=1.0, cv=cv_splitter, X=X2)
        assert brain_data2.cv_results_["scores"].shape == (3, 5)

    @e5y6_pending
    def test_fit_ridge_cv_predictions(self, small_brain_data_for_cv):
        """Test CV predictions are out-of-fold and stored as BrainData."""
        brain_data, X = small_brain_data_for_cv
        brain_data.fit(model="ridge", alpha=1.0, cv=3, X=X)

        cv_preds = brain_data.cv_results_["predictions"]
        assert isinstance(cv_preds, BrainData)
        assert cv_preds.shape == (24, 5)

        # Out-of-fold should differ from in-sample
        full_preds = brain_data.ridge_fitted_values
        assert not np.allclose(cv_preds.data, full_preds.data)

        assert np.isfinite(np.mean(brain_data.cv_results_["mean_score"]))

    @e5y6_pending
    def test_fit_ridge_cv_alpha_auto(self, small_brain_data_for_cv):
        """alpha='auto' triggers per-voxel α selection by default (v0.6).

        Breaking change: cv_results_['best_alpha'] is now (n_voxels,)
        when local_alpha=True (the new default). Pass local_alpha=False
        to get the legacy single-α-for-all-voxels behavior.
        """
        brain_data, X = small_brain_data_for_cv

        alphas = [0.1, 1.0, 10.0]
        brain_data.fit(model="ridge", alpha="auto", cv=3, alphas=alphas, X=X)

        # Should have both alpha selection and CV scoring results
        assert "best_alpha" in brain_data.cv_results_
        assert "alpha_scores" in brain_data.cv_results_
        assert "scores" in brain_data.cv_results_
        assert "mean_score" in brain_data.cv_results_

        # Per-voxel α: array of shape (n_voxels,), each entry from the alpha grid.
        best = brain_data.cv_results_["best_alpha"]
        assert isinstance(best, np.ndarray)
        assert best.shape == (5,)  # 5 voxels in the fixture
        assert np.all(np.isin(best, alphas))
        assert brain_data.cv_results_["alpha_scores"].shape == (3, 3, 5)
        assert brain_data.cv_results_["scores"].shape == (3, 5)
        # Model exposes the same per-voxel α via .alpha_ (post-fit attribute).
        np.testing.assert_array_equal(brain_data.model_.alpha_, best)

        # Check all expected keys and types
        expected_keys = {
            "scores",
            "mean_score",
            "predictions",
            "folds",
            "best_alpha",
            "alpha_scores",
        }
        assert set(brain_data.cv_results_.keys()) == expected_keys
        assert isinstance(brain_data.cv_results_["predictions"], BrainData)

    def test_fit_ridge_no_cv_backward_compat(self, small_brain_data_for_cv):
        """Test fit() without cv parameter doesn't create cv_results_."""
        brain_data, X = small_brain_data_for_cv
        brain_data.fit(model="ridge", alpha=1.0, X=X)

        assert not hasattr(brain_data, "cv_results_")
        assert hasattr(brain_data, "ridge_weights")

    def test_fit_ridge_cv_invalid_parameter(self, small_brain_data_for_cv):
        """Test fit() raises errors for invalid cv parameters."""
        brain_data, X = small_brain_data_for_cv

        with pytest.raises((TypeError, ValueError)):
            brain_data.fit(model="ridge", alpha=1.0, cv="invalid", X=X)

        with pytest.raises(ValueError):
            brain_data.fit(model="ridge", alpha=1.0, cv=-1, X=X)

        with pytest.raises(ValueError):
            brain_data.fit(model="ridge", alpha=1.0, cv=0, X=X)

    @e5y6_pending
    def test_fit_ridge_cv_with_insufficient_samples(self, tiny_brain_data_for_cv):
        """Test fit() raises error when cv folds > n_samples."""
        brain_data, X = tiny_brain_data_for_cv
        with pytest.raises(ValueError, match="Cannot have number of splits.*greater"):
            brain_data.fit(model="ridge", alpha=1.0, cv=10, X=X)

    @e5y6_pending
    def test_fit_ridge_cv_predict_consistency(self, small_brain_data_for_cv):
        """Test predict() returns full model predictions, not CV predictions."""
        brain_data, X = small_brain_data_for_cv
        brain_data.fit(model="ridge", alpha=1.0, cv=3, X=X)

        train_predictions = brain_data.predict(X=X)
        np.testing.assert_allclose(
            train_predictions.data, brain_data.ridge_fitted_values.data
        )
        assert not np.allclose(
            train_predictions.data, brain_data.cv_results_["predictions"].data
        )

    # ============ design estimated as given + rank diagnostics (GLM) ============

    @pytest.mark.slow
    def test_fit_estimates_every_column_given(self, minimal_brain_data):
        """fit() does not silently drop correlated regressors.

        Two regressors correlated at r=0.99 are collinear in the colloquial
        sense but the design is still full rank and estimable, so every column
        must appear in the betas. Dropping is the caller's decision, made
        explicitly via `DesignMatrix.clean()`.
        """
        n = len(minimal_brain_data)
        rng = np.random.default_rng(42)
        a = rng.standard_normal(n)
        b = a + 0.1 * rng.standard_normal(n)
        design_matrix = DesignMatrix({"Intercept": np.ones(n), "condA": a, "condB": b})
        r = abs(np.corrcoef(a, b)[0, 1])
        assert r > 0.95, f"setup invariant violated: |r|={r}"
        assert np.linalg.matrix_rank(design_matrix.to_numpy()) == 3

        minimal_brain_data.fit(model="glm", X=design_matrix)
        assert minimal_brain_data.glm_betas.shape[0] == 3

    def test_design_clean_kwargs_are_rejected(self, minimal_brain_data):
        """The implicit-cleaning kwargs were removed; passing them is an error."""
        n = len(minimal_brain_data)
        design_matrix = DesignMatrix({"Intercept": np.ones(n)})
        for kwarg in (
            "design_clean",
            "design_clean_thresh",
            "design_clean_exclude_confounds",
            "design_clean_fill_na",
        ):
            with pytest.raises(TypeError):
                minimal_brain_data.fit(model="glm", X=design_matrix, **{kwarg: False})

    @pytest.mark.slow
    def test_rank_deficient_design_warns(self, minimal_brain_data):
        """A singular design produces non-unique betas, so warn loudly."""
        n = len(minimal_brain_data)
        rng = np.random.default_rng(42)
        a = rng.standard_normal(n)
        design_matrix = DesignMatrix(
            {"Intercept": np.ones(n), "condA": a, "condA_dup": a}
        )
        assert np.linalg.matrix_rank(design_matrix.to_numpy()) == 2

        with pytest.warns(UserWarning, match="rank deficient"):
            minimal_brain_data.fit(model="glm", X=design_matrix)

        # Still fits (pseudo-inverse), and keeps every column.
        assert minimal_brain_data.glm_betas.shape[0] == 3

    @pytest.mark.slow
    def test_rank_deficient_warning_names_the_columns(self, minimal_brain_data):
        """The warning must be actionable: report rank, size, and next step."""
        n = len(minimal_brain_data)
        rng = np.random.default_rng(42)
        a = rng.standard_normal(n)
        design_matrix = DesignMatrix(
            {"Intercept": np.ones(n), "condA": a, "condA_dup": a}
        )
        with pytest.warns(UserWarning) as record:
            minimal_brain_data.fit(model="glm", X=design_matrix)
        msg = "\n".join(str(w.message) for w in record)
        assert "2" in msg and "3" in msg  # rank 2 of 3
        # Regularization is the recommended fix: ridge has a unique solution
        # even when X'X is singular, and is invariant to column order.
        assert "ridge" in msg
        assert "vif" in msg.lower()
        assert "clean" in msg

    @pytest.mark.slow
    def test_ridge_is_order_invariant_where_clean_is_not(self, minimal_brain_data):
        """Why the warning recommends ridge over dropping columns.

        Ridge shrinks collinear regressors toward each other and returns the
        same model regardless of column order. `clean()` keeps whichever of a
        correlated pair comes first, so it yields a different model when the
        design is built in a different order.
        """
        from nltools.data.designmatrix import DesignMatrix

        n = len(minimal_brain_data)
        rng = np.random.default_rng(0)
        a = rng.standard_normal(n)
        b = a + 0.14 * rng.standard_normal(n)
        c = rng.standard_normal(n)
        assert abs(np.corrcoef(a, b)[0, 1]) > 0.95

        def ridge_weights(X):
            bd = minimal_brain_data.copy()
            bd.fit(model="ridge", X=X, alpha=1.0)
            w = bd.ridge_weights
            return w.data if hasattr(w, "data") else np.asarray(w)

        w1 = ridge_weights(np.column_stack([a, b, c]))
        w2 = ridge_weights(np.column_stack([b, a, c]))
        # Swapping the two collinear columns swaps their weights. Tolerance is
        # float32 solver precision, not slack for order effects -- dropping a
        # column instead changes the model categorically (asserted below).
        np.testing.assert_allclose(w1[0], w2[1], atol=1e-5)
        np.testing.assert_allclose(w1[1], w2[0], atol=1e-5)

        # clean() instead keeps a different regressor depending on order.
        kept1 = DesignMatrix({"a": a, "b": b, "c": c}).clean(thresh=0.95).columns
        kept2 = DesignMatrix({"b": b, "a": a, "c": c}).clean(thresh=0.95).columns
        assert set(kept1) != set(kept2)

    @pytest.mark.slow
    def test_full_rank_design_does_not_warn(self, minimal_brain_data):
        """No spurious rank warning on a well-formed design."""
        n = len(minimal_brain_data)
        rng = np.random.default_rng(42)
        design_matrix = DesignMatrix(
            {
                "Intercept": np.ones(n),
                "condA": rng.standard_normal(n),
                "condB": rng.standard_normal(n),
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            minimal_brain_data.fit(model="glm", X=design_matrix)
        assert not [w for w in caught if "rank deficient" in str(w.message)]
        assert minimal_brain_data.glm_betas.shape[0] == 3


class TestWarnRankDeficient:
    """Unit tests for the rank-deficiency check `fit(model='glm')` runs.

    The helper is pure (array in, warning out), so these run without brain
    data. Integration coverage through `fit()` lives in TestBrainDataModeling
    (slow-marked).
    """

    @staticmethod
    def _check(X, columns=None):
        import polars as pl

        from nltools.data.braindata.modeling import _warn_if_rank_deficient

        X = np.asarray(X, dtype=float)
        model = (
            pl.DataFrame({c: X[:, i] for i, c in enumerate(columns)})
            if columns is not None
            else X
        )
        _warn_if_rank_deficient(X, model)

    def test_warns_with_named_category(self):
        from nltools.data.braindata.modeling import RankDeficientDesignWarning

        rng = np.random.default_rng(0)
        a = rng.standard_normal(30)
        with pytest.warns(RankDeficientDesignWarning, match="rank deficient"):
            self._check(
                np.column_stack([np.ones(30), a, a]),
                columns=["Intercept", "condA", "condA_dup"],
            )

    def test_message_offers_both_fixes_and_names_culprits(self):
        """Per the maintainer ask: helpful tips — try .clean(), try ridge."""
        rng = np.random.default_rng(0)
        a = rng.standard_normal(30)
        with pytest.warns(UserWarning) as record:
            self._check(
                np.column_stack([np.ones(30), a, a]),
                columns=["Intercept", "condA", "condA_dup"],
            )
        msg = "\n".join(str(w.message) for w in record)
        assert "clean" in msg
        assert "ridge" in msg
        assert "vif" in msg.lower()
        # The dependent columns are named, not just counted.
        assert "condA" in msg
        # The columns that ARE fine are not dragged into the message.
        assert "Intercept" not in msg

    def test_more_columns_than_rows_warns(self):
        """p > n is rank deficient by construction — the loudest case, not a skip."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 14))
        with pytest.warns(UserWarning, match="rank deficient") as record:
            self._check(X)
        msg = str(record[0].message)
        assert "14" in msg and "10" in msg

    def test_long_column_lists_are_truncated(self):
        """A 40-column design must not dump 40 names into the warning."""
        rng = np.random.default_rng(0)
        base = rng.standard_normal((50, 39))
        X = np.column_stack([base, base[:, 0]])  # last col duplicates col_0
        names = [f"col_{i}" for i in range(39)] + ["col_dup"]
        with pytest.warns(UserWarning) as record:
            self._check(X, columns=names)
        msg = str(record[0].message)
        # Only the implicated columns appear, not the full roster.
        assert sum(f"col_{i}," in msg or f"col_{i}." in msg for i in range(1, 39)) == 0

    def test_full_rank_is_silent(self):
        rng = np.random.default_rng(0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._check(rng.standard_normal((30, 4)))
        assert not caught

    def test_all_nonfinite_rows_do_not_crash(self):
        X = np.full((5, 3), np.nan)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            self._check(X)


class TestBrainDataTTest:
    def test_ttest_one_sample(self, minimal_brain_data):
        """One-sample t-test returns mean + t + z + p (all as BrainData)."""
        from scipy.stats import ttest_1samp

        result = minimal_brain_data.ttest()

        assert isinstance(result, dict)
        assert set(result.keys()) == {"mean", "t", "z", "p"}
        for key in ("mean", "t", "z", "p"):
            assert isinstance(result[key], BrainData)

        expected_t, expected_p = ttest_1samp(minimal_brain_data.data, 0.0, axis=0)
        np.testing.assert_allclose(result["t"].data, expected_t)
        np.testing.assert_allclose(result["p"].data, expected_p)
        np.testing.assert_allclose(
            result["mean"].data, minimal_brain_data.data.mean(axis=0)
        )
        assert result["t"].data.shape == (minimal_brain_data.data.shape[1],)

    def test_ttest2_removed(self):
        """ttest2 was a 0.6.0-dev convenience method; it must not exist."""
        assert not hasattr(BrainData, "ttest2")

    def test_ttest_parametric_honors_tail(self, minimal_brain_data):
        """tail=1 must reach the parametric path (was silently ignored pre-0.6.0)."""
        from scipy.stats import ttest_1samp

        result = minimal_brain_data.ttest(tail=1)
        _, expected_p = ttest_1samp(
            minimal_brain_data.data, 0.0, axis=0, alternative="greater"
        )
        np.testing.assert_allclose(result["p"].data, expected_p)
        with pytest.raises(ValueError, match="tail"):
            minimal_brain_data.ttest(tail="upper")

    def test_ttest_z_signed_and_monotonic(self, minimal_brain_data):
        """z = sign(t) * norm.isf(p/2); sign(z) == sign(t); monotonic with t."""
        result = minimal_brain_data.ttest()
        t_arr = np.asarray(result["t"].data)
        z_arr = np.asarray(result["z"].data)
        # non-zero-t voxels must agree in sign
        nz = t_arr != 0
        assert np.all(np.sign(t_arr[nz]) == np.sign(z_arr[nz]))
        # strong correlation (large-df t ≈ z; even at small df they stay
        # monotonic since z is derived from the same p)
        assert np.corrcoef(t_arr, z_arr)[0, 1] > 0.99

    def test_ttest_one_tailed_wrong_direction_z_is_finite(self, minimal_brain_data):
        """tail=1 on strongly negative data must not leak z = -inf.

        ``t.sf(t, df)`` saturates to exactly 1.0 for large negative t, and
        an unclipped ``norm.isf(1.0)`` is ``-inf`` — poisoning any downstream
        percentile / plotting of the z map.
        """
        bd = minimal_brain_data.copy()
        bd.data = bd.data - 10.0
        res = bd.ttest(tail=1)
        z = np.asarray(res["z"].data)
        assert np.all(np.isfinite(z))
        assert np.all(z < 0)

    def test_ttest_popmean(self, minimal_brain_data):
        """popmean kwarg shifts the null and the reported mean."""
        from scipy.stats import ttest_1samp

        result = minimal_brain_data.ttest(popmean=0.5)
        expected_t, _ = ttest_1samp(minimal_brain_data.data, 0.5, axis=0)
        np.testing.assert_allclose(result["t"].data, expected_t)
        np.testing.assert_allclose(
            result["mean"].data, minimal_brain_data.data.mean(axis=0) - 0.5
        )

    def test_ttest_single_image_raises(self, minimal_brain_data):
        """t-test on a single image should raise."""
        single = minimal_brain_data[0]
        with pytest.raises(ValueError, match="multiple images"):
            single.ttest()

    def test_ttest_permutation(self, minimal_brain_data):
        """permutation=True reports empirical p but still returns mean/t/z/p."""
        result = minimal_brain_data.ttest(
            permutation=True, n_permute=50, random_state=0
        )
        assert set(result.keys()) == {"mean", "t", "z", "p"}
        for key in ("mean", "t", "z", "p"):
            assert isinstance(result[key], BrainData)
        np.testing.assert_allclose(
            result["mean"].data, minimal_brain_data.data.mean(axis=0)
        )

    def test_ttest_permutation_popmean_tests_shifted_hypothesis(
        self, minimal_brain_data
    ):
        """permutation=True must test mean != popmean, not mean != 0.

        Data centered at popmean is null under the requested hypothesis, so
        the sign-flip p-values must look uniform — a zero-referenced test
        (the pre-fix bug) would return the minimum p at every voxel.
        """
        bd = minimal_brain_data.copy()
        bd.data = bd.data + 5.0  # ~N(5, 1) per voxel
        res = bd.ttest(popmean=5.0, permutation=True, n_permute=200, random_state=0)
        # Under the correct H0 (mean == 5) nothing should be at the floor
        # p = 1/(n_permute+1) ≈ 0.005; the zero-referenced bug puts every
        # voxel there.
        assert np.all(np.asarray(res["p"].data) > 0.02)
        # Sanity: the same data against popmean=0 is overwhelmingly significant.
        res0 = bd.ttest(popmean=0.0, permutation=True, n_permute=200, random_state=0)
        assert np.all(np.asarray(res0["p"].data) < 0.01)

    def test_ttest_permutation_popmean_mean_is_effect_size(self, minimal_brain_data):
        """Both branches must report mean(images) - popmean, per the docstring."""
        bd = minimal_brain_data.copy()
        bd.data = bd.data + 5.0
        expected = bd.data.mean(axis=0) - 5.0
        res_perm = bd.ttest(popmean=5.0, permutation=True, n_permute=20, random_state=0)
        np.testing.assert_allclose(res_perm["mean"].data, expected, rtol=1e-6)
        res_param = bd.ttest(popmean=5.0)
        np.testing.assert_allclose(res_param["mean"].data, expected, rtol=1e-6)

    # ── Shared one-sample contract (docs/development/specs/ttest.md) ────────

    @pytest.mark.parametrize("popmean", [0.0, 0.75])
    @pytest.mark.parametrize("tail,alternative", [(2, "two-sided"), (1, "greater")])
    def test_ttest_parametric_matches_scipy(
        self, minimal_brain_data, popmean, tail, alternative
    ):
        """Parametric mean/t/p match direct NumPy/SciPy for both tails."""
        from scipy.stats import ttest_1samp

        data = minimal_brain_data.data
        result = minimal_brain_data.ttest(popmean=popmean, tail=tail)
        assert set(result) == {"mean", "t", "z", "p"}
        expected_t, expected_p = ttest_1samp(
            data, popmean, axis=0, alternative=alternative
        )
        np.testing.assert_allclose(result["t"].data, expected_t)
        np.testing.assert_allclose(result["p"].data, expected_p)
        np.testing.assert_allclose(result["mean"].data, data.mean(axis=0) - popmean)

    def test_ttest_single_voxel(self):
        """A one-voxel stack still returns one image per key."""
        from scipy.stats import ttest_1samp

        rng = np.random.default_rng(0)
        bd = _brain_data_from_array(rng.standard_normal((12, 1)))
        result = bd.ttest(popmean=0.5)
        for key in ("mean", "t", "z", "p"):
            assert np.asarray(result[key].data).shape == (1,)
        expected_t, expected_p = ttest_1samp(bd.data, 0.5, axis=0)
        np.testing.assert_allclose(result["t"].data, expected_t)
        np.testing.assert_allclose(result["p"].data, expected_p)
        np.testing.assert_allclose(result["mean"].data, bd.data.mean(axis=0) - 0.5)

    def test_ttest_permutation_matches_engine_at_fixed_seed(self, minimal_brain_data):
        """Permutation p and the centered-mean null match a direct engine call."""
        from nltools.algorithms.inference import one_sample_permutation_test

        popmean = 0.25
        result = minimal_brain_data.ttest(
            popmean=popmean,
            permutation=True,
            n_permute=64,
            return_null=True,
            random_state=11,
        )
        engine = one_sample_permutation_test(
            minimal_brain_data.data - popmean,
            n_permute=64,
            tail=2,
            return_null=True,
            n_jobs=-1,
            random_state=11,
        )
        np.testing.assert_allclose(result["p"].data, engine["p"])
        np.testing.assert_allclose(result["mean"].data, engine["mean"])
        np.testing.assert_allclose(result["null_dist"], engine["null_dist"])
        assert result["null_dist"].shape == (64, minimal_brain_data.shape[1])

    def test_ttest_null_only_when_permuting_and_requested(self, minimal_brain_data):
        """`null_dist` appears only with permutation=True and return_null=True."""
        assert "null_dist" not in minimal_brain_data.ttest(return_null=True)
        assert "null_dist" not in minimal_brain_data.ttest(
            permutation=True, n_permute=16, random_state=0
        )
        with_null = minimal_brain_data.ttest(
            permutation=True, n_permute=16, return_null=True, random_state=0
        )
        without_null = minimal_brain_data.ttest(
            permutation=True, n_permute=16, return_null=False, random_state=0
        )
        assert set(with_null) == {"mean", "t", "z", "p", "null_dist"}
        for key in ("mean", "t", "z", "p"):
            np.testing.assert_array_equal(with_null[key].data, without_null[key].data)

    def test_ttest_single_voxel_null_keeps_feature_axis(self):
        """The null is never squeezed, even for one voxel."""
        rng = np.random.default_rng(1)
        bd = _brain_data_from_array(rng.standard_normal((10, 1)))
        result = bd.ttest(
            permutation=True, n_permute=32, return_null=True, random_state=2
        )
        assert result["null_dist"].shape == (32, 1)

    def test_ttest_permutation_reports_the_t_statistic(self, minimal_brain_data):
        """`t` is the observed SciPy statistic on the permutation path too."""
        from scipy.stats import ttest_1samp

        result = minimal_brain_data.ttest(
            permutation=True, n_permute=32, random_state=0
        )
        expected_t, _ = ttest_1samp(minimal_brain_data.data, 0.0, axis=0)
        np.testing.assert_allclose(result["t"].data, expected_t)

    def test_ttest_z_endpoints_stay_finite(self):
        """z is finite at both the p floor and p == 1.0."""
        rng = np.random.default_rng(5)
        bd = _brain_data_from_array(rng.standard_normal((30, 3)) * 1e-8 + 50.0)
        huge = bd.ttest()
        assert np.all(np.isfinite(huge["z"].data)) and np.all(huge["z"].data > 0)
        saturated = bd.ttest(popmean=100.0, tail=1)
        assert np.all(np.asarray(saturated["p"].data) == 1.0)
        assert np.all(np.isfinite(saturated["z"].data))
        assert np.all(np.asarray(saturated["z"].data) < 0)

    def test_ttest_constant_voxels_match_scipy(self):
        """Constant voxels keep SciPy's behavior; no new NaN policy."""
        from scipy.stats import ttest_1samp

        values = np.column_stack([np.ones(8), np.arange(8.0), np.zeros(8)])
        bd = _brain_data_from_array(values)
        with np.errstate(invalid="ignore", divide="ignore"):
            expected_t, expected_p = ttest_1samp(bd.data, 0.0, axis=0)
            result = bd.ttest()
        np.testing.assert_allclose(result["t"].data, expected_t)
        np.testing.assert_allclose(result["p"].data, expected_p)

    def test_ttest_results_are_owned_and_rows_cleared(self, minimal_brain_data):
        """Maps alias neither the input nor each other, and X/Y are cleared."""
        original = minimal_brain_data.data.copy()
        result = minimal_brain_data.ttest(
            permutation=True, n_permute=16, return_null=True, random_state=0
        )
        maps = [result[key] for key in ("mean", "t", "z", "p")]
        arrays = [m.data for m in maps] + [result["null_dist"]]
        for i, first in enumerate(arrays):
            assert not np.shares_memory(first, minimal_brain_data.data)
            for second in arrays[i + 1 :]:
                assert not np.shares_memory(first, second)
        for image in maps:
            assert image.X.is_empty()
            assert image.Y.is_empty()
        for array in arrays:
            array[...] = -999.0
        np.testing.assert_array_equal(minimal_brain_data.data, original)
        for key, image in zip(("mean", "t", "z", "p"), maps):
            assert np.all(np.asarray(image.data) == -999.0)


class TestBrainDataRidgeCV:
    """Splitter forwarding, generator rejection, and .size."""

    def test_size_property(self, minimal_brain_data):
        assert minimal_brain_data.size == minimal_brain_data.data.size

    @e5y6_pending
    def test_splitter_object_changes_alpha_selection(self, minimal_brain_data):
        """Different CV schemes produce different per-alpha scores."""
        n = minimal_brain_data.shape[0]
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, 8))

        b1 = minimal_brain_data.copy()
        b1.fit(
            model="ridge",
            X=X,
            alpha="auto",
            alphas=np.logspace(-2, 4, 10),
            cv=KFold(5, shuffle=False),
            scale=False,
        )
        b2 = minimal_brain_data.copy()
        b2.fit(
            model="ridge",
            X=X,
            alpha="auto",
            alphas=np.logspace(-2, 4, 10),
            cv=KFold(5, shuffle=True, random_state=0),
            scale=False,
        )
        # Same data, different splits → different alpha_scores.
        assert not np.allclose(
            b1.cv_results_["alpha_scores"], b2.cv_results_["alpha_scores"]
        )

    def test_generator_cv_rejected(self, minimal_brain_data):
        n = minimal_brain_data.shape[0]
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, 5))
        gen = KFold(5).split(X)
        with pytest.raises(TypeError, match="generator"):
            minimal_brain_data.fit(
                model="ridge", X=X, alpha="auto", cv=gen, scale=False
            )

    @e5y6_pending
    def test_fit_intercept_propagates_to_cv_path(self, minimal_brain_data):
        """fit_intercept=True is forwarded through compute_ridge_cv."""
        n = minimal_brain_data.shape[0]
        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, 5))

        # Non-trivial BOLD offset — without fit_intercept, CV path
        # produces strongly biased predictions (the original bug).
        bd = minimal_brain_data.copy()
        bd.data = bd.data + 100.0

        bd.fit(
            model="ridge",
            X=X,
            alpha="auto",
            alphas=np.logspace(-2, 2, 6),
            cv=KFold(5, shuffle=True, random_state=0),
            scale=False,
            standardize=None,
            fit_intercept=True,
        )
        # Held-out predictions live on the original BOLD scale.
        preds = bd.cv_results_["predictions"].data
        assert abs(preds.mean() - 100.0) < 5.0


@e5y6_pending
class TestBrainDataRidgePerVoxelAlpha:
    """v0.6 contract: bd.fit(model='ridge', alpha='auto', cv=K) selects α
    per-voxel by default and refits the full-data weights with those α.

    The previous BrainData CV path collapsed to a single global α even with
    Ridge.local_alpha=True (the default), so the per-voxel machinery in
    solve_ridge_cv was never reached and LORO produced wildly negative R².
    These tests pin down the new behavior.
    """

    @staticmethod
    def _per_voxel_fixture(n=80, p=12, n_voxels=6, snr_high=5.0, snr_low=0.2, seed=0):
        """Two voxels per noise regime so SNR drives α selection."""
        from nltools.data import BrainData
        import nibabel as nib

        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, p)).astype(np.float32)

        # True coefficients shared across voxels
        coef = rng.standard_normal((p, n_voxels)).astype(np.float32)
        signal = X @ coef
        # Half voxels: low noise (low SNR → larger α). Other half: high SNR.
        noise = rng.standard_normal((n, n_voxels)).astype(np.float32)
        scales = np.array(
            [snr_low if j < n_voxels // 2 else snr_high for j in range(n_voxels)],
            dtype=np.float32,
        )
        # Higher scale → noise dominates → wants larger α.
        Y = signal + noise * (1.0 / scales[None, :])

        spatial_shape = (n_voxels, 1, 1)
        mask_data = np.zeros(spatial_shape, dtype=bool)
        mask_data.flat[:n_voxels] = True
        affine = np.eye(4)
        volume_4d = np.zeros(spatial_shape + (n,), dtype=np.float32)
        for t in range(n):
            volume_t = np.zeros(spatial_shape, dtype=np.float32)
            volume_t.flat[:n_voxels] = Y[t]
            volume_4d[..., t] = volume_t

        bd = BrainData(
            nib.Nifti1Image(volume_4d, affine),
            mask=nib.Nifti1Image(mask_data.astype(np.float32), affine),
        )
        return bd, X, Y

    def test_best_alpha_is_per_voxel_array(self):
        bd, X, _ = self._per_voxel_fixture()
        alphas = np.logspace(-2, 4, 12)
        bd.fit(
            model="ridge",
            X=X,
            alpha="auto",
            alphas=alphas,
            cv=5,
            scale=False,
            fit_intercept=True,
        )
        best = bd.cv_results_["best_alpha"]
        assert isinstance(best, np.ndarray)
        assert best.shape == (bd.shape[1],)
        assert best is not bd.model_.alpha_
        assert bd.cv_results_["alpha_scores"] is not bd.model_.cv_scores_

        model_alpha = bd.model_.alpha_.copy()
        model_scores = bd.model_.cv_scores_.copy()
        best[0] = -1.0
        bd.cv_results_["alpha_scores"][0, 0, 0] = -1.0
        np.testing.assert_array_equal(bd.model_.alpha_, model_alpha)
        np.testing.assert_array_equal(bd.model_.cv_scores_, model_scores)

    def test_local_alpha_false_collapses_to_scalar(self):
        bd, X, _ = self._per_voxel_fixture()
        alphas = np.logspace(-2, 4, 12)
        bd.fit(
            model="ridge",
            X=X,
            alpha="auto",
            alphas=alphas,
            cv=5,
            scale=False,
            fit_intercept=True,
            local_alpha=False,
        )
        best = bd.cv_results_["best_alpha"]
        # All voxels share the same α; representation is scalar (or
        # per-voxel array with only one unique value).
        if isinstance(best, np.ndarray):
            assert best.shape == (bd.shape[1],)
            assert np.unique(best).size == 1
        else:
            assert isinstance(best, (int, float, np.floating, np.integer))

    def test_voxels_with_different_snr_pick_different_alphas(self):
        bd, X, _ = self._per_voxel_fixture(
            n=120, p=10, n_voxels=8, snr_high=10.0, snr_low=0.1, seed=1
        )
        alphas = np.logspace(-2, 4, 16)
        bd.fit(
            model="ridge",
            X=X,
            alpha="auto",
            alphas=alphas,
            cv=5,
            scale=False,
            fit_intercept=True,
        )
        best = bd.cv_results_["best_alpha"]
        assert isinstance(best, np.ndarray)
        # Low-SNR (noisy) voxels should pick larger α than high-SNR ones.
        n_voxels = bd.shape[1]
        low_snr = best[: n_voxels // 2]
        high_snr = best[n_voxels // 2 :]
        # Mean α over the noisy half is strictly larger than the clean half.
        assert low_snr.mean() > high_snr.mean()

    def test_full_data_coefs_match_per_voxel_alpha_refit(self):
        from nltools.algorithms.ridge import ridge_svd

        bd, X, Y = self._per_voxel_fixture()
        alphas = np.logspace(-2, 4, 12)
        bd.fit(
            model="ridge",
            X=X,
            alpha="auto",
            alphas=alphas,
            cv=5,
            scale=False,
            fit_intercept=True,
        )
        best = bd.cv_results_["best_alpha"]
        assert isinstance(best, np.ndarray) and best.shape == (bd.shape[1],)

        # Replicate the solver's centering: global mean across all rows of X / Y.
        X_offset = X.mean(axis=0)
        Y_offset = bd.data.mean(axis=0)
        Xc = X - X_offset
        Yc = bd.data - Y_offset

        coefs = bd.ridge_weights.data  # (n_features, n_voxels)
        for j in range(bd.shape[1]):
            expected = ridge_svd(Xc, Yc[:, j], alpha=float(best[j]))
            np.testing.assert_allclose(
                coefs[:, j],
                expected,
                rtol=1e-3,
                atol=1e-3,
                err_msg=f"voxel {j} weights diverge from per-α refit",
            )

    def test_held_out_predictions_use_per_voxel_alpha(self):
        from sklearn.model_selection import KFold
        from nltools.algorithms.ridge import ridge_svd

        bd, X, _ = self._per_voxel_fixture(
            n=120, p=10, n_voxels=8, snr_high=10.0, snr_low=0.1, seed=2
        )
        alphas = np.logspace(-2, 4, 12)
        bd.fit(
            model="ridge",
            X=X,
            alpha="auto",
            alphas=alphas,
            cv=KFold(5, shuffle=False),
            scale=False,
            fit_intercept=True,
        )
        per_voxel_preds = bd.cv_results_["predictions"].data

        # Build a baseline: same CV splits, but force a single global α.
        # If predictions use per-voxel α, this baseline differs.
        global_alpha = float(np.median(bd.cv_results_["best_alpha"]))
        baseline = np.zeros_like(per_voxel_preds)
        cv_splitter = KFold(5, shuffle=False)
        for train_idx, test_idx in cv_splitter.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr = bd.data[train_idx]
            X_off = X_tr.mean(axis=0)
            y_off = y_tr.mean(axis=0)
            coef = ridge_svd(X_tr - X_off, y_tr - y_off, alpha=global_alpha)
            baseline[test_idx] = (X_te - X_off) @ coef + y_off

        assert per_voxel_preds.shape == baseline.shape
        assert not np.allclose(per_voxel_preds, baseline)

    def test_local_alpha_named_kwarg_on_bd_fit(self):
        bd, X, _ = self._per_voxel_fixture()
        # Passing as named kwarg should work and should be forwarded.
        bd.fit(
            model="ridge",
            X=X,
            alpha="auto",
            alphas=np.logspace(-2, 4, 8),
            cv=3,
            scale=False,
            local_alpha=False,
        )
        assert bd.model_.local_alpha is False

    def test_loro_via_groupkfold_returns_sensible_per_voxel_results(self):
        """GroupKFold over per-run blocks → α selection respects splitter and
        per-voxel R² isn't disastrously negative."""
        from sklearn.model_selection import GroupKFold

        bd, X, _ = self._per_voxel_fixture(
            n=80, p=10, n_voxels=8, snr_high=8.0, snr_low=0.5, seed=3
        )
        # Synthetic per-run groups: 8 runs of 10 samples each.
        groups = np.repeat(np.arange(8), 10)
        splitter = GroupKFold(n_splits=8)
        # GroupKFold needs groups passed to .split, so we wrap it.

        class _GroupSplitter:
            def __init__(self, splitter, groups):
                self._s = splitter
                self._g = groups

            def split(self, X, y=None, groups=None):
                return self._s.split(X, y, self._g)

            def get_n_splits(self, X=None, y=None, groups=None):
                return self._s.get_n_splits(X, y, self._g)

        cv = _GroupSplitter(splitter, groups)

        bd.fit(
            model="ridge",
            X=X,
            alpha="auto",
            alphas=np.logspace(-2, 4, 12),
            cv=cv,
            scale=False,
            fit_intercept=True,
        )
        mean_score = bd.cv_results_["mean_score"]
        assert mean_score.shape == (bd.shape[1],)
        assert mean_score.max() > 0.0
        assert mean_score.mean() > -1.0


class TestWarnNearCollinear:
    """Unit tests for the near-collinearity check `fit(model='glm')` runs.

    Complements TestWarnRankDeficient: these designs are full rank, so the
    exact-deficiency warning stays silent, but the columns are correlated
    enough (pairwise |r|, or a large condition number) that OLS betas are
    unstable. The helper is pure (array in, warning out).
    """

    @staticmethod
    def _check(X, columns=None):
        import polars as pl

        from nltools.data.braindata.modeling import _warn_if_near_collinear

        X = np.asarray(X, dtype=float)
        model = (
            pl.DataFrame({c: X[:, i] for i, c in enumerate(columns)})
            if columns is not None
            else X
        )
        _warn_if_near_collinear(X, model)

    @staticmethod
    def _correlated_pair(n=60, r=0.97, seed=0):
        """Two unit-variance columns with exactly |r| = r (empirical)."""
        rng = np.random.default_rng(seed)
        a = rng.standard_normal(n)
        a = (a - a.mean()) / a.std()
        e = rng.standard_normal(n)
        e = e - e.mean()
        e = e - a * (e @ a) / (a @ a)  # orthogonalize against a
        e = e / e.std()
        b = r * a + np.sqrt(1 - r**2) * e
        return a, b

    def test_high_pairwise_r_warns_and_names_the_pair(self):
        from nltools.data.braindata.modeling import NearCollinearDesignWarning

        a, b = self._correlated_pair(r=0.97)
        rng = np.random.default_rng(7)
        other = rng.standard_normal(len(a))
        with pytest.warns(NearCollinearDesignWarning, match="nearly collinear") as rec:
            self._check(
                np.column_stack([a, b, other]), columns=["condA", "condB", "other"]
            )
        msg = str(rec[0].message)
        assert "condA" in msg and "condB" in msg
        # The uninvolved column is not dragged into the message.
        assert "other" not in msg

    def test_message_offers_the_three_fixes(self):
        """Same maintainer ask as the rank warning: vif, clean, ridge tips."""
        a, b = self._correlated_pair(r=0.97)
        with pytest.warns(UserWarning) as rec:
            self._check(np.column_stack([a, b]), columns=["condA", "condB"])
        msg = "\n".join(str(w.message) for w in rec)
        assert "vif" in msg.lower()
        assert "clean" in msg
        assert "ridge" in msg

    def test_orthogonal_design_is_silent(self):
        rng = np.random.default_rng(0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._check(rng.standard_normal((60, 4)))
        assert not caught

    def test_condition_number_catches_multi_column_dependence(self):
        """Three columns nearly summing to zero: no pairwise |r| >= 0.95, but
        the standardized design's condition number blows past the Belsley 30."""
        from nltools.data.braindata.modeling import NearCollinearDesignWarning

        rng = np.random.default_rng(3)
        n = 60
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        c = -(a + b) + 0.02 * rng.standard_normal(n)
        X = np.column_stack([a, b, c])
        # Preconditions: full rank, and the pairwise signal alone cannot fire.
        assert np.linalg.matrix_rank(X) == 3
        corr = np.abs(np.corrcoef(X, rowvar=False))
        np.fill_diagonal(corr, 0.0)
        assert corr.max() < 0.95
        with pytest.warns(NearCollinearDesignWarning, match="condition number"):
            self._check(X, columns=["a", "b", "c"])

    def test_intercept_and_drift_columns_do_not_warn(self):
        """Constant intercepts are excluded from the scan; a linear drift next
        to well-behaved regressors is not collinear."""
        rng = np.random.default_rng(5)
        n = 60
        X = np.column_stack(
            [
                np.ones(n),
                np.linspace(-1, 1, n),
                rng.standard_normal(n),
                rng.standard_normal(n),
            ]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._check(X, columns=[".nl_poly_0", ".nl_poly_1", "condA", "condB"])
        assert not caught

    def test_pair_list_is_truncated(self):
        """Many offending pairs must not flood the warning message."""
        a, _ = self._correlated_pair(n=80, r=0.97)
        cols = [a]
        for seed in range(1, 9):
            rng = np.random.default_rng(100 + seed)
            e = rng.standard_normal(len(a))
            e = e - e.mean()
            e = e - a * (e @ a) / (a @ a)
            e = e / e.std()
            cols.append(0.97 * a + np.sqrt(1 - 0.97**2) * e)
        X = np.column_stack(cols)
        names = ["base"] + [f"near_{i}" for i in range(1, 9)]
        with pytest.warns(UserWarning) as rec:
            self._check(X, columns=names)
        msg = str(rec[0].message)
        assert "more" in msg  # "... and N more"
        assert msg.count("&") <= 5

    def test_exact_deficiency_fires_only_the_rank_warning(self, minimal_brain_data):
        """Through fit(): an exactly rank-deficient design raises
        RankDeficientDesignWarning alone, never both warnings."""
        from nltools.data.braindata.modeling import (
            NearCollinearDesignWarning,
            RankDeficientDesignWarning,
        )

        n = len(minimal_brain_data)
        rng = np.random.default_rng(11)
        a = rng.standard_normal(n)
        design_matrix = DesignMatrix(
            {"Intercept": np.ones(n), "condA": a, "condA_dup": a}
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            minimal_brain_data.fit(model="glm", X=design_matrix)
        assert any(issubclass(w.category, RankDeficientDesignWarning) for w in caught)
        assert not any(
            issubclass(w.category, NearCollinearDesignWarning) for w in caught
        )

    def test_near_collinear_design_warns_through_fit(self, minimal_brain_data):
        """Through fit(): the v0.5 design_clean threshold case (r = 0.97) is no
        longer silent — it warns, and nothing is dropped."""
        from nltools.data.braindata.modeling import NearCollinearDesignWarning

        n = len(minimal_brain_data)
        a, b = self._correlated_pair(n=n, r=0.97, seed=21)
        design_matrix = DesignMatrix({"Intercept": np.ones(n), "condA": a, "condB": b})
        with pytest.warns(NearCollinearDesignWarning):
            minimal_brain_data.fit(model="glm", X=design_matrix)
        # Warning only: every regressor is still estimated.
        assert minimal_brain_data.glm_betas.shape[0] == 3


class TestGlmFacadeContract:
    """The `BrainData` GLM boundary defined in specs/glm.md and specs/braindata.md."""

    @staticmethod
    def _design(brain, seed=0):
        from nltools.data import DesignMatrix

        rng = np.random.default_rng(seed)
        return DesignMatrix(
            {
                "intercept": np.ones(len(brain)),
                "cond_a": rng.normal(size=len(brain)),
                "cond_b": rng.normal(size=len(brain)),
            }
        )

    # ---------------------------------------------------------------- fit

    def test_fit_exposes_the_glm_options_keyword_only(self):
        import inspect

        parameters = inspect.signature(BrainData.fit).parameters
        assert parameters["model"].default == "glm"
        for name, default in (
            ("X", None),
            ("glm_noise_model", "ols"),
            ("glm_bins", 100),
            ("glm_n_jobs", 1),
            ("inplace", True),
            ("random_state", None),
        ):
            assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert parameters[name].default == default
        assert "scale" not in parameters
        assert "standardize" not in parameters

    @pytest.mark.parametrize("removed", ["scale", "standardize"])
    def test_fit_rejects_removed_preprocessing_keywords(
        self, minimal_brain_data, removed
    ):
        design = self._design(minimal_brain_data)
        with pytest.raises(TypeError):
            minimal_brain_data.fit(model="glm", X=design, **{removed: None})

    @pytest.mark.parametrize(
        "option",
        [
            {"cv": 3},
            {"device": "gpu"},
            {"per_target_alpha": False},
            {"progress_bar": True},
        ],
    )
    def test_fit_glm_rejects_non_default_ridge_options(
        self, minimal_brain_data, option
    ):
        design = self._design(minimal_brain_data)
        with pytest.raises(ValueError, match="unselected estimator|does not accept"):
            minimal_brain_data.fit(model="glm", X=design, **option)

    def test_fit_ridge_rejects_non_default_glm_options(self, minimal_brain_data):
        X = np.random.default_rng(0).normal(size=(len(minimal_brain_data), 3))
        with pytest.raises(ValueError, match="unselected estimator|does not accept"):
            minimal_brain_data.fit(model="ridge", X=X, alpha=1.0, glm_bins=50)

    def test_fit_glm_rejects_unknown_keywords(self, minimal_brain_data):
        design = self._design(minimal_brain_data)
        with pytest.raises(TypeError):
            minimal_brain_data.fit(model="glm", X=design, t_r=2.0)

    def test_fit_glm_requires_a_design_matrix(self, minimal_brain_data):
        frame = pd.DataFrame({"intercept": np.ones(len(minimal_brain_data))})
        with pytest.raises(TypeError, match="DesignMatrix"):
            minimal_brain_data.fit(model="glm", X=frame)

    def test_fit_attaches_only_the_documented_state(self, minimal_brain_data):
        from nltools.models import Glm

        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)

        assert isinstance(minimal_brain_data.model_, Glm)
        assert minimal_brain_data.glm_betas.shape == (3, minimal_brain_data.shape[1])
        assert minimal_brain_data.glm_residual.shape == minimal_brain_data.shape
        assert minimal_brain_data.glm_predicted.shape == minimal_brain_data.shape
        assert minimal_brain_data.glm_r2.shape[-1] == minimal_brain_data.shape[1]
        for removed in ("glm_t", "glm_p", "glm_se", "X_", "design_matrix"):
            assert not hasattr(minimal_brain_data, removed)

    def test_refit_from_ridge_drops_the_ridge_training_design(self, minimal_brain_data):
        """`X_` belongs to the ridge fit; a GLM refit must not inherit it."""
        X = np.random.default_rng(0).normal(size=(len(minimal_brain_data), 3))
        minimal_brain_data.fit(model="ridge", X=X, alpha=1.0)
        assert hasattr(minimal_brain_data, "X_")

        design = self._design(minimal_brain_data)
        for inplace in (True, False):
            fitted = minimal_brain_data.fit(model="glm", X=design, inplace=inplace)
            assert not hasattr(fitted, "X_")
            assert not hasattr(fitted, "ridge_weights")

    def test_a_failed_fit_attaches_no_model(self, minimal_brain_data, monkeypatch):
        """`model_` is attached only once the estimator's own fit returns."""
        from nltools.models import Glm

        def boom(self, X, y):
            raise RuntimeError("estimator blew up")

        monkeypatch.setattr(Glm, "fit", boom)
        design = self._design(minimal_brain_data)
        with pytest.raises(RuntimeError, match="estimator blew up"):
            minimal_brain_data.fit(model="glm", X=design)

        assert not hasattr(minimal_brain_data, "model_")
        assert not hasattr(minimal_brain_data, "glm_betas")

    def test_fit_state_enumeration_is_exhaustive(self, minimal_brain_data):
        from nltools.data.braindata.utils import _FIT_STATE_ATTRIBUTES

        before = set(vars(minimal_brain_data))
        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)
        attached = set(vars(minimal_brain_data)) - before

        assert attached
        assert attached <= set(_FIT_STATE_ATTRIBUTES)
        assert "design_matrix" not in _FIT_STATE_ATTRIBUTES

    def test_in_place_mutation_clears_every_fitted_attribute(self, minimal_brain_data):
        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)

        minimal_brain_data += 1.0

        for name in ("model_", "glm_betas", "glm_residual", "glm_predicted", "glm_r2"):
            assert not hasattr(minimal_brain_data, name)

    def test_fit_inplace_false_leaves_the_source_untouched(self, minimal_brain_data):
        design = self._design(minimal_brain_data)
        original = minimal_brain_data.data.copy()

        fitted = minimal_brain_data.fit(model="glm", X=design, inplace=False)

        assert fitted is not minimal_brain_data
        np.testing.assert_array_equal(minimal_brain_data.data, original)
        for name in ("model_", "glm_betas", "glm_residual", "glm_predicted", "glm_r2"):
            assert hasattr(fitted, name)
            assert not hasattr(minimal_brain_data, name)

    def test_fit_does_not_preprocess_the_response(self, minimal_brain_data):
        design = self._design(minimal_brain_data)
        minimal_brain_data.data = minimal_brain_data.data + 100.0
        original = minimal_brain_data.data.copy()

        minimal_brain_data.fit(model="glm", X=design)

        np.testing.assert_allclose(minimal_brain_data.data, original)

    def test_fit_betas_match_least_squares(self, minimal_brain_data):
        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)

        expected = np.linalg.lstsq(
            design.to_numpy(), minimal_brain_data.data, rcond=None
        )[0]
        np.testing.assert_allclose(
            minimal_brain_data.glm_betas.data, expected, atol=1e-8
        )

    # ---------------------------------------------------- compute_contrasts

    def test_compute_contrasts_has_no_statistic_argument(self):
        import inspect

        parameters = inspect.signature(BrainData.compute_contrasts).parameters
        assert "statistic" not in parameters
        assert parameters["inference"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["inference"].default is False

    def test_compute_contrasts_before_fit_raises_runtime_error(
        self, minimal_brain_data
    ):
        with pytest.raises(RuntimeError):
            minimal_brain_data.compute_contrasts("cond_a - cond_b")

    def test_compute_contrasts_on_a_fitted_ridge_raises_value_error(
        self, minimal_brain_data
    ):
        X = np.random.default_rng(0).normal(size=(len(minimal_brain_data), 3))
        minimal_brain_data.fit(model="ridge", X=X, alpha=1.0)
        with pytest.raises(ValueError, match="Ridge"):
            minimal_brain_data.compute_contrasts([1, -1, 0])

    @pytest.mark.parametrize(
        "contrast", ["cond_a - cond_b", [0.0, 1.0, -1.0]], ids=["string", "vector"]
    )
    def test_effect_contrast_equals_beta_arithmetic(self, minimal_brain_data, contrast):
        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)

        effect = minimal_brain_data.compute_contrasts(contrast)

        assert isinstance(effect, BrainData)
        assert effect.X.is_empty() and effect.Y.is_empty()
        np.testing.assert_allclose(
            effect.data,
            np.array([0.0, 1.0, -1.0]) @ minimal_brain_data.glm_betas.data,
            atol=1e-10,
        )

    def test_effect_mapping_returns_the_same_keys(self, minimal_brain_data):
        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)

        results = minimal_brain_data.compute_contrasts(
            {"a_vs_b": "cond_a - cond_b", "just_a": [0.0, 1.0, 0.0]}
        )

        assert set(results) == {"a_vs_b", "just_a"}
        assert all(isinstance(value, BrainData) for value in results.values())

    def test_inference_returns_owned_brain_data_payloads(self, minimal_brain_data):
        from nltools.models import ContrastResult

        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)

        result = minimal_brain_data.compute_contrasts("cond_a - cond_b", inference=True)

        assert isinstance(result, ContrastResult)
        maps = [
            result.effect,
            result.variance,
            result.standard_error,
            result.statistic,
            result.z_score,
            result.p_value,
        ]
        assert all(isinstance(payload, BrainData) for payload in maps)
        assert all(payload.X.is_empty() and payload.Y.is_empty() for payload in maps)
        for left, right in itertools.combinations(maps, 2):
            assert not np.shares_memory(left.data, right.data)
        assert isinstance(result.degrees_of_freedom, float)

        statistic_before = result.statistic.data.copy()
        result.effect.data[0] = 1234.0
        assert minimal_brain_data.glm_betas.data[0, 0] != 1234.0
        np.testing.assert_array_equal(result.statistic.data, statistic_before)

    def test_inference_effect_equals_the_effect_only_call(self, minimal_brain_data):
        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)

        effect = minimal_brain_data.compute_contrasts("cond_a")
        result = minimal_brain_data.compute_contrasts("cond_a", inference=True)

        np.testing.assert_allclose(result.effect.data, effect.data)

    def test_inference_mapping_returns_keyed_results(self, minimal_brain_data):
        from nltools.models import ContrastResult

        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)

        results = minimal_brain_data.compute_contrasts(
            {"a_vs_b": "cond_a - cond_b"}, inference=True
        )

        assert set(results) == {"a_vs_b"}
        assert isinstance(results["a_vs_b"], ContrastResult)
        assert isinstance(results["a_vs_b"].statistic, BrainData)

    def test_facade_does_not_parse_contrasts(self):
        from nltools.data.braindata import modeling

        assert not hasattr(modeling, "parse_contrast_string")
        assert not hasattr(modeling, "_functional_contrast")

    # ------------------------------------------------------------ predict

    def test_predict_returns_an_independent_copy_of_glm_predicted(
        self, minimal_brain_data
    ):
        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)

        predicted = minimal_brain_data.predict()

        assert isinstance(predicted, BrainData)
        np.testing.assert_array_equal(
            predicted.data, minimal_brain_data.glm_predicted.data
        )
        assert predicted.data is not minimal_brain_data.glm_predicted.data
        predicted.data[0, 0] = 4321.0
        assert minimal_brain_data.glm_predicted.data[0, 0] != 4321.0
        assert predicted.X.equals(minimal_brain_data.X)

    def test_fitted_model_wins_over_attached_Y(self, minimal_brain_data):
        import polars as pl

        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)
        minimal_brain_data.Y = pl.DataFrame(
            {"label": np.arange(len(minimal_brain_data)) % 2}
        )

        predicted = minimal_brain_data.predict()

        assert isinstance(predicted, BrainData)
        np.testing.assert_array_equal(
            predicted.data, minimal_brain_data.glm_predicted.data
        )

    def test_only_a_fitted_model_wins_over_stored_labels(self, minimal_brain_data):
        """An unfitted `model_` is not a model to predict from."""
        import polars as pl

        from nltools.data.braindata.prediction import _resolve_stored_y
        from nltools.models import Glm

        labels = np.arange(len(minimal_brain_data)) % 2
        minimal_brain_data.Y = pl.DataFrame({"label": labels})
        minimal_brain_data.model_ = Glm()

        np.testing.assert_array_equal(
            _resolve_stored_y(minimal_brain_data, None), labels
        )

        minimal_brain_data.fit(model="glm", X=self._design(minimal_brain_data))
        assert _resolve_stored_y(minimal_brain_data, None) is None

    def test_predict_with_a_new_design_delegates_to_glm(self, minimal_brain_data):
        from nltools.data import DesignMatrix

        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)

        rng = np.random.default_rng(3)
        new = DesignMatrix(
            {
                "cond_b": rng.normal(size=6),
                "intercept": np.ones(6),
                "cond_a": rng.normal(size=6),
            }
        )
        predicted = minimal_brain_data.predict(X=new)

        assert isinstance(predicted, BrainData)
        assert predicted.shape == (6, minimal_brain_data.shape[1])
        np.testing.assert_allclose(
            predicted.data,
            new[["intercept", "cond_a", "cond_b"]].to_numpy()
            @ minimal_brain_data.model_.coef_,
        )
        assert predicted.X.is_empty() and predicted.Y.is_empty()

    @pytest.mark.parametrize("columns", [["intercept", "cond_a"], None])
    def test_predict_with_a_mismatched_design_raises(self, minimal_brain_data, columns):
        from nltools.data import DesignMatrix

        design = self._design(minimal_brain_data)
        minimal_brain_data.fit(model="glm", X=design)

        rng = np.random.default_rng(4)
        names = columns or ["intercept", "cond_a", "cond_b", "extra"]
        new = DesignMatrix({name: rng.normal(size=6) for name in names})
        with pytest.raises(ValueError, match="fitted design columns"):
            minimal_brain_data.predict(X=new)

    # ------------------------------------------------------------- report

    def test_report_is_removed(self, minimal_brain_data):
        assert not hasattr(BrainData, "report")
        assert not hasattr(minimal_brain_data, "report")

    # ------------------------------------------------- second-level design

    def test_second_level_glm_workflow(self, minimal_brain_data):
        from nltools.data import DesignMatrix
        from nltools.utils import concatenate

        first_level_design = self._design(minimal_brain_data)
        effects = []
        for seed in range(6):
            subject = minimal_brain_data.copy()
            subject.data = subject.data + np.random.default_rng(seed).normal(
                size=subject.shape
            )
            fitted = subject.fit(model="glm", X=first_level_design, inplace=False)
            effects.append(fitted.compute_contrasts("cond_a - cond_b"))

        group = concatenate(effects)
        assert group.shape == (6, minimal_brain_data.shape[1])

        second_level = DesignMatrix(
            {"intercept": np.ones(6), "age": np.linspace(-1, 1, 6)}
        )
        group.fit(model="glm", X=second_level, glm_noise_model="ols")
        result = group.compute_contrasts("age", inference=True)

        assert isinstance(result.statistic, BrainData)
        assert result.statistic.shape[-1] == minimal_brain_data.shape[1]
