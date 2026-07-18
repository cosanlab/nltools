import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from nltools.data import BrainData


class TestBrainDataModeling:
    def test_compute_contrasts_error_not_fitted(self, minimal_brain_data):
        """Test error when compute_contrasts() called before fit()."""
        with pytest.raises(RuntimeError, match="Must run .fit"):
            minimal_brain_data.compute_contrasts([1, -1, 0])

    @pytest.mark.slow
    def test_compute_contrasts(self, minimal_brain_data):
        """Test all contrast input types: numeric vector, string, dict."""
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condA": np.random.randn(len(minimal_brain_data)),
                "condB": np.random.randn(len(minimal_brain_data)),
            }
        )
        minimal_brain_data.fit(model="glm", X=design_matrix)

        # Numeric vector — single-image BrainData has 1D shape (n_voxels,)
        contrast = minimal_brain_data.compute_contrasts([0, 1, -1])
        assert isinstance(contrast, BrainData)
        assert contrast.shape[-1] == minimal_brain_data.shape[1]

        # String parsing
        contrast = minimal_brain_data.compute_contrasts("condA - condB")
        assert isinstance(contrast, BrainData)
        assert contrast.shape[-1] == minimal_brain_data.shape[1]

        # Dict of contrasts
        contrasts = {"A_vs_B": "condA - condB", "avg_effect": [0, 0.5, 0.5]}
        results = minimal_brain_data.compute_contrasts(contrasts)
        assert isinstance(results, dict)
        assert "A_vs_B" in results and "avg_effect" in results
        assert isinstance(results["A_vs_B"], BrainData)

    @pytest.mark.slow
    def test_compute_contrasts_invalid_length(self, minimal_brain_data):
        """Test error for invalid contrast vector length."""
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condA": np.random.randn(len(minimal_brain_data)),
                "condB": np.random.randn(len(minimal_brain_data)),
            }
        )
        minimal_brain_data.fit(model="glm", X=design_matrix)

        with pytest.raises(ValueError, match="Contrast vector length.*must match"):
            minimal_brain_data.compute_contrasts([1, -1])

    @pytest.mark.slow
    def test_compute_contrasts_type_distinguishes_t_and_beta(self, minimal_brain_data):
        """statistic='t' returns t-stats; statistic='beta' returns effect sizes.

        Regression guard: the earlier implementation always returned raw
        linear-combinations-of-betas (effect sizes) while advertising 't',
        which broke first-level thresholding.
        """
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condA": np.random.randn(len(minimal_brain_data)),
                "condB": np.random.randn(len(minimal_brain_data)),
            }
        )
        minimal_brain_data.fit(model="glm", X=design_matrix)

        t_map = minimal_brain_data.compute_contrasts("condA - condB", statistic="t")
        beta_map = minimal_brain_data.compute_contrasts(
            "condA - condB", statistic="beta"
        )
        z_map = minimal_brain_data.compute_contrasts("condA - condB", statistic="z")

        for m in (t_map, beta_map, z_map):
            assert isinstance(m, BrainData)
            assert m.shape[-1] == minimal_brain_data.shape[1]

        t_arr = np.asarray(t_map.data).squeeze()
        b_arr = np.asarray(beta_map.data).squeeze()
        z_arr = np.asarray(z_map.data).squeeze()

        # Beta is the effect size (linear combo of betas). t is effect / SE.
        # They must not be the same array (they were, before the fix).
        assert not np.allclose(t_arr, b_arr), (
            "t-map and beta-map should differ; compute_contrasts is returning "
            "raw effect sizes regardless of statistic"
        )
        # z and t are monotonically related (sign preserved, magnitude close)
        assert np.all(np.sign(t_arr) == np.sign(z_arr))
        assert np.corrcoef(t_arr, z_arr)[0, 1] > 0.999

    def test_compute_contrasts_invalid_type(self, minimal_brain_data):
        """Unknown statistic raises ValueError with supported values listed."""
        with pytest.raises(ValueError, match="statistic must be"):
            # not fitted yet but validation happens before that path; set dummy attr
            minimal_brain_data.glm_betas = minimal_brain_data[0]
            minimal_brain_data.model_ = object()
            minimal_brain_data.compute_contrasts([1, -1, 0], statistic="F")

    @pytest.mark.slow
    def test_compute_contrasts_all_single_returns_bundle(self, minimal_brain_data):
        """statistic='all' returns a flat dict with beta/t/z/p/se for one contrast."""
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condA": np.random.randn(len(minimal_brain_data)),
                "condB": np.random.randn(len(minimal_brain_data)),
            }
        )
        minimal_brain_data.fit(model="glm", X=design_matrix)

        res = minimal_brain_data.compute_contrasts("condA - condB", statistic="all")
        assert isinstance(res, dict)
        assert set(res.keys()) == {"beta", "t", "z", "p", "se"}
        for key in ("beta", "t", "z", "p", "se"):
            assert isinstance(res[key], BrainData)
            assert res[key].shape[-1] == minimal_brain_data.shape[1]

        # Consistency: individual calls agree with the bundle.
        t_only = minimal_brain_data.compute_contrasts("condA - condB", statistic="t")
        np.testing.assert_allclose(np.asarray(t_only.data), np.asarray(res["t"].data))

    @pytest.mark.slow
    def test_compute_contrasts_all_dict_returns_nested(self, minimal_brain_data):
        """statistic='all' + dict input returns nested {name: {beta,t,z,p,se}}."""
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "condA": np.random.randn(len(minimal_brain_data)),
                "condB": np.random.randn(len(minimal_brain_data)),
            }
        )
        minimal_brain_data.fit(model="glm", X=design_matrix)

        res = minimal_brain_data.compute_contrasts(
            {"A_vs_B": "condA - condB", "just_A": [0, 1, 0]},
            statistic="all",
        )
        assert set(res.keys()) == {"A_vs_B", "just_A"}
        for bundle in res.values():
            assert set(bundle.keys()) == {"beta", "t", "z", "p", "se"}
            for m in bundle.values():
                assert isinstance(m, BrainData)

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
        from nltools.models import Glm

        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "X1": np.random.randn(len(minimal_brain_data)),
            }
        )
        minimal_brain_data.fit(model="glm", noise_model="ols", X=design_matrix)

        assert hasattr(minimal_brain_data, "model_")
        assert isinstance(minimal_brain_data.model_, Glm)
        assert hasattr(minimal_brain_data, "glm_betas")
        assert hasattr(minimal_brain_data, "glm_t")

        predictions = minimal_brain_data.predict()
        assert predictions.shape == minimal_brain_data.shape

    @pytest.mark.slow
    def test_fit_passes_kwargs_to_model(self, minimal_brain_data):
        """Test fit() passes additional kwargs to model constructor."""
        X = np.random.randn(len(minimal_brain_data), 10)

        minimal_brain_data.fit(model="ridge", alpha=1.0, device="cpu", X=X)
        assert minimal_brain_data.model_.device == "cpu"

        design_matrix = pd.DataFrame({"Intercept": np.ones(len(minimal_brain_data))})
        minimal_brain_data.fit(model="glm", noise_model="ar1", X=design_matrix)
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
        assert minimal_brain_data.ridge_weights.mask is minimal_brain_data.mask

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

    def test_fit_inplace_false_returns_fit_dataclass_ridge(self, minimal_brain_data):
        """Test inplace=False returns Fit dataclass for Ridge."""
        from nltools.data.fitresults import Fit

        brain = minimal_brain_data.copy()
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

        fit = brain.fit(model="ridge", alpha=1.0, X=X_train, inplace=False)

        assert isinstance(fit, Fit)
        assert "fitted_values" in fit.available()
        assert "weights" in fit.available()
        assert "scores" in fit.available()
        assert fit.fitted_values.shape == brain.shape
        assert fit.weights.shape == (10, brain.shape[1])
        assert fit.scores.shape == (brain.shape[1],)
        assert not hasattr(brain, "ridge_weights")
        assert hasattr(brain, "model_")
        np.testing.assert_array_equal(brain.data, original_data)

    @pytest.mark.slow
    def test_fit_inplace_false_returns_fit_dataclass_ridge_cv(self, minimal_brain_data):
        """Test inplace=False returns Fit dataclass with CV results for Ridge."""
        from nltools.data.fitresults import Fit

        brain = minimal_brain_data.copy()
        X_train = np.random.randn(len(brain), 10)

        fit = brain.fit(model="ridge", alpha=1.0, X=X_train, cv=3, inplace=False)

        assert isinstance(fit, Fit)
        assert "cv_scores" in fit.available()
        assert "cv_mean_score" in fit.available()
        assert "cv_predictions" in fit.available()
        assert "cv_folds" in fit.available()
        assert fit.cv_scores.shape == (3, brain.shape[1])
        assert fit.cv_mean_score.shape == (brain.shape[1],)
        assert fit.cv_predictions.shape == brain.shape
        assert fit.cv_folds.shape == (len(brain),)

    @pytest.mark.slow
    def test_fit_inplace_false_returns_fit_dataclass_glm(self, minimal_brain_data):
        """Test inplace=False returns Fit dataclass for GLM."""
        from nltools.data.fitresults import Fit

        brain = minimal_brain_data.copy()
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(brain)),
                "X1": np.random.randn(len(brain)),
            }
        )
        original_data = brain.data.copy()

        fit = brain.fit(model="glm", noise_model="ols", X=design_matrix, inplace=False)

        assert isinstance(fit, Fit)
        assert "fitted_values" in fit.available()
        assert "betas" in fit.available()
        assert "t_stats" in fit.available()
        assert "p_values" in fit.available()
        assert "se" in fit.available()
        assert "residuals" in fit.available()
        assert "r2" in fit.available()
        assert fit.fitted_values.shape == brain.shape
        assert fit.betas.shape == (2, brain.shape[1])
        assert not hasattr(brain, "glm_betas")
        assert hasattr(brain, "model_")
        np.testing.assert_array_equal(brain.data, original_data)

    def test_fit_inplace_false_allows_predict(self, minimal_brain_data):
        """Test that inplace=False still allows predict() to work."""
        X_train = np.random.randn(len(minimal_brain_data), 10)
        minimal_brain_data.fit(model="ridge", alpha=1.0, X=X_train, inplace=False)

        X_test = np.random.randn(20, 10)
        predictions = minimal_brain_data.predict(X=X_test)
        assert predictions.shape == (20, minimal_brain_data.shape[1])

    def test_fit_inplace_false_serialization(self, minimal_brain_data):
        """Test Fit dataclass serialization roundtrip."""
        from nltools.data.fitresults import Fit
        import os

        X_train = np.random.randn(len(minimal_brain_data), 10)
        fit = minimal_brain_data.fit(model="ridge", alpha=1.0, X=X_train, inplace=False)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            np.savez(f.name, **fit.asdict())
            loaded = np.load(f.name)
            fit_reconstructed = Fit(**{k: loaded[k] for k in loaded.files})
            os.unlink(f.name)

        np.testing.assert_array_equal(
            fit.fitted_values, fit_reconstructed.fitted_values
        )
        np.testing.assert_array_equal(fit.weights, fit_reconstructed.weights)
        np.testing.assert_array_equal(fit.scores, fit_reconstructed.scores)

    @pytest.mark.slow
    def test_glm_fit_numerical_correctness(self, minimal_brain_data):
        """Test fit(model='glm') produces numerically correct results."""
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "X1": np.random.randn(len(minimal_brain_data)),
            }
        )

        minimal_brain_data.fit(model="glm", noise_model="ols", X=design_matrix)

        assert not np.isnan(minimal_brain_data.glm_betas.data).any()
        assert not np.allclose(minimal_brain_data.glm_betas.data, 0)
        assert not np.isnan(minimal_brain_data.glm_t.data).any()
        assert minimal_brain_data.model_.progress_bar is False

    @pytest.mark.slow
    def test_glm_fit_suppresses_drift_model_warning(self, minimal_brain_data):
        """Test fit(model='glm') suppresses drift_model warning."""
        import warnings

        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "X1": np.random.randn(len(minimal_brain_data)),
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            minimal_brain_data.fit(
                model="glm", noise_model="ols", X=design_matrix, drift_model="cosine"
            )

            drift_warnings = [
                warn
                for warn in w
                if "drift_model" in str(warn.message).lower()
                and "will be ignored" in str(warn.message).lower()
            ]
            assert len(drift_warnings) == 0

        assert minimal_brain_data.model_.is_fitted_
        assert minimal_brain_data.model_.progress_bar is False

        # Verify progress_bar=True is respected
        minimal_brain_data.fit(
            model="glm", noise_model="ols", X=design_matrix, progress_bar=True
        )
        assert minimal_brain_data.model_.progress_bar is True

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

    def test_fit_scale_applies_mean_scaling(self, minimal_brain_data):
        """scale=True applies nilearn per-voxel mean_scaling (percent signal change)."""
        from nilearn.glm.first_level import mean_scaling

        X = np.random.randn(len(minimal_brain_data), 10)
        bd = minimal_brain_data.copy()
        bd.data = bd.data + 100.0  # positive baseline -> clean PSC
        orig = bd.data.copy()
        bd.fit(model="ridge", alpha=1.0, X=X, scale=True, standardize=None)
        np.testing.assert_allclose(bd.data, mean_scaling(orig, axis=0)[0])

    def test_fit_standardize_zscore(self, minimal_brain_data):
        """standardize='zscore' z-scores each voxel across time; scale off = raw units."""
        X = np.random.randn(len(minimal_brain_data), 10)
        bd = minimal_brain_data.copy()
        bd.data = bd.data + 100.0
        oracle = bd.standardize(method="zscore").data
        bd.fit(model="ridge", alpha=1.0, X=X, scale=False, standardize="zscore")
        np.testing.assert_allclose(bd.data, oracle)

    def test_fit_scale_then_standardize_order(self, minimal_brain_data):
        """Preprocessing order is scale THEN standardize (a reversed order would
        corrupt via mean_scaling on centered data)."""
        from nilearn.glm.first_level import mean_scaling

        X = np.random.randn(len(minimal_brain_data), 10)
        bd = minimal_brain_data.copy()
        bd.data = bd.data + 100.0
        orig = bd.data.copy()
        scaled = bd.copy()
        scaled.data = mean_scaling(orig, axis=0)[0]
        oracle = scaled.standardize(method="center").data
        bd.fit(model="ridge", alpha=1.0, X=X, scale=True, standardize="center")
        np.testing.assert_allclose(bd.data, oracle)

    def test_fit_ridge_defaults_zscore_only(self, minimal_brain_data):
        """Ridge 'auto' default: standardize='zscore', scale OFF (no warning)."""
        import warnings

        X = np.random.randn(len(minimal_brain_data), 10)
        bd = minimal_brain_data.copy()
        bd.data = bd.data + 100.0
        oracle = bd.standardize(method="zscore").data
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bd.fit(model="ridge", alpha=1.0, X=X)  # no scale/standardize -> auto
        assert not any("redundant" in str(wi.message) for wi in w)
        np.testing.assert_allclose(bd.data, oracle)

    def test_scale_zscore_redundant_warns(self, minimal_brain_data):
        """scale=True + standardize='zscore' warns (scale is a no-op there)."""
        X = np.random.randn(len(minimal_brain_data), 10)
        bd = minimal_brain_data.copy()
        bd.data = bd.data + 100.0
        with pytest.warns(UserWarning, match="redundant"):
            bd.fit(model="ridge", alpha=1.0, X=X, scale=True, standardize="zscore")

    def test_fit_glm_defaults_no_preprocessing(self, minimal_brain_data):
        """GLM 'auto' default: no scaling, no standardization — data untouched."""
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "X1": np.random.randn(len(minimal_brain_data)),
            }
        )
        bd = minimal_brain_data.copy()
        bd.data = bd.data + 100.0
        orig = bd.data.copy()
        bd.fit(model="glm", noise_model="ols", X=design_matrix)
        np.testing.assert_allclose(bd.data, orig)

    def test_fit_scale_disabled_leaves_data_unchanged(self, minimal_brain_data):
        """scale=False, standardize=None leaves data unchanged."""
        X = np.random.randn(len(minimal_brain_data), 10)
        bd = minimal_brain_data.copy()
        orig = bd.data.copy()
        bd.fit(model="ridge", alpha=1.0, X=X, scale=False, standardize=None)
        np.testing.assert_allclose(bd.data, orig)

    def test_ridge_intercept_with_centering_warns(self, minimal_brain_data):
        """Ridge fit_intercept=True is redundant when the data is centered by
        standardization/scaling — warn loudly."""
        X = np.random.randn(len(minimal_brain_data), 10)
        bd = minimal_brain_data.copy()
        bd.data = bd.data + 100.0
        with pytest.warns(
            UserWarning, match="intercept.*redundant|redundant.*intercept"
        ):
            bd.fit(
                model="ridge", alpha=1.0, X=X, standardize="zscore", fit_intercept=True
            )

    def test_ridge_intercept_no_centering_ok(self, minimal_brain_data):
        """fit_intercept=True is fine (no warning) when no centering is applied —
        that is exactly the raw-offset case intercepts exist for."""
        import warnings

        X = np.random.randn(len(minimal_brain_data), 10)
        bd = minimal_brain_data.copy()
        bd.data = bd.data + 100.0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bd.fit(
                model="ridge",
                alpha=1.0,
                X=X,
                scale=False,
                standardize=None,
                fit_intercept=True,
            )
        assert not any("intercept" in str(wi.message).lower() for wi in w)

    def test_glm_predict_new_design_returns_brain_data(self, minimal_brain_data):
        """F182: bd.predict(X=new_design) works for GLM and returns BrainData
        holding X_new @ coef_ (parity with the Ridge facade path)."""
        design = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "X1": np.random.randn(len(minimal_brain_data)),
            }
        )
        minimal_brain_data.fit(model="glm", noise_model="ols", X=design)

        X_new = np.column_stack(
            [np.ones(6), np.random.randn(6)]
        )  # 6 new timepoints, same 2 regressors
        pred = minimal_brain_data.predict(X=X_new)

        assert isinstance(pred, BrainData)
        assert pred.data.shape == (6, minimal_brain_data.shape[1])
        np.testing.assert_allclose(pred.data, X_new @ minimal_brain_data.model_.coef_)

    def test_glm_report_returns_html(self, minimal_brain_data):
        """bd.report() delegates to nilearn and returns an HTMLReport."""
        design = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "X1": np.random.randn(len(minimal_brain_data)),
            }
        )
        minimal_brain_data.fit(model="glm", noise_model="ols", X=design)
        rep = minimal_brain_data.report(contrasts={"X1": np.array([0.0, 1.0])})
        assert type(rep).__name__ == "HTMLReport"

    def test_report_requires_fitted_glm(self, minimal_brain_data):
        with pytest.raises(RuntimeError, match="requires a fitted GLM"):
            minimal_brain_data.report(contrasts={"X1": np.array([0.0, 1.0])})

    def test_fit_scale_value_removed(self, minimal_brain_data):
        """scale_value is gone (nilearn PSC is fixed at x100)."""
        X = np.random.randn(len(minimal_brain_data), 10)
        with pytest.raises(TypeError):
            minimal_brain_data.fit(model="ridge", alpha=1.0, X=X, scale_value=1000.0)

    def test_fit_scale_inplace_false(self, minimal_brain_data):
        """fit() with preprocessing and inplace=False doesn't modify original."""
        from nltools.data.fitresults import Fit

        X = np.random.randn(len(minimal_brain_data), 10)
        original_data = minimal_brain_data.data.copy()

        result = minimal_brain_data.fit(
            model="ridge", alpha=1.0, X=X, inplace=False, scale=True, standardize=None
        )

        assert isinstance(result, Fit)
        np.testing.assert_allclose(minimal_brain_data.data, original_data)

    def test_predict_with_no_X_uses_training_data(self, minimal_brain_data):
        """Test predict() with no X returns predictions on training data."""
        X_train = np.random.randn(len(minimal_brain_data), 10)
        minimal_brain_data.fit(model="ridge", alpha=1.0, X=X_train)

        predictions_explicit = minimal_brain_data.predict(X=X_train)
        predictions_implicit = minimal_brain_data.predict()

        np.testing.assert_allclose(predictions_explicit.data, predictions_implicit.data)
        assert predictions_implicit.shape == minimal_brain_data.shape

    # ==================== Ridge CV Tests ====================

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

    def test_fit_ridge_cv_with_insufficient_samples(self, tiny_brain_data_for_cv):
        """Test fit() raises error when cv folds > n_samples."""
        brain_data, X = tiny_brain_data_for_cv
        with pytest.raises(ValueError, match="Cannot have number of splits.*greater"):
            brain_data.fit(model="ridge", alpha=1.0, cv=10, X=X)

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

    # ==================== design_clean kwargs (GLM only) ====================

    @pytest.mark.slow
    def test_design_clean_drops_perfectly_correlated(self, minimal_brain_data):
        """design_clean=True (default) drops perfectly correlated regressors."""
        n = len(minimal_brain_data)
        rng = np.random.default_rng(42)
        a = rng.standard_normal(n)
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(n),
                "condA": a,
                "condA_dup": a,  # r = 1.0 with condA
            }
        )
        minimal_brain_data.fit(model="glm", X=design_matrix)
        # condA_dup dropped, leaving Intercept and condA
        assert minimal_brain_data.glm_betas.shape[0] == 2

    @pytest.mark.slow
    def test_design_clean_false_keeps_correlated(self, minimal_brain_data):
        """design_clean=False keeps all regressors regardless of correlation."""
        n = len(minimal_brain_data)
        rng = np.random.default_rng(42)
        a = rng.standard_normal(n)
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(n),
                "condA": a,
                "condA_dup": a,
            }
        )
        minimal_brain_data.fit(model="glm", X=design_matrix, design_clean=False)
        assert minimal_brain_data.glm_betas.shape[0] == 3

    @pytest.mark.slow
    def test_design_clean_thresh_plumbing(self, minimal_brain_data):
        """design_clean_thresh changes the drop threshold."""
        n = len(minimal_brain_data)
        rng = np.random.default_rng(42)
        a = rng.standard_normal(n)
        # Construct b correlated with a at r ~= 0.7
        noise = rng.standard_normal(n)
        b = 0.7 * a + np.sqrt(1 - 0.7**2) * noise
        r = abs(np.corrcoef(a, b)[0, 1])
        assert 0.5 < r < 0.95, f"setup invariant violated: |r|={r}"

        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(n),
                "condA": a,
                "condB": b,
            }
        )

        # Default thresh=0.95: keeps both
        bd_default = minimal_brain_data.copy()
        bd_default.fit(model="glm", X=design_matrix)
        assert bd_default.glm_betas.shape[0] == 3

        # thresh=0.5: drops one
        bd_strict = minimal_brain_data.copy()
        bd_strict.fit(model="glm", X=design_matrix, design_clean_thresh=0.5)
        assert bd_strict.glm_betas.shape[0] == 2

    @pytest.mark.slow
    def test_design_clean_exclude_confounds_plumbing(self, minimal_brain_data):
        """design_clean_exclude_confounds=True skips confounds from correlation check."""
        from nltools.data.designmatrix import DesignMatrix

        n = len(minimal_brain_data)
        rng = np.random.default_rng(42)
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)

        # task1, task2, motion_x (= copy of task1, marked confound)
        dm = DesignMatrix(
            pd.DataFrame({"task1": a, "task2": b, "motion_x": a}),
            confounds=["motion_x"],
        )

        # Default exclude_confounds=False: motion_x correlated with task1 → dropped
        bd_default = minimal_brain_data.copy()
        bd_default.fit(model="glm", X=dm)
        assert bd_default.glm_betas.shape[0] == 2

        # exclude_confounds=True: motion_x excluded from check → all kept
        bd_excl = minimal_brain_data.copy()
        bd_excl.fit(model="glm", X=dm, design_clean_exclude_confounds=True)
        assert bd_excl.glm_betas.shape[0] == 3

    @pytest.mark.slow
    def test_design_clean_noop_on_clean_design(self, minimal_brain_data):
        """design_clean=True is a no-op on an already clean design."""
        n = len(minimal_brain_data)
        rng = np.random.default_rng(42)
        design_matrix = pd.DataFrame(
            {
                "Intercept": np.ones(n),
                "condA": rng.standard_normal(n),
                "condB": rng.standard_normal(n),
            }
        )
        minimal_brain_data.fit(model="glm", X=design_matrix)
        assert minimal_brain_data.glm_betas.shape[0] == 3


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

    def test_ttest2_two_sample(self, minimal_brain_data):
        """Two-sample voxelwise t-test returns dict of BrainData."""
        from scipy.stats import ttest_ind

        other = minimal_brain_data[:25]
        subset = minimal_brain_data[25:]

        result = subset.ttest2(other)

        assert set(result.keys()) == {"t", "p"}
        expected_t, expected_p = ttest_ind(subset.data, other.data, axis=0)
        np.testing.assert_allclose(result["t"].data, expected_t)
        np.testing.assert_allclose(result["p"].data, expected_p)

    def test_ttest2_welch(self, minimal_brain_data):
        """equal_var=False triggers Welch's t-test."""
        from scipy.stats import ttest_ind

        other = minimal_brain_data[:25]
        subset = minimal_brain_data[25:]

        result = subset.ttest2(other, equal_var=False)
        expected_t, _ = ttest_ind(subset.data, other.data, axis=0, equal_var=False)
        np.testing.assert_allclose(result["t"].data, expected_t)

    def test_ttest2_mismatched_voxels_raises(self, minimal_brain_data):
        """Mismatched n_voxels raises ValueError."""
        import nibabel as nib

        # Build a second BrainData with different n_voxels
        spatial_shape = (2, 2, 1)
        mask_data = np.zeros(spatial_shape, dtype=bool)
        mask_data.flat[:3] = True
        affine = np.eye(4)
        volume_4d = np.random.randn(*spatial_shape, 10)
        other = BrainData(
            nib.Nifti1Image(volume_4d, affine),
            mask=nib.Nifti1Image(mask_data.astype(np.float32), affine),
        )
        with pytest.raises(ValueError, match="n_voxels"):
            minimal_brain_data.ttest2(other)


class TestBrainDataRidgeCV:
    """Splitter forwarding, generator rejection, and .size."""

    def test_size_property(self, minimal_brain_data):
        assert minimal_brain_data.size == minimal_brain_data.data.size

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
