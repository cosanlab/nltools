"""Contract tests for `nltools.models._Glm` (development/specs/glm.md).

`_Glm` is a numerical estimator over a precomputed `DesignMatrix` and a
one- or two-dimensional response. It delegates fitting to nilearn's
`run_glm` and contrast inference to nilearn's `Contrast`, so the numerical
references here are direct nilearn calls, not hand-rolled formulas.
"""

import dataclasses
import inspect

import numpy as np
import pytest
from nilearn.glm import compute_contrast as nilearn_compute_contrast
from nilearn.glm.first_level import run_glm

from nltools.models import ContrastResult, _Glm


def reference_fit(design, y, noise_model="ols", **run_glm_kwargs):
    """Fit the same model straight through nilearn for a numerical reference.

    `run_glm` itself defaults to `'ar1'`; this helper mirrors `_Glm`'s own
    `'ols'` default so a reference call has to opt in to autoregression.
    """
    response = y[:, None] if y.ndim == 1 else y
    return run_glm(
        response, design.to_numpy(), noise_model=noise_model, **run_glm_kwargs
    )


class TestConstructor:
    def test_defaults(self):
        model = _Glm()
        assert model.noise_model == "ols"
        assert model.bins == 100
        assert model.n_jobs == 1
        assert model.random_state is None
        assert model.is_fitted_ is False

    def test_every_argument_is_keyword_only(self):
        parameters = inspect.signature(_Glm.__init__).parameters
        assert list(parameters) == [
            "self",
            "noise_model",
            "bins",
            "n_jobs",
            "random_state",
        ]
        assert all(
            parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            for name in parameters
            if name != "self"
        )

    @pytest.mark.parametrize("noise_model", ["ar1"])
    def test_accepted_noise_models(self, noise_model):
        assert _Glm(noise_model=noise_model).noise_model == noise_model

    @pytest.mark.parametrize("noise_model", ["ar"])
    def test_invalid_noise_model_raises_value_error(self, noise_model):
        with pytest.raises(ValueError, match="noise_model"):
            _Glm(noise_model=noise_model)

    @pytest.mark.parametrize("bins", [0])
    def test_non_positive_bins_raises_value_error(self, bins):
        with pytest.raises(ValueError, match="bins"):
            _Glm(bins=bins)

    @pytest.mark.parametrize("bins", [1.5])
    def test_non_integer_bins_raises_type_error(self, bins):
        with pytest.raises(TypeError, match="bins"):
            _Glm(bins=bins)


class TestFitValidation:
    def test_fit_returns_self(self, glm_design, glm_targets):
        model = _Glm()
        assert model.fit(glm_design, glm_targets) is model

    @pytest.mark.parametrize("frame", ["numpy"])
    def test_non_design_matrix_raises_type_error(self, glm_design, glm_targets, frame):
        import pandas as pd
        import polars as pl

        array = glm_design.to_numpy()
        X = {
            "numpy": array,
            "pandas": pd.DataFrame(array, columns=glm_design.columns),
            "polars": pl.DataFrame(dict(zip(glm_design.columns, array.T))),
        }[frame]
        with pytest.raises(TypeError, match="DesignMatrix"):
            _Glm().fit(X, glm_targets)

    def test_sample_count_mismatch_raises_value_error(self, glm_design):
        with pytest.raises(ValueError, match="sample"):
            _Glm().fit(glm_design, np.zeros((glm_design.shape[0] + 1, 2)))

    def test_three_dimensional_y_raises_value_error(self, glm_design):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            _Glm().fit(glm_design, np.zeros((glm_design.shape[0], 2, 2)))


class TestFittedState:
    def test_ols_coefficients_match_lstsq(self, glm_design, glm_targets, fitted_glm):
        expected = np.linalg.lstsq(glm_design.to_numpy(), glm_targets, rcond=None)[0]
        np.testing.assert_allclose(fitted_glm.coef_, expected, atol=1e-10)

    def test_no_intercept_is_ever_added(self, glm_design, glm_targets):
        without_intercept = glm_design[["condition_a", "condition_b"]]
        model = _Glm().fit(without_intercept, glm_targets)
        assert model.n_features_in_ == 2
        assert model.feature_names_in_ == ("condition_a", "condition_b")
        expected = np.linalg.lstsq(
            without_intercept.to_numpy(), glm_targets, rcond=None
        )[0]
        np.testing.assert_allclose(model.coef_, expected, atol=1e-10)

    def test_two_dimensional_shapes(self, glm_design, glm_targets, fitted_glm):
        n_samples, n_features = glm_design.shape
        assert fitted_glm.coef_.shape == (n_features, 3)
        assert fitted_glm.predicted_.shape == (n_samples, 3)
        assert fitted_glm.residuals_.shape == (n_samples, 3)
        assert fitted_glm.r2_.shape == (3,)
        assert fitted_glm.n_samples_ == n_samples
        assert fitted_glm.n_features_in_ == n_features
        assert fitted_glm.n_targets_ == 3
        assert fitted_glm.feature_names_in_ == tuple(glm_design.columns)
        assert isinstance(fitted_glm.feature_names_in_, tuple)
        assert fitted_glm.is_fitted_ is True

    def test_one_dimensional_target_is_squeezed(self, glm_design, glm_targets):
        n_samples, n_features = glm_design.shape
        model = _Glm().fit(glm_design, glm_targets[:, 0])
        assert model.coef_.shape == (n_features,)
        assert model.predicted_.shape == (n_samples,)
        assert model.residuals_.shape == (n_samples,)
        assert isinstance(model.r2_, float)
        assert model.n_targets_ == 1

    def test_single_column_target_keeps_its_axis(self, glm_design, glm_targets):
        n_samples, n_features = glm_design.shape
        model = _Glm().fit(glm_design, glm_targets[:, :1])
        assert model.coef_.shape == (n_features, 1)
        assert model.predicted_.shape == (n_samples, 1)
        assert model.residuals_.shape == (n_samples, 1)
        assert model.r2_.shape == (1,)
        assert model.n_targets_ == 1

    @pytest.mark.parametrize("targets", ["two_dimensional"])
    def test_coefficients_do_not_alias_the_retained_state(
        self, glm_design, glm_targets, targets
    ):
        """Mutating the public `coef_` must not reach later contrasts."""
        y = glm_targets[:, 0] if targets == "one_dimensional" else glm_targets
        model = _Glm().fit(glm_design, y)
        before = model.compute_contrasts([1, -1, 0])

        assert model.coef_.base is None
        model.coef_[0] = 999.0
        np.testing.assert_array_equal(model.compute_contrasts([1, -1, 0]), before)

    def test_predictions_and_residuals_are_in_observation_space(
        self, glm_design, glm_targets, fitted_glm
    ):
        np.testing.assert_allclose(
            fitted_glm.predicted_, glm_design.to_numpy() @ fitted_glm.coef_
        )
        np.testing.assert_allclose(
            fitted_glm.residuals_, glm_targets - fitted_glm.predicted_
        )

    def test_autoregressive_predictions_use_the_unwhitened_design(
        self, glm_design, ar_targets
    ):
        model = _Glm(noise_model="ar1").fit(glm_design, ar_targets)
        np.testing.assert_allclose(
            model.predicted_, glm_design.to_numpy() @ model.coef_
        )
        np.testing.assert_allclose(model.residuals_, ar_targets - model.predicted_)

    def test_autoregressive_coefficients_match_nilearn(self, glm_design, ar_targets):
        model = _Glm(noise_model="ar1").fit(glm_design, ar_targets)
        labels, results = reference_fit(glm_design, ar_targets, noise_model="ar1")
        expected = np.zeros_like(model.coef_)
        for label, result in results.items():
            expected[:, labels == label] = result.theta
        np.testing.assert_array_equal(model.coef_, expected)


class TestContrastResolution:
    def test_string_effect_matches_coefficient_arithmetic(self, fitted_glm):
        effect = fitted_glm.compute_contrasts("condition_a - condition_b")
        np.testing.assert_allclose(effect, fitted_glm.coef_[0] - fitted_glm.coef_[1])

    def test_weighted_string_expression(self, fitted_glm):
        effect = fitted_glm.compute_contrasts("2 * condition_a - condition_b")
        np.testing.assert_allclose(
            effect, 2 * fitted_glm.coef_[0] - fitted_glm.coef_[1]
        )

    def test_numeric_vector_effect(self, fitted_glm):
        effect = fitted_glm.compute_contrasts([1, -1, 0])
        np.testing.assert_allclose(effect, fitted_glm.coef_[0] - fitted_glm.coef_[1])

    def test_before_fit_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="fit"):
            _Glm().compute_contrasts("condition_a")

    def test_unknown_column_lists_available_names(self, fitted_glm):
        with pytest.raises(ValueError) as excinfo:
            fitted_glm.compute_contrasts("condition_z")
        message = str(excinfo.value)
        assert "condition_a" in message and "intercept" in message

    def test_invalid_expression_raises_value_error(self, fitted_glm):
        with pytest.raises(ValueError):
            fitted_glm.compute_contrasts("condition_a - -")

    def test_all_zero_expression_raises_value_error(self, fitted_glm):
        with pytest.raises(ValueError):
            fitted_glm.compute_contrasts("condition_a - condition_a")


class TestContrastInference:
    def test_matches_nilearn(self, glm_design, glm_targets, fitted_glm):
        vector = np.array([1.0, -1.0, 0.0])
        labels, results = reference_fit(glm_design, glm_targets)
        reference = nilearn_compute_contrast(labels, results, vector)

        result = fitted_glm.compute_contrasts([1, -1, 0], inference=True)
        assert isinstance(result, ContrastResult)
        np.testing.assert_allclose(result.effect, reference.effect_size().ravel())
        np.testing.assert_allclose(result.variance, reference.effect_variance().ravel())
        np.testing.assert_allclose(result.statistic, reference.stat().ravel())
        np.testing.assert_allclose(result.p_value, reference.p_value().ravel())
        np.testing.assert_allclose(result.z_score, reference.z_score().ravel())
        assert result.degrees_of_freedom == reference.dof

    def test_standard_error_is_the_raw_square_root_of_variance(self, fitted_glm):
        result = fitted_glm.compute_contrasts("condition_a", inference=True)
        np.testing.assert_array_equal(result.standard_error, np.sqrt(result.variance))

    def test_negative_variance_is_neither_clipped_nor_made_absolute(self, fitted_glm):
        """Spec: no `abs`, no clipping — a negative variance survives to a nan SE.

        A real fit cannot produce a negative dispersion, so the guard is
        exercised on a state whose sign has been flipped. `standard_error` is
        raw `np.sqrt(variance)` and may therefore be non-finite.
        """
        from nltools.models.glm import _contrast_statistics

        flipped = dataclasses.replace(
            fitted_glm._fit_state, dispersion=-fitted_glm._fit_state.dispersion
        )
        with np.errstate(invalid="ignore"):
            values = _contrast_statistics(flipped, np.array([1.0, -1.0, 0.0]))
        assert np.all(values["variance"] < 0)
        assert np.all(np.isnan(values["standard_error"]))
