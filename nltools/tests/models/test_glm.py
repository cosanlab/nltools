"""Contract tests for `nltools.models.Glm` (docs/development/specs/glm.md).

`Glm` is a numerical estimator over a precomputed `DesignMatrix` and a
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

import nltools.models
from nltools.data import DesignMatrix
from nltools.models import ContrastResult, Glm


def reference_fit(design, y, noise_model="ols", **run_glm_kwargs):
    """Fit the same model straight through nilearn for a numerical reference.

    `run_glm` itself defaults to `'ar1'`; this helper mirrors `Glm`'s own
    `'ols'` default so a reference call has to opt in to autoregression.
    """
    response = y[:, None] if y.ndim == 1 else y
    return run_glm(
        response, design.to_numpy(), noise_model=noise_model, **run_glm_kwargs
    )


class TestConstructor:
    def test_defaults(self):
        model = Glm()
        assert model.noise_model == "ols"
        assert model.bins == 100
        assert model.n_jobs == 1
        assert model.random_state is None
        assert model.is_fitted_ is False

    def test_every_argument_is_keyword_only(self):
        parameters = inspect.signature(Glm.__init__).parameters
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

    @pytest.mark.parametrize("noise_model", ["ols", "ar1", "ar2", "ar12"])
    def test_accepted_noise_models(self, noise_model):
        assert Glm(noise_model=noise_model).noise_model == noise_model

    @pytest.mark.parametrize(
        "noise_model", ["ar0", "ar", "ar-1", "ar1.5", "arx", "AR1", "gls", ""]
    )
    def test_invalid_noise_model_raises_value_error(self, noise_model):
        with pytest.raises(ValueError, match="noise_model"):
            Glm(noise_model=noise_model)

    @pytest.mark.parametrize("noise_model", [1, None, ["ar1"]])
    def test_non_string_noise_model_raises_type_error(self, noise_model):
        with pytest.raises(TypeError, match="noise_model"):
            Glm(noise_model=noise_model)

    @pytest.mark.parametrize("bins", [0, -1])
    def test_non_positive_bins_raises_value_error(self, bins):
        with pytest.raises(ValueError, match="bins"):
            Glm(bins=bins)

    @pytest.mark.parametrize("bins", [1.5, "100", True, None])
    def test_non_integer_bins_raises_type_error(self, bins):
        with pytest.raises(TypeError, match="bins"):
            Glm(bins=bins)

    @pytest.mark.parametrize(
        "removed", ["t_r", "smoothing_fwhm", "mask", "progress_bar", "verbose"]
    )
    def test_removed_arguments_raise_type_error(self, removed):
        with pytest.raises(TypeError):
            Glm(**{removed: 1})


class TestFitValidation:
    def test_fit_returns_self(self, glm_design, glm_targets):
        model = Glm()
        assert model.fit(glm_design, glm_targets) is model

    @pytest.mark.parametrize("frame", ["numpy", "pandas", "polars"])
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
            Glm().fit(X, glm_targets)

    def test_sample_count_mismatch_raises_value_error(self, glm_design):
        with pytest.raises(ValueError, match="sample"):
            Glm().fit(glm_design, np.zeros((glm_design.shape[0] + 1, 2)))

    def test_three_dimensional_y_raises_value_error(self, glm_design):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            Glm().fit(glm_design, np.zeros((glm_design.shape[0], 2, 2)))


class TestFittedState:
    def test_ols_coefficients_match_lstsq(self, glm_design, glm_targets, fitted_glm):
        expected = np.linalg.lstsq(glm_design.to_numpy(), glm_targets, rcond=None)[0]
        np.testing.assert_allclose(fitted_glm.coef_, expected, atol=1e-10)

    def test_no_intercept_is_ever_added(self, glm_design, glm_targets):
        without_intercept = glm_design[["condition_a", "condition_b"]]
        model = Glm().fit(without_intercept, glm_targets)
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
        model = Glm().fit(glm_design, glm_targets[:, 0])
        assert model.coef_.shape == (n_features,)
        assert model.predicted_.shape == (n_samples,)
        assert model.residuals_.shape == (n_samples,)
        assert isinstance(model.r2_, float)
        assert model.n_targets_ == 1

    def test_single_column_target_keeps_its_axis(self, glm_design, glm_targets):
        n_samples, n_features = glm_design.shape
        model = Glm().fit(glm_design, glm_targets[:, :1])
        assert model.coef_.shape == (n_features, 1)
        assert model.predicted_.shape == (n_samples, 1)
        assert model.residuals_.shape == (n_samples, 1)
        assert model.r2_.shape == (1,)
        assert model.n_targets_ == 1

    @pytest.mark.parametrize("targets", ["one_dimensional", "two_dimensional"])
    def test_coefficients_do_not_alias_the_retained_state(
        self, glm_design, glm_targets, targets
    ):
        """Mutating the public `coef_` must not reach later contrasts."""
        y = glm_targets[:, 0] if targets == "one_dimensional" else glm_targets
        model = Glm().fit(glm_design, y)
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
        model = Glm(noise_model="ar1").fit(glm_design, ar_targets)
        np.testing.assert_allclose(
            model.predicted_, glm_design.to_numpy() @ model.coef_
        )
        np.testing.assert_allclose(model.residuals_, ar_targets - model.predicted_)

    @pytest.mark.parametrize("noise_model", ["ols", "ar1"])
    def test_r2_is_copied_from_nilearn(self, glm_design, ar_targets, noise_model):
        model = Glm(noise_model=noise_model).fit(glm_design, ar_targets)
        labels, results = reference_fit(glm_design, ar_targets, noise_model=noise_model)
        expected = np.zeros(ar_targets.shape[1])
        for label, result in results.items():
            expected[labels == label] = result.r_square
        np.testing.assert_array_equal(model.r2_, expected)

    def test_autoregressive_coefficients_match_nilearn(self, glm_design, ar_targets):
        model = Glm(noise_model="ar1").fit(glm_design, ar_targets)
        labels, results = reference_fit(glm_design, ar_targets, noise_model="ar1")
        expected = np.zeros_like(model.coef_)
        for label, result in results.items():
            expected[:, labels == label] = result.theta
        np.testing.assert_array_equal(model.coef_, expected)

    def test_higher_order_autoregressive_fit_is_reproducible(
        self, glm_design, ar_targets
    ):
        first = Glm(noise_model="ar2", bins=3, random_state=0).fit(
            glm_design, ar_targets
        )
        second = Glm(noise_model="ar2", bins=3, random_state=0).fit(
            glm_design, ar_targets
        )
        np.testing.assert_array_equal(first.coef_, second.coef_)

        labels, results = reference_fit(
            glm_design, ar_targets, noise_model="ar2", bins=3, random_state=0
        )
        expected = np.zeros_like(first.coef_)
        for label, result in results.items():
            expected[:, labels == label] = result.theta
        np.testing.assert_array_equal(first.coef_, expected)


class TestFitStatePrivacy:
    def test_state_is_private(self, fitted_glm):
        assert not hasattr(fitted_glm, "fit_state")
        assert not hasattr(nltools.models, "GlmFitState")

    def test_state_is_frozen(self, fitted_glm):
        state = fitted_glm._fit_state
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.coefficients = np.zeros_like(state.coefficients)

    def test_state_preserves_nilearn_dtypes(self, glm_design, glm_targets, fitted_glm):
        _, results = reference_fit(glm_design, glm_targets)
        reference = next(iter(results.values()))
        state = fitted_glm._fit_state
        assert state.coefficients.dtype == reference.theta.dtype
        assert state.dispersion.dtype == np.asarray(reference.dispersion).dtype
        assert next(iter(state.covariances.values())).dtype == reference.cov.dtype

    def test_regression_results_are_discarded(self, fitted_glm):
        from nilearn.glm.regression import RegressionResults

        def holds_results(value, depth=0):
            if isinstance(value, RegressionResults):
                return True
            if depth > 3:
                return False
            if dataclasses.is_dataclass(value):
                return holds_results(vars(value), depth + 1)
            if isinstance(value, dict):
                return any(holds_results(item, depth + 1) for item in value.values())
            if isinstance(value, (list, tuple)):
                return any(holds_results(item, depth + 1) for item in value)
            return False

        assert not holds_results(vars(fitted_glm))


class TestRemovedSurface:
    @pytest.mark.parametrize(
        "removed",
        [
            "report",
            "score",
            "glm_",
            "residuals",
            "design_matrices_",
            "compute_contrast",
        ],
    )
    def test_removed_members_are_gone(self, removed, fitted_glm):
        assert not hasattr(Glm, removed)
        assert not hasattr(fitted_glm, removed)


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

    def test_mapping_returns_the_same_keys(self, fitted_glm):
        results = fitted_glm.compute_contrasts(
            {"a": "condition_a", "difference": [1, -1, 0]}
        )
        assert list(results) == ["a", "difference"]
        np.testing.assert_allclose(results["a"], fitted_glm.coef_[0])

    def test_before_fit_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="fit"):
            Glm().compute_contrasts("condition_a")

    @pytest.mark.parametrize("inference", [1, None, "true"])
    def test_inference_must_be_boolean(self, fitted_glm, inference):
        with pytest.raises(TypeError, match="inference must be a bool"):
            fitted_glm.compute_contrasts("condition_a", inference=inference)

    @pytest.mark.parametrize("inference", [np.True_, np.False_])
    def test_numpy_booleans_are_accepted_as_inference(self, fitted_glm, inference):
        result = fitted_glm.compute_contrasts("condition_a", inference=inference)
        assert isinstance(result, ContrastResult) is bool(inference)

    def test_unknown_column_lists_available_names(self, fitted_glm):
        with pytest.raises(ValueError) as excinfo:
            fitted_glm.compute_contrasts("condition_z")
        message = str(excinfo.value)
        assert "condition_a" in message and "intercept" in message

    def test_invalid_expression_raises_value_error(self, fitted_glm):
        with pytest.raises(ValueError):
            fitted_glm.compute_contrasts("condition_a - -")

    @pytest.mark.parametrize(
        "contrast",
        [
            None,
            True,
            ["condition_a", "condition_b"],
            [1 + 2j, 0, 0],
            [True, False, True],
        ],
    )
    def test_nonnumeric_boolean_and_complex_raise_type_error(
        self, fitted_glm, contrast
    ):
        with pytest.raises(TypeError):
            fitted_glm.compute_contrasts(contrast)

    @pytest.mark.parametrize(
        "contrast",
        [
            [],
            [1, -1],
            [1, -1, 0, 0],
            [0, 0, 0],
            [np.nan, 1, 0],
            [np.inf, 1, 0],
            [[1, -1, 0], [0, 1, -1]],
        ],
    )
    def test_empty_wrong_size_all_zero_nonfinite_and_2d_raise_value_error(
        self, fitted_glm, contrast
    ):
        with pytest.raises(ValueError):
            fitted_glm.compute_contrasts(contrast)

    def test_all_zero_expression_raises_value_error(self, fitted_glm):
        with pytest.raises(ValueError):
            fitted_glm.compute_contrasts("condition_a - condition_a")

    def test_non_string_mapping_key_raises_type_error(self, fitted_glm):
        with pytest.raises(TypeError):
            fitted_glm.compute_contrasts({0: "condition_a"})

    def test_empty_mapping_raises_value_error(self, fitted_glm):
        with pytest.raises(ValueError):
            fitted_glm.compute_contrasts({})


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

    def test_matches_nilearn_for_autoregressive_fits(self, glm_design, ar_targets):
        model = Glm(noise_model="ar1").fit(glm_design, ar_targets)
        labels, results = reference_fit(glm_design, ar_targets, noise_model="ar1")
        reference = nilearn_compute_contrast(
            labels, results, np.array([1.0, -1.0, 0.0])
        )

        result = model.compute_contrasts([1, -1, 0], inference=True)
        np.testing.assert_allclose(result.effect, reference.effect_size().ravel())
        np.testing.assert_allclose(result.variance, reference.effect_variance().ravel())
        np.testing.assert_allclose(result.statistic, reference.stat().ravel())
        np.testing.assert_allclose(result.p_value, reference.p_value().ravel())

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

    def test_effect_equals_the_effect_only_call(self, fitted_glm):
        effect = fitted_glm.compute_contrasts("condition_a - condition_b")
        result = fitted_glm.compute_contrasts(
            "condition_a - condition_b", inference=True
        )
        np.testing.assert_array_equal(result.effect, effect)

    def test_one_dimensional_target_returns_floats(self, glm_design, glm_targets):
        model = Glm().fit(glm_design, glm_targets[:, 0])
        assert isinstance(model.compute_contrasts("condition_a"), float)
        result = model.compute_contrasts("condition_a", inference=True)
        for field in (
            "effect",
            "variance",
            "standard_error",
            "statistic",
            "z_score",
            "p_value",
        ):
            assert isinstance(getattr(result, field), float)
        assert isinstance(result.degrees_of_freedom, float)

    def test_single_column_target_keeps_the_target_axis(self, glm_design, glm_targets):
        model = Glm().fit(glm_design, glm_targets[:, :1])
        assert model.compute_contrasts("condition_a").shape == (1,)
        result = model.compute_contrasts("condition_a", inference=True)
        assert result.effect.shape == (1,)
        assert result.p_value.shape == (1,)

    def test_multi_target_shapes(self, fitted_glm):
        assert fitted_glm.compute_contrasts("condition_a").shape == (3,)
        result = fitted_glm.compute_contrasts("condition_a", inference=True)
        assert result.statistic.shape == (3,)

    def test_mapping_returns_one_result_per_key(self, fitted_glm):
        results = fitted_glm.compute_contrasts(
            {"a": "condition_a", "b": "condition_b"}, inference=True
        )
        assert list(results) == ["a", "b"]
        assert all(isinstance(value, ContrastResult) for value in results.values())

    def test_results_own_their_arrays(self, fitted_glm):
        vector = np.array([1.0, -1.0, 0.0])
        coefficients_before = fitted_glm.coef_.copy()
        result = fitted_glm.compute_contrasts(vector, inference=True)
        other = fitted_glm.compute_contrasts(vector, inference=True)

        for field in ("effect", "variance", "standard_error", "statistic"):
            array = getattr(result, field)
            assert array.base is None
            assert array is not getattr(other, field)

        result.effect[0] = 12345.0
        vector[0] = 99.0
        np.testing.assert_array_equal(fitted_glm.coef_, coefficients_before)
        np.testing.assert_array_equal(
            fitted_glm._fit_state.coefficients, coefficients_before
        )
        np.testing.assert_allclose(
            other.effect, coefficients_before[0] - coefficients_before[1]
        )


class TestPredict:
    def test_returns_design_at_coefficients(self, glm_design, fitted_glm):
        np.testing.assert_allclose(
            fitted_glm.predict(glm_design), glm_design.to_numpy() @ fitted_glm.coef_
        )

    def test_new_design_prediction_equals_design_at_coef(self, glm_design, fitted_glm):
        """F182: a genuinely new-shaped design predicts as `X @ coef_`.

        The original finding was a `predict(X)` that documented a new-design
        path and then raised.
        """
        rng = np.random.RandomState(11)
        n_samples = 8
        new_design = DesignMatrix(
            {
                "condition_a": rng.randn(n_samples),
                "condition_b": rng.randn(n_samples),
                "intercept": np.ones(n_samples),
            },
            sampling_freq=glm_design.sampling_freq,
        )

        predictions = fitted_glm.predict(new_design)

        assert predictions.shape == (n_samples, fitted_glm.n_targets_)
        np.testing.assert_allclose(
            predictions, new_design.to_numpy() @ fitted_glm.coef_
        )

    def test_reorders_columns_to_the_fitted_order(self, glm_design, fitted_glm):
        shuffled = glm_design[["intercept", "condition_b", "condition_a"]]
        np.testing.assert_allclose(fitted_glm.predict(shuffled), fitted_glm.predicted_)

    def test_one_dimensional_fit_predicts_one_dimensional(
        self, glm_design, glm_targets
    ):
        model = Glm().fit(glm_design, glm_targets[:, 0])
        assert model.predict(glm_design).shape == (glm_design.shape[0],)

    def test_missing_column_raises_value_error(self, glm_design, fitted_glm):
        with pytest.raises(ValueError, match="condition_b"):
            fitted_glm.predict(glm_design[["condition_a", "intercept"]])

    def test_additional_column_raises_value_error(self, glm_design, fitted_glm):
        extra = glm_design.copy()
        extra["condition_c"] = 1.0
        with pytest.raises(ValueError, match="condition_c"):
            fitted_glm.predict(extra)

    @pytest.mark.parametrize("frame", ["numpy", "pandas"])
    def test_non_design_matrix_raises_type_error(self, glm_design, fitted_glm, frame):
        import pandas as pd

        array = glm_design.to_numpy()
        X = {"numpy": array, "pandas": pd.DataFrame(array, columns=glm_design.columns)}[
            frame
        ]
        with pytest.raises(TypeError, match="DesignMatrix"):
            fitted_glm.predict(X)

    def test_before_fit_raises(self, glm_design):
        with pytest.raises(ValueError, match="not fitted"):
            Glm().predict(glm_design)
