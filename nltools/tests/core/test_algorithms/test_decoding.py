"""Tests for the decoding functional core — whitelist and coefficient back-projection."""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    VarianceThreshold,
    f_classif,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import SVC, LinearSVC

from nltools.algorithms.decoding import (
    SUPPORTED_TRANSFORMERS,
    _BackProjectionError,
    _back_project_weight_maps,
    _validate_decoding_pipeline,
    _whitening_scale,
)

N_SAMPLES = 40
N_FEATURES = 6


@pytest.fixture
def binary_data():
    """A small, well-conditioned two-class problem."""
    rng = np.random.default_rng(0)
    y = np.tile([0, 1], N_SAMPLES // 2)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES)) + y[:, None]
    # Give the columns different scales so a StandardScaler actually does work.
    X *= np.arange(1, N_FEATURES + 1)
    return X, y


@pytest.fixture
def multiclass_data():
    """A small three-class problem."""
    rng = np.random.default_rng(1)
    y = np.tile([0, 1, 2], N_SAMPLES // 3 + 1)[:N_SAMPLES]
    X = rng.standard_normal((N_SAMPLES, N_FEATURES)) + y[:, None]
    return X, y


@pytest.fixture
def regression_data():
    """A small continuous-target problem."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    y = X @ np.arange(1.0, N_FEATURES + 1) + rng.standard_normal(N_SAMPLES) * 0.1
    return X, y


def svc():
    """The shortcut classifier, with the hyperparameters the shortcut table uses."""
    return LinearSVC(dual="auto", max_iter=10000)


# ---------------------------------------------------------------------------
# Shape and orientation of the extracted coefficients
# ---------------------------------------------------------------------------


class TestCoefficientShapes:
    def test_regression_gives_one_map(self, regression_data):
        X, y = regression_data
        fitted = Ridge().fit(X, y)

        maps = _back_project_weight_maps(fitted, N_FEATURES)

        assert maps.shape == (1, N_FEATURES)
        np.testing.assert_allclose(maps[0], fitted.coef_)

    def test_binary_classifier_gives_one_signed_map(self, binary_data):
        X, y = binary_data
        fitted = svc().fit(X, y)

        maps = _back_project_weight_maps(fitted, N_FEATURES)

        assert maps.shape == (1, N_FEATURES)
        # The sign convention is the estimator's own: classes_[1] vs classes_[0],
        # never negated on the way out.
        np.testing.assert_allclose(maps[0], fitted.coef_.ravel())

    def test_native_multiclass_gives_one_map_per_class(self, multiclass_data):
        X, y = multiclass_data
        fitted = LogisticRegression(max_iter=1000).fit(X, y)

        maps = _back_project_weight_maps(fitted, N_FEATURES)

        assert maps.shape == (3, N_FEATURES)
        np.testing.assert_allclose(maps, fitted.coef_)

    def test_coefficients_are_never_averaged_across_classes(self, multiclass_data):
        X, y = multiclass_data
        fitted = LogisticRegression(max_iter=1000).fit(X, y)

        maps = _back_project_weight_maps(fitted, N_FEATURES)

        assert maps.shape[0] == len(fitted.classes_)
        assert not np.allclose(maps[0], maps.mean(axis=0))


# ---------------------------------------------------------------------------
# One back-projection case per whitelisted transformer
# ---------------------------------------------------------------------------


class TestStandardScaler:
    def test_with_std_divides_by_scale(self, binary_data):
        X, y = binary_data
        pipe = make_pipeline(StandardScaler(), svc()).fit(X, y)

        maps = _back_project_weight_maps(pipe, N_FEATURES)

        scaler, estimator = pipe[0], pipe[-1]
        np.testing.assert_allclose(maps[0], estimator.coef_.ravel() / scaler.scale_)

    def test_without_std_leaves_weights_unchanged(self, binary_data):
        X, y = binary_data
        pipe = make_pipeline(StandardScaler(with_std=False), svc()).fit(X, y)

        maps = _back_project_weight_maps(pipe, N_FEATURES)

        np.testing.assert_allclose(maps[0], pipe[-1].coef_.ravel())

    def test_centering_alone_does_not_change_the_slope_map(self, regression_data):
        """Centering shifts the intercept, not the coefficients."""
        X, y = regression_data
        centered = make_pipeline(StandardScaler(with_std=False), LinearRegression())
        raw = LinearRegression().fit(X, y)

        maps = _back_project_weight_maps(centered.fit(X, y), N_FEATURES)

        np.testing.assert_allclose(maps[0], raw.coef_, rtol=1e-8)


class TestPca:
    def test_unwhitened_applies_weights_at_components(self, regression_data):
        X, y = regression_data
        pipe = make_pipeline(PCA(n_components=3), Ridge()).fit(X, y)

        maps = _back_project_weight_maps(pipe, N_FEATURES)

        pca, estimator = pipe[0], pipe[-1]
        np.testing.assert_allclose(maps[0], estimator.coef_ @ pca.components_)

    def test_whitened_divides_component_weights_by_the_component_scale(
        self, regression_data
    ):
        X, y = regression_data
        pipe = make_pipeline(PCA(n_components=3, whiten=True), Ridge()).fit(X, y)

        maps = _back_project_weight_maps(pipe, N_FEATURES)

        pca, estimator = pipe[0], pipe[-1]
        scale = np.sqrt(pca.explained_variance_)
        np.testing.assert_allclose(maps[0], (estimator.coef_ / scale) @ pca.components_)

    def test_whitened_and_unwhitened_differ(self, regression_data):
        """The whitening step is a real correction, not a no-op."""
        X, y = regression_data
        plain = make_pipeline(PCA(n_components=3), Ridge()).fit(X, y)
        white = make_pipeline(PCA(n_components=3, whiten=True), Ridge()).fit(X, y)

        assert not np.allclose(
            _back_project_weight_maps(plain, N_FEATURES),
            _back_project_weight_maps(white, N_FEATURES),
        )

    def test_multiclass_back_projects_every_class_row(self, multiclass_data):
        X, y = multiclass_data
        pipe = make_pipeline(PCA(n_components=3), LogisticRegression(max_iter=1000))
        pipe.fit(X, y)

        maps = _back_project_weight_maps(pipe, N_FEATURES)

        assert maps.shape == (3, N_FEATURES)
        np.testing.assert_allclose(maps, pipe[-1].coef_ @ pipe[0].components_)


class TestWhiteningScale:
    def test_zero_variance_is_floored_at_epsilon(self):
        variance = np.array([4.0, 0.0])

        scale = _whitening_scale(variance)

        eps = np.finfo(variance.dtype).eps
        np.testing.assert_allclose(scale, [2.0, eps])

    def test_values_below_epsilon_are_floored(self):
        variance = np.array([1e-40, 1.0])

        scale = _whitening_scale(variance)

        eps = np.finfo(variance.dtype).eps
        assert scale[0] == eps
        assert scale[1] == 1.0


@pytest.mark.parametrize(
    "selector",
    [
        pytest.param(VarianceThreshold(threshold=0.0), id="variance_threshold"),
        pytest.param(SelectKBest(f_classif, k=3), id="select_k_best"),
        pytest.param(RFE(svc(), n_features_to_select=3), id="rfe"),
    ],
)
class TestFeatureSelectors:
    def test_expands_to_the_fitted_input_width(self, selector, binary_data):
        X, y = binary_data
        pipe = make_pipeline(selector, svc()).fit(X, y)

        maps = _back_project_weight_maps(pipe, N_FEATURES)

        assert maps.shape == (1, N_FEATURES)

    def test_inserts_exact_zeros_at_unselected_positions(self, selector, binary_data):
        X, y = binary_data
        pipe = make_pipeline(selector, svc()).fit(X, y)

        maps = _back_project_weight_maps(pipe, N_FEATURES)

        support = pipe[0].get_support()
        assert np.all(maps[0][~support] == 0.0)
        np.testing.assert_allclose(maps[0][support], pipe[-1].coef_.ravel())


class TestPassthrough:
    @pytest.mark.parametrize("step", [None, "passthrough"], ids=["none", "passthrough"])
    def test_a_passthrough_step_changes_nothing(self, step, binary_data):
        X, y = binary_data
        pipe = Pipeline([("pre", step), ("clf", svc())]).fit(X, y)

        maps = _back_project_weight_maps(pipe, N_FEATURES)

        np.testing.assert_allclose(maps[0], pipe[-1].coef_.ravel())


class TestComposition:
    def test_steps_compose_in_reverse_order(self, binary_data):
        X, y = binary_data
        pipe = make_pipeline(
            StandardScaler(),
            SelectKBest(f_classif, k=4),
            PCA(n_components=2),
            svc(),
        ).fit(X, y)

        maps = _back_project_weight_maps(pipe, N_FEATURES)

        scaler, selector, pca, estimator = pipe[0], pipe[1], pipe[2], pipe[3]
        expected = estimator.coef_ @ pca.components_  # (1, 4)
        support = selector.get_support()
        expanded = np.zeros((1, N_FEATURES))
        expanded[:, support] = expected
        expected = expanded / scaler.scale_
        np.testing.assert_allclose(maps, expected)

    def test_order_matters(self, binary_data):
        """The walk follows the fitted order: scale-then-PCA is not PCA-then-scale."""
        X, y = binary_data
        first = make_pipeline(StandardScaler(), PCA(n_components=3), svc()).fit(X, y)
        second = make_pipeline(PCA(n_components=3), StandardScaler(), svc()).fit(X, y)

        assert not np.allclose(
            _back_project_weight_maps(first, N_FEATURES),
            _back_project_weight_maps(second, N_FEATURES),
        )


# ---------------------------------------------------------------------------
# OneVsRestClassifier
# ---------------------------------------------------------------------------


class TestOneVsRest:
    def test_binary_gives_the_sole_child_row(self, binary_data):
        X, y = binary_data
        fitted = OneVsRestClassifier(svc()).fit(X, y)

        maps = _back_project_weight_maps(fitted, N_FEATURES)

        assert maps.shape == (1, N_FEATURES)
        np.testing.assert_allclose(maps[0], fitted.estimators_[0].coef_.ravel())

    def test_multiclass_stacks_child_rows_in_class_order(self, multiclass_data):
        X, y = multiclass_data
        fitted = OneVsRestClassifier(svc()).fit(X, y)

        maps = _back_project_weight_maps(fitted, N_FEATURES)

        assert maps.shape == (3, N_FEATURES)
        for i, child in enumerate(fitted.estimators_):
            np.testing.assert_allclose(maps[i], child.coef_.ravel())

    def test_shared_preprocessing_before_it_is_back_projected(self, multiclass_data):
        X, y = multiclass_data
        pipe = make_pipeline(StandardScaler(), OneVsRestClassifier(svc())).fit(X, y)

        maps = _back_project_weight_maps(pipe, N_FEATURES)

        scaler, ovr = pipe[0], pipe[-1]
        for i, child in enumerate(ovr.estimators_):
            np.testing.assert_allclose(maps[i], child.coef_.ravel() / scaler.scale_)

    def test_a_child_without_coefficients_raises(self, multiclass_data):
        X, y = multiclass_data
        fitted = OneVsRestClassifier(SVC(kernel="rbf")).fit(X, y)

        with pytest.raises(ValueError, match="coef_"):
            _back_project_weight_maps(fitted, N_FEATURES)

    def test_it_must_be_the_final_step(self):
        # Never fitted: sklearn rejects a non-transformer intermediate step, and
        # the misplacement is visible from the structure alone.
        pipe = Pipeline([("ovr", OneVsRestClassifier(svc())), ("clf", svc())])

        with pytest.raises(ValueError, match="final"):
            _back_project_weight_maps(pipe, N_FEATURES)


# ---------------------------------------------------------------------------
# Rejection: unsupported transformers, non-linear estimators, width mismatches
# ---------------------------------------------------------------------------


class TestRejection:
    def test_an_estimator_without_coefficients_raises(self, binary_data):
        X, y = binary_data
        fitted = SVC(kernel="rbf").fit(X, y)

        with pytest.raises(ValueError, match="coef_"):
            _back_project_weight_maps(fitted, N_FEATURES)

    def test_the_error_is_a_value_error(self, binary_data):
        X, y = binary_data
        fitted = SVC(kernel="rbf").fit(X, y)

        with pytest.raises(_BackProjectionError):
            _back_project_weight_maps(fitted, N_FEATURES)
        assert issubclass(_BackProjectionError, ValueError)

    def test_an_unsupported_transformer_raises_even_with_inverse_transform(
        self, binary_data
    ):
        """`Normalizer` scales rows, so no coefficient back-projection exists."""
        X, y = binary_data
        pipe = make_pipeline(Normalizer(), svc()).fit(X, y)

        with pytest.raises(ValueError, match="Normalizer"):
            _back_project_weight_maps(pipe, N_FEATURES)

    def test_the_rejection_message_lists_the_supported_steps(self, binary_data):
        X, y = binary_data
        pipe = make_pipeline(Normalizer(), svc()).fit(X, y)

        with pytest.raises(ValueError, match="StandardScaler"):
            _back_project_weight_maps(pipe, N_FEATURES)

    def test_the_final_width_must_match_the_voxel_axis(self, binary_data):
        X, y = binary_data
        pipe = make_pipeline(StandardScaler(), svc()).fit(X, y)

        with pytest.raises(ValueError, match="width"):
            _back_project_weight_maps(pipe, N_FEATURES + 1)

    def test_a_step_whose_fitted_width_does_not_line_up_raises(self, binary_data):
        """Each step validates its own fitted input and output widths."""
        X, y = binary_data
        narrow_scaler = StandardScaler().fit(X[:, :4])
        wide = make_pipeline(PCA(n_components=2), svc()).fit(X, y)
        mismatched = Pipeline(
            [("scaler", narrow_scaler), ("pca", wide[0]), ("clf", wide[-1])]
        )

        with pytest.raises(ValueError, match="width"):
            _back_project_weight_maps(mismatched, N_FEATURES)


# ---------------------------------------------------------------------------
# The structural pre-fit check
# ---------------------------------------------------------------------------


class TestValidateDecodingPipeline:
    def test_an_unfitted_whitelisted_pipeline_passes(self):
        pipe = make_pipeline(StandardScaler(), PCA(n_components=2), svc())

        _validate_decoding_pipeline(pipe)  # no raise

    def test_an_unfitted_unsupported_transformer_raises_before_fitting(self):
        pipe = make_pipeline(Normalizer(), svc())

        with pytest.raises(ValueError, match="Normalizer"):
            _validate_decoding_pipeline(pipe)

    def test_one_vs_rest_must_be_final_before_fitting(self):
        pipe = Pipeline([("ovr", OneVsRestClassifier(svc())), ("clf", svc())])

        with pytest.raises(ValueError, match="final"):
            _validate_decoding_pipeline(pipe)

    def test_a_bare_estimator_passes(self):
        _validate_decoding_pipeline(svc())  # no raise

    def test_every_supported_transformer_is_whitelisted(self):
        names = {cls.__name__ for cls in SUPPORTED_TRANSFORMERS}

        assert names == {
            "StandardScaler",
            "PCA",
            "VarianceThreshold",
            "GenericUnivariateSelect",
            "SelectPercentile",
            "SelectKBest",
            "SelectFpr",
            "SelectFdr",
            "SelectFwe",
            "SelectFromModel",
            "RFE",
            "RFECV",
            "SequentialFeatureSelector",
        }
