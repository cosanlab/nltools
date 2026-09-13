"""Tests for the decoding functional core — whitelist and coefficient back-projection."""

import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC

from nltools.algorithms.decoding import (
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


@pytest.mark.parametrize(
    "selector",
    [pytest.param(SelectKBest(f_classif, k=3), id="select_k_best")],
)
class TestFeatureSelectors:
    def test_inserts_exact_zeros_at_unselected_positions(self, selector, binary_data):
        X, y = binary_data
        pipe = make_pipeline(selector, svc()).fit(X, y)

        maps = _back_project_weight_maps(pipe, N_FEATURES)

        support = pipe[0].get_support()
        assert np.all(maps[0][~support] == 0.0)
        np.testing.assert_allclose(maps[0][support], pipe[-1].coef_.ravel())


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


# ---------------------------------------------------------------------------
# OneVsRestClassifier
# ---------------------------------------------------------------------------


class TestOneVsRest:
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


# ---------------------------------------------------------------------------
# Rejection: unsupported transformers, non-linear estimators, width mismatches
# ---------------------------------------------------------------------------


class TestRejection:
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


# ---------------------------------------------------------------------------
# The structural pre-fit check
# ---------------------------------------------------------------------------


class TestValidateDecodingPipeline:
    def test_an_unfitted_unsupported_transformer_raises_before_fitting(self):
        pipe = make_pipeline(Normalizer(), svc())

        with pytest.raises(ValueError, match="Normalizer"):
            _validate_decoding_pipeline(pipe)
