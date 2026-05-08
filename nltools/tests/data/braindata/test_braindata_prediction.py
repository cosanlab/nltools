"""Tests for BrainData.predict() — kwargs API returning Predict dataclass."""

import warnings

import numpy as np
import pytest

from nltools.data.fitresults import Predict


# ---------------------------------------------------------------------------
# Mode dispatch / argument validation
# ---------------------------------------------------------------------------


class TestPredictDispatch:
    def test_cannot_specify_both_x_and_y(self, sim_brain_data):
        X = np.random.randn(len(sim_brain_data), 5)
        y = np.array([0, 1] * (len(sim_brain_data) // 2))

        with pytest.raises(ValueError, match="Cannot specify both X and y"):
            sim_brain_data.predict(X=X, y=y)

    def test_invalid_method(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        with pytest.raises(ValueError, match="Invalid method"):
            sim_brain_data.predict(y=y, method="bogus")

    def test_unknown_model_shortcut(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        with pytest.raises(ValueError, match="Unknown model"):
            sim_brain_data.predict(y=y, model="bogus")

    def test_invalid_reduce(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        with pytest.raises(ValueError, match="Unknown reduce"):
            sim_brain_data.predict(y=y, reduce="ica", n_components=2)


# ---------------------------------------------------------------------------
# Whole-brain MVPA — returns Predict with weight maps
# ---------------------------------------------------------------------------


class TestWholeBrain:
    def test_returns_predict_dataclass(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        result = sim_brain_data.predict(y=y, method="whole_brain", cv=3)

        assert isinstance(result, Predict)

    def test_classification_populates_expected_fields(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        n_voxels = sim_brain_data.shape[1]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        result = sim_brain_data.predict(y=y, method="whole_brain", cv=3, model="svm")

        assert result.predictions is not None
        assert result.predictions.shape == (n,)
        assert result.scores is not None
        assert result.scores.shape == (3,)
        assert isinstance(result.mean_score, float)
        assert isinstance(result.std_score, float)
        assert result.cv_folds is not None
        assert result.cv_folds.shape == (n,)
        assert result.weight_map is not None
        assert result.weight_map.shape == (n_voxels,)
        assert result.fold_weight_maps is not None
        assert result.fold_weight_maps.shape == (3, n_voxels)
        # whole_brain doesn't populate accuracy_map
        assert result.accuracy_map is None
        # refit=False by default
        assert result.final_estimator is None
        assert result.final_weight_map is None

    def test_regression_with_ridge(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        n_voxels = sim_brain_data.shape[1]
        y = np.random.RandomState(0).randn(n)

        result = sim_brain_data.predict(y=y, method="whole_brain", cv=3, model="ridge")

        assert isinstance(result, Predict)
        assert result.weight_map.shape == (n_voxels,)
        # mean_score should be a finite float (R² for regression)
        assert isinstance(result.mean_score, float)

    def test_logistic_shortcut(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        result = sim_brain_data.predict(
            y=y, method="whole_brain", cv=3, model="logistic"
        )
        assert result.weight_map is not None

    def test_ridge_classifier_shortcut(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        result = sim_brain_data.predict(
            y=y, method="whole_brain", cv=3, model="ridge_classifier"
        )
        assert result.weight_map is not None

    def test_lda_shortcut(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        result = sim_brain_data.predict(y=y, method="whole_brain", cv=3, model="lda")
        assert result.weight_map is not None

    def test_custom_sklearn_estimator(self, sim_brain_data):
        from sklearn.linear_model import LogisticRegression

        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        result = sim_brain_data.predict(
            y=y, method="whole_brain", cv=3, model=LogisticRegression(max_iter=1000)
        )
        assert isinstance(result, Predict)
        assert result.weight_map is not None

    def test_sklearn_pipeline_passthrough(self, sim_brain_data):
        """Pass a custom Pipeline as model — replaces .pipe() escape hatch."""
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC

        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        pipe = make_pipeline(
            StandardScaler(),
            SelectKBest(f_classif, k=min(2, sim_brain_data.shape[1])),
            LinearSVC(dual="auto", max_iter=10000),
        )
        result = sim_brain_data.predict(
            y=y, method="whole_brain", cv=3, model=pipe, standardize=False
        )
        assert isinstance(result, Predict)
        # weight_map may be None — SelectKBest masks the feature space; we
        # don't try to back-project here. Just confirm no crash + scores.
        assert result.scores is not None

    def test_non_linear_emits_warning_no_weight_map(self, sim_brain_data):
        from sklearn.svm import SVC

        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = sim_brain_data.predict(
                y=y, method="whole_brain", cv=3, model=SVC(kernel="rbf")
            )
        assert result.weight_map is None
        assert any("weight_map" in str(warn.message) for warn in w)


# ---------------------------------------------------------------------------
# standardize / reduce
# ---------------------------------------------------------------------------


class TestPreprocessing:
    def test_standardize_true_runs(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = sim_brain_data.predict(
            y=y, method="whole_brain", cv=3, standardize=True
        )
        assert result.weight_map is not None

    def test_standardize_false_runs(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = sim_brain_data.predict(
            y=y, method="whole_brain", cv=3, standardize=False
        )
        assert result.weight_map is not None

    def test_reduce_pca_back_projects_weight_map(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        n_voxels = sim_brain_data.shape[1]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        n_comp = min(3, n_voxels - 1)
        result = sim_brain_data.predict(
            y=y, method="whole_brain", cv=3, reduce="pca", n_components=n_comp
        )
        # weight_map must be in voxel space (back-projected through PCA), not
        # PC space
        assert result.weight_map.shape == (n_voxels,)
        assert result.fold_weight_maps.shape == (3, n_voxels)


# ---------------------------------------------------------------------------
# scoring='auto'
# ---------------------------------------------------------------------------


class TestScoringAuto:
    def test_auto_classifier_uses_accuracy(self, sim_brain_data):
        """Classifier + scoring='auto' → 'accuracy' → scores in [0, 1]."""
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = sim_brain_data.predict(
            y=y, method="whole_brain", cv=3, model="svm", scoring="auto"
        )
        assert np.all((result.scores >= 0) & (result.scores <= 1))

    def test_auto_regressor_uses_r2(self, sim_brain_data):
        """Regressor + scoring='auto' → 'r2' (can be negative)."""
        n = sim_brain_data.shape[0]
        y = np.random.RandomState(0).randn(n)
        result = sim_brain_data.predict(
            y=y, method="whole_brain", cv=3, model="ridge", scoring="auto"
        )
        # R² scores can be any float (including negative for bad fits) — just
        # confirm we got per-fold floats back
        assert result.scores is not None
        assert result.scores.dtype.kind == "f"

    def test_explicit_scoring_passthrough(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = sim_brain_data.predict(
            y=y,
            method="whole_brain",
            cv=3,
            model="svm",
            scoring="balanced_accuracy",
        )
        assert result.scores is not None


# ---------------------------------------------------------------------------
# inplace
# ---------------------------------------------------------------------------


class TestInplace:
    def test_inplace_false_returns_predict(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = sim_brain_data.predict(y=y, method="whole_brain", cv=3)
        assert isinstance(result, Predict)
        assert (
            not hasattr(sim_brain_data, "weight_map")
            or sim_brain_data.weight_map is None
            or sim_brain_data.weight_map is not result.weight_map
        )

    def test_inplace_true_mutates_self(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        n_voxels = sim_brain_data.shape[1]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = sim_brain_data.predict(y=y, method="whole_brain", cv=3, inplace=True)
        assert result is sim_brain_data
        assert sim_brain_data.weight_map is not None
        assert sim_brain_data.weight_map.shape == (n_voxels,)
        assert sim_brain_data.predictions is not None
        assert sim_brain_data.scores is not None
        assert isinstance(sim_brain_data.mean_score, float)


# ---------------------------------------------------------------------------
# Searchlight / ROI — accuracy_map populated, weight_map None
# ---------------------------------------------------------------------------


class TestSearchlight:
    def test_returns_predict_with_accuracy_map(self, minimal_brain_data):
        """Searchlight on a minimal 5-voxel fixture — fast."""
        n = minimal_brain_data.shape[0]
        n_voxels = minimal_brain_data.shape[1]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        result = minimal_brain_data.predict(
            y=y, method="searchlight", cv=3, radius_mm=4.0, n_jobs=1
        )
        assert isinstance(result, Predict)
        assert result.accuracy_map is not None
        assert result.accuracy_map.shape == (n_voxels,)
        # weight_map intentionally not provided for searchlight
        assert result.weight_map is None
        assert result.fold_weight_maps is None


class TestRefit:
    def test_refit_false_leaves_final_fields_none(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = sim_brain_data.predict(y=y, method="whole_brain", cv=3, refit=False)
        assert result.final_estimator is None
        assert result.final_weight_map is None

    def test_refit_true_populates_final_estimator_and_weight_map(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        n_voxels = sim_brain_data.shape[1]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = sim_brain_data.predict(
            y=y, method="whole_brain", cv=3, model="svm", refit=True
        )
        assert result.final_estimator is not None
        # final_estimator must be fitted (has .coef_ for linear models)
        assert hasattr(result.final_estimator, "named_steps") or hasattr(
            result.final_estimator, "coef_"
        )
        assert result.final_weight_map is not None
        assert result.final_weight_map.shape == (n_voxels,)

    def test_refit_with_pca_back_projects_final_weight_map(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        n_voxels = sim_brain_data.shape[1]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        n_comp = min(3, n_voxels - 1)
        result = sim_brain_data.predict(
            y=y,
            method="whole_brain",
            cv=3,
            reduce="pca",
            n_components=n_comp,
            refit=True,
        )
        assert result.final_weight_map.shape == (n_voxels,)


class TestPredictMulti:
    def test_predict_multi_deprecated(self, minimal_brain_data):
        """Deprecated .predict_multi() raises NotImplementedError pointing
        at the future Model class (per migration guide)."""
        with pytest.raises(
            NotImplementedError, match="predict_multi.*deprecated.*Model class"
        ):
            minimal_brain_data.predict_multi(algorithm="svm")
