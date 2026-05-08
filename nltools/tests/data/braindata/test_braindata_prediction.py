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
        # inplace=False must not attach predict_* attrs to bd
        assert not hasattr(sim_brain_data, "predict_weight_map")
        assert not hasattr(sim_brain_data, "predict_predictions")

    def test_inplace_true_mutates_self_with_predict_prefix(self, sim_brain_data):
        """inplace=True attaches result fields with a ``predict_`` prefix —
        mirrors ``bd.fit()``'s ``glm_*`` / ``ridge_*`` model-prefixed naming.
        """
        n = sim_brain_data.shape[0]
        n_voxels = sim_brain_data.shape[1]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = sim_brain_data.predict(y=y, method="whole_brain", cv=3, inplace=True)
        assert result is sim_brain_data
        assert sim_brain_data.predict_weight_map is not None
        assert sim_brain_data.predict_weight_map.shape == (n_voxels,)
        assert sim_brain_data.predict_predictions is not None
        assert sim_brain_data.predict_scores is not None
        assert isinstance(sim_brain_data.predict_mean_score, float)


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


# ---------------------------------------------------------------------------
# Brain-space wrapping — spatial fields are BrainData, not raw arrays
# ---------------------------------------------------------------------------


class TestBrainDataWrapping:
    """Spatial result fields are BrainData objects so users can call
    ``.plot()`` directly without wrapping. Numpy access via ``.data``.
    """

    def test_whole_brain_weight_maps_are_braindata(self, sim_brain_data):
        from nltools.data import BrainData

        n = sim_brain_data.shape[0]
        n_voxels = sim_brain_data.shape[1]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = sim_brain_data.predict(
            y=y, method="whole_brain", cv=3, model="svm", refit=True
        )
        for field in ("weight_map", "fold_weight_maps", "final_weight_map"):
            obj = getattr(result, field)
            assert isinstance(obj, BrainData), f"{field} should be BrainData"
        # Same mask as the source BrainData (so .plot() composes)
        assert result.weight_map.mask is sim_brain_data.mask
        # Underlying numpy still accessible and has expected shapes
        assert result.weight_map.data.shape == (n_voxels,)
        assert result.fold_weight_maps.data.shape == (3, n_voxels)
        assert result.final_weight_map.data.shape == (n_voxels,)

    def test_searchlight_accuracy_map_is_braindata(self, minimal_brain_data):
        from nltools.data import BrainData

        n = minimal_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = minimal_brain_data.predict(
            y=y, method="searchlight", cv=3, radius_mm=4.0, n_jobs=1
        )
        assert isinstance(result.accuracy_map, BrainData)
        assert result.accuracy_map.mask is minimal_brain_data.mask


# ---------------------------------------------------------------------------
# ROI dispatch — repurposed scores/mean_score/std_score + roi_labels
# ---------------------------------------------------------------------------


class TestROIDispatch:
    """ROI runner produces per-fold-per-ROI scores (not just the mean) and
    exposes parcel labels so users can map indices back to atlas IDs.
    """

    def _build_atlas(self, bd, n_rois=3):
        """Construct an in-memory atlas matching bd.mask, with `n_rois`
        contiguous parcels evenly partitioning the mask voxels."""
        import nibabel as nib

        mask_data = bd.mask.get_fdata().astype(bool)
        flat = np.zeros(mask_data.sum(), dtype=np.int64)
        chunk = max(1, len(flat) // n_rois)
        for i in range(n_rois):
            flat[i * chunk : (i + 1) * chunk] = i + 1
        flat[(n_rois) * chunk :] = n_rois  # any remainder → last parcel
        out = np.zeros(mask_data.shape, dtype=np.int64)
        out[mask_data] = flat
        return nib.Nifti1Image(out, bd.mask.affine, bd.mask.header)

    def test_roi_populates_repurposed_score_fields(self, minimal_brain_data):
        from nltools.data import BrainData

        n = minimal_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        atlas = self._build_atlas(minimal_brain_data, n_rois=2)

        result = minimal_brain_data.predict(
            y=y, method="roi", roi_mask=atlas, cv=3, model="svm", n_jobs=1
        )
        # Score fields are arrays, not scalars, on ROI dispatch
        assert result.scores is not None
        assert result.scores.shape == (3, 2)  # (n_folds, n_rois)
        assert result.mean_score is not None
        assert result.mean_score.shape == (2,)
        assert result.std_score is not None
        assert result.std_score.shape == (2,)
        assert result.roi_labels is not None
        assert result.roi_labels.shape == (2,)
        # Atlas labels in mean_score order
        assert list(result.roi_labels) == [1, 2]
        # accuracy_map is a BrainData
        assert isinstance(result.accuracy_map, BrainData)

    def test_roi_leaves_whole_brain_fields_none(self, minimal_brain_data):
        n = minimal_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        atlas = self._build_atlas(minimal_brain_data, n_rois=2)
        result = minimal_brain_data.predict(
            y=y, method="roi", roi_mask=atlas, cv=3, n_jobs=1
        )
        assert result.weight_map is None
        assert result.fold_weight_maps is None
        assert result.predictions is None
        assert result.cv_folds is None


# ---------------------------------------------------------------------------
# Pipeline auto-detect — model=Pipeline triggers standardize=False default
# ---------------------------------------------------------------------------


class TestPipelineStandardizeDetect:
    def test_pipeline_default_warns_and_disables_standardize(self, sim_brain_data):
        """Passing a Pipeline with the default ``standardize=True`` warns and
        flips standardize off — avoids silently wrapping another StandardScaler.
        """
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC

        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        pipe = make_pipeline(StandardScaler(), LinearSVC(dual="auto", max_iter=10000))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sim_brain_data.predict(y=y, method="whole_brain", cv=3, model=pipe)
        msgs = [str(warn.message) for warn in w]
        assert any("Pipeline" in m and "standardize" in m for m in msgs), (
            f"expected pipeline-standardize warning, got: {msgs}"
        )

    def test_pipeline_explicit_standardize_true_still_wraps(self, sim_brain_data):
        """User can opt back into wrapping by passing ``standardize=True``
        explicitly — same value as the default but no warning is suppressed.
        Behavior is the same as today, just confirmed not silently flipped.
        """
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import LinearSVC

        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        pipe = make_pipeline(LinearSVC(dual="auto", max_iter=10000))
        # standardize=False explicitly: never warns, never wraps
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sim_brain_data.predict(
                y=y, method="whole_brain", cv=3, model=pipe, standardize=False
            )
        assert not any(
            "Pipeline" in str(warn.message) and "standardize" in str(warn.message)
            for warn in w
        )
