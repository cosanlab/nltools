"""Tests for BrainData.predict() — kwargs API returning Predict dataclass."""

import inspect
import typing
import warnings

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    KFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from nltools.data import Predict
from nltools.data.braindata.prediction import _continuous_strata, _resolve_splitter


# ---------------------------------------------------------------------------
# Mode dispatch / argument validation
# ---------------------------------------------------------------------------


class TestPredictDispatch:
    def test_cannot_specify_both_x_and_y(self, sim_brain_data):
        X = np.random.randn(len(sim_brain_data), 5)
        y = np.array([0, 1] * (len(sim_brain_data) // 2))

        with pytest.raises(ValueError, match="Cannot specify both X and y"):
            sim_brain_data.predict(X=X, y=y)

    def test_invalid_spatial_scale(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        with pytest.raises(ValueError, match="Invalid spatial_scale"):
            sim_brain_data.predict(y=y, spatial_scale="bogus")

    def test_unknown_estimator_shortcut(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        with pytest.raises(ValueError, match="Unknown estimator shortcut"):
            sim_brain_data.predict(y=y, estimator="bogus")


class TestStoredYFallback:
    """predict() decodes against the stored ``.Y`` slot when y is omitted.

    ``BrainData.Y`` is row-aligned with ``.data`` (sliced together, carried
    through h5) — labels travel with the data, so per-subject decoding
    doesn't need an external label vector.
    """

    @staticmethod
    def _labels(bd):
        n = bd.shape[0]
        return np.array([0] * (n // 2) + [1] * (n - n // 2))

    def test_y_none_decodes_stored_single_column_Y(self, sim_brain_data):
        sim_brain_data.Y = {"label": self._labels(sim_brain_data)}
        result = sim_brain_data.predict(cv=3)
        assert isinstance(result, Predict)
        assert result.predictions.shape == (sim_brain_data.shape[0],)

    def test_y_none_matches_explicit_y(self, sim_brain_data):
        y = self._labels(sim_brain_data)
        sim_brain_data.Y = {"label": y}
        implicit = sim_brain_data.predict(cv=3)
        explicit = sim_brain_data.predict(y=y, cv=3)
        np.testing.assert_array_equal(implicit.predictions, explicit.predictions)
        assert implicit.mean_score == explicit.mean_score

    def test_y_column_name_selects_from_stored_Y(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        sim_brain_data.Y = {
            "label": self._labels(sim_brain_data),
            "run": np.arange(n) % 2,
        }
        result = sim_brain_data.predict(y="label", cv=3)
        assert isinstance(result, Predict)

    def test_y_none_multiple_Y_columns_raises(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        sim_brain_data.Y = {
            "label": self._labels(sim_brain_data),
            "run": np.arange(n) % 2,
        }
        with pytest.raises(ValueError, match="column"):
            sim_brain_data.predict(cv=3)

    def test_y_column_name_missing_raises(self, sim_brain_data):
        sim_brain_data.Y = {"label": self._labels(sim_brain_data)}
        with pytest.raises(ValueError, match="nope"):
            sim_brain_data.predict(y="nope", cv=3)

    def test_y_string_without_stored_Y_raises(self, sim_brain_data):
        with pytest.raises(ValueError, match="Y"):
            sim_brain_data.predict(y="label", cv=3)

    def test_groups_column_name_selects_from_stored_Y(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        runs = np.arange(n) % 3
        sim_brain_data.Y = {"label": self._labels(sim_brain_data), "run": runs}
        result = sim_brain_data.predict(y="label", cv=LeaveOneGroupOut(), groups="run")
        # LeaveOneGroupOut over 3 runs → 3 folds.
        assert result.scores.shape == (3,)

    def test_fitted_model_wins_over_stored_Y(self, minimal_brain_data):
        """A no-argument call predicts from the fitted model, not the labels."""
        n = minimal_brain_data.shape[0]
        X = np.random.default_rng(0).standard_normal((n, 3))
        minimal_brain_data.fit(model="ridge", X=X, ridge_alpha=1.0)
        minimal_brain_data.Y = {"label": np.arange(n) % 2}

        predicted = minimal_brain_data.predict()

        np.testing.assert_allclose(
            predicted.data, minimal_brain_data.ridge_fitted_values.data
        )


# ---------------------------------------------------------------------------
# Whole-brain MVPA — returns Predict with weight maps
# ---------------------------------------------------------------------------


class TestWholeBrain:
    def test_returns_predict_dataclass(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        result = sim_brain_data.predict(y=y, spatial_scale="whole_brain", cv=3)

        assert isinstance(result, Predict)

    def test_classification_populates_expected_fields(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        n_voxels = sim_brain_data.shape[1]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        result = sim_brain_data.predict(
            y=y, spatial_scale="whole_brain", cv=3, estimator="linear_svc"
        )

        assert result.spatial_scale == "whole_brain"
        assert result.scoring is None
        np.testing.assert_array_equal(result.classes, [0, 1])
        assert result.predictions.shape == (n,)
        assert result.cv_folds.shape == (n,)
        assert result.scores.shape == (3,)
        assert isinstance(result.mean_score, float)
        assert isinstance(result.std_score, float)
        # weight_map is the all-data refit (canonical), always populated for
        # linear models — no separate refit=True opt-in.
        assert result.weight_map.shape == (n_voxels,)
        # All-data fitted estimator, available for .predict() on new data.
        assert result.estimator is not None
        # ROI and searchlight fields stay None for whole-brain decoding.
        assert result.roi_labels is None
        assert result.score_map is None

    def test_scoring_specification_is_recorded(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        result = sim_brain_data.predict(y=y, cv=3, scoring="balanced_accuracy")

        assert result.scoring == "balanced_accuracy"

    def test_regression_with_ridge(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        n_voxels = sim_brain_data.shape[1]
        y = np.random.RandomState(0).randn(n)

        result = sim_brain_data.predict(
            y=y, spatial_scale="whole_brain", cv=3, estimator="ridge"
        )

        assert isinstance(result, Predict)
        assert result.weight_map.shape == (n_voxels,)
        # mean_score should be a finite float (R² for regression)
        assert isinstance(result.mean_score, float)
        # Regression has no class labels.
        assert result.classes is None

    def test_custom_sklearn_estimator(self, sim_brain_data):
        """A bare caller estimator is fitted as given — no scaler is wrapped."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        result = sim_brain_data.predict(
            y=y,
            spatial_scale="whole_brain",
            cv=3,
            estimator=LogisticRegression(max_iter=1000),
        )
        assert isinstance(result, Predict)
        assert result.weight_map is not None
        assert not isinstance(result.estimator, Pipeline)
        assert isinstance(result.estimator, LogisticRegression)

    def test_shortcut_estimator_standardizes_within_each_fold(self, sim_brain_data):
        """A shortcut name selects a predefined pipeline that scales per fold."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        result = sim_brain_data.predict(y=y, cv=3, estimator="linear_svc")
        assert isinstance(result.estimator, Pipeline)
        assert isinstance(result.estimator[0], StandardScaler)

    def test_sklearn_pipeline_passthrough(self, sim_brain_data):
        """A caller-supplied Pipeline is used exactly as given, unwrapped."""
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
            y=y, spatial_scale="whole_brain", cv=3, estimator=pipe
        )
        assert isinstance(result, Predict)
        assert result.estimator.named_steps.keys() == pipe.named_steps.keys()
        assert result.scores is not None
        # SelectKBest is whitelisted: the map is full width, with exact zeros
        # at the voxels the selector dropped.
        assert result.weight_map.shape == (sim_brain_data.shape[1],)
        support = result.estimator.named_steps["selectkbest"].get_support()
        assert np.all(result.weight_map.data[~support] == 0.0)

    def test_multiclass_gives_one_map_per_class_never_an_average(
        self, minimal_brain_data
    ):
        """Multiclass decoding returns one map per class, in ``classes`` order."""
        n = minimal_brain_data.shape[0]
        n_voxels = minimal_brain_data.shape[1]
        y = np.array([0, 1, 2] * (n // 3) + [0] * (n % 3))

        result = minimal_brain_data.predict(y=y, cv=3)

        np.testing.assert_array_equal(result.classes, [0, 1, 2])
        assert result.weight_map.shape == (3, n_voxels)
        maps = result.weight_map.data
        assert not np.allclose(maps[0], maps.mean(axis=0))
        assert result.scores.shape == (3,)
        assert result.predictions.shape == (n,)

    def test_binary_weight_map_is_one_signed_map(self, minimal_brain_data):
        n = minimal_brain_data.shape[0]
        n_voxels = minimal_brain_data.shape[1]
        y = np.array([0, 1] * (n // 2))

        result = minimal_brain_data.predict(y=y, cv=3)

        assert result.weight_map.shape == (n_voxels,)

    def test_non_linear_estimator_raises(self, sim_brain_data):
        """An estimator with no ``coef_`` is rejected, not silently degraded.

        Inverted from the 0.6.0-dev behaviour: `weight_map=None` used to be a
        reachable success state for whole-brain decoding. Every successful
        whole-brain result now carries a map.
        """
        from sklearn.svm import SVC

        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        with pytest.raises(ValueError, match="coef_"):
            sim_brain_data.predict(
                y=y, spatial_scale="whole_brain", cv=3, estimator=SVC(kernel="rbf")
            )


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


class TestScoring:
    def test_scoring_none_uses_the_estimator_score(self, minimal_brain_data):
        """`scoring=None` scores with the estimator's own `score` method.

        A regressor's `score` is R2, which goes negative on noise — no accuracy
        scorer can produce that, so the assertion distinguishes the two.
        """
        from sklearn.base import clone

        n = minimal_brain_data.shape[0]
        y = np.random.default_rng(0).standard_normal(n)
        result = minimal_brain_data.predict(y=y, cv=3, estimator="ridge")

        expected = []
        splitter = _resolve_splitter(3, classifier=False, grouped=False, n_rows=n)
        for train, test in splitter.split(minimal_brain_data.data, y):
            fitted = clone(result.estimator).fit(
                minimal_brain_data.data[train], y[train]
            )
            expected.append(fitted.score(minimal_brain_data.data[test], y[test]))
        np.testing.assert_allclose(result.scores, expected)
        assert np.any(result.scores < 0)

    def test_explicit_scoring_overrides_the_estimator_score(self, minimal_brain_data):
        """A callable scorer replaces the estimator's own `score`."""
        sentinel = 0.4242

        def constant_scorer(estimator, X, y):
            return sentinel

        n = minimal_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = minimal_brain_data.predict(y=y, cv=3, scoring=constant_scorer)
        np.testing.assert_allclose(result.scores, [sentinel] * 3)


# ---------------------------------------------------------------------------
# Searchlight / ROI — score_map populated
# ---------------------------------------------------------------------------


class TestSearchlight:
    def test_returns_predict_with_score_map(self, minimal_brain_data):
        """Searchlight on a minimal 5-voxel fixture — fast."""
        n = minimal_brain_data.shape[0]
        n_voxels = minimal_brain_data.shape[1]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        result = minimal_brain_data.predict(
            y=y, spatial_scale="searchlight", cv=3, radius=4.0, n_jobs=1
        )
        assert isinstance(result, Predict)
        assert result.spatial_scale == "searchlight"
        assert result.score_map.shape == (n_voxels,)
        np.testing.assert_array_equal(result.classes, [0, 1])
        # Searchlight exposes nothing but the score map.
        for field in ("predictions", "cv_folds", "scores", "estimator", "weight_map"):
            assert getattr(result, field) is None

    def test_score_summaries_raise_and_point_at_the_score_map(self, minimal_brain_data):
        n = minimal_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        result = minimal_brain_data.predict(
            y=y, spatial_scale="searchlight", cv=3, radius=4.0, n_jobs=1
        )
        for summary in ("mean_score", "std_score"):
            with pytest.raises(AttributeError, match="score_map"):
                getattr(result, summary)


class TestAllDataRefit:
    """The all-data refit is always-on for whole_brain dispatch — there's no
    ``refit=`` kwarg. ``weight_map`` is the canonical, publishable map (one
    legitimate estimator, all the data) and ``estimator`` is that fitted
    sklearn object. Fold-specific coefficient maps are not exposed.
    """

    def test_estimator_is_fitted_and_callable_on_new_data(self, sim_brain_data):
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = sim_brain_data.predict(
            y=y, spatial_scale="whole_brain", cv=3, estimator="linear_svc"
        )
        assert result.estimator is not None
        # Linear-model estimators expose .coef_ (possibly inside a Pipeline).
        assert hasattr(result.estimator, "named_steps") or hasattr(
            result.estimator, "coef_"
        )
        # The fitted estimator must work on a new design at the same width.
        new_X = np.random.RandomState(0).randn(4, sim_brain_data.shape[1])
        new_pred = result.estimator.predict(new_X)
        assert new_pred.shape == (4,)

    def test_weight_map_is_from_all_data_fit_not_cv_mean(self, sim_brain_data):
        """``weight_map`` is the all-data ``coef_`` back-projected to voxels.

        Not the average of the per-fold coefficients — that is the whole point
        of the always-on refit.
        """
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = sim_brain_data.predict(
            y=y, spatial_scale="whole_brain", cv=3, estimator="linear_svc"
        )
        est = result.estimator
        coef = est.named_steps["linearsvc"].coef_.ravel()
        scale = est.named_steps["standardscaler"].scale_
        np.testing.assert_allclose(result.weight_map.data, coef / scale)


# ---------------------------------------------------------------------------
# Brain-space wrapping — spatial fields are BrainData, not raw arrays
# ---------------------------------------------------------------------------


class TestBrainDataWrapping:
    """Spatial result fields are BrainData objects so users can call
    ``.plot()`` directly without wrapping. Numpy access via ``.data``.
    """

    def test_whole_brain_weight_map_is_braindata(self, sim_brain_data):
        from nltools.data import BrainData

        n = sim_brain_data.shape[0]
        n_voxels = sim_brain_data.shape[1]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = sim_brain_data.predict(
            y=y, spatial_scale="whole_brain", cv=3, estimator="linear_svc"
        )
        assert isinstance(result.weight_map, BrainData)
        # Equivalent but independently owned mask (so .plot() composes without aliasing)
        assert result.weight_map.mask is not sim_brain_data.mask
        np.testing.assert_array_equal(
            result.weight_map.mask.get_fdata(), sim_brain_data.mask.get_fdata()
        )
        np.testing.assert_array_equal(
            result.weight_map.mask.affine, sim_brain_data.mask.affine
        )
        # Underlying numpy still accessible and has expected shapes
        assert result.weight_map.data.shape == (n_voxels,)

    def test_searchlight_score_map_is_braindata(self, minimal_brain_data):
        from nltools.data import BrainData

        n = minimal_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = minimal_brain_data.predict(
            y=y, spatial_scale="searchlight", cv=3, radius=4.0, n_jobs=1
        )
        assert isinstance(result.score_map, BrainData)
        assert result.score_map.mask is not minimal_brain_data.mask
        np.testing.assert_array_equal(
            result.score_map.mask.get_fdata(), minimal_brain_data.mask.get_fdata()
        )
        np.testing.assert_array_equal(
            result.score_map.mask.affine, minimal_brain_data.mask.affine
        )


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

    def test_roi_populates_the_roi_field_set(self, minimal_brain_data):
        from nltools.data import BrainData

        n = minimal_brain_data.shape[0]
        n_voxels = minimal_brain_data.shape[1]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        atlas = self._build_atlas(minimal_brain_data, n_rois=2)

        result = minimal_brain_data.predict(
            y=y,
            spatial_scale="roi",
            roi_mask=atlas,
            cv=3,
            estimator="linear_svc",
            n_jobs=1,
        )
        assert result.spatial_scale == "roi"
        assert result.scores.shape == (3, 2)  # (n_folds, n_rois)
        assert result.mean_score.shape == (2,)
        assert result.std_score.shape == (2,)
        assert result.roi_labels.shape == (2,)
        # Atlas labels in parcel-score order
        assert list(result.roi_labels) == [1, 2]
        assert isinstance(result.score_map, BrainData)
        assert result.score_map.shape == (n_voxels,)
        assert isinstance(result.weight_map, BrainData)
        assert result.weight_map.shape == (n_voxels,)
        # Whole-brain-only fields stay None on ROI decoding.
        assert result.predictions is None
        assert result.cv_folds is None

    def test_roi_accepts_a_brain_data_atlas(self, minimal_brain_data):
        """A BrainData atlas resolves the same as the Nifti image it was built from."""
        from nltools.data import BrainData

        n = minimal_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        atlas_img = self._build_atlas(minimal_brain_data, n_rois=2)
        atlas_bd = BrainData(atlas_img, mask=minimal_brain_data.mask)

        result_from_img = minimal_brain_data.predict(
            y=y,
            spatial_scale="roi",
            roi_mask=atlas_img,
            cv=3,
            estimator="linear_svc",
            n_jobs=1,
        )
        result_from_bd = minimal_brain_data.predict(
            y=y,
            spatial_scale="roi",
            roi_mask=atlas_bd,
            cv=3,
            estimator="linear_svc",
            n_jobs=1,
        )
        np.testing.assert_array_equal(
            result_from_bd.roi_labels, result_from_img.roi_labels
        )
        np.testing.assert_allclose(
            result_from_bd.score_map.data, result_from_img.score_map.data
        )

    def test_roi_does_not_expose_an_estimator_mapping(self, minimal_brain_data):
        """ROI decoding hides its per-parcel models; only the maps come back."""
        n = minimal_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        atlas = self._build_atlas(minimal_brain_data, n_rois=2)

        result = minimal_brain_data.predict(
            y=y,
            spatial_scale="roi",
            roi_mask=atlas,
            cv=3,
            estimator="linear_svc",
            n_jobs=1,
        )
        assert result.estimator is None

    def test_roi_score_map_paints_each_parcel_mean_fold_score(self, minimal_brain_data):
        from nilearn.masking import apply_mask

        n = minimal_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        atlas = self._build_atlas(minimal_brain_data, n_rois=2)

        result = minimal_brain_data.predict(
            y=y,
            spatial_scale="roi",
            roi_mask=atlas,
            cv=3,
            estimator="linear_svc",
            n_jobs=1,
        )
        label_vec = apply_mask(atlas, minimal_brain_data.mask).astype(int)
        for index, label in enumerate(result.roi_labels):
            voxels = result.score_map.data[label_vec == label]
            np.testing.assert_allclose(voxels, result.mean_score[index])

    def test_roi_weight_map_matches_an_independent_per_parcel_fit(
        self, minimal_brain_data
    ):
        """Each voxel's weight equals its parcel's all-data ``coef_``.

        The runner no longer returns per-parcel estimators, so the reference is
        an independent fit of the same pipeline on the same parcel columns.
        """
        from nilearn.masking import apply_mask
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC

        n = minimal_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        atlas = self._build_atlas(minimal_brain_data, n_rois=2)
        result = minimal_brain_data.predict(
            y=y,
            spatial_scale="roi",
            roi_mask=atlas,
            cv=3,
            estimator="linear_svc",
            n_jobs=1,
        )

        label_vec = apply_mask(atlas, minimal_brain_data.mask).astype(int)
        weights = result.weight_map.data
        for label in result.roi_labels:
            cols = label_vec == label
            reference = make_pipeline(
                StandardScaler(), LinearSVC(dual="auto", max_iter=10000)
            ).fit(minimal_brain_data.data[:, cols], y)
            expected = (
                reference.named_steps["linearsvc"].coef_.ravel()
                / reference.named_steps["standardscaler"].scale_
            )
            np.testing.assert_allclose(weights[cols], expected)

    def test_roi_non_linear_model_raises(self, minimal_brain_data):
        """Non-linear ROI decoding is rejected, not degraded to a missing map.

        Inverted from the 0.6.0-dev behaviour, which warned once and returned
        ``weight_map=None`` for the whole call.
        """
        from sklearn.svm import SVC

        n = minimal_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        atlas = self._build_atlas(minimal_brain_data, n_rois=2)

        with pytest.raises(ValueError, match="coef_"):
            minimal_brain_data.predict(
                y=y,
                spatial_scale="roi",
                roi_mask=atlas,
                cv=3,
                estimator=SVC(kernel="rbf"),
                n_jobs=1,
            )

    def test_roi_multiclass_assembles_one_map_per_class(self, minimal_brain_data):
        """Multiclass ROI decoding paints every class map into parcel voxels."""
        from nilearn.masking import apply_mask

        n = minimal_brain_data.shape[0]
        n_voxels = minimal_brain_data.shape[1]
        y = np.array([0, 1, 2] * (n // 3) + [0] * (n % 3))
        atlas = self._build_atlas(minimal_brain_data, n_rois=2)

        result = minimal_brain_data.predict(
            y=y, spatial_scale="roi", roi_mask=atlas, cv=3, n_jobs=1
        )

        np.testing.assert_array_equal(result.classes, [0, 1, 2])
        assert result.weight_map.shape == (3, n_voxels)
        label_vec = apply_mask(atlas, minimal_brain_data.mask).astype(int)
        maps = result.weight_map.data
        assert np.isfinite(maps[:, label_vec != 0]).all()
        assert not np.allclose(maps[0], maps.mean(axis=0))


# ---------------------------------------------------------------------------
# Result ownership — every returned map is a new, independent BrainData
# ---------------------------------------------------------------------------


class TestReturnedMapOwnership:
    """Returned maps own their data and mask, and carry no row metadata.

    Their leading axis is a coefficient or score axis, not the source
    observations, so the spec's row-metadata rule clears ``.X`` and ``.Y``.
    """

    def _assert_independent(self, brain_map, source):
        from nltools.data import BrainData

        assert isinstance(brain_map, BrainData)
        assert brain_map.mask is not source.mask
        assert brain_map.X.is_empty()
        assert brain_map.Y.is_empty()
        assert not hasattr(brain_map, "model_")

        before = np.array(brain_map.data, copy=True)
        source.data[:] = source.data + 100.0
        np.testing.assert_array_equal(brain_map.data, before)

        brain_map.data[0] = 12345.0
        assert not np.any(source.data == 12345.0)
        brain_map.mask.get_fdata()[:] = 0
        assert source.mask.get_fdata().sum() > 0

    def test_whole_brain_weight_map_is_independent(self, minimal_brain_data):
        n = minimal_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = minimal_brain_data.predict(y=y, cv=3)
        self._assert_independent(result.weight_map, minimal_brain_data)

    def test_searchlight_score_map_is_independent(self, minimal_brain_data):
        n = minimal_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = minimal_brain_data.predict(
            y=y, spatial_scale="searchlight", cv=3, radius=4.0, n_jobs=1
        )
        self._assert_independent(result.score_map, minimal_brain_data)

    @pytest.mark.parametrize("field", ["score_map", "weight_map"])
    def test_roi_maps_are_independent(self, field, minimal_brain_data):
        import nibabel as nib

        mask_data = minimal_brain_data.mask.get_fdata().astype(bool)
        flat = np.ones(int(mask_data.sum()), dtype=np.int64)
        flat[len(flat) // 2 :] = 2
        atlas_data = np.zeros(mask_data.shape, dtype=np.int64)
        atlas_data[mask_data] = flat
        atlas = nib.Nifti1Image(
            atlas_data, minimal_brain_data.mask.affine, minimal_brain_data.mask.header
        )

        n = minimal_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = minimal_brain_data.predict(
            y=y, spatial_scale="roi", roi_mask=atlas, cv=3, n_jobs=1
        )
        self._assert_independent(getattr(result, field), minimal_brain_data)


# ---------------------------------------------------------------------------
# String class labels (F4): labels travel with the data as strings
# ---------------------------------------------------------------------------


class TestStringLabels:
    """Decoding with string class labels — the headline stored-Y use case."""

    def test_whole_brain_predict_with_string_labels(self, minimal_brain_data):
        n = minimal_brain_data.shape[0]
        labels = np.array(["face", "house"] * (n // 2))
        minimal_brain_data.Y = {"condition": labels}
        result = minimal_brain_data.predict(y="condition", cv=5)
        preds = np.asarray(result.predictions)
        assert preds.dtype.kind == "U"
        assert set(np.unique(preds)) <= {"face", "house"}
        assert 0.0 <= result.mean_score <= 1.0

    def test_whole_brain_numeric_predictions_stay_float(self, minimal_brain_data):
        """Regression path must keep float predictions (no int truncation)."""
        n = minimal_brain_data.shape[0]
        rng = np.random.default_rng(0)
        y = rng.standard_normal(n)
        result = minimal_brain_data.predict(y=y, estimator="ridge", cv=5)
        preds = np.asarray(result.predictions)
        assert preds.dtype.kind == "f"


# ---------------------------------------------------------------------------
# Signature contract (spec "Prediction and decoding")
# ---------------------------------------------------------------------------


class TestPredictSignature:
    """`BrainData.predict` exposes exactly the spec's keyword-only signature."""

    EXPECTED = (
        ("X", None),
        ("y", None),
        ("estimator", "linear_svc"),
        ("cv", None),
        ("groups", None),
        ("scoring", None),
        ("spatial_scale", "whole_brain"),
        ("roi_mask", None),
        ("radius", 10.0),
        ("n_jobs", 1),
        ("progress_bar", False),
    )

    @staticmethod
    def _labels(bd):
        n = bd.shape[0]
        return np.array([0] * (n // 2) + [1] * (n - n // 2))

    def test_signature_matches_spec(self):
        from nltools.data import BrainData

        params = list(inspect.signature(BrainData.predict).parameters.values())
        assert params[0].name == "self"
        rest = params[1:]
        assert [p.name for p in rest] == [name for name, _ in self.EXPECTED]
        assert [p.default for p in rest] == [default for _, default in self.EXPECTED]

    def test_every_argument_is_keyword_only(self):
        from nltools.data import BrainData

        rest = list(inspect.signature(BrainData.predict).parameters.values())[1:]
        assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in rest)

    def test_static_overloads_are_declared(self):
        """The two mode-specific return-type overloads are visible to type checkers."""
        from nltools.data import BrainData

        overloads = typing.get_overloads(BrainData.predict)
        returns = [inspect.signature(o).return_annotation for o in overloads]
        assert len(overloads) == 2
        assert any("BrainData" in str(r) for r in returns)
        assert any("Predict" in str(r) for r in returns)

    @pytest.mark.parametrize(
        "removed",
        ["model", "standardize", "reduce", "n_components", "inplace", "radius_mm"],
    )
    def test_removed_keyword_raises_type_error(self, minimal_brain_data, removed):
        y = self._labels(minimal_brain_data)
        with pytest.raises(TypeError):
            minimal_brain_data.predict(y=y, cv=3, **{removed: 1})

    def test_random_state_keyword_is_removed(self, minimal_brain_data):
        y = self._labels(minimal_brain_data)
        with pytest.raises(TypeError):
            minimal_brain_data.predict(y=y, cv=3, random_state=0)

    def test_scoring_auto_is_rejected(self, minimal_brain_data):
        y = self._labels(minimal_brain_data)
        with pytest.raises(ValueError, match="auto"):
            minimal_brain_data.predict(y=y, cv=3, scoring="auto")

    def test_multimetric_scoring_mapping_is_rejected(self, minimal_brain_data):
        y = self._labels(minimal_brain_data)
        with pytest.raises(ValueError, match="one value per"):
            minimal_brain_data.predict(
                y=y, cv=3, scoring={"acc": "accuracy", "bal": "balanced_accuracy"}
            )

    @pytest.mark.parametrize("alias", ["loo", "logo"])
    def test_cv_string_aliases_are_rejected(self, minimal_brain_data, alias):
        y = self._labels(minimal_brain_data)
        with pytest.raises(ValueError, match="LeaveOne"):
            minimal_brain_data.predict(y=y, cv=alias)

    @pytest.mark.parametrize(
        "abbreviation, canonical",
        [
            ("svm", "linear_svc"),
            ("logistic", "logistic_regression"),
            ("lda", "linear_discriminant_analysis"),
            ("svr", "linear_svr"),
        ],
    )
    def test_ambiguous_estimator_abbreviations_are_rejected(
        self, minimal_brain_data, abbreviation, canonical
    ):
        y = self._labels(minimal_brain_data)
        with pytest.raises(ValueError, match=canonical):
            minimal_brain_data.predict(y=y, cv=3, estimator=abbreviation)

    @pytest.mark.parametrize(
        "shortcut",
        [
            "linear_svc",
            "logistic_regression",
            "linear_discriminant_analysis",
            "ridge_classifier",
        ],
    )
    def test_classification_shortcuts_run(self, minimal_brain_data, shortcut):
        y = self._labels(minimal_brain_data)
        result = minimal_brain_data.predict(y=y, cv=3, estimator=shortcut)
        assert result.weight_map is not None

    @pytest.mark.parametrize("shortcut", ["ridge", "lasso", "linear_svr"])
    def test_regression_shortcuts_run(self, minimal_brain_data, shortcut):
        y = np.random.default_rng(0).standard_normal(minimal_brain_data.shape[0])
        result = minimal_brain_data.predict(y=y, cv=3, estimator=shortcut)
        assert result.weight_map is not None

    def test_linear_svr_shortcut_is_linear(self, minimal_brain_data):
        from sklearn.svm import LinearSVR

        y = np.random.default_rng(0).standard_normal(minimal_brain_data.shape[0])
        result = minimal_brain_data.predict(y=y, cv=3, estimator="linear_svr")
        assert isinstance(result.estimator[-1], LinearSVR)


# ---------------------------------------------------------------------------
# Invalid argument / state combinations (C26)
# ---------------------------------------------------------------------------


class TestInvalidCombinations:
    """Every argument/state combination the spec does not list raises early."""

    @staticmethod
    def _labels(bd):
        n = bd.shape[0]
        return np.array([0] * (n // 2) + [1] * (n - n // 2))

    @staticmethod
    def _fit_ridge(bd):
        X = np.random.default_rng(0).standard_normal((bd.shape[0], 3))
        bd.fit(model="ridge", X=X, ridge_alpha=1.0)
        return X

    def test_roi_mask_without_roi_scale_raises(self, minimal_brain_data):
        y = self._labels(minimal_brain_data)
        with pytest.raises(ValueError, match="roi_mask"):
            minimal_brain_data.predict(y=y, cv=3, roi_mask="atlas.nii.gz")

    def test_roi_scale_without_roi_mask_raises(self, minimal_brain_data):
        y = self._labels(minimal_brain_data)
        with pytest.raises(ValueError, match="roi_mask"):
            minimal_brain_data.predict(y=y, cv=3, spatial_scale="roi")

    def test_radius_outside_searchlight_raises(self, minimal_brain_data):
        y = self._labels(minimal_brain_data)
        with pytest.raises(ValueError, match="radius"):
            minimal_brain_data.predict(y=y, cv=3, radius=6.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"estimator": "ridge"},
            {"cv": 3},
            {"groups": np.zeros(50)},
            {"scoring": "r2"},
            {"spatial_scale": "roi"},
            {"roi_mask": "atlas.nii.gz"},
            {"radius": 6.0},
            {"n_jobs": 4},
            {"progress_bar": True},
        ],
    )
    def test_mvpa_arguments_on_the_fitted_model_path_raise(
        self, minimal_brain_data, kwargs
    ):
        X = self._fit_ridge(minimal_brain_data)
        with pytest.raises(ValueError, match="decoding"):
            minimal_brain_data.predict(X=X, **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"estimator": "ridge"},
            {"cv": 3},
            {"groups": np.zeros(50)},
            {"scoring": "r2"},
            {"spatial_scale": "roi"},
            {"roi_mask": "atlas.nii.gz"},
            {"radius": 6.0},
            {"n_jobs": 4},
            {"progress_bar": True},
        ],
    )
    def test_mvpa_arguments_on_the_no_argument_path_raise(
        self, minimal_brain_data, kwargs
    ):
        self._fit_ridge(minimal_brain_data)
        with pytest.raises(ValueError, match="decoding"):
            minimal_brain_data.predict(**kwargs)

    def test_the_defaults_themselves_are_accepted_on_a_fitted_call(
        self, minimal_brain_data
    ):
        """Passing every decoding argument at its default is still a fitted call."""
        X = self._fit_ridge(minimal_brain_data)
        predicted = minimal_brain_data.predict(
            X=X,
            estimator="linear_svc",
            cv=None,
            groups=None,
            scoring=None,
            spatial_scale="whole_brain",
            roi_mask=None,
            radius=10.0,
            n_jobs=1,
            progress_bar=False,
        )
        assert predicted.shape == minimal_brain_data.shape

    def test_no_model_no_labels_raises(self, minimal_brain_data):
        with pytest.raises(ValueError, match="fit"):
            minimal_brain_data.predict()


# ---------------------------------------------------------------------------
# MVPA target and group validation
# ---------------------------------------------------------------------------


class RaisingEstimator(BaseEstimator):
    """Estimator whose `fit` always raises — proves validation runs first."""

    def fit(self, X, y):
        raise AssertionError("fit must not be reached")

    def predict(self, X):
        raise AssertionError("predict must not be reached")


class TestMvpaTargets:
    def test_two_dimensional_y_is_rejected(self, minimal_brain_data):
        n = minimal_brain_data.shape[0]
        y = np.zeros((n, 2))
        with pytest.raises(ValueError, match="one-dimensional"):
            minimal_brain_data.predict(y=y, cv=3)

    def test_multilabel_indicator_y_is_rejected(self, minimal_brain_data):
        n = minimal_brain_data.shape[0]
        y = np.tile(np.array([[1, 0], [0, 1]]), (n // 2, 1))
        with pytest.raises(ValueError, match="one-dimensional"):
            minimal_brain_data.predict(y=y, cv=3)

    def test_y_row_count_mismatch_is_rejected(self, minimal_brain_data):
        y = np.zeros(minimal_brain_data.shape[0] + 1)
        with pytest.raises(ValueError, match="one value per row"):
            minimal_brain_data.predict(y=y, cv=3)

    def test_groups_row_count_mismatch_is_rejected(self, minimal_brain_data):
        n = minimal_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        with pytest.raises(ValueError, match="one value per row"):
            minimal_brain_data.predict(y=y, cv=3, groups=np.zeros(n + 1))

    def test_two_dimensional_groups_is_rejected(self, minimal_brain_data):
        n = minimal_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        with pytest.raises(ValueError, match="one value per row"):
            minimal_brain_data.predict(y=y, cv=3, groups=np.zeros((n, 2)))

    def test_target_validation_happens_before_any_fit(self, minimal_brain_data):
        y = np.zeros((minimal_brain_data.shape[0], 2))
        with pytest.raises(ValueError, match="one-dimensional"):
            minimal_brain_data.predict(y=y, cv=3, estimator=RaisingEstimator())


# ---------------------------------------------------------------------------
# Cross-validation partition rules
# ---------------------------------------------------------------------------


class TestCvPartition:
    @staticmethod
    def _labels(bd):
        n = bd.shape[0]
        return np.array([0] * (n // 2) + [1] * (n - n // 2))

    @staticmethod
    def _reference_folds(splitter, X, y, groups=None):
        """Fold index per row, as a reference splitter assigns them."""
        folds = np.empty(len(y), dtype=int)
        for index, (_, test) in enumerate(splitter.split(X, y, groups)):
            folds[test] = index
        return folds

    def test_cv_none_is_a_deterministic_unshuffled_five_fold(self, minimal_brain_data):
        """`cv=None` is StratifiedKFold(5) with no shuffle — not a seeded shuffle."""
        y = self._labels(minimal_brain_data)
        first = minimal_brain_data.predict(y=y)
        second = minimal_brain_data.predict(y=y)

        assert first.scores.shape == (5,)
        np.testing.assert_array_equal(first.cv_folds, second.cv_folds)
        np.testing.assert_array_equal(
            first.cv_folds,
            self._reference_folds(
                StratifiedKFold(n_splits=5), minimal_brain_data.data, y
            ),
        )

    def test_cv_none_on_a_regressor_stratifies_on_quantile_bins(
        self, minimal_brain_data
    ):
        """`cv=None` on a regressor is an unshuffled StratifiedKFold(5) on bins of `y`."""
        y = np.random.default_rng(0).standard_normal(minimal_brain_data.shape[0])
        first = minimal_brain_data.predict(y=y, estimator="ridge")
        second = minimal_brain_data.predict(y=y, estimator="ridge")

        np.testing.assert_array_equal(first.cv_folds, second.cv_folds)
        np.testing.assert_array_equal(
            first.cv_folds,
            self._reference_folds(
                StratifiedKFold(n_splits=5),
                minimal_brain_data.data,
                _continuous_strata(y, 5),
            ),
        )

    def test_integer_cv_with_groups_is_group_aware(self, minimal_brain_data):
        """An int plus `groups` promotes to the group-aware stratified splitter."""
        n = minimal_brain_data.shape[0]
        y = self._labels(minimal_brain_data)
        groups = np.arange(n) % 5

        with warnings.catch_warnings():
            # sklearn warns once per split when a splitter ignores `groups`,
            # which is exactly the silent leak this promotion removes.
            warnings.simplefilter("error", UserWarning)
            result = minimal_brain_data.predict(y=y, cv=4, groups=groups)

        assert result.scores.shape == (4,)
        np.testing.assert_array_equal(
            result.cv_folds,
            self._reference_folds(
                StratifiedGroupKFold(n_splits=4), minimal_brain_data.data, y, groups
            ),
        )
        # No group straddles the train/test boundary.
        assert all(
            len(np.unique(result.cv_folds[groups == g])) == 1 for g in np.unique(groups)
        )

    def test_integer_cv_with_groups_on_a_regressor_keeps_groups_whole(
        self, minimal_brain_data
    ):
        """The regressor promotion bins `y` and splits those bins group-aware."""
        n = minimal_brain_data.shape[0]
        y = np.random.default_rng(0).standard_normal(n)
        groups = np.arange(n) % 5

        result = minimal_brain_data.predict(y=y, estimator="ridge", cv=3, groups=groups)

        assert result.scores.shape == (3,)
        assert all(
            len(np.unique(result.cv_folds[groups == g])) == 1 for g in np.unique(groups)
        )

    def test_integer_cv_on_a_regressor_balances_fold_means(self):
        """Quantile-bin stratification is nltools' own addition to sklearn's grammar."""
        rng = np.random.default_rng(1)
        # Skewed and already ordered: the worst case for a contiguous KFold.
        y = np.sort(rng.exponential(size=120))
        X = rng.normal(size=(120, 3))
        stratified = _resolve_splitter(4, classifier=False, grouped=False, n_rows=120)

        stratified_means = [y[test].mean() for _, test in stratified.split(X, y)]
        plain_means = [y[test].mean() for _, test in KFold(n_splits=4).split(X, y)]

        # The folds never shuffle, so a within-bin gradient survives; the
        # spread across fold means is still a fraction of a plain KFold's.
        assert np.std(stratified_means) < np.std(plain_means) / 4
        assert abs(np.mean(stratified_means) - y.mean()) < 0.05
        assert stratified.get_n_splits(X, y) == 4

    def test_too_few_rows_for_a_stratified_regressor_raises_in_nltools_terms(self):
        """Quantile bins need two rows per fold; sklearn's "class" wording must not leak."""
        with pytest.raises(ValueError, match="8 row"):
            _resolve_splitter(5, classifier=False, grouped=False, n_rows=8)

    def test_continuous_strata_are_never_thinner_than_the_fold_count(self):
        """Ties at a quantile edge must not leave a bin scikit-learn calls a class."""
        tied = np.round(np.random.default_rng(0).standard_normal(32), 1)

        strata = _continuous_strata(tied, n_splits=2)

        assert np.bincount(strata).min() >= 2

    def test_continuous_strata_are_capped_quantile_bins(self):
        """Ten bins at most, each big enough for every fold, and ordinals pass through."""
        strata = _continuous_strata(
            np.random.default_rng(0).standard_normal(200), n_splits=5
        )

        assert strata.min() == 0 and strata.max() == 9
        assert np.bincount(strata).min() >= 2 * 5
        np.testing.assert_array_equal(
            _continuous_strata(np.repeat([1, 2, 3], 20), n_splits=5),
            np.repeat([0, 1, 2], 20),
        )

    def test_supplied_splitter_is_used_as_given(self, minimal_brain_data):
        n = minimal_brain_data.shape[0]
        y = self._labels(minimal_brain_data)
        groups = np.arange(n) % 3
        result = minimal_brain_data.predict(y=y, cv=LeaveOneGroupOut(), groups=groups)

        assert result.scores.shape == (3,)
        np.testing.assert_array_equal(
            result.cv_folds,
            self._reference_folds(
                LeaveOneGroupOut(), minimal_brain_data.data, y, groups
            ),
        )

    def test_overlapping_test_folds_are_rejected(self, minimal_brain_data):
        from sklearn.model_selection import ShuffleSplit

        y = self._labels(minimal_brain_data)
        with pytest.raises(ValueError, match="partition"):
            minimal_brain_data.predict(
                y=y, cv=ShuffleSplit(n_splits=3, test_size=0.5, random_state=0)
            )

    def test_repeated_test_folds_are_rejected(self, minimal_brain_data):
        from sklearn.model_selection import RepeatedKFold

        y = self._labels(minimal_brain_data)
        with pytest.raises(ValueError, match="partition"):
            minimal_brain_data.predict(
                y=y, cv=RepeatedKFold(n_splits=3, n_repeats=2, random_state=0)
            )

    def test_partition_validation_happens_before_any_fit(self, minimal_brain_data):
        from sklearn.model_selection import ShuffleSplit

        y = self._labels(minimal_brain_data)
        with pytest.raises(ValueError, match="partition"):
            minimal_brain_data.predict(
                y=y,
                cv=ShuffleSplit(n_splits=2, test_size=0.5, random_state=0),
                estimator=RaisingEstimator(),
            )

    def test_every_observation_gets_one_fold_index(self, minimal_brain_data):
        """Every row carries a real fold index — no `-1` "never tested" sentinel."""
        y = self._labels(minimal_brain_data)
        n = minimal_brain_data.shape[0]
        result = minimal_brain_data.predict(y=y, cv=5)

        assert np.all(result.cv_folds >= 0)
        assert set(np.unique(result.cv_folds)) == {0, 1, 2, 3, 4}
        assert list(np.bincount(result.cv_folds)) == [n // 5] * 5
        np.testing.assert_array_equal(
            result.cv_folds,
            self._reference_folds(
                StratifiedKFold(n_splits=5), minimal_brain_data.data, y
            ),
        )


# ---------------------------------------------------------------------------
# `predict` attaches nothing to the source
# ---------------------------------------------------------------------------


class TestNoPredictAttributes:
    @staticmethod
    def _labels(bd):
        n = bd.shape[0]
        return np.array([0] * (n // 2) + [1] * (n - n // 2))

    @staticmethod
    def _atlas(bd, n_rois=2):
        import nibabel as nib

        mask_data = bd.mask.get_fdata().astype(bool)
        flat = np.zeros(int(mask_data.sum()), dtype=np.int64)
        chunk = max(1, len(flat) // n_rois)
        for i in range(n_rois):
            flat[i * chunk : (i + 1) * chunk] = i + 1
        flat[n_rois * chunk :] = n_rois
        out = np.zeros(mask_data.shape, dtype=np.int64)
        out[mask_data] = flat
        return nib.Nifti1Image(out, bd.mask.affine, bd.mask.header)

    def _assert_clean(self, bd, before_attributes, before_data):
        """Nothing added, nothing removed, no data touched — fit state included."""
        assert not [name for name in vars(bd) if name.startswith("predict_")]
        assert set(vars(bd)) == before_attributes
        np.testing.assert_array_equal(bd.data, before_data)

    def test_whole_brain_attaches_nothing(self, minimal_brain_data):
        attributes = set(vars(minimal_brain_data))
        before = minimal_brain_data.data.copy()
        minimal_brain_data.predict(y=self._labels(minimal_brain_data), cv=3)
        self._assert_clean(minimal_brain_data, attributes, before)

    def test_searchlight_attaches_nothing(self, minimal_brain_data):
        attributes = set(vars(minimal_brain_data))
        before = minimal_brain_data.data.copy()
        minimal_brain_data.predict(
            y=self._labels(minimal_brain_data),
            cv=3,
            spatial_scale="searchlight",
            radius=4.0,
        )
        self._assert_clean(minimal_brain_data, attributes, before)

    def test_roi_attaches_nothing(self, minimal_brain_data):
        attributes = set(vars(minimal_brain_data))
        before = minimal_brain_data.data.copy()
        minimal_brain_data.predict(
            y=self._labels(minimal_brain_data),
            cv=3,
            spatial_scale="roi",
            roi_mask=self._atlas(minimal_brain_data),
        )
        self._assert_clean(minimal_brain_data, attributes, before)

    def test_decoding_leaves_fitted_state_intact(self, minimal_brain_data):
        """MVPA on a fitted object neither clears nor replaces the fit state."""
        n = minimal_brain_data.shape[0]
        X = np.random.default_rng(0).standard_normal((n, 3))
        minimal_brain_data.fit(model="ridge", X=X, ridge_alpha=1.0)
        fitted_model = minimal_brain_data.model_
        weights = minimal_brain_data.ridge_weights.data.copy()
        attributes = set(vars(minimal_brain_data))
        before = minimal_brain_data.data.copy()

        minimal_brain_data.predict(y=self._labels(minimal_brain_data), cv=3)

        self._assert_clean(minimal_brain_data, attributes, before)
        assert minimal_brain_data.model_ is fitted_model
        np.testing.assert_array_equal(minimal_brain_data.ridge_weights.data, weights)

    def test_prediction_state_inventory_is_gone(self):
        from nltools.data.braindata import utils

        assert not hasattr(utils, "_PREDICTION_STATE_ATTRIBUTES")
        assert not hasattr(utils, "_clear_prediction_state")
        assert not [
            name for name in utils._FIT_STATE_ATTRIBUTES if name.startswith("predict_")
        ]


# ---------------------------------------------------------------------------
# Built-in shortcut pipelines and multiclass strategy
# ---------------------------------------------------------------------------


CLASSIFICATION_SHORTCUTS = [
    "linear_svc",
    "logistic_regression",
    "linear_discriminant_analysis",
    "ridge_classifier",
]
REGRESSION_SHORTCUTS = ["ridge", "lasso", "linear_svr"]


def _binary_labels(bd):
    n = bd.shape[0]
    return np.array([0, 1] * (n // 2) + [0] * (n % 2))


def _three_class_labels(bd):
    n = bd.shape[0]
    return np.array([0, 1, 2] * (n // 3) + [0] * (n % 3))


class TestShortcutPipelines:
    """A shortcut name selects a predefined linear pipeline; callers' objects don't."""

    @pytest.mark.parametrize(
        "shortcut,final",
        [
            ("linear_svc", "LinearSVC"),
            ("logistic_regression", "LogisticRegression"),
            ("linear_discriminant_analysis", "LinearDiscriminantAnalysis"),
            ("ridge_classifier", "RidgeClassifier"),
            ("ridge", "Ridge"),
            ("lasso", "Lasso"),
            ("linear_svr", "LinearSVR"),
        ],
    )
    def test_each_shortcut_is_a_scaler_plus_its_linear_estimator(
        self, shortcut, final, minimal_brain_data
    ):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        y = (
            _binary_labels(minimal_brain_data)
            if shortcut in CLASSIFICATION_SHORTCUTS
            else np.random.default_rng(0).standard_normal(minimal_brain_data.shape[0])
        )

        result = minimal_brain_data.predict(y=y, cv=3, estimator=shortcut)

        assert isinstance(result.estimator, Pipeline)
        assert len(result.estimator.steps) == 2
        assert isinstance(result.estimator[0], StandardScaler)
        assert type(result.estimator[-1]).__name__ == final

    def test_linear_svc_keeps_its_hyperparameters(self, minimal_brain_data):
        """v0.5.1-dev carry-over: dual='auto', max_iter=10000 (audit note 7)."""
        y = _binary_labels(minimal_brain_data)

        result = minimal_brain_data.predict(y=y, cv=3, estimator="linear_svc")

        estimator = result.estimator[-1]
        assert estimator.dual == "auto"
        assert estimator.max_iter == 10000

    @pytest.mark.parametrize("shortcut", CLASSIFICATION_SHORTCUTS)
    def test_a_classification_shortcut_is_one_vs_rest_for_multiclass(
        self, shortcut, minimal_brain_data
    ):
        from sklearn.multiclass import OneVsRestClassifier

        y = _three_class_labels(minimal_brain_data)

        result = minimal_brain_data.predict(y=y, cv=3, estimator=shortcut)

        assert isinstance(result.estimator[-1], OneVsRestClassifier)
        assert len(result.estimator[-1].estimators_) == 3

    @pytest.mark.parametrize("shortcut", CLASSIFICATION_SHORTCUTS)
    def test_a_classification_shortcut_is_not_wrapped_for_binary(
        self, shortcut, minimal_brain_data
    ):
        from sklearn.multiclass import OneVsRestClassifier

        y = _binary_labels(minimal_brain_data)

        result = minimal_brain_data.predict(y=y, cv=3, estimator=shortcut)

        assert not isinstance(result.estimator[-1], OneVsRestClassifier)

    @pytest.mark.parametrize("shortcut", REGRESSION_SHORTCUTS)
    def test_a_regression_shortcut_is_never_wrapped(self, shortcut, minimal_brain_data):
        from sklearn.multiclass import OneVsRestClassifier

        y = np.random.default_rng(1).standard_normal(minimal_brain_data.shape[0])

        result = minimal_brain_data.predict(y=y, cv=3, estimator=shortcut)

        assert not isinstance(result.estimator[-1], OneVsRestClassifier)

    def test_a_caller_supplied_classifier_keeps_its_multiclass_strategy(
        self, minimal_brain_data
    ):
        """MVPA never wraps a caller's estimator, multiclass or not."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier

        y = _three_class_labels(minimal_brain_data)

        result = minimal_brain_data.predict(
            y=y, cv=3, estimator=LogisticRegression(max_iter=1000)
        )

        assert isinstance(result.estimator, LogisticRegression)
        assert not isinstance(result.estimator, OneVsRestClassifier)

    def test_a_caller_supplied_one_vs_rest_is_used_as_given(self, minimal_brain_data):
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import LinearSVC

        n_voxels = minimal_brain_data.shape[1]
        y = _three_class_labels(minimal_brain_data)

        result = minimal_brain_data.predict(
            y=y,
            cv=3,
            estimator=OneVsRestClassifier(LinearSVC(dual="auto", max_iter=10000)),
        )

        assert isinstance(result.estimator, OneVsRestClassifier)
        assert result.weight_map.shape == (3, n_voxels)


# ---------------------------------------------------------------------------
# Weight-map shapes and class semantics
# ---------------------------------------------------------------------------


class TestWeightMapShapes:
    def test_regression_is_one_map(self, minimal_brain_data):
        n_voxels = minimal_brain_data.shape[1]
        y = np.random.default_rng(2).standard_normal(minimal_brain_data.shape[0])

        result = minimal_brain_data.predict(y=y, cv=3, estimator="ridge")

        assert result.weight_map.shape == (n_voxels,)
        assert result.classes is None

    def test_binary_is_one_signed_map_for_the_second_class(self, minimal_brain_data):
        n_voxels = minimal_brain_data.shape[1]
        y = _binary_labels(minimal_brain_data)

        result = minimal_brain_data.predict(y=y, cv=3, estimator="linear_svc")

        assert result.weight_map.shape == (n_voxels,)
        np.testing.assert_array_equal(result.classes, [0, 1])
        estimator = result.estimator[-1]
        expected = estimator.coef_.ravel() / result.estimator[0].scale_
        np.testing.assert_allclose(result.weight_map.data, expected)

    def test_multiclass_one_vs_rest_stacks_children_in_class_order(
        self, minimal_brain_data
    ):
        n_voxels = minimal_brain_data.shape[1]
        y = _three_class_labels(minimal_brain_data)

        result = minimal_brain_data.predict(y=y, cv=3, estimator="linear_svc")

        assert result.weight_map.shape == (3, n_voxels)
        scale = result.estimator[0].scale_
        for index, child in enumerate(result.estimator[-1].estimators_):
            np.testing.assert_allclose(
                result.weight_map.data[index], child.coef_.ravel() / scale
            )

    def test_native_multiclass_keeps_the_estimator_rows(self, minimal_brain_data):
        from sklearn.linear_model import LogisticRegression

        n_voxels = minimal_brain_data.shape[1]
        y = _three_class_labels(minimal_brain_data)

        result = minimal_brain_data.predict(
            y=y, cv=3, estimator=LogisticRegression(max_iter=1000)
        )

        assert result.weight_map.shape == (3, n_voxels)
        np.testing.assert_allclose(result.weight_map.data, result.estimator.coef_)

    def test_every_successful_whole_brain_result_has_a_map(self, minimal_brain_data):
        """Including the two cases that used to degrade to ``weight_map=None``."""
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC

        binary = _binary_labels(minimal_brain_data)
        multiclass = _three_class_labels(minimal_brain_data)
        selecting = make_pipeline(
            StandardScaler(),
            SelectKBest(f_classif, k=3),
            LinearSVC(dual="auto", max_iter=10000),
        )

        assert minimal_brain_data.predict(y=binary, cv=3).weight_map is not None
        assert minimal_brain_data.predict(y=multiclass, cv=3).weight_map is not None
        assert (
            minimal_brain_data.predict(y=binary, cv=3, estimator=selecting).weight_map
            is not None
        )


# ---------------------------------------------------------------------------
# Pipeline validation — raise instead of warn and degrade
# ---------------------------------------------------------------------------


class TestPipelineValidation:
    def test_an_unsupported_transformer_raises_whole_brain(self, minimal_brain_data):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer
        from sklearn.svm import LinearSVC

        y = _binary_labels(minimal_brain_data)
        pipe = make_pipeline(Normalizer(), LinearSVC(dual="auto", max_iter=10000))

        with pytest.raises(ValueError, match="Normalizer"):
            minimal_brain_data.predict(y=y, cv=3, estimator=pipe)

    def test_an_unsupported_transformer_raises_before_any_fit(self, minimal_brain_data):
        """The whitelist is structural, so nothing is fitted before it raises."""
        from sklearn.base import BaseEstimator
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import Normalizer
        from sklearn.svm import LinearSVC

        fits = []

        class SpyingSVC(LinearSVC, BaseEstimator):
            def fit(self, X, y, **kwargs):
                fits.append(1)
                return super().fit(X, y, **kwargs)

        y = _binary_labels(minimal_brain_data)
        pipe = Pipeline([("norm", Normalizer()), ("clf", SpyingSVC(dual="auto"))])

        with pytest.raises(ValueError, match="Normalizer"):
            minimal_brain_data.predict(y=y, cv=3, estimator=pipe)
        assert fits == []

    def test_an_unsupported_transformer_raises_for_searchlight(
        self, minimal_brain_data
    ):
        """Searchlight uses the same transformer whitelist."""
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer
        from sklearn.svm import LinearSVC

        y = _binary_labels(minimal_brain_data)
        pipe = make_pipeline(Normalizer(), LinearSVC(dual="auto", max_iter=10000))

        with pytest.raises(ValueError, match="Normalizer"):
            minimal_brain_data.predict(
                y=y, cv=3, estimator=pipe, spatial_scale="searchlight", radius=4.0
            )

    def test_one_vs_rest_must_be_the_final_step(self, minimal_brain_data):
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC

        y = _three_class_labels(minimal_brain_data)
        pipe = Pipeline(
            [
                ("ovr", OneVsRestClassifier(LinearSVC(dual="auto"))),
                ("clf", LinearSVC(dual="auto")),
            ]
        )

        with pytest.raises(ValueError, match="final"):
            minimal_brain_data.predict(y=y, cv=3, estimator=pipe)


# ---------------------------------------------------------------------------
# v0.5.1 numerical regression guards
# ---------------------------------------------------------------------------


class TestV051NumericalGuards:
    """Pin the formulas v0.5.1 used, so the rewrite cannot silently drift.

    Reference values come from the v0.5.1 source (``649fd6c0``,
    ``nltools/data/brain_data.py::predict``), recomputed here — never read back
    out of the current implementation.
    """

    def test_pca_back_projection_matches_the_v051_formula(self, minimal_brain_data):
        """v0.5.1 'lassopcr': ``np.dot(pca.components_.T, lasso.coef_)``."""
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Lasso
        from sklearn.pipeline import make_pipeline

        y = np.random.default_rng(3).standard_normal(minimal_brain_data.shape[0])
        pipe = make_pipeline(PCA(n_components=3), Lasso(alpha=0.01))

        result = minimal_brain_data.predict(y=y, cv=3, estimator=pipe)

        pca, lasso = result.estimator[0], result.estimator[-1]
        v051 = np.dot(pca.components_.T, lasso.coef_)
        np.testing.assert_allclose(result.weight_map.data, v051)

    def test_weight_map_is_on_the_raw_voxel_scale(self, minimal_brain_data):
        """v0.5.1 fitted raw voxels, so its map was in voxel units.

        An unregularized fit makes the comparison exact: standardizing and then
        dividing the coefficients by ``scale_`` reproduces the raw-data fit
        v0.5.1's ``'linear'`` algorithm performed.
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        y = np.random.default_rng(4).standard_normal(minimal_brain_data.shape[0])
        pipe = make_pipeline(StandardScaler(), LinearRegression())

        result = minimal_brain_data.predict(y=y, cv=3, estimator=pipe)

        v051 = LinearRegression().fit(minimal_brain_data.data, y).coef_
        np.testing.assert_allclose(result.weight_map.data, v051, rtol=1e-8)

    def test_the_shortcut_map_divides_by_the_scaler_scale(self, minimal_brain_data):
        """The built-in pipelines standardize, so the map must be unscaled again."""
        y = _binary_labels(minimal_brain_data)

        result = minimal_brain_data.predict(y=y, cv=3, estimator="linear_svc")

        scaler, estimator = result.estimator[0], result.estimator[-1]
        np.testing.assert_allclose(
            result.weight_map.data, estimator.coef_.ravel() / scaler.scale_
        )

    def test_binary_coefficients_keep_the_v051_squeeze_and_sign(
        self, minimal_brain_data
    ):
        """v0.5.1 stored ``predictor.coef_.squeeze()`` — one row, never negated."""
        from sklearn.svm import LinearSVC

        y = _binary_labels(minimal_brain_data)

        result = minimal_brain_data.predict(
            y=y, cv=3, estimator=LinearSVC(dual="auto", max_iter=10000)
        )

        v051 = (
            LinearSVC(dual="auto", max_iter=10000)
            .fit(minimal_brain_data.data, y)
            .coef_.squeeze()
        )
        np.testing.assert_allclose(result.weight_map.data, v051)

    def test_multiclass_keeps_one_map_per_class_like_v051(self, minimal_brain_data):
        """v0.5.1 kept a per-class list of maps; averaging them is forbidden."""
        n_voxels = minimal_brain_data.shape[1]
        y = _three_class_labels(minimal_brain_data)

        result = minimal_brain_data.predict(y=y, cv=3, estimator="linear_svc")

        assert result.weight_map.shape == (len(result.classes), n_voxels)
        assert result.weight_map.shape != (n_voxels,)


# ---------------------------------------------------------------------------
# Whole-brain fold parallelism
# ---------------------------------------------------------------------------


class TestFoldParallelism:
    def test_parallel_folds_match_the_serial_path(self, minimal_brain_data):
        y = _binary_labels(minimal_brain_data)

        serial = minimal_brain_data.predict(y=y, cv=5, n_jobs=1)
        parallel = minimal_brain_data.predict(y=y, cv=5, n_jobs=2)

        np.testing.assert_array_equal(serial.predictions, parallel.predictions)
        np.testing.assert_array_equal(serial.cv_folds, parallel.cv_folds)
        # Bitwise: the folds are materialized once, so a worker-side numerical
        # difference is exactly what this test exists to catch.
        np.testing.assert_array_equal(serial.scores, parallel.scores)
        np.testing.assert_array_equal(serial.weight_map.data, parallel.weight_map.data)

    def test_n_jobs_reaches_the_whole_brain_fold_loop(
        self, monkeypatch, minimal_brain_data
    ):
        import joblib

        seen = []
        real_parallel = joblib.Parallel

        class SpyParallel(real_parallel):
            def __init__(self, *args, **kwargs):
                # Every construction, not just the first: an sklearn internal
                # that builds its own Parallel must not shadow the fold loop.
                seen.append(kwargs.get("n_jobs"))
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(joblib, "Parallel", SpyParallel)
        y = _binary_labels(minimal_brain_data)

        minimal_brain_data.predict(y=y, cv=3, n_jobs=2)

        assert 2 in seen
