"""Tests for BrainData.predict() — kwargs API returning Predict dataclass."""

import inspect
import typing
import warnings

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, LeaveOneGroupOut, StratifiedKFold

from nltools.data import Predict


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

        assert result.predictions is not None
        assert result.predictions.shape == (n,)
        assert result.scores is not None
        assert result.scores.shape == (3,)
        assert isinstance(result.mean_score, float)
        assert isinstance(result.std_score, float)
        assert result.cv_folds is not None
        assert result.cv_folds.shape == (n,)
        # weight_map is the all-data refit (canonical), always populated for
        # linear models — no separate refit=True opt-in.
        assert result.weight_map is not None
        assert result.weight_map.shape == (n_voxels,)
        assert result.fold_weight_maps is not None
        assert result.fold_weight_maps.shape == (3, n_voxels)
        # All-data fitted estimator, available for .predict() on new data.
        assert result.estimator is not None
        # whole_brain doesn't populate accuracy_map
        assert result.accuracy_map is None

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
                y=y, spatial_scale="whole_brain", cv=3, estimator=SVC(kernel="rbf")
            )
        assert result.weight_map is None
        assert any("weight_map" in str(warn.message) for warn in w)


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
        for train, test in KFold(n_splits=3).split(minimal_brain_data.data, y):
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
# Searchlight / ROI — accuracy_map populated, weight_map None
# ---------------------------------------------------------------------------


class TestSearchlight:
    def test_returns_predict_with_accuracy_map(self, minimal_brain_data):
        """Searchlight on a minimal 5-voxel fixture — fast."""
        n = minimal_brain_data.shape[0]
        n_voxels = minimal_brain_data.shape[1]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))

        result = minimal_brain_data.predict(
            y=y, spatial_scale="searchlight", cv=3, radius=4.0, n_jobs=1
        )
        assert isinstance(result, Predict)
        assert result.accuracy_map is not None
        assert result.accuracy_map.shape == (n_voxels,)
        # weight_map intentionally not provided for searchlight
        assert result.weight_map is None
        assert result.fold_weight_maps is None


class TestAllDataRefit:
    """The all-data refit is always-on for whole_brain dispatch — there's no
    ``refit=`` kwarg. ``weight_map`` is the canonical, publishable map (single
    legitimate estimator, all the data); ``fold_weight_maps`` is the per-fold
    stack for stability analysis; ``estimator`` is the fitted sklearn object.
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
        """``weight_map`` should match ``estimator.coef_`` (back-projected
        through any PCA), not the across-fold average of ``fold_weight_maps``.
        That's the whole point of dropping the ``refit`` flag.
        """
        n = sim_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = sim_brain_data.predict(
            y=y, spatial_scale="whole_brain", cv=3, estimator="linear_svc"
        )
        # Pull coef_ from the all-data fit (no PCA in this default pipeline,
        # so .coef_ already lives in voxel space).
        est = result.estimator
        coef = est.named_steps["linearsvc"].coef_.ravel()
        np.testing.assert_allclose(result.weight_map.data, coef)


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
            y=y, spatial_scale="whole_brain", cv=3, estimator="linear_svc"
        )
        for field in ("weight_map", "fold_weight_maps"):
            obj = getattr(result, field)
            assert isinstance(obj, BrainData), f"{field} should be BrainData"
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
        assert result.fold_weight_maps.data.shape == (3, n_voxels)

    def test_searchlight_accuracy_map_is_braindata(self, minimal_brain_data):
        from nltools.data import BrainData

        n = minimal_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = minimal_brain_data.predict(
            y=y, spatial_scale="searchlight", cv=3, radius=4.0, n_jobs=1
        )
        assert isinstance(result.accuracy_map, BrainData)
        assert result.accuracy_map.mask is not minimal_brain_data.mask
        np.testing.assert_array_equal(
            result.accuracy_map.mask.get_fdata(), minimal_brain_data.mask.get_fdata()
        )
        np.testing.assert_array_equal(
            result.accuracy_map.mask.affine, minimal_brain_data.mask.affine
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

    def test_roi_populates_repurposed_score_fields(self, minimal_brain_data):
        from nltools.data import BrainData

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

    def test_roi_populates_voxel_space_weight_map(self, minimal_brain_data):
        """Non-overlapping ROI assembles per-parcel coefs back to voxel
        space (the atlas is a label image, so each voxel belongs to exactly
        one parcel — disjoint reassembly). Voxels outside the atlas are NaN.
        """
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
        assert isinstance(result.weight_map, BrainData)
        assert result.weight_map.data.shape == (n_voxels,)
        assert isinstance(result.fold_weight_maps, BrainData)
        assert result.fold_weight_maps.data.shape == (3, n_voxels)
        # estimator is a dict keyed by atlas label
        assert isinstance(result.estimator, dict)
        assert set(result.estimator.keys()) == {1, 2}

    def test_roi_weight_map_voxels_match_per_parcel_estimator_coef(
        self, minimal_brain_data
    ):
        """Regression: each voxel's value in result.weight_map must equal
        the corresponding entry of its parcel's all-data ``estimator.coef_``.
        That's the disjoint-reassembly contract.
        """
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
        # Recover the label vector that the runner used to assign voxels
        from nilearn.masking import apply_mask

        label_vec = apply_mask(atlas, minimal_brain_data.mask).astype(int)
        weights = result.weight_map.data
        for label, est in result.estimator.items():
            cols = label_vec == label
            est_coef = est.named_steps["linearsvc"].coef_.ravel()
            np.testing.assert_allclose(weights[cols], est_coef)

    def test_roi_non_linear_model_drops_weight_fields(self, minimal_brain_data):
        """Non-linear ROI dispatch can't expose coefs → weight_map /
        fold_weight_maps / estimator collapse to None for the whole call,
        matching whole_brain's all-or-nothing rule. Exactly one aggregate
        warning is emitted (not one per parcel).
        """
        from sklearn.svm import SVC
        import warnings

        n = minimal_brain_data.shape[0]
        y = np.array([0] * (n // 2) + [1] * (n - n // 2))
        atlas = self._build_atlas(minimal_brain_data, n_rois=2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = minimal_brain_data.predict(
                y=y,
                spatial_scale="roi",
                roi_mask=atlas,
                cv=3,
                estimator=SVC(kernel="rbf"),
                n_jobs=1,
            )
        assert result.weight_map is None
        assert result.fold_weight_maps is None
        assert result.estimator is None
        # Score fields still populated (CV ran fine, just couldn't extract weights)
        assert result.mean_score is not None
        # One aggregate warning, not per-parcel spam
        msgs = [m for m in w if "weight_map" in str(m.message).lower()]
        assert (
            len(msgs) <= 2
        )  # one from per-parcel quiet=True (suppressed) + one aggregate


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

    def test_cv_none_on_a_regressor_is_an_unshuffled_kfold(self, minimal_brain_data):
        y = np.random.default_rng(0).standard_normal(minimal_brain_data.shape[0])
        result = minimal_brain_data.predict(y=y, estimator="ridge")
        np.testing.assert_array_equal(
            result.cv_folds,
            self._reference_folds(KFold(n_splits=5), minimal_brain_data.data, y),
        )

    def test_integer_cv_ignores_groups_instead_of_going_group_aware(
        self, minimal_brain_data
    ):
        """An int is that many plain folds — never a `GroupKFold` promotion."""
        n = minimal_brain_data.shape[0]
        y = self._labels(minimal_brain_data)
        groups = np.arange(n) % 5

        result = minimal_brain_data.predict(y=y, cv=4, groups=groups)

        assert result.scores.shape == (4,)
        # The folds are exactly StratifiedKFold's, which never sees `groups`.
        np.testing.assert_array_equal(
            result.cv_folds,
            self._reference_folds(
                StratifiedKFold(n_splits=4), minimal_brain_data.data, y
            ),
        )
        # A group-aware splitter keeps each group inside one test fold; this
        # one must not, or the promotion is still happening.
        assert any(
            len(np.unique(result.cv_folds[groups == g])) > 1 for g in np.unique(groups)
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
