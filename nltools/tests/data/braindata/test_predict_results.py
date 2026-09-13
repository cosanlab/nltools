"""Tests for the frozen structural `Predict` result record.

The spec's shape table (`docs/development/specs/braindata.md`, "Prediction and
decoding") is the contract: one record class, a required ``spatial_scale``
discriminator, and exactly one legal field combination per spatial scale.
"""

from dataclasses import FrozenInstanceError, fields

import nibabel as nib
import numpy as np
import pytest

from nltools.data import BrainData, Predict


N_SAMPLES, N_FOLDS, N_ROIS, N_VOXELS = 20, 4, 3, 125


@pytest.fixture(scope="module")
def tiny_mask():
    return nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.uint8), np.eye(4))


@pytest.fixture
def brain_map(tiny_mask):
    def _make(n_maps=None):
        shape = (N_VOXELS,) if n_maps is None else (n_maps, N_VOXELS)
        return BrainData(
            np.random.default_rng(0).standard_normal(shape), mask=tiny_mask
        )

    return _make


@pytest.fixture
def whole_brain_fields(brain_map):
    return {
        "spatial_scale": "whole_brain",
        "scoring": None,
        "classes": np.array([0, 1]),
        "predictions": np.arange(N_SAMPLES) % 2,
        "cv_folds": np.arange(N_SAMPLES) % N_FOLDS,
        "scores": np.linspace(0.4, 0.8, N_FOLDS),
        "estimator": "fitted-estimator",
        "weight_map": brain_map(),
    }


@pytest.fixture
def roi_fields(brain_map):
    return {
        "spatial_scale": "roi",
        "scoring": "accuracy",
        "classes": np.array([0, 1]),
        "scores": np.random.default_rng(1).random((N_FOLDS, N_ROIS)),
        "weight_map": brain_map(),
        "roi_labels": np.arange(1, N_ROIS + 1, dtype=np.int64),
        "score_map": brain_map(),
    }


@pytest.fixture
def searchlight_fields(brain_map):
    return {
        "spatial_scale": "searchlight",
        "scoring": "accuracy",
        "classes": np.array([0, 1]),
        "score_map": brain_map(),
    }


class TestPredictFieldSet:
    def test_fields_match_the_spec_table(self):
        assert [field.name for field in fields(Predict)] == [
            "spatial_scale",
            "scoring",
            "classes",
            "predictions",
            "cv_folds",
            "scores",
            "estimator",
            "weight_map",
            "roi_labels",
            "score_map",
        ]

    def test_spatial_scale_is_required(self):
        with pytest.raises(TypeError):
            Predict()

    def test_unknown_spatial_scale_is_rejected(self):
        with pytest.raises(ValueError, match="spatial_scale"):
            Predict(spatial_scale="voxel", score_map=None)


class TestWholeBrainConstruction:
    def test_construction(self, whole_brain_fields):
        result = Predict(**whole_brain_fields)

        assert result.spatial_scale == "whole_brain"
        assert result.scoring is None
        assert result.classes.shape == (2,)
        assert result.predictions.shape == (N_SAMPLES,)
        assert result.cv_folds.shape == (N_SAMPLES,)
        assert result.scores.shape == (N_FOLDS,)
        assert result.estimator == "fitted-estimator"
        assert result.weight_map.shape == (N_VOXELS,)
        assert result.roi_labels is None
        assert result.score_map is None

    def test_regression_leaves_classes_none(self, whole_brain_fields):
        result = Predict(**{**whole_brain_fields, "classes": None})
        assert result.classes is None


class TestRoiConstruction:
    def test_construction(self, roi_fields):
        result = Predict(**roi_fields)

        assert result.spatial_scale == "roi"
        assert result.scoring == "accuracy"
        assert result.scores.shape == (N_FOLDS, N_ROIS)
        assert result.roi_labels.shape == (N_ROIS,)
        assert result.score_map.shape == (N_VOXELS,)
        assert result.weight_map.shape == (N_VOXELS,)
        assert result.predictions is None
        assert result.cv_folds is None
        assert result.estimator is None

    def test_weight_map_cannot_be_absent(self, roi_fields):
        """Every successful ROI result carries a map — there is no degraded path."""
        with pytest.raises(ValueError, match="weight_map.*roi"):
            Predict(**{**roi_fields, "weight_map": None})


class TestSearchlightConstruction:
    def test_construction(self, searchlight_fields):
        result = Predict(**searchlight_fields)

        assert result.spatial_scale == "searchlight"
        assert result.score_map.shape == (N_VOXELS,)
        for field in ("predictions", "cv_folds", "scores", "estimator", "weight_map"):
            assert getattr(result, field) is None
        assert result.roi_labels is None

    def test_weight_map_is_rejected(self, searchlight_fields, brain_map):
        with pytest.raises(ValueError, match="weight_map.*searchlight"):
            Predict(**{**searchlight_fields, "weight_map": brain_map()})

    def test_score_map_is_required(self, searchlight_fields):
        with pytest.raises(ValueError, match="score_map.*searchlight"):
            Predict(**{**searchlight_fields, "score_map": None})


class TestProducerPairsFoldsWithScores:
    """`BrainData.predict` itself keeps `cv_folds` indexing into `scores`.

    The record no longer re-checks the pairing, so the producer is the only
    enforcement point: every fold index must name a fold score, and every fold
    score must be named by some index.
    """

    def test_cv_folds_index_the_fold_scores(self, minimal_brain_data):
        y = np.arange(minimal_brain_data.shape[0]) % 2

        result = minimal_brain_data.predict(y=y, cv=4)

        folds = np.asarray(result.cv_folds)
        assert folds.min() >= 0
        assert folds.max() < len(result.scores)
        assert set(np.unique(folds).tolist()) == set(range(len(result.scores)))


class TestScoreSummaries:
    def test_whole_brain_summaries_derive_from_scores(self, whole_brain_fields):
        result = Predict(**whole_brain_fields)

        assert isinstance(result.mean_score, float)
        assert result.mean_score == pytest.approx(float(result.scores.mean()))
        assert result.std_score == pytest.approx(float(result.scores.std()))

    def test_roi_summaries_are_one_value_per_parcel(self, roi_fields):
        result = Predict(**roi_fields)

        assert result.mean_score.shape == (N_ROIS,)
        assert result.std_score.shape == (N_ROIS,)
        np.testing.assert_allclose(result.mean_score, result.scores.mean(axis=0))
        np.testing.assert_allclose(result.std_score, result.scores.std(axis=0))

    def test_roi_summaries_ignore_failed_parcels(self, roi_fields):
        scores = np.array(roi_fields["scores"])
        scores[0, 0] = np.nan
        result = Predict(**{**roi_fields, "scores": scores})

        assert np.isfinite(result.mean_score).all()
        np.testing.assert_allclose(result.mean_score, np.nanmean(scores, axis=0))

    @pytest.mark.parametrize("summary", ["mean_score", "std_score"])
    def test_searchlight_summaries_raise(self, summary, searchlight_fields):
        result = Predict(**searchlight_fields)

        with pytest.raises(AttributeError, match="score_map"):
            getattr(result, summary)

    def test_summaries_track_the_stored_scores(self, whole_brain_fields):
        result = Predict(**whole_brain_fields)
        result.scores[0] = 0.0
        assert result.mean_score == pytest.approx(float(result.scores.mean()))


class TestPredictOwnership:
    def test_arrays_are_copied(self, whole_brain_fields):
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        result = Predict(**{**whole_brain_fields, "scores": scores})

        result.scores[0] = 9.9
        assert scores[0] == 0.1

    def test_maps_are_copied(self, whole_brain_fields, brain_map):
        weight_map = brain_map()
        result = Predict(**{**whole_brain_fields, "weight_map": weight_map})

        assert result.weight_map is not weight_map
        result.weight_map.data[0] = 9.9
        assert weight_map.data[0] != 9.9

    def test_copied_estimator_remains_fitted_and_usable(self, whole_brain_fields):
        from sklearn.svm import LinearSVC

        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0, 0, 1, 1])
        estimator = LinearSVC().fit(X, y)

        result = Predict(**{**whole_brain_fields, "estimator": estimator})

        assert result.estimator is not estimator
        np.testing.assert_array_equal(result.estimator.predict(X), y)

    def test_a_callable_scorer_is_stored_as_supplied(self, whole_brain_fields):
        def scorer(estimator, X, y):
            return 1.0

        result = Predict(**{**whole_brain_fields, "scoring": scorer})
        assert result.scoring is scorer


class TestPredictPickle:
    """Unpickling is the one path that bypasses `__post_init__`."""

    def _old_format_bytes(self):
        """Pickle bytes carrying the pre-0.6.0-dev field layout for `Predict`."""
        import copyreg
        import pickle

        old_state = {
            "predictions": np.arange(N_SAMPLES),
            "scores": np.zeros(N_FOLDS),
            "mean_score": 0.5,
            "std_score": 0.1,
            "cv_folds": np.arange(N_SAMPLES) % N_FOLDS,
            "roi_labels": None,
            "accuracy_map": None,
            "weight_map": None,
            "fold_weight_maps": None,
            "estimator": None,
        }

        class _OldRecord:
            def __reduce__(self):
                return (copyreg._reconstructor, (Predict, object, None), old_state)

        return pickle.dumps(_OldRecord())

    def test_old_format_pickle_raises_instead_of_reading_as_empty(self):
        import pickle

        with pytest.raises(ValueError, match="older nltools"):
            pickle.loads(self._old_format_bytes())

    def test_round_trip_preserves_the_record(self, roi_fields):
        import pickle

        restored = pickle.loads(pickle.dumps(Predict(**roi_fields)))

        assert restored.spatial_scale == "roi"
        np.testing.assert_allclose(restored.scores, Predict(**roi_fields).scores)
        assert restored.score_map.shape == (N_VOXELS,)
        assert restored.mean_score.shape == (N_ROIS,)


class TestPredictFrozenBindings:
    def test_cannot_modify_field(self, whole_brain_fields):
        result = Predict(**whole_brain_fields)
        with pytest.raises(FrozenInstanceError):
            result.scores = np.zeros(N_FOLDS)

    def test_cannot_add_attribute(self, whole_brain_fields):
        result = Predict(**whole_brain_fields)
        with pytest.raises(FrozenInstanceError):
            result.new_field = 1


class TestPredictAvailable:
    def test_searchlight_lists_only_populated_fields(self, searchlight_fields):
        result = Predict(**searchlight_fields)
        assert set(result.available()) == {
            "spatial_scale",
            "scoring",
            "classes",
            "score_map",
        }

    def test_scoring_is_listed_when_it_is_none(self, searchlight_fields):
        """`scoring=None` is a value — the estimator's own `score` — not an absence."""
        result = Predict(**{**searchlight_fields, "scoring": None})
        assert "scoring" in result.available()
        assert "predictions" not in result.available()

    def test_excludes_private(self, whole_brain_fields):
        result = Predict(**whole_brain_fields)
        object.__setattr__(result, "_priv", 1)
        assert "_priv" not in result.available()


class TestPredictAsDict:
    def test_default_excludes_none(self, searchlight_fields):
        keys = Predict(**searchlight_fields).asdict().keys()
        assert set(keys) == {"spatial_scale", "scoring", "classes", "score_map"}

    def test_scoring_is_included_when_it_is_none(self, searchlight_fields):
        result = Predict(**{**searchlight_fields, "scoring": None})
        assert result.asdict()["scoring"] is None
        assert "predictions" not in result.asdict()

    def test_include_none(self, searchlight_fields):
        full = Predict(**searchlight_fields).asdict(include_none=True)
        assert full["predictions"] is None
        assert full["spatial_scale"] == "searchlight"

    def test_excludes_private(self, searchlight_fields):
        result = Predict(**searchlight_fields)
        object.__setattr__(result, "_priv", 1)
        assert "_priv" not in result.asdict(include_none=True)
