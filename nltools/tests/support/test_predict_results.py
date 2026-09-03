"""Tests for the frozen structural `Predict` result record."""

from dataclasses import FrozenInstanceError
import importlib

import nibabel as nib
import numpy as np
import pytest

from nltools.data import BrainData, Predict
from nltools.data.braindata.utils import _PREDICTION_STATE_ATTRIBUTES


@pytest.fixture(scope="module")
def tiny_mask():
    return nib.Nifti1Image(np.ones((10, 10, 10), dtype=np.uint8), np.eye(4))


def test_results_are_exported_only_from_supported_data_namespace():
    from nltools import data

    assert data.Predict is Predict
    assert not hasattr(data, "Fit")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("nltools.data.fitresults")


def test_prediction_state_inventory_matches_predict_fields():
    expected = {f"predict_{name}" for name in Predict.__dataclass_fields__}

    assert set(_PREDICTION_STATE_ATTRIBUTES) == expected


class TestPredictCreation:
    def test_minimal_creation(self):
        """Predict allows construction with no fields (all None default)."""
        result = Predict()
        assert result.predictions is None
        assert result.weight_map is None
        assert result.mean_score is None

    def test_whole_brain_classification(self, tiny_mask):
        n_samples, n_voxels, n_folds = 100, 1000, 5
        result = Predict(
            predictions=np.random.randn(n_samples),
            scores=np.random.randn(n_folds),
            mean_score=0.72,
            std_score=0.05,
            cv_folds=np.arange(n_samples) % n_folds,
            weight_map=BrainData(np.random.randn(n_voxels), mask=tiny_mask),
            fold_weight_maps=BrainData(
                np.random.randn(n_folds, n_voxels), mask=tiny_mask
            ),
        )

        assert result.predictions.shape == (n_samples,)
        assert result.scores.shape == (n_folds,)
        assert result.mean_score == 0.72
        assert result.weight_map.shape == (n_voxels,)
        assert result.fold_weight_maps.shape == (n_folds, n_voxels)
        assert result.accuracy_map is None

    def test_searchlight_result(self, tiny_mask):
        n_voxels = 1000
        result = Predict(
            accuracy_map=BrainData(np.random.randn(n_voxels), mask=tiny_mask),
            mean_score=0.65,
        )
        assert result.accuracy_map.shape == (n_voxels,)
        assert result.weight_map is None

    def test_roi_result_with_repurposed_score_fields(self, tiny_mask):
        """ROI dispatch repurposes scores/mean_score/std_score with array
        shapes. ``roi_labels`` carries the atlas IDs in matching order.
        """
        n_folds, n_rois, n_voxels = 5, 200, 1000
        result = Predict(
            scores=np.random.rand(n_folds, n_rois),
            mean_score=np.random.rand(n_rois),
            std_score=np.random.rand(n_rois),
            roi_labels=np.arange(1, n_rois + 1, dtype=np.int64),
            accuracy_map=BrainData(np.random.rand(n_voxels), mask=tiny_mask),
        )
        assert result.scores.shape == (n_folds, n_rois)
        assert result.mean_score.shape == (n_rois,)
        assert result.std_score.shape == (n_rois,)
        assert result.roi_labels.shape == (n_rois,)
        assert "roi_labels" in result.available()
        # Whole-brain-only fields stay None on ROI dispatch
        assert result.weight_map is None
        assert result.fold_weight_maps is None

    def test_estimator_field(self, tiny_mask):
        """``estimator`` holds the all-data fitted sklearn estimator. There
        is no separate ``final_estimator`` / ``final_weight_map`` — the
        all-data fit is canonical and ``weight_map`` is its coefficients.
        """
        n_voxels = 1000
        from sklearn.svm import LinearSVC

        est = LinearSVC()
        result = Predict(
            mean_score=0.7,
            estimator=est,
            weight_map=BrainData(np.random.randn(n_voxels), mask=tiny_mask),
        )
        assert result.estimator is not est
        assert result.weight_map.shape == (n_voxels,)

    def test_copied_estimator_remains_fitted_and_usable(self):
        from sklearn.svm import LinearSVC

        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0, 0, 1, 1])
        estimator = LinearSVC().fit(X, y)

        result = Predict(estimator=estimator)

        assert result.estimator is not estimator
        np.testing.assert_array_equal(result.estimator.predict(X), y)

    @pytest.mark.parametrize(
        "field", ["accuracy_map", "weight_map", "fold_weight_maps"]
    )
    def test_map_fields_reject_raw_arrays(self, field):
        with pytest.raises(TypeError, match=f"{field}.*BrainData"):
            Predict(**{field: np.zeros(3)})


class TestPredictFrozenBindings:
    def test_cannot_modify_field(self):
        result = Predict(mean_score=0.5)
        with pytest.raises(FrozenInstanceError):
            result.mean_score = 0.9

    def test_cannot_add_attribute(self):
        result = Predict()
        with pytest.raises(FrozenInstanceError):
            result.new_field = 1

    def test_array_contents_remain_mutable(self):
        arr = np.array([0.1, 0.2, 0.3])
        result = Predict(scores=arr)
        result.scores[0] = 9.9
        assert result.scores[0] == 9.9
        assert arr[0] == 0.1


class TestPredictAvailable:
    def test_empty_available(self):
        assert Predict().available() == []

    def test_partial_available(self):
        result = Predict(
            mean_score=0.7,
            scores=np.array([0.6, 0.8]),
        )
        assert set(result.available()) == {"mean_score", "scores"}

    def test_excludes_private(self):
        result = Predict(mean_score=0.5)
        object.__setattr__(result, "_priv", 1)
        assert "_priv" not in result.available()


class TestPredictAsDict:
    def test_default_excludes_none(self):
        result = Predict(mean_score=0.5)
        d = result.asdict()
        assert d == {"mean_score": 0.5}

    def test_include_none(self):
        result = Predict(mean_score=0.5)
        d = result.asdict(include_none=True)
        assert "predictions" in d
        assert d["predictions"] is None
        assert d["mean_score"] == 0.5

    def test_excludes_private(self):
        result = Predict(mean_score=0.5)
        object.__setattr__(result, "_priv", 1)
        assert "_priv" not in result.asdict(include_none=True)
