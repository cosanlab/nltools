"""Tests for Predict dataclass (fitresults/).

Mirrors test_fit_results.py — Predict shares the frozen-dataclass pattern,
available()/asdict() introspection, and "None means not applicable to this
mode" semantics.
"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from nltools.data.fitresults import Predict


class TestPredictCreation:
    def test_minimal_creation(self):
        """Predict allows construction with no fields (all None default)."""
        result = Predict()
        assert result.predictions is None
        assert result.weight_map is None
        assert result.mean_score is None

    def test_whole_brain_classification(self):
        n_samples, n_voxels, n_folds = 100, 1000, 5
        result = Predict(
            predictions=np.random.randn(n_samples),
            scores=np.random.randn(n_folds),
            mean_score=0.72,
            std_score=0.05,
            cv_folds=np.arange(n_samples) % n_folds,
            weight_map=np.random.randn(n_voxels),
            fold_weight_maps=np.random.randn(n_folds, n_voxels),
        )

        assert result.predictions.shape == (n_samples,)
        assert result.scores.shape == (n_folds,)
        assert result.mean_score == 0.72
        assert result.weight_map.shape == (n_voxels,)
        assert result.fold_weight_maps.shape == (n_folds, n_voxels)
        assert result.accuracy_map is None

    def test_searchlight_result(self):
        n_voxels = 1000
        result = Predict(
            accuracy_map=np.random.randn(n_voxels),
            mean_score=0.65,
        )
        assert result.accuracy_map.shape == (n_voxels,)
        assert result.weight_map is None

    def test_roi_result_with_repurposed_score_fields(self):
        """ROI dispatch repurposes scores/mean_score/std_score with array
        shapes. ``roi_labels`` carries the atlas IDs in matching order.
        """
        n_folds, n_rois, n_voxels = 5, 200, 1000
        result = Predict(
            scores=np.random.rand(n_folds, n_rois),
            mean_score=np.random.rand(n_rois),
            std_score=np.random.rand(n_rois),
            roi_labels=np.arange(1, n_rois + 1, dtype=np.int64),
            accuracy_map=np.random.rand(n_voxels),
        )
        assert result.scores.shape == (n_folds, n_rois)
        assert result.mean_score.shape == (n_rois,)
        assert result.std_score.shape == (n_rois,)
        assert result.roi_labels.shape == (n_rois,)
        assert "roi_labels" in result.available()
        # Whole-brain-only fields stay None on ROI dispatch
        assert result.weight_map is None
        assert result.fold_weight_maps is None

    def test_estimator_field(self):
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
            weight_map=np.random.randn(n_voxels),
        )
        assert result.estimator is est
        assert result.weight_map.shape == (n_voxels,)


class TestPredictImmutability:
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
