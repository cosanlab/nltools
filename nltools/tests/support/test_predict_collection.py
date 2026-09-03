"""Tests for `PredictCollection` per-subject prediction results.

The container returned by ``BrainCollection.predict(y=...)``: one `Predict`
per subject plus the collection's per-subject metadata, with stacking
affordances that feed second-level inference (``weight_maps`` →
``BrainData (n_subjects, n_voxels)`` → ``permutation_test``).
"""

from dataclasses import FrozenInstanceError

import nibabel as nib
import numpy as np
import polars as pl
import pytest

from nltools.data import Predict, PredictCollection


@pytest.fixture(scope="module")
def tiny_mask():
    affine = np.eye(4) * 2
    affine[3, 3] = 1
    return nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.int8), affine)


def _whole_brain_predict(seed: int, mask) -> Predict:
    from nltools.data import BrainData

    rng = np.random.default_rng(seed)
    n_samples, n_voxels, n_folds = 12, 27, 3
    return Predict(
        predictions=rng.standard_normal(n_samples),
        scores=rng.standard_normal(n_folds),
        mean_score=float(seed) / 10.0,
        std_score=0.05,
        cv_folds=np.arange(n_samples) % n_folds,
        weight_map=BrainData(
            rng.standard_normal((1, n_voxels)).astype(np.float32), mask=mask
        ),
    )


class TestConstruction:
    def test_from_list_of_predicts(self, tiny_mask):
        pc = PredictCollection([_whole_brain_predict(i, tiny_mask) for i in range(3)])
        assert len(pc) == 3
        assert isinstance(pc[0], Predict)
        assert all(isinstance(r, Predict) for r in pc)

    def test_results_binding_is_frozen(self, tiny_mask):
        pc = PredictCollection([_whole_brain_predict(0, tiny_mask)])
        with pytest.raises(FrozenInstanceError):
            pc.results = ()

    def test_payloads_are_independently_owned(self, tiny_mask):
        result = _whole_brain_predict(0, tiny_mask)
        metadata = pl.DataFrame({"subject": ["s1"]})
        pc = PredictCollection([result], metadata=metadata)

        result.scores[0] = 99.0
        result.weight_map.data[0] = 99.0

        assert pc[0] is not result
        assert pc[0].scores[0] != 99.0
        assert not np.all(pc[0].weight_map.data == 99.0)
        assert pc.metadata is not metadata

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            PredictCollection([])

    def test_non_predict_raises(self):
        with pytest.raises(TypeError, match="Predict"):
            PredictCollection([np.zeros(3)])

    def test_metadata_row_count_must_match(self, tiny_mask):
        meta = pl.DataFrame({"subject": ["s1", "s2"]})
        with pytest.raises(ValueError, match="metadata"):
            PredictCollection(
                [_whole_brain_predict(0, tiny_mask)],
                metadata=meta,
            )


class TestScores:
    def test_mean_scores_stacks_scalars(self, tiny_mask):
        pc = PredictCollection([_whole_brain_predict(i, tiny_mask) for i in range(3)])
        np.testing.assert_allclose(pc.mean_scores, [0.0, 0.1, 0.2])

    def test_mean_scores_stacks_roi_arrays(self, tiny_mask):
        results = [
            Predict(mean_score=np.full(4, float(i)), std_score=np.zeros(4))
            for i in range(3)
        ]
        pc = PredictCollection(results)
        assert pc.mean_scores.shape == (3, 4)

    def test_scores_dataframe_carries_metadata(self, tiny_mask):
        meta = pl.DataFrame({"subject": ["s1", "s2", "s3"]})
        pc = PredictCollection(
            [_whole_brain_predict(i, tiny_mask) for i in range(3)],
            metadata=meta,
        )
        df = pc.scores
        assert isinstance(df, pl.DataFrame)
        assert df["subject"].to_list() == ["s1", "s2", "s3"]
        np.testing.assert_allclose(df["mean_score"].to_numpy(), [0.0, 0.1, 0.2])
        assert "std_score" in df.columns

    def test_scores_dataframe_without_metadata_has_subject_index(self, tiny_mask):
        pc = PredictCollection([_whole_brain_predict(i, tiny_mask) for i in range(2)])
        assert pc.scores["subject"].to_list() == [0, 1]

    def test_scores_dataframe_refuses_array_valued_scores(self):
        results = [Predict(mean_score=np.zeros(4)) for _ in range(2)]
        with pytest.raises(ValueError, match="mean_scores"):
            PredictCollection(results).scores


class TestMapStacking:
    def test_weight_maps_stack_to_braindata(self, tiny_mask):
        from nltools.data import BrainData

        pc = PredictCollection([_whole_brain_predict(i, tiny_mask) for i in range(3)])
        wm = pc.weight_maps
        assert isinstance(wm, BrainData)
        assert wm.data.shape == (3, 27)
        # Row i is subject i's weight map.
        np.testing.assert_allclose(wm.data[1], pc[1].weight_map.data.reshape(-1))

    def test_weight_maps_missing_raises(self):
        results = [Predict(mean_score=0.5) for _ in range(2)]
        with pytest.raises(ValueError, match="weight_map"):
            PredictCollection(results).weight_maps

    def test_accuracy_maps_stack_to_braindata(self, tiny_mask):
        from nltools.data import BrainData

        rng = np.random.default_rng(0)
        results = [
            Predict(
                accuracy_map=BrainData(
                    rng.standard_normal((1, 27)).astype(np.float32), mask=tiny_mask
                )
            )
            for _ in range(2)
        ]
        am = PredictCollection(results).accuracy_maps
        assert isinstance(am, BrainData)
        assert am.data.shape == (2, 27)


class TestIntrospection:
    def test_available_is_intersection(self, tiny_mask):
        full = _whole_brain_predict(0, tiny_mask)
        sparse = Predict(mean_score=0.5)
        pc = PredictCollection([full, sparse])
        assert "mean_score" in pc.available()
        assert "weight_map" not in pc.available()

    def test_repr_mentions_len(self, tiny_mask):
        pc = PredictCollection([_whole_brain_predict(i, tiny_mask) for i in range(3)])
        assert "3" in repr(pc)
