"""Regression test for F049: predict() MVPA hardcoded random_state=42.

``predict_mvpa`` built ``StratifiedKFold``/``KFold`` with a fixed
``random_state=42`` and ``BrainData.predict`` exposed no ``random_state``,
so integer-CV decoding was pinned to a single shuffle. ``predict`` now
accepts ``random_state`` and threads it into the fold splitter.
"""

import numpy as np


def _make_labels(n):
    y = np.zeros(n, dtype=int)
    y[n // 2 :] = 1
    return y


class TestPredictRandomState:
    def test_different_seeds_give_different_folds(self, minimal_brain_data):
        y = _make_labels(len(minimal_brain_data))

        r0 = minimal_brain_data.predict(
            y=y, spatial_scale="whole_brain", model="svm", cv=5, random_state=0
        )
        r1 = minimal_brain_data.predict(
            y=y, spatial_scale="whole_brain", model="svm", cv=5, random_state=1
        )

        assert not np.array_equal(r0.cv_folds, r1.cv_folds), (
            "distinct random_state values should yield different fold splits"
        )

    def test_same_seed_reproducible(self, minimal_brain_data):
        y = _make_labels(len(minimal_brain_data))

        r_a = minimal_brain_data.predict(
            y=y, spatial_scale="whole_brain", model="svm", cv=5, random_state=7
        )
        r_b = minimal_brain_data.predict(
            y=y, spatial_scale="whole_brain", model="svm", cv=5, random_state=7
        )

        np.testing.assert_array_equal(r_a.cv_folds, r_b.cv_folds)

    def test_default_random_state_still_runs(self, minimal_brain_data):
        """Omitting random_state must still work (default None)."""
        y = _make_labels(len(minimal_brain_data))
        r = minimal_brain_data.predict(y=y, model="svm", cv=5)
        assert r.cv_folds.shape == (len(minimal_brain_data),)
