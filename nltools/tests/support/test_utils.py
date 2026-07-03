"""Tests for cross-cutting helpers in ``nltools.utils``."""

import gc

import numpy as np
import pandas as pd

from nltools.utils import coalesced_gc


class TestCoalescedGC:
    """``coalesced_gc()`` collapses nilearn's per-copy gc storm into one sweep."""

    def test_single_real_collect_on_exit(self, monkeypatch):
        """Interim gc.collect() calls no-op; exactly one real collect on exit."""
        spy = _CountingCollect()
        monkeypatch.setattr(gc, "collect", spy)

        with coalesced_gc():
            gc.collect()
            gc.collect()
            gc.collect()

        # The three interior calls hit the no-op; only the exit sweep hits the spy.
        assert spy.count == 1

    def test_nesting_yields_single_real_collect(self, monkeypatch):
        """Nested contexts still produce exactly one real collect (outermost)."""
        spy = _CountingCollect()
        monkeypatch.setattr(gc, "collect", spy)

        with coalesced_gc():
            gc.collect()
            with coalesced_gc():
                gc.collect()
                gc.collect()
            gc.collect()

        assert spy.count == 1

    def test_env_var_disables_coalescing(self, monkeypatch):
        """NLTOOLS_NO_GC_COALESCE=1 makes it a pure passthrough."""
        monkeypatch.setenv("NLTOOLS_NO_GC_COALESCE", "1")
        spy = _CountingCollect()
        monkeypatch.setattr(gc, "collect", spy)

        with coalesced_gc():
            gc.collect()
            gc.collect()

        # Passthrough: every interior collect runs for real, no swap, no exit sweep.
        assert spy.count == 2

    def test_gc_collect_restored_after_exit(self, monkeypatch):
        """The real builtin is restored on exit even after coalescing."""
        real = gc.collect
        with coalesced_gc():
            assert gc.collect is not real  # swapped to the no-op inside
        assert gc.collect is real

    def test_wrapped_fit_numerically_identical(self, minimal_brain_data, monkeypatch):
        """A coalesced BrainData.fit matches the un-coalesced (passthrough) fit."""
        design = pd.DataFrame(
            {
                "Intercept": np.ones(len(minimal_brain_data)),
                "X1": np.random.RandomState(0).randn(len(minimal_brain_data)),
            }
        )

        # Coalesced path (default — fit is decorated with @coalesced_gc()).
        coalesced = minimal_brain_data.copy()
        coalesced.fit(model="glm", X=design)

        # Passthrough path (decorator becomes a no-op wrapper).
        monkeypatch.setenv("NLTOOLS_NO_GC_COALESCE", "1")
        passthrough = minimal_brain_data.copy()
        passthrough.fit(model="glm", X=design)

        for attr in ("glm_betas", "glm_t", "glm_p", "glm_se", "glm_residual"):
            a = getattr(coalesced, attr).data
            b = getattr(passthrough, attr).data
            assert np.array_equal(a, b), f"{attr} differs between coalesced/passthrough"


class _CountingCollect:
    """Stand-in for ``gc.collect`` that counts real invocations."""

    def __init__(self):
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return 0
