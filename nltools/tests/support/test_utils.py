"""Tests for cross-cutting helpers in ``nltools.utils``."""

import gc

import numpy as np
import pandas as pd

from nltools.utils import all_same, coalesced_gc


class TestAllSame:
    """``all_same()`` must actually detect an unequal element."""

    def test_detects_unequal_element(self):
        """The documented example: a differing item yields False (was broken —
        `np.all(generator)` wraps the generator in a truthy 0-d object array).
        """
        assert all_same([1, 1, 1]) is True or all_same([1, 1, 1])
        assert not all_same([1, 2, 1])

    def test_all_equal_and_single_and_empty(self):
        assert all_same([5, 5, 5])
        assert all_same(["a", "a"])
        assert all_same([7])  # single item trivially all-same

    def test_handles_array_like_items(self):
        """`np` import implies array items were intended; equal vs unequal arrays."""
        a, b = np.array([1, 2, 3]), np.array([1, 2, 3])
        c = np.array([1, 9, 3])
        assert all_same([a, b])
        assert not all_same([a, c])


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


class TestProgressHelpers:
    """``maybe_tqdm``/``make_progress_bar`` are the single library-wide progress mechanism."""

    def test_importable_from_nltools_utils(self):
        from nltools.utils import _NullProgressBar, make_progress_bar, maybe_tqdm

        assert callable(maybe_tqdm) and callable(make_progress_bar)
        assert callable(_NullProgressBar)

    def test_inference_layer_reexports_the_same_objects(self):
        from nltools import utils as top_utils
        from nltools.algorithms.inference import utils as inference_utils

        assert inference_utils.maybe_tqdm is top_utils.maybe_tqdm
        assert inference_utils.make_progress_bar is top_utils.make_progress_bar

    def test_maybe_tqdm_is_identity_when_disabled(self):
        from nltools.utils import maybe_tqdm

        it = range(3)
        assert maybe_tqdm(it, progress_bar=False) is it

    def test_make_progress_bar_disabled_is_inert_and_silent(self, capsys):
        from nltools.utils import make_progress_bar

        with make_progress_bar(progress_bar=False, total=5, desc="x") as bar:
            bar.update(2)
            bar.set_postfix(a=1)
            bar.set_description("y")
        assert capsys.readouterr().err == ""

    def test_only_utils_imports_tqdm(self):
        """Every progress bar must go through the shared helpers — no module
        outside ``nltools/utils.py`` may import tqdm directly."""
        import re
        from pathlib import Path

        import nltools

        pkg = Path(nltools.__file__).parent
        pattern = re.compile(r"^\s*(from tqdm[.\s]|import tqdm)", re.MULTILINE)
        offenders = [
            str(py.relative_to(pkg))
            for py in sorted(pkg.rglob("*.py"))
            if "tests" not in py.parts
            and py.relative_to(pkg) != Path("utils.py")
            and pattern.search(py.read_text())
        ]
        assert offenders == [], f"hand-rolled tqdm imports in: {offenders}"


class _CountingCollect:
    """Stand-in for ``gc.collect`` that counts real invocations."""

    def __init__(self):
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return 0
