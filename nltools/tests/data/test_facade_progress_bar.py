"""Facade-level ``progress_bar`` threading tests.

The engine-layer permutation/bootstrap functions expose ``progress_bar``
(default False). Before that, the facades below showed bars unconditionally —
so unless each facade exposes and forwards the kwarg, the capability is lost
entirely (always-off with no knob). These tests pin the threading.
"""

import contextlib
import inspect
import io

import numpy as np
import pytest

from nltools.data import Adjacency, BrainData

FACADE_METHODS = [
    Adjacency.similarity,
    Adjacency.ttest,
    Adjacency.bootstrap,
    BrainData.bootstrap,
]


def _stderr_of(fn):
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        fn()
    return buf.getvalue()


@pytest.mark.parametrize("method", FACADE_METHODS, ids=lambda m: f"{m.__qualname__}")
def test_facade_exposes_progress_bar_default_false(method):
    sig = inspect.signature(method)
    assert "progress_bar" in sig.parameters, f"{method.__qualname__} lacks progress_bar"
    param = sig.parameters["progress_bar"]
    assert param.default is False
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


class TestAdjacencyFacades:
    @pytest.fixture
    def pair(self):
        rng = np.random.default_rng(0)
        dat = rng.standard_normal((45, 2))
        return Adjacency(dat[:, 0]), Adjacency(dat[:, 1])

    @pytest.fixture
    def stack(self):
        rng = np.random.default_rng(1)
        return Adjacency(rng.standard_normal((6, 10)), matrix_type="distance_flat")

    def test_similarity_silent_by_default_bar_when_asked(self, pair):
        x, y = pair
        kwargs = {"n_permute": 20, "n_jobs": 1, "random_state": 0}
        assert _stderr_of(lambda: x.similarity(y, **kwargs)) == ""
        assert _stderr_of(lambda: x.similarity(y, progress_bar=True, **kwargs)) != ""

    def test_ttest_silent_by_default_bar_when_asked(self, stack):
        kwargs = {"permutation": True, "n_permute": 20, "n_jobs": 1, "random_state": 0}
        assert _stderr_of(lambda: stack.ttest(**kwargs)) == ""
        assert _stderr_of(lambda: stack.ttest(progress_bar=True, **kwargs)) != ""

    def test_bootstrap_silent_by_default_bar_when_asked(self, stack):
        kwargs = {"n_samples": 20, "n_jobs": 1, "random_state": 0}
        assert _stderr_of(lambda: stack.bootstrap("mean", **kwargs)) == ""
        assert (
            _stderr_of(lambda: stack.bootstrap("mean", progress_bar=True, **kwargs))
            != ""
        )


class TestBrainDataBootstrapFacade:
    def test_silent_by_default_bar_when_asked(self, minimal_brain_data):
        kwargs = {"n_samples": 20, "n_jobs": 1, "random_state": 0}
        assert _stderr_of(lambda: minimal_brain_data.bootstrap("mean", **kwargs)) == ""
        assert (
            _stderr_of(
                lambda: minimal_brain_data.bootstrap(
                    "mean", progress_bar=True, **kwargs
                )
            )
            != ""
        )
