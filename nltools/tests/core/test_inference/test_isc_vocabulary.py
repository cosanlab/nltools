"""Canonical kwarg vocabulary for the ISC family (v0.6.0 follow-up to #474).

Historically the isc engine used `metric` for the mean/median central-tendency
choice and `sim_metric` for the actual similarity metric, while the public
`isc()` wrapper spoke the canonical names (`summary`, `metric`) and translated.
The consolidation makes the engine itself canonical:

- `summary='median' | 'mean'` — central tendency of the ISC distribution
- `metric='correlation' | ...` — the similarity metric (canonical table)
- results expose `null_dist` (like every other inference result), not
  `null_distribution`
- the `isc` / `isc_group` wrappers expose and forward `progress_bar`
"""

import contextlib
import inspect
import io

import numpy as np
import pytest

from nltools.algorithms import isc, isc_group
from nltools.algorithms.inference import (
    isc_group_permutation_test,
    isc_permutation_test,
)

CANONICAL_SIGNATURES = [
    isc_permutation_test,
    isc_group_permutation_test,
    isc,
    isc_group,
]


@pytest.fixture
def subjects_data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((30, 8))


@pytest.fixture
def group_data():
    rng = np.random.default_rng(1)
    return rng.standard_normal((30, 6)), rng.standard_normal((30, 6))


@pytest.mark.parametrize("func", CANONICAL_SIGNATURES, ids=lambda f: f.__name__)
def test_summary_and_metric_are_canonical(func):
    params = inspect.signature(func).parameters
    assert "summary" in params, f"{func.__name__} lacks summary="
    assert params["summary"].default == "median"
    assert "sim_metric" not in params, f"{func.__name__} still exposes sim_metric="
    metric = params.get("metric")
    if func is isc_group or func is isc_group_permutation_test:
        # isc_group computes a pairwise-ISC difference; a similarity-metric knob
        # only exists where the engine exposes one.
        if metric is None:
            return
    assert metric is not None, f"{func.__name__} lacks metric="
    assert metric.default == "correlation", (
        f"{func.__name__}: metric= must be the similarity metric "
        f"(got default {metric.default!r} — the mean/median choice is summary=)"
    )


def test_braincollection_isc_uses_summary_kwarg():
    from nltools.data import BrainCollection

    for method in (BrainCollection.isc, BrainCollection.isc_test):
        params = inspect.signature(method).parameters
        assert "summary" in params, f"{method.__qualname__} lacks summary="
        assert params["summary"].default == "median"
        assert "metric" not in params, (
            f"{method.__qualname__}: the mean/median aggregation choice must be "
            "summary= — metric is reserved for similarity metrics"
        )


@pytest.mark.parametrize("func", [isc, isc_group], ids=lambda f: f.__name__)
def test_wrappers_expose_progress_bar(func):
    param = inspect.signature(func).parameters.get("progress_bar")
    assert param is not None, f"{func.__name__} lacks progress_bar="
    assert param.default is False
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


def test_isc_returns_null_dist_key(subjects_data):
    result = isc(subjects_data, n_samples=20, return_null=True, random_state=0)
    assert "null_dist" in result
    assert "null_distribution" not in result


def test_isc_group_returns_null_dist_key(group_data):
    g1, g2 = group_data
    result = isc_group(g1, g2, n_samples=20, return_null=True, random_state=0)
    assert "null_dist" in result
    assert "null_distribution" not in result


def test_isc_summary_kwarg_selects_central_tendency(subjects_data):
    r_median = isc(subjects_data, n_samples=20, summary="median", random_state=0)
    r_mean = isc(subjects_data, n_samples=20, summary="mean", random_state=0)
    assert r_median["isc"] != r_mean["isc"]
    with pytest.raises(ValueError, match="summary"):
        isc(subjects_data, n_samples=20, summary="mode", random_state=0)


def test_isc_group_summary_kwarg_validated(group_data):
    g1, g2 = group_data
    with pytest.raises(ValueError, match="summary"):
        isc_group(g1, g2, n_samples=20, summary="mode", random_state=0)


def test_engine_summary_kwarg_validated(subjects_data):
    with pytest.raises(ValueError, match="summary"):
        isc_permutation_test(subjects_data, n_permute=20, summary="mode")


class TestWrapperProgressBarThreading:
    def _stderr_of(self, fn):
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            fn()
        return buf.getvalue()

    def test_isc_silent_by_default_bar_when_asked(self, subjects_data):
        kwargs = {"n_samples": 20, "random_state": 0, "n_jobs": 1}
        assert self._stderr_of(lambda: isc(subjects_data, **kwargs)) == ""
        assert (
            self._stderr_of(lambda: isc(subjects_data, progress_bar=True, **kwargs))
            != ""
        )

    def test_isc_group_silent_by_default_bar_when_asked(self, group_data):
        g1, g2 = group_data
        kwargs = {"n_samples": 20, "random_state": 0, "n_jobs": 1}
        assert self._stderr_of(lambda: isc_group(g1, g2, **kwargs)) == ""
        assert (
            self._stderr_of(lambda: isc_group(g1, g2, progress_bar=True, **kwargs))
            != ""
        )
