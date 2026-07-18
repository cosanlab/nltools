"""Quick-mode smoke tests: each domain runner returns valid, timed results.

These exercise the real nltools API paths at tiny sizes (no numbers of record).

    uv run pytest benchmarks/tests/test_benches.py -q
"""

from __future__ import annotations

import pytest

from benchmarks import (
    bench_collection,
    bench_inference,
    bench_predict,
    bench_ridge,
)
from benchmarks.harness import BenchResult


def _assert_valid(results):
    assert results, "runner returned no results"
    for r in results:
        assert isinstance(r, BenchResult)
        assert r.seconds > 0
        assert r.peak_rss_mb >= 0
        assert r.name and r.domain


@pytest.mark.parametrize(
    "runner", [bench_ridge, bench_inference, bench_predict, bench_collection]
)
def test_runner_quick(runner):
    _assert_valid(runner.run(reps=1, quick=True))
