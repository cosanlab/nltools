"""Smoke tests for the benchmark harness (run explicitly, not in the main suite).

uv run pytest benchmarks/tests -q
"""

from __future__ import annotations

import numpy as np
import polars as pl

from benchmarks.harness import (
    BenchResult,
    benchmark,
    env_metadata,
    write_results,
)


def test_benchmark_measures_time_and_memory():
    """A ~200MB allocation should register in wall-clock and peak RSS."""

    def alloc_and_work():
        arr = np.ones(25_000_000, dtype=np.float64)  # ~200 MB
        return float(arr.sum())

    res = benchmark(
        alloc_and_work,
        domain="smoke",
        name="alloc_200mb",
        reps=2,
        warmup=1,
        params={"mb": 200},
    )
    assert isinstance(res, BenchResult)
    assert res.n_reps == 2
    assert res.seconds > 0
    # RSS peak sampler must see the transient 200MB buffer (allow slack).
    assert res.peak_rss_mb > 50
    assert res.peak_device_mb is None  # cpu device
    assert res.params["mb"] == 200


def test_setup_is_called_per_rep():
    """`setup` provides fresh per-rep input passed to fn."""
    seen = []

    def make_input():
        return len(seen)

    def consume(x):
        seen.append(x)

    benchmark(consume, domain="smoke", name="setup", reps=3, warmup=1, setup=make_input)
    # 1 warmup + 3 timed = 4 invocations, each with a fresh setup() value.
    assert len(seen) == 4


def test_result_row_flattens_params():
    res = BenchResult(
        domain="d",
        name="n",
        device="cpu",
        seconds=1.0,
        peak_rss_mb=2.0,
        peak_device_mb=None,
        n_reps=1,
        params={"n_voxels": 50000, "backend": "numpy"},
    )
    row = res.row()
    assert row["p_n_voxels"] == 50000
    assert row["p_backend"] == "numpy"
    assert "params" not in row


def test_env_metadata_has_core_keys():
    meta = env_metadata(device="cpu")
    for key in ("host", "python", "numpy", "polars", "nltools", "device"):
        assert key in meta


def test_write_results_roundtrips(tmp_path):
    results = [
        benchmark(lambda: sum(range(1000)), domain="smoke", name="a", reps=1, warmup=0),
        benchmark(lambda: sum(range(2000)), domain="smoke", name="b", reps=1, warmup=0),
    ]
    parquet_path, env_path = write_results(results, tmp_path, device="cpu")
    assert parquet_path.exists()
    assert env_path.exists()
    df = pl.read_parquet(parquet_path)
    assert df.height == 2
    assert set(df["name"]) == {"a", "b"}
