"""Speed + memory benchmark harness for nltools.

Every benchmark runs an operation ``reps`` times after ``warmup`` untimed runs,
recording per-rep wall-clock and peak memory. Memory is captured two ways so
each workload's real cost is visible:

- **peak_rss_mb** — process resident-set high-water mark during the op, sampled
  on a background thread (true peak, catches transient balloons). The headline
  for array/tensor workloads. Note: joblib/loky workers are separate processes,
  so their memory is not in the main-process RSS — this metric is most meaningful
  for single-process ops (e.g. `BrainCollection` lazy vs in-memory).
- **peak_device_mb** — GPU allocator peak (CUDA: exact via
  ``max_memory_allocated``; MPS: allocated-delta proxy, no true-peak API).

``tracemalloc`` is intentionally not used: it traps every allocation (inflating
allocation-heavy wall-clock ~5x) and its Python-object peak misses numpy/torch C
buffers anyway.

Results collect into ``BenchResult`` rows, serialize to parquet via polars, and
travel with an ``env.json`` provenance sidecar (versions, host, device).

Design note: this module has no nltools imports — it benchmarks arbitrary
callables, so the harness stays testable in isolation from the library.
"""

from __future__ import annotations

import gc
import json
import platform
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl


def _torch():
    try:
        import torch

        return torch
    except ModuleNotFoundError:
        return None


def _psutil():
    try:
        import psutil

        return psutil
    except ModuleNotFoundError:
        return None


class _RSSSampler(threading.Thread):
    """Poll process RSS on a background thread to capture a true peak."""

    def __init__(self, proc, interval: float = 0.005):
        super().__init__(daemon=True)
        self._proc = proc
        self._interval = interval
        self._stop_event = threading.Event()  # not `_stop`: Thread uses that name
        self.peak = 0

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.peak = max(self.peak, self._proc.memory_info().rss)
            except Exception:  # noqa: BLE001 — process introspection is best-effort
                pass
            self._stop_event.wait(self._interval)

    def stop(self) -> int:
        self._stop_event.set()
        self.join()
        return self.peak


class _DeviceProbe:
    """Reset + read the GPU allocator peak for the active device, if any."""

    def __init__(self, device: str):
        self.torch = _torch()
        self.kind: str | None = None
        self._mps_start = 0
        if self.torch is None:
            return
        if device.startswith("cuda") and self.torch.cuda.is_available():
            self.kind = "cuda"
        elif device.startswith("mps") and self.torch.backends.mps.is_available():
            self.kind = "mps"

    def reset(self) -> None:
        if self.kind == "cuda":
            self.torch.cuda.empty_cache()
            self.torch.cuda.reset_peak_memory_stats()
        elif self.kind == "mps":
            self.torch.mps.empty_cache()
            self._mps_start = self.torch.mps.current_allocated_memory()

    def sync(self) -> None:
        if self.kind == "cuda":
            self.torch.cuda.synchronize()
        elif self.kind == "mps":
            self.torch.mps.synchronize()

    def peak_mb(self) -> float | None:
        if self.kind == "cuda":
            return self.torch.cuda.max_memory_allocated() / 1e6
        if self.kind == "mps":
            # MPS exposes no true-peak API; allocated-delta is a lower bound.
            cur = self.torch.mps.current_allocated_memory()
            return max(0, cur - self._mps_start) / 1e6
        return None


@dataclass
class BenchResult:
    """One benchmarked condition: median speed + peak memory across reps."""

    domain: str
    name: str
    device: str
    seconds: float
    peak_rss_mb: float
    peak_device_mb: float | None
    n_reps: int
    params: dict[str, Any] = field(default_factory=dict)

    def row(self) -> dict[str, Any]:
        """Flatten to a parquet-friendly row (params inlined as ``p_*``)."""
        row = {k: v for k, v in vars(self).items() if k != "params"}
        for key, val in self.params.items():
            row[f"p_{key}"] = val
        return row


def benchmark(
    fn: Callable[..., Any],
    *,
    domain: str,
    name: str,
    device: str = "cpu",
    reps: int = 3,
    warmup: int = 1,
    setup: Callable[[], Any] | None = None,
    params: dict[str, Any] | None = None,
) -> BenchResult:
    """Time and memory-profile ``fn`` over ``reps`` runs (after ``warmup``).

    Args:
        fn: Operation to benchmark. Called as ``fn(setup())`` when ``setup`` is
            given (fresh per-rep input, e.g. for write benchmarks), else ``fn()``.
        domain: Coarse group (``"ridge"``, ``"inference"``, ``"collection"``, ...).
        name: Specific condition label.
        device: ``"cpu"``, ``"mps"``, or ``"cuda"`` — selects the device probe.
        reps: Timed repetitions; ``seconds`` is their median.
        warmup: Untimed runs first (JIT/allocator warmup, import side effects).
        setup: Optional zero-arg callable producing fresh input for each run.
        params: Condition parameters, stored and inlined into the parquet row.

    Returns:
        A `BenchResult` with median wall-clock and peak-across-reps memory.
    """
    probe = _DeviceProbe(device)
    ps = _psutil()
    proc = ps.Process() if ps is not None else None

    def _invoke() -> None:
        if setup is not None:
            fn(setup())
        else:
            fn()

    for _ in range(warmup):
        _invoke()

    times: list[float] = []
    rss_peaks: list[float] = []
    dev_peaks: list[float] = []

    for _ in range(reps):
        arg = setup() if setup is not None else None
        gc.collect()
        rss_before = proc.memory_info().rss if proc is not None else 0
        sampler = _RSSSampler(proc) if proc is not None else None
        probe.reset()
        if sampler is not None:
            sampler.start()

        # Timed region: RSS sampling + device probe only. `tracemalloc` is
        # deliberately NOT active here — it traps every allocation and inflates
        # allocation-heavy numpy/joblib wall-clock ~5x, and its Python-object
        # peak misses numpy/torch C buffers anyway. Peak RSS is the real signal.
        t0 = time.perf_counter()
        fn(arg) if setup is not None else fn()
        probe.sync()
        dt = time.perf_counter() - t0

        rss_peak = sampler.stop() if sampler is not None else rss_before
        dev = probe.peak_mb()

        times.append(dt)
        rss_peaks.append(max(0, rss_peak - rss_before) / 1e6)
        if dev is not None:
            dev_peaks.append(dev)

    return BenchResult(
        domain=domain,
        name=name,
        device=device,
        seconds=statistics.median(times),
        peak_rss_mb=max(rss_peaks),
        peak_device_mb=(max(dev_peaks) if dev_peaks else None),
        n_reps=reps,
        params=params or {},
    )


def env_metadata(device: str = "cpu") -> dict[str, Any]:
    """Capture reproducibility provenance for a benchmark run."""
    import numpy as np

    torch = _torch()
    meta: dict[str, Any] = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "device": device,
    }
    try:
        from importlib.metadata import version

        meta["nltools"] = version("nltools")
    except Exception:  # noqa: BLE001
        meta["nltools"] = "unknown"
    if torch is not None:
        meta["torch"] = torch.__version__
        meta["cuda_available"] = torch.cuda.is_available()
        meta["mps_available"] = torch.backends.mps.is_available()
    return meta


def write_results(
    results: list[BenchResult],
    out_dir: Path | str,
    *,
    device: str = "cpu",
    tag: str | None = None,
) -> tuple[Path, Path]:
    """Write results to ``<out_dir>/<host>[-<tag>].parquet`` + an ``env.json``.

    Returns the ``(parquet_path, env_path)`` written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    host = platform.node().split(".")[0] or "unknown"
    stem = f"{host}-{tag}" if tag else host

    df = pl.DataFrame([r.row() for r in results])
    parquet_path = out_dir / f"{stem}.parquet"
    df.write_parquet(parquet_path)

    env_path = out_dir / f"{stem}.env.json"
    env_path.write_text(json.dumps(env_metadata(device), indent=2))
    return parquet_path, env_path
