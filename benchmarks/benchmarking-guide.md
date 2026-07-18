# Benchmarking Guide

A small harness for measuring **speed and memory** of nltools at realistic
neuroimaging scales, across the CPU (NumPy) and GPU (PyTorch MPS/CUDA) backends.
It benchmarks the user-facing API (`BrainData.fit`/`.predict`, `BrainCollection`)
alongside the algorithm-layer primitives (`ridge_cv`, permutation tests).

## Layout

```
benchmarks/
  harness.py         # timing + peak-memory measurement, BenchResult, artifact writer
  workloads.py       # synthetic data: arrays, in-memory BrainData, on-disk subjects
  bench_ridge.py     # ridge_cv + BrainData.fit(model='ridge'), CPU vs GPU
  bench_predict.py   # BrainData.predict across whole_brain / roi / searchlight
  bench_inference.py # permutation tests (one/two-sample, correlation), CPU vs GPU
  bench_collection.py# BrainCollection over N subjects: lazy vs in-memory, n_jobs
  run.py             # CLI: run domains, write results/<host>.parquet + env.json
  build_docs.py      # regenerate docs/performance.md tables from the artifact
  results/           # committed parquet artifacts + env.json provenance
  tests/             # quick-mode smoke tests (not part of the package suite)
```

## What gets measured

Every condition runs `reps` timed repetitions after a warmup, recording:

- **wall-clock** — median across reps.
- **peak RSS** — process resident-set high-water mark, sampled on a background
  thread (the headline for array/tensor workloads). Joblib/loky workers are
  separate processes, so their memory is not in the main-process RSS — this is
  most meaningful for single-process ops (e.g. `BrainCollection` lazy vs
  in-memory).
- **peak GPU** — CUDA `max_memory_allocated` (exact) or an MPS allocated-delta
  proxy (no true-peak API on MPS).

`tracemalloc` is intentionally not used — it inflates allocation-heavy wall-clock
~5x and its Python peak misses NumPy/Torch C buffers.

## Running

```bash
# All domains, realistic sizes (writes benchmarks/results/<host>.parquet):
uv run python -m benchmarks.run

# Validate the harness quickly on tiny data:
uv run python -m benchmarks.run --quick

# A subset, more reps, tagged artifact:
uv run python -m benchmarks.run --domains ridge collection --reps 5 --tag trial

# See what would run:
uv run python -m benchmarks.run --dry-run
```

The GPU legs run automatically wherever a device is available (MPS on Apple
Silicon, CUDA on an NVIDIA box); they are skipped otherwise. To generate CUDA
numbers, run the same command on a CUDA host — the artifact is named per host,
so several machines' results coexist under `results/`.

## Regenerating the docs

```bash
uv run python -m benchmarks.run          # produce/refresh results/<host>.parquet
uv run python -m benchmarks.build_docs   # splice tables into docs/performance.md
```

`build_docs` only rewrites the block between the `<!-- BENCH:START -->` /
`<!-- BENCH:END -->` markers — the surrounding guide prose is preserved.

## Adding a benchmark

Add a `bench_*.py` exposing `run(reps: int, quick: bool) -> list[BenchResult]`,
call `benchmark(fn, domain=..., name=..., device=..., params=...)` per condition,
and register the module in `run.py`'s `DOMAINS`. Add a case to
`tests/test_benches.py` so `--quick` covers it. Keep `quick=True` sizes tiny
(seconds total) — they validate wiring, not performance.

## Notes

- **MPS is float32.** The GPU backend warns and runs in single precision; CPU
  paths keep float64. Speed comparisons are like-for-like on float32 inputs.
- **Searchlight** cost is O(n_voxels) model fits — it is swept at a small voxel
  count on purpose.
- **`BrainCollection` memory** is the reason the collection benchmark exists:
  lazy/path-backed mode should keep peak RSS ~flat as N subjects grows, while
  `lazy=False` scales with N.
