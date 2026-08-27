---
title: Architecture & Internals
description: How nltools is put together — the functional core, the class facades, and the algorithm substrate.
---

# Architecture & Internals

This section is the design reference for **contributors and maintainers** (and for AI
coding assistants working in the repo). It documents *how* nltools is built and *why*
— the invariants that keep the codebase coherent. For *what the public API does*, see
the [API reference](../api/data/brain_data.md); for a visual, interactive walkthrough,
see the [Design Tour](../design-tour.md).

## Functional core, imperative shell

nltools follows one organizing principle: **classes are facades and glue — all real
logic lives in pure functions.**

| Layer | Role | Where |
|---|---|---|
| **Imperative shell** | Four data classes that hold state and delegate. Each is a *facade over a submodule package* (io, modeling, plotting, …). | `nltools/data/{braindata,adjacency,designmatrix,collection}/` |
| **Functional core** | Pure functions — the actual computation. Containers in, containers out. Every user-facing function is importable flat from `nltools.algorithms`. | `nltools/algorithms/` (`corrections`, `outliers`, `signal`, `similarity`, `regression`, …), `utils`, `cross_validation`, `mask` |
| **Algorithm substrate** | Heavy numerical machinery with its own backend/parallel story. | `nltools/algorithms/{alignment,inference,ridge}/` |

The four facades and their submodules:

- **`BrainData`** — `io` · `analysis` · `modeling` · `prediction` · `bootstrap` ·
  `neighborhoods` · `cache` · `plotting` · `viewer` · `validation`
- **`Adjacency`** — `io` · `modeling` · `stats` · `spatial` · `plotting`
- **`DesignMatrix`** — `append` · `transforms` · `regressors` · `diagnostics` · `io` · `plotting`
- **`BrainCollection`** — `core` · `execution` · `inference` · `io` · `pipeline`

### Design rules

- **Pure functions first.** Classes compose and delegate to them, never the reverse.
- **Immutable state** in frozen dataclasses where it makes sense; prefer modern Python
  (type hints, `@dataclass(frozen=True)`, `|` unions).
- **Single source of truth.** Extract shared logic into helpers and import them; don't
  duplicate.
- **No underscore-prefixed module names** (`validation.py`, not `_validation.py`).
  Leading underscores are fine for internal functions/methods, just not filenames.
- **Facade translation at the boundary.** Internal algorithm-layer APIs may keep legacy
  parameter names; the class facade translates to the [canonical vocabulary](#canonical-api-vocabulary).
- **One GPU execution layer, run-or-raise.** Memory budgets, batch sizing, and OOM
  recovery live only in `algorithms.backends` (`device_memory_budget`,
  `auto_batch_size`, `compute_oom_safe`, `auto_n_jobs_for_arrays`); an algorithm
  supplies its per-item working-set estimate and never its own budget math (pinned by
  a source-scan test in `test_backends.py`). `max_gpu_memory_gb=None` — the default
  everywhere — means "measure the device". An explicit `device='gpu'` /
  `parallel='gpu'` either runs on the GPU or raises; `'auto'` is the one documented
  graceful-fallback path.
- **Generated column names live in the reserved `.nl_` namespace.** Any column nltools
  invents rather than the user — polynomial drift (`.nl_poly_0`), DCT cosines
  (`.nl_cosine_1`), spike indicators (`.nl_global_spike1`), and the run-separated
  variants a multi-run append produces (`.nl_r0_poly_0`) — is built with
  `nltools.utils.reserved_name()` / `run_separated_name()`. Code that needs to
  recognize nltools' own columns tests the prefix (`is_reserved_name`,
  `parse_run_separated`, or a domain predicate built on them like
  `designmatrix.utils.is_generated_intercept`) and **never** pattern-matches
  user-controlled names — no underscore counts, no substring tests. Users may then name
  their regressors anything without colliding with the machinery.

### Canonical API vocabulary

The four facades share one kwarg vocabulary (v0.6.0). The machine-readable source of
truth is [`docs/_data/api-vocabulary.yml`](https://github.com/cosanlab/nltools/blob/main/docs/_data/api-vocabulary.yml),
which also carries the enforcement rules `scripts/check_api_vocabulary.py` checks every
public signature against in CI. The table below is rendered from it:

<!-- AUTOGEN:api-vocabulary:index-table — generated from docs/_data/api-vocabulary.yml by scripts/build_api_vocabulary.py; run `uv run poe docs-generate` to update, do not edit by hand -->
| Concept | Canonical kwarg |
|---|---|
| Algorithm / variant choice | `method` |
| Spatial scale | `spatial_scale` (`'whole_brain' \| 'roi' \| 'searchlight'`) |
| Distance / similarity metric | `metric` |
| Central tendency | `summary` (`'mean' \| 'median'`) |
| Subject-level parallelism | `n_jobs: int = -1` |
| GPU / CPU selection | `device: str = "cpu"` — run-or-raise: explicit `'gpu'` never silently degrades to CPU; `'auto'` is the one graceful-fallback path |
| Backend (ridge/alignment internals) | `parallel: None \| 'cpu' \| 'gpu'` (the inference engine uses `device` as of v0.6.0) |
| Progress indicator | `progress_bar: bool = False` |
| Permutation count | `n_permute` |
| Bootstrap sample count | `n_samples` |
| Tail of test | `tail` (`2 \| 'two' \| 1 \| 'one'`; direction fixed by the test, never the data) |
| Threshold pair | `lower`, `upper`, `binarize` (+ `threshold` where bidirectional) |
| Diagonal flag | `include_diag: bool` |
| Radius (mm) | `radius_mm: float` |
<!-- /AUTOGEN:api-vocabulary:index-table -->

## The internals pages

- **[Execution model](execution-model.md)** — how `BrainCollection` runs per-subject
  work in parallel: path-backed-by-default caching, the `cache=` knob, HDF5 fit bundles,
  the pickling contract, and parallel write safety.
- **[Ridge internals](ridge-internals.md)** — the six mathematical tricks behind the
  GPU-accelerated ridge solver, and the backend abstraction.
- **[Inference internals](inference-internals.md)** — permutation and bootstrap testing:
  the algorithms, deterministic cross-backend RNG, p-value calculation, and numerical
  stability.
