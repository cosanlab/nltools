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
| **Functional core** | Pure functions — the actual computation. Containers in, containers out. | `stats`, `utils`, `cross_validation`, `mask` |
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

### Canonical API vocabulary

The four facades share one kwarg vocabulary (v0.6.0). The full table lives in the repo
[`CLAUDE.md`](https://github.com/cosanlab/nltools/blob/main/CLAUDE.md); the load-bearing
names:

| Concept | Canonical kwarg |
|---|---|
| Algorithm / variant choice | `method` |
| Spatial scale | `spatial_scale` (`'whole_brain' \| 'roi' \| 'searchlight'`) |
| Distance / similarity metric | `metric` |
| Subject-level parallelism | `n_jobs: int = -1` |
| GPU/CPU selection | `device: str = "cpu"` (BrainCollection) |
| Backend / parallel execution (algorithms) | `parallel: None \| 'cpu' \| 'gpu'` |
| Progress indicator | `progress_bar: bool = False` |
| Permutation count | `n_permute` |
| Bootstrap sample count | `n_samples` |

## The internals pages

- **[Execution model](execution-model.md)** — how `BrainCollection` runs per-subject
  work in parallel: path-backed-by-default caching, the `cache=` knob, HDF5 fit bundles,
  the pickling contract, and parallel write safety.
- **[Ridge internals](ridge-internals.md)** — the six mathematical tricks behind the
  GPU-accelerated ridge solver, and the backend abstraction.
- **[Inference internals](inference-internals.md)** — permutation and bootstrap testing:
  the algorithms, deterministic cross-backend RNG, p-value calculation, and numerical
  stability.
