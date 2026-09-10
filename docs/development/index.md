---
title: Architecture & Internals
description: How nltools is put together — the functional core, the class facades, and the algorithm substrate.
---

# Architecture & Internals

This section is the design reference for **contributors and maintainers** (and for AI
coding assistants working in the repo). It documents *how* nltools is built and *why*
— the invariants that keep the codebase coherent. For *what the public API does*, see
the reference: the [data classes](../api/data/brain_data.md), the
[functions by task](../api/tasks/loading.md), and the
[`nltools.algorithms` A–Z index](../api/algorithms.md); for release recovery scope, see the [recovery inventory](recovery-plan.md) and the [release verification matrix](release-verification.md).

## Functional core, imperative shell

nltools follows one organizing principle: **classes are facades and glue — all real
logic lives in pure functions.**

| Layer | Role | Where |
|---|---|---|
| **Imperative shell** | Three data classes that hold state and delegate. Each is a *facade over a submodule package* (io, modeling, plotting, …). | `nltools/data/{braindata,adjacency,designmatrix}/` |
| **Functional core** | Pure functions — the actual computation. Containers in, containers out. Every user-facing function is importable flat from `nltools.algorithms`. | `nltools/algorithms/` (`corrections`, `outliers`, `signal`, `similarity`, `regression`, …), `utils`, `cross_validation`, `mask` |
| **Algorithm substrate** | Heavy numerical machinery with its own backend/parallel story. | `nltools/algorithms/{alignment,inference}/` |

The three facades and their submodules:

- **`BrainData`** — `io` · `analysis` · `modeling` · `prediction` · `bootstrap` ·
  `neighborhoods` · `cache` · `plotting` · `viewer` · `validation`
- **`Adjacency`** — `io` · `modeling` · `stats` · `plotting`
- **`DesignMatrix`** — `append` · `transforms` · `regressors` · `diagnostics` · `io` · `plotting`

The [DesignMatrix contract](specs/designmatrix.md) defines direct Polars method
access, result ownership, metadata propagation and persistence.
The [Adjacency contract](specs/adjacency.md) defines matrix shapes, selection,
ownership, statistical result dimensions and explicit spatial projection.
The [one-sample t-test contract](specs/ttest.md) defines shared dictionary
results and inference semantics for BrainData and Adjacency.

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
  everywhere — means "measure the device"; when sizing batches, a measured budget is
  capped at a saturation ceiling (`BATCH_WORKING_SET_CEILING_GB`, 8 GB) because
  larger working sets add allocation cost without throughput gain, while an explicit
  `max_gpu_memory_gb` is always used verbatim. An explicit `device='gpu'` /
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

(canonical-api-vocabulary)=
### Canonical API vocabulary

The three facades share one kwarg vocabulary (v0.6.0). The machine-readable source of
truth is [`docs/_data/api-vocabulary.yml`](https://github.com/cosanlab/nltools/blob/main/docs/_data/api-vocabulary.yml),
which also carries the enforcement rules `scripts/check_api_vocabulary.py` checks every
public signature against in CI. The table below is rendered from it:

<!-- AUTOGEN:api-vocabulary:index-table — generated from docs/_data/api-vocabulary.yml by scripts/build_api_vocabulary.py; run `uv run poe docs-generate` to update, do not edit by hand -->
| Concept | Canonical kwarg |
|---|---|
| Algorithm / variant choice | `method` |
| Decoding estimator (MVPA) | `estimator: str \| BaseEstimator = 'linear_svc'` on `BrainData.predict` — a built-in shortcut name or any sklearn estimator / `Pipeline`. It names an sklearn object, so it is distinct from `method=`, which selects an algorithm variant |
| Spatial scale | `spatial_scale` (`'whole_brain' \| 'roi' \| 'searchlight'`) |
| Distance / similarity metric | `metric` |
| Central tendency | `summary` (`'mean' \| 'median'`) |
| Cross-validation spec | `cv` (`int \|` splitter `\| None`) on `BrainData.predict` — `None` is a deterministic five-fold `KFold`/`StratifiedKFold`; the `'loo'`/`'logo'` names are accepted only by `resolve_cv`, and `'loso'`/`'loro'` are gone everywhere. The grouping lives in `groups=` |
| Subject-level parallelism | `n_jobs: int = -1` |
| GPU / CPU selection | `device: str = "cpu"` — run-or-raise: explicit `'gpu'` never silently degrades to CPU; `'auto'` is the one graceful-fallback path |
| Backend (alignment internals) | `parallel: None \| 'cpu' \| 'gpu'` (the inference engine and `Ridge` use `device` as of v0.6.0) |
| Working-memory budget | `memory_budget_gb: float \| None = None` — device-neutral working-memory budget for internal batching; `None` measures the selected device with headroom |
| Progress indicator | `progress_bar: bool = False` |
| Permutation count | `n_permute` |
| Bootstrap sample count | `n_samples` |
| Tail of test | `tail` (`2 \| 'two' \| 1 \| 'one'`; direction fixed by the test, never the data) |
| Threshold pair | `lower`, `upper`, `binarize` (+ `threshold` where bidirectional) |
| Display autoscaling | `autoscale: bool = True` (viewer display window; `False` = raw magnitude range) |
| Display symmetry | `symmetric: bool \| 'auto' = 'auto'` (viewer positive/negative limbs) |
| Diagonal flag | `include_diag: bool` |
| Radius (mm) | `radius: float = 10.0` on `BrainData.predict` (millimeters, matching nilearn's searchlight); the other searchlight and surface entry points keep `radius_mm` |
| GLM-specific fit option | `glm_*` on `BrainData.fit` (`glm_noise_model`, `glm_bins`, `glm_n_jobs`) — a non-default one under `model='ridge'` raises `ValueError`; `random_state` keeps its bare name because both estimators use it |
| Ridge-specific fit option | `ridge_*` on `BrainData.fit` (`ridge_alpha`, `ridge_cv`, `ridge_search_iterations`, `ridge_dirichlet_concentration`, `ridge_device`, `ridge_memory_budget_gb`, `ridge_per_target_alpha`, `ridge_prefer_conservative_alpha`, `ridge_progress_bar`) — each maps onto the identically-named `Ridge` argument, and a non-default one under `model='glm'` raises `ValueError` |
| Contrast inference toggle | `inference: bool = False` on `compute_contrasts` — the effect alone by default (what a second-level model consumes); `True` returns the full `ContrastResult` |
<!-- /AUTOGEN:api-vocabulary:index-table -->

## The internals pages

- **[Ridge internals](ridge-internals.md)** — how `nltools.models.Ridge` adapts the
  Himalaya solvers: name translation, device and memory policy, and fitted state.
- **[Inference internals](inference-internals.md)** — permutation and bootstrap testing:
  the algorithms, deterministic cross-backend RNG, p-value calculation, and numerical
  stability.

## Deferred 0.6.1 design

`BrainCollection` and its exclusive execution, persistence, and prediction support
are deferred to 0.6.1. The [collection specification](specs/braincollection.md) and
[execution design](execution-model.md) preserve that work. Shared HDF5 persistence,
searchlight caching, estimators, alignment, and inference remain in 0.6.0.
