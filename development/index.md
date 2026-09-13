---
title: Architecture & Internals
description: How nltools is put together — the functional core, the class facades, and the algorithm substrate.
---

# Architecture & Internals

This section is the design reference for **contributors and maintainers** (and for AI
coding assistants working in the repo). It documents *how* nltools is built and *why*
— the invariants that keep the codebase coherent. For *what the public API does*, see
the reference: the [data classes](../docs/api/data/brain_data.md) and the namespace
pages, starting at [`nltools`](../docs/api/nltools.md); for release recovery scope, see the [recovery inventory](recovery-plan.md) and the [release verification matrix](release-verification.md).

## Functional core, imperative shell

nltools follows one organizing principle: **classes are facades and glue — all real
logic lives in pure functions.**

| Layer | Role | Where |
|---|---|---|
| **Imperative shell** | Three data classes that hold state and delegate. Each is a *facade over a submodule package* (io, modeling, plotting, …). | `nltools/data/{braindata,adjacency,designmatrix}/` |
| **Functional core** | Pure functions — the actual computation. Containers in, containers out. | `nltools/algorithms/` (`corrections`, `outliers`, `signal`, `similarity`, `regression`, `neighborhoods`, …), `utils` (stack levels, warning categories, optional imports, progress bars), `cross_validation`, `mask` |
| **Algorithm substrate** | Heavy numerical machinery with its own backend/parallel story. | `nltools/algorithms/{alignment,inference}/` |

Nine namespaces are user-facing — `nltools`, `nltools.data`, `nltools.algorithms`,
`nltools.io`, `nltools.datasets`, `nltools.mask`, `nltools.cross_validation`,
`nltools.plotting` and `nltools.utils` — and each one's `__all__` *is* its surface.
Every other module is internal and carries no `__all__`; an unprefixed name in an
internal module is not user-facing until `0nm2` lands the prefixes.

The three facades and their submodules:

- **`BrainData`** — `io` · `analysis` · `modeling` · `prediction` · `bootstrap` ·
  `plotting` · `viewer` · `validation`
- **`Adjacency`** — `io` · `modeling` · `stats` · `plotting`
- **`DesignMatrix`** — `append` · `transforms` · `regressors` · `diagnostics` · `io` · `plotting`

Three modules sit at `nltools/data/` level because more than one class needs them:
`ownership` (buffer-owning copies), `validation` (X/Y frame ingress) and `combine`
(`concatenate`).

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
- **One GPU execution layer, run-or-raise.** GPU execution means Himalaya ridge
  fitting (`_Ridge(device='gpu')`, `BrainData.fit(ridge_device='gpu')`) and the
  ridge bootstrap (`BrainData.bootstrap(device='gpu')`); nltools ships no GPU
  implementation of its own, and permutation, ISC and alignment run on CPU
  workers. Memory budgets, batch sizing, and OOM recovery for those ridge paths
  live only in `algorithms.backends` (`_device_memory_budget`, `_auto_batch_size`,
  `_compute_oom_safe`); an algorithm supplies its per-item working-set estimate and
  never its own budget math (pinned by a source-scan test in `test_backends.py`).
  A `None` memory budget — the default everywhere — means "measure the device";
  when sizing batches, a measured budget is capped at a saturation ceiling
  (`BATCH_WORKING_SET_CEILING_GB`, 8 GB) because larger working sets add
  allocation cost without throughput gain, while an explicit budget is always used
  verbatim. An explicit `device='gpu'` either runs on the GPU or raises; there is
  no silent CPU fallback and no `'auto'` value.
- **Generated column names live in the reserved `.nl_` namespace.** Any column nltools
  invents rather than the user — polynomial drift (`.nl_poly_0`), DCT cosines
  (`.nl_cosine_1`), spike indicators (`.nl_global_spike1`), and the run-separated
  variants a multi-run append produces (`.nl_r0_poly_0`) — is built with
  `nltools.data.designmatrix.utils._reserved_name()` / `_run_separated_name()`. The
  `DesignMatrix` package is the only writer of the namespace: code elsewhere that
  generates columns (`find_spikes`) names them plainly and hands the frame to
  `designmatrix.utils._design_from_generated`, which applies the prefix. Code that
  needs to recognize nltools' own columns tests the prefix (`_is_reserved_name`,
  `_parse_run_separated`, or a domain predicate built on them like
  `designmatrix.utils._is_generated_intercept`) and **never** pattern-matches
  user-controlled names — no underscore counts, no substring tests. Users may then name
  their regressors anything without colliding with the machinery.

### Canonical API vocabulary {#canonical-api-vocabulary}

The three facades share one kwarg vocabulary (v0.6.0). The machine-readable source of
truth is [`docs/_data/api-vocabulary.yml`](https://github.com/cosanlab/nltools/blob/main/docs/_data/api-vocabulary.yml),
which also carries the enforcement rules `scripts/check_api_vocabulary.py` checks every
public signature against in CI. The table below is rendered from it:

<!-- AUTOGEN:api-vocabulary:index-table — generated from docs/_data/api-vocabulary.yml by scripts/build_api_vocabulary.py; run `uv run poe docs-generate` to update, do not edit by hand -->
| Concept | Canonical kwarg |
|---|---|
| Second operand of a data-class method | `data` — the object a data class is combined with or compared against: `BrainData.append`, `Adjacency.append`, `DesignMatrix.append`, `BrainData.similarity` and `Adjacency.similarity` all name it `data` and take it as their only positional parameter, with everything after it keyword-only |
| Algorithm / variant choice | `method` — on `BrainData.standardize` and `DesignMatrix.standardize` it is one closed set: `*, method: str = 'center'` (<code>'center' &#124; 'zscore'</code>) |
| Decoding estimator (MVPA) | <code>estimator: str &#124; BaseEstimator = 'linear_svc'</code> on `BrainData.predict` — a built-in shortcut name or any sklearn estimator / `Pipeline`. It names an sklearn object, so it is distinct from `method=`, which selects an algorithm variant |
| Shortcut estimator options | <code>estimator_kwargs: dict &#124; None = None</code> on `BrainData.predict` — forwarded to the shortcut's sklearn constructor, merged over the shortcut's own defaults so a supplied key wins; a `ValueError` when `estimator` is an object, which is used exactly as supplied |
| Spatial scale | `spatial_scale` (<code>'whole_brain' &#124; 'roi' &#124; 'searchlight'</code>) |
| Distance / similarity metric | `metric` |
| Convolution kernel | <code>kernel: str &#124; np.ndarray = 'glover'</code> on `DesignMatrix.convolve` (v0.5.1 spelled it `conv_func`) — an HRF model name nilearn computes (<code>'glover' &#124; 'glover_time' &#124; 'glover_dispersion' &#124; 'spm' &#124; 'spm_time' &#124; 'spm_dispersion'</code>) or a caller-supplied array, 1-D for one kernel and 2-D for several. `DesignMatrix(events_file, hrf_model=)` takes the same six names plus `None` for raw boxcars |
| Central tendency | `summary` (<code>'mean' &#124; 'median'</code>) |
| ISC summary statistic | `summary_statistic` (<code>'pairwise' &#124; 'leave-one-out'</code>) on the ISC entry points — which cross-subject comparison is summarized, distinct from `summary`, which is the `'mean'`/`'median'` central tendency applied to it |
| Cross-validation spec | `cv` (<code>int &#124;</code> splitter <code>&#124; None</code>) on `BrainData.predict` — `None` is five folds and an int that many, both an unshuffled stratified K-fold (`StratifiedGroupKFold` when `groups=` is given); the `'loo'`/`'logo'`/`'loso'`/`'loro'` names are gone everywhere. The grouping lives in `groups=` |
| Subject-level parallelism | `n_jobs: int = -1` |
| GPU / CPU selection | `device: str = "cpu"` on the ridge entry points — `BrainData.fit(ridge_device=)`, `BrainData.bootstrap`, and the internal ridge estimator they drive — the only paths with a GPU implementation. Run-or-raise: explicit `'gpu'` either runs on the GPU or raises, and there is no `'auto'` |
| Alignment refinement count | `n_iter` on `_SRM` and `_DetSRM` — EM iterations or coordinate-descent iterations; everywhere else `n_iter` is a banned alias for `n_permute`/`n_samples`/`search_iterations` |
| Working-memory budget | <code>memory_budget_gb: float &#124; None = None</code> — device-neutral working-memory budget for internal batching; `None` measures the selected device with headroom |
| Progress indicator | `progress_bar: bool = False` |
| Permutation count | `n_permute` — including the `Adjacency` label-distance and silhouette plots (`plot_label_distance`, `plot_between_label_distance`, `plot_silhouette`), where it pairs with `permutation_test` |
| Permutation test toggle | `permutation_test: bool` on the `Adjacency` label-distance and silhouette plots — pairs with `n_permute`. It defaults to `False` on `Adjacency.plot_label_distance` and `True` on `Adjacency.plot_between_label_distance` and `Adjacency.plot_silhouette`, matching each figure's v0.5.1 behaviour |
| Bootstrap sample count | `n_samples` |
| Bootstrap statistic | `statistic` on `bootstrap` — a closed set of eight names (`'mean'`, `'median'`, `'std'`, `'sum'`, `'min'`, `'max'`, `'weights'`, `'predict'`); no callables and no dynamic dispatch to other methods |
| Interval confidence level | `confidence_level: float = 0.95` — one level in `(0, 1)`, not a `percentiles` pair; the reported bounds are the central percentile interval, elementwise marginal |
| Retain resampled draws | `return_samples: bool = False` on `bootstrap` — keeps every replicate (bootstrap axis first); it changes retention only, never the interval |
| Tail of test | `tail` (<code>2 &#124; 'two' &#124; 1 &#124; 'one'</code>; direction fixed by the test, never the data) |
| Threshold pair | `lower`, `upper`, `binarize` (+ `threshold` where bidirectional) |
| Draw result figures | `plot: bool = False` on `BrainData.predict` — draws the cross-validated regression scatter, or the binary-classification ROC and margin/probability figures, plus the weight map, as a side effect; the returned `Predict` is unchanged. `Adjacency.similarity(plot=)` carries the same meaning |
| Display autoscaling | `autoscale: bool = True` (viewer display window; `False` = raw magnitude range) |
| Display symmetry | <code>symmetric: bool &#124; 'auto' = 'auto'</code> (viewer positive/negative limbs) |
| Plot axis | <code>ax: matplotlib.axes.Axes &#124; None = None</code> on the data-class plotters (`BrainData.plot`, `Adjacency.plot`, `DesignMatrix.plot`) and the Adjacency helper plots, following matplotlib and seaborn |
| Plotted panel cap | `limit: int = 3` on `BrainData.plot` and `Adjacency.plot` — how many images or matrices of a stack are rendered; keyword-only on both |
| Diagonal flag | `include_diag: bool` |
| Radius (mm) | `radius: float` in millimeters everywhere, following nilearn's `SearchLight` and `NiftiSpheresMasker` — `10.0` on the searchlight entry points (`BrainData.predict`, `BrainData.distance`, `compute_searchlight_neighborhoods`), and the same millimeter unit for `create_sphere` and `Simulator` geometry, converted to voxels through the image affine |
| GLM-specific fit option | `glm_*` on `BrainData.fit` (`glm_noise_model`, `glm_bins`, `glm_n_jobs`) — a non-default one under `model='ridge'` raises `ValueError`; `random_state` keeps its bare name because both estimators use it |
| Ridge-specific fit option | `ridge_*` on `BrainData.fit` (`ridge_alpha`, `ridge_cv`, `ridge_search_iterations`, `ridge_dirichlet_concentration`, `ridge_device`, `ridge_memory_budget_gb`, `ridge_per_target_alpha`, `ridge_prefer_conservative_alpha`, `ridge_progress_bar`) — each maps onto the identically-named argument of the internal ridge estimator behind `model='ridge'`, and a non-default one under `model='glm'` raises `ValueError` |
| Contrast inference toggle | `inference: bool = False` on `compute_contrasts` — the effect alone by default (what a second-level model consumes); `True` returns the full `ContrastResult` |
<!-- /AUTOGEN:api-vocabulary:index-table -->

## The internals pages

- **[Ridge internals](ridge-internals.md)** — how `nltools.models._Ridge` adapts the
  Himalaya solvers: name translation, device and memory policy, and fitted state.
- **[Inference internals](inference-internals.md)** — permutation and bootstrap testing:
  the algorithms, deterministic cross-backend RNG, p-value calculation, and numerical
  stability.

## Deferred 0.6.1 design

`BrainCollection` and its exclusive execution, persistence, and prediction support
are deferred to 0.6.1. The [collection specification](specs/braincollection.md) and
[execution design](execution-model.md) preserve that work. Shared HDF5 persistence,
searchlight caching, estimators, alignment, and inference remain in 0.6.0.
