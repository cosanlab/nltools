---
title: Ridge internals
description: How nltools.models._Ridge adapts the Himalaya solvers — argument translation, device and memory policy, and fitted-state normalization.
---

# Ridge internals

`nltools.models._Ridge` is an adapter. All of the ridge numerics —
decomposition, the cross-validation loss, alpha selection, the banded Dirichlet
search, and coefficient refitting — come from
[Himalaya](https://github.com/gallantlab/himalaya) (Dupré la Tour et al., 2022),
pinned at `>=0.4.11,<0.5`. `nltools` owns everything around them: argument names
and validation, named feature-space alignment, device and memory policy,
fitted-state normalization, and bootstrap orchestration.

There is no `nltools.algorithms.ridge`. The in-house SVD solvers that lived
there through 0.6.0 development were removed once Himalaya became a dependency;
duplicating a tested numerical library was the wrong trade. The binding
contract is [`specs/ridge.md`](specs/ridge.md); this page explains how the
adapter meets it.

Code: `nltools/models/ridge.py`, with the device and memory layer in
`nltools/algorithms/backends.py`.

## What each side owns

| Concern | Owner |
|---|---|
| SVD, resolution matrices, alpha and target batching | Himalaya |
| Negative-MSE fold scores, conservative rule, tie-breaking | Himalaya |
| Dirichlet search over feature-space weights | Himalaya |
| Coefficient refitting at fixed hyperparameters | Himalaya (`solve_ridge_svd`) |
| Public argument names, defaults, and validation | nltools |
| Named feature spaces and prediction-time alignment | nltools |
| Candidate-weight validation and the underflow floor | nltools |
| Device resolution, memory budget, batch sizes | nltools (`backends.py`) |
| Fitted-state shapes, dtypes, and CPU normalization | nltools |

`nltools` must not copy Himalaya's solvers or restate its numerical tests. The
parity tests in `nltools/tests/models/test_ridge.py` compare the adapter's
output against direct Himalaya calls; they do not re-derive the math.

## Name translation

The public keywords are the nltools vocabulary; Himalaya's names stay internal.

| `_Ridge` keyword | Himalaya argument |
|---|---|
| `per_target_alpha` | `local_alpha` |
| `prefer_conservative_alpha` | `conservative` |
| `search_iterations` | `n_iter` |
| `dirichlet_concentration` | `concentration` |
| `memory_budget_gb` | derived `n_targets_batch`, `n_targets_batch_refit`, `n_alphas_batch` |

The adapter always passes `fit_intercept=False` and
`score_func=l2_neg_loss`. It exposes no scoring callback, no `solver_params`,
and no manual batch sizes. The removed spellings (`alphas`, `n_iter`,
`concentration`, `local_alpha`, `conservative`, `fit_intercept`, `backend`,
`solver_params`, `max_gpu_memory_gb`) raise `TypeError`; nothing is aliased or
translated for the caller.

## Which solver runs

| `X` | `alpha` | `cv` | Himalaya entry point |
|---|---|---|---|
| 2-D array | scalar | `None` | `solve_ridge_svd` |
| 2-D array | sequence | int or splitter | `solve_ridge_cv_svd` |
| name → 2-D mapping | sequence | int or splitter | `solve_group_ridge_random_search` |

`search_iterations` and `dirichlet_concentration` are the banded-only
arguments: a non-default value of either raises during an ordinary fit rather
than becoming a silent no-op. `random_state` is not one of them — ordinary Ridge
accepts and ignores it, because `BrainData.fit` forwards a single unprefixed
`random_state` to whichever estimator it builds.

A scalar `alpha` with a `cv`, a sequence without one, `alpha="auto"`, and a
scalar alpha for a banded fit are all rejected before any decomposition. `cv`
as an integer becomes an unshuffled `KFold`; a single-use split generator is
rejected because fitting traverses the splits more than once.

## Banded search: what nltools does before Himalaya

Himalaya's `n_iter` accepts an explicit `(n_iter, n_spaces)` array of candidate
feature-space weights instead of an integer count. The adapter uses that
opening: it draws the candidates with Himalaya's own
`generate_dirichlet_samples`, then validates and conditions them before they
reach the solver.

```python
with _scoped_himalaya_backend("numpy"):        # not the ambient backend
    candidates = generate_dirichlet_samples(
        n_samples=search_iterations, n_kernels=n_spaces,
        concentration=dirichlet_concentration, random_state=random_state,
    )
candidates = _prepare_feature_space_weights(candidates, dtype)  # nltools
deltas, weights, cv_scores = solve_group_ridge_random_search(
    Xs, Y, n_iter=candidates, ..., return_weights=True,
)
```

The sampler ends with `get_backend().asarray(gammas)`, so drawn candidates
otherwise inherit the dtype and device of whatever backend happened to be
globally active — a `device="cpu"` fit would stop being reproducible because
unrelated code left the global backend on MPS. The candidates are validated and
clamped on the host anyway, so they are drawn under an explicit `numpy` scope.

`_prepare_feature_space_weights` copies the candidates, requires every weight
to be finite and strictly positive and every row to sum to one, converts to the
feature dtype, and only then raises weights below `np.finfo(dtype).tiny` to
`tiny`. That floor is a numerical boundary, not a validation relaxation: the
random search scales each space by `sqrt(gamma)` and divides the same buffer
back afterwards, and on a float32 device a subnormal weight destroys the buffer
on the way back. A zero or negative weight is still an error. Upstream
[gallantlab/himalaya#107](https://github.com/gallantlab/himalaya/pull/107)
proposes the same clamp inside the solver; applying it on the nltools side of
the call makes a fork unnecessary.

The caller's arrays are never modified. Candidate preparation copies, and
Himalaya concatenates the feature spaces into a fresh buffer before scaling it.

## Recovering the fitted state

Himalaya reports the banded solution as `deltas = log(gamma / alpha)`, where
each `gamma` column sums to one. `deltas_` is not public state, so the adapter
converts it back to the two quantities the spec names, in float64 and by way of
a log-sum-exp so a float32 device cannot underflow the small weights:

```python
shifted = deltas - deltas.max(axis=0, keepdims=True)
feature_space_weights_ = exp(shifted) / exp(shifted).sum(axis=0, keepdims=True)
alpha_ = exp(-(deltas.max(axis=0) + log(exp(shifted).sum(axis=0))))
```

The recovered `alpha_` is then snapped back onto the candidate grid by nearest
log-distance. The log/exp round trip drifts by a few ULPs on float32, and the
bootstrap and the fixed refit both need the exact alpha the search selected.

`cv_scores_` is Himalaya's fold-averaged score at the selected alpha:
`(n_targets,)` for ordinary fits and `(search_iterations, n_targets)` for
banded ones, squeezed to a `float` or a 1-D array when `y` is one-dimensional.
Every fitted array is normalized to CPU NumPy regardless of the device that
produced it. There is no `intercept_` and no `deltas_`.

## One fixed-hyperparameter refit

`_refit_fixed_hyperparameters` is the package's only fixed-hyperparameter ridge
solve. Ordinary fixed-alpha fitting, the equivalence checks for the banded
refit, and every Ridge bootstrap resample — CPU or GPU, ordinary or banded —
route through it, so a resample cannot drift numerically from the full-data
fit.

Bootstrap replicates differ only in which rows they draw, so the engines in
`algorithms/inference/bootstrap.py` convert the design once into a
`_ResidentDesign` — the concatenated feature spaces and the response, already
on the backend in the working dtype — and each replicate passes its resample as
`row_indices`. On a GPU that keeps the host-to-device copy out of the replicate
loop; on the CPU it keeps the concatenation out of it. `_refit_resample` is the
one replicate implementation: it applies the same indices to the design and the
response and forwards the fitted `alpha_` and `feature_space_weights_`
untouched. The GPU driver's batch loop exists only to bound how many replicates
are retained before aggregation, and it uses the same
`backends._ridge_bootstrap_batch_size` / `_compute_oom_safe` machinery.

`_working_dtype` is the single dtype rule both `_Ridge.fit` and the shared refit
use: `float32` on MPS, which is float32-only, otherwise the promoted input
dtype with a `float32` floor. Applying it inside the refit is what keeps a
bootstrap from handing float64 to the MPS backend and triggering its downcast
warning on every replicate.

Measured on an Apple M3 (`torch-mps`, 100 replicates, shared alpha):

| Workload | Per-replicate transfer, float64 | Resident design, float32 |
|---|---|---|
| 120 obs x 40 feat → 4 000 voxels | 0.53 s | 0.50 s |
| 200 obs x 60 feat → 20 000 voxels | 3.69 s | 3.36 s |

The transfer is not the bottleneck at these shapes: profiling puts essentially
all of the remaining time inside Himalaya's `solve_ridge_svd` (33 ms per
replicate at 20 000 voxels, against 0.1 ms for the surrounding nltools code).
The pre-0.6.0 hand-written GPU SVD did the same algebra in 3.6 ms, so Himalaya
costs roughly 9x more per call here, and `torch-mps` is currently no faster than
the `numpy` backend for this workload (3.36 s vs 3.04 s). Recovering that would
mean re-implementing a solver, which this package does not do; the GPU path
remains correct and is retained for CUDA hosts and larger designs.

It accepts a scalar or per-target `alpha` and optional shared or per-target
feature-space weights, promotes the inputs to a floating working dtype, scales
space `k` by `sqrt(gamma[k])`, delegates the solve to `solve_ridge_svd`, and
scales the coefficients back into the original feature coordinates. The dtype
promotion is load-bearing rather than cosmetic: Himalaya solves in the dtype it
is handed, so an integer design matrix — one-hot or binary event regressors, an
ordinary thing in this domain — would truncate the shrinkage arithmetic and
return all-zero coefficients with no error. That scaling is exactly the per-space penalty the spec
defines:

```text
argmin_b ||X b - y||² + Σ_k (alpha / gamma[k]) ||b_k||²
```

Targets that selected the same weight vector share one decomposition. The
grouping is an implementation detail and never changes the result — a test
compares the grouped solve against one call per target. Weights shared by every
target (the common case) short-circuit to a single group without sorting.

The banded path does not double-refit: `solve_group_ridge_random_search` with
`return_weights=True` already multiplies its primal weights by `sqrt(gamma)`,
so `coef_` comes back in original coordinates.

## Device and memory

`device` accepts only `"cpu"` and `"gpu"`. There is no `"auto"` on this
estimator. `_resolve_backend` maps the request to an nltools `_Backend`, which
the adapter maps to a Himalaya backend:

| `device` | `_Backend.device` | Himalaya backend |
|---|---|---|
| `"cpu"` | `cpu` | `numpy` |
| `"gpu"` | `cuda` | `torch_cuda` |
| `"gpu"` | `mps` | `torch_mps` |

An explicit `"gpu"` runs on an accelerator or raises; `_resolve_backend` owns
that rule, so the estimator cannot silently degrade to CPU. On MPS the fit runs
in float32 — the backend supports nothing else — and Himalaya's documented
hybrid path may execute individual unsupported operations on the host. That
hybrid is by design, not a backend fallback.

Himalaya's backend is a module-level global, so the adapter scopes it:

```python
with _scoped_himalaya_backend(name):   # restores the previous backend in finally
    ...
```

The previous backend is restored after success and after an exception alike, so
a fit cannot leak its device into unrelated code.

Batch sizes are derived, never passed by the user. `memory_budget_gb=None`
measures the device through `backends._device_memory_budget`; an explicit
positive value is the budget verbatim. Two sizing functions supply only the
Himalaya-shaped working-set estimates and hand them to
`backends._auto_batch_size`. `_batch_sizes` sizes a whole cross-validated or
banded fit; `_refit_targets_batch` sizes the fixed-hyperparameter refit, whose
dominant allocation depends on whether the targets share an alpha:

| Function | Batch | Dominant allocation |
|---|---|---|
| `_batch_sizes` | `n_alphas_batch` | decomposition matrices, `(n_alphas_batch, n_features, n_samples)` |
| `_batch_sizes` | `n_targets_batch` | fold predictions, `(n_alphas_batch, n_samples, n_targets_batch)` |
| `_batch_sizes` | `n_targets_batch_refit` | refit weights, `(n_alphas_batch, n_features, n_targets_batch)` |
| `_refit_targets_batch` | `per_target_alpha=True` | `solve_ridge_svd`'s `(n_targets_batch, n_samples, n_samples)` block |
| `_refit_targets_batch` | `per_target_alpha=False` | one shared shrinkage operator, so `(n_samples + n_features)` per target |

All budget arithmetic, the saturation ceiling, and OOM recovery live in
`backends.py`. That is a hard invariant: an algorithm may estimate its own
working set but must never compute a budget.

## Backend abstraction

The nltools `_Backend` in `nltools/algorithms/backends.py` remains the device
abstraction for alignment and the bootstrap engines, and `_Ridge.backend_` is
the resolved instance (its `.name` reports `numpy`, `torch-cuda`, or
`torch-mps`). Himalaya owns every decomposition on the ridge paths; `_Backend` supplies the
device, the array module, and the memory budget only.

`_Backend` instances are picklable: `backend_` is public fitted state, so a
fitted `_Ridge` has to survive `copy.deepcopy`, `BrainData.copy()`, and
process-based `n_jobs` workers. `__getstate__` drops the live array module and
`__setstate__` recovers it from the pickled backend name.

`parallel=` stays an internal name in those subsystems. The public surface —
`_Ridge(device=...)`, `BrainData.fit(model='ridge', ridge_device=...)`,
`BrainData.bootstrap(device=...)` — uses the canonical `device` keyword, and
`scripts/check_api_vocabulary.py` enforces that against
`docs/_data/api-vocabulary.yml`.

## Alpha grids

| Use case | Alpha range | Notes |
|---|---|---|
| Exploratory | `[0.01, 0.1, 1, 10, 100]` | fast, coarse |
| Standard | `np.logspace(-2, 3, 10)` | publication quality |
| Thorough | `np.logspace(-3, 4, 20)` | capture nuance |

Log-spaced alphas cover a wide range cheaply. Himalaya's tie-break adds a
`1e-10 * log(alpha)` slope to the fold-averaged scores, so exactly tied
candidates resolve to the larger alpha.

## References

1. Dupré la Tour et al. (2022). himalaya: Ridge regression with multiple solvers.
2. Hoerl & Kennard (1970). Ridge regression. *Technometrics* 12(1):55–67.
3. Hastie et al. (2009). *The Elements of Statistical Learning* (2nd ed). Springer.
4. Nunez-Elizalde et al. (2019). Voxelwise encoding models with non-spherical priors.
   *Nature Neuroscience* 22:1060–1065.
5. Naselaris et al. (2011). Encoding and decoding in fMRI. *NeuroImage* 56(2):400–410.
