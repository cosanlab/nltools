# Ridge specification

> Release scope: BrainCollection and collection-only requirements in this specification
> are deferred to 0.6.1. Retained BrainData and estimator contracts remain targets for 0.6.0.

This file specifies `nltools.models._Ridge` and its Himalaya adapter. Code,
tests, and docstrings must implement this contract. Compatibility notes and
migration history belong elsewhere.

## Purpose and ownership

`_Ridge` fits ordinary and banded Ridge regressions. It estimates coefficients
and selects hyperparameters. Callers construct and preprocess `X` and `y`,
including any intercept column. `_Ridge` must not center, standardize, scale, or
add an intercept.

Himalaya defines the numerical behavior for Ridge fitting. `nltools` depends
on a tested Himalaya 0.4.x release and delegates decomposition,
cross-validation loss, hyperparameter selection, Dirichlet search, and
coefficient refitting to its public APIs. `nltools` must not copy Himalaya's
numerical solvers or duplicate its numerical tests.

`nltools` defines argument names and validation, named feature-space alignment,
device and memory policy, fitted-state normalization, bootstrap orchestration,
and `BrainData` and `BrainCollection` integration. Himalaya estimators remain
implementation details. Public `nltools` classes must not inherit from or
re-export them.

The `BrainData.fit(model="ridge", ...)` facade follows the same ownership rule.
It does not accept `scale` or `standardize`; callers compose the corresponding
`BrainData` methods before fitting.

The fitted `BrainData` does not retain the training features as `X_`.
No-argument prediction returns an independent copy of
`ridge_fitted_values`. Ridge coefficient and prediction bootstraps require the
training `X` explicitly; the fitted model supplies its selected `alpha_` but
does not retain a duplicate of the feature matrix.

Ridge bootstraps hold the fitted model's selected hyperparameters fixed. An
ordinary Ridge bootstrap holds `alpha_` fixed. A banded Ridge bootstrap holds
both `alpha_` and `feature_space_weights_` fixed. Bootstrap resampling measures
sampling uncertainty conditional on the selected model; it does not rerun
cross-validation or the banded random search within each resample.

`BrainData.fit` does not create a separate `cv_results_` dictionary or run a
second cross-validation pass for held-out predictions. `_Ridge.alpha_` and
`_Ridge.cv_scores_` are the only alpha-selection results. The facade stores
`ridge_weights`, `ridge_fitted_values`, and `ridge_r2` as independently owned
`BrainData` results. `ridge_r2` is the full-data, per-target value returned by
`_Ridge.score`; the facade does not use the ambiguous name `ridge_scores`, which
could be confused with the negative-MSE selection values in `cv_scores_`.
Every attached or returned `BrainData` follows the ownership contract in
`braindata.md`, including independent mask and masker state.

## Public API

```python
_Ridge(
    *,
    alpha: float | Sequence[float] | np.ndarray = 1.0,
    cv: int | BaseCrossValidator | None = None,
    search_iterations: int = 100,
    dirichlet_concentration: float | Sequence[float] = (0.1, 1.0),
    device: Literal["cpu", "gpu"] = "cpu",
    memory_budget_gb: float | None = None,
    per_target_alpha: bool = True,
    prefer_conservative_alpha: bool = False,
    random_state: int | None = None,
    progress_bar: bool = False,
)
```

`_Ridge` provides these methods:

```python
fit(X, y) -> _Ridge
predict(X) -> np.ndarray
score(X, y) -> float | np.ndarray
```

`fit` must return `self`.

## Inputs and model modes

`X` selects the model form:

- A two-dimensional array selects ordinary Ridge.
- A non-empty mapping from unique string names to two-dimensional arrays
  selects banded Ridge.
- All banded feature matrices must have the same sample count.
- `X` and `y` must have the same sample count.
- `y` must have one or two dimensions.

`predict` must accept the structure used by `fit`. Ordinary Ridge accepts one
matrix. Banded Ridge accepts a mapping with exactly the fitted feature-space
names. Mapping order may differ; `_Ridge` aligns spaces to
`feature_space_names_` before prediction. Each space must retain its fitted
feature count. The implementation may concatenate banded inputs internally.

## Alpha and cross-validation

`alpha` has two forms:

- A positive finite scalar requests a fixed-alpha fit and requires `cv=None`.
- A non-empty one-dimensional collection of positive finite values requests
  alpha selection and requires `cv`.

The string `"auto"` is invalid. There is no separate `alphas` argument.

Banded Ridge requires sequence-valued `alpha` and an explicit `cv`. A scalar
alpha is invalid for banded Ridge.

`cv` accepts an integer fold count or a reusable scikit-learn cross-validator.
An integer creates unshuffled K-fold splits. A single-use split generator is
invalid because fitting traverses the splits more than once.

`per_target_alpha=True` selects the highest-scoring alpha separately for each
target. `per_target_alpha=False` selects one alpha by averaging each
candidate's fold scores across targets.

## Selection criterion

Selection must reproduce Himalaya 0.4.11. For each held-out fold and target,
the score is negative mean squared error:

```text
-sum((y_true - y_pred) ** 2) / n_test_samples
```

Larger scores are better. Selection averages each candidate's scores across
folds before comparing alphas.

With `prefer_conservative_alpha=True`, a candidate is eligible when its mean
score is greater than the best alpha's mean score minus that best alpha's
standard deviation across folds. Fitting selects the largest eligible alpha.
When candidates have equal selection scores, fitting selects the larger alpha.
`prefer_conservative_alpha=True` is invalid with `per_target_alpha=False`.

`_Ridge.score()` does not participate in alpha selection. It returns R-squared
separately for each target. A one-dimensional target produces a `float`; a
two-dimensional target produces an array with shape `(n_targets,)`. A constant
target has a score of zero.

## Banded ridge search

Banded Ridge jointly searches feature-space weights on the simplex and the
candidate alphas. `search_iterations` sets the number of sampled weight
vectors. `dirichlet_concentration` parameterizes their Dirichlet distribution
and defaults to `(0.1, 1.0)`.

Every explicit candidate feature-space weight must be finite and strictly
positive before dtype conversion, and each candidate vector must sum to one
within numerical tolerance. Converting the candidates must not mutate an array
supplied by the caller.

After conversion to the feature matrix's dtype and device, the implementation
replaces weights below that dtype's minimum positive normal value with that
minimum before computing `sqrt(gamma)`. This is the smallest weight that the
MPS float32 backend can reliably use during square-root scaling and subsequent
restoration of the reusable feature buffer. The floor applies only at this
numerical boundary; it must not make zero or negative explicit weights valid.

`random_state` seeds only the banded random search. The cross-validator controls
split randomness. Ordinary Ridge accepts `random_state` and ignores it, because
`BrainData.fit` forwards one unprefixed `random_state` to whichever estimator it
builds (see `braindata.md`); rejecting it would break a documented facade
keyword that both estimators share.

The banded-only arguments are `search_iterations` and `dirichlet_concentration`.
For ordinary Ridge, non-default values of those two must raise an error.

## Numerical behavior

The model solves Ridge regression without adding an intercept:

```text
argmin_beta ||X @ beta - y||² + alpha * ||beta||²
```

`predict` computes `X @ coef_`. For banded Ridge, `X` denotes the feature
matrices concatenated in `feature_space_names_` order.

One-dimensional `y` must produce one-dimensional predictions. Two-dimensional
`y` must produce two-dimensional predictions with one column per target.

CPU and GPU execution implement the same equation and must agree within the
test suite's dtype-specific tolerances. `device` accepts only `"cpu"` or
`"gpu"`. `device="gpu"` must resolve to an available CUDA or MPS accelerator or
raise; it must not silently select a CPU backend. On MPS, operations that
PyTorch does not support, including some decompositions, may use Himalaya's
documented CPU fallback while supported operations remain on MPS. The MPS
backend therefore uses both devices by design.

`memory_budget_gb=None` measures the selected device's available memory with
conservative headroom. A positive explicit value supplies the working-memory
budget used to derive Himalaya's internal target, alpha, and refit batch sizes.
It is a budget rather than a hard process limit: inputs, outputs, allocator
overhead, and third-party libraries may consume additional memory. Internal
batch sizes and CPU-resident-target controls are not public API.

## Fitted state

After `fit`, the model exposes:

- `coef_`: shape `(n_features,)` for one-dimensional `y`, otherwise
  `(n_features, n_targets)`. Banded coefficients use concatenated feature-space
  order.
- `alpha_`: a scalar for a fixed or shared alpha, otherwise shape
  `(n_targets,)`.
- `cv_scores_`: `None` for a fixed-alpha fit. For ordinary Ridge it is the
  fold-averaged negative-MSE score at the selected alpha: a `float` for
  one-dimensional `y`, otherwise shape `(n_targets,)`. For banded Ridge its
  shape is `(search_iterations,)` for one-dimensional `y`, otherwise
  `(search_iterations, n_targets)`. Each banded entry is the fold-averaged score
  for the best alpha under that sampled feature-space weighting.
- `feature_space_weights_`: `None` for ordinary Ridge. For banded Ridge its
  strictly positive columns sum to one, and its shape is `(n_spaces,)` for
  one-dimensional `y`, otherwise `(n_spaces, n_targets)`.
- `feature_space_names_`: `None` for ordinary Ridge, otherwise a tuple of the
  fitted mapping keys in coefficient order.
- `feature_space_sizes_`: `None` for ordinary Ridge, otherwise a tuple of
  feature counts aligned with `feature_space_names_`.
- `backend_`: the resolved execution backend.
- `n_samples_`: the fitted sample count.
- `n_features_in_`: the total fitted feature count across all spaces.
- `is_fitted_`: `True` after a successful fit.

The model must not expose `intercept_` or `deltas_`.

The adapter exposes the selected alpha and simplex weights as `alpha_` and
`feature_space_weights_`, with the shapes specified above. Himalaya's `deltas_`
representation remains private.

The internal collection cache stores every fitted Ridge attribute specified
above exactly once, plus the facade-only fitted values and R-squared map. It
does not store training feature matrices, duplicate estimator fields on the
facade, or held-out predictions from another validation pass. Hydration
reconstructs independently owned estimator and facade state.

## Fixed-hyperparameter refitting

Ordinary fitting, banded fitting, and Ridge bootstrapping share one internal
fixed-hyperparameter refit implementation. It accepts scalar or per-target
`alpha` values and optional shared or per-target feature-space weights. It
returns coefficients in the original, unscaled feature coordinates.

For banded Ridge, feature-space weight `gamma[k, j]` and regularization
`alpha[j]` are equivalent to the per-space penalty
`alpha[j] / gamma[k, j]` for target `j`. Every `gamma` is finite and strictly
positive. Fixed refitting delegates its Ridge solve to Himalaya and returns
coefficients in the original feature coordinates.

Targets with the same selected feature-space weight vector may share a matrix
decomposition. This grouping is an implementation detail and must not change
the fitted coefficients. The final full-data fit and every bootstrap refit use
this same numerical path. Bootstrap code must not call the cross-validation or
random-search solvers.

## Himalaya adapter contract

The facade translates these `nltools` names when it calls Himalaya:

- `per_target_alpha`, not `local_alpha`
- `prefer_conservative_alpha`, not `conservative`
- `search_iterations`, not `n_iter`
- `dirichlet_concentration`, not `concentration`

The facade does not accept `fit_intercept`, a scoring callback, `alpha="auto"`,
`device="auto"`, Himalaya `solver_params`, or manual batch sizes. It always
configures Himalaya with no intercept and its native negative-MSE selection
loss. The adapter scopes and restores Himalaya's process-global backend,
normalizes fitted arrays to NumPy on the CPU, and exposes only the fitted state
specified above.

`BrainData.fit` translates `ridge_memory_budget_gb` to
`_Ridge.memory_budget_gb` and `ridge_progress_bar` to `_Ridge.progress_bar`.
`BrainData.bootstrap` uses the device-neutral name `memory_budget_gb`. CPU
bootstrap worker count and result streaming and GPU batch sizing derive from
the same budget policy; `n_jobs` remains an independent concurrency ceiling.

## Validation and errors

Validate inputs before decomposition or cross-validation. Reject:

- invalid `X` or `y` dimensions or unequal sample counts;
- empty or inconsistent banded inputs;
- non-string, missing, or additional feature-space names;
- empty, non-finite, non-positive, or multidimensional alpha collections;
- scalar `alpha` with non-`None` `cv`;
- sequence-valued `alpha` without `cv`;
- banded `X` with scalar `alpha`;
- `prefer_conservative_alpha=True` with `per_target_alpha=False`;
- unsupported devices;
- non-positive or non-finite `memory_budget_gb` values;
- non-default banded-only controls during ordinary Ridge fitting; and
- prediction inputs that do not match the fitted feature structure.

Each error must name the conflicting arguments or the mismatched dimension.
Removed names must raise `TypeError`; the API must not translate or alias them.

## Required tests

Tests must cover:

- every valid and invalid `alpha` and `cv` combination;
- ordinary and banded fitting;
- fitted and mismatched prediction structures;
- shared and per-target alpha selection;
- parity with Himalaya 0.4.11 negative-MSE selection, conservative tolerance,
  and tie-breaking;
- compact Himalaya-style selected CV scores;
- strictly positive feature-space weights, caller-owned candidate preservation,
  and dtype-specific underflow protection on NumPy and MPS;
- deterministic banded search under a fixed `random_state`;
- CPU and GPU parity within explicit tolerances;
- unavailable explicit GPU execution;
- every fitted attribute and its shape;
- banded prediction with reordered, missing, and additional feature spaces;
- absence of removed parameters, aliases, and attributes;
- agreement between the facade and Himalaya adapter names, defaults, and
  semantics;
- correct prefix translation at the `BrainData.fit` boundary;
- fixed-hyperparameter refits with scalar and per-target alphas and shared and
  per-target feature-space weights;
- equivalence of banded fixed refits to the corresponding generalized Ridge
  system;
- ordinary and banded bootstrap refits that hold selected hyperparameters fixed
  and never rerun model selection;
- banded bootstrap mapping alignment and common row resampling across spaces;
- automatic and explicit memory budgets for CPU workers and CPU/GPU batches;
- exact retained-tail percentile intervals when `return_samples=False`;
- full-distribution output-size preflight when `return_samples=True`;
- banded CPU/GPU parity; and
- ordinary bootstrap support for per-target selected alphas.

## Non-goals

This API does not construct features, preprocess data, add intercepts, accept
custom scoring functions, accept caller-supplied fold labels or groups, or
silently replace the requested execution backend. The documented hybrid MPS
path is not a backend fallback. For `cv=int`, fitting constructs unshuffled
K-fold splits; callers who need another design must pass a cross-validator.

## BrainCollection boundary

`BrainCollection.fit(model="ridge", X=...)` uses an unambiguous structural
grammar:

- one named mapping supplies shared banded feature spaces;
- any value coercible to a finite numeric two-dimensional array supplies one
  shared ordinary matrix, including a nested numeric list;
- otherwise, a collection-length sequence supplies either one two-dimensional
  ordinary matrix per member or one named mapping per member, without mixing
  the two modes;
- `None` uses each member's stored ordinary design.

For per-member inputs, the outer sequence length must equal the collection
length. Each member validates its own sample count and feature widths. Banded
mappings for every member must have the same feature-space name set; order may
differ and is aligned by name.

`BrainCollection.predict` accepts one shared matrix or mapping, or an outer
list providing one matrix or mapping per fitted member. Each fitted model
validates its own ordinary feature count or named banded structure. Cached
bundles retain `feature_space_names_`, `feature_space_sizes_`, coefficients,
selected alphas, and feature-space weights, so prediction preserves the full
banded model rather than flattening away its structure.

## BrainData bootstrap boundary

`BrainData.bootstrap(statistic="weights", X=...)` and
`BrainData.bootstrap(statistic="predict", X=..., X_test=...)` support both
ordinary and banded fitted Ridge models. `X` is always required and must contain
the training features in their original row order. Its row count must equal the
number of observations in the fitted `BrainData`. Fitting does not retain a
hidden training-feature snapshot, so omitting `X` raises even when the same
features were supplied to `fit`.

For ordinary Ridge, `X` and `X_test` are matrices with the fitted feature
count. For banded Ridge, both are mappings with exactly the fitted feature-space
names and widths. Mapping order may differ and is aligned to
`feature_space_names_`. All training spaces and `BrainData.data` are resampled
with the same row indices. `X_test` may have any row count but must preserve the
fitted feature structure.

Weight summaries use the same concatenated feature order as `coef_` and
`ridge_weights`; banded bootstraps do not introduce a second return grammar.
Prediction summaries use the test-observation order. When samples are retained,
the bootstrap axis precedes the ordinary result dimensions.

`"weights"` summarizes encoding-model coefficients. `"predict"` summarizes
brain-response predictions evaluated at `X_test`. Neither mode runs MVPA
decoding or returns decoding weight maps.

Explicit GPU execution must use an available CUDA or MPS backend or raise. MPS
may use Himalaya's documented hybrid CPU fallback for unsupported operations;
it must not silently replace the requested backend with a CPU backend.
Supporting banded bootstraps requires a GPU-capable fixed-hyperparameter refit
path; the ordinary scalar-alpha bootstrap implementation is not a valid
substitute.
