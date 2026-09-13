# GLM specification

> Release scope: BrainCollection and collection-only requirements in this specification
> are deferred to 0.6.1. Retained BrainData and estimator contracts remain targets for 0.6.0.

This file specifies `nltools.models._Glm` and its functional numerical core.
Code, tests, and docstrings must implement this contract. Compatibility notes
and migration history belong elsewhere.

## Purpose and ownership

`_Glm` is a numerical estimator that calls Nilearn's `run_glm` to fit ordinary
least squares and autoregressive models and `compute_contrast` for inference.
It shares the `fit` and `predict` method names with `_Ridge` and adds
`compute_contrasts`.

Callers supply a precomputed `DesignMatrix` and preprocessed `y` and add any
required intercept column to the design.

`_Glm` fits one run represented by a `(DesignMatrix, y)` pair, with one- or
two-dimensional `y`. It has no knowledge of `BrainData`, masks, NIfTI images,
events, or multi-run orchestration. It must not construct or retain a Nilearn
`FirstLevelModel`.

## Public API

```python
_Glm(
    *,
    noise_model: str = "ols",
    bins: int = 100,
    n_jobs: int = 1,
    random_state: int | None = None,
)
```

`_Glm` provides these methods:

```python
fit(X: DesignMatrix, y) -> _Glm
predict(X: DesignMatrix) -> np.ndarray
compute_contrasts(
    contrasts,
    *,
    inference: bool = False,
) -> float | np.ndarray | ContrastResult | dict[str, float | np.ndarray | ContrastResult]
```

`fit` must return `self`.

There is no report method or progress-bar argument. Nilearn's `run_glm`
verbosity is joblib logging for some autoregressive fits, not a progress bar.
There is no `score` method. `r2_` exposes training fit quality; unlike Ridge,
GLM has no nltools consumer that evaluates it as a predictive estimator.

There is no public `BaseModel`. `_Glm` and `_Ridge` are independent estimators
with different capabilities and input contracts. Small validation operations
may be shared as private functions, but neither class inherits an artificial
common interface. `_Ridge` retains its own predictive `score(X, y)` method.

## Inputs and model modes

`X` must be a precomputed `DesignMatrix` with shape
`(n_samples, n_features)`. Raw arrays and other DataFrame types are invalid.
`y` must have shape `(n_samples,)` or `(n_samples, n_targets)`. Their sample
counts must match.

`noise_model="ols"` fits ordinary least squares. A string of the form `"arN"`,
where `N` is a positive integer, fits Nilearn's autoregressive model of that
order. Other values are invalid.

`bins` controls Nilearn's discretization of AR coefficients. For AR(1), it is
the maximum number of histogram bins. For higher-order AR models, it is the
maximum number of K-means clusters. It must be a positive integer.

`n_jobs` controls Nilearn's parallel fitting of autoregressive groups. It is
passed unchanged to Nilearn and follows joblib's integer convention, including
`-1` for all available CPUs. It defaults to one because the default OLS fit
does not use this parallel path.

`random_state` seeds K-means for AR models of order two or greater. It does not
alter OLS or AR(1) results.

## Numerical behavior

Fitting delegates to `nilearn.glm.first_level.run_glm` with `y` as the response
matrix and the array representation of `X` as the design matrix. The
implementation expands a
one-dimensional `y` to a single-column matrix for fitting, then squeezes the
target axis from `coef_`, `predicted_`, `residuals_`, `r2_`, `predict`, and
contrast outputs.

The model is:

```text
y = X @ beta + error
```

`_Glm` never adds or estimates a separate intercept. Callers include an
intercept column in `X` when the model requires one.

Predictions are always defined in observation space:

```text
predicted = X @ coef_
```

Training residuals are:

```text
residuals_ = y - predicted_
```

This definition also applies to autoregressive fits. Nilearn's internal
whitened-design predictions must not become the public prediction or residual
contract.

A constant target has zero variance, so Nilearn's ratio is undefined for it.
Copying the value preserves the non-finite result and must not surface NumPy's
divide warning: an empty voxel inside a mask is ordinary input, not a fault the
caller can act on.

`r2_` is copied from each fitted Nilearn `RegressionResults.r_square` before
the full results are discarded. It is Nilearn's variance ratio:

```text
variance(whitened_design @ coef_) / variance(whitened_y)
```

For OLS the whitening operation is the identity; with an intercept this equals
conventional R-squared. For autoregressive models it is a pseudo-R-squared in
the whitened space. The public name remains `r2_`, and `BrainData.glm_r2`
wraps the same values, but their docstrings must state these semantics. nltools
must not independently recompute this quantity from predictions or residuals.

`predict` requires a `DesignMatrix` with exactly the fitted column names.
Columns may appear in a different order; `_Glm` reorders them to
`feature_names_in_` before multiplying. Missing, additional, or duplicate
columns raise `ValueError`. Raw arrays and other DataFrame types are invalid
because they cannot preserve the fitted coefficient-to-regressor relationship.

## Fitted state

After `fit`, the model exposes:

- `coef_`: shape `(n_features,)` for one-dimensional `y`, otherwise
  `(n_features, n_targets)`.
- `predicted_`: training predictions with the same shape as the fitted `y`.
- `residuals_`: training residuals with the same shape as the fitted `y`.
- `r2_`: a scalar for one-dimensional `y`, otherwise shape `(n_targets,)`.
- `n_samples_`: the fitted sample count.
- `n_features_in_`: the fitted feature count.
- `feature_names_in_`: a tuple containing the fitted `DesignMatrix` column
  names in coefficient order.
- `n_targets_`: the fitted target count. This is one for a one-dimensional
  target.
- `is_fitted_`: `True` after a successful fit.

The model privately retains a compact `_GlmFitState` for subsequent contrasts.
This immutable structural record contains fitted feature names, voxel model
labels, coefficients, covariance for each fitted label, dispersion, and
residual degrees of freedom. Its numerical arrays preserve the dtype returned
by Nilearn. After extracting this state and the public fitted arrays, `_Glm`
must discard the full `RegressionResults` objects returned by `run_glm`; those
objects also retain the response, whitened response, regression model, and
residual arrays.

`_GlmFitState` is private but is the authoritative fitted representation for
contrast computation and serialization. A single internal contrast function
accepts this state and computes the contrast effect and variance with the same
matrix operations as Nilearn's functional `compute_contrast`. It then uses
Nilearn's public `Contrast` result to calculate the statistic, p-value, and
z-score. Nilearn does not provide a supported constructor for rebuilding a
`SimpleRegressionResults` from serialized arrays, so the implementation must
not hydrate one through private attributes. No facade or persistence layer
implements a second contrast path.

## Contrasts

`_Glm.compute_contrasts` accepts one contrast or a mapping of named contrasts.
A single contrast is either a string expression or one real-valued array-like
vector. A string names a fitted design column or combines columns
arithmetically, for example `"condition_a - condition_b"` or
`"2 * condition_a - condition_b"`. String parsing uses `feature_names_in_` and
must follow Nilearn's `expression_to_contrast_vector` semantics.

Multiple contrasts must be supplied as a mapping whose string keys name the
results and whose values are string expressions or numeric vectors. An
unnamed sequence of contrast definitions is invalid. This leaves every flat
numeric sequence unambiguously available as one contrast vector.

A numeric input is converted to a float64 array and must have shape
`(n_features_in_,)`. Whether supplied directly or produced from an expression,
the resolved vector must contain only finite values and at least one nonzero
value.

`_Glm`, not `BrainData`, calls `expression_to_contrast_vector`. It must validate
the resolved contrast before calling Nilearn's `compute_contrast` and must not
pad a short vector with zeros. An input with the wrong length or rank, including
a nested sequence that becomes a matrix, is invalid; `_Glm` does not compute
F-contrasts.

With `inference=False`, the method returns only the effect:

```text
effect = contrast @ coef_
```

This is the default because effect estimates are the appropriate input to a
second-level model. It does not call Nilearn's contrast inference. For a
resolved numeric vector, the result is equivalent to direct coefficient
arithmetic; the string form is the user-facing syntax sugar that makes the
method useful for effect-only contrasts. A single input returns one effect. A
mapping returns a dictionary with the same keys and one effect per value.

With `inference=True`, the method tests the directional hypothesis represented
by the contrast against zero and returns all inferential outputs together:

```python
@dataclass(frozen=True)
class ContrastResult(Generic[Payload]):
    effect: Payload
    variance: Payload
    standard_error: Payload
    statistic: Payload
    z_score: Payload
    p_value: Payload
    degrees_of_freedom: float | np.ndarray
```

`ContrastResult` is one public generic result type shared by `_Glm`, `BrainData`,
and `BrainCollection`. It lives in `nltools.models.results` and is re-exported
from `nltools.models`, preserving the dependency direction from data facades to
models. Its fields cannot be rebound. Array payloads remain mutable, but each
result owns its arrays: they must not alias the input contrast, the model's
retained state, or another result. A single input with `inference=True` returns
one result; a mapping returns a dictionary with the same keys and one result
per value.

The fields mean:

- `effect`: the estimated linear combination of coefficients.
- `variance`: the estimated variance of `effect`.
- `standard_error`: computed by nltools as `np.sqrt` of Nilearn's returned
  variance, without taking an absolute value or clipping.
- `statistic`: the signed t-statistic for the null hypothesis that `effect` is
  zero.
- `z_score`: the signed normal-score equivalent of the directional p-value.
- `p_value`: Nilearn's one-sided upper-tail p-value. Negating the contrast tests
  the opposite direction.
- `degrees_of_freedom`: the residual degrees of freedom used for inference.

For a model fitted with one-dimensional `y`, `effect`, `variance`,
`standard_error`, `statistic`, `z_score`, and `p_value` are floats. A fitted
`y` with shape `(n_samples, 1)` preserves its target axis, so those fields are
arrays with shape `(1,)`. For any other two-dimensional `y`, they have shape
`(n_targets,)`.

The effect-only return follows the same scalar and target-axis rules.
`ContrastResult.effect` must equal a separate effect-only call for the same
contrast.

Inferential contrast computation uses the private `_GlmFitState` and shared
contrast function. Its effect and variance must match Nilearn's functional
`compute_contrast`, and its statistic, z-score, p-value, and degrees of freedom
must match the resulting Nilearn `Contrast`, after applying the specified
scalar or array shape conversion.

Contrast statistics must preserve Nilearn's numerical behavior. `_Glm` must not
apply `abs`, clip a negative estimated effect variance, or introduce an
nltools-specific exception. `standard_error` follows raw
`np.sqrt(variance)` semantics and may therefore be non-finite. The
Nilearn-provided `statistic` may use Nilearn's internal variance lower bound;
that bound must not alter the returned `variance` or `standard_error`.

Calling `compute_contrasts` before `fit` raises `RuntimeError`. Nonnumeric,
boolean, and complex-valued inputs raise `TypeError`. Empty, non-finite,
all-zero, incorrectly sized, or non-one-dimensional inputs raise `ValueError`.
An unknown design column or invalid expression raises `ValueError` and reports
the available column names. A mapping key that is not a string, an empty
mapping, or an unnamed sequence of contrast definitions is invalid.
`inference` must be a boolean.

## BrainData boundary

`BrainData.fit(model="glm", ...)` constructs and retains a fitted `_Glm` in
`model_`. The facade requires a precomputed `DesignMatrix`, delegates numerical
fitting to `_Glm`, and stores `glm_betas`, `glm_residual`, `glm_predicted`, and
`glm_r2` as `BrainData` results. Fitting does not compute or store eager
`glm_t`, `glm_p`, or `glm_se` maps.
Every attached or returned `BrainData` follows the ownership contract in
`braindata.md`, including independent mask and masker state.

The fitted `BrainData` does not retain the training input as `X_` or
`design_matrix`. Feature names and contrast state belong to `model_`.
No-argument prediction returns an independent copy of `glm_predicted`, so it
does not require the original design matrix.

The facade does not preprocess the response during fitting. `scale` and
`standardize` are not fit arguments; callers compose the corresponding
`BrainData` methods before `fit`. This keeps the fitted object's data,
predictions, residuals, and coefficients in the explicitly supplied response
space.

`BrainData.compute_contrasts` accepts the same single-contrast and named-mapping
forms and exposes the same `inference=False` control. The default returns an
effect `BrainData` for one contrast or a keyed dictionary of `BrainData`
effects for a mapping. For a numeric vector, each result is equivalent to
`contrast @ glm_betas.data`; for a string, it provides the convenient named
contrast syntax unavailable through arithmetic. With `inference=True`, it
returns `ContrastResult[BrainData]` for one contrast or a keyed dictionary of
those results for a mapping. `BrainData` forwards each original contrast
definition unchanged; both modes delegate parsing and calculation to the
fitted `_Glm`.

`BrainData.predict()` returns an independently owned copy of the stored
training predictions. `BrainData.predict(X=...)` requires a `DesignMatrix` for
a fitted GLM and delegates its named-column validation and alignment to
`_Glm.predict`. Ridge prediction retains its numerical feature-matrix contract.

`BrainCollection.compute_contrasts` exposes the same contrast forms and return
shape. Its payload is `BrainCollection`, so inferential calls return
`ContrastResult[BrainCollection]`. `degrees_of_freedom` is an array in subject
order because collection members may have different sample counts or designs.

For a string contrast, each collection member resolves the expression against
its own fitted feature names. Column order and unrelated nuisance regressors
may differ, but every name referenced by the expression must exist for every
member. For a numeric contrast, the first member's fitted feature order is
canonical. Every other member must have exactly the same feature-name set; its
coefficients are reordered to the canonical order before applying the vector.
Missing or additional features raise `ValueError` rather than allowing one
numeric vector to represent different estimands across members.

The internal collection cache stores `_GlmFitState` losslessly and stores every
other fitted numerical value exactly once. Hydration reconstructs the fitted
estimator and the facade's independently owned effect, prediction, residual,
and R-squared maps. The cache must not downcast Nilearn's state to float32. Both
OLS and autoregressive fits are supported. Loading cached state and computing a
contrast uses the same internal Nilearn-backed function as an in-memory `_Glm`;
the collection layer must not store the training design or reconstruct OLS
statistics from `X`, residuals, or a pseudoinverse.

For a stack of subject-level effect maps, `BrainData.ttest` provides the
intercept-only group test without retaining a fitted model. Its default p-value
is two-sided, unlike the directional p-value returned by GLM contrast
inference.

A multi-regressor second-level analysis fits `_Glm(noise_model="ols")` with a
second-level `DesignMatrix`. The design has one row per subject in the same
order as the stacked effect maps. The fitted `BrainData` then computes a
contrast on the second-level design coefficients. As in Nilearn's parametric
`SecondLevelModel`, this model estimates variance across effect maps; it does
not propagate first-level effect variance. The caller must choose OLS because
the generic facade cannot infer whether rows represent subjects or ordered
observations; autoregressive noise has no valid interpretation for subjects.

Model-specific facade arguments use the `glm_` prefix:

- `glm_noise_model` maps to `_Glm.noise_model`.
- `glm_bins` maps to `_Glm.bins`.
- `glm_n_jobs` maps to `_Glm.n_jobs` and defaults to one.

`random_state` retains its shared name because both Ridge and GLM use it.
`BrainCollection.fit` uses `n_jobs` for subject-level orchestration and
defaults `glm_n_jobs=1` for each inner fit, avoiding nested parallelism unless
the caller explicitly requests it.

The facade must not expose event construction, HRF, drift, high-pass,
smoothing, resampling, caching, subject-label, or report arguments.
