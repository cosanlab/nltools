---
title: models
label: page-models
---

Model classes for neuroimaging analysis.

Provides sklearn-compatible APIs for common neuroimaging analyses.

**Classes:**

Name | Description
---- | -----------
[`ContrastResult`](#models-contrastresult) | Frozen record of the inferential outputs of one contrast.
[`Glm`](#models-glm) | General linear model over a precomputed design matrix and a response.
[`Ridge`](#models-ridge) | Ridge regression over one or several named feature spaces.



## Classes

(models-contrastresult)=
### `ContrastResult`

```python
ContrastResult(effect: Payload, variance: Payload, standard_error: Payload, statistic: Payload, z_score: Payload, p_value: Payload, degrees_of_freedom: float | np.ndarray)
```

Frozen record of the inferential outputs of one contrast.

The one result type inferential contrast methods return. Its payload is
whatever the producer works in: `float` or `np.ndarray` for a `Glm`,
`BrainData` for the `BrainData` facade.

Fields cannot be rebound. Array payloads stay mutable, but each result owns
its arrays: they never alias the input contrast, a model's retained state,
or another result.

Every statistic describes the directional hypothesis that `effect` is zero,
so `p_value` is one-sided; negating the contrast tests the other direction.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`effect` | <code>Payload</code> | The estimated linear combination of coefficients.
`variance` | <code>Payload</code> | The estimated variance of `effect`.
`standard_error` | <code>Payload</code> | `np.sqrt` of `variance`, with no absolute value or clipping, so it may be non-finite.
`statistic` | <code>Payload</code> | The signed t-statistic for the null hypothesis that `effect` is zero.
`z_score` | <code>Payload</code> | The signed normal-score equivalent of the directional p-value.
`p_value` | <code>Payload</code> | The one-sided upper-tail p-value.
`degrees_of_freedom` | <code>float \| ndarray</code> | The residual degrees of freedom used for inference.

(models-glm)=
### `Glm`

```python
Glm(*, noise_model: str = 'ols', bins: int = 100, n_jobs: int = 1, random_state: int | None = None)
```

General linear model over a precomputed design matrix and a response.

Fits ordinary least squares or an autoregressive noise model with Nilearn's
`run_glm`, then exposes coefficients, predictions, residuals, R-squared, and
contrasts. The model is `y = X @ beta + error`: `Glm` never adds or
estimates an intercept, so include an intercept column in `X` when the model
needs one. Predictions and residuals are always in observation space, for
autoregressive fits as well as OLS.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`noise_model` | <code>str</code> | `'ols'` for ordinary least squares, or `'arN'` with `N` a positive integer for Nilearn's autoregressive model of that order (`'ar1'`, `'ar2'`, ...). Default `'ols'`. | <code>'ols'</code>
`bins` | <code>int</code> | Nilearn's discretization of the estimated AR coefficients — the maximum number of histogram bins for AR(1), and the maximum number of K-means clusters for higher orders. Must be positive. Default 100. | <code>100</code>
`n_jobs` | <code>int</code> | Number of CPUs Nilearn uses to fit autoregressive groups in parallel, following joblib's convention (`-1` is all cores). The default OLS fit does not use this path. Default 1. | <code>1</code>
`random_state` | <code>int \| None</code> | Seeds the K-means step for autoregressive models of order two or greater. It does not affect OLS or AR(1) results. Default None. | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`coef_` | <code>ndarray</code> | Fitted coefficients, shape `(n_features,)` for a one-dimensional `y` and `(n_features, n_targets)` otherwise.
`predicted_` | <code>ndarray</code> | Training predictions `X @ coef_`, same shape as the fitted `y`.
`residuals_` | <code>ndarray</code> | Training residuals `y - predicted_`, same shape as the fitted `y`.
`r2_` | <code>float \| ndarray</code> | Nilearn's `RegressionResults.r_square`, copied rather than recomputed. It is the variance ratio `variance(whitened_design @ coef_) / variance(whitened_y)`. For OLS the whitening is the identity, so with an intercept in the design this equals conventional R-squared; for autoregressive noise models it is a pseudo-R-squared in the whitened space. A float for a one-dimensional `y`, otherwise shape `(n_targets,)`.
`n_samples_` | <code>int</code> | Fitted sample count.
`n_features_in_` | <code>int</code> | Fitted feature count.
`feature_names_in_` | <code>tuple[str, ...]</code> | Fitted design column names in coefficient order.
`n_targets_` | <code>int</code> | Fitted target count; one for a one-dimensional `y`.
`is_fitted_` | <code>bool</code> | True after a successful fit.

**Methods:**

Name | Description
---- | -----------
[`compute_contrasts`](#models-compute-contrasts) | Compute one contrast or a named mapping of contrasts on the fitted model.
[`fit`](#models-fit) | Fit the model to one design matrix and response.
[`predict`](#models-predict) | Apply the fitted coefficients to a design matrix.



**Examples:**

```python
import numpy as np
from nltools.data import DesignMatrix
from nltools.models import Glm

n_samples = 100
rng = np.random.default_rng(0)
design = DesignMatrix(
    {
        "condition_a": rng.normal(size=n_samples),
        "condition_b": rng.normal(size=n_samples),
        "intercept": np.ones(n_samples),
    },
    sampling_freq=0.5,
)
y = rng.normal(size=(n_samples, 50))

model = Glm(noise_model="ar1").fit(design, y)
effects = model.compute_contrasts("condition_a - condition_b")
result = model.compute_contrasts("condition_a - condition_b", inference=True)
result.statistic  # → t-statistic per target
```

#### Methods

(models-compute-contrasts)=
##### `compute_contrasts`

```python
compute_contrasts(contrasts, *, inference: bool = False) -> float | np.ndarray | ContrastResult | dict
```

Compute one contrast or a named mapping of contrasts on the fitted model.

A contrast is a string expression over the fitted design column names —
`"condition_a - condition_b"`, `"2 * condition_a - condition_b"` — or a
real-valued vector with one weight per fitted column. Several contrasts
must be supplied as a mapping of names to those definitions, which
leaves every flat numeric sequence unambiguously available as one
contrast vector.

The default returns the effect `contrast @ coef_` only, the appropriate
input to a second-level model. With `inference=True`, the contrast is
tested against zero and every inferential output is returned together;
the p-value is Nilearn's one-sided upper-tail value, so negating the
contrast tests the opposite direction.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`contrasts` | <code>str \| array - like \| Mapping</code> | One contrast definition, or a mapping of string names to contrast definitions. | *required*
`inference` | <code>bool</code> | If True, return `ContrastResult` records instead of bare effects. Default False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>float \| ndarray \| [ContrastResult](#models-contrastresult) \| dict</code> | One effect, or one     `ContrastResult` when `inference=True`; a dictionary with the     same keys for a mapping. Effects and inferential fields are     floats for a model fitted on a one-dimensional `y`, shape `(1,)`     for a `(n_samples, 1)` response, and shape `(n_targets,)`     otherwise.

**Raises:**

Type | Description
---- | -----------
<code>RuntimeError</code> | If the model has not been fitted.
<code>TypeError</code> | If `inference` is not a bool, a mapping key is not a string, or a contrast is boolean, complex, or nonnumeric.
<code>ValueError</code> | If the mapping is empty, an expression is invalid or names an unknown column, or a resolved contrast is empty, non-finite, all zero, wrongly sized, or not one-dimensional.

**Examples:**

```python
model.compute_contrasts("condition_a - condition_b")
model.compute_contrasts([1, -1, 0])
model.compute_contrasts({"a_vs_b": "condition_a - condition_b"})
model.compute_contrasts("condition_a", inference=True).p_value
```

(models-fit)=
##### `fit`

```python
fit(X: DesignMatrix, y: DesignMatrix) -> Glm
```

Fit the model to one design matrix and response.

A one-dimensional `y` is expanded to a single column for Nilearn and the
target axis is squeezed back out of every fitted attribute and contrast
result.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[DesignMatrix](#page-data-design-matrix)</code> | Design of shape `(n_samples, n_features)`, including any intercept column the model needs. | *required*
`y` | <code>array - like</code> | Response of shape `(n_samples,)` or `(n_samples, n_targets)`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Glm](#models-glm)</code> | The fitted model, for method chaining.

**Raises:**

Type | Description
---- | -----------
<code>TypeError</code> | If `X` is not a `DesignMatrix`.
<code>ValueError</code> | If `y` is not one- or two-dimensional, or its sample count does not match `X`.

(models-predict)=
##### `predict`

```python
predict(X: DesignMatrix) -> np.ndarray
```

Apply the fitted coefficients to a design matrix.

`X` must carry exactly the fitted column names. They may appear in any
order; the columns are reordered to `feature_names_in_` before
multiplying, so the coefficient-to-regressor relationship survives.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[DesignMatrix](#page-data-design-matrix)</code> | Design with the fitted column names, in any order. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | `X @ coef_`, shape `(n_samples,)` for a model fitted on     a one-dimensional `y` and `(n_samples, n_targets)` otherwise.

**Raises:**

Type | Description
---- | -----------
<code>TypeError</code> | If `X` is not a `DesignMatrix`.
<code>ValueError</code> | If the model is not fitted, or `X` has duplicate, missing, or additional columns.

(models-ridge)=
### `Ridge`

```python
Ridge(*, alpha: float | Sequence[float] | np.ndarray = 1.0, cv: float | Sequence[float] | np.ndarray = None, search_iterations: int = 100, dirichlet_concentration: float | Sequence[float] = (0.1, 1.0), device: str = 'cpu', memory_budget_gb: float | None = None, per_target_alpha: bool = True, prefer_conservative_alpha: bool = False, random_state: int | None = None, progress_bar: bool = False)
```

Ridge regression over one or several named feature spaces.

Fits `argmin_b ||X @ b - y||^2 + alpha * ||b||^2` without an intercept.
Callers own preprocessing: `Ridge` never centers, scales, standardizes, or
adds an intercept column.

A two-dimensional `X` fits ordinary Ridge. A mapping from names to
two-dimensional arrays fits banded Ridge, which searches feature-space
weights on the simplex jointly with the alphas.

Himalaya defines the numerical behavior: the cross-validation loss is
negative mean squared error, and alpha selection, the Dirichlet search, and
coefficient refitting all come from its solvers.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`alpha` | <code>float \| Sequence[float] \| ndarray</code> | A positive finite scalar fits a fixed alpha and requires `cv=None`. A non-empty one-dimensional collection of positive finite values selects an alpha by cross-validation and requires `cv`. Default: `1.0`. | <code>1.0</code>
`cv` | <code>int \| BaseCrossValidator \| None</code> | An integer builds unshuffled K-fold splits; a reusable scikit-learn cross-validator is used as given. A single-use split generator is invalid because fitting traverses the splits more than once. Default: `None`. | <code>None</code>
`search_iterations` | <code>int</code> | Number of sampled feature-space weight vectors for banded Ridge. Default: `100`. | <code>100</code>
`dirichlet_concentration` | <code>float \| Sequence[float]</code> | Concentration parameter(s) of the Dirichlet distribution the candidate weights are drawn from. A list is cycled through across candidates. Default: `(0.1, 1.0)`. | <code>(0.1, 1.0)</code>
`device` | <code>str</code> | `'cpu'` or `'gpu'`. An explicit `'gpu'` resolves to CUDA or MPS or raises; it never falls back to a CPU backend. Default: `'cpu'`. | <code>'cpu'</code>
`memory_budget_gb` | <code>float \| None</code> | Working-memory budget in GB used to size Himalaya's internal batches. None measures the device with conservative headroom. It is a budget, not a hard process limit. Default: `None`. | <code>None</code>
`per_target_alpha` | <code>bool</code> | True selects the best alpha separately per target; False averages each candidate's fold scores across targets and selects one shared alpha. Default: `True`. | <code>True</code>
`prefer_conservative_alpha` | <code>bool</code> | True selects the largest alpha whose mean score beats the best alpha's mean score minus that alpha's standard deviation across folds. Invalid with `per_target_alpha=False`. Default: `False`. | <code>False</code>
`random_state` | <code>int \| None</code> | Seed for the banded random search only; the cross-validator controls split randomness. Ordinary Ridge accepts it and ignores it — it has no randomness of its own — so that `BrainData.fit` can keep forwarding one shared `random_state` to whichever estimator it builds. Default: `None`. | <code>None</code>
`progress_bar` | <code>bool</code> | Show a progress bar over the banded search. Default: `False`. | <code>False</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`coef_` | <code>ndarray</code> | `(n_features,)` for one-dimensional `y`, otherwise `(n_features, n_targets)`, in concatenated feature-space order.
`alpha_` | <code>float \| ndarray</code> | Scalar for a fixed or shared alpha, otherwise `(n_targets,)`.
`cv_scores_` | <code>float \| ndarray \| None</code> | None for a fixed-alpha fit. For ordinary Ridge, the fold-averaged negative-MSE score at the selected alpha. For banded Ridge, `(search_iterations,)` or `(search_iterations, n_targets)` fold-averaged scores.
`feature_space_weights_` | <code>ndarray \| None</code> | None for ordinary Ridge. Strictly positive weights whose columns sum to one, shaped `(n_spaces,)` or `(n_spaces, n_targets)`.
`feature_space_names_` | <code>tuple[str, ...] \| None</code> | Fitted mapping keys in coefficient order; None for ordinary Ridge.
`feature_space_sizes_` | <code>tuple[int, ...] \| None</code> | Feature counts aligned with `feature_space_names_`; None for ordinary Ridge.
`backend_` | <code>[Backend](#backends-backend)</code> | The resolved execution backend.
`n_samples_` | <code>int</code> | Fitted sample count.
`n_features_in_` | <code>int</code> | Total fitted feature count across spaces.
`is_fitted_` | <code>bool</code> | True after a successful fit.

**Methods:**

Name | Description
---- | -----------
[`fit`](#models-fit) | Fit the model.
[`predict`](#models-predict) | Predict targets for `X`.
[`score`](#models-score) | Return the coefficient of determination for each target.



**Examples:**

```python
import numpy as np
from nltools.models import Ridge

X = np.random.randn(100, 50)
y = np.random.randn(100)

model = Ridge(alpha=1.0).fit(X, y)
predictions = model.predict(X)

# Banded ridge over two named feature spaces
spaces = {"motion": np.random.randn(100, 6), "task": np.random.randn(100, 12)}
banded = Ridge(alpha=[1.0, 10.0, 100.0], cv=5, search_iterations=20)
banded.fit(spaces, y)
print(banded.feature_space_weights_)
```

#### Methods

##### `fit`

```python
fit(X, y) -> Ridge
```

Fit the model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray \| Mapping[str, ndarray]</code> | A `(n_samples, n_features)` matrix for ordinary Ridge, or a non-empty mapping of unique names to equally sampled 2-D matrices for banded Ridge. | *required*
`y` | <code>ndarray</code> | Targets of shape `(n_samples,)` or `(n_samples, n_targets)`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Ridge](#tasks-prediction-ridge)</code> | `self`.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If any input or argument combination is invalid.
<code>RuntimeError</code> | If `device='gpu'` and no accelerator is available.

##### `predict`

```python
predict(X) -> np.ndarray
```

Predict targets for `X`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray \| Mapping[str, ndarray]</code> | Features in the structure used for fitting. Banded mappings may be in any order; they are aligned to `feature_space_names_`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | `(n_samples,)` when fitted on one-dimensional `y`,     otherwise `(n_samples, n_targets)`.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If the model is not fitted, or `X` does not match the fitted feature structure.

(models-score)=
##### `score`

```python
score(X, y) -> float | np.ndarray
```

Return the coefficient of determination for each target.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray \| Mapping[str, ndarray]</code> | Features in the fitted structure. | *required*
`y` | <code>ndarray</code> | True targets, `(n_samples,)` or `(n_samples, n_targets)`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>float \| ndarray</code> | A `float` for one-dimensional `y`, otherwise an     array of shape `(n_targets,)`. A constant target scores zero.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If the model is not fitted, or the shapes disagree.
