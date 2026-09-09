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
[`Glm`](#models-glm) | General Linear Model for fMRI data analysis with sklearn-compatible API.
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
Glm(*, t_r: float | None = None, noise_model: str = 'ols', smoothing_fwhm: float | None = None, mask: nib.Nifti1Image | None = None, progress_bar: bool = False, **kwargs: bool)
```

General Linear Model for fMRI data analysis with sklearn-compatible API.

Wraps `nilearn.glm.first_level.FirstLevelModel` by composition, similar to
how `BrainData` holds masker objects. Provides the sklearn-style
fit/predict/score interface while exposing full nilearn GLM functionality
through the `glm_` property.

Unlike `Ridge`, which works with 2-D arrays (samples × features), `Glm`
works with 4-D neuroimaging data (x × y × z × time) and design matrices, so
it does not use the shared feature-matrix validation. `predict()` follows sklearn's
`LinearRegression` semantics: with no argument it returns the fitted values
on the training data; with a new design matrix it returns `X @ coef_`
(single-run fits only).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>float</code> | Repetition time (TR) in seconds. If None, inferred from the data. | <code>None</code>
`noise_model` | <code>str</code> | Noise model for temporal autocorrelation: `'ols'` (ordinary least squares, independent errors) or `'ar1'` (autoregressive AR(1), accounts for temporal correlation). Default `'ols'`. | <code>'ols'</code>
`smoothing_fwhm` | <code>float</code> | Full width at half maximum in mm for spatial smoothing. If None, no smoothing is applied. | <code>None</code>
`mask` | <code>Nifti1Image</code> | Mask defining the voxels to analyze. If None, uses the package brain-space mask (like `BrainData`). | <code>None</code>
`progress_bar` | <code>bool</code> | If True, enable nilearn's per-run progress output. Default False. | <code>False</code>
`**kwargs` | <code>dict</code> | Forwarded to `nilearn.glm.first_level.FirstLevelModel` (e.g. `drift_model`, `hrf_model`, `memory`). | <code>{}</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`is_fitted_` | <code>bool</code> | Whether the model has been fitted.
`coef_` | <code>ndarray \| list[ndarray]</code> | Beta matrix `(n_regressors, n_voxels)` after fitting a single run, or one per run for multi-run fits.
`mask` | <code>Nifti1Image</code> | Mask image used for analysis.
`glm_` | <code>FirstLevelModel</code> | The wrapped nilearn model, for advanced use.
`residuals` | <code>list[Nifti1Image]</code> | Residual images, one per run.
`design_matrices_` | <code>list[DataFrame]</code> | Design matrices used in fitting, one per run.

**Methods:**

Name | Description
---- | -----------
[`compute_contrast`](#models-compute-contrast) | Compute a contrast using nilearn's statistical inference.
[`fit`](#models-fit) | Fit GLM to fMRI data.
[`predict`](#models-predict) | Predict from the fitted GLM.
[`report`](#models-report) | Generate a nilearn HTML report for the fitted GLM.
[`score`](#models-score) | Return mean R² across voxels and runs.



**Examples:**

```python
import numpy as np
import pandas as pd
from nibabel import Nifti1Image
from nilearn.glm.first_level import make_first_level_design_matrix
from nltools.models import Glm

# Synthetic fMRI data and a matching design matrix
n_scans = 100
img = Nifti1Image(np.random.randn(20, 20, 20, n_scans), np.eye(4))
frame_times = np.arange(n_scans) * 2.0
events = pd.DataFrame(
    {"onset": [10, 30, 50, 70], "duration": [1, 1, 1, 1], "trial_type": ["task"] * 4}
)
design_matrix = make_first_level_design_matrix(frame_times, events)

model = Glm(t_r=2.0, noise_model="ar1")
model.fit(img, design_matrices=design_matrix)

task_effect = model.compute_contrast("task", output_type="stat")
fitted_values = model.predict()
residuals = model.residuals
```

#### Methods

(models-compute-contrast)=
##### `compute_contrast`

```python
compute_contrast(contrast_def: str | np.ndarray | list | dict, output_type: str = 'stat') -> nib.Nifti1Image | dict
```

Compute a contrast using nilearn's statistical inference.

This is the primary method for extracting results from a fitted GLM.
Delegates to `FirstLevelModel.compute_contrast` for inference with the
correct degrees of freedom.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`contrast_def` | <code>str \| ndarray \| list \| dict</code> | A regressor name (e.g. `'task'`), a contrast vector (e.g. `[1, -1, 0, 0]`), or a dict of named contrasts. | *required*
`output_type` | <code>str</code> | `'stat'` (t-statistic map, default), `'z_score'`, `'p_value'` (one-sided, per the nilearn/SPM directional-contrast convention; flip the contrast for the other direction), `'effect_size'` (beta), `'effect_variance'`, or `'all'` (a dict of every map). | <code>'stat'</code>

**Returns:**

Type | Description
---- | -----------
<code>Nifti1Image \| dict</code> | The contrast map, or a dict of all maps keyed     by output type when `output_type='all'`.

**Examples:**

```python
model.fit(img, design_matrices=design_matrix)

t_map = model.compute_contrast("task")  # by regressor name
contrast_map = model.compute_contrast([1, -1, 0])  # contrast vector

results = model.compute_contrast("task", output_type="all")
t_map, p_map = results["stat"], results["p_value"]
```

(models-fit)=
##### `fit`

```python
fit(X: nib.Nifti1Image | list[nib.Nifti1Image], y: None = None, *, design_matrices: pd.DataFrame | DesignMatrix | list[pd.DataFrame | DesignMatrix] | None = None, events: pd.DataFrame | list[pd.DataFrame] | None = None, **kwargs: pd.DataFrame | list[pd.DataFrame] | None) -> Glm
```

Fit GLM to fMRI data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>Nifti1Image \| list[Nifti1Image]</code> | 4-D fMRI image(s) to fit, a single run or a list of runs. | *required*
`y` | <code>None</code> | Not used; present for sklearn API compatibility. | <code>None</code>
`design_matrices` | <code>DataFrame \| [DesignMatrix](#page-data-design-matrix) \| list</code> | Design matrix or one per run, each of shape `(n_scans, n_regressors)`. `DesignMatrix` objects are converted to pandas at this boundary. | <code>None</code>
`events` | <code>DataFrame \| list[DataFrame]</code> | Event specifications for automatic design-matrix creation; an alternative to `design_matrices`. | <code>None</code>
`**kwargs` | <code>dict</code> | Forwarded to `FirstLevelModel.fit`. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[Glm](#models-glm)</code> | The fitted model (for method chaining).

<details class="note" open markdown="1">
<summary>Note</summary>

Unlike `Ridge.fit`, this method does not validate `X` as a 2-D array
because the GLM works with 4-D neuroimaging data; validation is
delegated to nilearn's `FirstLevelModel`.

</details>

(models-predict)=
##### `predict`

```python
predict(X: np.ndarray | pd.DataFrame | None = None) -> list[nib.Nifti1Image] | np.ndarray
```

Predict from the fitted GLM.

With `X=None`, returns the fitted values on the training data (one
`Nifti1Image` per run), matching sklearn's `LinearRegression` semantics.
With a new design matrix, returns `X @ coef_` as a 2-D array, mirroring
`Ridge.predict`; this requires a single-run fit.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray \| DataFrame</code> | New design matrix of shape `(n_samples, n_regressors)`. Default None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>list[Nifti1Image] \| ndarray</code> | Fitted images per run when `X`     is None; otherwise predictions of shape `(n_samples, n_voxels)`.

**Raises:**

Type | Description
---- | -----------
<code>NotImplementedError</code> | If X is given for a multi-run fit (a single new design is ambiguous across runs — fit per run instead).
<code>ValueError</code> | If X's column count does not match the fitted design.

(models-report)=
##### `report`

```python
report(contrasts = None, **kwargs)
```

Generate a nilearn HTML report for the fitted GLM.

Delegates to the underlying `FirstLevelModel.generate_report`, which
renders the design matrix, requested contrast maps, and model
parameters as a self-contained HTML report.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`contrasts` | <code>str \| list \| dict</code> | Contrast(s) to render, in the same forms as `compute_contrast`. | <code>None</code>
`**kwargs` | <code>dict</code> | Forwarded to nilearn's `generate_report` (e.g. `title`, `threshold`, `alpha`). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>HTMLReport</code> | nilearn report object; call `.save_as_html(path)` or     display it in a notebook.

(models-score)=
##### `score`

```python
score(X: None = None, y: None = None) -> float
```

Return mean R² across voxels and runs.

Computes average coefficient of determination (R²) from the fitted GLM.
Higher values indicate better model fit.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>None</code> | Not used; present for sklearn API compatibility. | <code>None</code>
`y` | <code>None</code> | Not used; present for sklearn API compatibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>float</code> | Mean R² across all non-NaN voxels and all runs, in `[0, 1]`.

<details class="note" open markdown="1">
<summary>Note</summary>

Averages nilearn's per-run `r_square_` maps. For voxel-wise R² maps,
access `glm_.r_square_` directly.

</details>

**Examples:**

```python
brain.fit(model="glm", X=design_matrix)
r2 = brain.model_.score()
```

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
