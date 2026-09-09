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
[`Ridge`](#models-ridge) | Ridge regression with optional GPU acceleration and banded ridge support.



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
Ridge(*, alpha: float | str = 1.0, cv: int | None = None, alphas: list[float] | np.ndarray | None = None, n_iter: int = 100, concentration: float | list[float] | None = None, device: str = 'cpu', local_alpha: bool = True, fit_intercept: bool = False, conservative: bool = False, random_state: int | None = None, progress_bar: bool = False)
```

Ridge regression with optional GPU acceleration and banded ridge support.

Wraps nltools SVD-based ridge regression algorithms with
scikit-learn compatible API. Supports single and multi-target
regression with optional GPU acceleration via PyTorch.

Supports both regular ridge (single feature space) and banded ridge
(multiple feature spaces). The model detects the input type: an array `X`
is a single feature space; a list `X` is multiple feature spaces and runs
banded (group) ridge with a random search over feature-space weights.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`alpha` | <code>float or 'auto', default=1.0</code> | Regularization strength. If 'auto', uses cross-validation to select optimal alpha from alphas parameter. | <code>1.0</code>
`cv` | <code>int or None, default=None</code> | Number of cross-validation folds (only used if alpha='auto') | <code>None</code>
`alphas` | <code>array-like or None, default=None</code> | Alpha values to try during cross-validation. Defaults to [0.1, 1.0, 10.0] if None. | <code>None</code>
`n_iter` | <code>int, default=100</code> | Number of random search iterations. Only used when X is a list (multiple feature spaces). Ignored for single feature space. | <code>100</code>
`concentration` | <code>float or list, default=[0.1, 1.0]</code> | Concentration parameter(s) for Dirichlet sampling of feature-space weights. Only used when X is a list. A value of 1 samples uniformly over the simplex, infinity gives equal weights, and a list is cycled through across iterations. | <code>None</code>
`device` | <code>str, default='cpu'</code> | Compute device. One of `'cpu'` (NumPy), `'gpu'` (PyTorch on CUDA/MPS when available, else torch-CPU), or `'auto'` (use a GPU if one is present, otherwise NumPy). Selects *where* the SVD/CV math runs; distinct from any CPU-core parallelism. | <code>'cpu'</code>
`local_alpha` | <code>bool, default=True</code> | If True, select best alpha independently for each target. If False, select single best alpha for all targets. | <code>True</code>
`fit_intercept` | <code>bool, default=False</code> | Whether to fit an intercept. | <code>False</code>
`conservative` | <code>bool, default=False</code> | If True, select largest alpha within 1 std of best score (more regularization). | <code>False</code>
`random_state` | <code>int or None, default=None</code> | Random seed for reproducibility (used for CV splits and random search) | <code>None</code>
`progress_bar` | <code>bool, default=False</code> | Whether to display progress bar during banded ridge fitting (when X is a list). Requires tqdm. Not used for single feature space ridge regression. | <code>False</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`coef_` | <code>ndarray of shape (n_features,) or (n_features, n_targets</code> | Ridge coefficients
`alpha_` | <code>float or ndarray</code> | Alpha value(s) used (selected via CV if alpha='auto')
`cv_scores_` | <code>ndarray</code> | Cross-validation scores (only if alpha='auto')
`deltas_` | <code>ndarray or None</code> | Feature space weights (only if X was a list) Shape: (n_spaces, n_targets). deltas = log(gamma / alpha)
`backend_` | <code>[Backend](#backends-backend)</code> | Resolved backend instance used for computation (its `.name` reports the concrete device, e.g. `'torch-cuda'`).
`is_fitted_` | <code>bool</code> | Whether the model has been fitted

**Methods:**

Name | Description
---- | -----------
[`fit`](#models-fit) | Fit ridge regression model.
[`predict`](#models-predict) | Predict using the ridge model.
[`score`](#models-score) | Return the coefficient of determination R^2 of the prediction.



**Examples:**

```python
import numpy as np
from nltools.models import Ridge

X = np.random.randn(100, 50)
y = np.random.randn(100)
model = Ridge(alpha=1.0)
model.fit(X, y)
y_pred = model.predict(X)

# Banded ridge with multiple feature spaces (automatic detection)
X1 = np.random.randn(100, 30)
X2 = np.random.randn(100, 20)
model = Ridge(alpha='auto', cv=5, n_iter=50)
model.fit([X1, X2], y)
print(f"Feature space weights: {model.deltas_}")
```

#### Methods

##### `fit`

```python
fit(X: np.ndarray | list[np.ndarray], y: np.ndarray) -> Ridge
```

Fit ridge regression model.

Supports both regular ridge (single feature space) and banded ridge
(multiple feature spaces). If X is a list, banded ridge is used.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray of shape (n_samples, n_features) or list of arrays</code> | Training data. If list, each element is a feature space for banded ridge. | *required*
`y` | <code>ndarray of shape (n_samples,) or (n_samples, n_targets)</code> | Target values | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Ridge](#models-ridge)</code> | Fitted model instance

##### `predict`

```python
predict(X: np.ndarray) -> np.ndarray
```

Predict using the ridge model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray of shape (n_samples, n_features)</code> | Samples to predict | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Predicted values, shape ``(n_samples,)`` or     ``(n_samples, n_targets)``.

##### `score`

```python
score(X: np.ndarray, y: np.ndarray) -> float | np.ndarray
```

Return the coefficient of determination R^2 of the prediction.

For multi-target regression (y is 2D), returns per-target R² scores.
For single-target regression (y is 1D), returns a scalar R².

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray of shape (n_samples, n_features)</code> | Test samples | *required*
`y` | <code>ndarray of shape (n_samples,) or (n_samples, n_targets)</code> | True values for X | *required*

**Returns:**

Type | Description
---- | -----------
<code>float \| ndarray</code> | A scalar R² when `y` is 1-D; an array of shape     `(n_targets,)` with per-target R² scores when `y` is 2-D.
