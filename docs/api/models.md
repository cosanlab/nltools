(models-models)=
## `models`

Model classes for neuroimaging analysis.

Provides sklearn-compatible APIs for common neuroimaging analyses.

**Modules:**

Name | Description
---- | -----------
[`base`](#models-base) | Base classes for nltools models.
[`glm`](#models-glm) | GLM model for neuroimaging data.
[`ridge`](#models-ridge) | Ridge regression model for neuroimaging data.

**Classes:**

Name | Description
---- | -----------
[`BaseModel`](#models-basemodel) | Abstract base class for all nltools models.
[`Glm`](#models-glm) | General Linear Model for fMRI data analysis with sklearn-compatible API.
[`Ridge`](#models-ridge) | Ridge regression with optional GPU acceleration and banded ridge support.



### Classes

(models-basemodel)=
#### `BaseModel`

```python
BaseModel()
```

Bases: <code>[ABC](#abc.ABC)</code>

Abstract base class for all nltools models.

Follows scikit-learn API conventions:
- fit(X, y) trains the model and returns self
- predict(X) generates predictions
- score(X, y) evaluates model performance

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`n_features_in_` | <code>[int](#int)</code> | Number of features seen during fit
`n_samples_` | <code>[int](#int)</code> | Number of samples seen during fit
`is_fitted_` | <code>[bool](#bool)</code> | Whether the model has been fitted

**Methods:**

Name | Description
---- | -----------
[`fit`](#models-fit) | Fit the model to training data.
[`predict`](#models-predict) | Generate predictions for new data.
[`score`](#models-score) | Evaluate model performance.

##### Methods

(models-fit)=
###### `fit`

```python
fit(X, y)
```

Fit the model to training data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray of shape (n_samples, n_features)</code> | Training data | *required*
`y` | <code>ndarray of shape (n_samples,) or (n_samples, n_targets)</code> | Target values | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BaseModel` |  | Fitted model instance

(models-predict)=
###### `predict`

```python
predict(X)
```

Generate predictions for new data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray of shape (n_samples, n_features)</code> | Data to predict on | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`ndarray` |  | Predicted values

(models-score)=
###### `score`

```python
score(X, y)
```

Evaluate model performance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray of shape (n_samples, n_features)</code> | Test data | *required*
`y` | <code>ndarray of shape (n_samples,) or (n_samples, n_targets)</code> | True values | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`float` |  | Model performance metric

(models-glm)=
#### `Glm`

```python
Glm(t_r = None, noise_model = 'ols', smoothing_fwhm = None, mask = None, progress_bar = False, **kwargs)
```

Bases: <code>[BaseModel](#nltools.models.base.BaseModel)</code>

General Linear Model for fMRI data analysis with sklearn-compatible API.

Wraps nilearn.glm.first_level.FirstLevelModel using composition pattern,
similar to how BrainData holds masker objects. Provides sklearn-style
interface (fit/predict/score) while exposing full nilearn GLM functionality.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>[float](#float)</code> | Repetition time (TR) in seconds. If None, will be inferred from data. | <code>None</code>
`noise_model` | <code>str, default='ols'</code> | Noise model for temporal autocorrelation ('ols' or 'ar1').<br>- 'ols': Ordinary Least Squares (assumes independent errors) - 'ar1': Autoregressive AR(1) model (accounts for temporal correlation) | <code>'ols'</code>
`smoothing_fwhm` | <code>[float](#float)</code> | Full-Width at Half Maximum (FWHM) in mm for spatial smoothing. If None, no smoothing is applied. | <code>None</code>
`mask` | <code>[Nifti1Image](#Nifti1Image)</code> | Mask image defining voxels to include in analysis. If None, uses MNI template mask (default, like BrainData). | <code>None</code>
`**kwargs` |  | Additional arguments passed to nilearn FirstLevelModel. | <code>{}</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`is_fitted_` | <code>[bool](#bool)</code> | Whether the model has been fitted

<details class="note" open markdown="1">
<summary>Note</summary>

Access fitted results via properties: ``glm_``, ``residuals``, ``design_matrices_``

</details>

**Examples:**

```pycon
>>> from nltools.models import GLMModel
>>> from nilearn.glm.first_level import make_first_level_design_matrix
>>> import pandas as pd
>>> import numpy as np
>>> from nibabel import Nifti1Image
>>>
>>> # Create synthetic fMRI data
>>> n_scans = 100
>>> fmri_data = np.random.randn(n_scans, 20, 20, 20)
>>> img = Nifti1Image(fmri_data.T, np.eye(4))
>>>
>>> # Create design matrix
>>> frame_times = np.arange(n_scans) * 2.0
>>> events = pd.DataFrame({
...     'onset': [10, 30, 50, 70],
...     'duration': [1, 1, 1, 1],
...     'trial_type': ['task', 'task', 'task', 'task']
... })
>>> design_matrix = make_first_level_design_matrix(frame_times, events)
>>>
>>> # Fit GLM
>>> model = GLMModel(t_r=2.0, noise_model='ar1')
>>> model.fit(img, design_matrices=design_matrix)
>>>
>>> # Compute contrast
>>> task_effect = model.compute_contrast('task', output_type='stat')
>>>
>>> # Get fitted values
>>> fitted_values = model.predict()
>>>
>>> # Access residuals
>>> residuals = model.residuals
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

Unlike Ridge which works with 2D arrays (samples × features), GLMModel
works with 4D neuroimaging data (x × y × z × time) and design matrices.
Therefore, it does not use BaseModel's input validation methods.

The predict() method follows sklearn's LinearRegression semantics:
- predict() returns fitted values (predictions on training data)
- predict(X) would generate predictions with new design matrix (future feature)

For advanced use cases, access the internal FirstLevelModel via the
``glm_`` property to use any nilearn-specific functionality.

</details>

**Methods:**

Name | Description
---- | -----------
[`compute_contrast`](#models-compute-contrast) | Compute contrast using nilearn for accurate statistical inference.
[`fit`](#models-fit) | Fit GLM to fMRI data.
[`predict`](#models-predict) | Generate predictions from fitted GLM.
[`score`](#models-score) | Return mean R² across voxels and runs.

##### Methods

(models-compute-contrast)=
###### `compute_contrast`

```python
compute_contrast(contrast_def, output_type = 'stat')
```

Compute contrast using nilearn for accurate statistical inference.

This is the primary method for extracting results from a fitted GLM.
Delegates to nilearn's FirstLevelModel.compute_contrast() for proper
statistical inference with correct degrees of freedom, etc.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`contrast_def` | <code>str, array-like, or dict</code> | Contrast specification: - str: Regressor name (e.g., 'task') - array-like: Contrast vector (e.g., [1, -1, 0, 0]) - dict: Multiple contrasts with names as keys | *required*
`output_type` | <code>str, default='stat'</code> | Type of output to return: - 'stat': T-statistic map (default) - 'z_score': Z-score map - 'p_value': P-value map - 'effect_size': Effect size (beta) map - 'effect_variance': Variance of effect size - 'all': Dictionary with all output types | <code>'stat'</code>

**Returns:**

Type | Description
---- | -----------
 | Nifti1Image or dict: Contrast map(s). If output_type='all', returns dict with all maps.

**Examples:**

```pycon
>>> # After fitting model
>>> model.fit(img, design_matrices=design_matrix)
>>>
>>> # Simple contrast by name
>>> t_map = model.compute_contrast('task')
>>>
>>> # Contrast vector
>>> contrast_map = model.compute_contrast([1, -1, 0])
>>>
>>> # Get all outputs
>>> results = model.compute_contrast('task', output_type='all')
>>> t_map = results['stat']
>>> p_map = results['p_value']
```

###### `fit`

```python
fit(X, y = None, design_matrices = None, events = None, **kwargs)
```

Fit GLM to fMRI data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>Nifti1Image or list of Nifti1Image</code> | 4D fMRI image(s) to fit. Can be single run or list of runs. | *required*
`y` | <code>None</code> | Not used, present for sklearn API compatibility. | <code>None</code>
`design_matrices` | <code>DataFrame, DesignMatrix, or list of DataFrame/DesignMatrix</code> | Design matrix or list of design matrices (one per run). Each should have shape (n_scans, n_regressors). Accepts both pandas DataFrames and nltools DesignMatrix objects. | <code>None</code>
`events` | <code>DataFrame or list of DataFrame</code> | Event specifications for automatic design matrix creation. Alternative to providing design_matrices directly. | <code>None</code>
`**kwargs` |  | Additional arguments passed to FirstLevelModel.fit() | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`GLMModel` |  | Fitted model instance (for method chaining)

<details class="notes" open markdown="1">
<summary>Notes</summary>

Unlike BaseModel's fit(), this method does not validate X as a 2D array
because GLM works with 4D neuroimaging data. Input validation is
delegated to nilearn's FirstLevelModel.

DesignMatrix objects are automatically converted to pandas DataFrames
for nilearn compatibility. The conversion is done at this boundary to
keep DesignMatrix Polars-native while maintaining nilearn integration.

</details>

###### `predict`

```python
predict(X = None)
```

Generate predictions from fitted GLM.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>DataFrame or None, default=None</code> | Design matrix for generating predictions.<br>- If None: returns fitted values (predictions on training data) - If DataFrame: generates predictions using new design matrix   (not yet implemented) | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | list of Nifti1Image: Predicted brain activity for each run

<details class="notes" open markdown="1">
<summary>Notes</summary>

Follows sklearn's LinearRegression semantics where predict() without
arguments returns fitted values (like calling predict(X_train)).

Future enhancement will support predict(X=new_design_matrix) to
generate predictions with different experimental designs.

</details>

###### `score`

```python
score(X = None, y = None)
```

Return mean R² across voxels and runs.

Computes average coefficient of determination (R²) from the fitted GLM.
Higher values indicate better model fit.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>None</code> | Not used, present for sklearn API compatibility. | <code>None</code>
`y` | <code>None</code> | Not used, present for sklearn API compatibility. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`float` |  | Mean R² across all voxels and runs. Range: [0, 1], higher is better.

<details class="notes" open markdown="1">
<summary>Notes</summary>

Extracts R² values from nilearn's FirstLevelModel.r_square attribute,
which returns a list of Nifti1Image objects (one per run).
Computes the mean across all non-NaN voxels and all runs.

For voxel-wise R² maps, access `glm_.r_square` directly.

</details>

**Examples:**

```pycon
>>> brain.fit(model='glm', X=design_matrix)
>>> r2 = brain.model_.score()
>>> print(f"Mean R²: {r2:.3f}")
```

(models-ridge)=
#### `Ridge`

```python
Ridge(alpha = 1.0, cv = None, alphas = None, n_iter = 100, concentration = [0.1, 1.0], backend = 'numpy', local_alpha = True, fit_intercept = False, conservative = False, random_state = None, progress_bar = False)
```

Bases: <code>[BaseModel](#nltools.models.base.BaseModel)</code>

Ridge regression with optional GPU acceleration and banded ridge support.

Wraps nltools SVD-based ridge regression algorithms with
scikit-learn compatible API. Supports single and multi-target
regression with optional GPU acceleration via PyTorch.

    Supports both regular ridge (single feature space) and banded ridge
    (multiple feature spaces). The model automatically detects the input type:
    - Array X: Single feature space → uses solve_ridge_cv
    - List X: Multiple feature spaces → uses solve_banded_ridge_cv (true banded/group ridge)

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`alpha` | <code>float or 'auto', default=1.0</code> | Regularization strength. If 'auto', uses cross-validation to select optimal alpha from alphas parameter. | <code>1.0</code>
`cv` | <code>int or None, default=None</code> | Number of cross-validation folds (only used if alpha='auto') | <code>None</code>
`alphas` | <code>array-like or None, default=None</code> | Alpha values to try during cross-validation. Defaults to [0.1, 1.0, 10.0] if None. | <code>None</code>
`n_iter` | <code>int, default=100</code> | Number of random search iterations. Only used when X is a list (multiple feature spaces). Ignored for single feature space. | <code>100</code>
`concentration` | <code>float or list, default=[0.1, 1.0]</code> | Concentration parameters for Dirichlet sampling. Only used when X is a list (multiple feature spaces). - A value of 1 corresponds to uniform sampling over the simplex. - A value of infinity corresponds to equal weights. - If a list, samples cycle through the list. | <code>[0.1, 1.0]</code>
`backend` | <code>str or Backend, default='numpy'</code> | Computational backend ('numpy', 'torch', or 'auto') | <code>'numpy'</code>
`local_alpha` | <code>bool, default=True</code> | If True, select best alpha independently for each target. If False, select single best alpha for all targets. | <code>True</code>
`fit_intercept` | <code>bool, default=False</code> | Whether to fit an intercept. | <code>False</code>
`conservative` | <code>bool, default=False</code> | If True, select largest alpha within 1 std of best score (more regularization). | <code>False</code>
`random_state` | <code>int or None, default=None</code> | Random seed for reproducibility (used for CV splits and random search) | <code>None</code>
`progress_bar` | <code>bool, default=False</code> | Whether to display progress bar during banded ridge fitting (when X is a list). Requires tqdm. Not used for single feature space ridge regression. | <code>False</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`coef_` | <code>ndarray of shape (n_features,) or (n_features, n_targets</code> | Ridge coefficients
`alpha_` | <code>[float](#float) or [ndarray](#ndarray)</code> | Alpha value(s) used (selected via CV if alpha='auto')
`cv_scores_` | <code>[ndarray](#ndarray)</code> | Cross-validation scores (only if alpha='auto')
`deltas_` | <code>[ndarray](#ndarray) or None</code> | Feature space weights (only if X was a list) Shape: (n_spaces, n_targets). deltas = log(gamma / alpha)
`backend_` | <code>[Backend](#Backend)</code> | Backend instance used for computation

**Examples:**

```pycon
>>> from nltools.models import Ridge
>>> import numpy as np
>>> X = np.random.randn(100, 50)
>>> y = np.random.randn(100)
>>> model = Ridge(alpha=1.0)
>>> model.fit(X, y)
Ridge(alpha=1.0, backend='numpy')
>>> y_pred = model.predict(X)
>>>
>>> # Banded ridge with multiple feature spaces (automatic detection)
>>> X1 = np.random.randn(100, 30)
>>> X2 = np.random.randn(100, 20)
>>> model = Ridge(alpha='auto', cv=5, n_iter=50)
>>> model.fit([X1, X2], y)
>>> print(f"Feature space weights: {model.deltas_}")
```

**Methods:**

Name | Description
---- | -----------
[`fit`](#models-fit) | Fit ridge regression model.
[`predict`](#models-predict) | Predict using the ridge model.
[`score`](#models-score) | Return the coefficient of determination R^2 of the prediction.

##### Methods

###### `fit`

```python
fit(X, y)
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

Name | Type | Description
---- | ---- | -----------
`Ridge` |  | Fitted model instance

###### `predict`

```python
predict(X)
```

Predict using the ridge model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray of shape (n_samples, n_features)</code> | Samples to predict | *required*

**Returns:**

Type | Description
---- | -----------
 | ndarray of shape (n_samples,) or (n_samples, n_targets): Predicted values

###### `score`

```python
score(X, y)
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
 | float or ndarray: - If y is 1D: scalar R² - If y is 2D: array of shape (n_targets,) with per-target R² scores



### Modules

(models-base)=
#### `base`

Base classes for nltools models.

Provides sklearn-compatible API for neuroimaging analysis.

**Classes:**

Name | Description
---- | -----------
[`BaseModel`](#models-basemodel) | Abstract base class for all nltools models.



##### Classes

###### `BaseModel`

```python
BaseModel()
```

Bases: <code>[ABC](#abc.ABC)</code>

Abstract base class for all nltools models.

Follows scikit-learn API conventions:
- fit(X, y) trains the model and returns self
- predict(X) generates predictions
- score(X, y) evaluates model performance

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`n_features_in_` | <code>[int](#int)</code> | Number of features seen during fit
`n_samples_` | <code>[int](#int)</code> | Number of samples seen during fit
`is_fitted_` | <code>[bool](#bool)</code> | Whether the model has been fitted

**Methods:**

Name | Description
---- | -----------
[`fit`](#models-fit) | Fit the model to training data.
[`predict`](#models-predict) | Generate predictions for new data.
[`score`](#models-score) | Evaluate model performance.



####### Attributes##

(models-is-fitted)=
###### `is_fitted_`

```python
is_fitted_ = False
```



####### Functions##

###### `fit`

```python
fit(X, y)
```

Fit the model to training data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray of shape (n_samples, n_features)</code> | Training data | *required*
`y` | <code>ndarray of shape (n_samples,) or (n_samples, n_targets)</code> | Target values | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BaseModel` |  | Fitted model instance

######## `predict`

```python
predict(X)
```

Generate predictions for new data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray of shape (n_samples, n_features)</code> | Data to predict on | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`ndarray` |  | Predicted values

######## `score`

```python
score(X, y)
```

Evaluate model performance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray of shape (n_samples, n_features)</code> | Test data | *required*
`y` | <code>ndarray of shape (n_samples,) or (n_samples, n_targets)</code> | True values | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`float` |  | Model performance metric

#### `glm`

GLM model for neuroimaging data.

Wraps nilearn.glm.first_level.FirstLevelModel with sklearn-compatible API.

**Classes:**

Name | Description
---- | -----------
[`Glm`](#models-glm) | General Linear Model for fMRI data analysis with sklearn-compatible API.



##### Classes

###### `Glm`

```python
Glm(t_r = None, noise_model = 'ols', smoothing_fwhm = None, mask = None, progress_bar = False, **kwargs)
```

Bases: <code>[BaseModel](#nltools.models.base.BaseModel)</code>

General Linear Model for fMRI data analysis with sklearn-compatible API.

Wraps nilearn.glm.first_level.FirstLevelModel using composition pattern,
similar to how BrainData holds masker objects. Provides sklearn-style
interface (fit/predict/score) while exposing full nilearn GLM functionality.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>[float](#float)</code> | Repetition time (TR) in seconds. If None, will be inferred from data. | <code>None</code>
`noise_model` | <code>str, default='ols'</code> | Noise model for temporal autocorrelation ('ols' or 'ar1').<br>- 'ols': Ordinary Least Squares (assumes independent errors) - 'ar1': Autoregressive AR(1) model (accounts for temporal correlation) | <code>'ols'</code>
`smoothing_fwhm` | <code>[float](#float)</code> | Full-Width at Half Maximum (FWHM) in mm for spatial smoothing. If None, no smoothing is applied. | <code>None</code>
`mask` | <code>[Nifti1Image](#Nifti1Image)</code> | Mask image defining voxels to include in analysis. If None, uses MNI template mask (default, like BrainData). | <code>None</code>
`**kwargs` |  | Additional arguments passed to nilearn FirstLevelModel. | <code>{}</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`is_fitted_` | <code>[bool](#bool)</code> | Whether the model has been fitted

<details class="note" open markdown="1">
<summary>Note</summary>

Access fitted results via properties: ``glm_``, ``residuals``, ``design_matrices_``

</details>

**Examples:**

```pycon
>>> from nltools.models import GLMModel
>>> from nilearn.glm.first_level import make_first_level_design_matrix
>>> import pandas as pd
>>> import numpy as np
>>> from nibabel import Nifti1Image
>>>
>>> # Create synthetic fMRI data
>>> n_scans = 100
>>> fmri_data = np.random.randn(n_scans, 20, 20, 20)
>>> img = Nifti1Image(fmri_data.T, np.eye(4))
>>>
>>> # Create design matrix
>>> frame_times = np.arange(n_scans) * 2.0
>>> events = pd.DataFrame({
...     'onset': [10, 30, 50, 70],
...     'duration': [1, 1, 1, 1],
...     'trial_type': ['task', 'task', 'task', 'task']
... })
>>> design_matrix = make_first_level_design_matrix(frame_times, events)
>>>
>>> # Fit GLM
>>> model = GLMModel(t_r=2.0, noise_model='ar1')
>>> model.fit(img, design_matrices=design_matrix)
>>>
>>> # Compute contrast
>>> task_effect = model.compute_contrast('task', output_type='stat')
>>>
>>> # Get fitted values
>>> fitted_values = model.predict()
>>>
>>> # Access residuals
>>> residuals = model.residuals
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

Unlike Ridge which works with 2D arrays (samples × features), GLMModel
works with 4D neuroimaging data (x × y × z × time) and design matrices.
Therefore, it does not use BaseModel's input validation methods.

The predict() method follows sklearn's LinearRegression semantics:
- predict() returns fitted values (predictions on training data)
- predict(X) would generate predictions with new design matrix (future feature)

For advanced use cases, access the internal FirstLevelModel via the
``glm_`` property to use any nilearn-specific functionality.

</details>

**Methods:**

Name | Description
---- | -----------
[`compute_contrast`](#models-compute-contrast) | Compute contrast using nilearn for accurate statistical inference.
[`fit`](#models-fit) | Fit GLM to fMRI data.
[`predict`](#models-predict) | Generate predictions from fitted GLM.
[`score`](#models-score) | Return mean R² across voxels and runs.



####### Attributes##

(models-design-matrices)=
###### `design_matrices_`

```python
design_matrices_
```

Design matrices used in fitting.

**Returns:**

Type | Description
---- | -----------
 | list of DataFrame: Design matrices for each run

######## `glm_`

```python
glm_
```

Access internal FirstLevelModel for advanced use.

Provides direct access to the wrapped nilearn FirstLevelModel
instance for advanced users who need functionality not exposed
by the sklearn-compatible interface.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`FirstLevelModel` |  | Internal nilearn FirstLevelModel instance

**Examples:**

```pycon
>>> # Access nilearn-specific attributes
>>> model.glm_.labels_
>>> model.glm_.results_
>>>
>>> # Use nilearn-specific methods
>>> model.glm_.generate_report()
```

######## `is_fitted_`

```python
is_fitted_ = False
```

######## `mask`

```python
mask = nib.load(get_brainspace().mask)
```

######## `noise_model`

```python
noise_model = noise_model
```

######## `progress_bar`

```python
progress_bar = progress_bar
```

######## `residuals`

```python
residuals
```

Residuals from fitted GLM.

**Returns:**

Type | Description
---- | -----------
 | list of Nifti1Image: Residual images for each run (observed - predicted)

######## `smoothing_fwhm`

```python
smoothing_fwhm = smoothing_fwhm
```

######## `t_r`

```python
t_r = t_r
```



####### Functions##

###### `compute_contrast`

```python
compute_contrast(contrast_def, output_type = 'stat')
```

Compute contrast using nilearn for accurate statistical inference.

This is the primary method for extracting results from a fitted GLM.
Delegates to nilearn's FirstLevelModel.compute_contrast() for proper
statistical inference with correct degrees of freedom, etc.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`contrast_def` | <code>str, array-like, or dict</code> | Contrast specification: - str: Regressor name (e.g., 'task') - array-like: Contrast vector (e.g., [1, -1, 0, 0]) - dict: Multiple contrasts with names as keys | *required*
`output_type` | <code>str, default='stat'</code> | Type of output to return: - 'stat': T-statistic map (default) - 'z_score': Z-score map - 'p_value': P-value map - 'effect_size': Effect size (beta) map - 'effect_variance': Variance of effect size - 'all': Dictionary with all output types | <code>'stat'</code>

**Returns:**

Type | Description
---- | -----------
 | Nifti1Image or dict: Contrast map(s). If output_type='all', returns dict with all maps.

**Examples:**

```pycon
>>> # After fitting model
>>> model.fit(img, design_matrices=design_matrix)
>>>
>>> # Simple contrast by name
>>> t_map = model.compute_contrast('task')
>>>
>>> # Contrast vector
>>> contrast_map = model.compute_contrast([1, -1, 0])
>>>
>>> # Get all outputs
>>> results = model.compute_contrast('task', output_type='all')
>>> t_map = results['stat']
>>> p_map = results['p_value']
```

######## `fit`

```python
fit(X, y = None, design_matrices = None, events = None, **kwargs)
```

Fit GLM to fMRI data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>Nifti1Image or list of Nifti1Image</code> | 4D fMRI image(s) to fit. Can be single run or list of runs. | *required*
`y` | <code>None</code> | Not used, present for sklearn API compatibility. | <code>None</code>
`design_matrices` | <code>DataFrame, DesignMatrix, or list of DataFrame/DesignMatrix</code> | Design matrix or list of design matrices (one per run). Each should have shape (n_scans, n_regressors). Accepts both pandas DataFrames and nltools DesignMatrix objects. | <code>None</code>
`events` | <code>DataFrame or list of DataFrame</code> | Event specifications for automatic design matrix creation. Alternative to providing design_matrices directly. | <code>None</code>
`**kwargs` |  | Additional arguments passed to FirstLevelModel.fit() | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`GLMModel` |  | Fitted model instance (for method chaining)

<details class="notes" open markdown="1">
<summary>Notes</summary>

Unlike BaseModel's fit(), this method does not validate X as a 2D array
because GLM works with 4D neuroimaging data. Input validation is
delegated to nilearn's FirstLevelModel.

DesignMatrix objects are automatically converted to pandas DataFrames
for nilearn compatibility. The conversion is done at this boundary to
keep DesignMatrix Polars-native while maintaining nilearn integration.

</details>

######## `predict`

```python
predict(X = None)
```

Generate predictions from fitted GLM.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>DataFrame or None, default=None</code> | Design matrix for generating predictions.<br>- If None: returns fitted values (predictions on training data) - If DataFrame: generates predictions using new design matrix   (not yet implemented) | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | list of Nifti1Image: Predicted brain activity for each run

<details class="notes" open markdown="1">
<summary>Notes</summary>

Follows sklearn's LinearRegression semantics where predict() without
arguments returns fitted values (like calling predict(X_train)).

Future enhancement will support predict(X=new_design_matrix) to
generate predictions with different experimental designs.

</details>

######## `score`

```python
score(X = None, y = None)
```

Return mean R² across voxels and runs.

Computes average coefficient of determination (R²) from the fitted GLM.
Higher values indicate better model fit.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>None</code> | Not used, present for sklearn API compatibility. | <code>None</code>
`y` | <code>None</code> | Not used, present for sklearn API compatibility. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`float` |  | Mean R² across all voxels and runs. Range: [0, 1], higher is better.

<details class="notes" open markdown="1">
<summary>Notes</summary>

Extracts R² values from nilearn's FirstLevelModel.r_square attribute,
which returns a list of Nifti1Image objects (one per run).
Computes the mean across all non-NaN voxels and all runs.

For voxel-wise R² maps, access `glm_.r_square` directly.

</details>

**Examples:**

```pycon
>>> brain.fit(model='glm', X=design_matrix)
>>> r2 = brain.model_.score()
>>> print(f"Mean R²: {r2:.3f}")
```



##### Methods

#### `ridge`

Ridge regression model for neuroimaging data.

Wraps nltools.algorithms.ridge with sklearn-compatible API.
Supports both regular ridge (single feature space) and banded ridge
(multiple feature spaces) with optional random search over feature weights.

**Classes:**

Name | Description
---- | -----------
[`Ridge`](#models-ridge) | Ridge regression with optional GPU acceleration and banded ridge support.



##### Classes

###### `Ridge`

```python
Ridge(alpha = 1.0, cv = None, alphas = None, n_iter = 100, concentration = [0.1, 1.0], backend = 'numpy', local_alpha = True, fit_intercept = False, conservative = False, random_state = None, progress_bar = False)
```

Bases: <code>[BaseModel](#nltools.models.base.BaseModel)</code>

Ridge regression with optional GPU acceleration and banded ridge support.

Wraps nltools SVD-based ridge regression algorithms with
scikit-learn compatible API. Supports single and multi-target
regression with optional GPU acceleration via PyTorch.

    Supports both regular ridge (single feature space) and banded ridge
    (multiple feature spaces). The model automatically detects the input type:
    - Array X: Single feature space → uses solve_ridge_cv
    - List X: Multiple feature spaces → uses solve_banded_ridge_cv (true banded/group ridge)

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`alpha` | <code>float or 'auto', default=1.0</code> | Regularization strength. If 'auto', uses cross-validation to select optimal alpha from alphas parameter. | <code>1.0</code>
`cv` | <code>int or None, default=None</code> | Number of cross-validation folds (only used if alpha='auto') | <code>None</code>
`alphas` | <code>array-like or None, default=None</code> | Alpha values to try during cross-validation. Defaults to [0.1, 1.0, 10.0] if None. | <code>None</code>
`n_iter` | <code>int, default=100</code> | Number of random search iterations. Only used when X is a list (multiple feature spaces). Ignored for single feature space. | <code>100</code>
`concentration` | <code>float or list, default=[0.1, 1.0]</code> | Concentration parameters for Dirichlet sampling. Only used when X is a list (multiple feature spaces). - A value of 1 corresponds to uniform sampling over the simplex. - A value of infinity corresponds to equal weights. - If a list, samples cycle through the list. | <code>[0.1, 1.0]</code>
`backend` | <code>str or Backend, default='numpy'</code> | Computational backend ('numpy', 'torch', or 'auto') | <code>'numpy'</code>
`local_alpha` | <code>bool, default=True</code> | If True, select best alpha independently for each target. If False, select single best alpha for all targets. | <code>True</code>
`fit_intercept` | <code>bool, default=False</code> | Whether to fit an intercept. | <code>False</code>
`conservative` | <code>bool, default=False</code> | If True, select largest alpha within 1 std of best score (more regularization). | <code>False</code>
`random_state` | <code>int or None, default=None</code> | Random seed for reproducibility (used for CV splits and random search) | <code>None</code>
`progress_bar` | <code>bool, default=False</code> | Whether to display progress bar during banded ridge fitting (when X is a list). Requires tqdm. Not used for single feature space ridge regression. | <code>False</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`coef_` | <code>ndarray of shape (n_features,) or (n_features, n_targets</code> | Ridge coefficients
`alpha_` | <code>[float](#float) or [ndarray](#ndarray)</code> | Alpha value(s) used (selected via CV if alpha='auto')
`cv_scores_` | <code>[ndarray](#ndarray)</code> | Cross-validation scores (only if alpha='auto')
`deltas_` | <code>[ndarray](#ndarray) or None</code> | Feature space weights (only if X was a list) Shape: (n_spaces, n_targets). deltas = log(gamma / alpha)
`backend_` | <code>[Backend](#Backend)</code> | Backend instance used for computation

**Examples:**

```pycon
>>> from nltools.models import Ridge
>>> import numpy as np
>>> X = np.random.randn(100, 50)
>>> y = np.random.randn(100)
>>> model = Ridge(alpha=1.0)
>>> model.fit(X, y)
Ridge(alpha=1.0, backend='numpy')
>>> y_pred = model.predict(X)
>>>
>>> # Banded ridge with multiple feature spaces (automatic detection)
>>> X1 = np.random.randn(100, 30)
>>> X2 = np.random.randn(100, 20)
>>> model = Ridge(alpha='auto', cv=5, n_iter=50)
>>> model.fit([X1, X2], y)
>>> print(f"Feature space weights: {model.deltas_}")
```

**Methods:**

Name | Description
---- | -----------
[`fit`](#models-fit) | Fit ridge regression model.
[`predict`](#models-predict) | Predict using the ridge model.
[`score`](#models-score) | Return the coefficient of determination R^2 of the prediction.



####### Attributes##

(models-alpha)=
###### `alpha`

```python
alpha = alpha
```

######## `alphas`

```python
alphas = alphas if alphas is not None else [0.1, 1.0, 10.0]
```

######## `backend`

```python
backend = backend
```

######## `concentration`

```python
concentration = concentration
```

######## `conservative`

```python
conservative = conservative
```

######## `cv`

```python
cv = cv
```

######## `fit_intercept`

```python
fit_intercept = fit_intercept
```

######## `is_fitted_`

```python
is_fitted_ = False
```

######## `local_alpha`

```python
local_alpha = local_alpha
```

######## `n_iter`

```python
n_iter = n_iter
```

######## `progress_bar`

```python
progress_bar = progress_bar
```

######## `random_state`

```python
random_state = random_state
```



####### Functions##

###### `fit`

```python
fit(X, y)
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

Name | Type | Description
---- | ---- | -----------
`Ridge` |  | Fitted model instance

######## `predict`

```python
predict(X)
```

Predict using the ridge model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray of shape (n_samples, n_features)</code> | Samples to predict | *required*

**Returns:**

Type | Description
---- | -----------
 | ndarray of shape (n_samples,) or (n_samples, n_targets): Predicted values

######## `score`

```python
score(X, y)
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
 | float or ndarray: - If y is 1D: scalar R² - If y is 2D: array of shape (n_targets,) with per-target R² scores



##### Methods