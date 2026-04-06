## `modeling`

BrainData modeling functions.

Standalone functions extracted from BrainData class methods for model fitting,
cross-validation, GLM estimation, Ridge regression, and contrast computation.

**Methods:**

Name | Description
---- | -----------
[`compute_contrasts`](#compute_contrasts) | Compute contrasts from fitted GLM results.
[`compute_ridge_cv`](#compute_ridge_cv) | Compute cross-validation results for Ridge regression.
[`cv`](#cv) | Create a cross-validation pipeline for this BrainData.
[`fit`](#fit) | Fit a model to brain imaging data.
[`fit_glm`](#fit_glm) | Fit GLM model and extract results (same logic as current regress()).
[`fit_ridge`](#fit_ridge) | Fit Ridge model and extract results.
[`parse_contrast_string`](#parse_contrast_string) | Parse a contrast string into a numeric contrast vector.
[`regress`](#regress) | Deprecated: Use fit(model='glm', X=design_matrix) instead.
[`to_fit_dataclass`](#to_fit_dataclass) | Convert BrainData fit results to Fit dataclass.



### Methods

#### `compute_contrasts`

```python
compute_contrasts(bd, contrasts, contrast_type = 't')
```

Compute contrasts from fitted GLM results.

This method computes contrasts as linear combinations of the GLM beta coefficients.
Must be called after .fit(model='glm', X=design_matrix) has been run.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`contrasts` |  | Can be:<br>- str: A string specifying the contrast using column names   e.g., "conditionA - conditionB" or "2*conditionA - conditionB - conditionC" - dict: Dictionary with contrast names as keys and contrast strings/vectors as values   e.g., {"main_effect": "conditionA - conditionB", "interaction": [1, -1, -1, 1]} - array: Numeric contrast vector matching the number of regressors   e.g., [1, -1, 0, 0] for a 4-regressor model | *required*
`contrast_type` | <code>[str](#str)</code> | Type of contrast statistic ('t' or 'F'). Default: 't' Note: Currently only 't' contrasts are supported. | <code>'t'</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or dict: If single contrast, returns BrainData object with contrast map.                If multiple contrasts (dict input), returns dict of BrainData objects.

**Examples:**

```pycon
>>> # Fit GLM model
>>> design_matrix = pd.DataFrame({
...     'intercept': np.ones(n_samples),
...     'conditionA': signal_a,
...     'conditionB': signal_b
... })
>>> brain.fit(model='glm', X=design_matrix)
>>>
>>> # Simple numeric contrast: A - B
>>> contrast1 = brain.compute_contrasts([0, 1, -1])
>>>
>>> # String-based contrast (more readable)
>>> contrast2 = brain.compute_contrasts("conditionA - conditionB")
>>>
>>> # Multiple contrasts at once
>>> contrasts = {
...     "A_vs_B": "conditionA - conditionB",
...     "avg_effect": [0, 0.5, 0.5],
...     "weighted": "2*conditionA - conditionB"
... }
>>> results = brain.compute_contrasts(contrasts)
>>> # results is a dict: {"A_vs_B": BrainData, "avg_effect": BrainData, ...}
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- String contrasts support coefficients: "2*A - B" or "0.5*A + 0.5*B"
- Column names must match design matrix columns exactly (case-sensitive)
- Contrast weights should sum to zero for proper inference in most cases

</details>

#### `compute_ridge_cv`

```python
compute_ridge_cv(bd, X, cv, alpha = None, alphas = None, backend = 'auto', **kwargs)
```

Compute cross-validation results for Ridge regression.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`X` | <code>[ndarray](#ndarray) or [list](#list)</code> | Training features. If ndarray, shape (n_samples, n_features). If list, list of feature spaces for banded ridge. | *required*
`cv` | <code>int, 'auto', or sklearn CV splitter</code> | Cross-validation specification | *required*
`alpha` | <code>[float](#float) or [auto](#auto)</code> | Regularization strength (extracted from model if not provided) | <code>None</code>
`alphas` | <code>[array](#array) - [like](#like)</code> | Alpha values to try for alpha selection | <code>None</code>
`backend` | <code>[str](#str)</code> | Computational backend ('numpy', 'torch', 'auto'). Default: 'auto' | <code>'auto'</code>
`**kwargs` | <code>[dict](#dict)</code> | Additional arguments (currently unused, for future extensibility) | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | Dictionary containing: - 'scores': (n_folds, n_voxels) array of R-squared per fold - 'mean_score': (n_voxels,) array of mean R-squared across folds - 'predictions': BrainData of out-of-fold predictions - 'folds': (n_samples,) array of fold indices - 'best_alpha': Selected alpha (if alpha selection performed) - 'alpha_scores': (n_folds, n_alphas, n_voxels) array (if alpha selection)

#### `cv`

```python
cv(bd, k = None, scheme = 'kfold', split_by = None, groups = None, random_state = None, **kwargs)
```

Create a cross-validation pipeline for this BrainData.

Returns a Pipeline object that enables fluent, chainable transforms
with cross-validation. Terminal methods like .predict() execute the
pipeline and return results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`k` |  | Number of folds (for kfold scheme). Defaults to 5. | <code>None</code>
`scheme` |  | CV scheme type. Options: - 'kfold': k-fold cross-validation (default) - 'loro': leave-one-run-out (requires split_by='runs' or groups) - 'bootstrap': bootstrap with out-of-bag test sets | <code>'kfold'</code>
`split_by` |  | Attribute name for group splits (e.g., 'runs'). If provided and groups is None, will try to get groups from bd.X[split_by] if bd.X is a DataFrame. | <code>None</code>
`groups` |  | Explicit group labels for CV splits. | <code>None</code>
`random_state` |  | Random seed for reproducibility. | <code>None</code>
`**kwargs` |  | Additional arguments passed to CVScheme. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainDataPipeline` |  | A pipeline object for method chaining.

**Examples:**

```pycon
>>> # Simple 5-fold CV with prediction
>>> result = brain.cv(k=5).predict(y, algorithm='ridge')
>>> print(f"Mean score: {result.mean_score:.3f}")
```

```pycon
>>> # With preprocessing
>>> result = (brain
...     .cv(k=5)
...     .normalize()
...     .reduce(n_components=50)
...     .predict(y))
```

```pycon
>>> # Leave-one-run-out CV
>>> result = brain.cv(scheme='loro', groups=run_labels).predict(y)
```

<details class="see-also" open markdown="1">
<summary>See Also</summary>

BrainDataPipeline: For available transforms and terminal methods.
CVScheme: For CV scheme configuration details.

</details>

#### `fit`

```python
fit(bd, model = None, X = None, cv = None, inplace = True, progress_bar = None, scale = True, scale_value = 100.0, **kwargs)
```

Fit a model to brain imaging data.

Creates and fits a model from string specification. The brain data
(bd.data) is always used as the target variable. Model and results
are stored for later use with predict().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`model` | <code>[str](#str)</code> | Model type: 'ridge', 'glm', or future model names | <code>None</code>
`X` | <code>[array](#array) - [like](#like) or [DataFrame](#DataFrame)</code> | Design matrix or feature matrix, shape (n_samples, n_features) - For GLM: Design matrix with regressors (n_samples must match bd.data) - For Ridge: Feature matrix for prediction (n_samples must match bd.data) | <code>None</code>
`cv` | <code>int, 'auto', or sklearn CV splitter</code> | Cross-validation specification (Ridge only): - int: Number of folds for k-fold CV (returns CV scores) - 'auto': Triggers alpha selection via CV (implies alpha='auto') - sklearn CV object: Custom CV splitter (e.g., KFold(3, shuffle=True)) - None: No CV (default, backward compatible) | <code>None</code>
`inplace` | <code>bool, default=True</code> | If True, mutate bd and return bd (backward compatible). If False, return Fit dataclass with results (bd unchanged). | <code>True</code>
`progress_bar` | <code>[bool](#bool)</code> | Display progress bar during fitting. - If None: Uses bd.verbose (default) - If True: Shows progress bar for long-running operations - If False: No progress bar | <code>None</code>
`scale` | <code>bool, default=True</code> | Apply grand-mean scaling before fitting. Calls bd.scale(scale_value) which divides all values by the global mean and multiplies by scale_value. This puts data in percent signal change units, which is standard for fMRI analysis. | <code>True</code>
`scale_value` | <code>float, default=100.0</code> | Target value for mean after scaling. Only used if scale=True. | <code>100.0</code>
`**kwargs` | <code>[dict](#dict)</code> | Additional arguments passed to model constructor - Ridge: alpha, alphas, backend, random_state - Glm: noise_model, minimize_memory, etc. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or Fit: If ``inplace=True``, returns bd (fitted BrainData). If ``inplace=False``, returns Fit dataclass with results.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`The`](#The) | <code>following are set on bd when ``inplace=True``</code> | 
[```model_```](#nltools.data.braindata.modeling.fit.``model_``) | <code>[BaseModel](#BaseModel)</code> | Fitted model instance (Ridge, Glm, etc.)
[```X_```](#nltools.data.braindata.modeling.fit.``X_``) | <code>[ndarray](#ndarray)</code> | Training data X, stored for predict() default
[```cv_results_```](#nltools.data.braindata.modeling.fit.``cv_results_``) | <code>[dict](#dict)</code> | Cross-validation results dict with keys 'scores',
[`glm_betas`](#glm_betas) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Beta coefficients (for model='glm')
[`glm_t`](#glm_t) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | T-statistics (for model='glm')
[`glm_p`](#glm_p) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | P-values (for model='glm')
[`glm_se`](#glm_se) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Standard errors (for model='glm')
[`glm_residual`](#glm_residual) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Residuals (for model='glm')
[`glm_predicted`](#glm_predicted) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Fitted values (for model='glm')
[`glm_r2`](#glm_r2) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | R-squared values (for model='glm')
[`ridge_weights`](#ridge_weights) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Model coefficients (for model='ridge')
[`ridge_fitted_values`](#ridge_fitted_values) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Fitted values (for model='ridge')
[`ridge_scores`](#ridge_scores) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | R-squared scores (for model='ridge')

**Examples:**

```pycon
>>> # Old behavior (backward compatible): mutate self
>>> brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
>>> print(f"CV R2: {brain_data.cv_results_['mean_score'].mean():.3f}")
>>> weights = brain_data.ridge_weights  # Access as attribute
>>>
>>> # New behavior: return Fit dataclass (self unchanged)
>>> fit = brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features, inplace=False)
>>> assert isinstance(fit, Fit)
>>> assert 'weights' in fit.available()
>>> assert not hasattr(brain_data, 'ridge_weights')  # brain_data unchanged
>>> print(f"CV R2: {fit.cv_mean_score.mean():.3f}")
>>>
>>> # GLM with Fit dataclass
>>> fit_glm = brain_data.fit(model='glm', X=design_matrix, inplace=False)
>>> assert 'betas' in fit_glm.available()
>>> assert 't_stats' in fit_glm.available()
```

#### `fit_glm`

```python
fit_glm(bd, X)
```

Fit GLM model and extract results (same logic as current regress()).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`X` |  | Design matrix (DataFrame or DesignMatrix). | *required*

<details class="note" open markdown="1">
<summary>Note</summary>

Sets glm_betas, glm_t, glm_p, glm_se, glm_residual, glm_predicted,
glm_r2, and design_matrix on bd.

</details>

#### `fit_ridge`

```python
fit_ridge(bd, X, cv = None, **kwargs)
```

Fit Ridge model and extract results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`X` | <code>[ndarray](#ndarray)</code> | Training features | *required*
`cv` | <code>int, 'auto', or sklearn CV splitter</code> | Cross-validation specification | <code>None</code>
`**kwargs` | <code>[dict](#dict)</code> | Additional arguments for CV (alpha, alphas, backend, etc.) | <code>{}</code>

<details class="note" open markdown="1">
<summary>Note</summary>

Sets ridge_weights, ridge_fitted_values, ridge_scores, and
cv_results_ (if cv provided) on bd.

</details>

#### `parse_contrast_string`

```python
parse_contrast_string(bd, contrast_str)
```

Parse a contrast string into a numeric contrast vector.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`contrast_str` | <code>[str](#str)</code> | Contrast string like "A - B" or "2*A - B - C" | *required*

**Returns:**

Type | Description
---- | -----------
 | np.array: Numeric contrast vector

#### `regress`

```python
regress(bd, design_matrix = None, noise_model = 'ols', mode = None, **kwargs)
```

Deprecated: Use fit(model='glm', X=design_matrix) instead.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`design_matrix` |  | Design matrix (unused, raises error). | <code>None</code>
`noise_model` |  | Noise model (unused, raises error). | <code>'ols'</code>
`mode` |  | Mode (unused, raises error). | <code>None</code>
`**kwargs` |  | Additional arguments (unused, raises error). | <code>{}</code>

#### `to_fit_dataclass`

```python
to_fit_dataclass(bd, model)
```

Convert BrainData fit results to Fit dataclass.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`model` | <code>[str](#str)</code> | Model type ('ridge' or 'glm') | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Fit` |  | Dataclass containing fit results

