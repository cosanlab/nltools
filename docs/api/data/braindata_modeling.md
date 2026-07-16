(data-braindata-modeling-modeling)=
## `modeling`

BrainData modeling functions.

Standalone functions extracted from BrainData class methods for model
fitting, GLM estimation, Ridge regression, and contrast computation.

**Methods:**

Name | Description
---- | -----------
[`compute_contrasts`](#data-braindata-modeling-compute-contrasts) | Compute contrasts from a fitted GLM.
[`compute_ridge_cv`](#data-braindata-modeling-compute-ridge-cv) | Held-out CV scores under a fixed Ridge α.
[`fit`](#data-braindata-modeling-fit) | Fit a model to brain imaging data.
[`fit_glm`](#data-braindata-modeling-fit-glm) | Fit GLM model and extract results (same logic as current regress()).
[`fit_ridge`](#data-braindata-modeling-fit-ridge) | Fit Ridge model and extract results.
[`parse_contrast_string`](#data-braindata-modeling-parse-contrast-string) | Parse a contrast string into a numeric contrast vector.
[`to_fit_dataclass`](#data-braindata-modeling-to-fit-dataclass) | Convert BrainData fit results to Fit dataclass.
[`ttest`](#data-braindata-modeling-ttest) | One-sample voxelwise t-test across images (axis 0).
[`ttest2`](#data-braindata-modeling-ttest2) | Two-sample voxelwise t-test between two BrainData stacks.



### Methods

(data-braindata-modeling-compute-contrasts)=
#### `compute_contrasts`

```python
compute_contrasts(bd, contrasts, contrast_type = 't')
```

Compute contrasts from a fitted GLM.

Delegates to the underlying ``nilearn.FirstLevelModel.compute_contrast`` so
t-statistics are computed with the full parameter covariance matrix —
linear-combination-of-stored-betas cannot do this correctly for multi-
regressor contrasts (it would ignore off-diagonal covariance and produce
an effect-size map, not a t-map).

Must be called after ``.fit(model='glm', X=design_matrix)`` has been run.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`contrasts` |  | Can be:<br>- str: a contrast expressed in terms of column names, e.g.   ``"conditionA - conditionB"`` or ``"2*conditionA - conditionB - conditionC"`` - array-like: a numeric contrast vector, one weight per regressor   (e.g. ``[1, -1, 0, 0]``) - dict: ``{name: contrast}`` for multiple contrasts at once | *required*
`contrast_type` | <code>[str](#str)</code> | What to return per contrast. One of:<br>- ``"t"`` (default): t-statistic map (for thresholding /   single-subject inference) - ``"z"``: z-score map - ``"p"``: p-value map - ``"beta"`` / ``"effect_size"``: effect-size (β) map — use this   when feeding into a second-level (group) analysis - ``"all"``: a bundle dict ``{"beta", "t", "z", "p", "se"}``   of BrainData maps for this one contrast. One fit, one call,   every view — effect size *and* inferential maps together so   group-level code never has to recompute beta separately. | <code>'t'</code>

**Returns:**

Type | Description
---- | -----------
 | Depends on inputs:<br>- single contrast (str or array) + scalar ``contrast_type``:   a single BrainData. - single contrast + ``contrast_type="all"``: a flat dict of five   BrainData keyed by ``"beta"``/``"t"``/``"z"``/``"p"``/``"se"``. - dict of contrasts + scalar ``contrast_type``: a dict   ``{name: BrainData}``. - dict of contrasts + ``contrast_type="all"``: a nested dict   ``{name: {"beta", "t", "z", "p", "se"}}``.

**Examples:**

```pycon
>>> data.fit(model="glm", X=dm)
>>> # Single-subject t-map, ready to threshold
>>> tmap = data.compute_contrasts("conditionA - conditionB")
>>> # Effect-size map for use as input to a group-level analysis
>>> beta = data.compute_contrasts(
...     "conditionA - conditionB", contrast_type="beta"
... )
>>> # Everything at once: threshold on res["t"], feed group on res["beta"]
>>> res = data.compute_contrasts(
...     "conditionA - conditionB", contrast_type="all"
... )
>>> res["t"].plot(threshold=3.09)
>>> group_effects.append(res["beta"])
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- String contrasts support coefficients: ``"2*A - B"`` or ``"0.5*A + 0.5*B"``.
- Column names must match design matrix columns exactly (case-sensitive).
- For group analysis, stack per-subject effect-size maps
  (``contrast_type="beta"`` or ``res["beta"]`` from ``contrast_type="all"``)
  and run a second-level test (e.g. ``BrainData.ttest``). Mixing first-level
  t-maps into a group one-sample test conflates effect magnitude with precision.

</details>

(data-braindata-modeling-compute-ridge-cv)=
#### `compute_ridge_cv`

```python
compute_ridge_cv(bd, X, cv, alpha = None, backend = 'auto', **kwargs)
```

Held-out CV scores under a fixed Ridge α.

Used only for the *fixed-α* + CV branch — alpha selection is now
handled by ``Ridge.fit`` (which delegates to ``solve_ridge_cv``) and
assembled into ``cv_results_`` by ``_assemble_ridge_cv_results``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`X` | <code>[ndarray](#ndarray)</code> | Training features, shape (n_samples, n_features). | *required*
`cv` | <code>int or sklearn CV splitter</code> | Cross-validation specification. | *required*
`alpha` | <code>[float](#float)</code> | Fixed regularization strength. If None, extracted from ``bd.model_.alpha``. | <code>None</code>
`backend` | <code>[str](#str)</code> | Computational backend ('numpy', 'torch', 'auto'). Default: 'auto' | <code>'auto'</code>
`**kwargs` |  | Additional kwargs (forward-compatibility). | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | ``{"scores", "mean_score", "predictions", "folds"}``.

(data-braindata-modeling-fit)=
#### `fit`

```python
fit(bd, model = 'glm', *, X = None, cv = None, local_alpha = True, fit_intercept = False, inplace = True, progress_bar = None, scale = True, scale_value = 100.0, design_clean = True, design_clean_thresh = 0.95, design_clean_exclude_confounds = False, design_clean_fill_na = 0, **kwargs)
```

Fit a model to brain imaging data.

Creates and fits a model from string specification. The brain data
(bd.data) is always used as the target variable. Model and results
are stored for later use with predict().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`model` | <code>[str](#str)</code> | Model type: 'ridge', 'glm', or future model names | <code>'glm'</code>
`X` | <code>[array](#array) - [like](#like) or [DataFrame](#DataFrame)</code> | Design matrix or feature matrix, shape (n_samples, n_features) - For GLM: Design matrix with regressors (n_samples must match bd.data) - For Ridge: Feature matrix for prediction (n_samples must match bd.data) | <code>None</code>
`cv` | <code>int, 'auto', or sklearn CV splitter</code> | Cross-validation specification (Ridge only): - int: Number of folds for k-fold CV (returns CV scores) - 'auto': Triggers alpha selection via CV (implies alpha='auto') - sklearn CV object: Custom CV splitter (e.g., KFold(3, shuffle=True)) - None: No CV (default, backward compatible) | <code>None</code>
`inplace` | <code>bool, default=True</code> | If True, mutate bd and return bd (backward compatible). If False, return Fit dataclass with results (bd unchanged). | <code>True</code>
`progress_bar` | <code>[bool](#bool)</code> | Display progress bar during fitting. - If None: Uses bd.verbose (default) - If True: Shows progress bar for long-running operations - If False: No progress bar | <code>None</code>
`scale` | <code>bool, default=True</code> | Apply grand-mean scaling before fitting. Calls bd.scale(scale_value) which divides all values by the global mean and multiplies by scale_value. This puts data in percent signal change units, which is standard for fMRI analysis. | <code>True</code>
`scale_value` | <code>float, default=100.0</code> | Target value for mean after scaling. Only used if scale=True. | <code>100.0</code>
`design_clean` | <code>bool, default=True</code> | GLM only. If True, run ``DesignMatrix.clean()`` on ``X`` before fitting to drop highly correlated regressors. Coerces ``X`` to ``DesignMatrix`` if needed. Ignored when ``model='ridge'``. | <code>True</code>
`design_clean_thresh` | <code>float, default=0.95</code> | GLM only. Correlation threshold passed to ``DesignMatrix.clean()`` (drops if ``abs(r) >= thresh``). Ignored when ``model='ridge'``. | <code>0.95</code>
`design_clean_exclude_confounds` | <code>bool, default=False</code> | GLM only. If True, ``DesignMatrix.clean()`` skips confound columns when checking correlations. Ignored when ``model='ridge'``. | <code>False</code>
`design_clean_fill_na` | <code>int, float, or None, default=0</code> | GLM only. Fill value for NaNs before correlation check in ``DesignMatrix.clean()``. Ignored when ``model='ridge'``. | <code>0</code>
`**kwargs` | <code>[dict](#dict)</code> | Additional arguments passed to model constructor - Ridge: alpha, alphas, backend, random_state - Glm: noise_model, minimize_memory, etc. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or Fit: If ``inplace=True``, returns bd (fitted BrainData). If ``inplace=False``, returns Fit dataclass with results.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`model_` | <code>[BaseModel](#BaseModel)</code> | Fitted model instance (Ridge, Glm, etc.). Set on bd when ``inplace=True``.
`X_` | <code>[ndarray](#ndarray)</code> | Training data X, stored for predict() default.
`cv_results_` | <code>[dict](#dict)</code> | Cross-validation results dict with keys 'scores', 'mean_score', 'predictions', 'folds', 'best_alpha', 'alpha_scores' (if cv is not None).
`glm_betas` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Beta coefficients (for model='glm')
`glm_t` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | T-statistics (for model='glm')
`glm_p` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | P-values (for model='glm')
`glm_se` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Standard errors (for model='glm')
`glm_residual` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Residuals (for model='glm')
`glm_predicted` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Fitted values (for model='glm')
`glm_r2` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | R-squared values (for model='glm')
`ridge_weights` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Model coefficients (for model='ridge')
`ridge_fitted_values` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Fitted values (for model='ridge')
`ridge_scores` | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | R-squared scores (for model='ridge')

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

(data-braindata-modeling-fit-glm)=
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

(data-braindata-modeling-fit-ridge)=
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

(data-braindata-modeling-parse-contrast-string)=
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

(data-braindata-modeling-to-fit-dataclass)=
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

(data-braindata-modeling-ttest)=
#### `ttest`

```python
ttest(bd, popmean = 0.0, permutation = False, n_permute = 5000, tail = 2, return_null = False, n_jobs = -1, random_state = None)
```

One-sample voxelwise t-test across images (axis 0).

For a BrainData stack of images (e.g. subject-level contrast maps with
shape ``(n_samples, n_voxels)``), test whether the per-voxel mean differs
from ``popmean``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance (must contain multiple images). | *required*
`popmean` |  | Population mean to test against. Default 0.0. | <code>0.0</code>
`permutation` |  | If True, use sign-flip permutation test via ``nltools.stats.one_sample_permutation_test``; the p-values come from the empirical null and the parametric t-statistic is still reported alongside for reference. | <code>False</code>
`n_permute` |  | Number of permutations (used only when ``permutation=True``). Default 5000. | <code>5000</code>
`tail` |  | Tail of the test (1 or 2). Default 2. | <code>2</code>
`return_null` |  | If True, also return the null distribution. Default False. | <code>False</code>
`n_jobs` |  | Number of parallel jobs. Default -1 (all cores). | <code>-1</code>
`random_state` |  | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | dict with four BrainData keys:<br>- ``"mean"``: voxelwise mean across images (effect-size estimate). - ``"t"``: parametric one-sample t-statistic. - ``"z"``: signed z-score, ``sign(t) * norm.isf(p/2)``, matching   nilearn's ``output_type='z_score'``. Useful for thresholding   on z at small df where t tails are heavier than normal. - ``"p"``: p-value (parametric, or permutation-based when   ``permutation=True``).
 | The effect size is always returned alongside the inferential maps so
 | group-level code never has to compute the mean separately.

(data-braindata-modeling-ttest2)=
#### `ttest2`

```python
ttest2(bd, other, equal_var = True)
```

Two-sample voxelwise t-test between two BrainData stacks.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | First BrainData (shape ``(n1, n_voxels)``). | *required*
`other` |  | Second BrainData (shape ``(n2, n_voxels)``). | *required*
`equal_var` |  | If True (default), standard two-sample t-test. If False, Welch's t-test. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | ``{"t": BrainData, "p": BrainData}``.

