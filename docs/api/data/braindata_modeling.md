---
title: data.braindata.modeling
label: page-data-braindata-modeling
---

Model fitting, contrasts, and statistical tests for `BrainData`.

GLM and ridge fitting (with cross-validation), contrast computation, one- and
two-sample t-tests, and the design-matrix diagnostics `fit` runs before a GLM.
`BrainData` methods delegate here.

**Classes:**

Name | Description
---- | -----------
[`NearCollinearDesignWarning`](#data-braindata-modeling-nearcollineardesignwarning) | The design matrix supplied to ``fit()`` is full rank but nearly collinear.
[`RankDeficientDesignWarning`](#data-braindata-modeling-rankdeficientdesignwarning) | The design matrix supplied to ``fit()`` is rank deficient.

**Functions:**

Name | Description
---- | -----------
[`check_unselected_estimator_options`](#data-braindata-modeling-check-unselected-estimator-options) | Reject `fit` options belonging to the estimator `model` did not select.
[`compute_contrasts`](#data-braindata-modeling-compute-contrasts) | Compute contrasts on a fitted GLM.
[`fit`](#data-braindata-modeling-fit) | Fit a model to brain imaging data.
[`fit_glm`](#data-braindata-modeling-fit-glm) | Fit `model` on `X` and attach it and the GLM results the facade owns.
[`fit_ridge`](#data-braindata-modeling-fit-ridge) | Fit `bd.model_` and attach the ridge results to `bd`.
[`ttest`](#data-braindata-modeling-ttest) | Run a one-sample voxelwise t-test across images (axis 0).



## Classes

(data-braindata-modeling-nearcollineardesignwarning)=
### `NearCollinearDesignWarning`

Bases: `UserWarning`

The design matrix supplied to ``fit()`` is full rank but nearly collinear.

Subclasses ``UserWarning`` so it participates in default filtering, while
remaining individually silenceable:
``warnings.filterwarnings("ignore", category=NearCollinearDesignWarning)``.

(data-braindata-modeling-rankdeficientdesignwarning)=
### `RankDeficientDesignWarning`

Bases: `UserWarning`

The design matrix supplied to ``fit()`` is rank deficient.

Subclasses ``UserWarning`` so it participates in default filtering, while
remaining individually silenceable:
``warnings.filterwarnings("ignore", category=RankDeficientDesignWarning)``.



## Functions

(data-braindata-modeling-check-unselected-estimator-options)=
### `check_unselected_estimator_options`

```python
check_unselected_estimator_options(model, supplied)
```

Reject `fit` options belonging to the estimator `model` did not select.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model` | <code>str</code> | The selected model, `'glm'` or `'ridge'`. | *required*
`supplied` | <code>Iterable[str]</code> | Names of the model-specific options the caller gave a non-default value. | *required*

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If any supplied name belongs to the unselected estimator.

(data-braindata-modeling-compute-contrasts)=
### `compute_contrasts`

```python
compute_contrasts(bd, contrasts, *, inference = False)
```

Compute contrasts on a fitted GLM.

Pure forwarding: the fitted `Glm` parses every contrast definition and
computes every number. This layer wraps each per-voxel array as an
independently owned `BrainData` map with cleared row metadata, because a
contrast map's leading axis no longer represents training observations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data fitted with `model='glm'`. | *required*
`contrasts` | <code>str \| array - like \| Mapping</code> | One contrast definition — a string expression over design column names or a flat numeric weight vector — or a mapping of names to those definitions. | *required*
`inference` | <code>bool</code> | If True, return `ContrastResult` records instead of bare effect maps. Default False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data) \| [ContrastResult](#models-contrastresult) \| dict</code> | One effect map, or one     `ContrastResult` of maps when `inference=True`; a dictionary with     the same keys for a mapping.

**Raises:**

Type | Description
---- | -----------
<code>RuntimeError</code> | If no model has been fitted.
<code>ValueError</code> | If the fitted model is not a `Glm`.

**Examples:**

```python
data.fit(model="glm", X=design)

# Effect map — the input a second-level model consumes
effect = data.compute_contrasts("conditionA - conditionB")

# First-level inference: every statistic in one record
result = data.compute_contrasts("conditionA - conditionB", inference=True)
result.statistic.plot(threshold=3.09)
```

<details class="note" open markdown="1">
<summary>Note</summary>

Contrast p-values are one-sided (the nilearn/SPM directional-contrast
convention): the contrast tests "A > B", so negate it for the other
direction. This is the documented exception to the library's two-tailed
default.

</details>

(data-braindata-modeling-fit)=
### `fit`

```python
fit(bd, model = 'glm', *, X = None, cv = None, device = 'cpu', per_target_alpha = True, glm_noise_model = 'ols', glm_bins = 100, glm_n_jobs = 1, inplace = True, random_state = None, progress_bar = False, **kwargs)
```

Fit a model to brain imaging data.

`bd.data` is always the response. The estimator and its results are stored
on the returned `BrainData` for later use with `predict` and, for a GLM,
`compute_contrasts`.

For `model='glm'` the design is diagnosed before estimation, as warnings
only — nothing is ever dropped, modified, or raised on. An exactly
rank-deficient design fires `RankDeficientDesignWarning`; a full-rank but
near-collinear design (a column pair with |r| >= 0.95, or a
column-standardized condition number above 30) fires
`NearCollinearDesignWarning` instead — never both. Each has its own
category so it can be silenced surgically with `warnings.filterwarnings`.

The facade does not preprocess the response. Compose `scale()` and
`standardize()` before `fit` when you want them, so the fitted object's
data, predictions, residuals, and coefficients stay in the response space
you supplied.

GLM options carry a `glm_` prefix. The ridge options (`cv`, `device`,
`per_target_alpha`, `progress_bar`, and additional `Ridge` constructor
arguments) keep their bare names for now, and `random_state` keeps its
bare name because both estimators use it. A non-default option belonging
to the estimator `model` did not select raises `ValueError`.

**Results stored on the returned `BrainData`:**

- `model_` — the fitted `Ridge` or `Glm`.
- GLM: `glm_betas`, `glm_residual`, `glm_predicted`, `glm_r2`.
- Ridge: `ridge_weights`, `ridge_fitted_values`, `ridge_scores`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data whose `.data` is the regression target. | *required*
`model` | <code>str</code> | `'glm'` (default) or `'ridge'`. | <code>'glm'</code>
`X` | <code>[DesignMatrix](#page-data-design-matrix) \| array - like \| Mapping</code> | Design matrix for a GLM — a precomputed `DesignMatrix` with `n_samples` matching `bd` — or a feature matrix for ridge. For banded ridge, a mapping of feature-space names to matrices. | <code>None</code>
`cv` | <code>int \| CV splitter \| None</code> | Ridge only. Cross-validation specification. An int is the number of unshuffled k-fold splits; an sklearn splitter is used as given; None (default) fits a fixed alpha. | <code>None</code>
`device` | <code>str</code> | Ridge only. Compute device for the ridge solve: `'cpu'` (default, NumPy) or `'gpu'` (PyTorch on CUDA/MPS, or an error when neither is available). | <code>'cpu'</code>
`per_target_alpha` | <code>bool</code> | Ridge only. If True (default), select a separate best alpha per voxel; if False, one shared alpha. | <code>True</code>
`glm_noise_model` | <code>str</code> | GLM only. `'ols'` (default) or `'arN'` for Nilearn's autoregressive model of order N (`'ar1'`, `'ar2'`, ...). | <code>'ols'</code>
`glm_bins` | <code>int</code> | GLM only. Nilearn's discretization of the estimated AR coefficients. Default 100. | <code>100</code>
`glm_n_jobs` | <code>int</code> | GLM only. CPUs Nilearn uses to fit autoregressive groups in parallel. The default OLS fit does not use this path. Default 1. | <code>1</code>
`inplace` | <code>bool</code> | If True (default), mutate `bd` and return it. If False, fit and return an independent `BrainData` copy while leaving every part of `bd` untouched. | <code>True</code>
`random_state` | <code>int \| None</code> | Seed shared by both estimators. Default None. | <code>None</code>
`progress_bar` | <code>bool</code> | Ridge only. Display a progress bar during fitting. Default False. | <code>False</code>
`**kwargs` | <code>dict</code> | Ridge only. Additional `Ridge` constructor arguments (`alpha`, `search_iterations`, ...). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | `bd` itself when `inplace=True`; otherwise an independently     owned fitted copy.

**Raises:**

Type | Description
---- | -----------
<code>TypeError</code> | If `model` is unknown, `X` is missing, `model='glm'` gets a design that is not a `DesignMatrix`, or `model='glm'` gets an unknown keyword.
<code>ValueError</code> | If `X` and `bd` disagree on sample count, or a non-default option belongs to the unselected estimator.

**Examples:**

```python
# inplace=True (default): results are stored on brain_data
brain_data.fit(model='ridge', alpha=1.0, X=features)
weights = brain_data.ridge_weights

# inplace=False: fit a copy; brain_data remains completely unchanged
fitted = brain_data.fit(model='glm', X=design, inplace=False)
effect = fitted.compute_contrasts('conditionA - conditionB')
```

(data-braindata-modeling-fit-glm)=
### `fit_glm`

```python
fit_glm(bd, X, model)
```

Fit `model` on `X` and attach it and the GLM results the facade owns.

Numerical fitting, coefficients, predictions, residuals, and R-squared all
come from `Glm`; this layer only wraps them as independently owned
`BrainData` results. `model_` is attached only once the fit succeeds, so a
failed fit never leaves an unfitted estimator on `bd`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data whose `.data` is the response. | *required*
`X` | <code>[DesignMatrix](#page-data-design-matrix)</code> | The training design. | *required*
`model` | <code>[Glm](#models-glm)</code> | An unfitted estimator. | *required*

<details class="note" open markdown="1">
<summary>Note</summary>

Sets `model_`, `glm_betas` (one map per design column), `glm_predicted`
and `glm_residual` (one row per training observation, row metadata
retained), and `glm_r2` (one fit-quality map). `glm_r2` carries
Nilearn's whitened variance-ratio semantics: conventional R-squared for
an OLS fit with an intercept, a pseudo-R-squared in the whitened space
for an autoregressive one.

</details>

(data-braindata-modeling-fit-ridge)=
### `fit_ridge`

```python
fit_ridge(bd, X)
```

Fit `bd.model_` and attach the ridge results to `bd`.

Alpha selection and the banded search belong to `Ridge`; this layer only
stores the results the facade owns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data with `bd.model_` already set to a `Ridge` instance. | *required*
`X` | <code>ndarray \| Mapping[str, ndarray]</code> | Training features. | *required*

<details class="note" open markdown="1">
<summary>Note</summary>

Sets `ridge_weights`, `ridge_fitted_values`, and `ridge_scores` on `bd`.

</details>

(data-braindata-modeling-ttest)=
### `ttest`

```python
ttest(bd, *, popmean = 0.0, permutation = False, n_permute = 5000, tail = 2, return_null = False, n_jobs = -1, random_state = None)
```

Run a one-sample voxelwise t-test across images (axis 0).

For a BrainData stack of images (e.g. subject-level contrast maps with
shape `(n_images, n_voxels)`), test whether the per-voxel mean differs from
`popmean`. Delegates the statistics to the shared one-sample contract in
`nltools.algorithms.inference.one_sample`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Stack of two or more images. | *required*
`popmean` | <code>float</code> | Population mean to test against. Default 0.0. | <code>0.0</code>
`permutation` | <code>bool</code> | If True, take p from a sign-flip permutation test on `images - popmean` via `one_sample_permutation_test`. The reported `t` stays the observed parametric statistic. Default False. | <code>False</code>
`n_permute` | <code>int</code> | Number of permutations, used only when `permutation=True`. Default 5000. | <code>5000</code>
`tail` | <code>int \| str</code> | `2` or `'two'` for two-tailed (default); `1` or `'one'` for one-tailed (mean > `popmean`). | <code>2</code>
`return_null` | <code>bool</code> | If True, also return the permutation null. Has no effect on the parametric path, which computes no null. Default False. | <code>False</code>
`n_jobs` | <code>int</code> | Number of parallel jobs. Default -1 (all cores). | <code>-1</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | `"mean"`, `"t"`, `"z"` and `"p"` as independent `BrainData` images     with observation metadata cleared. `"mean"` is the voxelwise mean     minus `popmean` — the effect relative to the tested null, equal to     the raw mean only when `popmean=0`. `"t"` is the observed one-sample     t-statistic on both paths. `"p"` is parametric, or the empirical     sign-flip p-value when `permutation=True`. `"z"` is the tail-aware     normal score of `p` (`sign(t) * norm.isf(p/2)` two-tailed), matching     nilearn's `output_type='z_score'`. With `permutation=True` and     `return_null=True` the dict also holds `"null_dist"`, an owned     `(n_permute, n_voxels)` array of centered means in the units of     `"mean"`. Maps are unthresholded. Apply a cutoff or a     multiple-comparison correction afterwards.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If `bd` contains fewer than 2 images.

**Examples:**

```python
result = contrast_maps.ttest()
significant = result["z"].data * (result["p"].data < 0.001)

perm = contrast_maps.ttest(
    permutation=True, n_permute=5000, return_null=True, random_state=0
)
perm["null_dist"].shape  # → (5000, n_voxels)
```
