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
[`compute_contrasts`](#data-braindata-modeling-compute-contrasts) | Compute contrasts from a fitted GLM.
[`fit`](#data-braindata-modeling-fit) | Fit a model to brain imaging data.
[`fit_glm`](#data-braindata-modeling-fit-glm) | Fit GLM model and extract results.
[`fit_ridge`](#data-braindata-modeling-fit-ridge) | Fit `bd.model_` and attach the ridge results to `bd`.
[`parse_contrast_string`](#data-braindata-modeling-parse-contrast-string) | Parse a contrast string into a numeric contrast vector.
[`resolve_preprocessing_defaults`](#data-braindata-modeling-resolve-preprocessing-defaults) | Resolve the ``'auto'`` scale/standardize sentinels to concrete values.
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

(data-braindata-modeling-compute-contrasts)=
### `compute_contrasts`

```python
compute_contrasts(bd, contrasts, statistic = 't')
```

Compute contrasts from a fitted GLM.

Uses nilearn's functional ``compute_contrast`` on the fitted
``(labels_, results_)`` so t-statistics are computed with the full per-voxel
parameter covariance (correct for OLS and AR) — a linear combination of
stored betas cannot do this for multi-regressor contrasts (it would ignore
off-diagonal covariance and produce an effect-size map, not a t-map).
Contrast maps stay in masked-array space; no unmasking to a Nifti.

Must be called after ``.fit(model='glm', X=design_matrix)`` has been run.

**Contrast forms.** A string names columns with optional coefficients
(``"conditionA - conditionB"``, ``"2*conditionA - conditionB - conditionC"``,
``"0.5*A + 0.5*B"``; names are case-sensitive and must match the design
exactly). An array-like is a numeric contrast vector with one weight per
regressor (``[1, -1, 0, 0]``). A dict ``{name: contrast}`` evaluates several
contrasts at once.

**Statistics.** ``"t"`` (default) is the t-statistic map for thresholding /
single-subject inference; ``"z"`` the z-score map; ``"p"`` the p-value map;
``"beta"`` / ``"effect_size"`` the effect-size (β) map to feed into a
second-level (group) analysis; ``"all"`` returns every view for one fit — a
dict ``{"beta", "t", "z", "p", "se"}`` of `BrainData` maps — so group-level
code never has to recompute beta separately.

Contrast p-values are **one-sided** (the nilearn/SPM directional-contrast
convention: a contrast tests "A > B"; flip the contrast for the other
direction). This is the documented exception to the library's two-tailed
default.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data fitted with ``model='glm'``. | *required*
`contrasts` | <code>str \| array - like \| dict</code> | One contrast (string or numeric vector) or a ``{name: contrast}`` dict of several; see above. | *required*
`statistic` | <code>str</code> | ``"t"`` (default), ``"z"``, ``"p"``, ``"beta"`` / ``"effect_size"``, or ``"all"``; see above. | <code>'t'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data) \| dict</code> | A single `BrainData` for one contrast with a scalar     ``statistic``; a dict keyed ``"beta"``/``"t"``/``"z"``/``"p"``/``"se"``     for one contrast with ``statistic="all"``; ``{name: BrainData}`` for     a dict of contrasts with a scalar ``statistic``; and a nested     ``{name: {"beta", "t", "z", "p", "se"}}`` for a dict of contrasts     with ``statistic="all"``.

**Raises:**

Type | Description
---- | -----------
<code>RuntimeError</code> | if ``.fit(model='glm')`` has not been run.
<code>ValueError</code> | if the contrast vector length or a column name is invalid, or if ``statistic`` is not one of the supported values.

**Examples:**

```python
data.fit(model="glm", X=dm)

# Single-subject t-map, ready to threshold
tmap = data.compute_contrasts("conditionA - conditionB")

# Effect-size map for use as input to a group-level analysis
beta = data.compute_contrasts("conditionA - conditionB", statistic="beta")

# Everything at once: threshold on res["t"], feed the group on res["beta"]
res = data.compute_contrasts("conditionA - conditionB", statistic="all")
res["t"].plot(threshold=3.09)
group_effects.append(res["beta"])
```

<details class="note" open markdown="1">
<summary>Note</summary>

For group analysis, stack per-subject effect-size maps
(``statistic="beta"`` or ``res["beta"]`` from ``statistic="all"``) and
run a second-level test (e.g. ``BrainData.ttest``). Mixing first-level
t-maps into a group one-sample test conflates effect magnitude with
precision.

</details>

(data-braindata-modeling-fit)=
### `fit`

```python
fit(bd, model = 'glm', *, X = None, cv = None, device = 'cpu', per_target_alpha = True, inplace = True, progress_bar = False, scale = 'auto', standardize = 'auto', **kwargs)
```

Fit a model to brain imaging data.

Creates and fits a model from string specification. The brain data
(bd.data) is always used as the target variable. Model and results
are stored for later use with predict().

For ``model='glm'`` the design is diagnosed before estimation, as warnings
only — nothing is ever dropped, modified, or raised on. An exactly
rank-deficient design fires `RankDeficientDesignWarning`; a full-rank but
near-collinear design (a column pair with |r| >= 0.95, or a
column-standardized condition number above 30) fires
`NearCollinearDesignWarning` instead — never both. Each has its own
category so it can be silenced surgically with
``warnings.filterwarnings``.

**Results stored on the returned `BrainData`:**

- `model_` — the fitted `Ridge` or `Glm` instance (always set, so `predict()`
  works).
- `X_` — the training design/features, used as the `predict()` default.
- GLM: `glm_betas`, `glm_t`, `glm_p`, `glm_se`, `glm_residual`,
  `glm_predicted`, `glm_r2` (each a `BrainData`).
- Ridge: `ridge_weights`, `ridge_fitted_values`, `ridge_scores` (each a
  `BrainData`).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data whose `.data` is the regression target. | *required*
`model` | <code>str</code> | `'glm'` (default) or `'ridge'`. | <code>'glm'</code>
`X` | <code>array - like \| DataFrame \| [DesignMatrix](#page-data-design-matrix)</code> | Design matrix (GLM) or feature matrix (ridge) of shape `(n_samples, n_features)`; `n_samples` must match `bd.data`. For banded ridge, a mapping of feature-space names to such matrices. | <code>None</code>
`cv` | <code>int \| CV splitter \| None</code> | Cross-validation specification, ridge only. An int is the number of unshuffled k-fold splits; an sklearn splitter (e.g. `KFold(3, shuffle=True)`) is used as given; None (default) fits a fixed alpha. Alpha selection needs both a sequence of candidate alphas and a `cv`. | <code>None</code>
`device` | <code>str</code> | Ridge only. Compute device for the ridge solve: `'cpu'` (NumPy) or `'gpu'` (PyTorch on CUDA/MPS, or an error when neither is available). Forwarded to `Ridge`. Ignored for `model='glm'`. Default: `'cpu'`. | <code>'cpu'</code>
`per_target_alpha` | <code>bool</code> | Ridge only. If True, select a separate best alpha per voxel; if False, select a single shared alpha across all voxels. Forwarded to `Ridge`. Default: True. | <code>True</code>
`inplace` | <code>bool</code> | If True, mutate `bd` and return it. If False, fit and return an independent `BrainData` copy while leaving every part of `bd` untouched. Default: True. | <code>True</code>
`progress_bar` | <code>bool</code> | Display a progress bar for long-running operations. Default: False. | <code>False</code>
`scale` | <code>bool \| str</code> | Apply percent-signal-change scaling to the data before fitting, via nilearn's per-voxel `mean_scaling` (each voxel's time-series is divided by its own temporal mean, de-meaned, and multiplied by 100). `'auto'` (default) resolves to False for both models — PSC is opt-in. Useful for GLM (interpretable % betas); for ridge it is redundant with `standardize='zscore'` (a warning is raised for that combination). Applied before `standardize`. | <code>'auto'</code>
`standardize` | <code>str \| None</code> | Standardize each voxel across observations after scaling: `'center'` (subtract the mean), `'zscore'` (subtract mean, divide by std), or None (off). `'auto'` (default) resolves to `'zscore'` for `model='ridge'` (so a shared alpha regularizes voxels fairly) and None for `model='glm'`. | <code>'auto'</code>
`**kwargs` | <code>dict</code> | Additional arguments passed to the model constructor — for `Ridge`: `alpha`, `search_iterations`, `random_state`; for `Glm`: `noise_model`, `minimize_memory`, etc. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | `bd` itself when `inplace=True`; otherwise an independently     owned fitted copy.

**Examples:**

```python
# inplace=True (default): results are stored as attributes on brain_data
brain_data.fit(model='ridge', alpha=[0.1, 1.0, 10.0], cv=5, X=features)
print(f"selected alpha: {brain_data.model_.alpha_}")
weights = brain_data.ridge_weights

# inplace=False: fit a copy; brain_data remains completely unchanged
fitted = brain_data.fit(
    model='ridge', alpha=1.0, X=features, inplace=False
)
weights = fitted.ridge_weights
assert not hasattr(brain_data, 'ridge_weights')

# The returned GLM copy can compute contrasts
fitted_glm = brain_data.fit(model='glm', X=design_matrix, inplace=False)
contrast = fitted_glm.compute_contrasts('conditionA - conditionB')
```

(data-braindata-modeling-fit-glm)=
### `fit_glm`

```python
fit_glm(bd, X)
```

Fit GLM model and extract results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data with `bd.model_` set to a `Glm` instance. | *required*
`X` | <code>DataFrame \| [DesignMatrix](#page-data-design-matrix)</code> | Design matrix. | *required*

<details class="note" open markdown="1">
<summary>Note</summary>

Sets `glm_betas`, `glm_t`, `glm_p`, `glm_se`, `glm_residual`,
`glm_predicted`, `glm_r2`, and `design_matrix` on `bd`.

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

(data-braindata-modeling-parse-contrast-string)=
### `parse_contrast_string`

```python
parse_contrast_string(bd, contrast_str)
```

Parse a contrast string into a numeric contrast vector.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data with a `design_matrix` from a prior GLM fit. | *required*
`contrast_str` | <code>str</code> | Contrast string like `"A - B"` or `"2*A - B - C"`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Numeric contrast vector, one weight per design column.

**Raises:**

Type | Description
---- | -----------
<code>RuntimeError</code> | If no design matrix is attached (`fit()` not called).
<code>ValueError</code> | If a column name is not in the design matrix.

(data-braindata-modeling-resolve-preprocessing-defaults)=
### `resolve_preprocessing_defaults`

```python
resolve_preprocessing_defaults(model, scale, standardize)
```

Resolve the ``'auto'`` scale/standardize sentinels to concrete values.

Per-model defaults for ``BrainData.fit``. ``scale`` (percent-signal-change)
is opt-in for both models. Ridge standardizes its targets by default so a
shared alpha regularizes voxels fairly; GLM does neither so betas stay in
native units.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model` | <code>str</code> | ``'ridge'`` or ``'glm'``. | *required*
`scale` | <code>bool or auto</code> | Requested scale flag. | *required*
`standardize` | <code>str, None, or 'auto'</code> | Requested standardize method. | *required*

**Returns:**

Type | Description
---- | -----------
<code>tuple</code> | ``(scale, standardize)`` with any ``'auto'`` resolved.

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
