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
[`compute_ridge_cv`](#data-braindata-modeling-compute-ridge-cv) | Held-out CV scores under a fixed Ridge α.
[`fit`](#data-braindata-modeling-fit) | Fit a model to brain imaging data.
[`fit_glm`](#data-braindata-modeling-fit-glm) | Fit GLM model and extract results.
[`fit_ridge`](#data-braindata-modeling-fit-ridge) | Fit Ridge model and extract results.
[`parse_contrast_string`](#data-braindata-modeling-parse-contrast-string) | Parse a contrast string into a numeric contrast vector.
[`resolve_preprocessing_defaults`](#data-braindata-modeling-resolve-preprocessing-defaults) | Resolve the ``'auto'`` scale/standardize sentinels to concrete values.
[`ttest`](#data-braindata-modeling-ttest) | One-sample voxelwise t-test across images (axis 0).
[`ttest2`](#data-braindata-modeling-ttest2) | Two-sample voxelwise t-test between two BrainData stacks.



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

(data-braindata-modeling-compute-ridge-cv)=
### `compute_ridge_cv`

```python
compute_ridge_cv(bd, X, cv, alpha = None, device = 'cpu')
```

Held-out CV scores under a fixed Ridge α.

Used only for the *fixed-α* + CV branch. When `alpha='auto'`, alpha selection
is handled by `Ridge.fit` (which delegates to `solve_ridge_cv`) and `fit`
assembles `cv_results_` from the fitted model instead.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data with `bd.model_` set to a `Ridge` instance. | *required*
`X` | <code>ndarray</code> | Training features, shape `(n_samples, n_features)`. | *required*
`cv` | <code>int \| CV splitter</code> | Cross-validation specification. | *required*
`alpha` | <code>float \| None</code> | Fixed regularization strength. If None, taken from `bd.model_.alpha`. | <code>None</code>
`device` | <code>str</code> | Compute device (`'cpu'`/`'gpu'`/`'auto'`). Default: `'cpu'`. | <code>'cpu'</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'scores'`, `'mean_score'`, `'predictions'`, `'folds'`.

(data-braindata-modeling-fit)=
### `fit`

```python
fit(bd, model = 'glm', *, X = None, cv = None, device = 'cpu', local_alpha = True, fit_intercept = False, inplace = True, progress_bar = False, scale = 'auto', standardize = 'auto', **kwargs)
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
- `cv_results_` — dict with keys `'scores'`, `'mean_score'`, `'predictions'`,
  `'folds'`, `'best_alpha'`, `'alpha_scores'` (ridge with `cv` only).
- GLM: `glm_betas`, `glm_t`, `glm_p`, `glm_se`, `glm_residual`,
  `glm_predicted`, `glm_r2` (each a `BrainData`).
- Ridge: `ridge_weights`, `ridge_fitted_values`, `ridge_scores` (each a
  `BrainData`).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data whose `.data` is the regression target. | *required*
`model` | <code>str</code> | `'glm'` (default) or `'ridge'`. | <code>'glm'</code>
`X` | <code>array - like \| DataFrame \| [DesignMatrix](#page-data-design-matrix)</code> | Design matrix (GLM) or feature matrix (ridge) of shape `(n_samples, n_features)`; `n_samples` must match `bd.data`. For banded ridge, a list of such matrices. | <code>None</code>
`cv` | <code>int \| str \| CV splitter \| None</code> | Cross-validation specification, ridge only. An int is the number of k-fold splits (returns CV scores); `'auto'` selects alpha via CV (implies `alpha='auto'`); an sklearn splitter (e.g. `KFold(3, shuffle=True)`) is used as given; None (default) runs no CV. | <code>None</code>
`device` | <code>str</code> | Ridge only. Compute device for the ridge solve/CV: `'cpu'` (NumPy), `'gpu'` (PyTorch on CUDA/MPS when available), or `'auto'` (GPU if present, else CPU). Forwarded to `Ridge` and the CV evaluation. Ignored for `model='glm'`. Default: `'cpu'`. | <code>'cpu'</code>
`local_alpha` | <code>bool</code> | Ridge only. If True, select a separate best alpha per voxel; if False, select a single shared alpha across all voxels. Forwarded to `Ridge`. Default: True. | <code>True</code>
`fit_intercept` | <code>bool</code> | Ridge only. If True, fit an intercept term. Redundant (and warned against) when the data is already centered via `scale` or `standardize`. Forwarded to `Ridge`. Default: False. | <code>False</code>
`inplace` | <code>bool</code> | If True, mutate `bd` and return it. If False, fit and return an independent `BrainData` copy while leaving every part of `bd` untouched. Default: True. | <code>True</code>
`progress_bar` | <code>bool</code> | Display a progress bar for long-running operations. Default: False. | <code>False</code>
`scale` | <code>bool \| str</code> | Apply percent-signal-change scaling to the data before fitting, via nilearn's per-voxel `mean_scaling` (each voxel's time-series is divided by its own temporal mean, de-meaned, and multiplied by 100). `'auto'` (default) resolves to False for both models — PSC is opt-in. Useful for GLM (interpretable % betas); for ridge it is redundant with `standardize='zscore'` (a warning is raised for that combination). Applied before `standardize`. | <code>'auto'</code>
`standardize` | <code>str \| None</code> | Standardize each voxel across observations after scaling: `'center'` (subtract the mean), `'zscore'` (subtract mean, divide by std), or None (off). `'auto'` (default) resolves to `'zscore'` for `model='ridge'` (so a shared alpha regularizes voxels fairly) and None for `model='glm'`. | <code>'auto'</code>
`**kwargs` | <code>dict</code> | Additional arguments passed to the model constructor — for `Ridge`: `alpha`, `alphas`, `random_state`; for `Glm`: `noise_model`, `minimize_memory`, etc. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | `bd` itself when `inplace=True`; otherwise an independently     owned fitted copy.

**Examples:**

```python
# inplace=True (default): results are stored as attributes on brain_data
brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
print(f"CV R2: {brain_data.cv_results_['mean_score'].mean():.3f}")
weights = brain_data.ridge_weights

# inplace=False: fit a copy; brain_data remains completely unchanged
fitted = brain_data.fit(
    model='ridge', alpha=1.0, cv=5, X=features, inplace=False
)
weights = fitted.ridge_weights
assert not hasattr(brain_data, 'ridge_weights')
print(f"CV R2: {fitted.cv_results_['mean_score'].mean():.3f}")

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
fit_ridge(bd, X, cv = None, device = 'cpu', **kwargs)
```

Fit Ridge model and extract results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data with `bd.model_` already set to a `Ridge` instance. | *required*
`X` | <code>ndarray \| list[ndarray]</code> | Training features (a list for banded ridge). | *required*
`cv` | <code>int \| str \| CV splitter \| None</code> | Cross-validation specification; see `fit`. | <code>None</code>
`device` | <code>str</code> | Compute device (`'cpu'`/`'gpu'`/`'auto'`) for the held-out CV evaluation, forwarded to `compute_ridge_cv`. Default: `'cpu'`. | <code>'cpu'</code>
`**kwargs` | <code>dict</code> | Additional ridge arguments for CV (`alpha`, etc.). | <code>{}</code>

<details class="note" open markdown="1">
<summary>Note</summary>

Sets `ridge_weights`, `ridge_fitted_values`, `ridge_scores`, and
`cv_results_` (if `cv` is given) on `bd`.

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

One-sample voxelwise t-test across images (axis 0).

For a BrainData stack of images (e.g. subject-level contrast maps with
shape ``(n_samples, n_voxels)``), test whether the per-voxel mean differs
from ``popmean``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Stack of two or more images. | *required*
`popmean` | <code>float</code> | Population mean to test against. Default 0.0. | <code>0.0</code>
`permutation` | <code>bool</code> | If True, use a sign-flip permutation test on ``images - popmean`` via ``nltools.algorithms.inference.one_sample_permutation_test``; the p-values come from the empirical null and the parametric t-statistic is still reported alongside for reference. Default False. | <code>False</code>
`n_permute` | <code>int</code> | Number of permutations (used only when ``permutation=True``). Default 5000. | <code>5000</code>
`tail` | <code>int \| str</code> | `2` or `'two'` for two-tailed (default); `1` or `'one'` for one-tailed (positive direction). | <code>2</code>
`return_null` | <code>bool</code> | Currently has no effect. The returned dict always contains exactly ``{"mean", "t", "z", "p"}`` and the null distribution is discarded even when this is True. Default False. | <code>False</code>
`n_jobs` | <code>int</code> | Number of parallel jobs. Default -1 (all cores). | <code>-1</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>dict[str, [BrainData](#page-data-brain-data)]</code> | Four keys. `"mean"` is the voxelwise mean across     images minus `popmean` (i.e. `mean(images) - popmean`, an effect-size     estimate; equals the raw voxelwise mean only when `popmean=0`); `"t"`     the parametric one-sample t-statistic; `"z"` the signed z-score,     `sign(t) * norm.isf(p/2)`, matching nilearn's `output_type='z_score'`     (useful for thresholding on z at small df where t tails are heavier     than normal); `"p"` the p-value (parametric, or permutation-based     when `permutation=True`). The effect size is always returned     alongside the inferential maps so group-level code never has to     compute the mean separately.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If ``bd`` contains fewer than 2 images.

(data-braindata-modeling-ttest2)=
### `ttest2`

```python
ttest2(bd, other, equal_var = True, tail = 2)
```

Two-sample voxelwise t-test between two BrainData stacks.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | First stack, shape ``(n1, n_voxels)``. | *required*
`other` | <code>[BrainData](#page-data-brain-data)</code> | Second stack, shape ``(n2, n_voxels)``. | *required*
`equal_var` | <code>bool</code> | If True (default), standard two-sample t-test. If False, Welch's t-test. | <code>True</code>
`tail` | <code>int \| str</code> | `2` or `'two'` for two-tailed (default); `1` or `'one'` for one-tailed (bd > other; swap the arguments for the other direction). | <code>2</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | ``{"t": BrainData, "p": BrainData}``.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If the two BrainData objects have mismatched n_voxels.
