---
title: data.fitresults
label: data-fitresults
---

Immutable container for model fitting results.

This module provides the Fit dataclass, which stores results from model fitting
operations in nltools. It uses pure numpy arrays and has no dependencies on
BrainData or other nltools data structures, making it suitable for standalone
use with inference algorithms.

**Classes:**

Name | Description
---- | -----------
[`Fit`](#data-fitresults-fit) | Immutable container for model fitting results.
[`Predict`](#data-fitresults-predict) | Immutable container for prediction / MVPA decoding results.
[`PredictCollection`](#data-fitresults-predictcollection) | Immutable container for per-subject decoding results.



**Examples:**

```python
import numpy as np
from nltools.data import BrainData

# BrainData workflow
brain = BrainData(data="brain_data.nii.gz")
fit = brain.fit(model="ridge", X=design_matrix, cv=5)
print(fit.available())
# ['fitted_values', 'weights', 'scores', 'cv_scores', 'cv_mean_score',
#  'cv_predictions', 'cv_folds']

# Inference algorithms directly
from nltools.algorithms import ridge_cv

X = np.random.randn(100, 5)
y = np.random.randn(100, 1000)
result = ridge_cv(X, y, cv=5)
result["cv_scores"].shape  # (5, 20, 1000)

# Save all non-None results, then load and reconstruct
np.savez("fit_results.npz", **fit.asdict())
loaded = np.load("fit_results.npz")
fit_reconstructed = Fit(**{k: loaded[k] for k in loaded.files})

# Export only specific fields
np.savez("weights_and_scores.npz", weights=fit.weights, scores=fit.scores)

# Introspection
if 'cv_scores' in fit.available():
    print(f"CV R2 range: [{fit.cv_mean_score.min():.3f}, {fit.cv_mean_score.max():.3f}]")

# Convert scalar and 1D results to a polars DataFrame
import polars as pl

results_dict = fit.asdict()
df = pl.DataFrame({k: v for k, v in results_dict.items() if v.ndim <= 1})
```

## Classes

(data-fitresults-fit)=
### `Fit`

```python
Fit(fitted_values: np.ndarray, weights: np.ndarray | None = None, scores: np.ndarray | None = None, betas: np.ndarray | None = None, t_stats: np.ndarray | None = None, p_values: np.ndarray | None = None, se: np.ndarray | None = None, residuals: np.ndarray | None = None, r2: np.ndarray | None = None, cv_scores: np.ndarray | None = None, cv_mean_score: np.ndarray | None = None, cv_predictions: np.ndarray | None = None, cv_folds: np.ndarray | None = None, cv_best_alpha: float | None = None, cv_alpha_scores: np.ndarray | None = None)
```

Immutable container for model fitting results.

Pure numpy arrays with minimal introspection methods. This allows
users to work directly with nltools inference algorithms without
requiring BrainData objects.

Attributes depend on model type and CV usage:

**Ridge (no CV):**
    weights (ndarray): Coefficients, shape (n_features, n_voxels)
    scores (ndarray): R² scores, shape (n_voxels,)
    fitted_values (ndarray): Training predictions, shape (n_samples, n_voxels)

**Ridge (with CV):**
    All above plus:
    cv_scores (ndarray): Per-fold R², shape (n_folds, n_voxels)
    cv_mean_score (ndarray): Mean R² across folds, shape (n_voxels,)
    cv_predictions (ndarray): Out-of-fold predictions, shape (n_samples, n_voxels)
    cv_folds (ndarray): Fold indices, shape (n_samples,)
    cv_best_alpha (float): Selected alpha (if alpha='auto')
    cv_alpha_scores (ndarray): Alpha selection scores (if alpha='auto')

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`fitted_values` | <code>ndarray</code> | Fitted values or predictions, always present.
`weights` | <code>ndarray \| None</code> | Model coefficients (Ridge).
`scores` | <code>ndarray \| None</code> | R² scores (Ridge).
`betas` | <code>ndarray \| None</code> | Beta coefficients (GLM).
`t_stats` | <code>ndarray \| None</code> | T-statistics (GLM).
`p_values` | <code>ndarray \| None</code> | P-values (GLM).
`se` | <code>ndarray \| None</code> | Standard errors (GLM).
`residuals` | <code>ndarray \| None</code> | Residuals (GLM).
`r2` | <code>ndarray \| None</code> | R² values (GLM).
`cv_scores` | <code>ndarray \| None</code> | Per-fold cross-validation scores.
`cv_mean_score` | <code>ndarray \| None</code> | Mean cross-validation score across folds.
`cv_predictions` | <code>ndarray \| None</code> | Out-of-fold predictions.
`cv_folds` | <code>ndarray \| None</code> | Fold indices for each sample.
`cv_best_alpha` | <code>float \| None</code> | Best alpha selected via cross-validation.
`cv_alpha_scores` | <code>ndarray \| None</code> | Cross-validation scores for each alpha tested.

<details class="note" open markdown="1">
<summary>Note</summary>

Methods: `available` returns the list of non-None attribute names
(excludes private fields); `asdict` converts to a dictionary,
optionally excluding None values.

</details>

**Methods:**

Name | Description
---- | -----------
[`asdict`](#data-fitresults-asdict) | Convert to dictionary.
[`available`](#data-fitresults-available) | Return list of non-None attribute names.



**GLM:**
    betas (ndarray): Beta coefficients, shape (n_regressors, n_voxels)
    t_stats (ndarray): T-statistics, shape (n_regressors, n_voxels)
    p_values (ndarray): P-values, shape (n_regressors, n_voxels)
    se (ndarray): Standard errors, shape (n_regressors, n_voxels)
    residuals (ndarray): Residuals, shape (n_samples, n_voxels)
    fitted_values (ndarray): Fitted values, shape (n_samples, n_voxels)
    r2 (ndarray): R² values, shape (n_voxels,)

**Examples:**

```python
import numpy as np
from nltools.data.fitresults import Fit

# Ridge without CV
fit = Fit(
    fitted_values=np.random.randn(100, 1000),
    weights=np.random.randn(5, 1000),
    scores=np.random.randn(1000),
)
fit.available()  # ['fitted_values', 'weights', 'scores']

# Ridge with CV
fit_cv = Fit(
    fitted_values=np.random.randn(100, 1000),
    weights=np.random.randn(5, 1000),
    scores=np.random.randn(1000),
    cv_scores=np.random.randn(5, 1000),
    cv_mean_score=np.random.randn(1000),
    cv_predictions=np.random.randn(100, 1000),
    cv_folds=np.arange(100) % 5,
)
'cv_scores' in fit_cv.available()  # True

# Immutability: assignment raises FrozenInstanceError (an AttributeError)
try:
    fit.scores = np.zeros(1000)
except AttributeError:
    print("Cannot modify frozen dataclass")

# Save to .npz, then load and reconstruct
np.savez("results.npz", **fit.asdict())
loaded = np.load("results.npz")
fit_reloaded = Fit(**{k: loaded[k] for k in loaded.files})
```

<details class="note" open markdown="1">
<summary>Note</summary>

- Frozen dataclass ensures results cannot be accidentally modified.
- All attributes are numpy arrays (except cv_best_alpha which is float).
- None values indicate that field was not computed for this model/method.
- Private fields (starting with _) are excluded from available() and asdict().

</details>

#### Methods

(data-fitresults-asdict)=
##### `asdict`

```python
asdict(include_none: bool = False) -> dict
```

Convert to dictionary.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`include_none` | <code>bool</code> | If True, include attributes with None values. Private fields (starting with _) are always excluded. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Dictionary of attribute names to values.

**Examples:**

```python
import numpy as np
from nltools.data.fitresults import Fit

fit = Fit(
    fitted_values=np.random.randn(100, 1000),
    weights=np.random.randn(5, 1000),
    scores=None,
)
d = fit.asdict(include_none=False)
'scores' in d  # False

d = fit.asdict(include_none=True)
'scores' in d  # True
d['scores'] is None  # True
```

(data-fitresults-available)=
##### `available`

```python
available() -> list
```

Return list of non-None attribute names.

Excludes private fields (starting with _).

**Returns:**

Type | Description
---- | -----------
<code>list</code> | Names of attributes that are not None.

**Examples:**

```python
import numpy as np
from nltools.data.fitresults import Fit

fit = Fit(
    fitted_values=np.random.randn(100, 1000),
    weights=np.random.randn(5, 1000),
)
fit.available()  # ['fitted_values', 'weights']
'scores' in fit.available()  # False
```

(data-fitresults-predict)=
### `Predict`

```python
Predict(predictions: np.ndarray | None = None, scores: np.ndarray | None = None, mean_score: Any = None, std_score: Any = None, cv_folds: np.ndarray | None = None, roi_labels: np.ndarray | None = None, accuracy_map: Any = None, weight_map: Any = None, fold_weight_maps: Any = None, estimator: Any = None, permutation_scores: np.ndarray | None = None, permutation_pvalue: Any = None)
```

Immutable container for prediction / MVPA decoding results.

Mirrors `Fit`: frozen, all fields default to `None`, populated
based on the dispatch path (`spatial_scale`, `y` vs `X`, `refit`) used
by `BrainData.predict`. Fields not applicable to the call remain
`None` and are filtered from `available` and `asdict`.

**Brain-space outputs are `BrainData` objects**, not raw arrays —
so `result.weight_map.plot()` works directly. Drop down to numpy via
`result.weight_map.data` if needed. Non-spatial fields (`predictions`,
`cv_folds`, scalar scores) stay as numpy.

Field shapes by dispatch:

**spatial_scale='whole_brain'** (with `y`):
    - `predictions`: `(n_samples,)` ndarray, OOF predictions from CV
    - `scores`: `(n_folds,)` ndarray, per-fold score
    - `mean_score`: float, mean across folds
    - `std_score`: float, std across folds
    - `cv_folds`: `(n_samples,)` ndarray, fold index per sample
    - `weight_map`: BrainData `(1, n_voxels)`, `coef_` from one
      model fit on the **full** `(X, y)`. The publishable map.
    - `fold_weight_maps`: BrainData `(n_folds, n_voxels)`, per-fold
      `coef_` stack — for stability analysis (e.g., across-fold std).
    - `estimator`: the fitted all-data sklearn estimator (use for
      `.predict()` on new data).

**spatial_scale='roi'** (with `y`):
    - `scores`: `(n_folds, n_rois)` ndarray
    - `mean_score`: `(n_rois,)` ndarray, mean across folds per parcel
    - `std_score`: `(n_rois,)` ndarray
    - `roi_labels`: `(n_rois,)` ndarray of atlas integer IDs in the
      same order as `mean_score` / `std_score` / `scores` axis 1
    - `accuracy_map`: BrainData `(1, n_voxels)`, every voxel inside
      parcel *i* set to that parcel's mean accuracy (others NaN)
    - `weight_map`: BrainData `(1, n_voxels)`, per-parcel `coef_`
      from each parcel's all-data fit, written back into voxel space
      (atlas is a label image so reassembly is disjoint). Voxels outside
      any parcel are NaN. Magnitudes across parcels are not directly
      comparable — different parcels live on different X distributions.
    - `fold_weight_maps`: BrainData `(n_folds, n_voxels)`
    - `estimator`: `dict[int, sklearn]` keyed by atlas label

    If any parcel can't expose `.coef_` (non-linear model, `SelectKBest`
    in pipeline), `weight_map` / `fold_weight_maps` / `estimator`
    all collapse to `None` for the whole call.

**spatial_scale='searchlight'** (with `y`):
    - `accuracy_map`: BrainData `(1, n_voxels)`, sphere-centered
      accuracy at each voxel

<details class="note" open markdown="1">
<summary>Note</summary>

Encoding-model timeseries prediction (`bd.predict(X=...)`) returns
a `BrainData` directly, not a `Predict` — the natural container for a
voxel timeseries.

Why the all-data fit is canonical: the CV mean of per-fold `coef_`
vectors doesn't correspond to any actual fitted estimator (each fold
saw a different subset). The all-data refit is a single, real model
with all the information used. CV gives the honest *score*; the refit
gives the publishable *map*. `fold_weight_maps` is still exposed for
stability analysis, and the CV-mean is one line away if you want it
(`fold_weight_maps.data.mean(axis=0)`).

Methods: `available` returns the names of non-None fields (excludes
private); `asdict` converts to a dict for serialization (private fields
always excluded).

</details>

**Methods:**

Name | Description
---- | -----------
[`asdict`](#data-fitresults-asdict) | Convert to dictionary.
[`available`](#data-fitresults-available) | Return names of non-None fields (excludes private).



#### Methods

##### `asdict`

```python
asdict(include_none: bool = False) -> dict
```

Convert to dictionary.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`include_none` | <code>bool</code> | If True, include fields with None values. Private fields (starting with _) are always excluded. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Dictionary of field names to values.

##### `available`

```python
available() -> list
```

Return names of non-None fields (excludes private).

(data-fitresults-predictcollection)=
### `PredictCollection`

```python
PredictCollection(results: tuple, metadata: Any = None, paths: tuple | None = None)
```

Immutable container for per-subject decoding results.

Returned by ``BrainCollection.predict(y=...)``: one `Predict` per subject
(each an independent within-subject model), plus the collection's
per-subject metadata. Sequence-like — ``len``, iteration, and integer
indexing all address the underlying `Predict` objects.

The stacking properties are the bridge to second-level inference: the
per-subject maps become one ``BrainData (n_subjects, n_voxels)``, ready
for a group test.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`results` | <code>tuple</code> | One `Predict` per subject, in collection order.
`metadata` | <code>Any</code> | Optional per-subject metadata (polars DataFrame, one row per subject), carried over from the source collection.
`paths` | <code>tuple \| None</code> | Optional on-disk predict-bundle paths, populated when the producing call cached its results (``cache=True``/``'auto'``).

<details class="note" open markdown="1">
<summary>Note</summary>

Properties: ``mean_scores`` / ``std_scores`` stack each subject's
score summary — ``(n_subjects,)`` for whole-brain decoding,
``(n_subjects, n_rois)`` for ROI. ``scores`` renders the whole-brain
case as a polars DataFrame alongside the metadata. ``weight_maps`` /
``accuracy_maps`` stack the per-subject brain maps into one
``BrainData``. ``available`` returns the field names populated on
*every* subject's result.

</details>

**Methods:**

Name | Description
---- | -----------
[`available`](#data-fitresults-available) | Return field names populated on every subject's `Predict`.



**Examples:**

```python
pc = collection.predict(y="condition", cv=5)
pc.scores                    # per-subject accuracy table
pc[0].weight_map.plot()      # one subject's decoder map

# Second-level inference on the decoder maps:
from nltools.algorithms import one_sample_permutation_test
group = one_sample_permutation_test(pc.weight_maps.data)
```

#### Methods

##### `available`

```python
available() -> list
```

Return field names populated on every subject's `Predict`.
