---
title: data.fitresults
label: page-data-fitresults
---

Immutable result containers returned by model fitting and decoding.

`Fit` holds the arrays produced by `BrainData.fit` (GLM or ridge, with or
without cross-validation); `Predict` holds the output of `BrainData.predict`
(scores, out-of-fold predictions, and brain-space weight / accuracy maps);
`PredictCollection` holds one `Predict` per subject from
`BrainCollection.predict`. All three are frozen dataclasses: fields not
computed for a given call are ``None`` and are dropped by `available` and
`asdict`.

**Classes:**

Name | Description
---- | -----------
[`Fit`](#data-fitresults-fit) | Immutable container for model fitting results.
[`Predict`](#data-fitresults-predict) | Immutable container for MVPA decoding results from `BrainData.predict`.
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

Plain numpy arrays with minimal introspection, so results can feed the
inference algorithms directly without a `BrainData`. Which fields are
populated depends on the model: **ridge** fills ``weights``, ``scores``,
and ``fitted_values``, plus the ``cv_*`` fields when fit with
cross-validation (``cv_best_alpha`` / ``cv_alpha_scores`` only under
``alpha='auto'``); **GLM** fills ``betas``, ``t_stats``, ``p_values``,
``se``, ``residuals``, ``r2``, and ``fitted_values``. Everything else is
``None`` and omitted from `available` and `asdict`.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`fitted_values` | <code>ndarray</code> | Fitted values / training predictions, ``(n_samples, n_voxels)``; always present.
`weights` | <code>ndarray \| None</code> | Ridge coefficients, ``(n_features, n_voxels)``.
`scores` | <code>ndarray \| None</code> | Ridge training R², ``(n_voxels,)``.
`betas` | <code>ndarray \| None</code> | GLM coefficients, ``(n_regressors, n_voxels)``.
`t_stats` | <code>ndarray \| None</code> | GLM t-statistics, ``(n_regressors, n_voxels)``.
`p_values` | <code>ndarray \| None</code> | GLM p-values, ``(n_regressors, n_voxels)``.
`se` | <code>ndarray \| None</code> | GLM standard errors, ``(n_regressors, n_voxels)``.
`residuals` | <code>ndarray \| None</code> | GLM residuals, ``(n_samples, n_voxels)``.
`r2` | <code>ndarray \| None</code> | GLM R², ``(n_voxels,)``.
`cv_scores` | <code>ndarray \| None</code> | Per-fold CV R², ``(n_folds, n_voxels)``.
`cv_mean_score` | <code>ndarray \| None</code> | Mean CV R² across folds, ``(n_voxels,)``.
`cv_predictions` | <code>ndarray \| None</code> | Out-of-fold predictions, ``(n_samples, n_voxels)``.
`cv_folds` | <code>ndarray \| None</code> | Fold index per sample, ``(n_samples,)``.
`cv_best_alpha` | <code>float \| None</code> | Alpha selected under ``alpha='auto'``.
`cv_alpha_scores` | <code>ndarray \| None</code> | Score per candidate alpha under ``alpha='auto'``.

**Methods:**

Name | Description
---- | -----------
[`asdict`](#data-fitresults-asdict) | Convert to dictionary.
[`available`](#data-fitresults-available) | Return list of non-None attribute names.



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

The dataclass is frozen, so results cannot be modified by accident.
Every field is a numpy array except ``cv_best_alpha`` (a float); a
``None`` value means the field was not computed for this model.

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

Immutable container for MVPA decoding results from `BrainData.predict`.

Mirrors `Fit`: frozen, every field defaults to ``None``, and which fields
are populated depends on ``spatial_scale``. Fields not applicable to the
call stay ``None`` and are dropped by `available` and `asdict`.

**Brain-space outputs are `BrainData` objects**, not raw arrays, so
``result.weight_map.plot()`` works directly (``.data`` gives the array).
Non-spatial fields (``predictions``, ``cv_folds``, scalar scores) are
numpy.

**Populated by `spatial_scale`.** ``'whole_brain'``: ``predictions``,
``scores``, ``mean_score``, ``std_score``, ``cv_folds``, ``weight_map``,
``fold_weight_maps``, ``estimator``. ``'roi'``: ``scores``,
``mean_score``, ``std_score``, ``roi_labels``, ``accuracy_map``,
``weight_map``, ``fold_weight_maps``, ``estimator`` — and if any parcel's
model cannot expose ``coef_`` (a non-linear model, or feature selection in
the pipeline), ``weight_map`` / ``fold_weight_maps`` / ``estimator`` are
all ``None`` for the whole call. ``'searchlight'``: ``accuracy_map`` only.

**Why the all-data fit is the canonical map.** The mean of per-fold
``coef_`` vectors corresponds to no actual fitted estimator (each fold saw
a different subset). The all-data refit is one real model using all the
information; CV gives the honest *score*, the refit gives the publishable
*map*. ``fold_weight_maps`` is still exposed for stability analysis, and
the CV mean is ``fold_weight_maps.data.mean(axis=0)``.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`predictions` | <code>ndarray \| None</code> | Out-of-fold CV predictions, ``(n_samples,)`` (whole-brain only).
`scores` | <code>ndarray \| None</code> | Per-fold score — ``(n_folds,)`` for whole-brain, ``(n_folds, n_rois)`` for ROI.
`mean_score` | <code>float \| ndarray \| None</code> | Mean score across folds — a float for whole-brain, ``(n_rois,)`` for ROI.
`std_score` | <code>float \| ndarray \| None</code> | Score standard deviation across folds, same form as ``mean_score``.
`cv_folds` | <code>ndarray \| None</code> | Fold index per sample, ``(n_samples,)`` (whole-brain only).
`roi_labels` | <code>ndarray \| None</code> | Atlas integer ids, ``(n_rois,)``, in the order of ``mean_score`` / ``std_score`` / ``scores`` axis 1 (ROI only).
`accuracy_map` | <code>[BrainData](#page-data-brain-data) \| None</code> | ``(1, n_voxels)`` map — for ROI, every voxel in parcel *i* holds that parcel's mean score (NaN outside parcels); for searchlight, the sphere-centered score at each voxel.
`weight_map` | <code>[BrainData](#page-data-brain-data) \| None</code> | ``(1, n_voxels)`` ``coef_`` of the model refit on all data — the publishable map. For ROI, each parcel's coefficients are written back into voxel space (NaN outside parcels); magnitudes are not comparable across parcels.
`fold_weight_maps` | <code>[BrainData](#page-data-brain-data) \| None</code> | ``(n_folds, n_voxels)`` stack of per-fold ``coef_`` for stability analysis.
`estimator` | <code>Any</code> | The fitted all-data sklearn estimator (whole-brain; use it to ``.predict()`` on new data), or a ``dict[int, estimator]`` keyed by atlas label (ROI). ``None`` when read back from a cached bundle.
`permutation_scores` | <code>ndarray \| None</code> | Label-permutation null from `BrainCollection.predict_group` — ``(n_permute,)`` for whole-brain, ``(n_permute, n_rois)`` for ROI, ``(n_permute, n_voxels)`` for searchlight.
`permutation_pvalue` | <code>float \| ndarray \| [BrainData](#page-data-brain-data) \| None</code> | Upper-tail permutation p-value — a float, ``(n_rois,)``, or a ``(1, n_voxels)`` `BrainData` map, matching ``permutation_scores``.

<details class="note" open markdown="1">
<summary>Note</summary>

Encoding-model timeseries prediction (``bd.predict(X=...)``) returns a
`BrainData` directly rather than a `Predict` — the natural container
for a voxel timeseries.

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
`results` | <code>tuple[[Predict](#data-fitresults-predict), ...]</code> | One `Predict` per subject, in collection order.
`metadata` | <code>DataFrame \| None</code> | Per-subject metadata (one row per subject), carried over from the source collection.
`paths` | <code>tuple[Path \| None, ...] \| None</code> | On-disk predict-bundle paths, populated when the producing call cached its results (``cache=True`` / ``'auto'``).
`mean_scores` | <code>ndarray</code> | Stacked per-subject mean CV score — ``(n_subjects,)`` for whole-brain decoding, ``(n_subjects, n_rois)`` for ROI.
`std_scores` | <code>ndarray</code> | Stacked per-subject score standard deviation across folds, same shape as ``mean_scores``.
`scores` | <code>DataFrame</code> | Per-subject score table — the metadata plus ``mean_score`` / ``std_score`` columns. Whole-brain decoding only; ROI results raise (use ``mean_scores`` / ``std_scores``).
`weight_maps` | <code>[BrainData](#page-data-brain-data)</code> | Per-subject decoder maps stacked into one ``(n_subjects, n_voxels)`` `BrainData`.
`accuracy_maps` | <code>[BrainData](#page-data-brain-data)</code> | Per-subject accuracy maps stacked into one ``(n_subjects, n_voxels)`` `BrainData`.

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
