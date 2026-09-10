---
title: data.braindata.prediction
label: page-data-braindata-prediction
---

BrainData prediction — timeseries (encoding) and MVPA (decoding).

Single entry point: `predict`. It resolves exactly one mode, validates every
argument for that mode, and returns either a new `BrainData` (fitted-model
prediction) or a frozen `Predict` record (MVPA). Nothing is attached to the
source object.

**Functions:**

Name | Description
---- | -----------
[`build_pipeline`](#data-braindata-prediction-build-pipeline) | Build the per-fold pipeline for `estimator`.
[`predict`](#data-braindata-prediction-predict) | Dispatch BrainData prediction to fitted-model prediction or MVPA decoding.
[`predict_mvpa`](#data-braindata-prediction-predict-mvpa) | Run cross-validated decoding on a `BrainData` and return a `Predict`.
[`predict_timeseries`](#data-braindata-prediction-predict-timeseries) | Predict voxel timeseries from a fitted encoding model.
[`resolve_estimator`](#data-braindata-prediction-resolve-estimator) | Resolve a shortcut name to an estimator, or pass an sklearn object through.
[`resolve_splits`](#data-braindata-prediction-resolve-splits) | Resolve `cv` into materialized train/test splits and check the partition.
[`validate_scoring`](#data-braindata-prediction-validate-scoring) | Reject the removed `'auto'` value and multimetric scoring mappings.

## Functions

(data-braindata-prediction-build-pipeline)=
### `build_pipeline`

```python
build_pipeline(estimator: Any, *, y: np.ndarray) -> Any
```

Build the per-fold pipeline for `estimator`.

A built-in shortcut selects a predefined pipeline: `StandardScaler` inside
each fold, then the linear estimator the shortcut names. A classification
shortcut on a multiclass target is wrapped in `OneVsRestClassifier`, which
gives one signed coefficient row per class instead of whatever multiclass
strategy the estimator happens to default to.

A caller-supplied estimator or `Pipeline` is used exactly as given — MVPA
adds, removes, and reconfigures nothing, and never overrides its multiclass
strategy. Callers who want one-vs-rest supply a `OneVsRestClassifier`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`estimator` | <code>Any</code> | A shortcut name or an sklearn estimator/`Pipeline`. | *required*
`y` | <code>ndarray</code> | The validated target vector, used only to decide whether a classification shortcut faces a multiclass problem. | *required*

**Returns:**

Type | Description
---- | -----------
<code>Any</code> | The estimator to clone and fit in every fold.

(data-braindata-prediction-predict)=
### `predict`

```python
predict(bd, *, X = None, y = None, estimator: Any = 'linear_svc', cv: Any = None, groups: Any = None, scoring: Any = None, spatial_scale: str = 'whole_brain', roi_mask: str = None, radius: float = 10.0, n_jobs: int = 1, progress_bar: bool = False)
```

Dispatch BrainData prediction to fitted-model prediction or MVPA decoding.

Implements `BrainData.predict`. See that method's docstring for full
parameter documentation.

(data-braindata-prediction-predict-mvpa)=
### `predict_mvpa`

```python
predict_mvpa(bd, *, y, estimator: Any, cv: Any, groups: Any, scoring: Any, spatial_scale: str, roi_mask: str, radius: float, n_jobs: int, progress_bar: bool) -> Predict
```

Run cross-validated decoding on a `BrainData` and return a `Predict`.

Every argument is validated, and the cross-validation folds are
materialized and checked, before a single model is fitted.

(data-braindata-prediction-predict-timeseries)=
### `predict_timeseries`

```python
predict_timeseries(bd, *, X = None)
```

Predict voxel timeseries from a fitted encoding model.

Returns a fresh ``BrainData`` whose ``.data`` is the predicted timeseries.
Encoding model prediction yields a brain image — the natural container is
``BrainData``, so it composes directly with downstream methods (`.plot()`,
`.standardize()`, etc.). MVPA decoding (``y=`` mode) returns ``Predict``.

With no ``X``, the fitted model returns an independent copy of the stored
training predictions and keeps their row metadata: ``glm_predicted`` for a
GLM, ``ridge_fitted_values`` for a Ridge. Neither retains the training
features, so a no-argument call never refits or re-multiplies. With an
explicit ``X``, structural validation and alignment belong to the
estimator's own ``predict`` — named design columns for `Glm`, named feature
spaces for a banded `Ridge` — and the result clears the source row metadata.

(data-braindata-prediction-resolve-estimator)=
### `resolve_estimator`

```python
resolve_estimator(estimator: Any)
```

Resolve a shortcut name to an estimator, or pass an sklearn object through.

(data-braindata-prediction-resolve-splits)=
### `resolve_splits`

```python
resolve_splits(cv, *, X, y, groups, classifier: bool) -> list
```

Resolve `cv` into materialized train/test splits and check the partition.

Materializing once means every runner — and every parallel worker — sees
the same folds, and it lets the partition rule be checked before any model
is fitted.

(data-braindata-prediction-validate-scoring)=
### `validate_scoring`

```python
validate_scoring(scoring) -> None
```

Reject the removed `'auto'` value and multimetric scoring mappings.
