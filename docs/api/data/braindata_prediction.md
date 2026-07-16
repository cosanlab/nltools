(data-braindata-prediction-prediction)=
## `prediction`

BrainData prediction — timeseries (encoding) and MVPA (decoding).

Single entry point: `predict`. Returns `Predict`
with fields populated based on dispatch. Mirrors `BrainData.fit` /
`Fit` patterns: frozen result dataclass, ``inplace=True`` mutates
self with attributes, ``inplace=False`` returns the dataclass.

**Methods:**

Name | Description
---- | -----------
[`build_pipeline`](#data-braindata-prediction-build-pipeline) | Build a per-fold scikit-learn preprocessing and model pipeline.
[`predict`](#data-braindata-prediction-predict) | Dispatch BrainData prediction to timeseries encoding or MVPA decoding.
[`predict_mvpa`](#data-braindata-prediction-predict-mvpa) | Cross-validated decoding. Returns Predict (or self if inplace=True).
[`predict_timeseries`](#data-braindata-prediction-predict-timeseries) | Predict voxel timeseries from a fitted encoding model.
[`resolve_model`](#data-braindata-prediction-resolve-model) | Resolve a string shortcut or pass through a sklearn estimator.
[`resolve_scoring`](#data-braindata-prediction-resolve-scoring) | Resolve scoring='auto' to 'accuracy' (classifier) or 'r2' (regressor).

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`VALID_SPATIAL_SCALES` |  | 

### Methods

(data-braindata-prediction-build-pipeline)=
#### `build_pipeline`

```python
build_pipeline(model, standardize: bool, reduce: str | None, n_components: str | None)
```

Build a per-fold scikit-learn preprocessing and model pipeline.

The pipeline contains an optional StandardScaler, optional PCA, and the
model. If only the model is needed, returns the model itself.

(data-braindata-prediction-predict)=
#### `predict`

```python
predict(bd, *, y = None, X = None, spatial_scale: str = 'whole_brain', model: Any = 'svm', cv: int = 5, standardize: bool = True, reduce: str | None = None, n_components: int | None = None, scoring: str = 'auto', groups: str = None, roi_mask: str = None, radius_mm: float = 10.0, inplace: bool = False, n_jobs: int = 1, progress_bar: bool = False)
```

Dispatch BrainData prediction to timeseries encoding or MVPA decoding.

Implements `BrainData.predict`. See the class docstring for full parameter
documentation.

(data-braindata-prediction-predict-mvpa)=
#### `predict_mvpa`

```python
predict_mvpa(bd, *, y, spatial_scale: str, model: Any, cv: Any, standardize: bool, reduce: str | None, n_components: int | None, scoring: str, groups: str, roi_mask: str, radius_mm: float, inplace: bool, n_jobs: int, progress_bar: bool) -> Predict | Any
```

Cross-validated decoding. Returns Predict (or self if inplace=True).

(data-braindata-prediction-predict-timeseries)=
#### `predict_timeseries`

```python
predict_timeseries(bd, *, X = None)
```

Predict voxel timeseries from a fitted encoding model.

Returns a fresh ``BrainData`` whose ``.data`` is the predicted timeseries.
Encoding model prediction yields a brain image — the natural container is
``BrainData``, so it composes directly with downstream methods (`.plot()`,
`.standardize()`, etc.). MVPA decoding (``y=`` mode) returns ``Predict``.

(data-braindata-prediction-resolve-model)=
#### `resolve_model`

```python
resolve_model(model: Any)
```

Resolve a string shortcut or pass through a sklearn estimator.

(data-braindata-prediction-resolve-scoring)=
#### `resolve_scoring`

```python
resolve_scoring(scoring: str, classifier: bool) -> str
```

Resolve scoring='auto' to 'accuracy' (classifier) or 'r2' (regressor).

