## `prediction`

BrainData prediction — timeseries (encoding) and MVPA (decoding).

Single entry point: :func:`predict`. Returns :class:`nltools.data.fitresults.Predict`
with fields populated based on dispatch. Mirrors :meth:`BrainData.fit` /
:class:`Fit` patterns: frozen result dataclass, ``inplace=True`` mutates
self with attributes, ``inplace=False`` returns the dataclass.

**Methods:**

Name | Description
---- | -----------
[`build_pipeline`](#build_pipeline) | Build a per-fold sklearn pipeline: optional StandardScaler → optional
[`predict`](#predict) | Implementation of :meth:`BrainData.predict`. See class docstring for
[`predict_mvpa`](#predict_mvpa) | Cross-validated decoding. Returns Predict (or self if inplace=True).
[`predict_timeseries`](#predict_timeseries) | Predict voxel timeseries from a fitted encoding model.
[`resolve_model`](#resolve_model) | Resolve a string shortcut or pass through a sklearn estimator.
[`resolve_scoring`](#resolve_scoring) | Resolve scoring='auto' to 'accuracy' (classifier) or 'r2' (regressor).

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`VALID_SPATIAL_SCALES`](#VALID_SPATIAL_SCALES) |  | 

### Methods

#### `build_pipeline`

```python
build_pipeline(model, standardize: bool, reduce: str | None, n_components: str | None)
```

Build a per-fold sklearn pipeline: optional StandardScaler → optional
PCA → model. If only the model is needed, returns the model itself.

#### `predict`

```python
predict(bd, *, y = None, X = None, spatial_scale: str = 'whole_brain', model: Any = 'svm', cv: int = 5, standardize: bool = True, reduce: str | None = None, n_components: int | None = None, scoring: str = 'auto', groups: str = None, roi_mask: str = None, radius_mm: float = 10.0, inplace: bool = False, n_jobs: int = 1, progress_bar: bool = False)
```

Implementation of :meth:`BrainData.predict`. See class docstring for
full parameter documentation.

#### `predict_mvpa`

```python
predict_mvpa(bd, *, y, spatial_scale: str, model: Any, cv: Any, standardize: bool, reduce: str | None, n_components: int | None, scoring: str, groups: str, roi_mask: str, radius_mm: float, inplace: bool, n_jobs: int, progress_bar: bool) -> Predict | Any
```

Cross-validated decoding. Returns Predict (or self if inplace=True).

#### `predict_timeseries`

```python
predict_timeseries(bd, *, X = None)
```

Predict voxel timeseries from a fitted encoding model.

Returns a fresh ``BrainData`` whose ``.data`` is the predicted timeseries.
Encoding model prediction yields a brain image — the natural container is
``BrainData``, so it composes directly with downstream methods (`.plot()`,
`.standardize()`, etc.). MVPA decoding (``y=`` mode) returns ``Predict``.

#### `resolve_model`

```python
resolve_model(model: Any)
```

Resolve a string shortcut or pass through a sklearn estimator.

#### `resolve_scoring`

```python
resolve_scoring(scoring: str, classifier: bool) -> str
```

Resolve scoring='auto' to 'accuracy' (classifier) or 'r2' (regressor).

