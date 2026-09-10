---
title: algorithms.decoding
label: page-algorithms-decoding
---

Coefficient back-projection for MVPA decoding pipelines.

A decoding pipeline preprocesses voxels before it fits, so the coefficients it
learns live on the preprocessed feature axis, not on the voxel axis a brain map
needs. These functions walk a *fitted* scikit-learn estimator or `Pipeline`
backwards and return coefficients on the original voxel axis.

Everything here is a pure function over fitted scikit-learn objects: no
`BrainData`, no file I/O, no global state. `nltools.data.braindata.prediction`
orchestrates; this module does the numerics.

Only the preprocessing steps in `SUPPORTED_TRANSFORMERS` are accepted. A
transformer outside that set is rejected even when it implements
`inverse_transform`, because inverting a *data* transformation is not the same
operation as back-projecting a *coefficient* vector: `Normalizer`, for
instance, rescales each observation rather than each feature, so its
coefficients have no fixed voxel-space image.

Centering is deliberately not undone. It shifts the intercept of the raw-space
decision function without changing its slope map, so `raw_data @ weight_map`
need not reproduce the decision function. Use the fitted estimator itself to
predict.

**Classes:**

Name | Description
---- | -----------
[`BackProjectionError`](#algorithms-decoding-backprojectionerror) | A fitted pipeline's coefficients cannot be projected onto the voxel axis.

**Functions:**

Name | Description
---- | -----------
[`back_project_step`](#algorithms-decoding-back-project-step) | Project coefficients backwards through one fitted preprocessing step.
[`back_project_weight_maps`](#algorithms-decoding-back-project-weight-maps) | Project a fitted pipeline's coefficients onto the original feature axis.
[`coefficient_rows`](#algorithms-decoding-coefficient-rows) | Return a fitted estimator's coefficients as ``(n_maps, n_final_features)``.
[`is_passthrough`](#algorithms-decoding-is-passthrough) | Whether a pipeline step is a placeholder that transforms nothing.
[`split_pipeline`](#algorithms-decoding-split-pipeline) | Split a decoding pipeline into its preprocessing steps and final estimator.
[`validate_decoding_pipeline`](#algorithms-decoding-validate-decoding-pipeline) | Check a pipeline's structure before anything is fitted.
[`whitening_scale`](#algorithms-decoding-whitening-scale) | Return the per-component scale a whitened `PCA` divides its output by.



## Classes

(algorithms-decoding-backprojectionerror)=
### `BackProjectionError`

Bases: `ValueError`

A fitted pipeline's coefficients cannot be projected onto the voxel axis.

A `ValueError` subclass, so callers that catch `ValueError` — including the
public `BrainData.predict` contract — see it as one, while the internal
runners can still tell it apart from an unrelated fit failure.



## Functions

(algorithms-decoding-back-project-step)=
### `back_project_step`

```python
back_project_step(weights: np.ndarray, step: Any) -> np.ndarray
```

Project coefficients backwards through one fitted preprocessing step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`weights` | <code>ndarray</code> | Coefficients on the step's *output* feature axis, ``(n_maps, n_output_features)``. | *required*
`step` | <code>Any</code> | The fitted transformer, or a passthrough placeholder. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Coefficients on the step's *input* feature axis,     ``(n_maps, n_input_features)``.

**Raises:**

Type | Description
---- | -----------
<code>[BackProjectionError](#algorithms-decoding-backprojectionerror)</code> | If the step is unsupported, or its fitted output width does not match the incoming coefficients.

(algorithms-decoding-back-project-weight-maps)=
### `back_project_weight_maps`

```python
back_project_weight_maps(fitted_estimator: Any, n_features: int) -> np.ndarray
```

Project a fitted pipeline's coefficients onto the original feature axis.

Starts from ``(n_maps, n_final_features)`` coefficients and walks the fitted
preprocessing steps in reverse order, validating each step's widths, until
the weights sit on the axis the pipeline was fitted from — the whole-brain,
parcel, or sphere voxel axis, depending on the caller.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fitted_estimator` | <code>Any</code> | A fitted estimator or `Pipeline`. | *required*
`n_features` | <code>int</code> | Width of the original feature axis the maps must land on. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | ``(n_maps, n_features)`` — one row for regression and binary     classification, one row per class for multiclass.

**Raises:**

Type | Description
---- | -----------
<code>[BackProjectionError](#algorithms-decoding-backprojectionerror)</code> | If the final estimator exposes no ``coef_``, a step is unsupported or misplaced, or any width does not line up.

**Examples:**

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

pipe = make_pipeline(StandardScaler(), LinearSVC()).fit(X, y)
maps = back_project_weight_maps(pipe, X.shape[1])
# → (1, n_voxels), in raw voxel units
```

(algorithms-decoding-coefficient-rows)=
### `coefficient_rows`

```python
coefficient_rows(final_estimator: Any) -> np.ndarray
```

Return a fitted estimator's coefficients as ``(n_maps, n_final_features)``.

One row for a regressor or a binary classifier — the signed map for
``classes_[1]`` versus ``classes_[0]`` — and one row per class, in
``classes_`` order, for a multiclass classifier. Rows are never averaged: a
mean across classes describes no fitted decision boundary.

`OneVsRestClassifier` is handled explicitly because it exposes no combined
``coef_``; its fitted children are read in class order instead.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`final_estimator` | <code>Any</code> | The fitted estimator ending the pipeline. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Coefficients, ``(n_maps, n_final_features)``.

**Raises:**

Type | Description
---- | -----------
<code>[BackProjectionError](#algorithms-decoding-backprojectionerror)</code> | If the estimator exposes no usable ``coef_``.

(algorithms-decoding-is-passthrough)=
### `is_passthrough`

```python
is_passthrough(step: Any) -> bool
```

Whether a pipeline step is a placeholder that transforms nothing.

(algorithms-decoding-split-pipeline)=
### `split_pipeline`

```python
split_pipeline(pipeline: Any) -> tuple[list, Any]
```

Split a decoding pipeline into its preprocessing steps and final estimator.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`pipeline` | <code>Any</code> | A `Pipeline`, or a bare estimator (a pipeline of one step). | *required*

**Returns:**

Type | Description
---- | -----------
<code>tuple</code> | ``(steps, final_estimator)``. A bare estimator has no steps.

(algorithms-decoding-validate-decoding-pipeline)=
### `validate_decoding_pipeline`

```python
validate_decoding_pipeline(pipeline: Any) -> None
```

Check a pipeline's structure before anything is fitted.

Catches the two failures that are visible without fitting: a preprocessing
step outside `SUPPORTED_TRANSFORMERS`, and a `OneVsRestClassifier` that is
not the final step. Whether the final estimator exposes ``coef_`` can only
be observed after a fit, so `back_project_weight_maps` checks that.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`pipeline` | <code>Any</code> | The estimator or `Pipeline` MVPA is about to fit. | *required*

**Raises:**

Type | Description
---- | -----------
<code>[BackProjectionError](#algorithms-decoding-backprojectionerror)</code> | If a step is unsupported or misplaced.

(algorithms-decoding-whitening-scale)=
### `whitening_scale`

```python
whitening_scale(explained_variance: np.ndarray) -> np.ndarray
```

Return the per-component scale a whitened `PCA` divides its output by.

``sqrt(explained_variance_)``, with values below the dtype's epsilon
replaced by that epsilon so a degenerate component cannot blow the
back-projected weights up to infinity.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`explained_variance` | <code>ndarray</code> | The fitted ``PCA.explained_variance_`` vector. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | The floored component scales, one per component.
