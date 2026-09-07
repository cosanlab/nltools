---
title: data.collection.execution
label: page-data-collection-execution
---

Parallel execution and on-disk bundle formats behind `BrainCollection`.

Every per-subject `BrainCollection` method runs its workers through this
module. Users meet it through `BrainCollectionWorkerError` (raised when a
worker fails, with the offending subject in the message), the readers and
writers for the HDF5 bundles that `BrainCollection.fit` and
`BrainCollection.predict` cache (`read_glm_bundle`, `read_ridge_bundle`,
`read_predict_bundle`, and their ``write_*`` counterparts,
`detect_bundle_kind`, `BUNDLE_SCHEMA_VERSION`), and `tqdm_joblib`, a
progress bar for joblib work. The execution model itself is described in
``docs/development/execution-model.md``.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`BUNDLE_SCHEMA_VERSION` |  | Schema version stamped into every HDF5 bundle; bumped on any breaking change to the on-disk layout.



**Classes:**

Name | Description
---- | -----------
[`BrainCollectionWorkerError`](#data-collection-execution-braincollectionworkererror) | Raised when a per-subject worker of a `BrainCollection` operation fails.
[`tqdm_joblib`](#data-collection-execution-tqdm-joblib) | Context manager that advances a progress bar as joblib workers *complete*.

**Functions:**

Name | Description
---- | -----------
[`detect_bundle_kind`](#data-collection-execution-detect-bundle-kind) | Classify an HDF5 file as a bundle kind, or ``None`` for plain data.
[`read_glm_bundle`](#data-collection-execution-read-glm-bundle) | Read a GLM bundle written by `write_glm_bundle`.
[`read_predict_bundle`](#data-collection-execution-read-predict-bundle) | Read a predict bundle written by `write_predict_bundle` back into a `Predict`.
[`read_ridge_bundle`](#data-collection-execution-read-ridge-bundle) | Read a ridge bundle written by `write_ridge_bundle`.
[`write_glm_bundle`](#data-collection-execution-write-glm-bundle) | Write a GLM fit bundle to ``out_path`` (atomic tmp+rename).
[`write_predict_bundle`](#data-collection-execution-write-predict-bundle) | Write a per-subject decoding bundle to ``out_path`` (atomic tmp+rename).
[`write_ridge_bundle`](#data-collection-execution-write-ridge-bundle) | Write a ridge fit bundle to ``out_path`` (atomic tmp+rename).

## Classes

(data-collection-execution-braincollectionworkererror)=
### `BrainCollectionWorkerError`

Bases: `RuntimeError`

Raised when a per-subject worker of a `BrainCollection` operation fails.

The message starts with the item's index and, when available, its
``subject`` and ``run`` metadata (``[idx=3, subject=sub-04] ValueError:
...``) so the offending item can be located. The original exception is
chained as ``__cause__``, preserving its traceback.

(data-collection-execution-tqdm-joblib)=
### `tqdm_joblib`

```python
tqdm_joblib(total: int, desc: str = '', disable: bool = False) -> None
```

Context manager that advances a progress bar as joblib workers *complete*.

Wrap a ``joblib.Parallel(...)`` call in ``with tqdm_joblib(total=n):`` to
get a bar that tracks finished tasks rather than dispatched ones. Works by
patching joblib's batch-completion callback for the duration of the
``with`` block.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`total` | <code>int</code> | Number of tasks the bar counts to. | *required*
`desc` | <code>str</code> | Label shown next to the bar. | <code>''</code>
`disable` | <code>bool</code> | If True, show nothing and patch nothing. | <code>False</code>



## Functions

(data-collection-execution-detect-bundle-kind)=
### `detect_bundle_kind`

```python
detect_bundle_kind(path: Path | str) -> str | None
```

Classify an HDF5 file as a bundle kind, or ``None`` for plain data.

Checks the ``bundle_kind`` attribute stamped by every bundle writer, then
falls back to sniffing datasets (a ``weights`` dataset means ridge, a
``betas`` dataset means GLM) for bundles written before that attribute
existed. A user-saved `BrainData` ``.h5`` image has neither and is not a
bundle.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`path` | <code>Path \| str</code> | File to inspect. | *required*

**Returns:**

Type | Description
---- | -----------
<code>str \| None</code> | ``'glm'``, ``'ridge'``, or ``'predict'``; ``None`` for     non-bundles, non-``.h5``/``.hdf5`` paths, and unreadable files.

(data-collection-execution-read-glm-bundle)=
### `read_glm_bundle`

```python
read_glm_bundle(path: Path) -> dict[str, Any]
```

Read a GLM bundle written by `write_glm_bundle`.

A ``bundle_schema_version`` mismatch raises; an nltools-version mismatch
only warns, since bundles are usually compatible within a minor version.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`path` | <code>Path</code> | Bundle path. | *required*

**Returns:**

Type | Description
---- | -----------
<code>dict[str, Any]</code> | The datasets (``betas``, ``residuals``, ``sigma2``,     ``r2``, ``X``, ``mask_bytes``) and decoded attributes (``affine``,     ``regressor_names``, ``scale``, ``standardize``, ``model_kwargs``,     ``step_id``, ``parent_step_id``, ``op``, ``kwargs``,     ``nltools_version``, ``bundle_schema_version``).

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If the bundle's schema version differs from `BUNDLE_SCHEMA_VERSION`.

(data-collection-execution-read-predict-bundle)=
### `read_predict_bundle`

```python
read_predict_bundle(path: Path)
```

Read a predict bundle written by `write_predict_bundle` back into a `Predict`.

Brain-map fields are rebuilt as `BrainData` on the embedded mask;
``estimator`` is always ``None`` (see `write_predict_bundle`). Same
schema/version handling as the fit-bundle readers.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`path` | <code>Path</code> | Bundle path. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Predict](#data-results-predict)</code> | The reconstructed result.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If the schema version differs from `BUNDLE_SCHEMA_VERSION`, or the file is a GLM/ridge bundle rather than a predict bundle.

(data-collection-execution-read-ridge-bundle)=
### `read_ridge_bundle`

```python
read_ridge_bundle(path: Path) -> dict[str, Any]
```

Read a ridge bundle written by `write_ridge_bundle`.

Same schema/version handling as `read_glm_bundle`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`path` | <code>Path</code> | Bundle path. | *required*

**Returns:**

Type | Description
---- | -----------
<code>dict[str, Any]</code> | The datasets (``weights``, ``intercept``,     ``cv_scores``, ``predictions``, ``scores``, ``X``, ``mask_bytes``)     and decoded attributes (``affine``, ``regressor_names``,     ``model_kwargs``, ``step_id``, ``parent_step_id``, ``op``,     ``kwargs``, ``nltools_version``, ``bundle_schema_version``).

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If the bundle's schema version differs from `BUNDLE_SCHEMA_VERSION`.

(data-collection-execution-write-glm-bundle)=
### `write_glm_bundle`

```python
write_glm_bundle(out_path: Path, *, betas: np.ndarray, residuals: np.ndarray, sigma2: np.ndarray, r2: np.ndarray, X: np.ndarray, mask_bytes: bytes, affine: np.ndarray, regressor_names: list[str], scale: bool, standardize: str | None, model_kwargs: dict, step_id: str, parent_step_id: str | None, op: str, op_kwargs: dict, nltools_version: str) -> Path
```

Write a GLM fit bundle to ``out_path`` (atomic tmp+rename).

Datasets: ``betas``, ``residuals``, ``sigma2``, ``r2``, ``X``, and
``mask`` (raw NIfTI bytes, so the bundle is portable across machines).
Attributes: ``affine``, ``regressor_names``, ``scale``, ``standardize``,
``model_kwargs``, ``nltools_version``, ``bundle_schema_version``, and the
lineage fields ``step_id``, ``parent_step_id``, ``op``, ``kwargs``
(dict-valued attributes are JSON-encoded).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`out_path` | <code>Path</code> | Destination ``.h5`` path. | *required*
`betas` | <code>ndarray</code> | Regression coefficients, ``(n_regressors, n_voxels)``. | *required*
`residuals` | <code>ndarray</code> | Residuals, ``(n_obs, n_voxels)``. | *required*
`sigma2` | <code>ndarray</code> | Residual variance per voxel. | *required*
`r2` | <code>ndarray</code> | R² per voxel. | *required*
`X` | <code>ndarray</code> | Design matrix used for the fit. | *required*
`mask_bytes` | <code>bytes</code> | The mask image serialized as NIfTI bytes. | *required*
`affine` | <code>ndarray</code> | The data's affine. | *required*
`regressor_names` | <code>list[str]</code> | Column names of ``X``. | *required*
`scale` | <code>bool</code> | Whether percent-signal-change scaling was applied. | *required*
`standardize` | <code>str \| None</code> | Standardization applied before fitting. | *required*
`model_kwargs` | <code>dict</code> | Extra fit keyword arguments. | *required*
`step_id` | <code>str</code> | Id of the cache step producing this bundle. | *required*
`parent_step_id` | <code>str \| None</code> | Id of the upstream step. | *required*
`op` | <code>str</code> | Operation name (e.g. ``'fit_glm'``). | *required*
`op_kwargs` | <code>dict</code> | Scalar kwargs of the operation. | *required*
`nltools_version` | <code>str</code> | Version of nltools writing the bundle. | *required*

**Returns:**

Type | Description
---- | -----------
<code>Path</code> | ``out_path``.

(data-collection-execution-write-predict-bundle)=
### `write_predict_bundle`

```python
write_predict_bundle(out_path: Path, *, result: Any, mask_bytes: bytes, affine: np.ndarray, model_spec: dict, step_id: str, parent_step_id: str | None, op: str, op_kwargs: dict, nltools_version: str) -> Path
```

Write a per-subject decoding bundle to ``out_path`` (atomic tmp+rename).

Datasets: every populated array field of the `Predict` (``predictions``,
``scores``, ``cv_folds``, ``roi_labels``, ``permutation_scores``,
``mean_score``, ``std_score``), the ``.data`` of each brain-map field
(``weight_map``, ``fold_weight_maps``, ``accuracy_map``), and ``mask``
(raw NIfTI bytes). Attributes: ``bundle_kind='predict'``,
``present_fields``, ``scalar_summaries``, ``permutation_pvalue`` (when
set), ``model_spec`` (JSON), ``affine``, and the shared lineage fields.

The fitted ``estimator`` is deliberately not persisted — pickled
estimators are version-fragile. ``model_spec`` carries what is needed to
refit one instead: the model shortcut name, or the estimator class and
its parameters, or an explicit not-refittable marker when the parameters
cannot be serialized.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`out_path` | <code>Path</code> | Destination ``.h5`` path. | *required*
`result` | <code>[Predict](#data-results-predict)</code> | The decoding result to persist. | *required*
`mask_bytes` | <code>bytes</code> | The mask image serialized as NIfTI bytes. | *required*
`affine` | <code>ndarray</code> | The data's affine. | *required*
`model_spec` | <code>dict</code> | Refit ingredients (model spec, ``spatial_scale``, ``cv``, ``scoring``, ``standardize``, ...). | *required*
`step_id` | <code>str</code> | Id of the cache step producing this bundle. | *required*
`parent_step_id` | <code>str \| None</code> | Id of the upstream step. | *required*
`op` | <code>str</code> | Operation name (e.g. ``'predict_mvpa'``). | *required*
`op_kwargs` | <code>dict</code> | Scalar kwargs of the operation. | *required*
`nltools_version` | <code>str</code> | Version of nltools writing the bundle. | *required*

**Returns:**

Type | Description
---- | -----------
<code>Path</code> | ``out_path``.

(data-collection-execution-write-ridge-bundle)=
### `write_ridge_bundle`

```python
write_ridge_bundle(out_path: Path, *, weights: np.ndarray, intercept: np.ndarray, cv_scores: np.ndarray, predictions: np.ndarray, scores: np.ndarray, X: np.ndarray, mask_bytes: bytes, affine: np.ndarray, regressor_names: list[str], model_kwargs: dict, step_id: str, parent_step_id: str | None, op: str, op_kwargs: dict, nltools_version: str) -> Path
```

Write a ridge fit bundle to ``out_path`` (atomic tmp+rename).

Same layout as `write_glm_bundle` with ridge datasets in place of the GLM
ones: ``weights``, ``intercept``, ``cv_scores``, ``predictions``,
``scores``, ``X``, ``mask``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`out_path` | <code>Path</code> | Destination ``.h5`` path. | *required*
`weights` | <code>ndarray</code> | Ridge coefficients, ``(n_features, n_voxels)``. | *required*
`intercept` | <code>ndarray</code> | Per-voxel intercepts. | *required*
`cv_scores` | <code>ndarray</code> | Per-fold cross-validation scores. | *required*
`predictions` | <code>ndarray</code> | Fitted values, ``(n_obs, n_voxels)``. | *required*
`scores` | <code>ndarray</code> | Training scores per voxel. | *required*
`X` | <code>ndarray</code> | Design matrix used for the fit. | *required*
`mask_bytes` | <code>bytes</code> | The mask image serialized as NIfTI bytes. | *required*
`affine` | <code>ndarray</code> | The data's affine. | *required*
`regressor_names` | <code>list[str]</code> | Column names of ``X``. | *required*
`model_kwargs` | <code>dict</code> | Extra fit keyword arguments. | *required*
`step_id` | <code>str</code> | Id of the cache step producing this bundle. | *required*
`parent_step_id` | <code>str \| None</code> | Id of the upstream step. | *required*
`op` | <code>str</code> | Operation name (e.g. ``'fit_ridge'``). | *required*
`op_kwargs` | <code>dict</code> | Scalar kwargs of the operation. | *required*
`nltools_version` | <code>str</code> | Version of nltools writing the bundle. | *required*

**Returns:**

Type | Description
---- | -----------
<code>Path</code> | ``out_path``.
