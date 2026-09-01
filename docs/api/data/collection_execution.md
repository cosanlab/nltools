(data-collection-execution-execution)=
## `execution`

Parallel execution machinery for BrainCollection.

Holds the worker-side dataclasses (``_ItemTask``, ``_DesignContext``), the
single parallel primitive (``_apply``), the worker-error type, and the
HDF5 fit-bundle IO. Every per-subject method on ``BrainCollection`` routes
through ``_apply`` here.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`BUNDLE_SCHEMA_VERSION` |  | 



**Classes:**

Name | Description
---- | -----------
[`BrainCollectionWorkerError`](#data-collection-execution-braincollectionworkererror) | Raised in the parent process when a worker fails inside ``_apply``.
[`tqdm_joblib`](#data-collection-execution-tqdm-joblib) | Context manager that updates a tqdm bar as joblib workers complete.

**Methods:**

Name | Description
---- | -----------
[`detect_bundle_kind`](#data-collection-execution-detect-bundle-kind) | Classify an HDF5 file as a bundle kind, or ``None`` for plain data.
[`read_glm_bundle`](#data-collection-execution-read-glm-bundle) | Read a GLM bundle. Validates ``bundle_schema_version``.
[`read_predict_bundle`](#data-collection-execution-read-predict-bundle) | Read a predict bundle back into a `Predict` (``estimator`` is ``None``).
[`read_ridge_bundle`](#data-collection-execution-read-ridge-bundle) | Read a ridge bundle. Same schema/version handling as ``read_glm_bundle``.
[`write_glm_bundle`](#data-collection-execution-write-glm-bundle) | Write a GLM fit bundle to ``out_path`` (atomic tmp+rename).
[`write_predict_bundle`](#data-collection-execution-write-predict-bundle) | Write a per-subject decoding bundle to ``out_path`` (atomic tmp+rename).
[`write_ridge_bundle`](#data-collection-execution-write-ridge-bundle) | Write a ridge fit bundle to ``out_path`` (atomic tmp+rename).

### Classes

(data-collection-execution-braincollectionworkererror)=
#### `BrainCollectionWorkerError`

Bases: <code>[RuntimeError](#RuntimeError)</code>

Raised in the parent process when a worker fails inside ``_apply``.

Wraps the original exception via ``raise ... from e`` so the full
traceback is preserved. The message embeds subject/run context from
``_ItemTask.metadata_row`` so users can locate the offending item.

(data-collection-execution-tqdm-joblib)=
#### `tqdm_joblib`

```python
tqdm_joblib(total: int, desc: str = '', disable: bool = False) -> None
```

Context manager that updates a tqdm bar as joblib workers complete.

Replaces today's submit-time wrapper, which advances on dispatch rather
than completion. Monkey-patches ``BatchCompletionCallBack.__call__`` for
the duration of the ``with`` block.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`desc` |  | 
`disable` |  | 
`total` |  | 

### Methods

(data-collection-execution-detect-bundle-kind)=
#### `detect_bundle_kind`

```python
detect_bundle_kind(path: Path | str) -> str | None
```

Classify an HDF5 file as a bundle kind, or ``None`` for plain data.

The structured replacement for bare-suffix checks, which misclassified
user-saved ``BrainData`` ``.h5`` images as fit bundles. Detection order:

1. The ``bundle_kind`` attr (``'glm'`` | ``'ridge'`` | ``'predict'``) —
   stamped by every bundle writer.
2. Dataset sniff for dev-cycle bundles written before the attr existed:
   a file carrying ``bundle_schema_version`` with a ``weights`` dataset
   is a ridge bundle, with ``betas`` a GLM bundle.
3. Otherwise not a bundle (e.g. a user-saved BrainData ``.h5``).

Non-``.h5``/``.hdf5`` paths and unreadable files return ``None``.

(data-collection-execution-read-glm-bundle)=
#### `read_glm_bundle`

```python
read_glm_bundle(path: Path) -> dict[str, Any]
```

Read a GLM bundle. Validates ``bundle_schema_version``.

Schema-version mismatch raises with a migration message; nltools-version
mismatch logs a warning but does not refuse — bundles are usually
forward-compatible within a minor version.

(data-collection-execution-read-predict-bundle)=
#### `read_predict_bundle`

```python
read_predict_bundle(path: Path)
```

Read a predict bundle back into a `Predict` (``estimator`` is ``None``).

Brain-map fields are rebuilt as ``BrainData`` on the embedded mask. Same
schema/version handling as the fit-bundle readers; refuses non-predict
bundles with a pointer to the right reader.

(data-collection-execution-read-ridge-bundle)=
#### `read_ridge_bundle`

```python
read_ridge_bundle(path: Path) -> dict[str, Any]
```

Read a ridge bundle. Same schema/version handling as ``read_glm_bundle``.

(data-collection-execution-write-glm-bundle)=
#### `write_glm_bundle`

```python
write_glm_bundle(out_path: Path, *, betas: np.ndarray, residuals: np.ndarray, sigma2: np.ndarray, r2: np.ndarray, X: np.ndarray, mask_bytes: bytes, affine: np.ndarray, regressor_names: list[str], scale: bool, standardize: str | None, model_kwargs: dict, step_id: str, parent_step_id: str | None, op: str, op_kwargs: dict, nltools_version: str) -> Path
```

Write a GLM fit bundle to ``out_path`` (atomic tmp+rename).

Layout (see ``docs/development/execution-model.md``):
    /betas, /residuals, /sigma2, /r2, /X, /mask
    attrs: affine, regressor_names, scale, standardize, model_kwargs,
           nltools_version, bundle_schema_version,
           step_id, parent_step_id, op, kwargs (JSON-encoded).

Mask is embedded as a dataset (raw NIfTI bytes) so the bundle is
portable across machines. Uses ``h5py.File(..., locking=False)``.

(data-collection-execution-write-predict-bundle)=
#### `write_predict_bundle`

```python
write_predict_bundle(out_path: Path, *, result: Any, mask_bytes: bytes, affine: np.ndarray, model_spec: dict, step_id: str, parent_step_id: str | None, op: str, op_kwargs: dict, nltools_version: str) -> Path
```

Write a per-subject decoding bundle to ``out_path`` (atomic tmp+rename).

Layout (see ``docs/development/execution-model.md``):
    datasets: every populated array field of the `Predict` (predictions,
    scores, cv_folds, roi_labels, permutation_scores, mean_score,
    std_score), each brain-map field's ``.data`` (weight_map,
    fold_weight_maps, accuracy_map), and /mask (raw NIfTI bytes).
    attrs: bundle_kind='predict', present_fields, scalar_summaries,
    permutation_pvalue (when set), model_spec (JSON — the refit
    ingredients; its ``model`` entry is a structured spec from
    ``_serialize_model_spec``: shortcut name or estimator class + params,
    or an explicit ``refittable: false`` marker when the estimator's
    params can't be serialized), affine, plus the shared lineage attrs.

The fitted ``estimator`` is deliberately not persisted; rebuild one via
``nltools.data.braindata.prediction._model_from_spec`` when the spec is
refittable.

(data-collection-execution-write-ridge-bundle)=
#### `write_ridge_bundle`

```python
write_ridge_bundle(out_path: Path, *, weights: np.ndarray, intercept: np.ndarray, cv_scores: np.ndarray, predictions: np.ndarray, scores: np.ndarray, X: np.ndarray, mask_bytes: bytes, affine: np.ndarray, regressor_names: list[str], model_kwargs: dict, step_id: str, parent_step_id: str | None, op: str, op_kwargs: dict, nltools_version: str) -> Path
```

Write a ridge fit bundle to ``out_path`` (atomic tmp+rename).

Parallel layout to ``write_glm_bundle`` with ridge-specific datasets
(``weights``, ``intercept``, ``cv_scores``, ``predictions``, ``scores``).



### Modules