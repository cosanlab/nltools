## `execution`

Parallel execution machinery for BrainCollection.

Holds the worker-side dataclasses (``_ItemTask``, ``_DesignContext``), the
single parallel primitive (``_apply``), the worker-error type, and the
HDF5 fit-bundle IO. Every per-subject method on ``BrainCollection`` routes
through ``_apply`` here.

**Classes:**

Name | Description
---- | -----------
[`BrainCollectionWorkerError`](#BrainCollectionWorkerError) | Raised in the parent process when a worker fails inside ``_apply``.
[`tqdm_joblib`](#tqdm_joblib) | Context manager that updates a tqdm bar as joblib workers complete.

**Methods:**

Name | Description
---- | -----------
[`read_glm_bundle`](#read_glm_bundle) | Read and validate a GLM bundle.
[`read_ridge_bundle`](#read_ridge_bundle) | Read a ridge bundle.
[`write_glm_bundle`](#write_glm_bundle) | Write a GLM fit bundle to ``out_path`` (atomic tmp+rename).
[`write_ridge_bundle`](#write_ridge_bundle) | Write a ridge fit bundle to ``out_path`` (atomic tmp+rename).

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`BUNDLE_SCHEMA_VERSION`](#BUNDLE_SCHEMA_VERSION) |  | 

### Methods

#### `read_glm_bundle`

```python
read_glm_bundle(path: Path) -> dict[str, Any]
```

Read and validate a GLM bundle.

Validates ``bundle_schema_version``. A schema-version mismatch raises with
a migration message; nltools-version
mismatch logs a warning but does not refuse — bundles are usually
forward-compatible within a minor version.

#### `read_ridge_bundle`

```python
read_ridge_bundle(path: Path) -> dict[str, Any]
```

Read a ridge bundle.

Uses the same schema and version handling as ``read_glm_bundle``.

#### `write_glm_bundle`

```python
write_glm_bundle(out_path: Path, *, betas: np.ndarray, residuals: np.ndarray, sigma2: np.ndarray, r2: np.ndarray, X: np.ndarray, mask_bytes: bytes, affine: np.ndarray, regressor_names: list[str], scale: bool, scale_value: float, model_kwargs: dict, step_id: str, parent_step_id: str | None, op: str, op_kwargs: dict, nltools_version: str) -> Path
```

Write a GLM fit bundle to ``out_path`` (atomic tmp+rename).

Layout (see SPEC §"HDF5 fit bundle"):
    /betas, /residuals, /sigma2, /r2, /X, /mask
    attrs: affine, regressor_names, scale, scale_value, model_kwargs,
           nltools_version, bundle_schema_version,
           step_id, parent_step_id, op, kwargs (JSON-encoded).

Mask is embedded as a dataset (raw NIfTI bytes) so the bundle is
portable across machines. Uses ``h5py.File(..., locking=False)``.

#### `write_ridge_bundle`

```python
write_ridge_bundle(out_path: Path, *, weights: np.ndarray, cv_scores: np.ndarray, predictions: np.ndarray, scores: np.ndarray, X: np.ndarray, mask_bytes: bytes, affine: np.ndarray, regressor_names: list[str], model_kwargs: dict, step_id: str, parent_step_id: str | None, op: str, op_kwargs: dict, nltools_version: str) -> Path
```

Write a ridge fit bundle to ``out_path`` (atomic tmp+rename).

Parallel layout to ``write_glm_bundle`` with ridge-specific datasets
(``weights``, ``cv_scores``, ``predictions``, ``scores``).

