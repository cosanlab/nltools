---
title: data.collection.core
label: page-data-collection-core
---

Pure helpers behind `BrainCollection`: metadata coercion, mask and cache-dir resolution, run/step ids.

`coerce_metadata` and `resolve_mask` normalize constructor inputs;
`resolve_cache_dir` applies the cache-location precedence; `make_run_id`
and `make_step_dirname` name the cache root and its per-operation
subdirectories.

**Functions:**

Name | Description
---- | -----------
[`coerce_metadata`](#data-collection-core-coerce-metadata) | Coerce a metadata input into a polars DataFrame of length ``n_subjects``.
[`make_run_id`](#data-collection-core-make-run-id) | Build a fresh ``run_id`` of the form ``{timestamp}_{token}``.
[`make_step_dirname`](#data-collection-core-make-step-dirname) | Name a cache step subdirectory: ``{timestamp}_{seq}_{token}_{op}_{key_kwargs}``.
[`resolve_cache_dir`](#data-collection-core-resolve-cache-dir) | Resolve ``cache_dir`` in precedence order: explicit arg → ``NLTOOLS_CACHE_DIR`` → ``./.nltools_cache``.
[`resolve_mask`](#data-collection-core-resolve-mask) | Resolve a mask spec into a Nifti1Image.



## Functions

(data-collection-core-coerce-metadata)=
### `coerce_metadata`

```python
coerce_metadata(metadata: pl.DataFrame | pd.DataFrame | dict | None, n_subjects: int) -> pl.DataFrame
```

Coerce a metadata input into a polars DataFrame of length ``n_subjects``.

Metadata holds simple per-subject values only; DataFrames and arrays
(designs, confounds, sample masks) travel alongside it in their own
per-item slots.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metadata` | <code>DataFrame \| DataFrame \| dict \| None</code> | A polars or pandas DataFrame, a dict of columns, or ``None`` for a default ``subject`` column (``sub-0001``, ...). | *required*
`n_subjects` | <code>int</code> | Required number of rows. | *required*

**Returns:**

Type | Description
---- | -----------
<code>DataFrame</code> | One row per subject.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If the row count does not equal ``n_subjects``.

(data-collection-core-make-run-id)=
### `make_run_id`

```python
make_run_id(now: datetime | None = None) -> str
```

Build a fresh ``run_id`` of the form ``{timestamp}_{token}``.

The timestamp is UTC ``YYYYMMDDTHHMMSS``; the token is 8 random hex
characters, so ids sort lexicographically by time and do not collide
across processes.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`now` | <code>datetime \| None</code> | Timestamp to use; ``None`` means the current UTC time. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>str</code> | The run id.

(data-collection-core-make-step-dirname)=
### `make_step_dirname`

```python
make_step_dirname(op: str, kwargs: dict[str, Any] | None = None, *, now: datetime | None = None) -> str
```

Name a cache step subdirectory: ``{timestamp}_{seq}_{token}_{op}_{key_kwargs}``.

Each call yields a unique name (random token), so running the same op
with the same parameters twice produces two subdirectories and never
overwrites. The zero-padded ``seq`` is a process-monotonic counter placed
after the second-resolution timestamp, so lexicographic order tracks
creation order even for calls within the same second.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`op` | <code>str</code> | Short operation name (e.g. ``'smooth'``). | *required*
`kwargs` | <code>dict[str, Any] \| None</code> | Scalar kwargs to slug into the name (``{'fwhm': 6.0}`` → ``fwhm-6.0``); ``None`` values are skipped. | <code>None</code>
`now` | <code>datetime \| None</code> | Timestamp to use; ``None`` means the current UTC time. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>str</code> | The directory name (no path components).

(data-collection-core-resolve-cache-dir)=
### `resolve_cache_dir`

```python
resolve_cache_dir(cache_dir: Path | str | None) -> Path | None
```

Resolve ``cache_dir`` in precedence order: explicit arg → ``NLTOOLS_CACHE_DIR`` → ``./.nltools_cache``.

The environment variable is consulted only when the default
``'./.nltools_cache'`` was passed. The result is the cache *parent*; the
collection appends its own ``run_id`` subdirectory at construction.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cache_dir` | <code>Path \| str \| None</code> | Requested location, or ``None`` to ask for an auto-cleaned temp dir. | *required*

**Returns:**

Type | Description
---- | -----------
<code>Path \| None</code> | The resolved absolute path, or ``None`` when ``None`` was     passed.

(data-collection-core-resolve-mask)=
### `resolve_mask`

```python
resolve_mask(mask: nib.Nifti1Image | Path | str) -> nib.Nifti1Image
```

Resolve a mask spec into a Nifti1Image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask` | <code>Nifti1Image \| Path \| str</code> | An image, a path, or an nltools template name such as ``'3mm-MNI152-2009c'`` (resolved the same way `BrainData` resolves template masks). | *required*

**Returns:**

Type | Description
---- | -----------
<code>Nifti1Image</code> | The loaded mask.

**Raises:**

Type | Description
---- | -----------
<code>TypeError</code> | For any other input type.
