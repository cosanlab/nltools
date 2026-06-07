## `core`

Module-level helpers for BrainCollection.

Pure functions: metadata coercion, mask resolution, run/step ID generation,
step-directory naming. No class state lives here.

**Methods:**

Name | Description
---- | -----------
[`coerce_metadata`](#coerce_metadata) | Coerce a metadata input into a polars DataFrame of length ``n_subjects``.
[`make_run_id`](#make_run_id) | Build a fresh ``run_id`` of the form ``{timestamp}_{uuid8}``.
[`make_step_dirname`](#make_step_dirname) | Name a step subdir: ``{timestamp}_{uuid8}_{op}_{key_kwargs}/``.
[`resolve_cache_dir`](#resolve_cache_dir) | Resolve ``cache_dir`` per the spec's precedence rules.
[`resolve_mask`](#resolve_mask) | Resolve a mask spec into a Nifti1Image.



### Methods

#### `coerce_metadata`

```python
coerce_metadata(metadata: pl.DataFrame | pd.DataFrame | dict | None, n_subjects: int) -> pl.DataFrame
```

Coerce a metadata input into a polars DataFrame of length ``n_subjects``.

Accepts polars/pandas DataFrames or a dict-of-columns. ``None`` yields a
DataFrame with a default ``subject`` column (``sub-0001``, ...).

Polars ``metadata`` cannot hold DataFrames or arrays — those belong in
the parallel slots (``designs``, ``_confounds``, ``_sample_masks``).

#### `make_run_id`

```python
make_run_id(now: datetime | None = None) -> str
```

Build a fresh ``run_id`` of the form ``{timestamp}_{uuid8}``.

Timestamp is UTC ``YYYYMMDDTHHMMSS``; the uuid tail is 8 hex chars from
``secrets.token_hex(4)``. Lex-sortable, collision-free across processes.

#### `make_step_dirname`

```python
make_step_dirname(op: str, kwargs: dict[str, Any] | None = None, *, now: datetime | None = None) -> str
```

Name a step subdir: ``{timestamp}_{uuid8}_{op}_{key_kwargs}/``.

Each call yields a unique name (UUID tail) — same op + same params
twice produces two subdirs, never overwriting.

#### `resolve_cache_dir`

```python
resolve_cache_dir(cache_dir: Path | str | None) -> Path | None
```

Resolve ``cache_dir`` per the spec's precedence rules.

Order: explicit arg → ``NLTOOLS_CACHE_DIR`` env var → ``./.nltools_cache``.
Returns ``None`` when the caller passes ``None`` (signaling tempdir mode).
The returned path is *not* yet decorated with a ``run_id`` subdir; that
happens at construction time on the instance.

#### `resolve_mask`

```python
resolve_mask(mask: nib.Nifti1Image | Path | str) -> nib.Nifti1Image
```

Resolve a mask spec into a Nifti1Image.

Accepts a Nifti1Image, a path, or a known nltools template string
(e.g. ``"3mm-MNI152-2009c"``). String templates dispatch to the same
resolver used by ``BrainData``.

