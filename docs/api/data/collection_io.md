(data-collection-io-io)=
## `io`

IO and constructors for BrainCollection.

Constructors (``from_bids``, ``from_glob``, ``from_paths``, ``read``),
write, load/unload, cache plumbing, and ``memory_estimate``. Anything that
crosses the disk boundary lives here.

**Methods:**

Name | Description
---- | -----------
[`discover_bids`](#data-collection-io-discover-bids) | Walk the BIDS dataset and return aligned per-item lists.
[`from_bids`](#data-collection-io-from-bids) | Build a ``BrainCollection`` from a BIDS dataset.
[`from_glob`](#data-collection-io-from-glob) | Build a collection by globbing for BOLD images (and optionally designs).
[`from_paths`](#data-collection-io-from-paths) | Build a collection from explicit lists of brain (and design) paths.
[`load`](#data-collection-io-load) | Materialize path-backed items into ``BrainData``.
[`memory_estimate`](#data-collection-io-memory-estimate) | Human-readable RAM estimate if every item were loaded.
[`read`](#data-collection-io-read) | Inverse of ``write()``: read images + ``metadata.csv`` from ``directory``.
[`unload`](#data-collection-io-unload) | Drop in-memory data for items that have backing paths.
[`write`](#data-collection-io-write) | Write a clean, portable copy of ``bc`` outside the cache root.



### Classes

### Methods

(data-collection-io-discover-bids)=
#### `discover_bids`

```python
discover_bids(root: Path | str | Any, *, task: str | None, space: str | None, sub_labels: list[str] | None, img_filters: list[tuple[str, str]] | None, derivatives_folder: str, confounds_strategy: str | tuple[str, ...] | None, confounds_kwargs: dict | None, TR: float | str) -> dict[str, list]
```

Walk the BIDS dataset and return aligned per-item lists.

Returns a dict with keys: ``bold_paths``, ``events_dfs``, ``confounds_dfs``,
``sample_masks``, ``metadata_rows``, ``TRs``. Each list is the same length
(one entry per BOLD file). Anything missing for an item is ``None``.

Errors (see ``docs/development/execution-model.md``):
  - Missing TR with ``TR='infer'``: raise.
  - ``task=None`` + ``pair_events=True``: caller silently downgrades.
  - fmriprep absent + ``confounds_strategy`` set: raise.
  - pybids not installed: raise ``ImportError``.

(data-collection-io-from-bids)=
#### `from_bids`

```python
from_bids(cls: type[BrainCollection], root: Path | str | Any, *, mask: nib.Nifti1Image | Path | str, task: str | None = None, space: str | None = None, sub_labels: list[str] | None = None, img_filters: list[tuple[str, str]] | None = None, derivatives_folder: str = 'derivatives', pair_events: bool = True, confounds_strategy: str | tuple[str, ...] | None = None, confounds_kwargs: dict | None = None, TR: float | str = 'infer', cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Build a ``BrainCollection`` from a BIDS dataset.

Delegates discovery to ``nilearn.glm.first_level.first_level_from_bids``
(which wraps pybids), drops the returned ``models``, and keeps paths +
events/confounds DataFrames. Per-item ``DesignMatrix`` is built from the
events DataFrame; convolution / drift / confound merging is **not** done
here — that's the user's ``transform_designs`` step.

See ``docs/development/execution-model.md`` for edge cases.

(data-collection-io-from-glob)=
#### `from_glob`

```python
from_glob(cls: type[BrainCollection], pattern: str, *, mask: nib.Nifti1Image | Path | str, design_pattern: str | None = None, pattern_groups: dict[str, int] | str | None = None, sort: bool = True, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Build a collection by globbing for BOLD images (and optionally designs).

``pattern_groups`` extracts metadata from filename wildcards. Pass
``{column_name: wildcard_index}`` (0-based) to capture each ``*`` in
``pattern`` into a metadata column.

(data-collection-io-from-paths)=
#### `from_paths`

```python
from_paths(cls: type[BrainCollection], brain_paths: list[Path | str], *, mask: nib.Nifti1Image | Path | str, design_paths: list[Path | str | None] | None = None, metadata: pl.DataFrame | pd.DataFrame | dict | None = None, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Build a collection from explicit lists of brain (and design) paths.

Always lazy — items are stored as ``Path`` and loaded on demand.

(data-collection-io-load)=
#### `load`

```python
load(bc: BrainCollection, indices: list[int] | None = None) -> BrainCollection
```

Materialize path-backed items into ``BrainData``.

Mutates ``bc`` in place. This is the only mutation method besides
``unload`` and does not allocate a step
subdir, does not write to disk, does not produce a new identity.

(data-collection-io-memory-estimate)=
#### `memory_estimate`

```python
memory_estimate(bc: BrainCollection) -> str
```

Human-readable RAM estimate if every item were loaded.

Reports ``n_subjects``, the per-item shape (or "unknown" if path-backed
and not yet loaded), and an estimated total in MB/GB based on float32.

(data-collection-io-read)=
#### `read`

```python
read(cls: type[BrainCollection], directory: Path | str, *, mask: nib.Nifti1Image | Path | str, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Inverse of ``write()``: read images + ``metadata.csv`` from ``directory``.

Discovers items by globbing ``image_*.nii*`` (matches the ``write()``
default pattern) and pairs them with rows from ``metadata.csv`` if it
exists. Does **not** recover from cache subdirs in v0.6.0.

(data-collection-io-unload)=
#### `unload`

```python
unload(bc: BrainCollection, indices: list[int] | None = None) -> BrainCollection
```

Drop in-memory data for items that have backing paths.

Mutates in place. This is a no-op for items that don't have a backing path
because dropping them would lose data.

(data-collection-io-write)=
#### `write`

```python
write(bc: BrainCollection, directory: Path | str, *, pattern: str = 'image_{i:04d}.nii.gz', metadata_file: str | None = 'metadata.csv') -> list[Path]
```

Write a clean, portable copy of ``bc`` outside the cache root.

Inverse of ``BrainCollection.read()``. Writes one NIfTI per item under
``directory`` plus a metadata CSV. Skips the cache layout entirely so
the result is shareable / archival.

