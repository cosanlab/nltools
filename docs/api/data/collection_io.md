---
title: data.collection.io
label: page-data-collection-io
---

Constructors and disk IO for `BrainCollection`.

Builds collections from a BIDS dataset (`from_bids`, `discover_bids`), a glob
(`from_glob`), explicit paths (`from_paths`), or a directory written by
`write` (`read`); moves items between disk and memory (`load`, `unload`); and
estimates the memory a fully loaded collection would need
(`memory_estimate`). The `BrainCollection` methods of the same names
delegate here.

**Functions:**

Name | Description
---- | -----------
[`discover_bids`](#data-collection-io-discover-bids) | Walk a BIDS dataset and return aligned per-run lists.
[`from_bids`](#data-collection-io-from-bids) | Build a `BrainCollection` from a BIDS dataset.
[`from_glob`](#data-collection-io-from-glob) | Build a collection by globbing for brain images (and optionally designs).
[`from_paths`](#data-collection-io-from-paths) | Build a collection from explicit lists of brain (and design) paths.
[`load`](#data-collection-io-load) | Load path-backed items into memory as `BrainData`.
[`memory_estimate`](#data-collection-io-memory-estimate) | Human-readable RAM estimate if every item were loaded.
[`read`](#data-collection-io-read) | Read a collection from a directory written by `write`.
[`unload`](#data-collection-io-unload) | Drop in-memory data for items that have a backing path.
[`write`](#data-collection-io-write) | Write a clean, portable copy of ``bc`` outside the cache root.

## Functions

(data-collection-io-discover-bids)=
### `discover_bids`

```python
discover_bids(root: Path | str | Any, *, task: str | None, space: str | None, sub_labels: list[str] | None, img_filters: list[tuple[str, str]] | None, derivatives_folder: str, confounds_strategy: str | tuple[str, ...] | None, confounds_kwargs: dict | None, TR: float | str) -> dict[str, list]
```

Walk a BIDS dataset and return aligned per-run lists.

With ``task=None`` only BOLD files are discovered (no events, no
confounds); designs are left unset by the caller.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`root` | <code>Path \| str</code> | BIDS dataset root. | *required*
`task` | <code>str \| None</code> | BIDS task label, or ``None`` for BOLD-only discovery. | *required*
`space` | <code>str \| None</code> | Only keep images in this ``space-`` entity. | *required*
`sub_labels` | <code>list[str] \| None</code> | Restrict to these subject labels. | *required*
`img_filters` | <code>list[tuple[str, str]] \| None</code> | Extra BIDS ``(entity, value)`` filters on image filenames. | *required*
`derivatives_folder` | <code>str</code> | Preprocessed-derivatives folder name under ``root``. | *required*
`confounds_strategy` | <code>str \| tuple[str, ...] \| None</code> | fMRIPrep confounds strategy forwarded to nilearn's ``load_confounds``. | *required*
`confounds_kwargs` | <code>dict \| None</code> | Extra keyword arguments for ``load_confounds``. | *required*
`TR` | <code>float \| str</code> | Repetition time in seconds, or ``'infer'``. | *required*

**Returns:**

Type | Description
---- | -----------
<code>dict[str, list]</code> | Keys ``bold_paths``, ``events_dfs``, ``confounds_dfs``,     ``sample_masks``, ``metadata_rows``, ``TRs``. Every list has one     entry per BOLD file; anything missing for a run is ``None``.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If ``TR='infer'`` finds no repetition time for a run, or no BOLD files match.
<code>ImportError</code> | If nilearn/pybids are unavailable, or ``confounds_strategy`` is set without fMRIPrep support in nilearn.

(data-collection-io-from-bids)=
### `from_bids`

```python
from_bids(cls: type[BrainCollection], root: Path | str | Any, *, mask: nib.Nifti1Image | Path | str, task: str | None = None, space: str | None = None, sub_labels: list[str] | None = None, img_filters: list[tuple[str, str]] | None = None, derivatives_folder: str = 'derivatives', pair_events: bool = True, confounds_strategy: str | tuple[str, ...] | None = None, confounds_kwargs: dict | None = None, TR: float | str = 'infer', cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Build a `BrainCollection` from a BIDS dataset.

Discovery goes through ``nilearn.glm.first_level.first_level_from_bids``
(which wraps pybids); the returned models are discarded and only the BOLD
paths plus events/confounds DataFrames are kept. Each events DataFrame
becomes an unconvolved `DesignMatrix` — HRF convolution, drift terms, and
confound columns are left to `BrainCollection.transform_designs`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cls` | <code>type[[BrainCollection](#page-data-brain-collection)]</code> | The collection class to construct. | *required*
`root` | <code>Path \| str</code> | BIDS dataset root. | *required*
`mask` | <code>Nifti1Image \| Path \| str</code> | Mask shared by every item. | *required*
`task` | <code>str \| None</code> | BIDS task label. ``None`` discovers BOLD files without pairing events (designs are all ``None``). | <code>None</code>
`space` | <code>str \| None</code> | Only keep images in this ``space-`` entity. | <code>None</code>
`sub_labels` | <code>list[str] \| None</code> | Restrict to these subject labels. | <code>None</code>
`img_filters` | <code>list[tuple[str, str]] \| None</code> | Extra BIDS ``(entity, value)`` filters on image filenames. | <code>None</code>
`derivatives_folder` | <code>str</code> | Preprocessed-derivatives folder name under ``root``. | <code>'derivatives'</code>
`pair_events` | <code>bool</code> | If True (and ``task`` is set), build a `DesignMatrix` from each run's ``events.tsv``. Runs without one get ``None`` and a warning. | <code>True</code>
`confounds_strategy` | <code>str \| tuple[str, ...] \| None</code> | fMRIPrep confounds strategy forwarded to nilearn's ``load_confounds``. | <code>None</code>
`confounds_kwargs` | <code>dict \| None</code> | Extra keyword arguments for ``load_confounds``. | <code>None</code>
`TR` | <code>float \| str</code> | Repetition time in seconds, or ``'infer'`` to read it from the BIDS sidecars. | <code>'infer'</code>
`cache_dir` | <code>Path \| str \| None</code> | Cache location; see `BrainCollection`. | <code>'./.nltools_cache'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | A lazy, path-backed collection with per-run designs,     confounds, and sample masks attached.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If no BOLD files are discovered.

(data-collection-io-from-glob)=
### `from_glob`

```python
from_glob(cls: type[BrainCollection], pattern: str, *, mask: nib.Nifti1Image | Path | str, design_pattern: str | None = None, pattern_groups: dict[str, int] | str | None = None, sort: bool = True, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Build a collection by globbing for brain images (and optionally designs).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cls` | <code>type[[BrainCollection](#page-data-brain-collection)]</code> | The collection class to construct. | *required*
`pattern` | <code>str</code> | Glob matching one brain image per subject. | *required*
`mask` | <code>Nifti1Image \| Path \| str</code> | Mask shared by every item. | *required*
`design_pattern` | <code>str \| None</code> | Glob matching per-subject design files, paired positionally with the images (match counts must agree). | <code>None</code>
`pattern_groups` | <code>dict[str, int] \| None</code> | Metadata extracted from the filename wildcards — ``{column_name: wildcard_index}`` (0-based) captures each ``*`` in ``pattern`` into a metadata column. | <code>None</code>
`sort` | <code>bool</code> | If True, sort matched paths before pairing. | <code>True</code>
`cache_dir` | <code>Path \| str \| None</code> | Cache location; see `BrainCollection`. | <code>'./.nltools_cache'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | A lazy, path-backed collection.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If ``pattern`` matches nothing, or ``design_pattern`` matches a different number of files.

(data-collection-io-from-paths)=
### `from_paths`

```python
from_paths(cls: type[BrainCollection], brain_paths: list[Path | str], *, mask: nib.Nifti1Image | Path | str, design_paths: list[Path | str | None] | None = None, metadata: pl.DataFrame | pd.DataFrame | dict | None = None, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Build a collection from explicit lists of brain (and design) paths.

Always lazy — items are stored as paths and loaded on demand.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cls` | <code>type[[BrainCollection](#page-data-brain-collection)]</code> | The collection class to construct. | *required*
`brain_paths` | <code>list[Path \| str]</code> | One brain image path per subject. | *required*
`mask` | <code>Nifti1Image \| Path \| str</code> | Mask shared by every item. | *required*
`design_paths` | <code>list[Path \| str \| None] \| None</code> | Per-subject design paths aligned with ``brain_paths`` (``None`` entries allowed). | <code>None</code>
`metadata` | <code>DataFrame \| DataFrame \| dict \| None</code> | Per-subject table, one row per path. | <code>None</code>
`cache_dir` | <code>Path \| str \| None</code> | Cache location; see `BrainCollection`. | <code>'./.nltools_cache'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | A lazy, path-backed collection.

(data-collection-io-load)=
### `load`

```python
load(bc: BrainCollection, indices: list[int] | None = None) -> BrainCollection
```

Load path-backed items into memory as `BrainData`.

Mutates ``bc`` in place (`load` and `unload` are the only operations
that do). Nothing is written to disk and no cache step is recorded.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#page-data-brain-collection)</code> | The collection to load. | *required*
`indices` | <code>list[int] \| None</code> | Items to load; ``None`` loads all. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | ``bc`` itself, for chaining.

(data-collection-io-memory-estimate)=
### `memory_estimate`

```python
memory_estimate(bc: BrainCollection) -> str
```

Human-readable RAM estimate if every item were loaded.

The shape is read from the first in-memory item, or from the first item
loaded on demand when none is in memory. The total assumes float32.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#page-data-brain-collection)</code> | The collection to estimate. | *required*

**Returns:**

Type | Description
---- | -----------
<code>str</code> | ``n_subjects``, the per-item shape, and the estimated total in     human-readable units.

(data-collection-io-read)=
### `read`

```python
read(cls: type[BrainCollection], directory: Path | str, *, mask: nib.Nifti1Image | Path | str, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Read a collection from a directory written by `write`.

Discovers items by globbing ``image_*.nii*`` (the `write` default
pattern) and pairs them with the rows of ``metadata.csv`` when present.
Only this portable layout is readable — not the cache directories.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cls` | <code>type[[BrainCollection](#page-data-brain-collection)]</code> | The collection class to construct. | *required*
`directory` | <code>Path \| str</code> | Directory produced by `write`. | *required*
`mask` | <code>Nifti1Image \| Path \| str</code> | Mask shared by every item. | *required*
`cache_dir` | <code>Path \| str \| None</code> | Cache location; see `BrainCollection`. | <code>'./.nltools_cache'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | A lazy, path-backed collection.

**Raises:**

Type | Description
---- | -----------
<code>FileNotFoundError</code> | If ``directory`` does not exist.
<code>ValueError</code> | If it holds no ``image_*.nii*`` files.

(data-collection-io-unload)=
### `unload`

```python
unload(bc: BrainCollection, indices: list[int] | None = None) -> BrainCollection
```

Drop in-memory data for items that have a backing path.

Mutates ``bc`` in place. Items without a backing path are left untouched,
since dropping them would lose the data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#page-data-brain-collection)</code> | The collection to unload. | *required*
`indices` | <code>list[int] \| None</code> | Items to unload; ``None`` unloads all. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | ``bc`` itself, for chaining.

(data-collection-io-write)=
### `write`

```python
write(bc: BrainCollection, directory: Path | str, *, pattern: str = 'image_{i:04d}.nii.gz', metadata_file: str | None = 'metadata.csv') -> list[Path]
```

Write a clean, portable copy of ``bc`` outside the cache root.

Inverse of `read`. Writes one NIfTI per item under ``directory`` plus a
metadata CSV, without the cache layout, so the result is shareable and
archival.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#page-data-brain-collection)</code> | The collection to write. | *required*
`directory` | <code>Path \| str</code> | Output directory (created if missing). | *required*
`pattern` | <code>str</code> | Filename template per item, formatted with ``i`` (item index). | <code>'image_{i:04d}.nii.gz'</code>
`metadata_file` | <code>str \| None</code> | CSV filename for the metadata table, or ``None`` to skip it. | <code>'metadata.csv'</code>

**Returns:**

Type | Description
---- | -----------
<code>list[Path]</code> | Written NIfTI paths, in item order.
