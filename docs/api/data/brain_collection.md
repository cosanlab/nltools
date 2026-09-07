---
title: BrainCollection
label: page-data-brain-collection
---

```python
BrainCollection(brains: list, *, mask: nib.Nifti1Image | Path | str, designs: list | None = None, metadata: pl.DataFrame | pd.DataFrame | dict | None = None, lazy: bool = True, cache_dir: Path | str | None = './.nltools_cache')
```

A lazy, parallel collection of `BrainData` — one item per subject — whose API mirrors `BrainData`.

Every item shares one mask and lines up with an optional paired
`DesignMatrix` (`designs`) and one row of `metadata` (a polars
DataFrame). Build one from explicit lists (`BrainCollection(...)`) or
from disk with `from_bids`, `from_glob`, `from_paths`, or `read`.

Items are **lazy** by default: each is held as a file path and loaded
into a `BrainData` only when accessed (`bc[i]`, iteration) or inside a
worker. `load` / `unload` switch items between the two states in place
and are the only methods that mutate a collection.

Per-subject methods (`smooth`, `fit`, `predict`, `map`, `apply`, ...)
run over every item in parallel and return a **new** collection. With
``cache='auto'`` (the default) their outputs are written to the
collection's `cache_root` when the inputs were path-backed and kept in
memory otherwise; ``cache=True``/``False`` force either behavior. Every
cached step lands in its own subdirectory and the chain of steps that
produced a collection is available from `steps`; `cleanup` removes the
whole cache root. Group reductions (`mean`, `ttest`, `isc`, ...)
return in-memory `BrainData` (or dicts of them) and never cache.

Indexing: ``bc[i]`` → `BrainData`; ``bc[i:j]``, ``bc[list]``,
``bc[bool_mask]``, ``bc[polars_expr]`` → `BrainCollection`;
``bc['sub-01']`` → `BrainData` looked up in ``metadata['subject']``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brains` | <code>list[[BrainData](#page-data-brain-data) \| Path \| str]</code> | One brain image per subject — in-memory `BrainData` objects or paths to NIfTI/HDF5 files. | *required*
`mask` | <code>Nifti1Image \| Path \| str</code> | Mask shared by every item — an image, a path, or an nltools template name (e.g. ``'3mm-MNI152-2009c'``). | *required*
`designs` | <code>list[[DesignMatrix](#page-data-design-matrix) \| Path \| str \| None] \| None</code> | Optional per-subject designs, aligned positionally with ``brains`` (``None`` entries allowed; length must match). | <code>None</code>
`metadata` | <code>DataFrame \| DataFrame \| dict \| None</code> | Per-subject table (one row per item). ``None`` creates a default ``subject`` column (``sub-0001``, ...). | <code>None</code>
`lazy` | <code>bool</code> | If True, path items stay as paths until accessed; if False, they are loaded into `BrainData` up front. | <code>True</code>
`cache_dir` | <code>Path \| str \| None</code> | Where cached outputs go. Precedence: explicit arg → ``NLTOOLS_CACHE_DIR`` env var → ``./.nltools_cache``. ``None`` uses a temp dir that is removed at process exit. Resolved once, at construction. | <code>'./.nltools_cache'</code>



**Attributes:**

Name | Type | Description
---- | ---- | -----------
`cache_root` | <code>Path</code> | Run-scoped cache directory shared by clones.
`designs` | <code>list</code> | Per-subject paired designs (a copy of the list; ``None`` where unpaired).
`is_loaded` | <code>list[bool]</code> | Per-item flag — True iff the slot holds a ``BrainData`` (not a path).
`mask` | <code>Nifti1Image</code> | Shared mask image for the collection.
`metadata` | <code>DataFrame</code> | Per-subject metadata as a polars DataFrame (one row per item).
`n_subjects` | <code>int</code> | Number of subjects (items) in the collection.
`n_voxels` | <code>int</code> | Voxel count from the mask.
`shape` | <code>tuple[int, int \| None, int]</code> | Collection shape as ``(n_subjects, n_obs_or_None_if_ragged, n_voxels)``.

**Methods:**

Name | Description
---- | -----------
[`align`](#data-brain-collection-align) | Functionally align subjects into a common space via `LocalAlignment`.
[`anova`](#data-brain-collection-anova) | One-way ANOVA across subjects grouped by ``groups``.
[`apply`](#data-brain-collection-apply) | Call ``BrainData.<op>(*args, **kwargs)`` on every item in parallel.
[`cleanup`](#data-brain-collection-cleanup) | Remove ``cache_root`` and invalidate every clone derived from ``self``.
[`cleanup_all`](#data-brain-collection-cleanup-all) | Remove every ``.nltools_cache/{run_id}/`` under ``directory``.
[`compute_contrasts`](#data-brain-collection-compute-contrasts) | Compute per-subject contrast maps from fit-bundle items (the output of `fit`).
[`concat`](#data-brain-collection-concat) | Stack all subject maps into a single `BrainData` (subjects as rows).
[`detrend`](#data-brain-collection-detrend) | Detrend every subject's image in parallel (delegates to `BrainData.detrend`).
[`filter`](#data-brain-collection-filter) | Filter to a subset by predicate, polars expression, or boolean array.
[`fit`](#data-brain-collection-fit) | Fit a GLM or ridge model to every subject in parallel (delegates to `BrainData.fit`).
[`from_bids`](#data-brain-collection-from-bids) | Build a collection from a BIDS dataset, pairing each BOLD run with its events and confounds.
[`from_glob`](#data-brain-collection-from-glob) | Build a collection by glob-matching brain images (and optional designs).
[`from_paths`](#data-brain-collection-from-paths) | Build a collection from explicit lists of brain (and design) paths.
[`isc`](#data-brain-collection-isc) | Inter-subject correlation (ISC) across the time dimension.
[`isc_test`](#data-brain-collection-isc-test) | Bootstrap inference on ISC (per-voxel p-values).
[`iter_pairs`](#data-brain-collection-iter-pairs) | Yield ``(BrainData, DesignMatrix | None)`` pairs.
[`load`](#data-brain-collection-load) | Load path-backed items into memory, in place.
[`map`](#data-brain-collection-map) | Apply an arbitrary ``fn(BrainData) -> BrainData`` to each item in parallel.
[`max`](#data-brain-collection-max) | Voxelwise maximum across subjects as a single `BrainData`.
[`mean`](#data-brain-collection-mean) | Voxelwise mean across subjects as a single `BrainData`.
[`median`](#data-brain-collection-median) | Voxelwise median across subjects as a single `BrainData`.
[`memory_estimate`](#data-brain-collection-memory-estimate) | Human-readable RAM estimate if every item were loaded into memory.
[`min`](#data-brain-collection-min) | Voxelwise minimum across subjects as a single `BrainData`.
[`permutation_test`](#data-brain-collection-permutation-test) | One-sample sign-flipping permutation test across subjects.
[`permutation_test2`](#data-brain-collection-permutation-test2) | Two-sample permutation test between this collection and ``other``.
[`predict`](#data-brain-collection-predict) | Per-subject decoding (``y``) or predict-after-fit (``X_new``).
[`predict_group`](#data-brain-collection-predict-group) | Group MVPA: subjects as samples → one model → ``Predict``.
[`read`](#data-brain-collection-read) | Read a collection previously saved by `write`.
[`resample`](#data-brain-collection-resample) | Resample every subject's image to a target space in parallel.
[`smooth`](#data-brain-collection-smooth) | Spatially smooth every subject's image in parallel (delegates to `BrainData.smooth`).
[`standardize`](#data-brain-collection-standardize) | Standardize every subject's image in parallel (delegates to `BrainData.standardize`).
[`std`](#data-brain-collection-std) | Voxelwise standard deviation across subjects as a single `BrainData`.
[`steps`](#data-brain-collection-steps) | Cache subdirectories of the steps that produced this collection's items, oldest to newest.
[`sum`](#data-brain-collection-sum) | Voxelwise sum across subjects as a single `BrainData`.
[`threshold`](#data-brain-collection-threshold) | Threshold every subject's image in parallel (delegates to `BrainData.threshold`).
[`transform_designs`](#data-brain-collection-transform-designs) | Map ``fn(dm) -> DesignMatrix`` over each paired design.
[`ttest`](#data-brain-collection-ttest) | One-sample t-test across subjects (delegates to `inference.ttest`).
[`ttest2`](#data-brain-collection-ttest2) | Two-sample t-test between this collection and ``other`` (subject-level).
[`unload`](#data-brain-collection-unload) | Drop in-memory data for items that have a backing path, in place.
[`var`](#data-brain-collection-var) | Voxelwise variance across subjects as a single `BrainData`.
[`write`](#data-brain-collection-write) | Write a clean, portable copy of the collection outside the cache root.

## Methods

(data-brain-collection-align)=
### `align`

```python
align(*, method: str = 'procrustes', spatial_scale: str = 'searchlight', radius_mm: float = 10.0, roi_mask: nib.Nifti1Image | None = None, n_features: int | None = None, n_iter: int = 3, device: str = 'cpu', return_model: bool = False, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto')
```

Functionally align subjects into a common space via `LocalAlignment`.

Loads every subject into memory — the aligner needs all of them at
once.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>str</code> | Alignment solver (e.g. ``'procrustes'``). | <code>'procrustes'</code>
`spatial_scale` | <code>str</code> | Alignment spatial scale — ``'searchlight'`` (default, overlapping spheres) or ``'roi'`` (non-overlapping parcels). Whole-brain alignment is not supported at the collection level. | <code>'searchlight'</code>
`radius_mm` | <code>float</code> | Searchlight sphere radius in mm (``spatial_scale='searchlight'``). | <code>10.0</code>
`roi_mask` | <code>Nifti1Image \| None</code> | Parcellation/ROI mask (used when ``spatial_scale='roi'``). | <code>None</code>
`n_features` | <code>int \| None</code> | Optional target feature count for the common space. | <code>None</code>
`n_iter` | <code>int</code> | LocalAlignment solver iteration count (not a permutation count). | <code>3</code>
`device` | <code>str</code> | Backend selector (``'cpu'``/``'gpu'``). | <code>'cpu'</code>
`return_model` | <code>bool</code> | If True, also return the fitted `LocalAlignment`. | <code>False</code>
`n_jobs` | <code>int</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>Literal['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection) \| tuple[[BrainCollection](#page-data-brain-collection), [LocalAlignment](#tasks-alignment-localalignment)]</code> | A new     collection of aligned data, or a ``(collection, model)`` tuple     when ``return_model=True``.

(data-brain-collection-anova)=
### `anova`

```python
anova(groups: str | list | np.ndarray) -> dict
```

One-way ANOVA across subjects grouped by ``groups``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`groups` | <code>str \| list \| ndarray</code> | A metadata column name, or a list/ndarray of length ``n_subjects`` giving each subject's group label. | *required*

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Dict with ``{'F', 'p'}`` `BrainData` maps plus ``df_between`` and     ``df_within`` degrees of freedom.

(data-brain-collection-apply)=
### `apply`

```python
apply(op: str, *args: str, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto', **kwargs: Literal['auto', True, False]) -> BrainCollection
```

Call ``BrainData.<op>(*args, **kwargs)`` on every item in parallel.

The generic form of the per-subject methods (`smooth`, `standardize`,
...) — use it for any `BrainData` method that returns a `BrainData`
and has no dedicated wrapper here. The method name is passed as ``op``
rather than ``method`` because several `BrainData` methods take a
``method=`` keyword of their own (`standardize`, `detrend`, ...).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`op` | <code>str</code> | Name of the `BrainData` method to call. | *required*
`*args` | <code>tuple</code> | Positional arguments forwarded to the method. | <code>()</code>
`n_jobs` | <code>int</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>Literal['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>
`**kwargs` | <code>dict</code> | Keyword arguments forwarded to the method. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | A new collection of the returned items.

(data-brain-collection-cleanup)=
### `cleanup`

```python
cleanup() -> None
```

Remove ``cache_root`` and invalidate every clone derived from ``self``.

Idempotent — calling twice is a no-op. Path-backed items in any
clone become unloadable after this; use ``bc.write(...)`` first to
materialize a portable copy if needed.

(data-brain-collection-cleanup-all)=
### `cleanup_all`

```python
cleanup_all(directory: Path | str = '.') -> None
```

Remove every ``.nltools_cache/{run_id}/`` under ``directory``.

A wide brush — this also removes caches belonging to other live
collections created in the same directory. Prefer `cleanup` on the
collection you are done with.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`directory` | <code>Path \| str</code> | Directory whose ``.nltools_cache`` to clear. | <code>'.'</code>

(data-brain-collection-compute-contrasts)=
### `compute_contrasts`

```python
compute_contrasts(contrasts: str | list[str] | dict[str, np.ndarray], *, statistic: str = 'beta', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection | dict[str, BrainCollection] | dict[str, dict[str, BrainCollection]]
```

Compute per-subject contrast maps from fit-bundle items (the output of `fit`).

Each per-subject NIfTI gets a JSON sidecar recording its lineage
(``step_id``, ``parent_step_id``, ``op``, ``kwargs``,
``nltools_version``).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`contrasts` | <code>str \| list[str] \| dict[str, ndarray]</code> | A contrast expression over regressor names (``'A - B'``, ``'2*A - B'``), a list of them, or a dict mapping contrast names to expressions or weight vectors. | *required*
`statistic` | <code>str</code> | Which map to return — ``'beta'``, ``'t'``, ``'z'``, ``'p'``, ``'se'``, or ``'all'`` for every one. | <code>'beta'</code>
`n_jobs` | <code>int</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>Literal['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection) \| dict[str, [BrainCollection](#page-data-brain-collection)] \| dict[str, dict[str, [BrainCollection](#page-data-brain-collection)]]</code> | A `BrainCollection` for one contrast and one statistic; a dict     keyed by contrast name for several contrasts and one statistic;     a dict keyed by statistic for one contrast with     ``statistic='all'``; and a nested ``{name: {stat: collection}}``     dict for several contrasts with ``statistic='all'``.

(data-brain-collection-concat)=
### `concat`

```python
concat() -> BrainData
```

Stack all subject maps into a single `BrainData` (subjects as rows).

(data-brain-collection-detrend)=
### `detrend`

```python
detrend(*, method: str = 'linear', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Detrend every subject's image in parallel (delegates to `BrainData.detrend`).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>str</code> | Detrending method (``'linear'`` or ``'constant'``). | <code>'linear'</code>
`n_jobs` | <code>int</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>Literal['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | A new collection of detrended items.

(data-brain-collection-filter)=
### `filter`

```python
filter(predicate: Callable[[Any], Any] | list | np.ndarray | pl.Series | pd.Series) -> BrainCollection
```

Filter to a subset by predicate, polars expression, or boolean array.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`predicate` | <code>Callable \| Expr \| list \| ndarray \| Series \| Series</code> | A callable ``fn(BrainData) -> bool`` evaluated per item (loads each item), a polars expression over ``metadata``, or a boolean array-like of length ``n_subjects``. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | The matching items, sharing this collection's     cache root.

(data-brain-collection-fit)=
### `fit`

```python
fit(model: str = 'glm', X: DesignMatrix | list | Callable | None = None, *, scale: bool | str = 'auto', standardize: str | None = 'auto', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto', **model_kwargs: Literal['auto', True, False]) -> BrainCollection
```

Fit a GLM or ridge model to every subject in parallel (delegates to `BrainData.fit`).

Each item becomes an HDF5 *fit bundle* holding the fitted arrays, the
design, and lineage attributes. Feed the result to `compute_contrasts`
(GLM or ridge) or `predict(X_new=...)` (ridge).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model` | <code>str</code> | ``'glm'`` or ``'ridge'``. | <code>'glm'</code>
`X` | <code>[DesignMatrix](#page-data-design-matrix) \| list \| Callable \| None</code> | The design. ``None`` uses each subject's paired design from `designs` (all must be set); a single `DesignMatrix` is shared across subjects; a list gives one design per subject (length ``n_subjects``); a callable is invoked per subject as ``fn(ctx) -> DesignMatrix``, where ``ctx`` exposes ``bd`` (the loaded `BrainData`), ``dm`` (the paired design or ``None``), ``confounds``, ``sample_mask``, ``metadata`` (that subject's row), and the BIDS entities ``subject``, ``session``, ``run``, ``task``, ``TR``, ``bold_path``, ``events_path``, ``confounds_path``. | <code>None</code>
`scale` | <code>bool \| str</code> | Percent-signal-change scaling before fitting; ``'auto'`` resolves to ``False`` for both models. | <code>'auto'</code>
`standardize` | <code>str \| None</code> | Per-voxel standardization before fitting; ``'auto'`` resolves to ``'zscore'`` for ridge and ``None`` for GLM. | <code>'auto'</code>
`n_jobs` | <code>int</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>Literal['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>
`**model_kwargs` | <code>dict</code> | Forwarded to `BrainData.fit` (e.g. ``alpha``, ``cv`` for ridge). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | A new collection whose items are per-subject fit     bundles.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If ``model`` is unknown, or ``X`` is ``None`` while some items have no paired design.
<code>NotImplementedError</code> | For ``model='glm'`` with a ``noise_model`` other than ``'ols'`` — AR models are available per subject via `BrainData.fit`.

(data-brain-collection-from-bids)=
### `from_bids`

```python
from_bids(root: Path | str | Any, *, mask: nib.Nifti1Image | Path | str, task: str | None = None, space: str | None = None, sub_labels: list[str] | None = None, img_filters: list[tuple[str, str]] | None = None, derivatives_folder: str = 'derivatives', pair_events: bool = True, confounds_strategy: str | tuple[str, ...] | None = None, confounds_kwargs: dict | None = None, TR: float | str = 'infer', cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Build a collection from a BIDS dataset, pairing each BOLD run with its events and confounds.

Discovery goes through nilearn's ``first_level_from_bids``; each run's
``events.tsv`` becomes an unconvolved `DesignMatrix` (add HRF
convolution, drift, and confound columns yourself with
`transform_designs`). Metadata gets one row per run with ``subject``,
``session``, ``run``, ``task``, ``space``, ``bold_path``, and ``TR``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`root` | <code>Path \| str</code> | BIDS dataset root. | *required*
`mask` | <code>Nifti1Image \| Path \| str</code> | Mask shared by every item. | *required*
`task` | <code>str \| None</code> | BIDS task label. ``None`` discovers BOLD files without pairing events (designs are all ``None``). | <code>None</code>
`space` | <code>str \| None</code> | Only keep images in this ``space-`` entity. | <code>None</code>
`sub_labels` | <code>list[str] \| None</code> | Restrict to these subject labels. | <code>None</code>
`img_filters` | <code>list[tuple[str, str]] \| None</code> | Extra BIDS ``(entity, value)`` filters on image filenames. | <code>None</code>
`derivatives_folder` | <code>str</code> | Preprocessed-derivatives folder name under ``root``. | <code>'derivatives'</code>
`pair_events` | <code>bool</code> | If True (and ``task`` is set), build a `DesignMatrix` from each run's ``events.tsv``. Runs without one get ``None`` and a warning. | <code>True</code>
`confounds_strategy` | <code>str \| tuple[str, ...] \| None</code> | fMRIPrep confounds strategy forwarded to nilearn's ``load_confounds``; requires fMRIPrep-style derivatives. | <code>None</code>
`confounds_kwargs` | <code>dict \| None</code> | Extra keyword arguments for ``load_confounds``. | <code>None</code>
`TR` | <code>float \| str</code> | Repetition time in seconds, or ``'infer'`` to read it from the BIDS sidecars. | <code>'infer'</code>
`cache_dir` | <code>Path \| str \| None</code> | Cache location; see the class constructor. | <code>'./.nltools_cache'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | A lazy, path-backed collection.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If no BOLD files match, or ``TR='infer'`` finds no repetition time for a run.
<code>ImportError</code> | If nilearn/pybids are unavailable, or ``confounds_strategy`` is set without fMRIPrep support.

(data-brain-collection-from-glob)=
### `from_glob`

```python
from_glob(pattern: str, *, mask: nib.Nifti1Image | Path | str, design_pattern: str | None = None, pattern_groups: dict[str, int] | str | None = None, sort: bool = True, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Build a collection by glob-matching brain images (and optional designs).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`pattern` | <code>str</code> | Glob pattern matching the per-subject brain image files. | *required*
`mask` | <code>Nifti1Image \| Path \| str</code> | Shared mask image, path, or nltools template name. | *required*
`design_pattern` | <code>str \| None</code> | Optional glob matching per-subject design files, paired positionally with the brain images. | <code>None</code>
`pattern_groups` | <code>dict[str, int] \| str \| None</code> | Regex capture-group spec used to extract metadata (e.g. subject/run) from each matched path. | <code>None</code>
`sort` | <code>bool</code> | If True, sort matched paths before pairing (stable ordering). | <code>True</code>
`cache_dir` | <code>Path \| str \| None</code> | Cache-directory precedence: explicit arg → ``NLTOOLS_CACHE_DIR`` env → ``./.nltools_cache``; ``None`` for a temp dir. | <code>'./.nltools_cache'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | A lazy, path-backed `BrainCollection`.

(data-brain-collection-from-paths)=
### `from_paths`

```python
from_paths(brain_paths: list, *, mask: nib.Nifti1Image | Path | str, design_paths: list | None = None, metadata: pl.DataFrame | pd.DataFrame | dict | None = None, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Build a collection from explicit lists of brain (and design) paths.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_paths` | <code>list</code> | Per-subject brain image paths. | *required*
`mask` | <code>Nifti1Image \| Path \| str</code> | Shared mask image, path, or nltools template name. | *required*
`design_paths` | <code>list \| None</code> | Optional per-subject design paths, aligned positionally with ``brain_paths`` (length must match, ``None`` entries allowed). | <code>None</code>
`metadata` | <code>DataFrame \| DataFrame \| dict \| None</code> | Optional per-subject metadata (polars/pandas DataFrame or dict-of-columns), one row per path. | <code>None</code>
`cache_dir` | <code>Path \| str \| None</code> | Cache-directory precedence: explicit arg → ``NLTOOLS_CACHE_DIR`` env → ``./.nltools_cache``; ``None`` for a temp dir. | <code>'./.nltools_cache'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | A lazy, path-backed `BrainCollection`.

(data-brain-collection-isc)=
### `isc`

```python
isc(*, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, summary: str = 'median') -> dict
```

Inter-subject correlation (ISC) across the time dimension.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>str</code> | ``'loo'`` (leave-one-out template) or ``'pairwise'`` (all subject pairs). | <code>'loo'</code>
`roi_mask` | <code>Nifti1Image \| Path \| str \| None</code> | Optional ROI/atlas mask restricting the computation to those voxels. The returned maps carry the ROI mask. If None, ISC is computed across the collection's whole-brain mask. | <code>None</code>
`summary` | <code>str</code> | Aggregation across subjects/pairs (e.g. ``'median'``). | <code>'median'</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Dict ``{'isc', 'per_subject'}`` for ``method='loo'`` or     ``{'isc', 'pairs'}`` for ``method='pairwise'`` (``'isc'`` is a     `BrainData` map).

(data-brain-collection-isc-test)=
### `isc_test`

```python
isc_test(*, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, n_samples: int = 5000, summary: str = 'median', tail: int | str = 2, random_state: int | None = None) -> dict
```

Bootstrap inference on ISC (per-voxel p-values).

Resamples subjects with replacement, recomputes ISC each draw, and
derives a per-voxel p-value from the null centered at 0.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>str</code> | ``'loo'`` or ``'pairwise'`` (matches `isc`). | <code>'loo'</code>
`roi_mask` | <code>Nifti1Image \| Path \| str \| None</code> | Optional ROI/atlas mask restricting the computation to those voxels. The returned maps carry the ROI mask. If None, ISC is computed across the collection's whole-brain mask. | <code>None</code>
`n_samples` | <code>int</code> | Number of bootstrap resamples. | <code>5000</code>
`summary` | <code>str</code> | Aggregation across subjects/pairs (e.g. ``'median'``). | <code>'median'</code>
`tail` | <code>int \| str</code> | `2`/`'two'` (two-tailed, default) or `1`/`'one'` (one-tailed: ISC > 0). | <code>2</code>
`random_state` | <code>int \| None</code> | Seed for the bootstrap RNG. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Dict ``{'isc', 'p', 'null_dist'}`` (``'isc'`` and ``'p'`` are     `BrainData` maps).

(data-brain-collection-iter-pairs)=
### `iter_pairs`

```python
iter_pairs() -> Iterator[tuple]
```

Yield ``(BrainData, DesignMatrix | None)`` pairs.

(data-brain-collection-load)=
### `load`

```python
load(indices: list[int] | None = None) -> BrainCollection
```

Load path-backed items into memory, in place.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`indices` | <code>list[int] \| None</code> | Items to load; ``None`` loads all. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | ``self``, for chaining.

(data-brain-collection-map)=
### `map`

```python
map(fn: Callable, *, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Apply an arbitrary ``fn(BrainData) -> BrainData`` to each item in parallel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fn` | <code>Callable</code> | Function taking one loaded `BrainData` and returning a `BrainData`. | *required*
`n_jobs` | <code>int</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>Literal['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | A new collection of the returned items.

(data-brain-collection-max)=
### `max`

```python
max() -> BrainData
```

Voxelwise maximum across subjects as a single `BrainData`.

(data-brain-collection-mean)=
### `mean`

```python
mean() -> BrainData
```

Voxelwise mean across subjects as a single `BrainData`.

(data-brain-collection-median)=
### `median`

```python
median() -> BrainData
```

Voxelwise median across subjects as a single `BrainData`.

(data-brain-collection-memory-estimate)=
### `memory_estimate`

```python
memory_estimate() -> str
```

Human-readable RAM estimate if every item were loaded into memory.

The per-item shape is read from the first in-memory item, or from the
first item loaded on demand when none is in memory.

**Returns:**

Type | Description
---- | -----------
<code>str</code> | ``n_subjects``, the per-item shape, and the estimated float32     total in human-readable units.

(data-brain-collection-min)=
### `min`

```python
min() -> BrainData
```

Voxelwise minimum across subjects as a single `BrainData`.

(data-brain-collection-permutation-test)=
### `permutation_test`

```python
permutation_test(*, n_permute: int = 5000, tail: int | str = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None, progress_bar: bool = False) -> dict
```

One-sample sign-flipping permutation test across subjects.

Delegates to the inference engine's `one_sample_permutation_test`
over the stacked subject data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_permute` | <code>int</code> | Number of sign-flip permutations. | <code>5000</code>
`tail` | <code>int \| str</code> | 1 for one-tailed, 2 for two-tailed. | <code>2</code>
`device` | <code>str</code> | Execution backend — ``None`` (single-threaded numpy), ``'cpu'`` (joblib parallel), or ``'gpu'`` (PyTorch). | <code>'cpu'</code>
`return_null` | <code>bool</code> | If True, include the null distribution in the result. | <code>False</code>
`n_jobs` | <code>int</code> | CPU workers when ``device='cpu'`` (-1 = all cores). | <code>-1</code>
`random_state` | <code>int \| None</code> | Seed for the sign-flip RNG. | <code>None</code>
`progress_bar` | <code>bool</code> | Whether to display a progress bar. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Dict ``{'mean', 'p'}`` of `BrainData` maps, plus     ``'null_dist'`` when ``return_null=True``.

(data-brain-collection-permutation-test2)=
### `permutation_test2`

```python
permutation_test2(other: BrainCollection, *, n_permute: int = 5000, tail: int | str = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Two-sample permutation test between this collection and ``other``.

Uses random label shuffling of the pooled subjects, delegating to the
inference engine's `two_sample_permutation_test`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` | <code>[BrainCollection](#page-data-brain-collection)</code> | The second collection to compare against. | *required*
`n_permute` | <code>int</code> | Number of label-shuffle permutations. | <code>5000</code>
`tail` | <code>int \| str</code> | 1 for one-tailed, 2 for two-tailed. | <code>2</code>
`device` | <code>str</code> | Execution backend — ``None`` (single-threaded numpy), ``'cpu'`` (joblib parallel), or ``'gpu'`` (PyTorch). | <code>'cpu'</code>
`return_null` | <code>bool</code> | If True, include the null distribution in the result. | <code>False</code>
`n_jobs` | <code>int</code> | CPU workers when ``device='cpu'`` (-1 = all cores). | <code>-1</code>
`random_state` | <code>int \| None</code> | Seed for the shuffling RNG. | <code>None</code>
`progress_bar` | <code>bool</code> | Whether to display a progress bar. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Dict ``{'mean', 'p'}`` of `BrainData` maps (``mean`` is the group     difference), plus ``'null_dist'`` when ``return_null=True``.

(data-brain-collection-predict)=
### `predict`

```python
predict(y: str | list | np.ndarray | None = None, *, X_new: np.ndarray | None = None, spatial_scale: str = 'whole_brain', model: str = 'svm', cv: int | str = 5, groups: str | list | np.ndarray | None = None, roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float = 10.0, scoring: str = 'auto', standardize: bool = True, n_jobs: int = -1, random_state: int | None = None, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto')
```

Per-subject decoding (``y``) or predict-after-fit (``X_new``).

The per-subject counterpart to every other method on this class —
one operation per subject, no cross-subject mixing. (For **group
MVPA** — subjects as samples, one model across the collection — use
`predict_group`.) Dispatched by which argument is provided:

1. **Per-subject decoding** (``y``, or omitted with stored labels):
   maps `BrainData.predict` over subjects — one model per subject,
   cross-validated within that subject's own rows — and returns a
   `PredictCollection` carrying the collection's metadata. Stack the
   per-subject decoder maps for second-level inference via
   ``result.weight_maps``.
2. **Predict-after-fit** (``X_new``): map each subject's fitted
   ridge model over a new design matrix, returning a
   ``BrainCollection`` of predicted maps. Requires ridge fit-bundle
   items (``.fit(model='ridge', cache=True)``).

Labels travel with the data: with ``y`` omitted, each subject
decodes its own single-column ``.Y``; ``y='name'`` picks a column of
each subject's ``.Y``, and ``groups='name'`` does the same for a
within-subject grouping variable (e.g. run). Alternatively pass one
shared label array (applied to every subject) or a list of arrays
(one per subject, in collection order).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` | <code>str \| list \| ndarray \| None</code> | Per-subject decoding targets — ``None`` (each subject's single-column ``.Y``), a ``.Y`` column name, one shared array, or a list of per-subject arrays. | <code>None</code>
`X_new` | <code>ndarray \| None</code> | New design matrix for predict-after-fit (mode 2). | <code>None</code>
`spatial_scale` | <code>str</code> | One of ``'whole_brain'``, ``'roi'``, or ``'searchlight'``. | <code>'whole_brain'</code>
`model` | <code>str</code> | Model name or sklearn estimator (see ``BrainData.predict``). | <code>'svm'</code>
`cv` | <code>int \| str</code> | Within-subject CV — an int fold count (default 5, honoring ``groups`` via the Group variants), ``'loo'``, ``'logo'`` (with ``groups``, e.g. leave-one-run-out), or an sklearn splitter. | <code>5</code>
`groups` | <code>str \| list \| ndarray \| None</code> | Within-subject grouping variable — a ``.Y`` column name, one shared array, or a list of per-subject arrays. | <code>None</code>
`roi_mask` | <code>Nifti1Image \| Path \| str \| None</code> | Atlas image for ``spatial_scale='roi'``. | <code>None</code>
`radius_mm` | <code>float</code> | Searchlight radius. | <code>10.0</code>
`scoring` | <code>str</code> | ``'auto'`` → accuracy (classifier) / r2 (regressor). | <code>'auto'</code>
`standardize` | <code>bool</code> | Standardize features within each CV fold. | <code>True</code>
`n_jobs` | <code>int</code> | CPU workers (subject-level; each subject decodes with ``n_jobs=1`` to avoid nested parallelism). | <code>-1</code>
`random_state` | <code>int \| None</code> | Seed for shuffled int-``cv`` folds. | <code>None</code>
`progress_bar` | <code>bool</code> | Whether to display a progress bar. | <code>False</code>
`cache` | <code>Literal['auto', True, False]</code> | ``'auto'`` (cache when the source is path-backed), ``True``, or ``False``. Caching writes one predict bundle (``.h5``) per subject holding the result's ingredients — never a pickled estimator, so cached results have ``estimator=None``. | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[PredictCollection](#data-results-predictcollection) \| [BrainCollection](#page-data-brain-collection)</code> | `PredictCollection` (mode 1) or     `BrainCollection` (mode 2).

(data-brain-collection-predict-group)=
### `predict_group`

```python
predict_group(y: str | list | np.ndarray, *, spatial_scale: str = 'whole_brain', model: str = 'svm', cv: int | str = 'logo', groups: str | np.ndarray | None = None, roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float = 10.0, scoring: str = 'auto', standardize: bool = True, n_permute: int = 0, n_jobs: int = -1, random_state: int | None = None, progress_bar: bool = False)
```

Group MVPA: subjects as samples → one model → ``Predict``.

Stacks the collection into a ``(n_subjects, n_voxels)`` matrix and
trains a **single** model with subjects as samples (unlike the
per-subject methods, this deliberately collapses across subjects).
Requires single-map-per-subject items — run
``compute_contrasts(...)`` first for GLM/ridge bundles.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` | <code>str \| list \| ndarray</code> | Labels/targets, one per subject — an array/list, or the name of a metadata column. | *required*
`spatial_scale` | <code>str</code> | One of ``'whole_brain'``, ``'roi'``, or ``'searchlight'``. | <code>'whole_brain'</code>
`model` | <code>str</code> | Model name (see ``BrainData.predict``). | <code>'svm'</code>
`cv` | <code>int \| str</code> | ``'logo'`` (leave-one-group-out, default — with the default ``groups`` this is leave-one-subject-out), ``'loo'`` (leave-one-out), an int fold count, or an sklearn splitter. An int spec **honors** ``groups``: it resolves to `StratifiedGroupKFold` (classifiers) / `GroupKFold` (regressors) so a group never straddles a train/test boundary. | <code>'logo'</code>
`groups` | <code>str \| ndarray \| None</code> | Group labels, or a metadata column name. Defaults to one group per subject; pass ``groups='run'`` (or any metadata column) for e.g. leave-one-run-out under ``cv='logo'``. | <code>None</code>
`roi_mask` | <code>Nifti1Image \| Path \| str \| None</code> | Restrict to an ROI. | <code>None</code>
`radius_mm` | <code>float</code> | Searchlight radius. | <code>10.0</code>
`scoring` | <code>str</code> | ``'auto'`` → accuracy (classifier) / r2 (regressor). | <code>'auto'</code>
`standardize` | <code>bool</code> | Standardize features within each CV fold. | <code>True</code>
`n_permute` | <code>int</code> | If ``> 0``, also build a label-permutation null of the CV score — shuffle ``y`` and re-score the identical CV (scoring only; no refit/weight-map work) — attached as ``permutation_scores`` and ``permutation_pvalue`` (Phipson-Smyth upper-tail). Forms by ``spatial_scale``: whole_brain → null ``(n_permute,)``, p float; roi → null ``(n_permute, n_rois)``, p ``(n_rois,)``; searchlight → null ``(n_permute, n_voxels)``, p a `BrainData` map (NaN where the observed accuracy map is NaN). Default 0 (no null). | <code>0</code>
`n_jobs` | <code>int</code> | CPU workers. | <code>-1</code>
`random_state` | <code>int \| None</code> | Seed for the permutation-null label shuffling. | <code>None</code>
`progress_bar` | <code>bool</code> | Whether to display a progress bar. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[Predict](#data-results-predict)</code> | Result with CV attributes, plus the permutation-null fields     when ``n_permute > 0``.

(data-brain-collection-read)=
### `read`

```python
read(directory: Path | str, *, mask: nib.Nifti1Image | Path | str, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Read a collection previously saved by `write`.

Discovers ``image_*.nii*`` files in ``directory`` and pairs them with
the rows of ``metadata.csv`` when present. Only the portable layout
written by `write` is readable this way — not the cache directories.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`directory` | <code>Path \| str</code> | Directory produced by `write`. | *required*
`mask` | <code>Nifti1Image \| Path \| str</code> | Mask shared by every item. | *required*
`cache_dir` | <code>Path \| str \| None</code> | Cache location; see the class constructor. | <code>'./.nltools_cache'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | A lazy, path-backed collection.

(data-brain-collection-resample)=
### `resample`

```python
resample(target, *, interpolation: str = 'continuous', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Resample every subject's image to a target space in parallel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` | <code>Nifti1Image \| Path \| str</code> | Target image whose grid and affine every item is resampled onto. | *required*
`interpolation` | <code>str</code> | Interpolation method (``'continuous'``, ``'linear'``, ``'nearest'``). | <code>'continuous'</code>
`n_jobs` | <code>int</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>Literal['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | A new collection of resampled items.

(data-brain-collection-smooth)=
### `smooth`

```python
smooth(fwhm: float, *, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Spatially smooth every subject's image in parallel (delegates to `BrainData.smooth`).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fwhm` | <code>float</code> | Gaussian kernel full-width at half-maximum, in mm. | *required*
`n_jobs` | <code>int</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>Literal['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | A new collection of smoothed items.

(data-brain-collection-standardize)=
### `standardize`

```python
standardize(*, axis: int = 0, method: str = 'center', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Standardize every subject's image in parallel (delegates to `BrainData.standardize`).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>int</code> | Axis along which to standardize (0 = across observations). | <code>0</code>
`method` | <code>str</code> | Standardization variant (e.g. ``'center'``, ``'zscore'``). | <code>'center'</code>
`n_jobs` | <code>int</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>Literal['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | A new `BrainCollection` of standardized items.

(data-brain-collection-std)=
### `std`

```python
std() -> BrainData
```

Voxelwise standard deviation across subjects as a single `BrainData`.

(data-brain-collection-steps)=
### `steps`

```python
steps() -> list[Path]
```

Cache subdirectories of the steps that produced this collection's items, oldest to newest.

One entry per upstream cached operation. Empty when the collection was
constructed directly or no ancestor wrote to disk.

**Returns:**

Type | Description
---- | -----------
<code>list[Path]</code> | Step directories under `cache_root`.

(data-brain-collection-sum)=
### `sum`

```python
sum() -> BrainData
```

Voxelwise sum across subjects as a single `BrainData`.

(data-brain-collection-threshold)=
### `threshold`

```python
threshold(*, lower: float | None = None, upper: float | None = None, binarize: bool = False, coerce_nan: bool = True, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Threshold every subject's image in parallel (delegates to `BrainData.threshold`).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`lower` | <code>float \| None</code> | Values below this are zeroed (or set NaN); ``None`` disables. | <code>None</code>
`upper` | <code>float \| None</code> | Values above this are zeroed (or set NaN); ``None`` disables. | <code>None</code>
`binarize` | <code>bool</code> | If True, set surviving voxels to 1. | <code>False</code>
`coerce_nan` | <code>bool</code> | If True, coerce thresholded-out voxels to NaN instead of 0. | <code>True</code>
`n_jobs` | <code>int</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>Literal['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | A new `BrainCollection` of thresholded items.

(data-brain-collection-transform-designs)=
### `transform_designs`

```python
transform_designs(fn: Callable, *, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Map ``fn(dm) -> DesignMatrix`` over each paired design.

Items with no paired design are skipped (kept as ``None``). Runs in
the parent process — designs are small — so ``n_jobs``,
``progress_bar``, and ``cache`` are accepted for consistency with the
other per-subject methods but ignored.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fn` | <code>Callable</code> | Function taking one `DesignMatrix` and returning the transformed `DesignMatrix`. | *required*
`n_jobs` | <code>int</code> | Ignored. | <code>-1</code>
`progress_bar` | <code>bool</code> | Ignored. | <code>False</code>
`cache` | <code>Literal['auto', True, False]</code> | Ignored. | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | A new collection with the same items and the     transformed designs.

(data-brain-collection-ttest)=
### `ttest`

```python
ttest(*, popmean: float = 0.0, tail: int | str = 2) -> dict
```

One-sample t-test across subjects (delegates to `inference.ttest`).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`popmean` | <code>float</code> | Null-hypothesis population mean to test against. | <code>0.0</code>
`tail` | <code>int \| str</code> | `2`/`'two'` (two-tailed, default) or `1`/`'one'` (one-tailed: mean > popmean; negate the data for the other direction). | <code>2</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Dict ``{'mean', 't', 'z', 'p'}`` of `BrainData` maps.

(data-brain-collection-ttest2)=
### `ttest2`

```python
ttest2(other: BrainCollection, *, equal_var: bool = True, tail: int | str = 2) -> dict
```

Two-sample t-test between this collection and ``other`` (subject-level).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` | <code>[BrainCollection](#page-data-brain-collection)</code> | The second collection to compare against. | *required*
`equal_var` | <code>bool</code> | If True, pooled-variance t-test; if False, Welch's test. | <code>True</code>
`tail` | <code>int \| str</code> | `2`/`'two'` (two-tailed, default) or `1`/`'one'` (one-tailed: self > other; swap the operands for the other direction). | <code>2</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Dict ``{'mean', 't', 'z', 'p'}`` of `BrainData` maps (``mean`` is the     group difference).

(data-brain-collection-unload)=
### `unload`

```python
unload(indices: list[int] | None = None) -> BrainCollection
```

Drop in-memory data for items that have a backing path, in place.

Items without a backing path (constructed from in-memory `BrainData`
or produced with ``cache=False``) are left untouched, since dropping
them would lose the data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`indices` | <code>list[int] \| None</code> | Items to unload; ``None`` unloads all. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection)</code> | ``self``, for chaining.

(data-brain-collection-var)=
### `var`

```python
var() -> BrainData
```

Voxelwise variance across subjects as a single `BrainData`.

(data-brain-collection-write)=
### `write`

```python
write(directory: Path | str, *, pattern: str = 'image_{i:04d}.nii.gz', metadata_file: str | None = 'metadata.csv') -> list[Path]
```

Write a clean, portable copy of the collection outside the cache root.

Inverse of `BrainCollection.read`. Writes one NIfTI per item plus an
optional metadata CSV, skipping the internal cache layout so the result
is shareable/archival.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`directory` | <code>Path \| str</code> | Output directory (created if missing). | *required*
`pattern` | <code>str</code> | Filename template per item, formatted with ``i`` (item index). | <code>'image_{i:04d}.nii.gz'</code>
`metadata_file` | <code>str \| None</code> | CSV filename for the metadata table, or ``None`` to skip. | <code>'metadata.csv'</code>

**Returns:**

Type | Description
---- | -----------
<code>list[Path]</code> | List of written NIfTI paths, in item order.
