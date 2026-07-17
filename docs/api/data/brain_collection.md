(data-brain-collection-braincollection)=
## `BrainCollection`

```python
BrainCollection(brains: list, *, mask: nib.Nifti1Image | Path | str, designs: list | None = None, metadata: pl.DataFrame | pd.DataFrame | dict | None = None, lazy: bool = True, cache_dir: Path | str | None = './.nltools_cache') -> None
```

Parallel, lazy iterator of ``BrainData`` whose API mirrors ``BrainData``.

Constructed via ``__init__`` (explicit lists) or one of the classmethod
factories (``from_bids``, ``from_glob``, ``from_paths``, ``read``).

See ``SPEC.md`` §"Public API" for the full contract; key invariants:
  - Per-subject ops route through ``execution._apply`` and return a
    lightweight clone via ``self._clone(...)`` over the same cache root.
  - Path-backed by default after parallel ops; ``cache='auto'`` follows
    source state. ``cache=`` is only accepted on collection-returning ops.
  - ``load`` / ``unload`` are the only methods that mutate ``self``.

Internal state (mutable list at top level; per-item slots are parallel):

  _items          list[BrainData | Path]        per-item brain data
  _mask           nib.Nifti1Image               shared mask (by reference)
  _designs        list[DesignMatrix | Path | None]
  _confounds      list[pd.DataFrame | None]
  _sample_masks   list[np.ndarray | None]
  _metadata       pl.DataFrame                  simple-typed columns only
  _cache_root     Path | None                   shared by clones
  _step_id        str | None                    this collection's step id
  _parent_step_id str | None                    upstream step id (lineage)
  _step_dirs      list[Path]                    lineage of step subdirs
                                                that produced these items
  _source_paths   list[Path | None]             per-item backing path
                                                (None for in-memory only)

**Methods:**

Name | Description
---- | -----------
[`align`](#data-brain-collection-align) | Functionally align subjects into a common space via `LocalAlignment`.
[`anova`](#data-brain-collection-anova) | One-way ANOVA across subjects grouped by ``groups``.
[`apply`](#data-brain-collection-apply) | Call ``BrainData.<op>(*args, **kwargs)`` on every item in parallel.
[`cleanup`](#data-brain-collection-cleanup) | Remove ``cache_root`` and invalidate every clone derived from ``self``.
[`cleanup_all`](#data-brain-collection-cleanup-all) | Remove every ``.nltools_cache/{run_id}/`` under ``directory``.
[`compute_contrasts`](#data-brain-collection-compute-contrasts) | Compute per-subject contrast maps from fit-bundle items.
[`concat`](#data-brain-collection-concat) | Stack all subject maps into a single `BrainData` (subjects as rows).
[`cv`](#data-brain-collection-cv) | Build a CV pipeline for cross-subject prediction.
[`detrend`](#data-brain-collection-detrend) | Detrend every subject's image in parallel (delegates to `BrainData.detrend`).
[`filter`](#data-brain-collection-filter) | Filter to a subset by predicate, polars expression, or boolean array.
[`fit`](#data-brain-collection-fit) | Per-subject fit; returns a path-backed collection of HDF5 fit bundles.
[`from_bids`](#data-brain-collection-from-bids) | Auto-pair BOLD with events.tsv (→ ``DesignMatrix``) and confounds.tsv.
[`from_glob`](#data-brain-collection-from-glob) | Build a collection by glob-matching brain images (and optional designs).
[`from_paths`](#data-brain-collection-from-paths) | Build a collection from explicit lists of brain (and design) paths.
[`isc`](#data-brain-collection-isc) | Inter-subject correlation (ISC) across the time dimension.
[`isc_test`](#data-brain-collection-isc-test) | Bootstrap inference on ISC (per-voxel p-values).
[`iter_pairs`](#data-brain-collection-iter-pairs) | Yield ``(BrainData, DesignMatrix | None)`` pairs.
[`load`](#data-brain-collection-load) | Materialize path-backed items in place. Returns ``self`` for chaining.
[`map`](#data-brain-collection-map) | Apply an arbitrary ``fn(BrainData) -> BrainData`` to each item in parallel.
[`max`](#data-brain-collection-max) | Voxelwise maximum across subjects as a single `BrainData`.
[`mean`](#data-brain-collection-mean) | Voxelwise mean across subjects as a single `BrainData`.
[`median`](#data-brain-collection-median) | Voxelwise median across subjects as a single `BrainData`.
[`memory_estimate`](#data-brain-collection-memory-estimate) | Human-readable RAM estimate if every item were loaded into memory.
[`min`](#data-brain-collection-min) | Voxelwise minimum across subjects as a single `BrainData`.
[`permutation_test`](#data-brain-collection-permutation-test) | One-sample sign-flipping permutation test across subjects.
[`permutation_test2`](#data-brain-collection-permutation-test2) | Two-sample permutation test between this collection and ``other``.
[`predict`](#data-brain-collection-predict) | Two distinct paths, dispatched by argument:
[`read`](#data-brain-collection-read) | Inverse of ``write()``. Does not recover from cache subdirs in v0.6.0.
[`resample`](#data-brain-collection-resample) | Resample every subject's image to a target space in parallel.
[`smooth`](#data-brain-collection-smooth) | Spatially smooth every subject's image in parallel (delegates to `BrainData.smooth`).
[`standardize`](#data-brain-collection-standardize) | Standardize every subject's image in parallel (delegates to `BrainData.standardize`).
[`std`](#data-brain-collection-std) | Voxelwise standard deviation across subjects as a single `BrainData`.
[`steps`](#data-brain-collection-steps) | Step subdirs that produced this collection's items, oldest to newest.
[`sum`](#data-brain-collection-sum) | Voxelwise sum across subjects as a single `BrainData`.
[`threshold`](#data-brain-collection-threshold) | Threshold every subject's image in parallel (delegates to `BrainData.threshold`).
[`transform_designs`](#data-brain-collection-transform-designs) | Map ``fn(dm) -> DesignMatrix`` over each paired design.
[`ttest`](#data-brain-collection-ttest) | One-sample t-test across subjects (delegates to `inference.ttest`).
[`ttest2`](#data-brain-collection-ttest2) | Two-sample t-test between this collection and ``other`` (subject-level).
[`unload`](#data-brain-collection-unload) | Drop in-memory data for items with backing paths. Returns ``self``.
[`var`](#data-brain-collection-var) | Voxelwise variance across subjects as a single `BrainData`.
[`write`](#data-brain-collection-write) | Write a clean, portable copy of the collection outside the cache root.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`cache_root` | <code>[Path](#pathlib.Path)</code> | Run-scoped cache directory shared by clones. Raises if unset.
`designs` | <code>[list](#list)</code> | Per-subject paired designs (a copy of the list; ``None`` where unpaired).
`is_loaded` | <code>[list](#list)[[bool](#bool)]</code> | Per-item flag — True iff the slot holds a ``BrainData`` (not a path).
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image)</code> | Shared mask image for the collection. Raises if the mask is unset.
`metadata` | <code>[DataFrame](#polars.DataFrame)</code> | Per-subject metadata as a polars DataFrame (one row per item).
`n_subjects` | <code>[int](#int)</code> | Number of subjects (items) in the collection.
`n_voxels` | <code>[int](#int)</code> | Voxel count from the mask. Raises if mask is unset.
`shape` | <code>[tuple](#tuple)[[int](#int), [int](#int) \| None, [int](#int)]</code> | ``(n_subjects, n_obs_or_None_if_ragged, n_voxels)``.

``cache_dir`` precedence: explicit arg → ``NLTOOLS_CACHE_DIR`` env →
``./.nltools_cache``. Pass ``None`` for an auto-cleaned tempdir.
Resolved at construction and frozen on the instance.

### Methods

(data-brain-collection-align)=
#### `align`

```python
align(*, method: str = 'procrustes', spatial_scale: str = 'searchlight', radius_mm: float = 10.0, roi_mask: nib.Nifti1Image | None = None, n_features: int | None = None, n_iter: int = 3, device: str = 'cpu', return_model: bool = False, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto')
```

Functionally align subjects into a common space via `LocalAlignment`.

Materializes all subjects (algorithm constraint in v0.6.0).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Alignment solver (e.g. ``'procrustes'``). | <code>'procrustes'</code>
`spatial_scale` | <code>[str](#str)</code> | Alignment spatial scale — ``'searchlight'`` (default, overlapping spheres) or ``'roi'`` (non-overlapping parcels). Whole-brain alignment is not supported at the collection level. | <code>'searchlight'</code>
`radius_mm` | <code>[float](#float)</code> | Searchlight sphere radius in mm (``spatial_scale='searchlight'``). | <code>10.0</code>
`roi_mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| None</code> | Parcellation/ROI mask (used when ``spatial_scale='roi'``). | <code>None</code>
`n_features` | <code>[int](#int) \| None</code> | Optional target feature count for the common space. | <code>None</code>
`n_iter` | <code>[int](#int)</code> | LocalAlignment solver iteration count (not a permutation count). | <code>3</code>
`device` | <code>[str](#str)</code> | Backend selector (``'cpu'``/``'gpu'``). | <code>'cpu'</code>
`return_model` | <code>[bool](#bool)</code> | If True, also return the fitted `LocalAlignment`. | <code>False</code>
`n_jobs` | <code>[int](#int)</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>[bool](#bool)</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>[Literal](#typing.Literal)['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
 | A new `BrainCollection` of aligned data, or a
 | ``(BrainCollection, LocalAlignment)`` tuple when
 | ``return_model=True``.

(data-brain-collection-anova)=
#### `anova`

```python
anova(groups: str | list | np.ndarray) -> dict
```

One-way ANOVA across subjects grouped by ``groups``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`groups` | <code>[str](#str) \| [list](#list) \| [ndarray](#numpy.ndarray)</code> | A metadata column name, or a list/ndarray of length ``n_subjects`` giving each subject's group label. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict with ``{'F', 'p'}`` `BrainData` maps plus ``df_between`` and
<code>[dict](#dict)</code> | ``df_within`` degrees of freedom.

(data-brain-collection-apply)=
#### `apply`

```python
apply(op: str, *args: str, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto', **kwargs: Literal['auto', True, False]) -> BrainCollection
```

Call ``BrainData.<op>(*args, **kwargs)`` on every item in parallel.

All per-subject methods (``smooth``, ``standardize``, ...) reduce to
this. Centralizes the ``_apply`` plumbing and the cache-knob handling.
``op`` is named ``op`` (not ``method``) to avoid colliding with
``BrainData`` methods that themselves take a ``method=`` kwarg
(``standardize``, ``detrend``, ...).

(data-brain-collection-cleanup)=
#### `cleanup`

```python
cleanup() -> None
```

Remove ``cache_root`` and invalidate every clone derived from ``self``.

Idempotent — calling twice is a no-op. Path-backed items in any
clone become unloadable after this; use ``bc.write(...)`` first to
materialize a portable copy if needed.

(data-brain-collection-cleanup-all)=
#### `cleanup_all`

```python
cleanup_all(directory: Path | str = '.') -> None
```

Remove every ``.nltools_cache/{run_id}/`` under ``directory``.

Wide brush — can kill sibling sessions in the same cwd. Prefer
``bc.cleanup()`` for surgical removal.

(data-brain-collection-compute-contrasts)=
#### `compute_contrasts`

```python
compute_contrasts(contrasts: str | list[str] | dict[str, np.ndarray], *, statistic: str = 'beta', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection | dict[str, BrainCollection] | dict[str, dict[str, BrainCollection]]
```

Compute per-subject contrast maps from fit-bundle items.

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)] \| [dict](#dict)[[str](#str), [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]]</code> | single contrast + single ``statistic`` → ``BrainCollection``
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)] \| [dict](#dict)[[str](#str), [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]]</code> | multiple contrasts (single type)            → ``dict[str, BrainCollection]``
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)] \| [dict](#dict)[[str](#str), [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]]</code> | ``statistic='all'`` (single contrast)   → ``dict['beta'|'t'|'z'|'p'|'se', BrainCollection]``
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)] \| [dict](#dict)[[str](#str), [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]]</code> | multiple contrasts + ``statistic='all'`` → nested                                              ``dict[name, dict[stat, BrainCollection]]``

Each per-subject NIfTI gets a JSON sidecar with lineage attrs
(``step_id``, ``parent_step_id``, ``op``, ``kwargs``,
``nltools_version``).

(data-brain-collection-concat)=
#### `concat`

```python
concat() -> BrainData
```

Stack all subject maps into a single `BrainData` (subjects as rows).

(data-brain-collection-cv)=
#### `cv`

```python
cv(*, k: int | None = None, method: str = 'kfold', split_by: str | None = None, groups: np.ndarray | None = None, n: int = 1000, random_state: int | None = None) -> BrainCollectionPipeline
```

Build a CV pipeline for cross-subject prediction.

See ``pipeline.py`` for the builder API. The pipeline's ``predict``
terminal returns a ``BrainData`` with CV attrs attached.

(data-brain-collection-detrend)=
#### `detrend`

```python
detrend(*, method: str = 'linear', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Detrend every subject's image in parallel (delegates to `BrainData.detrend`).

(data-brain-collection-filter)=
#### `filter`

```python
filter(predicate: Callable[[Any], Any] | list | np.ndarray | pl.Series | pd.Series) -> BrainCollection
```

Filter to a subset by predicate, polars expression, or boolean array.

(data-brain-collection-fit)=
#### `fit`

```python
fit(model: str = 'glm', X: DesignMatrix | list | Callable | None = None, *, scale: bool | str = 'auto', standardize: str | None = 'auto', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto', **model_kwargs: Literal['auto', True, False]) -> BrainCollection
```

Per-subject fit; returns a path-backed collection of HDF5 fit bundles.

``X`` resolution priority:
  - ``None``         → use ``self.designs`` (must be set per subject)
  - ``DesignMatrix`` → shared across all subjects
  - ``list``         → per-subject (len == n_subjects)
  - ``callable``     → ``fn(ctx: _DesignContext) -> DesignMatrix``

(data-brain-collection-from-bids)=
#### `from_bids`

```python
from_bids(root: Path | str | Any, *, mask: nib.Nifti1Image | Path | str, task: str | None = None, space: str | None = None, sub_labels: list[str] | None = None, img_filters: list[tuple[str, str]] | None = None, derivatives_folder: str = 'derivatives', pair_events: bool = True, confounds_strategy: str | tuple[str, ...] | None = None, confounds_kwargs: dict | None = None, TR: float | str = 'infer', cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Auto-pair BOLD with events.tsv (→ ``DesignMatrix``) and confounds.tsv.

Full design and edge cases: SPEC §"``from_bids`` — concrete design".

(data-brain-collection-from-glob)=
#### `from_glob`

```python
from_glob(pattern: str, *, mask: nib.Nifti1Image | Path | str, design_pattern: str | None = None, pattern_groups: dict[str, int] | str | None = None, sort: bool = True, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Build a collection by glob-matching brain images (and optional designs).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`pattern` | <code>[str](#str)</code> | Glob pattern matching the per-subject brain image files. | *required*
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str)</code> | Shared mask image, path, or nltools template name. | *required*
`design_pattern` | <code>[str](#str) \| None</code> | Optional glob matching per-subject design files, paired positionally with the brain images. | <code>None</code>
`pattern_groups` | <code>[dict](#dict)[[str](#str), [int](#int)] \| [str](#str) \| None</code> | Regex capture-group spec used to extract metadata (e.g. subject/run) from each matched path. | <code>None</code>
`sort` | <code>[bool](#bool)</code> | If True, sort matched paths before pairing (stable ordering). | <code>True</code>
`cache_dir` | <code>[Path](#pathlib.Path) \| [str](#str) \| None</code> | Cache-directory precedence: explicit arg → ``NLTOOLS_CACHE_DIR`` env → ``./.nltools_cache``; ``None`` for a temp dir. | <code>'./.nltools_cache'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | A lazy, path-backed `BrainCollection`.

(data-brain-collection-from-paths)=
#### `from_paths`

```python
from_paths(brain_paths: list, *, mask: nib.Nifti1Image | Path | str, design_paths: list | None = None, metadata: pl.DataFrame | pd.DataFrame | dict | None = None, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Build a collection from explicit lists of brain (and design) paths.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_paths` | <code>[list](#list)</code> | Per-subject brain image paths. | *required*
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str)</code> | Shared mask image, path, or nltools template name. | *required*
`design_paths` | <code>[list](#list) \| None</code> | Optional per-subject design paths, aligned positionally with ``brain_paths`` (length must match, ``None`` entries allowed). | <code>None</code>
`metadata` | <code>[DataFrame](#polars.DataFrame) \| [DataFrame](#pandas.DataFrame) \| [dict](#dict) \| None</code> | Optional per-subject metadata (polars/pandas DataFrame or dict-of-columns), one row per path. | <code>None</code>
`cache_dir` | <code>[Path](#pathlib.Path) \| [str](#str) \| None</code> | Cache-directory precedence: explicit arg → ``NLTOOLS_CACHE_DIR`` env → ``./.nltools_cache``; ``None`` for a temp dir. | <code>'./.nltools_cache'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | A lazy, path-backed `BrainCollection`.

(data-brain-collection-isc)=
#### `isc`

```python
isc(*, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, metric: str = 'median') -> dict
```

Inter-subject correlation (ISC) across the time dimension.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ``'loo'`` (leave-one-out template) or ``'pairwise'`` (all subject pairs). | <code>'loo'</code>
`roi_mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str) \| None</code> | Optional ROI/atlas mask restricting the computation to those voxels. The returned maps carry the ROI mask. If None, ISC is computed across the collection's whole-brain mask. | <code>None</code>
`metric` | <code>[str](#str)</code> | Aggregation across subjects/pairs (e.g. ``'median'``). | <code>'median'</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict ``{'isc', 'per_subject'}`` for ``method='loo'`` or
<code>[dict](#dict)</code> | ``{'isc', 'pairs'}`` for ``method='pairwise'`` (``'isc'`` is a
<code>[dict](#dict)</code> | `BrainData` map).

(data-brain-collection-isc-test)=
#### `isc_test`

```python
isc_test(*, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, n_samples: int = 5000, metric: str = 'median', random_state: int | None = None) -> dict
```

Bootstrap inference on ISC (per-voxel p-values).

Resamples subjects with replacement, recomputes ISC each draw, and
derives a per-voxel two-tailed p-value from the null centered at 0.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ``'loo'`` or ``'pairwise'`` (matches `isc`). | <code>'loo'</code>
`roi_mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str) \| None</code> | Optional ROI/atlas mask restricting the computation to those voxels. The returned maps carry the ROI mask. If None, ISC is computed across the collection's whole-brain mask. | <code>None</code>
`n_samples` | <code>[int](#int)</code> | Number of bootstrap resamples. | <code>5000</code>
`metric` | <code>[str](#str)</code> | Aggregation across subjects/pairs (e.g. ``'median'``). | <code>'median'</code>
`random_state` | <code>[int](#int) \| None</code> | Seed for the bootstrap RNG. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict ``{'isc', 'p', 'null_distribution'}`` (``'isc'`` and ``'p'`` are
<code>[dict](#dict)</code> | `BrainData` maps).

(data-brain-collection-iter-pairs)=
#### `iter_pairs`

```python
iter_pairs() -> Iterator[tuple]
```

Yield ``(BrainData, DesignMatrix | None)`` pairs.

(data-brain-collection-load)=
#### `load`

```python
load(indices: list[int] | None = None) -> BrainCollection
```

Materialize path-backed items in place. Returns ``self`` for chaining.

(data-brain-collection-map)=
#### `map`

```python
map(fn: Callable, *, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Apply an arbitrary ``fn(BrainData) -> BrainData`` to each item in parallel.

(data-brain-collection-max)=
#### `max`

```python
max() -> BrainData
```

Voxelwise maximum across subjects as a single `BrainData`.

(data-brain-collection-mean)=
#### `mean`

```python
mean() -> BrainData
```

Voxelwise mean across subjects as a single `BrainData`.

(data-brain-collection-median)=
#### `median`

```python
median() -> BrainData
```

Voxelwise median across subjects as a single `BrainData`.

(data-brain-collection-memory-estimate)=
#### `memory_estimate`

```python
memory_estimate() -> str
```

Human-readable RAM estimate if every item were loaded into memory.

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | A string reporting ``n_subjects``, the per-item shape (or "unknown"
<code>[str](#str)</code> | for path-backed items not yet loaded), and an estimated float32
<code>[str](#str)</code> | total in MB/GB.

(data-brain-collection-min)=
#### `min`

```python
min() -> BrainData
```

Voxelwise minimum across subjects as a single `BrainData`.

(data-brain-collection-permutation-test)=
#### `permutation_test`

```python
permutation_test(*, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

One-sample sign-flipping permutation test across subjects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_permute` | <code>[int](#int)</code> | Number of sign-flip permutations. | <code>5000</code>
`tail` | <code>[int](#int)</code> | 1 for one-tailed, 2 for two-tailed. | <code>2</code>
`device` | <code>[str](#str)</code> | Backend selector (currently informational). | <code>'cpu'</code>
`return_null` | <code>[bool](#bool)</code> | If True, include the null distribution in the result. | <code>False</code>
`n_jobs` | <code>[int](#int)</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Seed for the sign-flip RNG. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict ``{'mean', 'p'}`` of `BrainData` maps, plus
<code>[dict](#dict)</code> | ``'null_distribution'`` when ``return_null=True``.

(data-brain-collection-permutation-test2)=
#### `permutation_test2`

```python
permutation_test2(other: BrainCollection, *, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

Two-sample permutation test between this collection and ``other``.

Uses random label shuffling of the pooled subjects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | The second collection to compare against. | *required*
`n_permute` | <code>[int](#int)</code> | Number of label-shuffle permutations. | <code>5000</code>
`tail` | <code>[int](#int)</code> | 1 for one-tailed, 2 for two-tailed. | <code>2</code>
`device` | <code>[str](#str)</code> | Backend selector (currently informational). | <code>'cpu'</code>
`return_null` | <code>[bool](#bool)</code> | If True, include the null distribution in the result. | <code>False</code>
`n_jobs` | <code>[int](#int)</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Seed for the shuffling RNG. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict ``{'mean', 'p'}`` of `BrainData` maps (``mean`` is the group
<code>[dict](#dict)</code> | difference), plus ``'null_distribution'`` when ``return_null=True``.

(data-brain-collection-predict)=
#### `predict`

```python
predict(y: str | list | np.ndarray | None = None, *, X_new: np.ndarray | None = None, spatial_scale: str = 'whole_brain', model: str = 'svm', cv: int | str = 'loso', groups: str | np.ndarray | None = None, roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float = 10.0, scoring: str = 'auto', standardize: bool = True, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto')
```

Two distinct paths, dispatched by argument:

  ``y=`` only    → group MVPA (subjects as samples) → ``Predict``
  ``X_new=`` only → per-subject predict-after-fit  → ``BrainCollection``
  both / neither → raise

``predict(y=...)`` requires single-map-per-subject items (run
``compute_contrasts(...)`` first if you have GLM/ridge bundles).

(data-brain-collection-read)=
#### `read`

```python
read(directory: Path | str, *, mask: nib.Nifti1Image | Path | str, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Inverse of ``write()``. Does not recover from cache subdirs in v0.6.0.

(data-brain-collection-resample)=
#### `resample`

```python
resample(target, *, interpolation: str = 'continuous', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Resample every subject's image to a target space in parallel.

Delegates to `BrainData.resample`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` |  | Resampling target (image, affine/shape spec, or template) passed through to `BrainData.resample`. | *required*
`interpolation` | <code>[str](#str)</code> | Interpolation method (``'continuous'``, ``'linear'``, ``'nearest'``). | <code>'continuous'</code>
`n_jobs` | <code>[int](#int)</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>[bool](#bool)</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>[Literal](#typing.Literal)['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | A new `BrainCollection` of resampled items.

(data-brain-collection-smooth)=
#### `smooth`

```python
smooth(fwhm: float, *, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Spatially smooth every subject's image in parallel (delegates to `BrainData.smooth`).

(data-brain-collection-standardize)=
#### `standardize`

```python
standardize(*, axis: int = 0, method: str = 'center', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Standardize every subject's image in parallel (delegates to `BrainData.standardize`).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>[int](#int)</code> | Axis along which to standardize (0 = across observations). | <code>0</code>
`method` | <code>[str](#str)</code> | Standardization variant (e.g. ``'center'``, ``'zscore'``). | <code>'center'</code>
`n_jobs` | <code>[int](#int)</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>[bool](#bool)</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>[Literal](#typing.Literal)['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | A new `BrainCollection` of standardized items.

(data-brain-collection-std)=
#### `std`

```python
std() -> BrainData
```

Voxelwise standard deviation across subjects as a single `BrainData`.

(data-brain-collection-steps)=
#### `steps`

```python
steps() -> list[Path]
```

Step subdirs that produced this collection's items, oldest to newest.

Lineage chain accumulated through clones (one entry per upstream
cached op). Empty when the collection was constructed directly or
no ancestor wrote to disk.

(data-brain-collection-sum)=
#### `sum`

```python
sum() -> BrainData
```

Voxelwise sum across subjects as a single `BrainData`.

(data-brain-collection-threshold)=
#### `threshold`

```python
threshold(*, lower: float | None = None, upper: float | None = None, binarize: bool = False, coerce_nan: bool = True, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Threshold every subject's image in parallel (delegates to `BrainData.threshold`).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`lower` | <code>[float](#float) \| None</code> | Values below this are zeroed (or set NaN); ``None`` disables. | <code>None</code>
`upper` | <code>[float](#float) \| None</code> | Values above this are zeroed (or set NaN); ``None`` disables. | <code>None</code>
`binarize` | <code>[bool](#bool)</code> | If True, set surviving voxels to 1. | <code>False</code>
`coerce_nan` | <code>[bool](#bool)</code> | If True, coerce thresholded-out voxels to NaN instead of 0. | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>[bool](#bool)</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>[Literal](#typing.Literal)['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | A new `BrainCollection` of thresholded items.

(data-brain-collection-transform-designs)=
#### `transform_designs`

```python
transform_designs(fn: Callable, *, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Map ``fn(dm) -> DesignMatrix`` over each paired design.

Items with no paired design are skipped (kept as ``None``). Runs in
the parent process — designs are small. ``n_jobs``/``progress_bar``/
``cache`` are accepted for surface consistency but ignored.

(data-brain-collection-ttest)=
#### `ttest`

```python
ttest(*, popmean: float = 0.0) -> dict
```

One-sample t-test across subjects (delegates to `inference.ttest`).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`popmean` | <code>[float](#float)</code> | Null-hypothesis population mean to test against. | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict ``{'mean', 't', 'z', 'p'}`` of `BrainData` maps.

(data-brain-collection-ttest2)=
#### `ttest2`

```python
ttest2(other: BrainCollection, *, equal_var: bool = True) -> dict
```

Two-sample t-test between this collection and ``other`` (subject-level).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | The second collection to compare against. | *required*
`equal_var` | <code>[bool](#bool)</code> | If True, pooled-variance t-test; if False, Welch's test. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dict ``{'mean', 't', 'z', 'p'}`` of `BrainData` maps (``mean`` is the
<code>[dict](#dict)</code> | group difference).

(data-brain-collection-unload)=
#### `unload`

```python
unload(indices: list[int] | None = None) -> BrainCollection
```

Drop in-memory data for items with backing paths. Returns ``self``.

(data-brain-collection-var)=
#### `var`

```python
var() -> BrainData
```

Voxelwise variance across subjects as a single `BrainData`.

(data-brain-collection-write)=
#### `write`

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
`directory` | <code>[Path](#pathlib.Path) \| [str](#str)</code> | Output directory (created if missing). | *required*
`pattern` | <code>[str](#str)</code> | Filename template per item, formatted with ``i`` (item index). | <code>'image_{i:04d}.nii.gz'</code>
`metadata_file` | <code>[str](#str) \| None</code> | CSV filename for the metadata table, or ``None`` to skip. | <code>'metadata.csv'</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[Path](#pathlib.Path)]</code> | List of written NIfTI paths, in item order.

