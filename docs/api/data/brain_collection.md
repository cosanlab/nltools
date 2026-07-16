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
[`align`](#data-brain-collection-align) | 
[`anova`](#data-brain-collection-anova) | 
[`apply`](#data-brain-collection-apply) | Call ``BrainData.<op>(*args, **kwargs)`` on every item in parallel.
[`cleanup`](#data-brain-collection-cleanup) | Remove ``cache_root`` and invalidate every clone derived from ``self``.
[`cleanup_all`](#data-brain-collection-cleanup-all) | Remove every ``.nltools_cache/{run_id}/`` under ``directory``.
[`compute_contrasts`](#data-brain-collection-compute-contrasts) | Compute per-subject contrast maps from fit-bundle items.
[`concat`](#data-brain-collection-concat) | 
[`cv`](#data-brain-collection-cv) | Build a CV pipeline for cross-subject prediction.
[`detrend`](#data-brain-collection-detrend) | 
[`filter`](#data-brain-collection-filter) | Filter to a subset by predicate, polars expression, or boolean array.
[`fit`](#data-brain-collection-fit) | Per-subject fit; returns a path-backed collection of HDF5 fit bundles.
[`from_bids`](#data-brain-collection-from-bids) | Auto-pair BOLD with events.tsv (→ ``DesignMatrix``) and confounds.tsv.
[`from_glob`](#data-brain-collection-from-glob) | 
[`from_paths`](#data-brain-collection-from-paths) | 
[`isc`](#data-brain-collection-isc) | 
[`isc_test`](#data-brain-collection-isc-test) | 
[`iter_pairs`](#data-brain-collection-iter-pairs) | Yield ``(BrainData, DesignMatrix | None)`` pairs.
[`load`](#data-brain-collection-load) | Materialize path-backed items in place. Returns ``self`` for chaining.
[`map`](#data-brain-collection-map) | Apply an arbitrary ``fn(BrainData) -> BrainData`` to each item in parallel.
[`max`](#data-brain-collection-max) | 
[`mean`](#data-brain-collection-mean) | 
[`median`](#data-brain-collection-median) | 
[`memory_estimate`](#data-brain-collection-memory-estimate) | 
[`min`](#data-brain-collection-min) | 
[`permutation_test`](#data-brain-collection-permutation-test) | 
[`permutation_test2`](#data-brain-collection-permutation-test2) | 
[`predict`](#data-brain-collection-predict) | Two distinct paths, dispatched by argument:
[`read`](#data-brain-collection-read) | Inverse of ``write()``. Does not recover from cache subdirs in v0.6.0.
[`resample`](#data-brain-collection-resample) | 
[`smooth`](#data-brain-collection-smooth) | 
[`standardize`](#data-brain-collection-standardize) | 
[`std`](#data-brain-collection-std) | 
[`steps`](#data-brain-collection-steps) | Step subdirs that produced this collection's items, oldest to newest.
[`sum`](#data-brain-collection-sum) | 
[`threshold`](#data-brain-collection-threshold) | 
[`transform_designs`](#data-brain-collection-transform-designs) | Map ``fn(dm) -> DesignMatrix`` over each paired design.
[`ttest`](#data-brain-collection-ttest) | 
[`ttest2`](#data-brain-collection-ttest2) | 
[`unload`](#data-brain-collection-unload) | Drop in-memory data for items with backing paths. Returns ``self``.
[`var`](#data-brain-collection-var) | 
[`write`](#data-brain-collection-write) | 

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`cache_root` | <code>[Path](#pathlib.Path)</code> | 
`designs` | <code>[list](#list)</code> | 
`is_loaded` | <code>[list](#list)[[bool](#bool)]</code> | Per-item flag — True iff the slot holds a ``BrainData`` (not a path).
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image)</code> | 
`metadata` | <code>[DataFrame](#polars.DataFrame)</code> | 
`n_subjects` | <code>[int](#int)</code> | 
`n_voxels` | <code>[int](#int)</code> | Voxel count from the mask. Raises if mask is unset.
`shape` | <code>[tuple](#tuple)[[int](#int), [int](#int) \| None, [int](#int)]</code> | ``(n_subjects, n_obs_or_None_if_ragged, n_voxels)``.

``cache_dir`` precedence: explicit arg → ``NLTOOLS_CACHE_DIR`` env →
``./.nltools_cache``. Pass ``None`` for an auto-cleaned tempdir.
Resolved at construction and frozen on the instance.

### Methods

(data-brain-collection-align)=
#### `align`

```python
align(*, method: str = 'procrustes', scheme: str = 'searchlight', radius_mm: float = 10.0, parcellation: nib.Nifti1Image | None = None, n_features: int | None = None, n_iter: int = 3, device: str = 'cpu', return_model: bool = False, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto')
```

(data-brain-collection-anova)=
#### `anova`

```python
anova(groups: str | list | np.ndarray) -> dict
```

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
compute_contrasts(contrasts: str | list[str] | dict[str, np.ndarray], *, contrast_type: str = 'beta', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection | dict[str, BrainCollection] | dict[str, dict[str, BrainCollection]]
```

Compute per-subject contrast maps from fit-bundle items.

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)] \| [dict](#dict)[[str](#str), [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]]</code> | single contrast + single ``contrast_type`` → ``BrainCollection``
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)] \| [dict](#dict)[[str](#str), [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]]</code> | multiple contrasts (single type)            → ``dict[str, BrainCollection]``
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)] \| [dict](#dict)[[str](#str), [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]]</code> | ``contrast_type='all'`` (single contrast)   → ``dict['beta'|'t'|'z'|'p'|'se', BrainCollection]``
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)] \| [dict](#dict)[[str](#str), [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]]</code> | multiple contrasts + ``contrast_type='all'`` → nested                                              ``dict[name, dict[stat, BrainCollection]]``

Each per-subject NIfTI gets a JSON sidecar with lineage attrs
(``step_id``, ``parent_step_id``, ``op``, ``kwargs``,
``nltools_version``).

(data-brain-collection-concat)=
#### `concat`

```python
concat() -> BrainData
```

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

(data-brain-collection-filter)=
#### `filter`

```python
filter(predicate: Callable[[Any], Any] | list | np.ndarray | pl.Series | pd.Series) -> BrainCollection
```

Filter to a subset by predicate, polars expression, or boolean array.

(data-brain-collection-fit)=
#### `fit`

```python
fit(model: str = 'glm', X: DesignMatrix | list | Callable | None = None, *, scale: bool = True, scale_value: float = 100.0, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto', **model_kwargs: Literal['auto', True, False]) -> BrainCollection
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

(data-brain-collection-from-paths)=
#### `from_paths`

```python
from_paths(brain_paths: list, *, mask: nib.Nifti1Image | Path | str, design_paths: list | None = None, metadata: pl.DataFrame | pd.DataFrame | dict | None = None, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

(data-brain-collection-isc)=
#### `isc`

```python
isc(*, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float | None = 6.0, metric: str = 'median', device: str = 'cpu', n_jobs: int = -1, progress_bar: bool = False) -> dict
```

(data-brain-collection-isc-test)=
#### `isc_test`

```python
isc_test(*, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float | None = 6.0, n_permute: int = 5000, permutation_method: str = 'bootstrap', metric: str = 'median', device: str = 'cpu', n_jobs: int = -1, progress_bar: bool = False, random_state: int | None = None) -> dict
```

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

(data-brain-collection-mean)=
#### `mean`

```python
mean() -> BrainData
```

(data-brain-collection-median)=
#### `median`

```python
median() -> BrainData
```

(data-brain-collection-memory-estimate)=
#### `memory_estimate`

```python
memory_estimate() -> str
```

(data-brain-collection-min)=
#### `min`

```python
min() -> BrainData
```

(data-brain-collection-permutation-test)=
#### `permutation_test`

```python
permutation_test(*, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

(data-brain-collection-permutation-test2)=
#### `permutation_test2`

```python
permutation_test2(other: BrainCollection, *, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

(data-brain-collection-predict)=
#### `predict`

```python
predict(y: str | list | np.ndarray | None = None, *, X_new: np.ndarray | None = None, spatial_scale: str = 'whole_brain', model: str = 'svm', cv: int | str = 'loso', groups: str | np.ndarray | None = None, roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float = 10.0, scoring: str = 'auto', standardize: bool = True, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto')
```

Two distinct paths, dispatched by argument:

  ``y=`` only    → group MVPA (subjects as samples) → ``BrainData``
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

(data-brain-collection-smooth)=
#### `smooth`

```python
smooth(fwhm: float, *, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

(data-brain-collection-standardize)=
#### `standardize`

```python
standardize(*, axis: int = 0, method: str = 'center', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

(data-brain-collection-std)=
#### `std`

```python
std() -> BrainData
```

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

(data-brain-collection-threshold)=
#### `threshold`

```python
threshold(*, lower: float | None = None, upper: float | None = None, binarize: bool = False, coerce_nan: bool = True, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

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

(data-brain-collection-ttest2)=
#### `ttest2`

```python
ttest2(other: BrainCollection, *, equal_var: bool = True) -> dict
```

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

(data-brain-collection-write)=
#### `write`

```python
write(directory: Path | str, *, pattern: str = 'image_{i:04d}.nii.gz', metadata_file: str | None = 'metadata.csv') -> list[Path]
```

