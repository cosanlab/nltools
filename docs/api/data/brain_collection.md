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

**Methods:**

Name | Description
---- | -----------
[`align`](#align) | 
[`anova`](#anova) | 
[`apply`](#apply) | Call ``BrainData.<method>(*args, **kwargs)`` on every item in parallel.
[`cleanup`](#cleanup) | Remove ``cache_root`` and invalidate every clone derived from ``self``.
[`cleanup_all`](#cleanup_all) | Remove every ``.nltools_cache/{run_id}/`` under ``directory``.
[`compute_contrasts`](#compute_contrasts) | Compute per-subject contrast maps from fit-bundle items.
[`concat`](#concat) | 
[`cv`](#cv) | Build a cross-validation pipeline for cross-subject prediction.
[`detrend`](#detrend) | 
[`filter`](#filter) | Filter to a subset by predicate, mask, or boolean Series.
[`fit`](#fit) | Per-subject fit; returns a path-backed collection of HDF5 fit bundles.
[`from_bids`](#from_bids) | Auto-pair BOLD with events.tsv (→ ``DesignMatrix``) and confounds.tsv.
[`from_glob`](#from_glob) | 
[`from_paths`](#from_paths) | 
[`isc`](#isc) | 
[`isc_test`](#isc_test) | 
[`iter_pairs`](#iter_pairs) | Yield ``(BrainData, DesignMatrix | None)`` pairs.
[`load`](#load) | Materialize path-backed items in place.
[`map`](#map) | Apply an arbitrary ``fn(BrainData) -> BrainData`` to each item in parallel.
[`max`](#max) | 
[`mean`](#mean) | 
[`median`](#median) | 
[`memory_estimate`](#memory_estimate) | 
[`min`](#min) | 
[`permutation_test`](#permutation_test) | 
[`permutation_test2`](#permutation_test2) | 
[`predict`](#predict) | Dispatch prediction according to the provided target argument.
[`read`](#read) | Read a collection written by ``write()``.
[`resample`](#resample) | 
[`smooth`](#smooth) | 
[`standardize`](#standardize) | 
[`std`](#std) | 
[`steps`](#steps) | List step subdirs under ``cache_root``, oldest to newest (lex-sorted).
[`sum`](#sum) | 
[`threshold`](#threshold) | 
[`transform_designs`](#transform_designs) | Map a function over paired ``DesignMatrix``es.
[`ttest`](#ttest) | 
[`ttest2`](#ttest2) | 
[`unload`](#unload) | Drop in-memory data for items with backing paths.
[`var`](#var) | 
[`write`](#write) | 

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`cache_root`](#cache_root) | <code>[Path](#pathlib.Path)</code> | 
[`designs`](#designs) | <code>[list](#list)</code> | 
[`is_loaded`](#is_loaded) | <code>[list](#list)[[bool](#bool)]</code> | Per-item flag — True iff the slot holds a ``BrainData`` (not a path).
[`mask`](#mask) | <code>[Nifti1Image](#nibabel.Nifti1Image)</code> | 
[`metadata`](#metadata) | <code>[DataFrame](#polars.DataFrame)</code> | 
[`n_subjects`](#n_subjects) | <code>[int](#int)</code> | 
[`n_voxels`](#n_voxels) | <code>[int](#int)</code> | Return the voxel count from the mask.
[`shape`](#shape) | <code>[tuple](#tuple)[[int](#int), [int](#int) \| None, [int](#int)]</code> | ``(n_subjects, n_obs_or_None_if_ragged, n_voxels)``.

``cache_dir`` precedence: explicit arg → ``NLTOOLS_CACHE_DIR`` env →
``./.nltools_cache``. Pass ``None`` for an auto-cleaned tempdir.
Resolved at construction and frozen on the instance.

### Methods

#### `align`

```python
align(*, method: str = 'procrustes', scheme: str = 'searchlight', radius_mm: float = 10.0, parcellation: nib.Nifti1Image | None = None, n_features: int | None = None, n_iter: int = 3, device: str = 'cpu', return_model: bool = False, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto')
```

#### `anova`

```python
anova(groups: str | list | np.ndarray) -> dict
```

#### `apply`

```python
apply(method: str, *args: str, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto', **kwargs: Literal['auto', True, False]) -> BrainCollection
```

Call ``BrainData.<method>(*args, **kwargs)`` on every item in parallel.

All per-subject methods (``smooth``, ``standardize``, ...) reduce to
this. Centralizes the ``_apply`` plumbing and the cache-knob handling.

#### `cleanup`

```python
cleanup() -> None
```

Remove ``cache_root`` and invalidate every clone derived from ``self``.

#### `cleanup_all`

```python
cleanup_all(directory: Path | str = '.') -> None
```

Remove every ``.nltools_cache/{run_id}/`` under ``directory``.

Wide brush — can kill sibling sessions in the same cwd. Prefer
``bc.cleanup()`` for surgical removal.

#### `compute_contrasts`

```python
compute_contrasts(contrasts: str | list[str] | dict[str, np.ndarray], *, contrast_type: str = 'beta', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection | dict[str, BrainCollection]
```

Compute per-subject contrast maps from fit-bundle items.

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | single contrast + single ``contrast_type`` → ``BrainCollection``
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | multiple contrasts                          → ``dict[str, BrainCollection]``
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | ``contrast_type='all'``                     → ``dict['beta'|'t'|'z'|'p'|'se', BrainCollection]``

#### `concat`

```python
concat() -> BrainData
```

#### `cv`

```python
cv(*, k: int | None = None, method: str = 'kfold', split_by: str | None = None, groups: np.ndarray | None = None, n: int = 1000, random_state: int | None = None) -> BrainCollectionPipeline
```

Build a cross-validation pipeline for cross-subject prediction.

See ``pipeline.py`` for details.

#### `detrend`

```python
detrend(*, method: str = 'linear', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

#### `filter`

```python
filter(predicate: Callable | list | np.ndarray | pl.Series | pd.Series) -> BrainCollection
```

Filter to a subset by predicate, mask, or boolean Series.

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

#### `from_bids`

```python
from_bids(root: Path | str | Any, *, mask: nib.Nifti1Image | Path | str, task: str | None = None, space: str | None = None, sub_labels: list[str] | None = None, img_filters: list[tuple[str, str]] | None = None, derivatives_folder: str = 'derivatives', pair_events: bool = True, confounds_strategy: str | tuple[str, ...] | None = None, confounds_kwargs: dict | None = None, TR: float | str = 'infer', cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Auto-pair BOLD with events.tsv (→ ``DesignMatrix``) and confounds.tsv.

Full design and edge cases: SPEC §"``from_bids`` — concrete design".

#### `from_glob`

```python
from_glob(pattern: str, *, mask: nib.Nifti1Image | Path | str, design_pattern: str | None = None, pattern_groups: dict[str, int] | str | None = None, sort: bool = True, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

#### `from_paths`

```python
from_paths(brain_paths: list, *, mask: nib.Nifti1Image | Path | str, design_paths: list | None = None, metadata: pl.DataFrame | pd.DataFrame | dict | None = None, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

#### `isc`

```python
isc(*, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float | None = 6.0, metric: str = 'median', device: str = 'cpu', n_jobs: int = -1, progress_bar: bool = False) -> dict
```

#### `isc_test`

```python
isc_test(*, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float | None = 6.0, n_permute: int = 5000, permutation_method: str = 'bootstrap', metric: str = 'median', device: str = 'cpu', n_jobs: int = -1, progress_bar: bool = False, random_state: int | None = None) -> dict
```

#### `iter_pairs`

```python
iter_pairs() -> Iterator[tuple]
```

Yield ``(BrainData, DesignMatrix | None)`` pairs.

#### `load`

```python
load(indices: list[int] | None = None) -> BrainCollection
```

Materialize path-backed items in place.

Returns ``self`` for chaining.

#### `map`

```python
map(fn: Callable, *, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Apply an arbitrary ``fn(BrainData) -> BrainData`` to each item in parallel.

#### `max`

```python
max() -> BrainData
```

#### `mean`

```python
mean() -> BrainData
```

#### `median`

```python
median() -> BrainData
```

#### `memory_estimate`

```python
memory_estimate() -> str
```

#### `min`

```python
min() -> BrainData
```

#### `permutation_test`

```python
permutation_test(*, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

#### `permutation_test2`

```python
permutation_test2(other: BrainCollection, *, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

#### `predict`

```python
predict(y: str | list | np.ndarray | None = None, *, X_new: np.ndarray | None = None, spatial_scale: str = 'whole_brain', estimator: str = 'svm', cv: int | str = 'loso', groups: str | np.ndarray | None = None, roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float = 10.0, scoring: str = 'accuracy', standardize: bool = True, return_weights: bool = True, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto', **kwargs: Literal['auto', True, False])
```

Dispatch prediction according to the provided target argument.

- ``y=`` only → group MVPA (subjects as samples) → ``BrainData``
- ``X_new=`` only → per-subject predict-after-fit → ``BrainCollection``
- both / neither → raise

``predict(y=...)`` requires single-map-per-subject items (run
``compute_contrasts(...)`` first if you have GLM/ridge bundles).

#### `read`

```python
read(directory: Path | str, *, mask: nib.Nifti1Image | Path | str, cache_dir: Path | str | None = './.nltools_cache') -> BrainCollection
```

Read a collection written by ``write()``.

This does not recover from cache subdirectories in v0.6.0.

#### `resample`

```python
resample(target, *, interpolation: str = 'continuous', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

#### `smooth`

```python
smooth(fwhm: float, *, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

#### `standardize`

```python
standardize(*, axis: int = 0, method: str = 'center', n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

#### `std`

```python
std() -> BrainData
```

#### `steps`

```python
steps() -> list[Path]
```

List step subdirs under ``cache_root``, oldest to newest (lex-sorted).

#### `sum`

```python
sum() -> BrainData
```

#### `threshold`

```python
threshold(*, lower: float | None = None, upper: float | None = None, binarize: bool = False, coerce_nan: bool = True, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

#### `transform_designs`

```python
transform_designs(fn: Callable, *, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto') -> BrainCollection
```

Map a function over paired ``DesignMatrix``es.

``fn`` may take either a ``DesignMatrix`` or a ``DesignContext``;
the wrapper inspects arity and dispatches.

#### `ttest`

```python
ttest(*, popmean: float = 0.0) -> dict
```

#### `ttest2`

```python
ttest2(other: BrainCollection, *, equal_var: bool = True) -> dict
```

#### `unload`

```python
unload(indices: list[int] | None = None) -> BrainCollection
```

Drop in-memory data for items with backing paths.

Returns ``self``.

#### `var`

```python
var() -> BrainData
```

#### `write`

```python
write(directory: Path | str, *, pattern: str = 'image_{i:04d}.nii.gz', metadata_file: str | None = 'metadata.csv') -> list[Path]
```

