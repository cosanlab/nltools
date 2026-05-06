## `inference`

Group-level reductions and cross-subject ops for BrainCollection.

Module-level functions that the ``BrainCollection`` facade delegates to.
Reductions stream from path-backed inputs (Welford-style) and produce
in-memory ``BrainData`` (or dicts of them); they never path-back their
own output.

**Methods:**

Name | Description
---- | -----------
[`align`](#align) | Functional alignment (Procrustes / SRM / hyperalignment).
[`anova`](#anova) | One-way ANOVA across subjects.
[`concat`](#concat) | Stack along axis 0 → ``BrainData`` of shape ``(n_total_obs, n_voxels)``.
[`isc`](#isc) | Inter-subject correlation.
[`isc_test`](#isc_test) | Permutation/bootstrap inference on ISC.
[`max_`](#max_) | Per-voxel max across subjects. Streams.
[`mean`](#mean) | Mean across subjects (leading axis). Streams from path-backed input.
[`median`](#median) | Median across subjects. Not streaming-friendly — materializes.
[`min_`](#min_) | Per-voxel min across subjects. Streams.
[`permutation_test`](#permutation_test) | Sign-flipping permutation test across subjects.
[`permutation_test2`](#permutation_test2) | Two-sample permutation test between two collections.
[`std`](#std) | Std across subjects. Streams via Welford; ddof=1 by default.
[`sum_`](#sum_) | Sum across subjects. Streams.
[`ttest`](#ttest) | One-sample t-test across subjects.
[`ttest2`](#ttest2) | Two-sample t-test between two collections.
[`var`](#var) | Variance across subjects. Streams via Welford; ddof=1 by default.



### Classes

### Methods

#### `align`

```python
align(bc: BrainCollection, *, method: str = 'procrustes', scheme: str = 'searchlight', radius_mm: float = 10.0, parcellation: nib.Nifti1Image | None = None, n_features: int | None = None, n_iter: int = 3, device: str = 'cpu', return_model: bool = False, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto')
```

Functional alignment (Procrustes / SRM / hyperalignment).

Returns ``BrainCollection`` (aligned data) or
``(BrainCollection, LocalAlignment)`` when ``return_model=True``.
Materializes all subjects in v0.6 (algorithm constraint, see SPEC
streaming-algorithms table).

#### `anova`

```python
anova(bc: BrainCollection, groups: str | list | np.ndarray) -> dict[str, BrainData]
```

One-way ANOVA across subjects.

``groups`` is a metadata column name, a list, or an ndarray of length
``n_subjects``.

#### `concat`

```python
concat(bc: BrainCollection) -> BrainData
```

Stack along axis 0 → ``BrainData`` of shape ``(n_total_obs, n_voxels)``.

Not streamable — the operation *is* materialization. Items must share
a voxel dimension; mismatched shapes raise.

#### `isc`

```python
isc(bc: BrainCollection, *, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float | None = 6.0, metric: str = 'median', device: str = 'cpu', n_jobs: int = -1, progress_bar: bool = False) -> dict
```

Inter-subject correlation.

LOO method streams (two passes, sum-trick); pairwise streams two
subjects at a time. Voxelwise/searchlight path goes through
``nltools.algorithms.inference.isc`` after the v0.6 streaming rewrite.

#### `isc_test`

```python
isc_test(bc: BrainCollection, *, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float | None = 6.0, n_permute: int = 5000, permutation_method: str = 'bootstrap', metric: str = 'median', device: str = 'cpu', n_jobs: int = -1, progress_bar: bool = False, random_state: int | None = None) -> dict
```

Permutation/bootstrap inference on ISC.

#### `max_`

```python
max_(bc: BrainCollection) -> BrainData
```

Per-voxel max across subjects. Streams.

#### `mean`

```python
mean(bc: BrainCollection) -> BrainData
```

Mean across subjects (leading axis). Streams from path-backed input.

#### `median`

```python
median(bc: BrainCollection) -> BrainData
```

Median across subjects. Not streaming-friendly — materializes.

#### `min_`

```python
min_(bc: BrainCollection) -> BrainData
```

Per-voxel min across subjects. Streams.

#### `permutation_test`

```python
permutation_test(bc: BrainCollection, *, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

Sign-flipping permutation test across subjects.

Materializes all subjects (or memmaps); see SPEC streaming-algorithms
table — sign-flipping needs the full set in memory by design.

#### `permutation_test2`

```python
permutation_test2(bc: BrainCollection, other: BrainCollection, *, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

Two-sample permutation test between two collections.

#### `std`

```python
std(bc: BrainCollection) -> BrainData
```

Std across subjects. Streams via Welford; ddof=1 by default.

#### `sum_`

```python
sum_(bc: BrainCollection) -> BrainData
```

Sum across subjects. Streams.

#### `ttest`

```python
ttest(bc: BrainCollection, *, popmean: float = 0.0) -> dict[str, BrainData]
```

One-sample t-test across subjects.

Returns ``{'mean', 't', 'z', 'p'}`` — same shape contract as
``BrainData.ttest``. Streams from path-backed input via Welford.

#### `ttest2`

```python
ttest2(bc: BrainCollection, other: BrainCollection, *, equal_var: bool = True) -> dict[str, BrainData]
```

Two-sample t-test between two collections.

#### `var`

```python
var(bc: BrainCollection) -> BrainData
```

Variance across subjects. Streams via Welford; ddof=1 by default.

