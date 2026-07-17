(data-collection-inference-inference)=
## `inference`

Group-level reductions and cross-subject ops for BrainCollection.

Module-level functions that the ``BrainCollection`` facade delegates to.
Reductions stream from path-backed inputs (Welford-style) and produce
in-memory ``BrainData`` (or dicts of them); they never path-back their
own output.

**Methods:**

Name | Description
---- | -----------
[`align`](#data-collection-inference-align) | Functional alignment via ``LocalAlignment``.
[`anova`](#data-collection-inference-anova) | One-way ANOVA across subjects.
[`concat`](#data-collection-inference-concat) | Stack along axis 0 → ``BrainData`` of shape ``(n_total_obs, n_voxels)``.
[`isc`](#data-collection-inference-isc) | Inter-subject correlation across the time dimension.
[`isc_test`](#data-collection-inference-isc-test) | Bootstrap inference on ISC.
`max_` | Per-voxel max across subjects. Streams.
[`mean`](#data-collection-inference-mean) | Mean across subjects (leading axis). Streams from path-backed input.
[`median`](#data-collection-inference-median) | Median across subjects. Materializes (not streaming-friendly).
`min_` | Per-voxel min across subjects. Streams.
[`permutation_test`](#data-collection-inference-permutation-test) | Sign-flipping permutation test across subjects (one-sample).
[`permutation_test2`](#data-collection-inference-permutation-test2) | Two-sample permutation test by random label shuffling.
[`std`](#data-collection-inference-std) | Std across subjects. Streams via Welford; ddof=1.
`sum_` | Sum across subjects. Streams.
[`ttest`](#data-collection-inference-ttest) | One-sample t-test across subjects.
[`ttest2`](#data-collection-inference-ttest2) | Two-sample t-test between two collections (subject-level).
[`var`](#data-collection-inference-var) | Variance across subjects. Streams via Welford; ddof=1.



### Classes

### Methods

(data-collection-inference-align)=
#### `align`

```python
align(bc: BrainCollection, *, method: str = 'procrustes', spatial_scale: str = 'searchlight', radius_mm: float = 10.0, roi_mask: nib.Nifti1Image | None = None, n_features: int | None = None, n_iter: int = 3, device: str = 'cpu', return_model: bool = False, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto')
```

Functional alignment via ``LocalAlignment``.

Materializes all subjects (algorithm constraint in v0.6.0). Returns
a new ``BrainCollection`` of aligned data, or
``(BrainCollection, LocalAlignment)`` when ``return_model=True``.

(data-collection-inference-anova)=
#### `anova`

```python
anova(bc: BrainCollection, groups: str | list | np.ndarray) -> dict[str, BrainData | int]
```

One-way ANOVA across subjects.

``groups`` is a metadata column name, a list, or an ndarray of length
``n_subjects``. Returns ``{'F', 'p', 'df_between', 'df_within'}``.

(data-collection-inference-concat)=
#### `concat`

```python
concat(bc: BrainCollection) -> BrainData
```

Stack along axis 0 → ``BrainData`` of shape ``(n_total_obs, n_voxels)``.

Not streamable — the operation *is* materialization. 1D items are
promoted to ``(1, n_voxels)`` before concatenation.

(data-collection-inference-isc)=
#### `isc`

```python
isc(bc: BrainCollection, *, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float | None = 6.0, metric: str = 'median', device: str = 'cpu', n_jobs: int = -1, progress_bar: bool = False) -> dict
```

Inter-subject correlation across the time dimension.

method='loo' uses the leave-one-out template approach (each subject
correlated with the average of the others). method='pairwise' computes
all subject pairs. Both materialize all subjects in v0.6.0; the
streaming rewrite is deferred to a later release.

Returns ``{'isc', 'per_subject'}`` for ``loo`` or ``{'isc', 'pairs'}``
for ``pairwise``.

(data-collection-inference-isc-test)=
#### `isc_test`

```python
isc_test(bc: BrainCollection, *, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, radius_mm: float | None = 6.0, n_samples: int = 5000, metric: str = 'median', device: str = 'cpu', n_jobs: int = -1, progress_bar: bool = False, random_state: int | None = None) -> dict
```

Bootstrap inference on ISC.

Resamples subjects with replacement, recomputes ISC each draw, and
derives a per-voxel p-value from the null distribution centered at 0.

(data-collection-inference-max)=
#### `max_`

```python
max_(bc: BrainCollection) -> BrainData
```

Per-voxel max across subjects. Streams.

(data-collection-inference-mean)=
#### `mean`

```python
mean(bc: BrainCollection) -> BrainData
```

Mean across subjects (leading axis). Streams from path-backed input.

(data-collection-inference-median)=
#### `median`

```python
median(bc: BrainCollection) -> BrainData
```

Median across subjects. Materializes (not streaming-friendly).

(data-collection-inference-min)=
#### `min_`

```python
min_(bc: BrainCollection) -> BrainData
```

Per-voxel min across subjects. Streams.

(data-collection-inference-permutation-test)=
#### `permutation_test`

```python
permutation_test(bc: BrainCollection, *, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

Sign-flipping permutation test across subjects (one-sample).

Per SPEC streaming-algorithms table, sign-flipping needs all subjects
in memory by design. ``device`` is currently informational; backend
selection is deferred to the parametric stats path.

(data-collection-inference-permutation-test2)=
#### `permutation_test2`

```python
permutation_test2(bc: BrainCollection, other: BrainCollection, *, n_permute: int = 5000, tail: int = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None) -> dict
```

Two-sample permutation test by random label shuffling.

(data-collection-inference-std)=
#### `std`

```python
std(bc: BrainCollection) -> BrainData
```

Std across subjects. Streams via Welford; ddof=1.

(data-collection-inference-sum)=
#### `sum_`

```python
sum_(bc: BrainCollection) -> BrainData
```

Sum across subjects. Streams.

(data-collection-inference-ttest)=
#### `ttest`

```python
ttest(bc: BrainCollection, *, popmean: float = 0.0) -> dict[str, BrainData]
```

One-sample t-test across subjects.

Returns ``{'mean', 't', 'z', 'p'}`` — same shape contract as
``BrainData.ttest``. Streams from path-backed input via Welford.

(data-collection-inference-ttest2)=
#### `ttest2`

```python
ttest2(bc: BrainCollection, other: BrainCollection, *, equal_var: bool = True) -> dict[str, BrainData]
```

Two-sample t-test between two collections (subject-level).

(data-collection-inference-var)=
#### `var`

```python
var(bc: BrainCollection) -> BrainData
```

Variance across subjects. Streams via Welford; ddof=1.

