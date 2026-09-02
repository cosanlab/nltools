---
title: data.collection.inference
label: page-data-collection-inference
---

Group-level reductions and cross-subject analyses for `BrainCollection`.

Voxelwise summaries (`mean`, `std`, `var`, `median`, `sum_`, `min_`,
`max_`, `concat`), group tests (`ttest`, `ttest2`, `anova`,
`permutation_test`, `permutation_test2`), inter-subject correlation (`isc`,
`isc_test`), and functional alignment (`align`). The `BrainCollection`
methods of the same names delegate here. Where the math allows (means,
variances, t-tests, leave-one-out ISC) inputs are streamed one subject at a
time, so peak memory stays near one subject's worth of data; `median`,
`concat`, the permutation tests, pairwise ISC, and `align` load every
subject. Every result is an in-memory `BrainData` (or a dict of them) and is
never cached to disk.

**Functions:**

Name | Description
---- | -----------
[`align`](#data-collection-inference-align) | Functionally align subjects into a common space via `LocalAlignment`.
[`anova`](#data-collection-inference-anova) | One-way ANOVA across subjects.
[`concat`](#data-collection-inference-concat) | Stack every subject's rows into one `BrainData` of shape ``(n_total_obs, n_voxels)``.
[`isc`](#data-collection-inference-isc) | Inter-subject correlation (ISC) across the time dimension.
[`isc_test`](#data-collection-inference-isc-test) | Bootstrap inference on ISC (per-voxel p-values).
`max_` | Voxelwise maximum across subjects.
[`mean`](#data-collection-inference-mean) | Voxelwise mean across subjects.
[`median`](#data-collection-inference-median) | Voxelwise median across subjects.
`min_` | Voxelwise minimum across subjects.
[`permutation_test`](#data-collection-inference-permutation-test) | One-sample sign-flipping permutation test across subjects.
[`permutation_test2`](#data-collection-inference-permutation-test2) | Two-sample permutation test by random label shuffling of the pooled subjects.
[`std`](#data-collection-inference-std) | Voxelwise standard deviation across subjects (``ddof=1``).
`sum_` | Voxelwise sum across subjects.
[`ttest`](#data-collection-inference-ttest) | One-sample t-test across subjects.
[`ttest2`](#data-collection-inference-ttest2) | Two-sample t-test between two collections (subject-level).
[`var`](#data-collection-inference-var) | Voxelwise variance across subjects (``ddof=1``).

## Functions

(data-collection-inference-align)=
### `align`

```python
align(bc: BrainCollection, *, method: str = 'procrustes', spatial_scale: str = 'searchlight', radius_mm: float = 10.0, roi_mask: nib.Nifti1Image | None = None, n_features: int | None = None, n_iter: int = 3, device: str = 'cpu', return_model: bool = False, n_jobs: int = -1, progress_bar: bool = False, cache: Literal['auto', True, False] = 'auto')
```

Functionally align subjects into a common space via `LocalAlignment`.

Loads every subject — the aligner needs all of them at once. Outputs are
cached under the collection's cache root by the same ``cache`` rule as
the per-subject methods.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#page-data-brain-collection)</code> | The subjects to align. | *required*
`method` | <code>str</code> | Alignment solver (e.g. ``'procrustes'``). | <code>'procrustes'</code>
`spatial_scale` | <code>str</code> | ``'searchlight'`` (overlapping spheres) or ``'roi'`` (non-overlapping parcels). | <code>'searchlight'</code>
`radius_mm` | <code>float</code> | Searchlight sphere radius in mm. | <code>10.0</code>
`roi_mask` | <code>Nifti1Image \| None</code> | Parcellation used when ``spatial_scale='roi'``. | <code>None</code>
`n_features` | <code>int \| None</code> | Optional feature count for the common space. | <code>None</code>
`n_iter` | <code>int</code> | Solver iterations. | <code>3</code>
`device` | <code>str</code> | ``'cpu'`` or ``'gpu'``. | <code>'cpu'</code>
`return_model` | <code>bool</code> | If True, also return the fitted `LocalAlignment`. | <code>False</code>
`n_jobs` | <code>int</code> | Parallel worker count (``-1`` uses all cores). | <code>-1</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. | <code>False</code>
`cache` | <code>Literal['auto', True, False]</code> | Cache policy for the result (``'auto'`` follows source state). | <code>'auto'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#page-data-brain-collection) \| tuple[[BrainCollection](#page-data-brain-collection), [LocalAlignment](#tasks-alignment-localalignment)]</code> | The aligned     collection, or ``(collection, model)`` when ``return_model=True``.

(data-collection-inference-anova)=
### `anova`

```python
anova(bc: BrainCollection, groups: str | list | np.ndarray) -> dict[str, BrainData | int]
```

One-way ANOVA across subjects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#page-data-brain-collection)</code> | One map per subject. | *required*
`groups` | <code>str \| list \| ndarray</code> | A metadata column name, or a list / array of length ``n_subjects`` giving each subject's group label. | *required*

**Returns:**

Type | Description
---- | -----------
<code>dict[str, [BrainData](#page-data-brain-data) \| int]</code> | ``{'F', 'p'}`` maps plus the     ``'df_between'`` and ``'df_within'`` degrees of freedom.

(data-collection-inference-concat)=
### `concat`

```python
concat(bc: BrainCollection) -> BrainData
```

Stack every subject's rows into one `BrainData` of shape ``(n_total_obs, n_voxels)``.

Loads every item (the operation *is* materialization). Single-image items
are promoted to ``(1, n_voxels)`` before concatenation.

(data-collection-inference-isc)=
### `isc`

```python
isc(bc: BrainCollection, *, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, summary: str = 'median') -> dict
```

Inter-subject correlation (ISC) across the time dimension.

``method='loo'`` correlates each subject with the average of the others
and streams in two passes (peak memory about two subjects);
``method='pairwise'`` correlates every subject pair and loads all
subjects at once.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#page-data-brain-collection)</code> | One timeseries per subject, aligned in time. | *required*
`method` | <code>str</code> | ``'loo'`` or ``'pairwise'``. | <code>'loo'</code>
`roi_mask` | <code>Nifti1Image \| Path \| str \| None</code> | Optional ROI restricting the computation; the returned maps then carry the ROI mask rather than the collection's whole-brain mask. | <code>None</code>
`summary` | <code>str</code> | How to aggregate across subjects or pairs — ``'median'`` or ``'mean'`` (Fisher-z averaged). | <code>'median'</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | ``{'isc', 'per_subject'}`` for ``'loo'`` or ``{'isc', 'pairs'}``     for ``'pairwise'``, where ``'isc'`` is a `BrainData` map and the     other entry is the per-subject / per-pair correlation array.

(data-collection-inference-isc-test)=
### `isc_test`

```python
isc_test(bc: BrainCollection, *, method: str = 'loo', roi_mask: nib.Nifti1Image | Path | str | None = None, n_samples: int = 5000, summary: str = 'median', tail: int | str = 2, random_state: int | None = None) -> dict
```

Bootstrap inference on ISC (per-voxel p-values).

Resamples subjects with replacement, recomputes ISC on each draw, and
derives a per-voxel p-value from the null distribution centered at 0.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#page-data-brain-collection)</code> | One timeseries per subject, aligned in time. | *required*
`method` | <code>str</code> | ``'loo'`` or ``'pairwise'`` (as in `isc`). | <code>'loo'</code>
`roi_mask` | <code>Nifti1Image \| Path \| str \| None</code> | Optional ROI restricting the computation; the returned maps then carry the ROI mask. | <code>None</code>
`n_samples` | <code>int</code> | Number of bootstrap resamples. | <code>5000</code>
`summary` | <code>str</code> | ``'median'`` or ``'mean'`` aggregation (as in `isc`). | <code>'median'</code>
`tail` | <code>int \| str</code> | ``2``/``'two'`` for two-tailed, or ``1``/``'one'`` for one-tailed (ISC > 0). | <code>2</code>
`random_state` | <code>int \| None</code> | Seed for the bootstrap RNG. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | ``{'isc', 'p', 'null_dist'}`` — ``'isc'`` and ``'p'`` are     `BrainData` maps; ``'null_dist'`` is the bootstrap array.

(data-collection-inference-max)=
### `max_`

```python
max_(bc: BrainCollection) -> BrainData
```

Voxelwise maximum across subjects.

Streams one subject at a time from path-backed input.

(data-collection-inference-mean)=
### `mean`

```python
mean(bc: BrainCollection) -> BrainData
```

Voxelwise mean across subjects.

Streams one subject at a time from path-backed input.

(data-collection-inference-median)=
### `median`

```python
median(bc: BrainCollection) -> BrainData
```

Voxelwise median across subjects.

Loads every item into memory (a median cannot be streamed).

(data-collection-inference-min)=
### `min_`

```python
min_(bc: BrainCollection) -> BrainData
```

Voxelwise minimum across subjects.

Streams one subject at a time from path-backed input.

(data-collection-inference-permutation-test)=
### `permutation_test`

```python
permutation_test(bc: BrainCollection, *, n_permute: int = 5000, tail: int | str = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None, progress_bar: bool = False) -> dict
```

One-sample sign-flipping permutation test across subjects.

Loads every subject (sign-flipping needs the full stack) and delegates to
`one_sample_permutation_test`, so ``device`` and ``n_jobs`` select the
execution backend.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#page-data-brain-collection)</code> | One map per subject. | *required*
`n_permute` | <code>int</code> | Number of sign-flip permutations. | <code>5000</code>
`tail` | <code>int \| str</code> | ``1`` for one-tailed, ``2`` for two-tailed. | <code>2</code>
`device` | <code>str</code> | ``'cpu'`` (joblib parallel) or ``'gpu'`` (PyTorch). | <code>'cpu'</code>
`return_null` | <code>bool</code> | If True, include the null distribution. | <code>False</code>
`n_jobs` | <code>int</code> | CPU workers when ``device='cpu'`` (``-1`` = all cores). | <code>-1</code>
`random_state` | <code>int \| None</code> | Seed for the sign-flip RNG. | <code>None</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | ``{'mean', 'p'}`` `BrainData` maps, plus ``'null_dist'`` when     ``return_null=True``.

(data-collection-inference-permutation-test2)=
### `permutation_test2`

```python
permutation_test2(bc: BrainCollection, other: BrainCollection, *, n_permute: int = 5000, tail: int | str = 2, device: str = 'cpu', return_null: bool = False, n_jobs: int = -1, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Two-sample permutation test by random label shuffling of the pooled subjects.

Loads every subject and delegates to `two_sample_permutation_test`, so
``device`` and ``n_jobs`` select the execution backend.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#page-data-brain-collection)</code> | First group, one map per subject. | *required*
`other` | <code>[BrainCollection](#page-data-brain-collection)</code> | Second group. | *required*
`n_permute` | <code>int</code> | Number of label-shuffle permutations. | <code>5000</code>
`tail` | <code>int \| str</code> | ``1`` for one-tailed, ``2`` for two-tailed. | <code>2</code>
`device` | <code>str</code> | ``'cpu'`` (joblib parallel) or ``'gpu'`` (PyTorch). | <code>'cpu'</code>
`return_null` | <code>bool</code> | If True, include the null distribution. | <code>False</code>
`n_jobs` | <code>int</code> | CPU workers when ``device='cpu'`` (``-1`` = all cores). | <code>-1</code>
`random_state` | <code>int \| None</code> | Seed for the shuffling RNG. | <code>None</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | ``{'mean', 'p'}`` `BrainData` maps (``'mean'`` is the group     difference), plus ``'null_dist'`` when ``return_null=True``.

(data-collection-inference-std)=
### `std`

```python
std(bc: BrainCollection) -> BrainData
```

Voxelwise standard deviation across subjects (``ddof=1``).

Streams one subject at a time via Welford's algorithm.

(data-collection-inference-sum)=
### `sum_`

```python
sum_(bc: BrainCollection) -> BrainData
```

Voxelwise sum across subjects.

Streams one subject at a time from path-backed input.

(data-collection-inference-ttest)=
### `ttest`

```python
ttest(bc: BrainCollection, *, popmean: float = 0.0, tail: int | str = 2) -> dict[str, BrainData]
```

One-sample t-test across subjects.

Streams one subject at a time via Welford's algorithm. The z map is
derived from the reported p, so it matches the requested tail.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#page-data-brain-collection)</code> | One map per subject. | *required*
`popmean` | <code>float</code> | Null-hypothesis population mean. | <code>0.0</code>
`tail` | <code>int \| str</code> | ``2``/``'two'`` for two-tailed, or ``1``/``'one'`` for one-tailed (mean > ``popmean``; negate the data for the other direction). | <code>2</code>

**Returns:**

Type | Description
---- | -----------
<code>dict[str, [BrainData](#page-data-brain-data)]</code> | ``{'mean', 't', 'z', 'p'}`` maps — the same     contract as `BrainData.ttest`.

(data-collection-inference-ttest2)=
### `ttest2`

```python
ttest2(bc: BrainCollection, other: BrainCollection, *, equal_var: bool = True, tail: int | str = 2) -> dict[str, BrainData]
```

Two-sample t-test between two collections (subject-level).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>[BrainCollection](#page-data-brain-collection)</code> | First group, one map per subject. | *required*
`other` | <code>[BrainCollection](#page-data-brain-collection)</code> | Second group. | *required*
`equal_var` | <code>bool</code> | If True, pooled-variance t-test; if False, Welch's. | <code>True</code>
`tail` | <code>int \| str</code> | ``2``/``'two'`` for two-tailed, or ``1``/``'one'`` for one-tailed (``bc`` > ``other``; swap the operands for the other direction). | <code>2</code>

**Returns:**

Type | Description
---- | -----------
<code>dict[str, [BrainData](#page-data-brain-data)]</code> | ``{'mean', 't', 'z', 'p'}`` maps, where     ``'mean'`` is the group difference.

(data-collection-inference-var)=
### `var`

```python
var(bc: BrainCollection) -> BrainData
```

Voxelwise variance across subjects (``ddof=1``).

Streams one subject at a time via Welford's algorithm.
