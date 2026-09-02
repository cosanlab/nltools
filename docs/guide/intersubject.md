---
title: Intersubject correlation
---

Intersubject correlation asks how much of a subject's timecourse is shared with everyone else
watching the same movie. [`isc`](../api/tasks/intersubject.md#tasks-intersubject-isc) takes one
`(n_observations, n_subjects)` array for a voxel, a parcel, or a component, correlates every pair of
subjects, and summarizes the pairwise matrix with the median by default, following Chen et al.
(2016). `summary='mean'` averages after a Fisher r-to-z transform instead, which avoids inflating
the estimate; anything else is not a `summary`.

`method=` picks the null. `'bootstrap'` (the default) resamples subjects with replacement and gives
percentile p-values. `'circle_shift'` and `'phase_randomize'` are surrogate nulls that rotate or
phase-scramble each timeseries, preserving its autocorrelation. Use them to test against any
time-locked structure at all rather than against subject sampling. The bootstrap counts draws, so
its kwarg is `n_samples`, not `n_permute`.

Goal | Use | Notes
--- | --- | ---
ISC of one timeseries set | [`isc`](../api/tasks/intersubject.md#tasks-intersubject-isc)`(data, method='bootstrap')` | Returns `{'isc', 'p', 'ci', ...}`; `data` is observations × subjects
Surrogate null instead | `isc(..., method='circle_shift' \| 'phase_randomize')` | Preserves temporal autocorrelation
Leave-one-out ISC | [`isc_permutation_test`](../api/tasks/intersubject.md#tasks-intersubject-isc-permutation-test)`(summary_statistic='leave-one-out')` | The engine under `isc`, with `device='gpu'` available
Region-to-region | [`isfc`](../api/tasks/intersubject.md#tasks-intersubject-isfc) | Takes a list of per-subject `(n_obs, n_regions)` matrices
Moment-to-moment synchrony | [`isps`](../api/tasks/intersubject.md#tasks-intersubject-isps) | Band-limited phase synchrony; set `sampling_freq=` and the band
Compare two groups | [`isc_group`](../api/tasks/intersubject.md#tasks-intersubject-isc-group) | `method='permute'` shuffles group labels; `'bootstrap'` resamples
Whole-brain ISC map | [`BrainCollection.isc`](../api/data/brain_collection.md#data-brain-collection-isc)`(method='loo')` | Streams; peak memory ~2 subjects regardless of N
Whole-brain ISC + p-values | [`BrainCollection.isc_test`](../api/data/brain_collection.md#data-brain-collection-isc-test) | Bootstrap over subjects; needs all subjects resident
Restrict to a region | `bc.isc(roi_mask=...)` | Same call, fewer voxels

## One timeseries

```python
from nltools.algorithms import isc, isc_group

result = isc(data, n_samples=1000, summary="median", method="bootstrap", random_state=0)
result["isc"], result["p"], result["ci"]

shifted = isc(data, method="circle_shift", n_samples=1000, random_state=0)
diff = isc_group(group1, group2, n_samples=1000, method="permute", random_state=0)
```

`isc_group` returns `isc_group_difference` alongside `p` and `ci`. `isfc` wants a *list* of
per-subject region matrices, not one stacked array, and `isps` returns `average_angle`,
`vector_length`, and `p` per timepoint:

```python
from nltools.algorithms import isfc, isps

conn = isfc(per_subject_matrices, method="average")
phase = isps(data, sampling_freq=0.5, low_cut=0.04, high_cut=0.07)
```

## Whole brain, across subjects

```python
bc = BrainCollection(brains, mask=mask)

isc_map = bc.isc(method="loo", summary="median")          # {'isc', 'per_subject'}
tested = bc.isc_test(method="loo", n_samples=1000, summary="median", random_state=0)
```

`method='loo'` correlates each subject against the average of the others and streams the data, so
memory stays at roughly two subjects no matter how many you have. `method='pairwise'` computes all
`n(n-1)/2` pairs and has to materialize every subject; so does `isc_test`, which needs random
subject access across bootstrap draws.

## Gotchas

- `BrainCollection.isc` and `isc_test` accept neither `n_jobs` nor `device`. They are streaming
  reductions, not per-subject parallel operations. The array-level `isc_permutation_test` is where
  `device='gpu'` lives, and it pays off for long timeseries with many permutations. See
  [Performance & GPU](../performance.md).
- `exclude_self_corr=True` (the default) sets a subject's correlation with itself to NaN when the
  bootstrap draws them twice. Turning it off inflates ISC.
- Use `n_samples` for the bootstrap and `n_permute` only where the docs say permutation. Mixing them
  up is the most common error on this page.

Next: [Functional alignment](alignment.md), or the
[ISC tutorial](../tutorials/workflows/04_isc.md) for a full naturalistic-data walkthrough.
