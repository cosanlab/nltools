---
title: Intersubject correlation
---

Intersubject correlation asks how much of a subject's timecourse is shared with everyone else
watching the same movie. [`isc`](../api/tasks/intersubject.md#nltools.algorithms.isc) takes one
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
ISC of one timeseries set | [`isc`](../api/tasks/intersubject.md#nltools.algorithms.isc)`(data, method='bootstrap')` | Returns `{'isc', 'p', 'ci', ...}`; `data` is observations × subjects
Surrogate null instead | <code>isc(..., method='circle_shift' &#124; 'phase_randomize')</code> | Preserves temporal autocorrelation
Leave-one-out ISC | [`isc_permutation_test`](../api/tasks/intersubject.md#nltools.algorithms.isc_permutation_test)`(summary_statistic='leave-one-out')` | The engine under `isc`, with `device='gpu'` available
Region-to-region | [`isfc`](../api/tasks/intersubject.md#nltools.algorithms.isfc) | Takes a list of per-subject `(n_obs, n_regions)` matrices
Moment-to-moment synchrony | [`isps`](../api/tasks/intersubject.md#nltools.algorithms.isps) | Band-limited phase synchrony; set `sampling_freq=` and the band
Compare two groups | [`isc_group`](../api/tasks/intersubject.md#nltools.algorithms.isc_group) | `method='permute'` shuffles group labels; `'bootstrap'` resamples

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

## Gotchas

- `isc_permutation_test` accepts `device="gpu"` for array-level inference. See the
  `n_jobs` vs `device` guidance in [Statistics & inference](statistics-and-inference.md).
- `exclude_self_corr=True` (the default) sets a subject's correlation with itself to NaN when the
  bootstrap draws them twice. Turning it off inflates ISC.
- Use `n_samples` for the bootstrap and `n_permute` only where the docs say permutation. Mixing them
  up is the most common error on this page.

Next: [Functional alignment](alignment.md), or the
[ISC tutorial](../tutorials/workflows/04_isc.md) for a full naturalistic-data walkthrough.
