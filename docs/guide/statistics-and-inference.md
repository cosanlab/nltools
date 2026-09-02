---
title: Statistics & inference
---

Group inference in nltools is voxelwise and mostly non-parametric.
[`BrainData.ttest`](../api/data/brain_data.md#data-brain-data-ttest) runs a one-sample test at
every voxel of a stacked `(n_subjects, n_voxels)` object and returns `{'mean', 't', 'z', 'p'}` as
`BrainData` maps; [`ttest2`](../api/data/brain_data.md#data-brain-data-ttest2) is the two-sample
version. Pass `permutation=True` for a sign-flipping null instead of the parametric t, and
`popmean=` to test against something other than zero. Everything here also exists as a plain
function on numpy arrays; see the [inference reference](../api/tasks/inference.md).

Four kwargs carry most of the meaning.

- **`tail`.** `2` or `'two'` (the default) is two-tailed; `1` or `'one'` tests the test's own
  positive direction, fixed by the test and never inferred from your data. GLM contrast p-values
  are the one exception: they are always one-sided.
- **`n_permute` vs `n_samples`.** Permutations and bootstrap draws. They are never the same kwarg,
  and a function takes whichever one matches its null. Both default to `5000`; going below `1000`
  earns you a warning.
- **`n_jobs` vs `device`.** `n_jobs` is CPU worker count (`-1` = all cores); `device` is `'cpu'`,
  `'gpu'`, or `'auto'`. They are independent. An explicit `device='gpu'` runs on the GPU or raises;
  only `'auto'` falls back silently.
- **`random_state`.** Set it and results reproduce exactly, including across CPU and GPU backends.

Goal | Use | Notes
--- | --- | ---
One-sample voxelwise test | [`ttest`](../api/data/brain_data.md#data-brain-data-ttest)`(popmean=0.0)` | Returns `{'mean', 't', 'z', 'p'}`
Two-sample voxelwise test | [`ttest2`](../api/data/brain_data.md#data-brain-data-ttest2)`(other, equal_var=)` | Returns `{'t', 'p'}`
Non-parametric one-sample | `ttest(permutation=True, n_permute=)` | Sign flipping; also [`one_sample_permutation_test`](../api/tasks/inference.md#tasks-inference-one-sample-permutation-test)
Non-parametric two-sample | [`two_sample_permutation_test`](../api/tasks/inference.md#tasks-inference-two-sample-permutation-test) | Group-label shuffling
Correlated time series | [`timeseries_correlation_permutation_test`](../api/tasks/inference.md#tasks-inference-timeseries-correlation-permutation-test) | `method='circle_shift'` or `'phase_randomize'` preserves autocorrelation
Build a timeseries null | [`circle_shift`](../api/tasks/inference.md#tasks-inference-circle-shift), [`phase_randomize`](../api/tasks/inference.md#tasks-inference-phase-randomize) | The surrogate generators used above
Confidence intervals | [`BrainData.bootstrap`](../api/data/brain_data.md#data-brain-data-bootstrap), [`Adjacency.bootstrap`](../api/data/adjacency.md#data-adjacency-bootstrap) | `stat='mean'` returns a `BrainData`; model stats return a dict
Matrix comparison | [`matrix_permutation_test`](../api/tasks/similarity.md#tasks-similarity-matrix-permutation-test), [`Adjacency.ttest`](../api/data/adjacency.md#data-adjacency-ttest) | See [Similarity & RSA](similarity-and-rsa.md)
FDR / Holm-Bonferroni | [`fdr`](../api/tasks/inference.md#tasks-inference-fdr), [`holm_bonf`](../api/tasks/inference.md#tasks-inference-holm-bonf) | Both return a *p-threshold*, or `-1` if nothing survives
Apply a threshold | [`threshold`](../api/tasks/inference.md#tasks-inference-threshold), [`BrainData.threshold`](../api/data/brain_data.md#data-brain-data-threshold) | The function thresholds by a p-map; the method by value (`upper=`/`lower=`)

## Group test, corrected

```python
result = group.ttest()                         # mean, t, z, p
p = result["p"].data

fdr(p, q=0.05), holm_bonf(p, alpha=0.05)       # p-thresholds
sig = threshold(result["z"], result["p"], thr=fdr(p, q=0.05))
```

`fdr` and `holm_bonf` return the p-value cutoff, not a mask. Feed it to `threshold`, which zeroes
every voxel of the statistic map whose p exceeds it. When nothing survives they return `-1`, so
check before you plot. `BrainData.threshold(upper=, lower=)` is a different operation: it censors by
the statistic's own value, with `binarize=True` and `cluster_threshold=` for cluster-extent masking.

Swap the parametric t for sign flipping with one kwarg:

```python
perm = group.ttest(permutation=True, n_permute=1000, tail=2, random_state=0)
```

## Bootstrap

```python
roi = group.apply_mask(create_sphere([0, -20, 20], radius=10))
boot = roi.bootstrap("mean", n_samples=1000, random_state=0)
```

`stat='mean'` (or `'median'`, `'std'`, `'sum'`, `'min'`, `'max'`) returns a single `BrainData` of the
bootstrap estimate. `'weights'` and `'predict'` bootstrap a fitted `Ridge` model and return a dict
of `mean`, `std`, `Z`, `p`, `ci_lower`, `ci_upper`. The CPU path holds every draw in memory, so
restrict to an ROI before bootstrapping a whole brain at `n_samples=5000`.
[`Adjacency.bootstrap`](../api/data/adjacency.md#data-adjacency-bootstrap) always returns that dict.

## Plain arrays

```python
from nltools.algorithms import (
    one_sample_permutation_test,
    two_sample_permutation_test,
    timeseries_correlation_permutation_test,
)

one_sample_permutation_test(a, n_permute=1000, random_state=0)["p"]
two_sample_permutation_test(a, b, n_permute=1000, random_state=0)["p"]
timeseries_correlation_permutation_test(
    a, b, method="circle_shift", n_permute=1000, metric="pearson", random_state=0
)["p"]
```

Correlating two autocorrelated time series with an ordinary shuffle null gives p-values that are
far too small. `method='circle_shift'` rotates one series; `method='phase_randomize'` scrambles its
Fourier phases. Both preserve the autocorrelation the naive null throws away.

Next: [Intersubject correlation](intersubject.md), or
[Performance & GPU](../performance.md) for when `device='gpu'` is worth it.
