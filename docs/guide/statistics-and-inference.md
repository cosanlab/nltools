---
title: Statistics & inference
---

Group inference in nltools is voxelwise and mostly non-parametric.
[`BrainData.ttest`](../api/data/brain_data.md#nltools.data.braindata.BrainData.ttest) runs a one-sample test at
every voxel of a stacked `(n_subjects, n_voxels)` object and returns `{'mean', 't', 'z', 'p'}` as
`BrainData` maps. Pass `permutation=True` for a sign-flipping null instead of the parametric t, and
`popmean=` to test against something other than zero. Everything here also exists as a plain
function on numpy arrays; see the [inference reference](../api/tasks/inference.md).

Four kwargs carry most of the meaning.

- **`tail`.** `2` or `'two'` (the default) is two-tailed; `1` or `'one'` tests the test's own
  positive direction, fixed by the test and never inferred from your data. GLM contrast p-values
  are the one exception: they are always one-sided.
- **`n_permute` vs `n_samples`.** Permutations and bootstrap draws. They are never the same kwarg,
  and a function takes whichever one matches its null. Both default to `5000`; going below `1000`
  earns you a warning.
- **`n_jobs`.** The joblib worker count (`-1` = all cores). It is a speed knob only: a seeded run
  gives the same numbers at every worker count. `device='gpu'` exists on the ridge paths
  (`Ridge`, `BrainData.fit(ridge_device=)`, `BrainData.bootstrap`) and nowhere else; there it runs
  on the GPU or raises.
- **`random_state`.** Set it and results reproduce exactly.

Goal | Use | Notes
--- | --- | ---
One-sample voxelwise test | [`ttest`](../api/data/brain_data.md#nltools.data.braindata.BrainData.ttest)`(popmean=0.0)` | Returns `{'mean', 't', 'z', 'p'}`
Non-parametric one-sample | `ttest(permutation=True, n_permute=)` | Sign flipping; add `return_null=True` for the `'null_dist'` array. Also [`one_sample_permutation_test`](../api/tasks/inference.md#nltools.algorithms.one_sample_permutation_test)
Non-parametric two-sample | [`two_sample_permutation_test`](../api/tasks/inference.md#nltools.algorithms.two_sample_permutation_test) | Group-label shuffling
Correlated time series | [`timeseries_correlation_permutation_test`](../api/tasks/inference.md#nltools.algorithms.timeseries_correlation_permutation_test) | `method='circle_shift'` or `'phase_randomize'` preserves autocorrelation
Build a timeseries null | [`circle_shift`](../api/tasks/inference.md#nltools.algorithms.circle_shift), [`phase_randomize`](../api/tasks/inference.md#nltools.algorithms.phase_randomize) | The surrogate generators used above
Confidence intervals | [`BrainData.bootstrap`](../api/data/brain_data.md#nltools.data.braindata.BrainData.bootstrap), [`Adjacency.bootstrap`](../api/data/adjacency.md#nltools.data.adjacency.Adjacency.bootstrap) | Returns a `BootstrapResult`: `.estimate`, `.standard_error`, `.ci_lower`, `.ci_upper`
Matrix comparison | [`matrix_permutation_test`](../api/tasks/similarity.md#nltools.algorithms.matrix_permutation_test), [`Adjacency.ttest`](../api/data/adjacency.md#nltools.data.adjacency.Adjacency.ttest) | `Adjacency.ttest` takes the same kwargs and returns the same keys, one edgewise `Adjacency` each. See [Similarity & RSA](similarity-and-rsa.md)
FDR / Holm-Bonferroni | [`fdr`](../api/tasks/inference.md#nltools.algorithms.fdr), [`holm_bonf`](../api/tasks/inference.md#nltools.algorithms.holm_bonf) | Both return a *p-threshold*, or `-1` if nothing survives
Apply a threshold | [`threshold`](../api/tasks/inference.md#nltools.algorithms.threshold), [`BrainData.threshold`](../api/data/brain_data.md#nltools.data.braindata.BrainData.threshold) | The function thresholds by a p-map; the method by value (`upper=`/`lower=`)

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

`t` stays the observed parametric statistic; only `p` (and therefore `z`) comes from the
permutation null. Add `return_null=True` to keep that null as `perm["null_dist"]`, an
`(n_permute, n_voxels)` array of centered means for your own correction.

## Bootstrap

```python
roi = group.apply_mask(create_sphere([0, -20, 20], radius=10))
boot = roi.bootstrap("mean", n_samples=1000, random_state=0)

boot.estimate.plot()                       # the mean map itself
boot.ci_lower, boot.ci_upper               # the 95% percentile interval
```

Every statistic returns the same `BootstrapResult`: `estimate` (the statistic on the *unresampled*
data — not the average of the draws), `standard_error` (the `ddof=1` deviation across draws), and
`ci_lower`/`ci_upper`. All four are `BrainData` maps of identical shape.
[`Adjacency.bootstrap`](../api/data/adjacency.md#nltools.data.adjacency.Adjacency.bootstrap) returns the same record
with `Adjacency` payloads.

The six basic statistics — `'mean'`, `'median'`, `'std'`, `'sum'`, `'min'`, `'max'` — reduce the
data itself. `'weights'` and `'predict'` bootstrap a fitted `Ridge`, taking the training features
back explicitly. `confidence_level=` sets one level (default `0.95`), not a percentile pair, and
the bounds are elementwise marginal: the nominal level applies per voxel, with no
multiple-comparison control across the map.

Draws are aggregated as they complete rather than collected, so what the run holds is the retained
tail — about `(1 - confidence_level)` of the draws per output element — plus one dispatch window,
not all `n_samples` maps. That is a large saving, not a free lunch: the tail still grows with
`n_samples`, so a whole-brain 95% bootstrap at `n_samples=5000` needs roughly
`230000 x 258 x 8 bytes ~= 0.5 GB` and an ROI is still the cheaper way to explore.
`return_samples=True` keeps every draw as `boot.samples` and costs the full
`n_samples x output_size`. Whatever the run will hold is checked *before* it resamples: if it does
not fit, `bootstrap` says so and names the `memory_budget_gb=` override, and it never quietly
shrinks the run.

There is no `p` or `z` in the result. A bootstrap hypothesis test is a separate API; if you want a
normal-approximation z, compute `boot.estimate.data / boot.standard_error.data` yourself and be
explicit about the assumption.

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

Next: [Intersubject correlation](intersubject.md).
