---
title: Design matrices & GLM
---

A [`DesignMatrix`](../api/data/design_matrix.md) is one row per timepoint, one column per
regressor, plus the metadata a GLM needs: the sampling frequency, which columns are already
HRF-convolved, and which are nuisance. Build one from a BIDS events `.tsv`, a confounds table, a
numpy array, or a dict. An events file is detected by its `onset` and `duration` columns: each
`trial_type` becomes a boxcar named `<type>_c0` and is convolved with the Glover HRF for you.
Pass `hrf_model=None` to keep raw boxcars and call [`convolve`](../api/data/design_matrix.md#nltools.data.designmatrix.DesignMatrix.convolve)
yourself.

Nuisance regressors go in through `append(..., axis=1, as_confounds=True)`, which marks them so
`convolve` skips them and `vif` ignores them. Add low-frequency drift with
[`add_poly`](../api/data/design_matrix.md#nltools.data.designmatrix.DesignMatrix.add_poly) (polynomials) or
[`add_dct_basis`](../api/data/design_matrix.md#nltools.data.designmatrix.DesignMatrix.add_dct_basis) (cosines). Pick
one, not both.

Goal | Use | Notes
--- | --- | ---
Events → convolved regressors | `DesignMatrix(events_tsv, run_length=, TR=)` | `hrf_model='glover'` by default; `None` for boxcars
Convolve later | [`convolve`](../api/data/design_matrix.md#nltools.data.designmatrix.DesignMatrix.convolve) | `kernel='glover'` (or another nilearn HRF model name) or your own kernel array; skips confounds
Add nuisance columns | [`append`](../api/data/design_matrix.md#nltools.data.designmatrix.DesignMatrix.append)`(..., axis=1, as_confounds=True)` | `axis=0` stacks runs and keeps confounds separate per run
Drift | [`add_poly`](../api/data/design_matrix.md#nltools.data.designmatrix.DesignMatrix.add_poly) / [`add_dct_basis`](../api/data/design_matrix.md#nltools.data.designmatrix.DesignMatrix.add_dct_basis) | Generated names carry the reserved `.nl_` prefix
Check the design | [`vif`](../api/data/design_matrix.md#nltools.data.designmatrix.DesignMatrix.vif), [`corr`](../api/data/design_matrix.md#nltools.data.designmatrix.DesignMatrix.corr), [`plot`](../api/data/design_matrix.md#nltools.data.designmatrix.DesignMatrix.plot) | `vif` excludes confounds by default
Drop redundant columns | [`clean`](../api/data/design_matrix.md#nltools.data.designmatrix.DesignMatrix.clean) | Removes columns correlated above `thresh` (default `0.95`)
Fit a first-level model | [`BrainData.fit`](../api/data/brain_data.md#nltools.data.braindata.BrainData.fit)`(model='glm', X=design)` | `model='ridge'` for the penalized fit; see [Prediction](prediction.md)
Contrasts | [`compute_contrasts`](../api/data/brain_data.md#nltools.data.braindata.BrainData.compute_contrasts)`("A - B", inference=)` | Effect map by default; `inference=True` returns a `ContrastResult`
Group test | [`concatenate`](../api/tasks/loading.md#nltools.data.combine.concatenate) → [`ttest`](../api/data/brain_data.md#nltools.data.braindata.BrainData.ttest) | Returns `{'mean', 't', 'z', 'p'}` of `BrainData`
Correct and threshold | [`fdr`](../api/tasks/inference.md#nltools.algorithms.fdr), [`holm_bonf`](../api/tasks/inference.md#nltools.algorithms.holm_bonf), [`threshold`](../api/tasks/inference.md#nltools.algorithms.threshold) | See [Statistics & inference](statistics-and-inference.md)

## First level

```python
from nltools.data import BrainData, DesignMatrix

bold = BrainData("sub-01_bold.nii.gz")
events = DesignMatrix("sub-01_events.tsv", run_length=bold.shape[0], TR=2.0)
confounds = DesignMatrix("sub-01_confounds.tsv", run_length="infer", TR=2.0)

design = events.append(confounds, axis=1, as_confounds=True).add_poly(2)
design.vif()                 # one VIF per non-confound regressor

bold.fit(X=design)
effect = bold.compute_contrasts("face_c0 - house_c0")
con = bold.compute_contrasts("face_c0 - house_c0", inference=True)
```

`fit` keeps the fitted model on the object, and the model remembers its column names, so
`compute_contrasts` can name columns directly. A contrast is a string of column names with
optional coefficients (`"2*A - B - C"`), a numeric weight vector, or a `{name: contrast}` dict to
evaluate several at once. The default returns the effect map — the input a group analysis
consumes. `inference=True` returns a `ContrastResult` carrying `effect`, `variance`,
`standard_error`, `statistic`, `z_score`, `p_value`, and `degrees_of_freedom` together, so you can
threshold `con.statistic` now and reuse `con.effect` at the group level. Contrast p-values are
one-sided, following the nilearn/SPM convention that a contrast tests "A > B"; flip the contrast
for the other direction. This is the one place in nltools where the default is not two-tailed.

For a multi-regressor group model, stack the per-subject effect maps and fit a second-level
`DesignMatrix` with one row per subject, then contrast its coefficients:

```python
group = concatenate(effects)                                   # (n_subjects, n_voxels)
second_level = DesignMatrix({"intercept": np.ones(len(group)), "age": ages})
group.fit(model="glm", X=second_level)
result = group.compute_contrasts("age", inference=True)
```

This estimates variance *across* effect maps; it does not propagate first-level effect variance.
`ttest()` remains the concise intercept-only version of the same test.

## Group level

```python
from nltools import concatenate
from nltools.algorithms import fdr, threshold

group = concatenate(betas)                                # (n_subjects, n_voxels)
result = group.ttest()                                    # mean, t, z, p
group_z = threshold(result["z"], result["p"], thr=0.001)  # uncorrected
fdr_thr = fdr(result["p"].data, q=0.05)                   # -1 if nothing survives
```

## Design warnings

One warning describes your design, not a bug in nltools. Read it instead of filtering it.

- [`RankDeficientDesignWarning`](../api/tasks/design-and-glm.md#nltools.data.braindata.modeling.RankDeficientDesignWarning).
  A column is an exact linear combination of others (a duplicated regressor, an intercept added
  twice, a condition that never occurs in this run). The fit uses a pseudo-inverse, so contrasts
  involving that column are not interpretable. Fix the design; `clean()` handles the common case.

It subclasses `UserWarning`, so you can silence it individually. Do that only once you know
which regressors triggered it.

Next: [Statistics & inference](statistics-and-inference.md), or the
[GLM workflow tutorial](../tutorials/workflows/01_glm.md) for the whole pipeline end to end.
