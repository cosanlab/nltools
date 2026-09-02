---
title: Design matrices & GLM
---

A [`DesignMatrix`](../api/data/design_matrix.md) is one row per timepoint, one column per
regressor, plus the metadata a GLM needs: the sampling frequency, which columns are already
HRF-convolved, and which are nuisance. Build one from a BIDS events `.tsv`, a confounds table, a
numpy array, or a dict. An events file is detected by its `onset` and `duration` columns: each
`trial_type` becomes a boxcar named `<type>_c0` and is convolved with the Glover HRF for you.
Pass `hrf_model=None` to keep raw boxcars and call [`convolve`](../api/data/design_matrix.md#data-design-matrix-convolve)
yourself.

Nuisance regressors go in through `append(..., axis=1, as_confounds=True)`, which marks them so
`convolve` skips them and `vif` ignores them. Add low-frequency drift with
[`add_poly`](../api/data/design_matrix.md#data-design-matrix-add-poly) (polynomials) or
[`add_dct_basis`](../api/data/design_matrix.md#data-design-matrix-add-dct-basis) (cosines). Pick
one, not both.

Goal | Use | Notes
--- | --- | ---
Events → convolved regressors | `DesignMatrix(events_tsv, run_length=, TR=)` | `hrf_model='glover'` by default; `None` for boxcars
Convolve later | [`convolve`](../api/data/design_matrix.md#data-design-matrix-convolve) | `conv_func='hrf'` or your own kernel array; skips confounds
Add nuisance columns | [`append`](../api/data/design_matrix.md#data-design-matrix-append)`(..., axis=1, as_confounds=True)` | `axis=0` stacks runs and keeps confounds separate per run
Drift | [`add_poly`](../api/data/design_matrix.md#data-design-matrix-add-poly) / [`add_dct_basis`](../api/data/design_matrix.md#data-design-matrix-add-dct-basis) | Generated names carry the reserved `.nl_` prefix
Check the design | [`vif`](../api/data/design_matrix.md#data-design-matrix-vif), [`corr`](../api/data/design_matrix.md#data-design-matrix-corr), [`plot`](../api/data/design_matrix.md#data-design-matrix-plot) | `vif` excludes confounds by default
Drop redundant columns | [`clean`](../api/data/design_matrix.md#data-design-matrix-clean) | Removes columns correlated above `thresh` (default `0.95`)
Fit a first-level model | [`BrainData.fit`](../api/data/brain_data.md#data-brain-data-fit)`(model='glm', X=design)` | `model='ridge'` for the penalized fit; see [Prediction](prediction.md)
Contrasts | [`compute_contrasts`](../api/data/brain_data.md#data-brain-data-compute-contrasts)`("A - B", statistic=)` | `statistic='t'` (default), `'z'`, `'p'`, `'beta'`, or `'all'`
Group test | [`concatenate`](../api/tasks/loading.md#tasks-loading-concatenate) → [`ttest`](../api/data/brain_data.md#data-brain-data-ttest) | Returns `{'mean', 't', 'z', 'p'}` of `BrainData`
Correct and threshold | [`fdr`](../api/tasks/inference.md#tasks-inference-fdr), [`holm_bonf`](../api/tasks/inference.md#tasks-inference-holm-bonf), [`threshold`](../api/tasks/inference.md#tasks-inference-threshold) | See [Statistics & inference](statistics-and-inference.md)

## First level

```python
from nltools.data import BrainData, DesignMatrix

bold = BrainData("sub-01_bold.nii.gz")
events = DesignMatrix("sub-01_events.tsv", run_length=bold.shape[0], TR=2.0)
confounds = DesignMatrix("sub-01_confounds.tsv", run_length="infer", TR=2.0)

design = events.append(confounds, axis=1, as_confounds=True).add_poly(2)
design.vif()                 # one VIF per non-confound regressor

bold.fit(X=design)
con = bold.compute_contrasts("face_c0 - house_c0", statistic="all")
```

`fit` attaches the design to the object, so `compute_contrasts` can name columns directly. A
contrast is a string of column names with optional coefficients (`"2*A - B - C"`), a numeric weight
vector, or a `{name: contrast}` dict to evaluate several at once. `statistic='all'` returns
`beta`, `t`, `z`, `p`, and `se` from one fit, so you can threshold the `t` map now and reuse the
`beta` map at the group level. Contrast p-values are one-sided, following the nilearn/SPM
convention that a contrast tests "A > B"; flip the contrast for the other direction. This is the
one place in nltools where the default is not two-tailed.

## Group level

```python
from nltools.utils import concatenate
from nltools.algorithms import fdr, threshold

group = concatenate(betas)                                # (n_subjects, n_voxels)
result = group.ttest()                                    # mean, t, z, p
group_z = threshold(result["z"], result["p"], thr=0.001)  # uncorrected
fdr_thr = fdr(result["p"].data, q=0.05)                   # -1 if nothing survives
```

## Design warnings

Two warnings describe your design, not a bug in nltools. Read them instead of filtering them.

- [`NearCollinearDesignWarning`](../api/tasks/design-and-glm.md#tasks-design-and-glm-nearcollineardesignwarning).
  The design is full rank but two or more regressors are nearly the same. Betas will be unstable
  and their signs can flip. Usually you are modeling the same variance twice: detrend motion
  regressors before adding polynomial drift, or drop one of the pair.
- [`RankDeficientDesignWarning`](../api/tasks/design-and-glm.md#tasks-design-and-glm-rankdeficientdesignwarning).
  A column is an exact linear combination of others (a duplicated regressor, an intercept added
  twice, a condition that never occurs in this run). The fit uses a pseudo-inverse, so contrasts
  involving that column are not interpretable. Fix the design; `clean()` handles the common case.

Both subclass `UserWarning`, so you can silence them individually. Do that only once you know
which regressors triggered them.

Next: [Statistics & inference](statistics-and-inference.md), or the
[GLM workflow tutorial](../tutorials/workflows/01_glm.md) for the whole pipeline end to end.
