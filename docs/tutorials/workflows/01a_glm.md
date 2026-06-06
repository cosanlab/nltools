---
file_format: mystnb
authors:
  - Eshin Jolly
kernelspec:
  name: python3
  display_name: Python 3
execute:
  allow_errors: false
---

```{code-cell} python
:tags: [remove-cell]
import matplotlib.pyplot as plt
```

# GLM Analysis

## Introduction

In the previous two tutorials you saw the two core data classes on their own: `BrainData` holds the imaging timeseries (**y**) and `DesignMatrix` holds your experimental design (**X**). This tutorial brings them together in a classic GLM analysis. Conventionally, we proceed in two stages:

- **first-level analysis:** we take a single subject from preprocessed fMRI time-series -> statistical maps (e.g. betas)
- **second-level analysis:** we first-level statistical maps from *multiple* subjects -> group map (e.g. thresholded t-stats)

```{code-cell} python
from nltools.data import BrainData, DesignMatrix
```

In this tutorial, we'll use the **language localizer demo** dataset from `nilearn` — 10 subjects watching blocks of sentences (`language`) and blocks of consonant strings (`string`).

```{code-cell} python
from nilearn.datasets import fetch_language_localizer_demo_dataset
from nilearn.interfaces.bids import get_bids_files

DATASET = fetch_language_localizer_demo_dataset(verbose=0)
print(DATASET.description)
```

## Single Subject Analysis (first-level)

Each subject's functional derivative lives at `derivatives/sub-XX/func/` and has 3 files we need:

1. preprocessed BOLD timeseries
2. events TSV with the task onsets
3. confounds TSV with six motion regressors

We can use `get_bids_files()` from `nilearn` and create quick helper function:

```{code-cell} python
def get_sub_files(sub: str) -> dict:
    """Return one subject's BOLD, events, confounds, and TR from BIDS."""
    import json
    from pathlib import Path
    from nilearn.interfaces.bids import get_bids_files

    datapath = Path(DATASET["data_dir"])
    events = get_bids_files(datapath, file_tag="events", file_type="tsv", sub_label=sub)[0]
    bold = get_bids_files(
        datapath / "derivatives", file_tag="bold", file_type="nii.gz", sub_label=sub
    )[0]
    sidecar = get_bids_files(
        datapath / "derivatives", file_tag="bold", file_type="json", sub_label=sub
    )[0]
    _confounds = get_bids_files(
        datapath / "derivatives", file_type="tsv", modality_folder="func", sub_label=sub
    )[0]
    return {
        "bold": bold,
        "events": events,
        "confounds": _confounds,
        "TR": json.loads(Path(sidecar).read_text())["RepetitionTime"],
    }
```

## Step 1: Single-subject first-level GLM

### Load the timeseries

We can pass the filepath directly to `BrainData` which will **automatically resample** to the closest mm MNI template space (e.g. 3mm in this case):

```{code-cell} python
s1 = get_sub_files("01")

brain = BrainData(s1["bold"])
brain
```

```{code-cell} python
brain.plot()
```

### Prepare the design matrix

We can pass any BIDS-style events file (with `onset`, `duration`, `trial_type` columns). `DesignMatrix` builds boxcar regressors aligned to the BOLD timeseries — convolution is an explicit step you call later:

```{code-cell} python
designmat = DesignMatrix(s1["events"], run_length=brain.shape[0], TR=s1["TR"])
designmat.plot()
```

Next, pull in the motion confounds from `confounds.tsv` and *append* them as *nuisance* columns. Confounds files already have one row per TR, so `run_length='infer'` accepts whatever the file contains:

```{code-cell} python
_confounds = DesignMatrix(s1["confounds"], run_length="infer", TR=s1["TR"])
designmat_1 = designmat.append(_confounds, axis=1, as_confounds=True)
designmat_1.plot()
```

Next, lets add some polynomial drift terms:

```{code-cell} python
designmat_2 = designmat_1.add_poly(2)
designmat_2.plot()
```

And finally perform convolution on our events (language & string):

```{code-cell} python
designmat_3 = designmat_2.convolve()
designmat_3.plot()
```

In practice we can use *method-chaining* to do this more succinctly:

```{code-cell} python
events = DesignMatrix(s1["events"], run_length=brain.shape[0], TR=s1["TR"])
_confounds = DesignMatrix(s1["confounds"], run_length="infer", TR=s1["TR"])
designmat_4 = events.append(_confounds, axis=1, as_confounds=True).add_poly(2).convolve()
```

### Fit the GLM

We can estimate the model by passing the `DesignMatrix` to the `.fit()` method of `BrainData`. This will create several `.glm_*` attributes that hold the results

```{code-cell} python
brain.fit(X=designmat_4)
# beta-maps
brain.glm_betas
```

```{code-cell} python
brain.glm_betas
```

```{code-cell} python
# Beta-map for "language"
brain.glm_betas[0].plot()
```

Each `.glm_*` attribute holds one map per regressor. `glm_betas[i]` is the effect-size map and `glm_t[i]` is the marginal t-map for regressor `i`.

### Compute a contrast

`compute_contrasts(..., contrast_type="all")` returns the effect size, t, z, p, and SE maps for one contrast in a single call — so group code can grab the effect size while thresholding code grabs the t-map, no double work.

Note that `.convolve()` renames each column to `<col>_c0` (and `_c1`, `_c2`, … if you pass a multi-kernel basis), so the contrast string references the post-convolution names:

```{code-cell} python
result = brain.compute_contrasts(
    "language_c0 - string_c0",
    contrast_type="all",
)
print("Keys:", sorted(result.keys()))
```

Each entry is a `BrainData`. Threshold the t-map at a conventional `|t| > 3.09` (two-tailed p ≈ 0.001 for this df):

```{code-cell} python
result["t"].plot(
    title="sub-01: language > string (t)",
    threshold=3.09,
)
```

Even at one subject the left-lateralized fronto-temporal language network is visible. For group-level input we'll use `result["beta"]` instead — effect-size maps, not t-maps, are the right inputs to a second-level one-sample test (see the note at the end of this section).

## Step 2: Scale up to a group analysis

`nltools` offers 2 ways work with *multiple* subject/run data. You can always just write a python function and a `for` loop:

### Manual: First-level loop over all 10 subjects

The same fitting recipe applies to every subject. We collect one effect-size map per subject:

```python
def fit_first_level(sub: str, contrast: str = "language_c0 - string_c0") -> BrainData:
    """Load sub's BOLD, build the DM, fit GLM, return the effect-size map."""
    p = get_sub_files(sub)
    bd = BrainData(p["bold"])
    dm = DesignMatrix(p["events"], run_length=bd.shape[0], TR=p["TR"])
    _confounds = DesignMatrix(p["confounds"], run_length="infer", TR=p["TR"])
    dm = dm.append(_confounds, axis=1, as_confounds=True)
    dm = dm.add_dct_basis(duration=128).add_poly(order=2, include_lower=True)
    dm = dm.convolve()
    bd.fit(model="glm", X=dm)
    return bd.compute_contrasts(contrast, contrast_type="beta")
```

Then we can use `concatenate` to stack a list of single-image `BrainData` into into 1 `BrainData` (with n_subjects x n_voxels):

```python
effect_maps = [fit_first_level(sub) for sub in subjects]
group = concatenate(effect_maps)
```

And use `BrainData.ttest()` which returns — the effect-size (mean), the parametric t, a signed z-score (computed from the p via `sign(t) * norm.isf(p/2)`, matching nilearn's convention), and the p-value:

```python
group_result = group.ttest()
```

### Easier: `BrainCollection`

```{code-cell} python
# TODO: Add BrainCollection approach in this section

```

### Visualizing Results

At 10 subjects our df is small (9), so t-tails are heavier than normal and `z` is the better scale for fixed thresholds. Show the group z-map thresholded at `|z| > 3.09` (uncorrected p ≈ 0.001):

```{code-cell} python
# group_result["z"].plot(title="Group: language > string (z, unc |z| > 3.09)", threshold=3.09);
```

This is the left-lateralized fronto-temporal language network reported in the original nilearn BIDS analysis — a nice sanity check that nltools's first-level + group path lines up with the canonical result.

And the mean effect-size map shows the *magnitude* of the language-vs-string response (in whatever units the fit used), independent of inference:

```{code-cell} python
# group_result["mean"].plot(title="Group mean effect size: language - string");
```

### Multiple comparisons correction

A voxel-wise threshold at unc p < 0.001 doesn't correct for the ~70k voxels we just tested. `nltools.stats.fdr` finds the p-threshold controlling the false-discovery rate, and `threshold` applies it to a stat map:

```{code-cell} python
# p_arr = np.asarray(group_result["p"].data)
# fdr_thr = fdr(p_arr, q=0.05)
# print(f"FDR p-threshold (q=0.05): {fdr_thr:.4g}")
# n_sig = int((p_arr < fdr_thr).sum()) if fdr_thr > 0 else 0
# print(f"Voxels surviving FDR: {n_sig} / {p_arr.size}")
```

```{code-cell} python
# if fdr_thr > 0:
#     z_fdr = threshold(group_result["z"], group_result["p"], thr=fdr_thr)
#     z_fdr.plot(title=f"Group: language > string (FDR q=0.05)");
# else:
#     print("No voxels survive FDR at q=0.05 — showing unthresholded z-map.")
#     group_result["z"].plot(title="Group: language > string (unthresholded z)");
```

Bonferroni for reference — one of the most conservative corrections, thresholding at α / n_voxels:

```{code-cell} python
# bonf_thr = 0.05 / p_arr.size
# n_bonf = int((p_arr < bonf_thr).sum())
# print(f"Bonferroni p-threshold: {bonf_thr:.2e}")
# print(f"Voxels surviving Bonferroni: {n_bonf}")
```

## Putting it all together

The complete flow in compact form:

```{code-cell} python
# 1. Fetch + resolve paths
# dataset = fetch_language_localizer_demo_dataset(verbose=0)
# root = Path(dataset.data_dir)
# subjects = sorted(p.name for p in root.glob("sub-*") if p.is_dir())

# 2. First-level per subject → effect-size map
# effect_maps = [fit_first_level(sub, "language_c0 - string_c0") for sub in subjects]

# 3. Stack and run group one-sample test
# group = concatenate(effect_maps)
# group_result = group.ttest()

# 4. Threshold and plot
# group_result["z"].plot(threshold=3.09)
```

| Stage | What it does | Key API |
|---|---|---|
| Load data | Fetch BIDS dataset + per-subject paths | `fetch_language_localizer_demo_dataset()` |
| Build design | Events → boxcar regressors + confounds + drift, then convolve | `DesignMatrix(events_path, run_length=..., TR=...)`, `dm.append(confounds_dm, axis=1, as_confounds=True)`, `dm.add_dct_basis()`, `dm.add_poly()`, `dm.convolve()` |
| Fit first level | OLS at every voxel | `brain.fit(model="glm", X=dm)` |
| Contrast | Linear combination of β with full covariance | `brain.compute_contrasts("A - B", contrast_type="all")` |
| Stack subjects | Concatenate first-level effect-size maps | `concatenate([...])` |
| Group test | Voxel-wise one-sample t-test returning `{mean, t, z, p}` | `group.ttest()` |
| MC correction | FDR / Bonferroni thresholding | `nltools.stats.fdr`, `threshold` |

> **Note**
>
> **Why feed *effect sizes* into the group test, not t-stats?** A first-level t-statistic is `β / SE(β)`, and SE differs across subjects for reasons that aren't about the effect of interest (scan length, motion, noise floor). Running a group one-sample test on effect-size maps keeps the group inference about the effect; stacking t-maps conflates effect magnitude with first-level precision.
