---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
execute:
  allow_errors: false
---

# First-Level and Group GLM Analysis

## Introduction

In the previous two tutorials you saw the two core data classes on their own: `BrainData` holds the imaging timeseries (**y**) and `DesignMatrix` holds your experimental design (**X**). This tutorial brings them together in a classic first-level GLM, then scales that same flow up to a 10-subject group analysis — all on one real BIDS dataset.

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$

`BrainData.fit(model="glm", X=dm)` estimates **β** at every voxel, `BrainData.compute_contrasts()` tests linear combinations of those betas, and `BrainData.ttest()` takes the per-subject effect-size maps and runs a second-level one-sample test across subjects.

:::{tip}
For a refresher on either class on its own, see the [BrainData basics](../basics/01_brain_data.md) and [DesignMatrix basics](../basics/02_design_matrix.md) tutorials.
:::

```{code-cell} python3
import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nilearn.datasets import fetch_language_localizer_demo_dataset

from nltools.data import BrainData
from nltools.io import onsets_to_dm
from nltools.stats import fdr, threshold
from nltools.utils import concatenate
```

## Loading Data

We'll use the **language localizer demo** dataset from `nilearn` — 10 subjects watching blocks of sentences (`language`) and blocks of consonant strings (`string`), ~750 MB, BIDS-organized, already preprocessed and normalized to MNI space. First-time download takes a minute or two and is then cached in `~/nilearn_data/`.

```{code-cell} python3
dataset = fetch_language_localizer_demo_dataset(verbose=0)
root = Path(dataset.data_dir)

subjects = sorted(p.name for p in root.glob("sub-*") if p.is_dir())
print(f"Found {len(subjects)} subjects: {subjects}")
```

Each subject's functional derivative lives at `derivatives/sub-XX/func/` and has three useful files: the preprocessed BOLD, a sidecar JSON (for TR), and a confounds TSV with six motion regressors. The events TSV sits in `sub-XX/func/`.

```{code-cell} python3
def subject_files(sub: str) -> dict:
    """Resolve paths for one subject's BOLD, events, confounds, and TR."""
    deriv = root / "derivatives" / sub / "func"
    bold = deriv / f"{sub}_task-languagelocalizer_desc-preproc_bold.nii.gz"
    sidecar = deriv / f"{sub}_task-languagelocalizer_desc-preproc_bold.json"
    return {
        "bold":      bold,
        "events":    root / sub / "func" / f"{sub}_task-languagelocalizer_events.tsv",
        "confounds": deriv / f"{sub}_task-languagelocalizer_desc-confounds_regressors.tsv",
        "TR":        json.loads(sidecar.read_text())["RepetitionTime"],
    }

paths = subject_files("sub-01")
paths
```

## Step 1: Single-subject first-level GLM

### Load the BOLD

The preprocessed BOLD is already on an MNI 3 mm grid, so `BrainData` loads it directly with no resampling.

```{code-cell} python3
data = BrainData(str(paths["bold"]), verbose=False)
print(f"BrainData: {data.shape}  (timepoints x voxels)")
print(f"TR:        {paths['TR']}s")
```

### Build the DesignMatrix

The events TSV is BIDS-style (`onset`, `duration`, `trial_type`). `onsets_to_dm` convolves each condition with the HRF and returns a regressor matrix aligned to the BOLD timeseries:

```{code-cell} python3
events = pd.read_csv(paths["events"], sep="\t")
events.head()
```

```{code-cell} python3
dm = onsets_to_dm(
    timings=events,
    run_length=data.shape[0],
    TR=paths["TR"],
    hrf_model="glover",
)
list(dm.columns)
```

Next, pull in the six motion confounds from `confounds.tsv` and stack them on as nuisance columns. `DesignMatrix.append(df, axis=1)` accepts a pandas DataFrame directly and marks the new columns as nuisance, so a multi-run vertical append would keep them separated per run:

```{code-cell} python3
confounds = pd.read_csv(paths["confounds"], sep="\t")
dm = dm.append(confounds, axis=1)
list(dm.columns)
```

Finally add a DCT high-pass filter and polynomial drift terms:

```{code-cell} python3
dm = dm.add_dct_basis(duration=128).add_poly(order=2, include_lower=True)
print(f"Full design matrix: {dm.shape}  (timepoints x regressors)")
dm.plot(figsize=(10, 6));
```

The block-like DCT columns and smooth polynomial columns on the right soak up scanner drift so it doesn't get mis-attributed to the task conditions.

### Fit the GLM

```{code-cell} python3
data.fit(model="glm", X=dm)
print(f"glm_betas: {data.glm_betas.shape}   (regressors x voxels)")
print(f"glm_t:     {data.glm_t.shape}")
print(f"glm_p:     {data.glm_p.shape}")
```

Each `.glm_*` attribute holds one map per regressor. `glm_betas[i]` is the effect-size map and `glm_t[i]` is the marginal t-map for regressor `i`. For a contrast *across* regressors you need `compute_contrasts` — the per-regressor maps cannot be combined by hand because t-statistic arithmetic requires the full parameter covariance.

### Compute a contrast

`compute_contrasts(..., contrast_type="all")` returns the effect size, t, z, p, and SE maps for one contrast in a single call — so group code can grab the effect size while thresholding code grabs the t-map, no double work.

```{code-cell} python3
result = data.compute_contrasts(
    "language - string",
    contrast_type="all",
)
print("Keys:", sorted(result.keys()))
```

Each entry is a `BrainData`. Threshold the t-map at a conventional `|t| > 3.09` (two-tailed p ≈ 0.001 for this df):

```{code-cell} python3
result["t"].plot(
    title="sub-01: language > string (t)",
    threshold=3.09,
);
```

Even at one subject the left-lateralized fronto-temporal language network is visible. For group-level input we'll use `result["beta"]` instead — effect-size maps, not t-maps, are the right inputs to a second-level one-sample test (see the note at the end of this section).

## Step 2: Scale up to a group analysis

### First-level loop over all 10 subjects

The same fitting recipe applies to every subject. We collect one effect-size map per subject:

```{code-cell} python3
def fit_first_level(sub: str, contrast: str = "language - string") -> BrainData:
    """Load sub's BOLD, build the DM, fit GLM, return the effect-size map."""
    p = subject_files(sub)
    bd = BrainData(str(p["bold"]), verbose=False)
    dm = onsets_to_dm(
        timings=pd.read_csv(p["events"], sep="\t"),
        run_length=bd.shape[0],
        TR=p["TR"],
        hrf_model="glover",
    )
    dm = dm.append(pd.read_csv(p["confounds"], sep="\t"), axis=1)
    dm = dm.add_dct_basis(duration=128).add_poly(order=2, include_lower=True)
    bd.fit(model="glm", X=dm)
    return bd.compute_contrasts(contrast, contrast_type="beta")

effect_maps = [fit_first_level(sub) for sub in subjects]
print(f"Collected {len(effect_maps)} first-level effect-size maps")
print(f"Each map: {effect_maps[0].shape}")
```

### Stack into one group-level `BrainData`

`concatenate` stacks a list of single-image `BrainData` into an (n_subjects, n_voxels) array, preserving the shared mask:

```{code-cell} python3
group = concatenate(effect_maps)
print(f"Group stack: {group.shape}  (subjects x voxels)")
```

### One-sample t-test across subjects

`BrainData.ttest` returns the full bundle — the effect-size (mean), the parametric t, a signed z-score (computed from the p via `sign(t) * norm.isf(p/2)`, matching nilearn's convention), and the p-value:

```{code-cell} python3
group_result = group.ttest()
print("Keys:", sorted(group_result.keys()))
```

At 10 subjects our df is small (9), so t-tails are heavier than normal and `z` is the better scale for fixed thresholds. Show the group z-map thresholded at `|z| > 3.09` (uncorrected p ≈ 0.001):

```{code-cell} python3
group_result["z"].plot(
    title="Group: language > string (z, unc |z| > 3.09)",
    threshold=3.09,
);
```

This is the left-lateralized fronto-temporal language network reported in the original nilearn BIDS analysis — a nice sanity check that nltools's first-level + group path lines up with the canonical result.

And the mean effect-size map shows the *magnitude* of the language-vs-string response (in whatever units the fit used), independent of inference:

```{code-cell} python3
group_result["mean"].plot(title="Group mean effect size: language - string");
```

### Multiple comparisons correction

A voxel-wise threshold at unc p < 0.001 doesn't correct for the ~70k voxels we just tested. `nltools.stats.fdr` finds the p-threshold controlling the false-discovery rate, and `threshold` applies it to a stat map:

```{code-cell} python3
p_arr = np.asarray(group_result["p"].data)
fdr_thr = fdr(p_arr, q=0.05)
print(f"FDR p-threshold (q=0.05): {fdr_thr:.4g}")
n_sig = int((p_arr < fdr_thr).sum()) if fdr_thr > 0 else 0
print(f"Voxels surviving FDR: {n_sig} / {p_arr.size}")
```

```{code-cell} python3
if fdr_thr > 0:
    z_fdr = threshold(group_result["z"], group_result["p"], thr=fdr_thr)
    z_fdr.plot(title=f"Group: language > string (FDR q=0.05)");
else:
    print("No voxels survive FDR at q=0.05 — showing unthresholded z-map.")
    group_result["z"].plot(title="Group: language > string (unthresholded z)");
```

Bonferroni for reference — one of the most conservative corrections, thresholding at α / n_voxels:

```{code-cell} python3
bonf_thr = 0.05 / p_arr.size
n_bonf = int((p_arr < bonf_thr).sum())
print(f"Bonferroni p-threshold: {bonf_thr:.2e}")
print(f"Voxels surviving Bonferroni: {n_bonf}")
```

## Putting it all together

The complete flow in compact form:

```{code-cell} python3
# 1. Fetch + resolve paths
dataset = fetch_language_localizer_demo_dataset(verbose=0)
root = Path(dataset.data_dir)
subjects = sorted(p.name for p in root.glob("sub-*") if p.is_dir())

# 2. First-level per subject → effect-size map
effect_maps = [fit_first_level(sub, "language - string") for sub in subjects]

# 3. Stack and run group one-sample test
group = concatenate(effect_maps)
group_result = group.ttest()

# 4. Threshold and plot
group_result["z"].plot(threshold=3.09)
```

| Stage | What it does | Key API |
|---|---|---|
| Load data | Fetch BIDS dataset + per-subject paths | `fetch_language_localizer_demo_dataset()` |
| Build design | Events → HRF-convolved regressors + confounds + drift | `onsets_to_dm`, `dm.append(df, axis=1)`, `dm.add_dct_basis()`, `dm.add_poly()` |
| Fit first level | OLS at every voxel | `data.fit(model="glm", X=dm)` |
| Contrast | Linear combination of β with full covariance | `data.compute_contrasts("A - B", contrast_type="all")` |
| Stack subjects | Concatenate first-level effect-size maps | `concatenate([...])` |
| Group test | Voxel-wise one-sample t-test returning `{mean, t, z, p}` | `group.ttest()` |
| MC correction | FDR / Bonferroni thresholding | `nltools.stats.fdr`, `threshold` |

:::{note}
**Why feed *effect sizes* into the group test, not t-stats?** A first-level t-statistic is `β / SE(β)`, and SE differs across subjects for reasons that aren't about the effect of interest (scan length, motion, noise floor). Running a group one-sample test on effect-size maps keeps the group inference about the effect; stacking t-maps conflates effect magnitude with first-level precision.
:::

## Next Steps

- **Two-stage GLM**: per-run first-level + across-run second-level within a subject — see [Two-Stage GLM](07_two_stage_glm.md)
- **Decoding**: ask "can the brain tell these conditions apart?" rather than "where in the brain?" — see [Decoding](05_decoding.md)
- **RSA**: characterize the *structure* of neural patterns across conditions — see [RSA](04_rsa.md)
