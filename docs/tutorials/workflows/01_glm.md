---
# AUTO-GENERATED from 01_glm.py by scripts/marimo_to_myst.py — DO NOT EDIT.
# Edit the marimo notebook, then run `uv run poe tutorials-build`.
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# GLM Analysis

**What it answers.** *Where* in the brain does activity track your task design? The general linear model (GLM) is the mass-univariate workhorse of task fMRI: fit one regression per voxel, then test contrasts between conditions. Use it when you have a known design and want a statistical map of effects.

For the underlying theory, see the GLM chapters in [dartbrains](https://dartbrains.org). This tutorial is about *running* the analysis in nltools.
<!---->
**How it works.** A GLM analysis runs in two stages:

- **First level (single subject).** Regress each voxel's timeseries on the design matrix **X** → one β (and t) per regressor. A *contrast* is a linear combination of βs, giving a per-subject effect map.
- **Second level (group).** Stack the per-subject contrast **effect-size** maps and run a one-sample test across subjects.

Feed *effect sizes* (βs), not first-level t-maps, into the group test: a first-level t is `β / SE(β)`, and SE varies across subjects for reasons unrelated to the effect (scan length, motion). Stacking t-maps would conflate effect magnitude with first-level precision.

```{code-cell} python3
import numpy as np
from joblib import Memory

from nltools.data import BrainData, DesignMatrix
from nltools.stats import fdr, threshold
from nltools.utils import concatenate

# Memoize per-subject fits to disk (.cache/ is git-ignored) so re-running
# the notebook reloads results instead of refitting every voxel.
memory = Memory(".cache/tutorials", verbose=0)
```

## How to do it

We use the **language localizer demo** from `nilearn` — 10 subjects viewing blocks of sentences (`language`) vs. consonant strings (`string`). Each subject's BIDS derivatives give us three files: the preprocessed BOLD, an events TSV, and a confounds TSV.

```{code-cell} python3
import json
from pathlib import Path

from nilearn.datasets import fetch_language_localizer_demo_dataset
from nilearn.interfaces.bids import get_bids_files

DATASET = fetch_language_localizer_demo_dataset(verbose=0)
DATA_DIR = Path(DATASET["data_dir"])

def get_sub_files(sub: str) -> dict:
    """Resolve one subject's BOLD, events, confounds, and TR from BIDS."""
    derivatives = DATA_DIR / "derivatives"
    sidecar = get_bids_files(derivatives, file_tag="bold", file_type="json", sub_label=sub)[0]
    return {
        "bold": get_bids_files(derivatives, file_tag="bold", file_type="nii.gz", sub_label=sub)[0],
        "events": get_bids_files(DATA_DIR, file_tag="events", file_type="tsv", sub_label=sub)[0],
        "confounds": get_bids_files(derivatives, file_type="tsv", modality_folder="func", sub_label=sub)[0],
        "TR": json.loads(Path(sidecar).read_text())["RepetitionTime"],
    }
```

### First level (single subject)

The recipe for one subject: load the BOLD (`BrainData` resamples to standard MNI automatically), build the design, and fit. Building a `DesignMatrix` from a BIDS events file creates boxcar regressors and **convolves them with the canonical (Glover) HRF for you** — columns come back as `language_c0` / `string_c0` (pass `hrf_model=None` for raw boxcars to `.convolve()` yourself). We append the motion confounds as nuisance columns and add polynomial drift. Wrapping it in `memory.cache` means each subject is fit once, then reloaded from disk.

```{code-cell} python3
@memory.cache
def first_level(sub: str, contrast: str = "language_c0 - string_c0"):
    """Fit one subject's GLM; return its design and the contrast bundle.

    We return only the lightweight design and contrast maps (not the
    fitted model, which carries residuals and a copy of the data) so the
    on-disk cache stays small.
    """
    f = get_sub_files(sub)
    brain = BrainData(f["bold"])
    events = DesignMatrix(f["events"], run_length=brain.shape[0], TR=f["TR"])
    confounds = DesignMatrix(f["confounds"], run_length="infer", TR=f["TR"])
    brain.fit(X=events.append(confounds, axis=1, as_confounds=True).add_poly(2))
    return brain.design_matrix, brain.compute_contrasts(contrast, contrast_type="all")
```

```{code-cell} python3
design, contrasts = first_level("01")
design.plot()  # the design we just fit
```

The helper returns the `language > string` contrast as a bundle — `beta`, `t`, `z`, `p`, `se` — computed in one call with `contrast_type="all"`, so we can threshold the t-map here *and* reuse the β map for the group analysis below.

```{code-cell} python3
contrasts["t"].plot(method="slices", threshold=3.09, title="sub-01: language > string (t)")
```

Even at one subject the left-lateralized fronto-temporal language network is visible (`|t| > 3.09`, two-tailed p ≈ 0.001).

### Second level (group)

The same cached recipe runs per subject, returning one **effect-size** (β) map each. We loop over eight of the ten demo subjects.

```{code-cell} python3
SUBJECTS = ["01", "02", "03", "04", "05", "06", "07", "08"]
beta_maps = []
for sub in SUBJECTS:
    _, sub_contrasts = first_level(sub)
    beta_maps.append(sub_contrasts["beta"])
```

`concatenate` stacks the per-subject maps into one `(n_subjects, n_voxels)` `BrainData`. `BrainData.ttest` runs a voxelwise one-sample test, returning the effect-size `mean`, the parametric `t`, a signed `z`, and `p`. `nltools.stats.threshold` keeps the `z` values whose `p` clears a cutoff — here voxelwise `p < 0.001`.

```{code-cell} python3
group = concatenate(beta_maps)
group_result = group.ttest()
group_z = threshold(group_result["z"], group_result["p"], thr=0.001)
group_z.plot(method="slices", title="Group: language > string (voxelwise p < 0.001)")
```

### Multiple-comparisons correction

That `p < 0.001` map is *uncorrected* — it ignores that we ran tens of thousands of tests. `nltools.stats.fdr` returns the p-threshold controlling the false-discovery rate. Whole-brain correction is stringent: on a ten-subject demo, far fewer voxels survive than at the uncorrected threshold — exactly the inflation that correction guards against. Restricting the search to an ROI (see the [MVPA tutorial](03_mvpa.md)) recovers power.

```{code-cell} python3
p_values = np.asarray(group_result["p"].data)
n_voxels = p_values.size
fdr_thr = fdr(p_values, q=0.05)
bonf_thr = 0.05 / n_voxels

n_uncorrected = int((p_values < 0.001).sum())
n_fdr = int((p_values <= fdr_thr).sum()) if fdr_thr > 0 else 0
n_bonferroni = int((p_values < bonf_thr).sum())

print(f"voxels surviving, out of {n_voxels}:")
print(f"  uncorrected (p < 0.001):  {n_uncorrected:5d}")
print(f"  FDR (q = 0.05):           {n_fdr:5d}")
print(f"  Bonferroni (p < 0.05/N):  {n_bonferroni:5d}")
```

## Recap

| Stage | What it does | Key API |
|---|---|---|
| Build design | BIDS events → HRF-convolved regressors + confounds + drift | `DesignMatrix(events, run_length=, TR=)`, `.append(confounds, axis=1, as_confounds=True)`, `.add_poly()` |
| First level | OLS at every voxel | `brain.fit(X=design)` |
| Contrast | Linear combination of βs (effect size + inference) | `brain.compute_contrasts("A - B", contrast_type="all")` |
| Stack subjects | Concatenate first-level β maps | `concatenate([...])` |
| Group test | Voxelwise one-sample t-test → `{mean, t, z, p}` | `group.ttest()` |
| Correction | FDR threshold | `nltools.stats.fdr`, `nltools.stats.threshold` |

The per-subject loop is the explicit path; `BrainCollection` will wrap multi-subject fitting into a single call once it lands on this branch.

**Next steps**

- [Encoding models](02_encoding.md) — predict brain activity *from* stimulus features (GLM vs. Ridge).
- [Multivariate pattern analysis](03_mvpa.md) — decode conditions and compare representational geometry.
- [Inter-subject correlation](04_isc.md) — shared responses to naturalistic stimuli.
