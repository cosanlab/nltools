---
# AUTO-GENERATED from 04_isc.py by scripts/marimo_to_myst.py — DO NOT EDIT.
# Edit the marimo notebook, then run `uv run poe tutorials-build`.
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Inter-Subject Correlation

**What it answers.** Which brain regions respond *consistently across people* to a shared naturalistic stimulus (a movie, a story)? There's no explicit design matrix to model — instead, ISC uses other subjects' responses as the model, asking where the stimulus drives a common, time-locked signal.

For the theory, see the ISC material in [naturalistic-data](https://naturalistic-data.org). This tutorial runs one in nltools.
<!---->
**How it works.** ISC runs in two stages, like a GLM's first/second level:

- **Compute** the cross-subject similarity per region. Either **pairwise** (correlate every pair of subjects, then summarize) or **leave-one-out** (correlate each subject against the mean of the others). LOO is larger because each subject is compared to a denoised group average.
- **Group inference** on whether that similarity exceeds chance, via a permutation/bootstrap test that respects the temporal structure.

```{code-cell} python3
import warnings

import numpy as np
from joblib import Memory

from nltools.algorithms.inference.isc import isc_permutation_test
from nltools.data import BrainData
from nltools.mask import roi_to_brain_from_atlas
from nltools.templates import fetch_resource

memory = Memory(".cache/tutorials", verbose=0)
warnings.filterwarnings("ignore", message="Cannot detect name collisions")
```

## How to do it

We use nilearn's **development_fmri** dataset — children and adults watching the same short Pixar movie. For each subject we extract a region-mean timeseries with the bundled k50 atlas, giving one `(timepoints, regions)` array per subject; stacking them is the `(timepoints, subjects, regions)` input ISC expects. (In a full analysis you'd regress the provided confounds first.)

```{code-cell} python3
from nilearn.datasets import fetch_development_fmri

N_SUBJECTS = 12
DATA = fetch_development_fmri(n_subjects=N_SUBJECTS, verbose=0)
ATLAS = fetch_resource("masks/default/3mm-MNI152-2009fsl-k50.nii.gz")

@memory.cache
def region_timeseries(n_subjects):
    """Region-mean timeseries per subject (slow load; cached to disk)."""
    # extract_roi returns (n_regions, n_timepoints); transpose to (time, region).
    return [BrainData(DATA.func[i]).extract_roi(ATLAS).T for i in range(n_subjects)]

series = region_timeseries(N_SUBJECTS)
isc_data = np.stack(series, axis=1)  # (timepoints, subjects, regions)
print(f"ISC input: {isc_data.shape}  (timepoints, subjects, regions)")
```

### Compute ISC + group inference

`isc_permutation_test` does both stages in one call: it computes the per-region ISC (`summary_statistic="pairwise"`) and returns a permutation p-value per region.

```{code-cell} python3
pairwise = isc_permutation_test(
    isc_data,
    summary_statistic="pairwise",
    metric="median",
    n_permute=1000,
    random_state=0,
    progress_bar=False,
)
isc_values = np.asarray(pairwise["isc"])
p_values = np.asarray(pairwise["p"])
print(f"pairwise ISC — median {np.median(isc_values):.3f}, max {isc_values.max():.3f}")
print(f"regions significant (p < 0.05): {(p_values < 0.05).sum()} / {isc_values.size}")
```

Paint the per-region ISC back onto the brain with `roi_to_brain_from_atlas`. Sensory regions that track the movie's audio and visuals should show the highest synchrony:

```{code-cell} python3
from nilearn.image import math_img

brain_mask = math_img("img > 0", img=ATLAS)  # binary mask defining the output grid
isc_map = roi_to_brain_from_atlas(isc_values, atlas=ATLAS, source_mask=brain_mask)
isc_map.plot(
    method="slices", title="Inter-subject correlation (pairwise, per region)", cmap="hot", colorbar=True
)
```

### Pairwise vs. leave-one-out

The two summary statistics rank regions almost identically, but LOO values are systematically larger — each subject is compared against a less noisy group mean:

```{code-cell} python3
import matplotlib.pyplot as plt

loo = isc_permutation_test(
    isc_data,
    summary_statistic="leave-one-out",
    metric="median",
    n_permute=1000,
    random_state=0,
    progress_bar=False,
)
loo_values = np.asarray(loo["isc"])

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(isc_values, loo_values, alpha=0.7)
lims = [min(isc_values.min(), loo_values.min()), max(isc_values.max(), loo_values.max())]
ax.plot(lims, lims, "k--", linewidth=1, label="y = x")
ax.set_xlabel("pairwise ISC")
ax.set_ylabel("leave-one-out ISC")
ax.set_title("Pairwise vs. leave-one-out (per region)")
ax.legend()
fig
```

## Recap

| Stage | What it does | Key API |
|---|---|---|
| Region timeseries | Extract region means per subject, stack to `(time, subjects, regions)` | `BrainData(func).extract_roi(atlas).T` |
| Compute + test | Per-region ISC + permutation p-value | `isc_permutation_test(data, summary_statistic="pairwise", n_permute=)` |
| Leave-one-out | Each subject vs. the group mean | `summary_statistic="leave-one-out"` |
| Project to brain | Paint per-region values onto voxels | `roi_to_brain_from_atlas(values, atlas=, source_mask=)` |

**Next steps**

- [GLM analysis](01_glm.md) — model a known design instead of using subjects as each other's model.
- [Encoding models](02_encoding.md) — predict brain activity from explicit stimulus features.
- [Multivariate pattern analysis](03_mvpa.md) — decode conditions and compare representational geometry.
