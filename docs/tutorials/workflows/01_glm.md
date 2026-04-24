---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
execute:
  allow_errors: false
---

# First-Level GLM Analysis

## Introduction

In the previous two tutorials you saw the two core data classes on their own: `BrainData` holds the imaging timeseries (**y**) and `DesignMatrix` holds your experimental design (**X**). This tutorial brings them together in a classic first-level GLM:

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$

`BrainData.fit(model="glm", X=dm)` estimates **β** at every voxel simultaneously, and `BrainData.compute_contrasts()` lets you test linear combinations of those betas.

:::{tip}
If you'd like a refresher on either class on its own, see the [BrainData basics](../basics/01_brain_data.md) and [DesignMatrix basics](../basics/02_design_matrix.md) tutorials.
:::

```{code-cell} python3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nilearn.datasets import fetch_localizer_first_level

from nltools.data import BrainData, DesignMatrix
from nltools.io import onsets_to_dm
```

## Loading Data

We'll use the **localizer** dataset from `nilearn` — a single-subject, single-run fMRI experiment with ten conditions spanning motor, visual, auditory, language, and arithmetic tasks. The data are already preprocessed and normalized to MNI space, so we can load the BOLD volume directly into `BrainData` without any resampling.

```{code-cell} python3
localizer = fetch_localizer_first_level(verbose=0)

# 4D BOLD volume (already in MNI 3mm space)
data = BrainData(localizer.epi_img, verbose=False)

print(f"BrainData shape: {data.shape}  (timepoints x voxels)")
print(f"TR:              {localizer.t_r}s")
```

The events come as a BIDS-style `events.tsv` file with one row per trial — `onset`, `duration`, and `trial_type`. Here's what the start of the run looks like:

```{code-cell} python3
events = pd.read_csv(localizer.events, sep="\t")
events.head()
```

```{code-cell} python3
# Which conditions appear?
events["trial_type"].unique().tolist()
```

We turn those events into a `DesignMatrix` with `onsets_to_dm`, which convolves each condition's onsets with an HRF and returns a regressor matrix aligned to the BOLD timeseries:

```{code-cell} python3
dm = onsets_to_dm(
    timings=events,
    run_length=data.shape[0],
    TR=localizer.t_r,
    hrf_model="glover",
)

print(f"DesignMatrix shape: {dm.shape}   (timepoints x regressors)")
list(dm.columns)
```

## Inspecting the Design Matrix

Before fitting, it's always a good idea to visualize the design. Rows are timepoints, columns are regressors:

```{code-cell} python3
dm.plot(figsize=(10, 6));
```

Each column is a condition convolved with the HRF. The HRF accounts for the ~5 second delay between neural activity and the peak BOLD response — see the [DesignMatrix tutorial](../basics/02_design_matrix.md) for details on `.convolve()`.

## Adding Nuisance Regressors

Raw fMRI data contains low-frequency drift and other noise sources unrelated to the task. We model these with **nuisance regressors** so they don't contaminate the task estimates.

Two common approaches (often used together):

- **DCT basis functions** (`.add_dct_basis`): a discrete cosine high-pass filter. The `duration` sets the cutoff period in seconds — slower signals get filtered out.
- **Polynomial drift** (`.add_poly`): Legendre polynomials capturing slow trends. Order 0 is the intercept, 1 is linear drift, 2 is quadratic, etc.

```{code-cell} python3
dm_full = dm.add_dct_basis(duration=128).add_poly(order=2, include_lower=True)

print(f"Task regressors:     {dm.shape[1]}")
print(f"After adding drift:  {dm_full.shape[1]}")
```

```{code-cell} python3
dm_full.plot(figsize=(12, 8));
```

Notice the block-like DCT columns and smooth polynomial columns on the right. These soak up variance from scanner drift so it doesn't get mis-attributed to the task conditions.

## Fitting the GLM

Now we bring `BrainData` and `DesignMatrix` together. `.fit()` runs an OLS regression at every voxel:

```{code-cell} python3
data.fit(model="glm", X=dm_full)
```

Under the hood, `nltools` will:

1. Scale the data to percent signal change (so betas are interpretable)
2. Run an OLS regression at each voxel
3. Store the results as new `.glm_*` attributes on the `BrainData` object

Each result is itself a `BrainData` object, so you can immediately visualize or do further computation with it:

```{code-cell} python3
print(f"Betas (weights):       {data.glm_betas.shape}")
print(f"t-statistics:          {data.glm_t.shape}")
print(f"p-values:              {data.glm_p.shape}")
print(f"Standard errors:       {data.glm_se.shape}")
print(f"R-squared (per voxel): {data.glm_r2.shape}")
```

The beta map for a single condition is just one row of `glm_betas`:

```{code-cell} python3
regressor_names = list(dm_full.columns)
reading_idx = regressor_names.index("sentence_reading")

reading_beta = data.glm_betas[reading_idx]
reading_beta.plot(title="Beta map: sentence reading")
```

## Computing Contrasts

Beta maps tell you the response to each condition, but usually we care about **differences** between conditions. That's what contrasts are for. `compute_contrasts()` accepts three input forms.

### String notation (simplest)

Write the comparison as a string using the regressor names:

```{code-cell} python3
data.compute_contrasts("sentence_reading - sentence_listening").plot(
    title="Reading > Listening"
)
```

### Numeric vector (full control)

For complex or non-standard contrasts, pass a numeric vector with one weight per regressor in the design matrix:

```{code-cell} python3
# Average activation across all four button-press (motor) conditions
contrast_vec = np.zeros(len(regressor_names))

motor_conditions = [
    "audio_left_hand_button_press",
    "audio_right_hand_button_press",
    "visual_left_hand_button_press",
    "visual_right_hand_button_press",
]
for cond in motor_conditions:
    contrast_vec[regressor_names.index(cond)] = 1.0 / len(motor_conditions)

contrast_vec
```

```{code-cell} python
# Use it
data.compute_contrasts(contrast_vec).plot(title="Average motor response")
```

### Dictionary (multiple contrasts at once)

Pass a dictionary to compute several contrasts in one call:

```{code-cell} python3
contrasts = {
    "reading_vs_listening": "sentence_reading - sentence_listening",
    "motor_left_vs_right":  "audio_left_hand_button_press - audio_right_hand_button_press",
    "computation_vs_checkerboard":
        "audio_computation - horizontal_checkerboard",
}

# 3 t-maps - one for each contrast
data.compute_contrasts(contrasts)
```

## Putting It All Together

Here's the complete workflow in a compact form:

```{code-cell} python3
# 1. Load data
localizer = fetch_localizer_first_level(verbose=0)
data = BrainData(localizer.epi_img, verbose=False)

events = pd.read_csv(localizer.events, sep="\t")
dm = onsets_to_dm(
    timings=events,
    run_length=data.shape[0],
    TR=localizer.t_r,
    hrf_model="glover",
)

# 2. Add nuisance regressors
dm_full = dm.add_dct_basis(duration=128).add_poly(order=2, include_lower=True)

# 3. Fit GLM
data.fit(model="glm", X=dm_full)

# 4. Compute contrasts
reading_vs_listening = data.compute_contrasts(
    "sentence_reading - sentence_listening"
)
```

| Step | What it does | Key method |
|------|-------------|------------|
| Load data | Fetch BOLD + events, wrap as `BrainData` | `fetch_localizer_first_level()` + `BrainData()` |
| Build design | Convert events.tsv → HRF-convolved regressors | `onsets_to_dm(events, run_length, TR)` |
| Nuisance regressors | Model scanner drift and noise | `dm.add_dct_basis().add_poly()` |
| Fit model | Estimate betas at every voxel | `data.fit(model="glm", X=dm)` |
| Contrasts | Test differences between conditions | `data.compute_contrasts("A - B")` |
| Access results | Beta maps, t-stats, p-values | `data.glm_betas`, `glm_t`, `glm_p` |

## Next Steps

- **Group analysis**: Combine first-level contrasts across subjects for population-level inference — see [Group Analysis](02_group_analysis.md)
- **Two-stage GLM**: Run the full two-stage approach (first-level per run, second-level across runs) — see [Two-Stage GLM](07_two_stage_glm.md)
- **Decoding**: Instead of asking "where in the brain?", ask "can the brain tell these conditions apart?" — see [Decoding](05_decoding.md)
