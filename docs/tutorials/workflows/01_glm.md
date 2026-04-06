---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
execute:
  allow_errors: false
---

# First-Level GLM Analysis

This tutorial walks through a complete first-level GLM analysis: loading fMRI data and a design matrix, fitting the model, and computing contrasts. Along the way, we'll introduce how nltools' two core classes --- `BrainData` and `DesignMatrix` --- work together.

:::{tip}
If you want deeper coverage of either class on its own, see the [BrainData basics](../basics/01_brain_data.md) and [DesignMatrix basics](../basics/02_design_matrix.md) tutorials.
:::

## The General Linear Model in a Nutshell

The GLM models each voxel's timeseries **y** as a weighted combination of regressors plus noise:

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$

- **y** is the BOLD timeseries at a single voxel (one column of your `BrainData`)
- **X** is the design matrix (your `DesignMatrix`) --- task regressors convolved with the HRF, plus nuisance terms
- **beta** are the regression weights we want to estimate
- **epsilon** is residual noise

In nltools, `BrainData` holds **y** and `DesignMatrix` holds **X**. Fitting the GLM with `BrainData.fit()` estimates **beta** at every voxel simultaneously.

```{code-cell} python3
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from nltools.data import BrainData, DesignMatrix
from nltools.datasets import fetch_haxby
```

## Step 1: Load Data

We'll use the Haxby visual object recognition dataset. `fetch_haxby` returns paired `BrainData` and `DesignMatrix` objects --- one per run --- with the design matrices already HRF-convolved.

```{code-cell} python3
brain_data, design_matrices = fetch_haxby(n_subjects=1, verbose=0)

# Grab the first run
data = brain_data[0]
dm = design_matrices[0]

print(f"BrainData shape: {data.shape}  (timepoints x voxels)")
print(f"DesignMatrix shape: {dm.shape}  (timepoints x regressors)")
```

The `BrainData` object stores the 4D fMRI timeseries as a 2D matrix (timepoints x voxels) plus a brain mask that maps voxels back to 3D space. The `DesignMatrix` stores the experimental design with one column per condition.

```{code-cell} python3
print(f"Conditions: {list(dm.columns)}")
```

## Step 2: Inspect the Design Matrix

Before fitting, it's always a good idea to visualize the design matrix. The `heatmap()` method shows a traditional SPM-style view where rows are timepoints and columns are regressors:

```{code-cell} python3
dm.heatmap(figsize=(10, 6))
plt.title("Design Matrix (task regressors only)")
plt.show()
```

Each column represents a condition (face, house, cat, etc.) convolved with the hemodynamic response function. The HRF accounts for the ~5 second delay between neural activity and the peak BOLD response.

## Step 3: Add Nuisance Regressors

Raw fMRI data contains low-frequency drift and other noise sources that aren't related to the task. We model these with **nuisance regressors** so they don't contaminate our task estimates.

Two common approaches (often used together):

- **DCT basis functions**: A discrete cosine transform high-pass filter. The `duration` parameter sets the cutoff period in seconds --- signals slower than this are filtered out. 128 seconds is a common choice.
- **Polynomial drift**: Legendre polynomials that capture slow trends. Order 0 is the intercept (mean signal), order 1 captures linear drift, order 2 captures quadratic drift, etc.

```{code-cell} python3
dm_full = dm.add_dct_basis(duration=128).add_poly(order=2, include_lower=True)

print(f"Task regressors: {dm.shape[1]}")
print(f"After adding nuisance: {dm_full.shape[1]}")
```

```{code-cell} python3
dm_full.heatmap(figsize=(12, 8))
plt.title("Complete Design Matrix (task + nuisance)")
plt.show()
```

Notice the block-like DCT columns and smooth polynomial columns on the right side of the heatmap. These soak up variance from scanner drift so it doesn't get attributed to the task conditions.

## Step 4: Fit the GLM

Now we bring `BrainData` and `DesignMatrix` together. The `fit()` method estimates beta weights at every voxel in the brain:

```{code-cell} python3
data.fit(model="glm", X=dm_full)
```

That's it --- one line. Under the hood, nltools:
1. Scales the data to percent signal change (so betas are interpretable)
2. Runs an OLS regression at each voxel
3. Stores the results as new attributes on the `BrainData` object

Let's see what we got:

```{code-cell} python3
print("GLM results:")
print(f"  Betas (weights):       {data.glm_betas.shape}")
print(f"  t-statistics:          {data.glm_t.shape}")
print(f"  p-values:              {data.glm_p.shape}")
print(f"  Standard errors:       {data.glm_se.shape}")
print(f"  R-squared (per voxel): {data.glm_r2.shape}")
```

Each result is itself a `BrainData` object, so you can immediately visualize or do further computation with it. For example, the beta map for a single condition is one row of `glm_betas`:

```{code-cell} python3
# Find which row corresponds to the "face" condition
regressor_names = list(dm_full.columns)
face_idx = regressor_names.index("face")

face_beta = data.glm_betas[face_idx]
face_beta.plot(title="Beta map: face condition")
```

## Step 5: Compute Contrasts

Beta maps tell you the response to each condition, but usually we care about **differences** between conditions. That's what contrasts are for.

### String notation (simplest)

The most intuitive way --- just write the comparison as a string:

```{code-cell} python3
face_vs_house = data.compute_contrasts("face - house")
face_vs_house.plot(title="Face > House (t-statistic)")
```

This computes a t-statistic at every voxel testing whether the face response is significantly greater than the house response. You can also use coefficients: `"2*face - house - cat"`.

### Numeric vector (for full control)

For complex or non-standard contrasts, pass a numeric vector with one weight per regressor in the design matrix:

```{code-cell} python3
# Average activation across all visual object categories
regressor_names = list(dm_full.columns)
contrast_vec = np.zeros(len(regressor_names))

visual_conditions = ["face", "house", "cat", "bottle", "scissors", "shoe", "chair"]
for cond in visual_conditions:
    if cond in regressor_names:
        contrast_vec[regressor_names.index(cond)] = 1.0 / len(visual_conditions)

all_visual = data.compute_contrasts(contrast_vec)
all_visual.plot(title="Average Visual Response (all object categories)")
```

### Dictionary (multiple contrasts at once)

Pass a dictionary to compute several contrasts in one call:

```{code-cell} python3
contrasts = {
    "face_vs_house": "face - house",
    "face_vs_scrambled": "face - scrambledpix",
    "face_only": "face",
}

results = data.compute_contrasts(contrasts)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, result) in zip(axes, results.items()):
    result.plot(ax=ax, title=name)
plt.tight_layout()
plt.show()
```

## Putting It All Together

Here's the complete workflow in a compact form:

```{code-cell} python3
# 1. Load data
brain_data, design_matrices = fetch_haxby(n_subjects=1, verbose=0)
data, dm = brain_data[0], design_matrices[0]

# 2. Add nuisance regressors
dm_full = dm.add_dct_basis(duration=128).add_poly(order=2, include_lower=True)

# 3. Fit GLM
data.fit(model="glm", X=dm_full)

# 4. Compute contrasts
face_vs_house = data.compute_contrasts("face - house")
```

| Step | What it does | Key method |
|------|-------------|------------|
| Load data | Get paired BrainData + DesignMatrix | `fetch_haxby()` |
| Nuisance regressors | Model scanner drift and noise | `dm.add_dct_basis().add_poly()` |
| Fit model | Estimate betas at every voxel | `data.fit(model="glm", X=dm)` |
| Contrasts | Test differences between conditions | `data.compute_contrasts("A - B")` |
| Access results | Beta maps, t-stats, p-values | `data.glm_betas`, `glm_t`, `glm_p` |

## Next Steps

- **Group analysis**: Combine first-level contrasts across subjects for population-level inference --- see [Group Analysis](02_group_analysis.md)
- **Two-stage GLM**: Run the full two-stage approach (first-level per run, second-level across runs) --- see [Two-Stage GLM](07_two_stage_glm.md)
- **Decoding**: Instead of asking "where in the brain?", ask "can the brain tell these conditions apart?" --- see [Decoding](05_decoding.md)
