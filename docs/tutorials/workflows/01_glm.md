---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# GLM Analysis

First-level GLM workflow: load data, fit model, compute contrasts.

```{code-cell} python3
import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt

from nltools.datasets import fetch_haxby
```

## Load Data

```{code-cell} python3
# Load Haxby dataset (single subject)
brain_data, design_matrices = fetch_haxby(n_subjects=1, verbose=1)

data = brain_data[0]
dm = design_matrices[0]

print(f"Data shape: {data.shape}")
print(f"Design matrix: {dm.shape}")
print(f"Conditions: {list(dm.columns)}")
```

## Visualize Design Matrix

```{code-cell} python3
dm.heatmap(figsize=(12, 6))
plt.title("Design Matrix")
```

## Add Nuisance Regressors

```{code-cell} python3
# Add DCT basis (high-pass filter) and polynomial drift
dm_filt = dm.add_dct_basis(duration=128).add_poly(order=2, include_lower=True)

print(f"Original: {dm.shape[1]} columns")
print(f"With nuisance: {dm_filt.shape[1]} columns")
```

## Fit GLM

```{code-cell} python3
data.fit(model="glm", X=dm_filt)

print("GLM results available:")
print(f"  glm_betas: {data.glm_betas.shape}")
print(f"  glm_t: {data.glm_t.shape}")
print(f"  glm_p: {data.glm_p.shape}")
```

## Compute Contrasts

Three ways to specify contrasts:

```{code-cell} python3
# Method 1: String notation
face_vs_house = data.compute_contrasts("face - house")
face_vs_house.plot(title="Face > House (t-statistic)")
```

```{code-cell} python3
# Method 2: Numeric vector (for complex contrasts)
regressor_names = list(dm_filt.columns)
contrast_vec = np.zeros(len(regressor_names))

# Average of all visual conditions
for cond in ["face", "house", "cat", "bottle", "scissors", "shoe", "chair"]:
    if cond in regressor_names:
        contrast_vec[regressor_names.index(cond)] = 1.0 / 7

all_visual = data.compute_contrasts(contrast_vec)
all_visual.plot(title="Average Visual Response")
```

```{code-cell} python3
# Method 3: Dictionary for multiple contrasts
contrasts = {
    "face_vs_house": "face - house",
    "face_vs_scrambled": "face - scrambledpix",
    "face_only": "face",
}

results = data.compute_contrasts(contrasts)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, (name, result) in enumerate(results.items()):
    result.plot(ax=axes[idx], title=name)
plt.tight_layout()
```

## Summary

| Step | Method |
|------|--------|
| Load data | `fetch_haxby()` |
| Add nuisance | `dm.add_dct_basis().add_poly()` |
| Fit model | `data.fit(model='glm', X=dm)` |
| Contrasts | `data.compute_contrasts("A - B")` |
| Access results | `data.glm_betas`, `glm_t`, `glm_p` |

For group analysis, see `02_group_analysis.py`.
