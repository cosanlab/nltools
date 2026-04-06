---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# BrainCollection: Multi-Subject Data Container

This tutorial covers the `BrainCollection` class for working with data from
multiple subjects or runs.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Create BrainCollections from paths, BrainData objects, or BIDS layouts
- Use 3-axis indexing (images × timepoints × voxels)
- Compute aggregations across subjects (mean, std)
- Run group inference (t-tests, permutation tests)
- Fit encoding models with ridge regression
- Use ISC (inter-subject correlation) for naturalistic data

## Introduction

When working with multiple subjects, you often need to:
- Compute group-level statistics (are effects consistent across subjects?)
- Run encoding/decoding models on each subject
- Measure similarity between subjects (ISC)

`BrainCollection` provides a container that makes these operations simple
while handling memory efficiently through lazy loading.

**Key features**:
- 3-axis semantics: `(n_images, n_observations, n_voxels)`
- Lazy loading: only load data when needed
- Group inference: t-tests, permutation tests, ANOVA
- Encoding models: fit_ridge, fit_glm
- Multi-subject: ISC, hyperalignment

## Creating a BrainCollection

```{code-cell} python3
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for script execution

import numpy as np
import pandas as pd

from nltools.data import BrainData, BrainCollection
from nltools.datasets import fetch_haxby

# Load example dataset (multiple subjects)
data, design = fetch_haxby(n_subjects=3)

# Create collection from list of BrainData
# BrainCollection needs a common mask - use the mask from first subject
mask = data[0].mask
bc = BrainCollection(data, mask=mask)
print(bc)
print(f"Number of subjects: {bc.n_images}")
print(f"Shape: {bc.shape}")
```

The shape shows `(n_images, n_observations, n_voxels)`:
- **n_images**: Number of subjects/runs
- **n_observations**: Timepoints per subject (may vary)
- **n_voxels**: Number of voxels in the common mask

### From File Paths (Lazy Loading)

For large datasets, lazy loading avoids memory issues:

```{code-cell} python3
# In practice, you'd have paths to actual files
# bc_lazy = BrainCollection(paths, mask=mask, lazy=True)
#
# With lazy=True:
# - Data is only loaded when accessed
# - Memory-efficient for large multi-subject datasets
# - Check loading state with bc.is_loaded
```

### From Glob Patterns

```{code-cell} python3
# Load all subject files matching a pattern
# bc = BrainCollection.from_glob(
#     "derivatives/sub-*/func/*_bold.nii.gz",
#     mask=mask,
#     lazy=True,
# )
```

### From BIDS Layout

If you have a BIDS dataset and pybids installed:

```{code-cell} python3
# from bids import BIDSLayout
# layout = BIDSLayout("/path/to/bids")
#
# bc = BrainCollection.from_bids(
#     layout,
#     mask=mask,
#     task="rest",
#     suffix="bold",
# )
```

## 3-Axis Indexing

`BrainCollection` supports NumPy-style indexing across all three axes:

```{code-cell} python3
# Get first subject (returns BrainData)
first_subject = bc[0]
print(f"First subject: {first_subject.shape}")

# Get timepoint 10 from all subjects (returns BrainCollection)
timepoint_10 = bc[:, 10]
print(f"Timepoint 10 across subjects: {timepoint_10.shape}")

# Get specific voxels from all subjects
# subset = bc[:, :, :100]
```

## Aggregation Methods

Compute statistics across different axes:

```{code-cell} python3
# Mean across subjects (axis=0) -> BrainData
group_mean = bc.mean(axis=0)
print(f"Group mean shape: {group_mean.shape}")

# Mean across time (axis=1) -> BrainCollection (one value per subject)
temporal_mean = bc.mean(axis=1)
print(f"Temporal mean shape: {temporal_mean.shape}")

# Standard deviation across subjects
group_std = bc.std(axis=0)
```

## Group Inference

Test whether effects are consistent across subjects:

```{code-cell} python3
# One-sample t-test: is the mean significantly different from zero?
# Useful for testing if activation is consistent across subjects
t_stat, p_val = bc.mean(axis=1).ttest()
print(f"T-statistic shape: {t_stat.shape}")
print(f"Significant voxels: {(p_val.data < 0.05).sum()}")
```

### Permutation Tests

For non-parametric inference:

```{code-cell} python3
# Permutation test (more robust, slower)
# result = bc.mean(axis=1).permutation_test(n_permute=5000)
# print(f"Permutation p-values: {result['p'].shape}")
```

### Comparing Groups

Compare two groups of subjects:

```{code-cell} python3
# Split into two groups
group1 = bc[:2]  # First two subjects
group2 = bc[2:]  # Remaining subjects

# Two-sample t-test
# Note: need equal timepoints, so using temporal mean
t_stat, p_val = group1.mean(axis=1).ttest2(group2.mean(axis=1))
```

## Transformation Methods

Apply functions to each subject:

```{code-cell} python3
# Map a function over each subject
def zscore_subject(bd):
    """Z-score each subject's data."""
    result = BrainData(mask=bd.mask)
    result.data = (bd.data - bd.data.mean(axis=0)) / bd.data.std(axis=0)
    return result


bc_zscored = bc.map(zscore_subject, axis=0, show_progress=False)

# Filter to subjects meeting a criterion
# bc_filtered = bc.filter(lambda bd: bd.data.mean() > 0)
```

## Encoding Models with Ridge Regression

Fit encoding models to predict brain activity from stimulus features:

```{code-cell} python3
# Create example features (would typically come from your experiment)
n_timepoints = bc[0].shape[0]
n_features = 5
X = np.random.randn(n_timepoints, n_features)

# Fit ridge regression using unified fit() API
# Returns BrainCollection of CV scores by default
scores = bc.fit(model="ridge", X=X, cv=3, show_progress=False)
print(f"Scores shape: {scores[0].shape}")

# Get weights instead of scores
weights = bc.fit(model="ridge", X=X, output="weights", cv=None, show_progress=False)
print(f"Weights shape: {weights[0].shape}")
```

### Accessing Weights and Scores

The result includes weights that can be used for group-level inference:

```{code-cell} python3
# If result is a BrainCollection of weights:
# - Test if feature 0 has consistent weights across subjects
# feature_weights = weights[:, 0, :]  # All subjects, feature 0, all voxels
# t_stat, p_val = feature_weights.ttest()  # Group-level test
```

## Inter-Subject Correlation (ISC)

Measure how similarly subjects respond to naturalistic stimuli:

```{code-cell} python3
# Compute ISC across subjects
# High ISC indicates stimulus-driven activity consistent across subjects
isc_result = bc.isc(method="loo", show_progress=False)
print(f"ISC shape: {isc_result['isc'].shape}")
print(f"Mean ISC: {isc_result['isc'].data.mean():.3f}")
```

### ISC with Statistical Testing

Test whether ISC is significantly greater than zero:

```{code-cell} python3
# Permutation test for ISC significance
# isc_test_result = bc.isc_test(
#     method="loo",
#     n_permute=1000,
#     show_progress=False,
# )
# print(f"Significant ISC voxels: {(isc_test_result['p'].data < 0.05).sum()}")
```

## GLM Workflow

Fit first-level GLMs across subjects:

```{code-cell} python3
# Create example events DataFrame
events = pd.DataFrame(
    {
        "onset": [0, 10, 20, 30, 40],
        "duration": [5, 5, 5, 5, 5],
        "trial_type": ["A", "B", "A", "B", "A"],
    }
)

# Fit GLM using unified fit() API with DesignMatrix
# dm = DesignMatrix.from_events(events, t_r=2.0, n_scans=bc[0].shape[0])
# betas = bc.fit(model="glm", X=dm, show_progress=False)
#
# # Or use fit_from_events() convenience method
# betas = bc.fit_from_events(
#     events=events,
#     t_r=2.0,  # TR in seconds
#     return_stats=["t"],
#     show_progress=False,
# )
#
# # Compute contrasts
# contrast = betas.compute_contrasts("A - B")
#
# # Group-level test
# t_stat, p_val = contrast.ttest()
```

## Memory Efficiency

`BrainCollection` is designed for memory-efficient processing:

1. **Lazy loading**: Data loaded only when accessed
2. **Batch processing**: Process subjects one at a time
3. **Progress tracking**: Monitor long-running operations

For very large datasets:

```{code-cell} python3
# Process in batches to control memory
# for batch in bc.iter_batches(batch_size=5, axis=0):
#     # Process each batch of 5 subjects
#     batch_mean = batch.mean(axis=0)
```

## Metadata

Attach subject-level metadata:

```{code-cell} python3
# Add metadata - must match number of images in collection
n_images = len(data)
metadata = pd.DataFrame(
    {
        "subject_id": [f"sub-{i + 1:02d}" for i in range(n_images)],
        "age": np.random.randint(20, 40, n_images),
        "group": ["control" if i % 2 == 0 else "patient" for i in range(n_images)],
    }
)

bc_with_meta = BrainCollection(data, mask=mask, metadata=metadata)
print(bc_with_meta.metadata.head())

# Filter by metadata
# controls = bc_with_meta.filter(lambda bd, meta: meta["group"] == "control")
```

## Summary

`BrainCollection` provides a powerful interface for multi-subject analyses:

| Operation | Method | Returns |
|-----------|--------|---------|
| Group mean | `bc.mean(axis=0)` | BrainData |
| Subject means | `bc.mean(axis=1)` | BrainCollection |
| T-test | `bc.ttest()` | (t_stat, p_val) |
| Permutation test | `bc.permutation_test()` | dict |
| Ridge encoding | `bc.fit(model="ridge", X=X)` | BrainCollection |
| ISC | `bc.isc()` | dict |
| GLM | `bc.fit(model="glm", X=dm)` | BrainCollection |
| GLM from events | `bc.fit_from_events(events)` | BrainCollection |

For more advanced workflows, see the workflow tutorials.
