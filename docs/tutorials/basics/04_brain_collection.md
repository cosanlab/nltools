---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# BrainCollection Basics

## Learning Objectives

By the end of this tutorial, you will be able to:
- Create `BrainCollection` objects from BrainData objects or file paths
- Use 3-axis indexing (images x time x voxels)
- Compute group-level aggregations (mean, std)
- Run group inference (t-tests, permutation tests)
- Apply preprocessing across subjects
- Use ISC for naturalistic data analysis

## Introduction

`BrainCollection` is a container for multi-subject (or multi-run) neuroimaging data. It provides:

- **3-axis semantics**: `(n_images, n_observations, n_voxels)` — images are subjects/runs, observations are timepoints
- **Lazy loading**: data is only loaded from disk when accessed, keeping memory usage low
- **Group operations**: aggregation, inference, and modeling applied across subjects in parallel

```{code-cell} python3
import os

os.environ["TQDM_DISABLE"] = "1"  # Suppress progress bars in tutorial output

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltools.data import BrainData, BrainCollection
from nltools.datasets import fetch_haxby
```

## Creating a BrainCollection

### From BrainData Objects

```{code-cell} python3
# Load Haxby dataset (3 subjects, each with multiple runs)
data, design = fetch_haxby(n_subjects=3)

# data is a list of BrainData objects (one per subject)
mask = data[0].mask
bc = BrainCollection(data, mask=mask)
print(bc)
```

The shape shows `(n_images, n_observations, n_voxels)`:
- **n_images**: number of subjects/runs
- **n_observations**: timepoints per subject (may vary)
- **n_voxels**: voxels in the common mask

```{code-cell} python3
print(f"Images (subjects): {bc.n_images}")
print(f"Voxels: {bc.n_voxels}")
print(f"Shape: {bc.shape}")
```

### From File Paths (Lazy Loading)

For large datasets, pass file paths instead of loaded data. With `lazy=True` (the default), data is only loaded when accessed:

```python
# Example: paths to subject files
paths = ["sub-01_bold.nii.gz", "sub-02_bold.nii.gz", "sub-03_bold.nii.gz"]
bc_lazy = BrainCollection(paths, mask=mask)  # lazy=True by default
```

### From Glob Patterns

```python
bc = BrainCollection.from_glob(
    "derivatives/sub-*/func/*_bold.nii.gz",
    mask=mask,
)
```

## 3-Axis Indexing

`BrainCollection` supports indexing across all three dimensions:

```{code-cell} python3
# Get first subject (returns BrainData)
first_subject = bc[0]
print(f"First subject: {first_subject.shape}")

# Get first 2 subjects (returns BrainCollection)
subset = bc[:2]
print(f"Subset: {subset.shape}")
```

Multi-dimensional indexing selects across time and voxel axes:

```{code-cell} python3
# Get timepoint 5 from all subjects
tp5 = bc[:, 5]
print(f"Timepoint 5 across subjects: {tp5.shape}")

# Get first 10 timepoints from first 2 subjects
early = bc[:2, :10]
print(f"First 2 subjects, 10 timepoints: {early.shape}")
```

## Aggregation

Compute statistics across different axes:

```{code-cell} python3
# Mean across subjects (axis=0) → BrainData
group_mean = bc.mean(axis=0)
print(f"Group mean: {group_mean.shape}")

# Mean across time per subject (axis=1) → BrainCollection
temporal_means = bc.mean(axis=1)
print(f"Temporal means: {temporal_means.shape}")

# Grand mean across subjects AND time (axis=(0,1)) → BrainData
grand_mean = bc.mean(axis=(0, 1))
print(f"Grand mean: {grand_mean.shape}")

# Standard deviation across subjects
group_std = bc.std(axis=0)
```

```{code-cell} python3
# Visualize the group mean
grand_mean.plot(title="Group Mean Activation")
```

## Group Inference

### One-Sample T-Test

Test whether activation is consistently different from zero across subjects. This requires one observation per subject, so we first average across time:

```{code-cell} python3
# Average each subject across time, then t-test across subjects
subject_means = bc.mean(axis=1)
t_stat, p_val = subject_means.ttest()
print(f"T-statistic: {t_stat.shape}")
print(f"Significant voxels (p < 0.001): {(p_val.data < 0.001).sum()}")
```

```{code-cell} python3
# Visualize significant regions
t_stat.plot(title="Group T-statistic", threshold=3.0)
```

### Two-Sample T-Test

Compare two groups of subjects:

```{code-cell} python3
group1 = bc[:2].mean(axis=1)
group2 = bc[2:].mean(axis=1)

t_stat2, p_val2 = group1.ttest2(group2)
print(f"Two-sample t-test: {(p_val2.data < 0.05).sum()} voxels at p < 0.05")
```

### Permutation Test

Non-parametric inference via sign-flipping:

```{code-cell} python3
perm_result = subject_means.permutation_test(n_permute=500)
print(f"Permutation test keys: {list(perm_result.keys())}")
print(f"Significant voxels (p < 0.05): {(perm_result['p'].data < 0.05).sum()}")
```

## Preprocessing

Apply preprocessing to all subjects at once:

```{code-cell} python3
# Standardize each subject (mean-center per voxel)
bc_centered = bc.standardize(axis=0, method="center", show_progress=False)
print(f"Centered: {bc_centered.shape}")

# Smooth all subjects
bc_smooth = bc.smooth(fwhm=6, show_progress=False)
print(f"Smoothed: {bc_smooth.shape}")
```

### Custom Transforms with `map()`

Apply any function to each subject:

```{code-cell} python3
# Z-score each subject independently
bc_zscored = bc.map(
    lambda bd: bd.standardize(method="zscore"),
    axis=0,
    show_progress=False,
)
print(f"Z-scored: {bc_zscored.shape}")
```

## Metadata

Attach subject-level metadata for filtering and analysis:

```{code-cell} python3
# Metadata must have one row per image in the collection
n = bc.n_images
metadata = pd.DataFrame({
    "subject_id": [f"sub-{i+1:02d}" for i in range(n)],
    "age": np.random.randint(20, 40, n),
    "group": ["control" if i % 2 == 0 else "patient" for i in range(n)],
})

bc_meta = BrainCollection(data, mask=mask, metadata=metadata)
print(bc_meta.metadata)
```

```{code-cell} python3
# Filter by metadata using a boolean mask
control_mask = bc_meta.metadata["group"] == "control"
controls = bc_meta.filter(control_mask)
print(f"Controls: {controls.n_images} images")
```

## Memory Management

`BrainCollection` tracks which images are loaded and estimates memory usage:

```{code-cell} python3
print(f"Loaded: {bc.is_loaded}")
print(f"Memory estimate: {bc.memory_estimate()}")
```

For large datasets, use batch processing:

```{code-cell} python3
# Process in batches of 2
for i, batch in enumerate(bc.iter_batches(batch_size=2, axis=0, show_progress=False)):
    print(f"Batch {i}: {batch.shape}")
```

## Converting Between Formats

```{code-cell} python3
# To a list of BrainData objects
brain_list = bc.to_list()
print(f"List: {len(brain_list)} BrainData objects")

# To a single stacked BrainData (concatenate all subjects)
stacked = bc.to_stacked()
print(f"Stacked: {stacked.shape}")

# Back to BrainCollection from stacked
bc_from_stacked = BrainCollection.from_stacked(stacked, n_images=bc.n_images)
print(f"From stacked: {bc_from_stacked.shape}")
```

## File I/O

```{code-cell} python3
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    # Write all subjects to a directory
    paths = bc.mean(axis=1).write(tmpdir, pattern="sub_{i:02d}.nii.gz")
    print(f"Wrote {len(paths)} files")

    # Load back
    bc_loaded = BrainCollection(paths, mask=mask)
    print(f"Loaded: {bc_loaded.shape}")
```

## Summary

| Operation | Method | Returns |
|-----------|--------|---------|
| Group mean | `bc.mean(axis=0)` | BrainData |
| Subject means | `bc.mean(axis=1)` | BrainCollection |
| Grand mean | `bc.mean(axis=(0,1))` | BrainData |
| One-sample t-test | `bc.ttest()` | (t_stat, p_val) |
| Two-sample t-test | `bc.ttest2(other)` | (t_stat, p_val) |
| Permutation test | `bc.permutation_test()` | dict |
| Custom transform | `bc.map(fn, axis=0)` | BrainCollection |
| Standardize | `bc.standardize()` | BrainCollection |
| Smooth | `bc.smooth(fwhm)` | BrainCollection |
| ISC | `bc.isc()` | dict |

For encoding models (ridge, GLM) and ISC workflows, see the [workflow tutorials](../workflows/01_glm.md).
