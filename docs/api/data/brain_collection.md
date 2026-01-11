# `nltools.data.BrainCollection`

Multi-subject data container for group-level neuroimaging analyses.

## Overview

`BrainCollection` is a container class for working with data from multiple subjects or runs. It provides 3-axis indexing (images × timepoints × voxels), lazy loading for memory efficiency, and integrated methods for group inference, encoding models, and inter-subject correlation analysis.

## Key Features

- **3-axis semantics**: Index by images (subjects), timepoints, and voxels
- **Lazy loading**: Memory-efficient processing of large multi-subject datasets
- **Group inference**: One-sample t-tests, two-sample t-tests, permutation tests
- **Encoding models**: `fit_ridge()` and `fit_glm()` with cross-validation
- **ISC analysis**: `isc()` and `isc_test()` for naturalistic neuroimaging
- **Transformations**: `map()`, `filter()`, `mean()`, `std()` across axes
- **Predictions**: `predict()` for encoding/decoding with fitted models
- **Metadata**: Attach subject-level metadata for filtering and grouping

## Quick Start

```python
from nltools.data import BrainData, BrainCollection
from nltools.datasets import fetch_haxby

# Load multi-subject data
data, _ = fetch_haxby(n_subjects=5)
bc = BrainCollection(data, mask=data[0].mask)

# 3-axis indexing
first_subject = bc[0]              # BrainData
timepoint_10 = bc[:, 10]           # BrainCollection
subset_voxels = bc[:, :, :1000]    # BrainCollection

# Group statistics
group_mean = bc.mean(axis=0)       # BrainData (mean across subjects)
subject_means = bc.mean(axis=1)    # BrainCollection (mean across time)

# Group inference
t_stat, p_val = subject_means.ttest()

# ISC for naturalistic data
isc_result = bc.isc(method="loo")

# Ridge encoding
import numpy as np
X = np.random.randn(bc[0].shape[0], 10)  # (timepoints, features)
result = bc.fit_ridge(X=X, cv=3)
```

## Construction Methods

| Method | Use Case |
|--------|----------|
| `BrainCollection(data, mask)` | From list of BrainData or file paths |
| `BrainCollection.from_glob(pattern, mask)` | From glob pattern matching files |
| `BrainCollection.from_bids(layout, mask)` | From pybids BIDSLayout |
| `BrainCollection.from_stacked(brain_data, axis)` | Split stacked BrainData |

## Main Methods

### Aggregation

| Method | Description |
|--------|-------------|
| `mean(axis)` | Mean across specified axis |
| `std(axis)` | Standard deviation across axis |
| `sum(axis)` | Sum across axis |

### Group Inference

| Method | Description |
|--------|-------------|
| `ttest()` | One-sample t-test vs zero |
| `ttest2(other)` | Two-sample t-test |
| `permutation_test(n_permute)` | Non-parametric permutation test |
| `anova(*groups)` | One-way ANOVA |

### Encoding/Decoding

| Method | Description |
|--------|-------------|
| `fit_ridge(X, cv)` | Fit ridge regression for each subject |
| `fit_glm(events, t_r)` | Fit first-level GLM for each subject |
| `predict(X)` | Generate predictions from fitted model |
| `compute_contrasts(contrast)` | Compute contrasts from fitted GLM |

### ISC (Inter-Subject Correlation)

| Method | Description |
|--------|-------------|
| `isc(method)` | Compute inter-subject correlation |
| `isc_test(n_permute)` | ISC with permutation testing |

### Transformations

| Method | Description |
|--------|-------------|
| `map(func, axis)` | Apply function to each element |
| `filter(predicate)` | Filter based on condition |
| `apply(func)` | Apply function returning scalar |

## API Reference

```{eval-rst}
.. autoclass:: nltools.data.BrainCollection
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__, __getitem__, __len__
```

## See Also

- {doc}`brain_data` - Single-subject data container
- {doc}`adjacency` - Similarity/distance matrices
- {doc}`design_matrix` - Design matrices for GLM
- {doc}`../../migration-guide` - Migration guide for v0.6.0 changes
