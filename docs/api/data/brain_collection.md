# `nltools.data.BrainCollection`

Multi-subject data container for group-level neuroimaging analyses.

## Overview

`BrainCollection` is a container class for working with data from multiple subjects or runs. It provides 3-axis indexing (images x timepoints x voxels), lazy loading for memory efficiency, and integrated methods for group inference, encoding models, and inter-subject correlation analysis.

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

## API Reference

```{eval-rst}
.. autoclass:: nltools.data.BrainCollection
    :no-members:
    :show-inheritance:
```

### Constructor

```{eval-rst}
.. automethod:: nltools.data.BrainCollection.__init__
```

### Properties

```{eval-rst}
.. autoattribute:: nltools.data.BrainCollection.n_images
.. autoattribute:: nltools.data.BrainCollection.n_voxels
.. autoattribute:: nltools.data.BrainCollection.mask
.. autoattribute:: nltools.data.BrainCollection.metadata
.. autoattribute:: nltools.data.BrainCollection.is_loaded
.. autoattribute:: nltools.data.BrainCollection.shape
```

### Data Access & Indexing

```{eval-rst}
.. automethod:: nltools.data.BrainCollection.__getitem__
.. automethod:: nltools.data.BrainCollection.__len__
.. automethod:: nltools.data.BrainCollection.__iter__
.. automethod:: nltools.data.BrainCollection.iter_batches
```

### Loading

```{eval-rst}
.. automethod:: nltools.data.BrainCollection.load
.. automethod:: nltools.data.BrainCollection.unload
.. automethod:: nltools.data.BrainCollection.memory_estimate
```

### Aggregation

```{eval-rst}
.. automethod:: nltools.data.BrainCollection.mean
.. automethod:: nltools.data.BrainCollection.std
.. automethod:: nltools.data.BrainCollection.var
.. automethod:: nltools.data.BrainCollection.sum
.. automethod:: nltools.data.BrainCollection.min
.. automethod:: nltools.data.BrainCollection.max
.. automethod:: nltools.data.BrainCollection.median
```

### Conversion

```{eval-rst}
.. automethod:: nltools.data.BrainCollection.to_tensor
.. automethod:: nltools.data.BrainCollection.to_list
.. automethod:: nltools.data.BrainCollection.to_stacked
```

### Constructors

```{eval-rst}
.. automethod:: nltools.data.BrainCollection.from_bids
.. automethod:: nltools.data.BrainCollection.from_glob
.. automethod:: nltools.data.BrainCollection.from_stacked
```

### Transforms

```{eval-rst}
.. automethod:: nltools.data.BrainCollection.map
.. automethod:: nltools.data.BrainCollection.filter
.. automethod:: nltools.data.BrainCollection.align
.. automethod:: nltools.data.BrainCollection.standardize
.. automethod:: nltools.data.BrainCollection.smooth
.. automethod:: nltools.data.BrainCollection.threshold
.. automethod:: nltools.data.BrainCollection.detrend
```

### Inference

```{eval-rst}
.. automethod:: nltools.data.BrainCollection.ttest
.. automethod:: nltools.data.BrainCollection.ttest2
.. automethod:: nltools.data.BrainCollection.permutation_test
.. automethod:: nltools.data.BrainCollection.permutation_test2
.. automethod:: nltools.data.BrainCollection.anova
.. automethod:: nltools.data.BrainCollection.isc
.. automethod:: nltools.data.BrainCollection.isc_test
```

### Modeling

```{eval-rst}
.. automethod:: nltools.data.BrainCollection.cv
.. automethod:: nltools.data.BrainCollection.fit
.. automethod:: nltools.data.BrainCollection.fit_glm
.. automethod:: nltools.data.BrainCollection.fit_from_events
.. automethod:: nltools.data.BrainCollection.fit_ridge
```

### Prediction

```{eval-rst}
.. automethod:: nltools.data.BrainCollection.predict
.. automethod:: nltools.data.BrainCollection.compute_contrasts
.. automethod:: nltools.data.BrainCollection.select_feature
```

### I/O

```{eval-rst}
.. automethod:: nltools.data.BrainCollection.write
```

## Submodules

The implementation is organized into functional submodules. These contain the
standalone functions that back the methods above:

- {doc}`collection_constructors` -- `from_bids`, `from_glob`, `from_stacked`
- {doc}`collection_transforms` -- `map`, `filter`, `align`, `standardize`, `smooth`, `threshold`, `detrend`
- {doc}`collection_inference` -- `ttest`, `permutation_test`, `anova`, `isc`, `isc_test`
- {doc}`collection_modeling` -- `fit`, `fit_glm`, `fit_ridge`, `fit_from_events`, `cv`
- {doc}`collection_prediction` -- `predict`, `compute_contrasts`
- {doc}`collection_io` -- `write`
- {doc}`collection_pipeline` -- `BrainCollectionPipeline`, `BrainCollectionCVResult`, `FittedBrainCollection`

## See Also

- {doc}`brain_data` - Single-subject data container
- {doc}`adjacency` - Similarity/distance matrices
- {doc}`design_matrix` - Design matrices for GLM
- {doc}`../../migration-guide` - Migration guide for v0.6.0 changes
