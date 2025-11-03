# `nltools.data.BrainData`

Main class for representing neuroimaging data as vectors.

## Overview

`BrainData` is the core data structure in nltools for working with neuroimaging data. It stores brain images as 2D arrays (voxels × images) rather than 4D matrices, making data manipulation and analysis more intuitive. Think of it as a neuroimaging-specific pandas DataFrame.

## Key Features

- **Data manipulation**: Arithmetic operations, concatenation, indexing
- **Preprocessing**: Smoothing, filtering, detrending, standardization
- **Masking**: ROI extraction, sphere creation, mask application
- **Statistical modeling**: Fit/predict API for Ridge and GLM models
- **Immutable results**: `fit(inplace=False)` returns Fit dataclass for serialization
- **Visualization**: Quick plotting methods for brain maps
- **File I/O**: Support for NIfTI, HDF5, and NeuroVault formats

## Quick Start

```python
from nltools.data import BrainData
from nltools.datasets import fetch_pain

# Load example data
data = fetch_pain()

# Basic operations
smoothed = data.smooth(fwhm=6)
standardized = data.standardize()

# Statistical modeling (mutating mode - default)
data.fit(model='ridge', X=design_matrix)
predictions = data.predict(X=test_data)

# Statistical modeling (immutable mode - returns Fit dataclass)
from nltools.data import Fit
fit = data.fit(model='ridge', X=design_matrix, inplace=False)
assert isinstance(fit, Fit)
assert 'weights' in fit.available()
```

## Fit Method

The `fit()` method supports two modes:

**Default mode** (`inplace=True`): Mutates BrainData object, adds attributes
```python
brain.fit(model='ridge', X=features, alpha=1.0)
weights = brain.ridge_weights  # Access as attribute
```

**Immutable mode** (`inplace=False`): Returns Fit dataclass, BrainData unchanged
```python
fit = brain.fit(model='ridge', X=features, alpha=1.0, inplace=False)
weights = fit.weights  # Access from Fit object
assert not hasattr(brain, 'ridge_weights')  # brain unchanged
```

See the [Migration Guide](../migration-guide.md#pattern-10-fit-dataclass-braindata-fit-inplace-false) for more details.

## Full API Reference

```{eval-rst}
.. autoclass:: nltools.data.BrainData
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

## See Also

- {doc}`adjacency` - Similarity/distance matrices
- {doc}`design_matrix` - Design matrices for GLM
- {doc}`../models` - Statistical models (Ridge, etc.)
- {doc}`../stats` - Statistical functions
- {doc}`../migration-guide` - Migration guide for v0.6.0 changes
