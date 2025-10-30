# Brain_Data

Main class for representing neuroimaging data as vectors.

## Overview

`Brain_Data` is the core data structure in nltools for working with neuroimaging data. It stores brain images as 2D arrays (voxels × images) rather than 4D matrices, making data manipulation and analysis more intuitive. Think of it as a neuroimaging-specific pandas DataFrame.

## Key Features

- **Data manipulation**: Arithmetic operations, concatenation, indexing
- **Preprocessing**: Smoothing, filtering, detrending, standardization
- **Masking**: ROI extraction, sphere creation, mask application
- **Statistical modeling**: Fit/predict API for Ridge and GLM models
- **Visualization**: Quick plotting methods for brain maps
- **File I/O**: Support for NIfTI, HDF5, and NeuroVault formats

## Quick Start

```python
from nltools.data import Brain_Data
from nltools.datasets import fetch_pain

# Load example data
data = fetch_pain()

# Basic operations
smoothed = data.smooth(fwhm=6)
standardized = data.standardize()

# Statistical modeling
data.fit(model='ridge', X=design_matrix)
predictions = data.predict(X=test_data)
```

## Full API Reference

```{eval-rst}
.. autoclass:: nltools.data.Brain_Data
    :members:
    :undoc-members:
    :show-inheritance:

.. automethod:: __init__
```

## See Also

- {doc}`adjacency` - Similarity/distance matrices
- {doc}`design_matrix` - Design matrices for GLM
- {doc}`../models` - Statistical models (Ridge, etc.)
- {doc}`../stats` - Statistical functions
