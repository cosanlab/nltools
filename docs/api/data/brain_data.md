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

## API Reference

```{eval-rst}
.. autoclass:: nltools.data.BrainData
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

## Submodules

The `BrainData` class delegates to standalone functions in focused submodules.
Each facade method on the class calls into these modules, which can also be
used directly:

- {doc}`braindata_io` -- Loading, mask initialization, file I/O
- {doc}`braindata_analysis` -- Similarity, thresholding, smoothing, decomposition, alignment, etc.
- {doc}`braindata_modeling` -- Model fitting (Ridge, GLM), contrasts
- {doc}`braindata_prediction` -- MVPA decoding, encoding model prediction
- {doc}`braindata_bootstrap` -- Bootstrap inference
- {doc}`braindata_plotting` -- Visualization (glass brain, flatmap, timeseries)
- {doc}`braindata_pipeline` -- BrainDataPipeline and BrainDataCVResult

## See Also

- {doc}`adjacency` - Similarity/distance matrices
- {doc}`design_matrix` - Design matrices for GLM
- {doc}`../models` - Statistical models (Ridge, etc.)
- {doc}`../stats` - Statistical functions
- {doc}`../../migration-guide` - Migration guide for v0.6.0 changes
