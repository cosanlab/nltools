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
    :no-members:
    :show-inheritance:
```

### Constructor

```{eval-rst}
.. automethod:: nltools.data.BrainData.__init__
```

### Properties

```{eval-rst}
.. autoattribute:: nltools.data.BrainData.shape
.. autoattribute:: nltools.data.BrainData.dtype
.. autoattribute:: nltools.data.BrainData.is_empty
```

### Data I/O

```{eval-rst}
.. automethod:: nltools.data.BrainData.to_nifti
.. automethod:: nltools.data.BrainData.resample_to
.. automethod:: nltools.data.BrainData.write
.. automethod:: nltools.data.BrainData.upload_neurovault
```

### Data Manipulation

```{eval-rst}
.. automethod:: nltools.data.BrainData.copy
.. automethod:: nltools.data.BrainData.append
.. automethod:: nltools.data.BrainData.create_empty
.. automethod:: nltools.data.BrainData.mean
.. automethod:: nltools.data.BrainData.median
.. automethod:: nltools.data.BrainData.std
.. automethod:: nltools.data.BrainData.sum
.. automethod:: nltools.data.BrainData.astype
```

### Preprocessing

```{eval-rst}
.. automethod:: nltools.data.BrainData.scale
.. automethod:: nltools.data.BrainData.standardize
.. automethod:: nltools.data.BrainData.smooth
.. automethod:: nltools.data.BrainData.filter
.. automethod:: nltools.data.BrainData.detrend
.. automethod:: nltools.data.BrainData.threshold
.. automethod:: nltools.data.BrainData.r_to_z
.. automethod:: nltools.data.BrainData.z_to_r
.. automethod:: nltools.data.BrainData.find_spikes
.. automethod:: nltools.data.BrainData.temporal_resample
```

### Masking and ROI Extraction

```{eval-rst}
.. automethod:: nltools.data.BrainData.apply_mask
.. automethod:: nltools.data.BrainData.extract_roi
.. automethod:: nltools.data.BrainData.regions
```

### Similarity and Distance

```{eval-rst}
.. automethod:: nltools.data.BrainData.similarity
.. automethod:: nltools.data.BrainData.distance
.. automethod:: nltools.data.BrainData.multivariate_similarity
.. automethod:: nltools.data.BrainData.transform_pairwise
.. automethod:: nltools.data.BrainData.icc
```

### Decomposition and Alignment

```{eval-rst}
.. automethod:: nltools.data.BrainData.decompose
.. automethod:: nltools.data.BrainData.align
```

### Statistical Modeling

```{eval-rst}
.. automethod:: nltools.data.BrainData.fit
.. automethod:: nltools.data.BrainData.compute_contrasts
.. automethod:: nltools.data.BrainData.cv
```

### Prediction and Decoding

```{eval-rst}
.. automethod:: nltools.data.BrainData.predict
```

### Bootstrap Inference

```{eval-rst}
.. automethod:: nltools.data.BrainData.bootstrap
```

### Visualization

```{eval-rst}
.. automethod:: nltools.data.BrainData.plot
.. automethod:: nltools.data.BrainData.plot_flatmap
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
