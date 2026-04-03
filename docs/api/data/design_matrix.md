# `nltools.data.DesignMatrix`

Class for creating and manipulating design matrices for GLM analyses.

## Overview

`DesignMatrix` provides neuroimaging-specific functionality for building and manipulating design matrices. It uses Polars DataFrames internally (v0.6.0+) for improved performance (2-5x speedup) while maintaining backward compatibility with pandas-style operations.

## Key Features

- **HRF convolution**: Convolve stimulus onsets with hemodynamic response
- **Polynomial trends**: Add linear, quadratic, or higher-order trends
- **DCT basis**: Add discrete cosine transform basis for high-pass filtering
- **Visualization**: Quick plotting of design matrix structure
- **Polars backend**: Fast operations with Polars (v0.6.0+)
- **Pandas compatible**: Automatic conversion to pandas for nilearn compatibility
- **File I/O**: CSV, TSV, and other supported formats

## API Reference

```{eval-rst}
.. autoclass:: nltools.data.DesignMatrix
    :no-members:
    :show-inheritance:
```

### Constructor

```{eval-rst}
.. automethod:: nltools.data.DesignMatrix.__init__
```

### Properties

```{eval-rst}
.. autoattribute:: nltools.data.DesignMatrix.shape
.. autoattribute:: nltools.data.DesignMatrix.columns
.. autoattribute:: nltools.data.DesignMatrix.is_empty
.. autoattribute:: nltools.data.DesignMatrix.empty
```

### Data Access

```{eval-rst}
.. automethod:: nltools.data.DesignMatrix.__getitem__
.. automethod:: nltools.data.DesignMatrix.__setitem__
.. automethod:: nltools.data.DesignMatrix.__len__
```

### Data Manipulation

```{eval-rst}
.. automethod:: nltools.data.DesignMatrix.copy
.. automethod:: nltools.data.DesignMatrix.drop
.. automethod:: nltools.data.DesignMatrix.fillna
.. automethod:: nltools.data.DesignMatrix.replace_data
.. automethod:: nltools.data.DesignMatrix.reset_index
.. automethod:: nltools.data.DesignMatrix.sum
```

### Transformations

```{eval-rst}
.. automethod:: nltools.data.DesignMatrix.zscore
.. automethod:: nltools.data.DesignMatrix.standardize
.. automethod:: nltools.data.DesignMatrix.downsample
.. automethod:: nltools.data.DesignMatrix.upsample
```

### HRF & Regressors

```{eval-rst}
.. automethod:: nltools.data.DesignMatrix.convolve
.. automethod:: nltools.data.DesignMatrix.add_poly
.. automethod:: nltools.data.DesignMatrix.add_dct_basis
```

### Concatenation

```{eval-rst}
.. automethod:: nltools.data.DesignMatrix.append
```

### Diagnostics

```{eval-rst}
.. automethod:: nltools.data.DesignMatrix.vif
.. automethod:: nltools.data.DesignMatrix.clean
.. automethod:: nltools.data.DesignMatrix.details
```

### Visualization & I/O

```{eval-rst}
.. automethod:: nltools.data.DesignMatrix.heatmap
.. automethod:: nltools.data.DesignMatrix.to_pandas
.. automethod:: nltools.data.DesignMatrix.to_numpy
.. automethod:: nltools.data.DesignMatrix.write
```

## Submodules

The `DesignMatrix` class delegates to standalone functions in focused submodules.
Each facade method on the class calls into these modules, which can also be
used directly:

- {doc}`design_matrix_transforms` -- Z-scoring, standardization, downsampling, upsampling
- {doc}`design_matrix_regressors` -- HRF convolution, polynomial trends, DCT basis
- {doc}`design_matrix_append` -- Horizontal and vertical concatenation, multi-run separation
- {doc}`design_matrix_diagnostics` -- VIF, collinearity cleaning, summaries
- {doc}`design_matrix_io` -- Heatmap visualization, pandas/numpy conversion, file I/O

## See Also

- {doc}`brain_data` - Brain imaging data for GLM
- {doc}`../algorithms` - HRF models and convolution
- {doc}`../filereader` - Reading onset files
- {doc}`../../migration-guide` - Migration guide for v0.6.0 changes
