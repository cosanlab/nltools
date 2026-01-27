# `nltools.data.DesignMatrix`

Class for creating and manipulating design matrices for GLM analyses.

## Overview

`DesignMatrix` provides neuroimaging-specific functionality for building and manipulating design matrices. It uses Polars DataFrames internally (v0.6.0+) for improved performance (2-5× speedup) while maintaining backward compatibility with pandas-style operations.

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
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

## See Also

- {doc}`brain_data` - Brain imaging data for GLM
- {doc}`../algorithms` - HRF models and convolution
- {doc}`../filereader` - Reading onset files
- {doc}`../../migration-guide` - Migration guide for v0.6.0 changes
