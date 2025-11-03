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

## Performance

**v0.6.0 Changes**: DesignMatrix now uses Polars instead of pandas internally, providing:
- **2-5× faster** operations (especially statistics and concatenation)
- **Lower memory usage** (Apache Arrow format)
- **Better type safety** and error messages
- **Automatic conversion** to pandas for nilearn GLM compatibility

See the [Migration Guide](../migration-guide.md#designmatrix-pandas-polars) for details.

## Quick Start

```python
from nltools.data import DesignMatrix
import numpy as np

# Create from array or pandas DataFrame
dm = DesignMatrix(data=df)

# Add polynomial trends
dm = dm.add_poly(order=2, include_lower=True)

# Add DCT basis for filtering
dm = dm.add_dct_basis(duration=180)

# Convolve with HRF
convolved = dm.convolve(hrf_model='glover')

# Visualize
dm.heatmap()
```

## Full API Reference

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
- {doc}`../migration-guide` - Migration guide for v0.6.0 changes
