# Design_Matrix

Class for creating and manipulating design matrices for GLM analyses.

## Overview

`Design_Matrix` extends pandas DataFrame with neuroimaging-specific functionality for building and manipulating design matrices. It provides convenient methods for convolution with HRF, adding polynomial trends, discrete cosine transforms, and visualizing design matrices.

## Key Features

- **HRF convolution**: Convolve stimulus onsets with hemodynamic response
- **Polynomial trends**: Add linear, quadratic, or higher-order trends
- **DCT basis**: Add discrete cosine transform basis for high-pass filtering
- **Visualization**: Quick plotting of design matrix structure
- **Pandas compatible**: Inherits all pandas DataFrame methods
- **File I/O**: CSV, TSV, and other pandas-supported formats

## Quick Start

```python
from nltools.data import Design_Matrix
import numpy as np

# Create from pandas DataFrame
dm = Design_Matrix(data=df)

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
.. autoclass:: nltools.data.Design_Matrix
    :members:
    :undoc-members:
    :show-inheritance:

.. automethod:: __init__
```

## See Also

- {doc}`brain_data` - Brain imaging data for GLM
- {doc}`../algorithms` - HRF models and convolution
- {doc}`../filereader` - Reading onset files
