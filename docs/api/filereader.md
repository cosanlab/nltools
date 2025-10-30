# File Readers

Utilities for reading neuroimaging-related file formats.

## Overview

The `nltools.file_reader` module provides functions for reading various file formats commonly used in neuroimaging analyses, with a focus on converting stimulus timing files into design matrices.

## Key Functions

**onsets_to_dm** - Convert onset files to design matrices
- Read stimulus timing files (e.g., from E-Prime, PsychoPy)
- Support for multiple file formats (CSV, TSV, text)
- Automatic creation of Design_Matrix objects
- Handle event durations and amplitudes

## Quick Start

```python
from nltools.file_reader import onsets_to_dm

# Read onset file and create design matrix
dm = onsets_to_dm(
    'stimulus_onsets.csv',
    TR=2.0,
    n_volumes=200,
    header=True
)

# Convolve with HRF
dm_convolved = dm.convolve()

# Use in GLM
data.fit(model='glm', X=dm_convolved)
```

## Full API Reference

```{eval-rst}
.. automodule:: nltools.file_reader
    :members:
    :undoc-members:
    :show-inheritance:
```

## See Also

- {doc}`data/design_matrix` - Design_Matrix class
- {doc}`algorithms` - HRF models for convolution
- {doc}`data/brain_data` - Brain_Data.fit() for GLM