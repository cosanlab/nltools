# `nltools.utils`

Helper functions for data validation, file operations, and common tasks.

## Overview

The `nltools.utils` module provides utility functions that support the core functionality of nltools. These include data validation, type checking, file path resolution, concatenation helpers, and bootstrap utilities.

## Key Functions

**Data Validation**
- `check_brain_data()` - Validate BrainData instances
- `check_brain_data_is_single()` - Check for single image
- `check_square_numpy_matrix()` - Validate square matrices

**File Operations**
- `get_resource_path()` - Resolve paths to package resources
- `attempt_to_import()` - Safe optional dependency import

**Data Manipulation**
- `concatenate()` - Concatenate BrainData objects
- `set_decomposition_algorithm()` - Configure decomposition backend

**Bootstrap Utilities**
- `_bootstrap_apply_func()` - Apply function to bootstrap samples
- `summarize_bootstrap()` - Summarize bootstrap distributions

**HDF5 Support**
- `to_h5()` - Write data to HDF5 format

## Quick Start

```python
from nltools.utils import (
    check_brain_data,
    concatenate,
    get_resource_path
)

# Validate BrainData
check_brain_data(data)

# Concatenate multiple BrainData objects
combined = concatenate([data1, data2, data3])

# Get path to package resource
mask_path = get_resource_path('MNI152_T1_2mm_brain_mask.nii.gz')
```

## Full API Reference

```{eval-rst}
.. automodule:: nltools.utils
    :members:
    :undoc-members:
    :show-inheritance:
```

## See Also

- {doc}`data/brain_data` - Main data class
- {doc}`stats` - Statistical functions
- {doc}`prefs` - MNI template preferences