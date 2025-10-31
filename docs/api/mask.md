# `nltools.mask`

Functions for creating, manipulating, and applying brain masks.

## Overview

The `nltools.mask` module provides tools for working with brain masks and ROIs. These functions extend nilearn's masking capabilities with convenient wrappers for common neuroimaging tasks like sphere creation, mask expansion, and ROI manipulation.

## Key Functions

**Mask Creation**
- `create_sphere()` - Create spherical ROI masks from coordinates
- `expand_mask()` - Convert masked vector back to full brain image
- `collapse_mask()` - Extract data from image using mask

**Mask Manipulation**
- `roi_to_brain()` - Convert ROI to BrainData object
- `threshold()` - Threshold masks by intensity

## Quick Start

```python
from nltools.mask import create_sphere, expand_mask
from nltools.prefs import MNI_Template

# Create 10mm sphere at coordinates
mask = create_sphere([0, 0, 0], radius=10, mask=MNI_Template.mask)

# Apply mask to extract data
data_vector = collapse_mask(brain_image, mask)

# Convert back to full brain
brain_image = expand_mask(data_vector, mask)
```

## Full API Reference

```{eval-rst}
.. automodule:: nltools.mask
    :members:
    :undoc-members:
    :show-inheritance:
```

## See Also

- {doc}`data/brain_data` - BrainData.apply_mask() and .extract_roi()
- {doc}`prefs` - MNI template resources
- nilearn.masking - Underlying masking functions