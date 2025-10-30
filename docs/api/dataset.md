# Datasets

Dataset loading utilities and example data.

## Overview

The `nltools.datasets` module provides functions for downloading and loading example neuroimaging datasets. These datasets are useful for testing, tutorials, and reproducing published analyses.

## Available Datasets

**fetch_pain** - Pain anticipation dataset
- Multisubject fMRI data
- Pain anticipation task
- Useful for prediction and classification examples

**fetch_emotion** - Emotion regulation dataset
- fMRI data from emotion regulation task
- Multiple conditions
- Good for GLM and contrast examples

**get_resource_path** - Access package resources
- Get paths to built-in masks and templates
- Access example data files

## Quick Start

```python
from nltools.datasets import fetch_pain, fetch_emotion

# Load pain dataset
pain_data = fetch_pain()
print(pain_data.shape)  # (n_voxels, n_images)

# Load emotion dataset
emotion_data = fetch_emotion()

# Use in analyses
pain_data.fit(model='ridge', X=design_matrix)
```

## Full API Reference

```{eval-rst}
.. automodule:: nltools.datasets
    :members:
    :undoc-members:
    :show-inheritance:
```

## See Also

- {doc}`data/brain_data` - Brain_Data class for loaded datasets
- {doc}`prefs` - MNI template resources