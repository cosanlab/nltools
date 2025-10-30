# Adjacency

Class for representing similarity and distance matrices.

## Overview

`Adjacency` is a specialized data structure for working with symmetric similarity or distance matrices. It handles the storage, manipulation, and visualization of pairwise relationships between observations (e.g., subjects, stimuli, brain regions).

## Key Features

- **Storage**: Efficient storage as 1D vector (upper/lower triangle)
- **Operations**: Arithmetic, thresholding, statistical tests
- **Visualization**: Heatmaps, network graphs, dendrograms
- **Statistics**: Permutation tests, distance correlations
- **Transformations**: Distance ↔ similarity conversions
- **File I/O**: Save/load matrices with labels

## Quick Start

```python
from nltools.data import Adjacency, Brain_Data

# Create from pairwise distances
data = Brain_Data('brain_images.nii.gz')
similarity = data.distance(metric='correlation')

# Threshold and visualize
thresholded = similarity.threshold(upper=0.8)
thresholded.plot()

# Statistical testing
stats = similarity.ttest(permutation=True, n_permutations=5000)
```

## Full API Reference

```{eval-rst}
.. autoclass:: nltools.data.Adjacency
    :members:
    :undoc-members:
    :show-inheritance:

.. automethod:: __init__
```

## See Also

- {doc}`brain_data` - Brain imaging data
- {doc}`../stats` - Statistical functions for matrices
- {doc}`../analysis` - ROC analysis tools
