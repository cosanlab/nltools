# `nltools.data.Adjacency`

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

## API Reference

```{eval-rst}
.. autoclass:: nltools.data.Adjacency
    :members:
    :undoc-members:
    :show-inheritance:
    :special-members: __init__
```

## See Also

- {doc}`brain_data` - Brain imaging data
- {doc}`../stats` - Statistical functions for matrices
- {doc}`../analysis` - ROC analysis tools
