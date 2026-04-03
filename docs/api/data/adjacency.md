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
    :no-members:
    :show-inheritance:
```

### Constructor

```{eval-rst}
.. automethod:: nltools.data.Adjacency.__init__
```

### Properties

```{eval-rst}
.. autoattribute:: nltools.data.Adjacency.shape
.. autoattribute:: nltools.data.Adjacency.vector_shape
.. autoattribute:: nltools.data.Adjacency.n_nodes
.. autoattribute:: nltools.data.Adjacency.is_empty
```

### Data Access

```{eval-rst}
.. automethod:: nltools.data.Adjacency.copy
.. automethod:: nltools.data.Adjacency.append
.. automethod:: nltools.data.Adjacency.squareform
.. automethod:: nltools.data.Adjacency.to_square
```

### Arithmetic

Adjacency supports element-wise arithmetic with scalars and other
Adjacency instances via ``+``, ``-``, ``*``, ``/``.

### Conversion

```{eval-rst}
.. automethod:: nltools.data.Adjacency.distance_to_similarity
.. automethod:: nltools.data.Adjacency.r_to_z
.. automethod:: nltools.data.Adjacency.z_to_r
```

### Basic Statistics

```{eval-rst}
.. automethod:: nltools.data.Adjacency.mean
.. automethod:: nltools.data.Adjacency.std
.. automethod:: nltools.data.Adjacency.median
.. automethod:: nltools.data.Adjacency.sum
.. automethod:: nltools.data.Adjacency.cluster_summary
```

### I/O

```{eval-rst}
.. automethod:: nltools.data.Adjacency.write
.. automethod:: nltools.data.Adjacency.to_graph
```

### Similarity & Distance

```{eval-rst}
.. automethod:: nltools.data.Adjacency.similarity
.. automethod:: nltools.data.Adjacency.distance
```

### Thresholding & Transforms

```{eval-rst}
.. automethod:: nltools.data.Adjacency.threshold
```

### Statistical Tests

```{eval-rst}
.. automethod:: nltools.data.Adjacency.ttest
.. automethod:: nltools.data.Adjacency.stats_label_distance
.. automethod:: nltools.data.Adjacency.plot_label_distance
.. automethod:: nltools.data.Adjacency.plot_silhouette
```

### Modeling & Inference

```{eval-rst}
.. automethod:: nltools.data.Adjacency.bootstrap
.. automethod:: nltools.data.Adjacency.regress
.. automethod:: nltools.data.Adjacency.social_relations_model
.. automethod:: nltools.data.Adjacency.generate_permutations
```

### Visualization

```{eval-rst}
.. automethod:: nltools.data.Adjacency.plot
.. automethod:: nltools.data.Adjacency.plot_mds
```

## Submodules

The `Adjacency` class delegates to standalone functions in focused submodules.
Each facade method on the class calls into these modules, which can also be
used directly:

- {doc}`adjacency_stats` -- Similarity, thresholding, t-tests, label distance, silhouette
- {doc}`adjacency_modeling` -- Bootstrap inference, OLS regression, SRM, permutation generation
- {doc}`adjacency_plotting` -- Heatmaps, MDS scatter plots
- {doc}`adjacency_io` -- CSV/HDF5 I/O, NetworkX graph conversion

## See Also

- {doc}`brain_data` - Brain imaging data
- {doc}`../stats` - Statistical functions for matrices
- {doc}`../analysis` - ROC analysis tools
