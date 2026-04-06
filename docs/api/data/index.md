---
title: Data Classes
---

# API Classes

The four core data classes in nltools. Each page documents the class constructor, all methods, and attributes.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} BrainData
:link: brain_data
The primary class for neuroimaging data — loading, indexing, arithmetic, statistics, modeling, and visualization.
:::

:::{grid-item-card} Adjacency
:link: adjacency
Symmetric and directed adjacency matrices stored in compact vector form — thresholding, similarity, regression, and graph operations.
:::

:::{grid-item-card} DesignMatrix
:link: design_matrix
Polars-based design matrices for GLM analysis — HRF convolution, polynomial drift terms, multi-run concatenation, and diagnostics.
:::

:::{grid-item-card} BrainCollection
:link: brain_collection
Collections of BrainData objects with 3-axis indexing — group-level aggregation, inference, and batch operations.
:::

::::
