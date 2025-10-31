# Adjacency Basics

## Learning Objectives

By the end of this tutorial, you will be able to:
- Create `Adjacency` objects for representing connectivity matrices
- Compute correlation and similarity matrices
- Threshold and manipulate connectivity matrices
- Visualize network structures
- Perform basic graph-theoretic analyses
- Convert between different network representations

## Introduction

The `Adjacency` class in nltools represents connectivity or similarity matrices between brain regions or images. Common use cases:
- **Functional connectivity**: Correlations between ROI timeseries
- **Representational similarity**: Pattern similarity matrices (RSA)
- **Behavioral similarity**: Subject similarity based on traits
- **Spatial similarity**: Structural covariance networks

**This tutorial covers the fundamentals of working with Adjacency matrices.**

## Creating Adjacency Objects

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltools.data import Adjacency, BrainData
from nltools.datasets import fetch_pain

# Adjacency objects can be created from:
# 1. Square symmetric matrices
# 2. Vectorized upper/lower triangles
# 3. Correlation/covariance calculations

# Load example data
data = fetch_pain()
print(f"Data shape: {data.shape()}")
```

## Method 1: From Correlation Matrix

```python
# Compute pairwise correlations between images
# This creates a similarity matrix: how similar is each image to every other?

corr_matrix = data.similarity(metric='correlation')

print(f"Similarity matrix type: {type(corr_matrix)}")
print(f"Shape: {corr_matrix.shape()}")

# Visualize correlation matrix
corr_matrix.plot(figsize=(10, 8), vmin=-1, vmax=1)
plt.title('Intersubject Correlation Matrix')
plt.show()
```

## Method 2: From Distance Matrix

```python
# Compute pairwise distances (dissimilarity)

dist_matrix = data.similarity(metric='euclidean')

print(f"Distance matrix shape: {dist_matrix.shape()}")

# Visualize
dist_matrix.plot(figsize=(10, 8))
plt.title('Pairwise Distance Matrix')
plt.show()
```

## Method 3: From Custom Matrix

```python
# Create adjacency from pre-computed matrix

n_nodes = 10
# Generate random symmetric matrix
random_matrix = np.random.randn(n_nodes, n_nodes)
random_matrix = (random_matrix + random_matrix.T) / 2  # Make symmetric

# Create Adjacency object
adj = Adjacency(data=random_matrix, matrix_type='similarity')

print(f"Custom adjacency: {adj.shape()}")
adj.plot()
plt.title('Random Symmetric Matrix')
plt.show()
```

## Indexing and Slicing

```python
# Index specific elements
print(f"Element [0, 1]: {corr_matrix[0, 1]}")

# Slice rows/columns
subset = corr_matrix[:10, :10]
print(f"Subset shape: {subset.shape()}")

# Extract diagonal
diag_vals = np.diag(corr_matrix.squareform())
print(f"Diagonal values (should be 1.0 for correlation): {diag_vals[:5]}")
```

## Vectorized Representation

```python
# Adjacency matrices are symmetric, so we only need upper/lower triangle

# Get vectorized form (upper triangle, no diagonal)
vector = corr_matrix.data

print(f"Full matrix shape: {corr_matrix.squareform().shape}")
print(f"Vectorized shape: {vector.shape}")
print(f"Relationship: n*(n-1)/2 = {84*83//2}")

# Convert back to square matrix
square = corr_matrix.squareform()
print(f"Square form: {square.shape}")
```

## Thresholding

```python
# Threshold weak connections

# Absolute threshold
thresh_abs = corr_matrix.threshold(lower=0.3)
print(f"Edges with r > 0.3: {(thresh_abs.data > 0).sum()}")

# Percentile threshold (keep top 10% of connections)
thresh_pct = corr_matrix.threshold(upper='90%')
print(f"Top 10% of connections: {(thresh_pct.data > 0).sum()}")

# Visualize thresholded matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

corr_matrix.plot(axes=axes[0], vmin=-1, vmax=1)
axes[0].set_title('Original Correlation Matrix')

thresh_abs.plot(axes=axes[1], vmin=0, vmax=1)
axes[1].set_title('Thresholded (r > 0.3)')

plt.tight_layout()
plt.show()
```

## Graph Metrics

```python
# Compute basic graph properties

# Degree: Number of connections per node
degrees = np.sum(thresh_abs.squareform() > 0, axis=1)

print(f"Node degrees: {degrees}")
print(f"Mean degree: {degrees.mean():.2f}")
print(f"Max degree: {degrees.max()}")

# Visualize degree distribution
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(degrees, bins=20, edgecolor='black', alpha=0.7)
ax.set_xlabel('Node Degree')
ax.set_ylabel('Frequency')
ax.set_title('Degree Distribution')
plt.show()
```

## Clustering Coefficient

```python
# Clustering coefficient: How interconnected are a node's neighbors?

# TODO: Implement clustering coefficient calculation
# clustering = thresh_abs.clustering_coefficient()

print("Note: Advanced graph metrics to be implemented")
print("For now, use networkx or igraph for detailed graph analysis")
```

## Network Visualization

```python
# Visualize network as graph

# For publication-quality network plots, export to external tools
# Export as adjacency list or matrix for networkx/igraph

# Simple heatmap visualization
fig, ax = plt.subplots(figsize=(10, 8))

# Reorder matrix by hierarchical clustering
# TODO: Implement reordering
# thresh_abs_sorted = thresh_abs.reorder(method='hierarchical')

# For now, just plot
thresh_abs.plot(axes=ax, vmin=0, vmax=1)
ax.set_title('Thresholded Network (r > 0.3)')
plt.show()
```

## Community Detection

```python
# Detect communities (clusters of highly interconnected nodes)

# TODO: Implement community detection
# communities = thresh_abs.detect_communities(method='louvain')

print("Note: Community detection to be implemented")
print("Use community detection algorithms from networkx or python-louvain")
```

## Applications: ROI-to-ROI Connectivity

```python
# Compute functional connectivity between brain regions

# In practice:
# 1. Extract ROI timeseries for each region
# 2. Compute pairwise correlations
# 3. Create Adjacency object

# Simulate 5 ROI timeseries
n_rois = 5
n_timepoints = 100

roi_timeseries = np.random.randn(n_timepoints, n_rois)

# Add some structure (ROIs 0-1 correlated, ROIs 3-4 correlated)
roi_timeseries[:, 1] = roi_timeseries[:, 0] + np.random.randn(n_timepoints) * 0.3
roi_timeseries[:, 4] = roi_timeseries[:, 3] + np.random.randn(n_timepoints) * 0.3

# Compute correlation matrix
fc_matrix = np.corrcoef(roi_timeseries.T)

# Create Adjacency object
fc = Adjacency(data=fc_matrix, matrix_type='similarity')

# Visualize
fc.plot(figsize=(6, 5))
plt.title('ROI-to-ROI Functional Connectivity')

# Label ROIs
roi_labels = ['DLPFC_L', 'DLPFC_R', 'ACC', 'Insula_L', 'Insula_R']
plt.xticks(range(n_rois), roi_labels, rotation=45, ha='right')
plt.yticks(range(n_rois), roi_labels)
plt.tight_layout()
plt.show()
```

## Applications: Representational Similarity Analysis (RSA)

```python
# Measure pattern similarity for different conditions

# Simulate neural patterns for different stimuli
n_conditions = 6
n_voxels = 1000

patterns = np.random.randn(n_conditions, n_voxels)

# Create structure: conditions 0-2 similar, conditions 3-5 similar
patterns[1] = patterns[0] + np.random.randn(n_voxels) * 0.2
patterns[2] = patterns[0] + np.random.randn(n_voxels) * 0.2
patterns[4] = patterns[3] + np.random.randn(n_voxels) * 0.2
patterns[5] = patterns[3] + np.random.randn(n_voxels) * 0.2

# Compute representational dissimilarity matrix (RDM)
rdm = 1 - np.corrcoef(patterns)  # Dissimilarity = 1 - correlation

# Create Adjacency object
rsa = Adjacency(data=rdm, matrix_type='distance')

# Visualize RDM
rsa.plot(figsize=(7, 6))
plt.title('Representational Dissimilarity Matrix')

condition_labels = ['Faces1', 'Faces2', 'Faces3', 'Objects1', 'Objects2', 'Objects3']
plt.xticks(range(n_conditions), condition_labels, rotation=45, ha='right')
plt.yticks(range(n_conditions), condition_labels)
plt.tight_layout()
plt.show()

# The block structure shows faces are similar to faces, objects to objects
```

## Comparing Adjacency Matrices

```python
# Compare two connectivity matrices (e.g., different conditions or groups)

# Create two matrices with different structure
np.random.seed(42)
matrix1 = np.random.randn(20, 20)
matrix1 = (matrix1 + matrix1.T) / 2

matrix2 = matrix1 + np.random.randn(20, 20) * 0.5
matrix2 = (matrix2 + matrix2.T) / 2

adj1 = Adjacency(data=matrix1, matrix_type='similarity')
adj2 = Adjacency(data=matrix2, matrix_type='similarity')

# Compute correlation between vectorized matrices
from scipy.stats import pearsonr
r, p = pearsonr(adj1.data, adj2.data)

print(f"Matrix similarity: r = {r:.3f}, p = {p:.4e}")

# Visualize difference
diff = Adjacency(data=matrix1 - matrix2, matrix_type='directed')

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

adj1.plot(axes=axes[0])
axes[0].set_title('Matrix 1')

adj2.plot(axes=axes[1])
axes[1].set_title('Matrix 2')

diff.plot(axes=axes[2], vmin=-2, vmax=2)
axes[2].set_title('Difference (1 - 2)')

plt.tight_layout()
plt.show()
```

## Save and Load

```python
# Save adjacency matrix
corr_matrix.write('similarity_matrix.csv')

# Load adjacency matrix
loaded = Adjacency('similarity_matrix.csv', matrix_type='similarity')

print(f"Loaded matrix shape: {loaded.shape()}")
assert np.allclose(corr_matrix.data, loaded.data), "Loaded data should match"
```

## Sanity Checks

```python
# Verify adjacency properties

# Check symmetry (for undirected networks)
square_matrix = corr_matrix.squareform()
assert np.allclose(square_matrix, square_matrix.T), "Matrix should be symmetric"

# Check diagonal (should be 1.0 for correlation)
diag = np.diag(square_matrix)
assert np.allclose(diag, 1.0), "Diagonal should be 1.0 for correlation matrix"

# Check bounds for correlation
assert (corr_matrix.data >= -1).all() and (corr_matrix.data <= 1).all(), \
    "Correlations should be in [-1, 1]"

# Check vectorization
n_nodes = corr_matrix.shape()[0]
expected_length = n_nodes * (n_nodes - 1) // 2
assert len(corr_matrix.data) == expected_length, \
    f"Vector length should be {expected_length}"

print("✓ All sanity checks passed!")
```

## Summary

In this tutorial, you learned how to:
- ✓ Create `Adjacency` objects from matrices or similarity calculations
- ✓ Work with vectorized representations of symmetric matrices
- ✓ Threshold connectivity matrices
- ✓ Compute basic graph metrics (degree, clustering)
- ✓ Visualize networks with heatmaps
- ✓ Apply adjacency to functional connectivity and RSA
- ✓ Compare connectivity matrices across conditions

## Next Steps

- **Tutorial: Connectivity Analysis** - Advanced network analyses
- **Tutorial: Representational Similarity** - Deep dive into RSA methods
- **Advanced**: Dynamic connectivity and time-varying networks
- **Advanced**: Graph-theoretic measures (modularity, efficiency, etc.)

## Clean Up

```python
import os
if os.path.exists('similarity_matrix.csv'):
    os.remove('similarity_matrix.csv')
```
