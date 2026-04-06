---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Adjacency Basics

## Learning Objectives

By the end of this tutorial, you will be able to:
- Create `Adjacency` objects from matrices and vectorized representations
- Understand the relationship between square and vector forms
- Threshold and binarize connectivity matrices
- Visualize matrices as heatmaps and MDS plots
- Compare matrices with permutation-based similarity tests
- Apply Adjacency to functional connectivity and RSA

## Introduction

The `Adjacency` class represents connectivity or similarity matrices. It stores data efficiently as the upper triangle vector and reconstructs the full square matrix on demand.

Common use cases:
- **Functional connectivity**: Correlations between ROI timeseries
- **Representational similarity**: Pattern similarity matrices (RSA)
- **Behavioral similarity**: Subject-level similarity based on traits or responses

`Adjacency` supports two matrix types: `"similarity"` (higher = more similar) and `"distance"` (higher = more dissimilar).

```{code-cell} python3
import numpy as np
import matplotlib.pyplot as plt

from nltools.data import Adjacency
```

## Creating Adjacency Objects

### From a Square Matrix

```{code-cell} python3
n_nodes = 10

# Generate a random symmetric matrix
random_matrix = np.random.randn(n_nodes, n_nodes)
random_matrix = (random_matrix + random_matrix.T) / 2
np.fill_diagonal(random_matrix, 0)

adj = Adjacency(data=random_matrix, matrix_type="similarity")
print(adj)
```

### From Brain Data

Compute pairwise distances between brain images:

```{code-cell} python3
from nltools.datasets import fetch_pain

data = fetch_pain()

# Use a subset for speed
subset = data[:20]
dist_matrix = subset.distance(metric="correlation")
print(f"Distance matrix: {dist_matrix.shape}")
```

## Shape and Storage

`Adjacency` distinguishes between the logical shape (square matrix) and the stored vector:

```{code-cell} python3
print(f"Logical shape: {adj.shape}")
print(f"Number of nodes: {adj.n_nodes}")
print(f"Vector length: {adj.vector_shape}")
print(f"Expected: n*(n-1)/2 = {n_nodes * (n_nodes - 1) // 2}")
```

Use `squareform()` to get the full matrix:

```{code-cell} python3
square = adj.squareform()
print(f"Square matrix: {square.shape}")
print(f"Symmetric: {np.allclose(square, square.T)}")
```

## Visualization

### Heatmap

```{code-cell} python3
adj.plot()
plt.title("Random Similarity Matrix")
plt.show()
```

### With Labels

```{code-cell} python3
# Create a labeled adjacency
roi_names = [f"ROI_{i}" for i in range(n_nodes)]
adj_labeled = Adjacency(data=random_matrix, matrix_type="similarity", labels=roi_names)
adj_labeled.plot()
plt.title("Labeled Matrix")
plt.show()
```

### MDS Plot

Multidimensional scaling visualizes the structure of a distance matrix in 2D:

```{code-cell} python3
dist_matrix.plot_mds(n_components=2, figsize=(6, 5))
plt.title("MDS of Image Distances")
plt.show()
```

## Thresholding

Remove weak connections by absolute value or percentile:

```{code-cell} python3
# Absolute threshold: keep edges > 0.3
thresh = adj.threshold(lower=0.3)
print(f"Edges above 0.3: {(thresh.data > 0).sum()} / {len(thresh.data)}")

# Percentile threshold: keep top 10%
thresh_pct = adj.threshold(lower="90%")
print(f"Top 10% edges: {(thresh_pct.data > 0).sum()}")

# Binarize: convert to binary connectivity
binary = adj.threshold(lower=0.3, binarize=True)
print(f"Binary values: {np.unique(binary.data)}")
```

```{code-cell} python3
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

adj.plot(axes=axes[0])
axes[0].set_title("Original")

thresh.plot(axes=axes[1])
axes[1].set_title("Thresholded (> 0.3)")

binary.plot(axes=axes[2])
axes[2].set_title("Binarized")

plt.tight_layout()
plt.show()
```

## Statistics

### Summary Statistics

```{code-cell} python3
print(f"Mean: {adj.mean():.4f}")
print(f"Std: {adj.std():.4f}")
print(f"Median: {adj.median():.4f}")
```

### Comparing Two Matrices

Use `similarity()` to test whether two matrices are related, with permutation-based inference:

```{code-cell} python3
# Create two related matrices
np.random.seed(42)
m1 = np.random.randn(15, 15)
m1 = (m1 + m1.T) / 2
np.fill_diagonal(m1, 0)

m2 = m1 + np.random.randn(15, 15) * 0.5
m2 = (m2 + m2.T) / 2
np.fill_diagonal(m2, 0)

adj1 = Adjacency(m1, matrix_type="similarity")
adj2 = Adjacency(m2, matrix_type="similarity")

# Permutation test for matrix similarity
result = adj1.similarity(adj2, metric="spearman", n_permute=5000)
print(f"Spearman r = {result['correlation']:.3f}, p = {result['p']:.4f}")
```

## Fisher's r-to-z Transform

When averaging or comparing correlation matrices, apply the Fisher transform first:

```{code-cell} python3
# Create a correlation matrix
corr_data = np.corrcoef(np.random.randn(8, 50))
np.fill_diagonal(corr_data, 0)
corr_adj = Adjacency(corr_data, matrix_type="similarity")

# Transform to z-scores for statistical testing
z_adj = corr_adj.r_to_z()
print(f"Original range: [{corr_adj.data.min():.3f}, {corr_adj.data.max():.3f}]")
print(f"Z-scored range: [{z_adj.data.min():.3f}, {z_adj.data.max():.3f}]")

# Transform back
r_adj = z_adj.z_to_r()
print(f"Round-trip check: {np.allclose(corr_adj.data, r_adj.data, atol=1e-10)}")
```

## Arithmetic

`Adjacency` supports element-wise arithmetic:

```{code-cell} python3
diff = adj1 - adj2
scaled = adj1 * 2
summed = adj1 + adj2

print(f"Difference mean: {diff.mean():.4f}")
print(f"Scaled mean: {scaled.mean():.4f}")
```

## Application: Functional Connectivity

```{code-cell} python3
# Simulate 5 ROI timeseries with correlation structure
np.random.seed(0)
n_rois = 5
n_timepoints = 100

roi_ts = np.random.randn(n_timepoints, n_rois)
# Make ROIs 0-1 correlated and ROIs 3-4 correlated
roi_ts[:, 1] = roi_ts[:, 0] + np.random.randn(n_timepoints) * 0.3
roi_ts[:, 4] = roi_ts[:, 3] + np.random.randn(n_timepoints) * 0.3

fc_matrix = np.corrcoef(roi_ts.T)
np.fill_diagonal(fc_matrix, 0)

roi_labels = ["DLPFC_L", "DLPFC_R", "ACC", "Insula_L", "Insula_R"]
fc = Adjacency(fc_matrix, matrix_type="similarity", labels=roi_labels)

fc.plot()
plt.title("ROI-to-ROI Functional Connectivity")
plt.show()
```

## Application: Representational Similarity Analysis

```{code-cell} python3
# Simulate neural patterns with category structure
np.random.seed(1)
n_conditions = 6
n_voxels = 1000

patterns = np.random.randn(n_conditions, n_voxels)
# Faces (0-2) similar to each other, objects (3-5) similar to each other
patterns[1] = patterns[0] + np.random.randn(n_voxels) * 0.2
patterns[2] = patterns[0] + np.random.randn(n_voxels) * 0.2
patterns[4] = patterns[3] + np.random.randn(n_voxels) * 0.2
patterns[5] = patterns[3] + np.random.randn(n_voxels) * 0.2

# Representational dissimilarity matrix
rdm = 1 - np.corrcoef(patterns)
np.fill_diagonal(rdm, 0)

condition_labels = ["Face1", "Face2", "Face3", "Object1", "Object2", "Object3"]
rsa = Adjacency(rdm, matrix_type="distance", labels=condition_labels)

rsa.plot()
plt.title("Representational Dissimilarity Matrix")
plt.show()
```

Notice the block-diagonal structure: faces are similar to faces (low dissimilarity), objects to objects.

## Stacking Multiple Matrices

Use `append()` to stack matrices (e.g., one per subject) for group-level analysis:

```{code-cell} python3
# Simulate FC matrices for 5 subjects
matrices = []
for _ in range(5):
    ts = np.random.randn(100, 5)
    ts[:, 1] = ts[:, 0] + np.random.randn(100) * 0.3
    fc = np.corrcoef(ts.T)
    np.fill_diagonal(fc, 0)
    matrices.append(Adjacency(fc, matrix_type="similarity"))

# Stack into a single object
stacked = matrices[0]
for m in matrices[1:]:
    stacked = stacked.append(m)

print(f"Stacked: {len(stacked)} matrices, {stacked.n_nodes} nodes each")

# Group mean
group_mean = stacked.mean()
print(f"Group mean shape: {group_mean.shape}")

# One-sample t-test across subjects (requires stacked matrices)
ttest_result = stacked.ttest()
print(f"T-test: {(ttest_result['p'].data < 0.05).sum()} significant edges (p < 0.05)")
```

## File I/O

```{code-cell} python3
import os
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    # Save as square matrix CSV
    path = os.path.join(tmpdir, "adjacency.csv")
    adj.write(path, method="square")
    print(f"Saved: {adj.shape}")

    # Load back
    loaded = Adjacency(path, matrix_type="similarity")
    print(f"Loaded: {loaded.shape}")
    print(f"Round-trip check: {np.allclose(adj.data, loaded.data)}")
```

## Summary

In this tutorial you learned:
- **Creating**: from square matrices, vectors, or `BrainData.distance()`
- **Storage**: upper triangle vector with `squareform()` reconstruction
- **Visualization**: `plot()` heatmaps, `plot_mds()` for structure
- **Thresholding**: absolute, percentile, and binarization
- **Statistics**: `mean()`, `ttest()`, `similarity()` with permutation tests
- **Transforms**: `r_to_z()` / `z_to_r()` for Fisher transforms
- **Stacking**: `append()` for group-level analysis
- **Applications**: functional connectivity and RSA

Next, explore [BrainCollection](04_brain_collection.md) for multi-subject analyses, or see the [RSA workflow](../workflows/04_rsa.md) for a complete representational similarity analysis.
