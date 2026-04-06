---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Adjacency Basics

The `Adjacency` class for connectivity and similarity matrices.

```{code-cell} python3
import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_1samp

from nltools.data import Adjacency
from nltools.datasets import fetch_haxby
```

## Create Adjacency Objects

```{code-cell} python3
# From brain data correlation
data, design = fetch_haxby(n_subjects=1)
brain = data[0][:20]  # First 20 timepoints
print(f"Data shape: {brain.shape}")

corr_matrix = brain.distance(metric="correlation")
print(f"Similarity matrix: {corr_matrix.shape}")
```

```{code-cell} python3
# From custom matrix
n_nodes = 10
random_matrix = np.random.randn(n_nodes, n_nodes)
random_matrix = (random_matrix + random_matrix.T) / 2
np.fill_diagonal(random_matrix, 0)

adj = Adjacency(data=random_matrix, matrix_type="similarity")
print(f"Custom adjacency: {adj.shape}")
```

## Visualization

```{code-cell} python3
adj.plot()
plt.title("Random Symmetric Matrix")
plt.close()
```

## Shape and Vectorized Storage

```{code-cell} python3
print(f"Logical shape: {adj.shape}")
print(f"Number of nodes: {adj.n_nodes}")
print(f"Vector shape: {adj.vector_shape}")

# Convert to square
square = adj.squareform()
print(f"Squareform: {square.shape}")
```

## Thresholding

```{code-cell} python3
n_nodes = 20
base = np.random.randn(n_nodes, n_nodes) * 0.3
structured = base + base.T
np.fill_diagonal(structured, 0)

adj_struct = Adjacency(data=structured, matrix_type="similarity")

# Absolute threshold
thresh = adj_struct.threshold(lower=0.3)
print(f"Edges above 0.3: {(thresh.data > 0).sum()}")

# Percentile threshold
thresh_pct = adj_struct.threshold(lower="90%")
print(f"Top 10% edges: {(thresh_pct.data > 0).sum()}")
```

## Graph Metrics

```{code-cell} python3
square_matrix = thresh.squareform()
degrees = np.sum(np.abs(square_matrix) > 0, axis=1)

print(f"Degrees: {degrees}")
print(f"Mean degree: {degrees.mean():.2f}")
```

## Functional Connectivity Example

```{code-cell} python3
n_rois = 5
n_timepoints = 100

# Simulate ROI timeseries
roi_timeseries = np.random.randn(n_timepoints, n_rois)

# Add correlation structure
roi_timeseries[:, 1] = roi_timeseries[:, 0] + np.random.randn(n_timepoints) * 0.3
roi_timeseries[:, 4] = roi_timeseries[:, 3] + np.random.randn(n_timepoints) * 0.3

fc_matrix = np.corrcoef(roi_timeseries.T)
np.fill_diagonal(fc_matrix, 0)

fc = Adjacency(data=fc_matrix, matrix_type="similarity")
print(f"FC shape: {fc.shape}")
print(f"Value range: [{fc.data.min():.2f}, {fc.data.max():.2f}]")

fc.plot()
plt.title("Functional Connectivity")
plt.close()
```

## RSA Example

```{code-cell} python3
n_conditions = 6
n_voxels = 1000

# Simulate patterns with structure
patterns = np.random.randn(n_conditions, n_voxels)
patterns[1] = patterns[0] + np.random.randn(n_voxels) * 0.2
patterns[2] = patterns[0] + np.random.randn(n_voxels) * 0.2
patterns[4] = patterns[3] + np.random.randn(n_voxels) * 0.2
patterns[5] = patterns[3] + np.random.randn(n_voxels) * 0.2

# Compute RDM (1 - correlation)
corr = np.corrcoef(patterns)
rdm = 1 - corr
np.fill_diagonal(rdm, 0)

rsa = Adjacency(data=rdm, matrix_type="distance")
print(f"RDM: {rsa.shape}")

rsa.plot()
plt.title("Representational Dissimilarity Matrix")
plt.close()
```

## Comparing Matrices

```{code-cell} python3
np.random.seed(42)
matrix1 = np.random.randn(20, 20)
matrix1 = (matrix1 + matrix1.T) / 2
np.fill_diagonal(matrix1, 0)

matrix2 = matrix1 + np.random.randn(20, 20) * 0.5
matrix2 = (matrix2 + matrix2.T) / 2
np.fill_diagonal(matrix2, 0)

adj1 = Adjacency(data=matrix1, matrix_type="similarity")
adj2 = Adjacency(data=matrix2, matrix_type="similarity")

r, p = pearsonr(adj1.data, adj2.data)
print(f"Matrix similarity: r = {r:.3f}, p = {p:.4e}")
```

## Statistical Testing

```{code-cell} python3
t_stat, p_val = ttest_1samp(adj1.data, 0)
print(f"Mean: {adj1.data.mean():.4f} (SD = {adj1.data.std():.4f})")
print(f"T-test vs 0: t = {t_stat:.2f}, p = {p_val:.4e}")
```
