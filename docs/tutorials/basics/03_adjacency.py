# %% [markdown]
# # Adjacency Basics
#
# This tutorial covers the `Adjacency` class for connectivity and similarity matrices.
#
# ## Learning Objectives
#
# By the end of this tutorial, you will be able to:
# - Create `Adjacency` objects for representing connectivity matrices
# - Compute correlation and similarity matrices
# - Threshold and manipulate connectivity matrices
# - Visualize network structures
# - Perform basic graph-theoretic analyses
# - Convert between different network representations

# %% [markdown]
# ## Introduction
#
# The `Adjacency` class in nltools represents connectivity or similarity matrices
# between brain regions or images. Common use cases:
# - **Functional connectivity**: Correlations between ROI timeseries
# - **Representational similarity**: Pattern similarity matrices (RSA)
# - **Behavioral similarity**: Subject similarity based on traits
# - **Spatial similarity**: Structural covariance networks

# %%
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

import numpy as np
import matplotlib.pyplot as plt

from nltools.data import Adjacency
from nltools.datasets import fetch_haxby

# %% [markdown]
# ## Creating Adjacency Objects
#
# Adjacency objects can be created from:
# 1. Square symmetric matrices
# 2. Vectorized upper/lower triangles
# 3. Correlation/covariance calculations from BrainData

# %%
# Load example data
data, design = fetch_haxby(n_subjects=1)
brain = data[0][:20]  # Use first 20 timepoints for speed

print(f"Data shape: {brain.shape}")

# %% [markdown]
# ## Method 1: From Correlation Matrix

# %%
# Compute pairwise correlations between images
# This creates a similarity matrix: how similar is each image to every other?
corr_matrix = brain.distance(metric="correlation")

print(f"Similarity matrix type: {type(corr_matrix)}")
print(f"Shape: {corr_matrix.shape}")

# %% [markdown]
# ## Method 2: From Custom Matrix

# %%
# Create adjacency from pre-computed matrix
n_nodes = 10

# Generate random symmetric matrix
random_matrix = np.random.randn(n_nodes, n_nodes)
random_matrix = (random_matrix + random_matrix.T) / 2  # Make symmetric
np.fill_diagonal(random_matrix, 0)  # Zero diagonal for distance matrix

# Create Adjacency object
adj = Adjacency(data=random_matrix, matrix_type="similarity")

print(f"Custom adjacency shape: {adj.shape}")

# %% [markdown]
# ## Visualizing Adjacency Matrices

# %%
# Plot the similarity matrix
adj.plot()
plt.title("Random Symmetric Matrix")
plt.close()

# %% [markdown]
# ## Shape and Vectorized Representation
#
# Adjacency matrices are symmetric, so internally we store only the upper triangle
# as a vector. The `.shape` property returns the logical `(n_nodes, n_nodes)` shape,
# while `.vector_shape` shows the internal storage.

# %%
# Shape returns (n_nodes, n_nodes) - the logical matrix dimensions
print(f"Logical shape: {adj.shape}")
print(f"Number of nodes: {adj.n_nodes}")

# Internal vectorized storage (upper triangle, no diagonal)
print(f"Vector shape: {adj.vector_shape}")
vector_length = adj.n_nodes * (adj.n_nodes - 1) // 2
print(f"Expected vector length (n*(n-1)/2): {vector_length}")

# Convert to full square matrix
square = adj.squareform()
print(f"Squareform shape: {square.shape}")

# %% [markdown]
# ## Thresholding

# %%
# Threshold weak connections

# Create correlation matrix with some structure
n_nodes = 20
base = np.random.randn(n_nodes, n_nodes) * 0.3
structured = base + base.T
np.fill_diagonal(structured, 0)

adj_struct = Adjacency(data=structured, matrix_type="similarity")

# Threshold: keep only strong positive connections
thresh = adj_struct.threshold(lower=0.3)
print(f"Edges above 0.3: {(thresh.data > 0).sum()}")

# Percentile threshold (keep top 10%)
thresh_pct = adj_struct.threshold(lower="90%")
print(f"Top 10% edges: {(thresh_pct.data > 0).sum()}")

# %% [markdown]
# ## Graph Metrics

# %%
# Compute basic graph properties

# Get square matrix
square_matrix = thresh.squareform()

# Degree: Number of connections per node
degrees = np.sum(np.abs(square_matrix) > 0, axis=1)

print(f"Node degrees: {degrees}")
print(f"Mean degree: {degrees.mean():.2f}")
print(f"Max degree: {degrees.max()}")

# %% [markdown]
# ## Applications: ROI-to-ROI Connectivity

# %%
# Simulate functional connectivity between brain regions
n_rois = 5
n_timepoints = 100

# Simulate ROI timeseries
roi_timeseries = np.random.randn(n_timepoints, n_rois)

# Add structure: ROIs 0-1 correlated, ROIs 3-4 correlated
roi_timeseries[:, 1] = roi_timeseries[:, 0] + np.random.randn(n_timepoints) * 0.3
roi_timeseries[:, 4] = roi_timeseries[:, 3] + np.random.randn(n_timepoints) * 0.3

# Compute correlation matrix
fc_matrix = np.corrcoef(roi_timeseries.T)
np.fill_diagonal(fc_matrix, 0)  # Zero diagonal

# Create Adjacency object
fc = Adjacency(data=fc_matrix, matrix_type="similarity")

print("Functional Connectivity Matrix:")
print(f"  Shape: {fc.shape}")
print(f"  Value range: [{fc.data.min():.2f}, {fc.data.max():.2f}]")

# Visualize
fc.plot()
plt.title("ROI-to-ROI Functional Connectivity")
plt.close()

# %% [markdown]
# ## Applications: Representational Similarity Analysis (RSA)

# %%
# Measure pattern similarity for different conditions
n_conditions = 6
n_voxels = 1000

# Simulate neural patterns for different stimuli
patterns = np.random.randn(n_conditions, n_voxels)

# Create structure: conditions 0-2 similar, conditions 3-5 similar
patterns[1] = patterns[0] + np.random.randn(n_voxels) * 0.2
patterns[2] = patterns[0] + np.random.randn(n_voxels) * 0.2
patterns[4] = patterns[3] + np.random.randn(n_voxels) * 0.2
patterns[5] = patterns[3] + np.random.randn(n_voxels) * 0.2

# Compute representational dissimilarity matrix (RDM)
# Dissimilarity = 1 - correlation
corr = np.corrcoef(patterns)
rdm = 1 - corr
np.fill_diagonal(rdm, 0)

# Create Adjacency object
rsa = Adjacency(data=rdm, matrix_type="distance")

print("Representational Dissimilarity Matrix:")
print(f"  Shape: {rsa.shape}")

# Visualize RDM
rsa.plot()
plt.title("Representational Dissimilarity Matrix")
plt.close()

# The block structure shows faces are similar to faces, objects to objects

# %% [markdown]
# ## Comparing Adjacency Matrices

# %%
# Compare two connectivity matrices (e.g., different conditions or groups)
from scipy.stats import pearsonr  # noqa: E402

# Create two matrices with different structure
np.random.seed(42)
matrix1 = np.random.randn(20, 20)
matrix1 = (matrix1 + matrix1.T) / 2
np.fill_diagonal(matrix1, 0)

matrix2 = matrix1 + np.random.randn(20, 20) * 0.5
matrix2 = (matrix2 + matrix2.T) / 2
np.fill_diagonal(matrix2, 0)

adj1 = Adjacency(data=matrix1, matrix_type="similarity")
adj2 = Adjacency(data=matrix2, matrix_type="similarity")

# Compute correlation between vectorized matrices
r, p = pearsonr(adj1.data, adj2.data)

print(f"Matrix similarity: r = {r:.3f}, p = {p:.4e}")

# %% [markdown]
# ## Statistical Testing

# %%
# Test if matrix values are significantly different from zero
mean_val = adj1.data.mean()
std_val = adj1.data.std()
n = len(adj1.data)

# One-sample t-test against zero
from scipy.stats import ttest_1samp  # noqa: E402

t_stat, p_val = ttest_1samp(adj1.data, 0)

print(f"Mean connectivity: {mean_val:.4f} (SD = {std_val:.4f})")
print(f"One-sample t-test vs 0: t = {t_stat:.2f}, p = {p_val:.4e}")

# %% [markdown]
# ## Summary
#
# In this tutorial, you learned how to:
# - ✓ Create `Adjacency` objects from matrices or similarity calculations
# - ✓ Work with vectorized representations of symmetric matrices
# - ✓ Threshold connectivity matrices
# - ✓ Compute basic graph metrics (degree)
# - ✓ Visualize networks with heatmaps
# - ✓ Apply adjacency to functional connectivity and RSA
# - ✓ Compare connectivity matrices across conditions
#
# ## Next Steps
#
# - **[GLM Workflow](../workflows/01_glm)**: First-level fMRI analysis
# - **[RSA Workflow](../workflows/04_rsa)**: Full representational similarity analysis
