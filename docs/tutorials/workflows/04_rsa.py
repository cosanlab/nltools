# %% [markdown]
# # Representational Similarity Analysis
#
# *Written by Luke Chang, updated for v0.6.0*
#
# Representational Similarity Analysis (RSA) is a multivariate technique that links
# disparate types of data based on shared structure in their similarity matrices.
# This technique was initially proposed by [Kriegeskorte et al. (2008)](https://www.frontiersin.org/articles/10.3389/neuro.06.004.2008/full).
#
# Unlike multivariate classification, RSA does not directly map brain activity onto
# a measure. Instead, it compares similarities between brain activity patterns and
# theoretical models using "second-order isomorphisms."
#
# In this tutorial we will cover:
# 1. Computing pattern similarity across conditions
# 2. Visualizing similarity with heatmaps and MDS
# 3. Testing representational hypotheses
# 4. Using the Adjacency class for similarity matrices

# %%
import numpy as np
import matplotlib.pyplot as plt
from nltools.data import Adjacency
from nltools.datasets import fetch_haxby
from nltools.utils import concatenate

# %% [markdown]
# ## RSA Conceptual Overview
#
# ### The Key Idea
#
# RSA asks: Does the pattern of brain activity across conditions have a similar
# *structure* to some theoretical model?
#
# For example, in the Haxby dataset, we might hypothesize that:
# - Face and house patterns are very different (they activate different brain regions)
# - Similar categories (e.g., animate objects) have more similar patterns
#
# ### Steps in RSA
#
# 1. **Estimate patterns**: Get a brain map for each condition (e.g., beta from GLM)
# 2. **Compute brain similarity**: Calculate pairwise similarity between all conditions
# 3. **Define model similarity**: Create a theoretical prediction of similarity structure
# 4. **Compare**: Correlate brain similarity matrix with model matrix

# %% [markdown]
# ## Loading and Preparing Data
#
# We'll use the Haxby dataset and fit a GLM to get condition-specific beta patterns.
# The dataset has 8 object categories: face, house, cat, bottle, scissors,
# shoe, chair, and scrambled.

# %%
print("Loading Haxby dataset...")
all_data, all_dm = fetch_haxby(n_subjects="all", verbose=1)

# Use first subject, first run for this tutorial
data = all_data[0][0].copy()
dm = all_dm[0][0]

print(f"Data shape: {data.shape}")
print(f"Conditions: {list(dm.columns)}")

# %% [markdown]
# ### Fitting GLM to Get Beta Patterns
#
# For RSA, we need a single brain pattern for each condition.
# We fit a GLM and extract the beta (coefficient) for each regressor.
# These betas represent the condition-specific activation patterns.

# %%
# Define conditions of interest (exclude nuisance regressors)
conditions = ["bottle", "cat", "chair", "face", "house", "scissors", "shoe"]
print(f"Conditions for RSA: {conditions}")

# Add nuisance regressors and fit GLM
dm_filt = dm.add_dct_basis(duration=128).add_poly(order=1, include_lower=True)
data.fit(model="glm", X=dm_filt)

# Extract beta patterns for each condition
condition_betas = []
for cond in conditions:
    beta = data.compute_contrasts(cond)
    condition_betas.append(beta)

print(f"\nExtracted beta patterns for {len(condition_betas)} conditions")

# Stack into a single BrainData object
patterns = concatenate(condition_betas)
patterns.Y = conditions
print(f"Pattern matrix shape: {patterns.shape} (conditions x voxels)")

# %% [markdown]
# ## Computing Pattern Similarity
#
# Now we compute the pairwise similarity between all conditions.
# We'll use correlation distance, which measures pattern similarity
# regardless of overall activation magnitude.

# %%
# Compute pairwise distance (correlation distance)
similarity = patterns.distance(metric="correlation")

# Add condition labels
similarity.labels = conditions

print(f"Similarity matrix size: {similarity.shape}")

# %% [markdown]
# ### Visualizing the Similarity Matrix
#
# Let's visualize the pattern similarity across conditions.
# Darker red = more similar patterns, darker blue = more different patterns.

# %%
# Convert distance to similarity (1 - distance)
sim_matrix = 1 - similarity

# Plot similarity matrix
fig = sim_matrix.plot(vmin=-0.5, vmax=1, cmap="RdBu_r")
plt.title("Pattern Similarity Across Conditions", fontsize=14)

# %% [markdown]
# ### Multidimensional Scaling (MDS)
#
# MDS projects the high-dimensional patterns into a 2D space for visualization.
# Conditions that are close together have similar brain patterns.

# %%
# Plot MDS
fig = similarity.plot_mds(n_components=2)
plt.title("MDS: Pattern Relationships", fontsize=14)

# %% [markdown]
# ## Testing a Representational Hypothesis
#
# Now let's test a specific hypothesis about the representational structure.
#
# **Hypothesis**: Animate objects (face, cat) have similar patterns that are
# distinct from inanimate objects (house, bottle, scissors, shoe, chair).

# %%
# Create a model similarity matrix
# Animate: face, cat
# Inanimate: house, bottle, scissors, shoe, chair

n_conditions = len(conditions)
model = np.zeros((n_conditions, n_conditions))

# Find indices for each category
# Animate: face, cat
# Inanimate: house, bottle, scissors, shoe, chair
animate_idx = [conditions.index(c) for c in ["face", "cat"] if c in conditions]
inanimate_idx = [
    conditions.index(c)
    for c in ["bottle", "chair", "house", "scissors", "shoe"]
    if c in conditions
]

print(f"Animate conditions: {[conditions[i] for i in animate_idx]}")
print(f"Inanimate conditions: {[conditions[i] for i in inanimate_idx]}")

# Set high similarity within categories
for i in animate_idx:
    for j in animate_idx:
        model[i, j] = 1

for i in inanimate_idx:
    for j in inanimate_idx:
        model[i, j] = 1

# Create Adjacency object
model_adj = Adjacency(model, matrix_type="similarity", labels=conditions)

print("\nModel matrix (animate vs inanimate):")
model_adj.plot(vmin=0, vmax=1, cmap="RdBu_r")
plt.title("Model: Animate vs Inanimate", fontsize=14)

# %% [markdown]
# ### Comparing Brain and Model Similarity
#
# We use Spearman correlation to compare the brain similarity matrix
# with our model matrix. This tests whether the brain's representational
# structure matches our hypothesis.

# %%
# Compare brain similarity with model
result = sim_matrix.similarity(model_adj, metric="spearman", n_permute=5000)

print("RSA Results:")
print(f"  Correlation (rho): {result['correlation']:.3f}")
print(f"  P-value: {result['p']:.4f}")

if result["p"] < 0.05:
    print("\n  ✓ The brain's representational structure significantly")
    print("    matches the animate/inanimate model!")
else:
    print("\n  ✗ No significant match with the model")
    print("    (May need more subjects or a better model)")

# %% [markdown]
# ## Alternative Hypothesis: Face vs House
#
# Let's test a more specific hypothesis based on the classic Haxby finding:
# Face and house patterns should be maximally distinct.

# %%
# Create face vs house model
model_fh = np.ones((n_conditions, n_conditions))

# Face and house should be dissimilar (set to 0)
if "face" in conditions and "house" in conditions:
    face_idx = conditions.index("face")
    house_idx = conditions.index("house")
    model_fh[face_idx, house_idx] = 0
    model_fh[house_idx, face_idx] = 0

model_fh_adj = Adjacency(model_fh, matrix_type="similarity", labels=conditions)

# Plot
model_fh_adj.plot(vmin=0, vmax=1, cmap="RdBu_r")
plt.title("Model: Face-House Dissimilarity", fontsize=14)

# %%
# Test the face vs house model
result_fh = sim_matrix.similarity(model_fh_adj, metric="spearman", n_permute=5000)

print("Face vs House RSA Results:")
print(f"  Correlation (rho): {result_fh['correlation']:.3f}")
print(f"  P-value: {result_fh['p']:.4f}")

# %% [markdown]
# ## Examining Specific Pattern Relationships
#
# Let's look at the actual similarity values between key conditions.

# %%
# Get the full similarity matrix as numpy array
sim_array = sim_matrix.squareform()

# Print key similarities
print("Pattern Similarities (correlation):")
print("-" * 40)

key_pairs = [
    ("face", "house"),
    ("face", "cat"),
    ("house", "chair"),
    ("bottle", "scissors"),
]

for cond1, cond2 in key_pairs:
    if cond1 in conditions and cond2 in conditions:
        i = conditions.index(cond1)
        j = conditions.index(cond2)
        print(f"  {cond1:10} - {cond2:10}: {sim_array[i, j]:.3f}")

# %% [markdown]
# ## Working with the Adjacency Class
#
# The `Adjacency` class provides many useful methods for working with
# similarity and distance matrices.

# %%
# Basic properties
print("Adjacency Object Properties:")
print(f"  Matrix type: {sim_matrix.matrix_type}")
print(f"  Shape: {sim_matrix.shape}")
print(f"  Labels: {sim_matrix.labels}")

# %%
# Threshold the similarity matrix
high_sim = sim_matrix.threshold(upper=0.3, lower=None)
print("\nCondition pairs with similarity > 0.3:")
high_sim.plot(vmin=0, vmax=1, cmap="RdBu_r")
plt.title("High Similarity Pairs Only", fontsize=14)

# %% [markdown]
# ## Summary
#
# In this tutorial, we covered:
#
# 1. **Pattern similarity**: Computing pairwise correlations between condition patterns
# 2. **Visualization**: Using heatmaps and MDS to explore representational structure
# 3. **Hypothesis testing**: Comparing brain similarity to theoretical models
# 4. **Adjacency class**: Working with similarity matrices in nltools
#
# ### Key Methods Used
#
# | Method | Description |
# |--------|-------------|
# | `BrainData.distance()` | Compute pairwise distance/similarity |
# | `Adjacency.plot()` | Visualize similarity matrix |
# | `Adjacency.plot_mds()` | Project to 2D/3D with MDS |
# | `Adjacency.similarity()` | Compare two similarity matrices |
# | `Adjacency.threshold()` | Threshold similarity values |
#
# ### Key Concepts
#
# - RSA compares the *structure* of brain representations to theoretical models
# - Uses second-order isomorphism (correlating correlation matrices)
# - Powerful for testing computational hypotheses about brain function
# - Can link brain data to other modalities (behavior, models, other species)
#
# ### Next Steps
#
# - Tutorial 06: Multivariate Pattern Analysis (MVPA) and Decoding
