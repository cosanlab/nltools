# %% [markdown]
# # Representational Similarity Analysis (RSA)
#
# RSA compares the similarity structure of neural representations with
# theoretical models. Instead of asking "which voxel responds to X?", RSA
# asks "does the brain organize information the way our model predicts?"
#
# ## Learning Objectives
#
# - Understand Representational Dissimilarity Matrices (RDMs)
# - Use `RSATerminal` for model-based RSA
# - Compare neural and model RDMs
# - Run permutation tests for significance

# %%
import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform

from nltools.pipelines.terminals import RSATerminal

# %% [markdown]
# ## What is RSA?
#
# RSA compares representations through Representational Dissimilarity Matrices:
#
# 1. Compute pairwise dissimilarity between all conditions (neural RDM)
# 2. Compare to a theoretical model (model RDM)
# 3. Higher correlation = brain uses similar representation
#
# ```
# Neural patterns:          Neural RDM:              Model RDM:
# Face  → [1,0,1,0]         Face House Cat Dog       Face House Cat Dog
# House → [0,1,0,1]    →    Face  0    1    0   1      Face  0    1    0   1
# Cat   → [1,0,0,1]         House 1    0    1   0      House 1    0    1   0
# Dog   → [0,1,1,0]         Cat   0    1    0   1      Cat   0    1    0   1
#                           Dog   1    0    1   0      Dog   1    0    1   0
#
# Neural RDM ↔ Model RDM → Correlation (r) + p-value
# ```

# %% [markdown]
# ## Creating RDMs
#
# An RDM is a symmetric matrix showing dissimilarity between all pairs.

# %%
np.random.seed(42)

n_conditions = 6  # e.g., 6 categories: face, house, cat, dog, car, tool
n_voxels = 100

# Create neural patterns for each condition
# Simulate category structure: faces/cats similar, houses/cars similar
patterns = np.random.randn(n_conditions, n_voxels)

# Add category structure
# Animate: faces (0), cats (2), dogs (4)
# Inanimate: houses (1), cars (3), tools (5)
patterns[0, :30] += 1.0  # Face
patterns[2, :30] += 0.8  # Cat (similar to face)
patterns[4, :30] += 0.6  # Dog (somewhat similar)

patterns[1, 30:60] += 1.0  # House
patterns[3, 30:60] += 0.8  # Car (similar to house)
patterns[5, 30:60] += 0.6  # Tool (somewhat similar)

print(f"Patterns shape: {patterns.shape}")
print(f"  Conditions: {n_conditions}")
print(f"  Features/voxels: {n_voxels}")

# %% [markdown]
# ## Computing Neural RDMs
#
# Use correlation distance (1 - correlation) for dissimilarity.

# %%
from sklearn.metrics import pairwise_distances

# Compute neural RDM using correlation distance
neural_rdm = pairwise_distances(patterns, metric="correlation")

print(f"Neural RDM shape: {neural_rdm.shape}")
print(f"  Min dissimilarity: {neural_rdm.min():.3f}")
print(f"  Max dissimilarity: {neural_rdm.max():.3f}")

# Visualize
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(neural_rdm, cmap="viridis")
ax.set_title("Neural RDM")
ax.set_xlabel("Condition")
ax.set_ylabel("Condition")
labels = ["Face", "House", "Cat", "Car", "Dog", "Tool"]
ax.set_xticks(range(6))
ax.set_xticklabels(labels, rotation=45)
ax.set_yticks(range(6))
ax.set_yticklabels(labels)
plt.colorbar(im, label="Dissimilarity")
plt.tight_layout()
plt.close()

print("Plot created: Neural RDM")

# %% [markdown]
# ## Creating Model RDMs
#
# Model RDMs encode theoretical predictions about how conditions relate.

# %%
# Model 1: Animate vs Inanimate
# Animate: Face, Cat, Dog (indices 0, 2, 4)
# Inanimate: House, Car, Tool (indices 1, 3, 5)
model_animate = np.zeros((6, 6))
animate = [0, 2, 4]
inanimate = [1, 3, 5]

for i in animate:
    for j in inanimate:
        model_animate[i, j] = 1
        model_animate[j, i] = 1

print("Animate/Inanimate Model:")
print(model_animate)

# %%
# Model 2: Visual similarity (faces vs non-faces)
model_face = np.ones((6, 6))
np.fill_diagonal(model_face, 0)
# Faces and cats are visually similar
model_face[0, 2] = 0.3
model_face[2, 0] = 0.3

print("\nFace similarity Model (subset):")
print(model_face[:3, :3])

# %% [markdown]
# ## Running RSA with RSATerminal
#
# `RSATerminal` correlates neural RDMs with model RDMs using permutation testing.

# %%
# Create RSA terminal with animate/inanimate model
terminal = RSATerminal(
    model_rdm=model_animate,
    method="spearman",  # Rank correlation
    n_permute=1000
)

# Run RSA
result = terminal.fit_evaluate(neural_rdm, random_state=42)

print("RSA Result:")
print(f"  Correlation: {result.correlation:.3f}")
print(f"  p-value: {result.p_value:.4f}")
print(f"  95% CI: [{result.ci[0]:.3f}, {result.ci[1]:.3f}]")
print(f"  Method: {result.method}")
print(f"  N conditions: {result.n_conditions}")

# %% [markdown]
# ## Comparing Multiple Models
#
# Test which theoretical model best explains the neural data.

# %%
# Create additional models

# Model 3: Random (control)
np.random.seed(123)
model_random = np.random.rand(6, 6)
model_random = (model_random + model_random.T) / 2  # Make symmetric
np.fill_diagonal(model_random, 0)

# Test all models
models = {
    "Animate/Inanimate": model_animate,
    "Face similarity": model_face,
    "Random": model_random
}

print("Model Comparison:")
print("=" * 50)

for name, model in models.items():
    terminal = RSATerminal(model_rdm=model, method="spearman", n_permute=500)
    result = terminal.fit_evaluate(neural_rdm, random_state=42)
    sig = "*" if result.p_value < 0.05 else ""
    print(f"{name:20s}: r = {result.correlation:+.3f}, p = {result.p_value:.4f} {sig}")

# %% [markdown]
# ## Correlation Methods
#
# | Method | Description | Best For |
# |--------|-------------|----------|
# | `spearman` | Rank correlation | Ordinal relationships (default) |
# | `pearson` | Linear correlation | Interval data |
# | `kendall` | Concordance | Robust to outliers |

# %%
# Compare correlation methods
methods = ["spearman", "pearson", "kendall"]

print("\nCorrelation Method Comparison:")
for method in methods:
    terminal = RSATerminal(model_rdm=model_animate, method=method, n_permute=300)
    result = terminal.fit_evaluate(neural_rdm, random_state=42)
    print(f"  {method:10s}: r = {result.correlation:+.3f}")

# %% [markdown]
# ## Using RSATerminal with Raw Data
#
# RSATerminal can also compute the RDM automatically from pattern data.

# %%
# Pass patterns directly (n_conditions, n_features)
terminal = RSATerminal(model_rdm=model_animate, method="spearman", n_permute=500)

# RSATerminal will compute neural RDM using correlation distance
result_from_patterns = terminal.fit_evaluate(patterns, random_state=42)

print("RSA from raw patterns:")
print(f"  Correlation: {result_from_patterns.correlation:.3f}")
print(f"  p-value: {result_from_patterns.p_value:.4f}")

# %% [markdown]
# ## Condensed RDM Format
#
# RDMs can be stored in condensed form (upper triangle only) for efficiency.

# %%
# Convert square RDM to condensed form
neural_condensed = squareform(neural_rdm, checks=False)
model_condensed = squareform(model_animate, checks=False)

print(f"Square RDM shape: {neural_rdm.shape}")
print(f"Condensed shape: {neural_condensed.shape}")
print(f"  (6 conditions → 6*5/2 = 15 pairs)")

# RSATerminal accepts either format
terminal = RSATerminal(model_rdm=model_condensed, n_permute=300)
result_condensed = terminal.fit_evaluate(neural_condensed, random_state=42)
print(f"\nRSA with condensed format: r = {result_condensed.correlation:.3f}")

# %% [markdown]
# ## Visualizing Model Comparison

# %%
# Bar plot comparing model fits
model_names = list(models.keys())
correlations = []
p_values = []

for name, model in models.items():
    terminal = RSATerminal(model_rdm=model, method="spearman", n_permute=500)
    result = terminal.fit_evaluate(neural_rdm, random_state=42)
    correlations.append(result.correlation)
    p_values.append(result.p_value)

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(model_names))
colors = ['green' if p < 0.05 else 'gray' for p in p_values]
bars = ax.bar(x, correlations, color=colors)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('Spearman Correlation')
ax.set_title('RSA: Model Comparison')
ax.set_xticks(x)
ax.set_xticklabels(model_names)

# Add significance markers
for i, (corr, p) in enumerate(zip(correlations, p_values)):
    if p < 0.05:
        ax.annotate('*', (i, corr + 0.02), ha='center', fontsize=14)

plt.tight_layout()
plt.close()

print("Plot created: Model comparison bar chart")

# %% [markdown]
# ## Permutation Testing
#
# Permutation tests shuffle condition labels to create a null distribution.
#
# - **Null hypothesis**: No relationship between neural and model RDMs
# - **P-value**: Proportion of permuted correlations >= observed correlation
# - **More permutations** = More precise p-value (use 5000+ for publication)

# %% [markdown]
# ## Best Practices
#
# 1. **Use enough permutations**: 5000+ for publication
# 2. **Choose appropriate metric**: Spearman for ordinal, Pearson for interval
# 3. **Multiple comparison correction**: FDR when testing many ROIs/models
# 4. **Report effect sizes**: Correlation values, not just p-values
# 5. **Visualize RDMs**: Always inspect your RDMs visually

# %% [markdown]
# ## Summary
#
# | Method | Description |
# |--------|-------------|
# | `RSATerminal(model_rdm)` | Create RSA analysis |
# | `method='spearman'` | Rank correlation (default) |
# | `method='pearson'` | Pearson correlation |
# | `method='kendall'` | Kendall's tau |
# | `terminal.fit_evaluate(data)` | Run RSA and get result |
# | `result.correlation` | Correlation coefficient |
# | `result.p_value` | Permutation p-value |
# | `result.ci` | Confidence interval |
#
# ## The Full Workflow
#
# ```python
# from nltools.pipelines.terminals import RSATerminal
# from sklearn.metrics import pairwise_distances
#
# # 1. Create model RDM based on theory
# model_rdm = create_theoretical_model(n_conditions=6)
#
# # 2. Compute neural RDM from patterns
# neural_rdm = pairwise_distances(patterns, metric='correlation')
#
# # 3. Run RSA with permutation test
# terminal = RSATerminal(
#     model_rdm=model_rdm,
#     method='spearman',
#     n_permute=5000
# )
# result = terminal.fit_evaluate(neural_rdm)
#
# print(f"r = {result.correlation:.3f}, p = {result.p_value:.4f}")
# ```
#
# ## Next Steps
#
# - **[ISC Analysis](09_isc_analysis)**: Inter-subject correlation
# - **[Multi-Subject Decoding](08_multi_subject_decoding)**: LOSO CV
# - **[Pipeline Basics](06_pipeline_basics)**: Single-subject pipelines
