---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
execute:
  skip: true
---

# Representational Similarity Analysis

RSA links disparate data types via shared structure in similarity matrices.

```{code-cell} python3
import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from nltools.data import Adjacency
from nltools.datasets import fetch_haxby
from nltools.utils import concatenate
```

## Load Data

```{code-cell} python3
print("Loading Haxby dataset...")
all_data, all_dm = fetch_haxby(n_subjects="all", verbose=1)

# First subject, first run
data = all_data[0][0].copy()
dm = all_dm[0][0]

print(f"Data shape: {data.shape}")
print(f"Conditions: {list(dm.columns)}")
```

## Get Beta Patterns via GLM

```{code-cell} python3
conditions = ["bottle", "cat", "chair", "face", "house", "scissors", "shoe"]
print(f"Conditions: {conditions}")

dm_filt = dm.add_dct_basis(duration=128).add_poly(order=1, include_lower=True)
data.fit(model="glm", X=dm_filt)

# Extract beta patterns
condition_betas = []
for cond in conditions:
    beta = data.compute_contrasts(cond)
    condition_betas.append(beta)

patterns = concatenate(condition_betas)
patterns.Y = conditions
print(f"Pattern matrix: {patterns.shape}")
```

## Compute Pattern Similarity

```{code-cell} python3
similarity = patterns.distance(metric="correlation")
similarity.labels = conditions

print(f"Similarity matrix: {similarity.shape}")
```

```{code-cell} python3
# Convert distance to similarity (1 - distance)
sim_matrix = 1 - similarity

sim_matrix.plot(vmin=-0.5, vmax=1, cmap="RdBu_r")
plt.title("Pattern Similarity")
plt.close()
```

## MDS Visualization

```{code-cell} python3
similarity.plot_mds(n_components=2)
plt.title("MDS: Pattern Relationships")
plt.close()
```

## Test Animate vs Inanimate Hypothesis

```{code-cell} python3
n_conditions = len(conditions)
model = np.zeros((n_conditions, n_conditions))

animate_idx = [conditions.index(c) for c in ["face", "cat"] if c in conditions]
inanimate_idx = [
    conditions.index(c)
    for c in ["bottle", "chair", "house", "scissors", "shoe"]
    if c in conditions
]

print(f"Animate: {[conditions[i] for i in animate_idx]}")
print(f"Inanimate: {[conditions[i] for i in inanimate_idx]}")

# High similarity within categories
for i in animate_idx:
    for j in animate_idx:
        model[i, j] = 1

for i in inanimate_idx:
    for j in inanimate_idx:
        model[i, j] = 1

model_adj = Adjacency(model, matrix_type="similarity", labels=conditions)
model_adj.plot(vmin=0, vmax=1, cmap="RdBu_r")
plt.title("Model: Animate vs Inanimate")
plt.close()
```

```{code-cell} python3
# Compare brain and model
result = sim_matrix.similarity(model_adj, metric="spearman", n_permute=5000)

print("RSA Results:")
print(f"  rho: {result['correlation']:.3f}")
print(f"  p: {result['p']:.4f}")
```

## Test Face vs House Model

```{code-cell} python3
model_fh = np.ones((n_conditions, n_conditions))

if "face" in conditions and "house" in conditions:
    face_idx = conditions.index("face")
    house_idx = conditions.index("house")
    model_fh[face_idx, house_idx] = 0
    model_fh[house_idx, face_idx] = 0

model_fh_adj = Adjacency(model_fh, matrix_type="similarity", labels=conditions)

model_fh_adj.plot(vmin=0, vmax=1, cmap="RdBu_r")
plt.title("Model: Face-House Dissimilarity")
plt.close()
```

```{code-cell} python3
result_fh = sim_matrix.similarity(model_fh_adj, metric="spearman", n_permute=5000)

print("Face vs House RSA:")
print(f"  rho: {result_fh['correlation']:.3f}")
print(f"  p: {result_fh['p']:.4f}")
```

## Examine Specific Pairs

```{code-cell} python3
sim_array = sim_matrix.squareform()

print("Pattern Similarities:")
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
        print(f"  {cond1} - {cond2}: {sim_array[i, j]:.3f}")
```

## Adjacency Properties

```{code-cell} python3
print(f"Matrix type: {sim_matrix.matrix_type}")
print(f"Shape: {sim_matrix.shape}")
print(f"Labels: {sim_matrix.labels}")
```

```{code-cell} python3
# Threshold
high_sim = sim_matrix.threshold(upper=0.3, lower=None)
high_sim.plot(vmin=0, vmax=1, cmap="RdBu_r")
plt.title("High Similarity Only (> 0.3)")
plt.close()
```
