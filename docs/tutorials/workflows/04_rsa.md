---
file_format: mystnb
authors:
  - Eshin Jolly
kernelspec:
  name: python3
  display_name: Python 3
execute:
  allow_errors: false
---

```{code-cell} python
:tags: [remove-cell]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

# Representational Similarity Analysis (RSA)

## Introduction

Decoding asks **"can the spatial pattern predict the condition?"**. RSA flips the question once more:

> **What is the *structure* of the patterns relative to one another?**

Instead of training a classifier, we compute a **representational dissimilarity matrix (RDM)** — pairwise distances between condition patterns — and ask whether that geometry matches a hypothesis (e.g. *animates cluster together; faces are far from houses*).

The big win: the RDM is a common currency. Once your brain patterns and your hypothesis are both expressed as RDMs, comparing them is a single correlation. The same shape can also relate brain data to behavior, model layers, or another subject — anything you can turn into a distance matrix.

```{code-cell} python
from nltools.data import BrainData, Adjacency
```

## Load Haxby (one subject)

Same dataset as the decoding tutorial — `sub-2` viewing 8 object categories across 12 runs — auto-resampled to MNI 3mm by `BrainData`:

```{code-cell} python
from nilearn.datasets import fetch_haxby

HAXBY = fetch_haxby(subjects=[2], verbose=0)
brain = BrainData(HAXBY.func[0])
labels = pd.read_csv(HAXBY.session_target[0], sep=" ")

print(f"brain.shape: {brain.shape}  (TRs, voxels)")
print(f"conditions:  {sorted(labels['labels'].unique())}")
```

We'll work with the seven object categories (dropping `rest` and `scrambledpix`) and build one mean pattern per condition. Mean-of-TRs is a deliberately simple estimator — it's enough to surface the well-known animate / face-vs-place structure without re-doing the GLM tutorial:

```{code-cell} python
conditions = ["bottle", "cat", "chair", "face", "house", "scissors", "shoe"]

condition_patterns = []
for cond in conditions:
    idx = (labels["labels"] == cond).to_numpy()
    pattern = brain[idx].data.mean(axis=0)
    condition_patterns.append(pattern)

patterns = BrainData(np.stack(condition_patterns), mask=brain.mask)
patterns.Y = pd.DataFrame({"condition": conditions})
print(f"patterns.shape: {patterns.shape}  (n_conditions, n_voxels)")
```

## Compute the RDM

`BrainData.distance(metric='correlation')` returns an `Adjacency` whose entry *(i,j)* is `1 − corr(pattern_i, pattern_j)` — the canonical RSA dissimilarity:

```{code-cell} python
rdm = patterns.distance(metric="correlation")
rdm.labels = conditions
print(f"rdm.shape: {rdm.shape}  (matrix_type={rdm.matrix_type})")
```

```{code-cell} python
fig, ax = plt.subplots(figsize=(5, 4))
rdm.plot(vmin=0, vmax=2, cmap="RdBu_r", axes=ax)
ax.set_title("Representational dissimilarity (1 − Pearson r)")
fig
```

For visualization it's often easier to read similarity than distance — `distance_to_similarity` flips the sign for you:

```{code-cell} python
sim = rdm.distance_to_similarity(metric="correlation")
sim.labels = conditions

fig, ax = plt.subplots(figsize=(5, 4))
sim.plot(vmin=-1, vmax=1, cmap="RdBu_r", axes=ax)
ax.set_title("Pattern similarity (Pearson r)")
fig
```

## MDS — see the geometry

A 2D MDS projection of the RDM gives a quick spatial intuition for which conditions cluster:

```{code-cell} python
fig, ax = plt.subplots(figsize=(5, 5))
rdm.plot_mds(n_components=2, labels=conditions, ax=ax)
ax.set_title("MDS of the brain RDM")
fig
```

## Test a hypothesis: animate vs inanimate

The cleanest RSA workflow is *hypothesis as RDM*. Build a model RDM where same-category pairs have distance 0 and across-category pairs have distance 1:

```{code-cell} python
animate = {"face", "cat"}
is_animate = np.array([c in animate for c in conditions])
animate_rdm = (is_animate[:, None] != is_animate[None, :]).astype(float)

animate_model = Adjacency(animate_rdm, matrix_type="distance", labels=conditions)

fig, ax = plt.subplots(figsize=(5, 4))
animate_model.plot(vmin=0, vmax=1, cmap="RdBu_r", axes=ax)
ax.set_title("Model RDM: animate vs inanimate")
fig
```

Compare brain RDM and model RDM — Spearman rank correlation with a permutation null:

```{code-cell} python
result = rdm.similarity(animate_model, metric="spearman", n_permute=1000, random_state=0)
print(f"animate vs inanimate:  rho = {result['correlation']:.3f}  p = {result['p']:.4f}")
```

## A second hypothesis: face vs house

Faces and houses anchor the classic Haxby ventral-temporal story — faces drive FFA, houses drive PPA. A targeted model RDM is just a single off-diagonal entry pulled apart from the rest:

```{code-cell} python
face_house_rdm = np.zeros((len(conditions), len(conditions)))
i, j = conditions.index("face"), conditions.index("house")
face_house_rdm[i, j] = face_house_rdm[j, i] = 1.0

face_house_model = Adjacency(face_house_rdm, matrix_type="distance", labels=conditions)
result_fh = rdm.similarity(face_house_model, metric="spearman", n_permute=1000, random_state=0)
print(f"face ↔ house dissim:   rho = {result_fh['correlation']:.3f}  p = {result_fh['p']:.4f}")
```

## Inspect specific pairs

Sometimes you just want the numbers. `squareform()` returns the dense `n × n` array; the labels list gives row/column meaning:

```{code-cell} python
rdm_array = rdm.squareform()

pairs = [("face", "house"), ("face", "cat"), ("house", "chair"), ("bottle", "scissors")]
print("Pairwise dissimilarity (1 − r):")
for a, b in pairs:
    i, j = conditions.index(a), conditions.index(b)
    print(f"  {a:<8s} ↔ {b:<8s}  {rdm_array[i, j]:.3f}")
```

## Putting it together

| Step | Call | Returns |
|---|---|---|
| Per-condition pattern stack | `BrainData(np.stack(means), mask=brain.mask)` | `BrainData` `(n_conditions, n_voxels)` |
| Brain RDM | `patterns.distance(metric='correlation')` | `Adjacency` (matrix_type='distance') |
| Flip to similarity | `rdm.distance_to_similarity(metric='correlation')` | `Adjacency` (matrix_type='similarity') |
| MDS embedding | `rdm.plot_mds(n_components=2, labels=...)` | matplotlib axes |
| Compare to model | `rdm.similarity(model, metric='spearman', n_permute=K)` | `dict` with `correlation`, `p` |

> **When mean-of-TRs isn't enough.** Mean patterns are a fine first pass and great for tutorials, but real RSA studies usually use **GLM betas** (or t-stats, which downweight noisy voxels). Swap the pattern-extraction step for `data.fit(model='glm', X=dm).compute_contrasts(cond)` — the rest of this tutorial stays the same. The [GLM tutorial](01_glm.md) covers that machinery.

## Next Steps

- **Searchlight RSA**: roving sphere — `patterns.distance(spatial_scale='searchlight', radius_mm=8.0)` returns one RDM per searchlight, ready to compare to a model RDM voxelwise.
- **ROI RSA**: pass a parcellation — `patterns.distance(spatial_scale='roi', roi_mask=atlas)` for one RDM per region.
- **Decoding view**: instead of distance structure, can a classifier separate the conditions? — see [Decoding](05_decoding.md).
