---
title: Similarity & RSA
---

Representational similarity analysis has two halves: turn brain patterns into a representational
dissimilarity matrix (RDM), and compare that RDM to a model RDM.
[`BrainData.distance`](../api/data/brain_data.md#data-brain-data-distance) does the first and
returns an [`Adjacency`](../api/data/adjacency.md), a square matrix stored as its upper triangle.
[`Adjacency.similarity`](../api/data/adjacency.md#data-adjacency-similarity) does the second.

Two kwargs are easy to mix up. `metric=` is the correlation type used to compare the two matrices
(`'spearman'` by default, or `'pearson'` / `'kendall'`). `method=` is the *permutation scheme*:
`'2d'` shuffles rows and columns together (the Mantel test, the right null for a symmetric RDM),
`'1d'` shuffles the vectorized entries, and `None` skips the test entirely.

Goal | Use | Notes
--- | --- | ---
Brain RDM | [`BrainData.distance`](../api/data/brain_data.md#data-brain-data-distance)`(metric='correlation')` | Any scipy metric; `'euclidean'` is the default
Per-ROI or per-searchlight RDMs | `distance(..., spatial_scale='roi', roi_mask=)` or `'searchlight', radius_mm=` | Returns an ordinary stack in explicit ROI or voxel order
Model RDM | `Adjacency(square_matrix, matrix_type='distance')` | `'similarity'` and `'directed'` are the other types
Compare two RDMs | [`Adjacency.similarity`](../api/data/adjacency.md#data-adjacency-similarity)`(other, metric=, method='2d')` | Returns `{'correlation', 'p', ...}`; a stack returns a list
Paint a stack's result on the brain | `roi_to_brain_from_atlas(...)` or `nilearn.masking.unmask(...)` | Retain the aligned atlas/ROI order or source-mask voxel order
Compare two raw matrices | [`matrix_permutation_test`](../api/tasks/similarity.md#tasks-similarity-matrix-permutation-test) | The Mantel test on plain arrays; `include_diag=False` by default
Row-wise pattern similarity | [`compute_similarity`](../api/tasks/similarity.md#tasks-similarity-compute-similarity) | One image against many; `metric='correlation'`, `'spearman'`, `'cosine'`, `'dot_product'`
Average correlations | [`fisher_r_to_z`](../api/tasks/similarity.md#tasks-similarity-fisher-r-to-z) / [`fisher_z_to_r`](../api/tasks/similarity.md#tasks-similarity-fisher-z-to-r) | Also `Adjacency.r_to_z` / `.z_to_r`, returning independent copies
Show two RDMs together | [`plot_stacked_adjacency`](../api/tasks/similarity.md#tasks-similarity-plot-stacked-adjacency) | Brain RDM above the diagonal, model RDM below
Cluster structure | [`plot_mds`](../api/data/adjacency.md#data-adjacency-plot-mds), [`plot_silhouette`](../api/data/adjacency.md#data-adjacency-plot-silhouette), [`plot_label_distance`](../api/data/adjacency.md#data-adjacency-plot-label-distance) | All take `labels=`

## Brain RDM vs model RDM

```python
from nltools.data import Adjacency

brain_rdm = brain.distance(metric="correlation")
model_rdm = Adjacency(
    np.abs(levels[:, None] - levels[None, :]), matrix_type="distance"
)

stats = brain_rdm.similarity(
    model_rdm, metric="spearman", n_permute=1000, random_state=0
)
stats["correlation"], stats["p"]
```

`plot_stacked_adjacency` is the picture that goes with that number. It draws one matrix above the
diagonal and the other below on a shared scale, so you can see where the two agree, not just how
much:

```python
from nltools.plotting import plot_stacked_adjacency

plot_stacked_adjacency(brain_rdm, model_rdm)
```

`normalize=True` (the default) rescales both matrices before stacking, which matters whenever the
two are on different scales, say a correlation-distance brain RDM against a model RDM in stimulus
units. Extra keyword arguments go to seaborn's heatmap.

## Per-ROI RSA, painted back on the brain

Give `distance` a labeled parcellation and it returns one RDM per parcel.
Align the atlas to the brain mask first and keep the sorted nonzero labels
present within that mask. This is the stack order used by `distance`:

```python
from nilearn.image import resample_to_img
from nilearn.masking import apply_mask
from nltools.mask import roi_to_brain_from_atlas

aligned_atlas = resample_to_img(
    atlas, brain.mask, interpolation="nearest", force_resample=True, copy_header=True,
)
roi_labels = np.unique(apply_mask(aligned_atlas, brain.mask).astype(int))
roi_labels = roi_labels[roi_labels != 0]
rdms = brain.distance(metric="correlation", spatial_scale="roi", roi_mask=aligned_atlas)
scores = rdms.similarity(model_rdm, metric="spearman", method=None)
brain_map = roi_to_brain_from_atlas(
    np.array([score["correlation"] for score in scores]),
    atlas=aligned_atlas, source_mask=brain.mask, roi_labels=roi_labels,
)
```

`similarity` returns one result dict per parcel. If you select or reorder matrices,
apply the same selection to `roi_labels` before painting their values.
`spatial_scale="searchlight"` returns one matrix per masked voxel in source-mask
order. Map its per-center scores with `nilearn.masking.unmask(values, brain.mask)`;
for a selected subset, restore the selected positions in a full mask-length
vector first. Adjacency carries no spatial mapping state.

## Raw arrays

Every method above has a function underneath that takes plain numpy:

```python
from nltools.algorithms import compute_similarity, fisher_r_to_z, matrix_permutation_test

r = compute_similarity(brain.data[0], brain.data[1:], metric="correlation")
fisher_r_to_z(r)

matrix_permutation_test(
    brain_rdm.squareform(), model_rdm.squareform(),
    n_permute=1000, metric="spearman", random_state=0,
)
```

Next: [Statistics & inference](statistics-and-inference.md), or the
[MVPA tutorial](../tutorials/workflows/03_mvpa.md), which runs RSA at all three spatial scales.
