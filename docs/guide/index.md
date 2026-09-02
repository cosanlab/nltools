---
title: User Guide
---

One page per task: which nltools call to reach for, in what order, and what will bite you.
The [Tutorials](../tutorials/index.md) are executed narratives you read start to finish; the
[Reference](../api/data/brain_data.md) is the generated signature for every public symbol. These
pages sit between the two, short and linked into both.

## The four data classes

Everything in nltools flows through four objects. Each flattens a neuroimaging structure into a
2-D array so ordinary array operations work on it.

Class | Holds | You get one from
--- | --- | ---
[`BrainData`](../api/data/brain_data.md) | Images × in-mask voxels | A NIfTI path, a list of paths, an array + mask, an `.h5` file
[`DesignMatrix`](../api/data/design_matrix.md) | Timepoints × regressors | A BIDS events `.tsv`, a confounds table, an array, a dict
[`Adjacency`](../api/data/adjacency.md) | Matrices × node pairs | `BrainData.distance`, a square matrix, a stack of them
[`BrainCollection`](../api/data/brain_collection.md) | One `BrainData` per subject | `from_bids`, `from_glob`, `from_paths`, `read`

Three flows cover most analyses:

- **Univariate.** Load a run → build a design → `fit` → `compute_contrasts` → stack the
  per-subject maps → `ttest` → threshold. See [Design matrices & GLM](design-and-glm.md).
- **Multivariate.** Load images → `predict` (decoding) or `distance` → `Adjacency` →
  `similarity` (RSA). See [Prediction](prediction.md) and [Similarity & RSA](similarity-and-rsa.md).
- **Many subjects.** Build a `BrainCollection` and call the same method names; they run per
  subject in parallel and cache to disk. See [Working with many subjects](collections.md).

## Pages

Page | Covers
--- | ---
[Loading data & masks](loading.md) | `BrainData` from files and URLs, masks, the resolution rule, HDF5, bundled datasets
[Design matrices & GLM](design-and-glm.md) | Events → regressors, confounds, drift, collinearity warnings, `fit`, contrasts, group tests
[Prediction: encoding & decoding](prediction.md) | `predict`, cross-validation specs, ROI and searchlight scales, ridge encoding, ROC
[Similarity & RSA](similarity-and-rsa.md) | Brain RDMs, model RDMs, Mantel tests, `plot_stacked_adjacency`, painting results back on the brain
[Functional alignment](alignment.md) | Hyperalignment vs SRM vs local alignment, common models, transforming new subjects
[Statistics & inference](statistics-and-inference.md) | t-tests, permutation, bootstrap, FDR/Holm, `tail`, `n_permute` vs `n_samples`, `n_jobs` vs `device`
[Intersubject correlation](intersubject.md) | `isc`, `isfc`, `isps`, group comparisons, whole-brain ISC over a collection
[Plotting](plotting.md) | Which plot for which object, thresholds, interactive viewers, saving
[Atlases & cluster reports](atlases.md) | Bundled parcellations, anatomical labels, parcel summaries, cluster tables
[Working with many subjects](collections.md) | Lazy items, the cache, the fit → contrast → group chain, `predict` vs `predict_group`, errors
[Performance & GPU](../performance.md) | When the GPU backend pays off, and by how much

## Kwarg conventions

- `method=` picks an algorithm variant; `metric=` picks a distance or similarity measure;
  `summary=` picks a central tendency (`'mean'` or `'median'`). They are never interchangeable.
- `spatial_scale=` (`'whole_brain'`, `'roi'`, `'searchlight'`) picks the scale an analysis runs at.
- `n_jobs=` sets CPU workers; `device=` picks CPU or GPU. An explicit `device='gpu'` runs on the
  GPU or raises; only `'auto'` falls back.
- `n_permute=` counts permutations; `n_samples=` counts bootstrap draws.
- `random_state=` makes any resampling reproducible, including on the GPU.

The full manifest is in the [architecture notes](../development/index.md#canonical-api-vocabulary).
