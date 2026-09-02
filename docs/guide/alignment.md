---
title: Functional alignment
---

Anatomical normalization lines up sulci; it does not line up *function*. Functional alignment
learns a per-subject transform into a shared response space, estimated from responses to a common
stimulus. Use it before any analysis that assumes voxel *i* means the same thing in every
subject: cross-subject decoding, group RSA, ISC on fine-grained patterns.

Pick a method by what you have and what you need back:

Method | Use when | Trade-off
--- | --- | ---
`'procrustes'` (hyperalignment) | Aligning one subject to a reference subject or common model | Orthogonal rotation, no dimensionality reduction; invertible
[`HyperAlignment`](../api/tasks/alignment.md#tasks-alignment-hyperalignment) | Building a common model from a group, iteratively | Same voxel count out; `n_iter=2` is usually enough
[`SRM`](../api/tasks/alignment.md#tasks-alignment-srm) | You want a low-dimensional shared response and a noise model | Probabilistic, slower; `features=` sets the shared dimensionality
[`DetSRM`](../api/tasks/alignment.md#tasks-alignment-detsrm) | Same, without the probabilistic machinery | Faster and deterministic; the usual default
[`LocalAlignment`](../api/tasks/alignment.md#tasks-alignment-localalignment) | One transform per ROI or searchlight, not one for the whole brain | Respects local topography; far more compute

Goal | Use | Notes
--- | --- | ---
Align one subject to another | [`BrainData.align`](../api/data/brain_data.md#data-brain-data-align)`(target, method='procrustes')` | Returns `transformed`, `transformation_matrix`, `common_model`, `disparity`, `scale`
Build a group common model | [`align`](../api/tasks/alignment.md#tasks-alignment-align)`(list_of_arrays, method=)` | `'procrustes'`, `'probabilistic_srm'`, `'deterministic_srm'`
Project a new subject in | `BrainData.align(common_model, method='deterministic_srm')` | The target is the fitted model array, not a subject
Local (ROI/searchlight) | `BrainData.align(..., spatial_scale='roi'\|'searchlight')`, or [`LocalAlignment`](../api/tasks/alignment.md#tasks-alignment-localalignment) | Needs `roi_mask=` or `radius_mm=`
Align a whole collection | [`BrainCollection.align`](../api/data/brain_collection.md#data-brain-collection-align) | `spatial_scale='searchlight'` by default; `return_model=True` keeps the transforms
Raw matrix superposition | [`procrustes`](../api/tasks/alignment.md#tasks-alignment-procrustes) | Returns `(mtx1, mtx2, disparity, R, scale)`
Test two matrices' similarity | [`procrustes_distance`](../api/tasks/alignment.md#tasks-alignment-procrustes-distance) | Permutation test on the Procrustes disparity
Match state maps across groups | [`align_states`](../api/tasks/alignment.md#tasks-alignment-align-states) | For comparing decompositions, not timeseries

## Pairwise and group

```python
out = subjects[0].align(subjects[1], method="procrustes")
out["transformed"], out["transformation_matrix"], out["disparity"]
```

`align` on a list of arrays fits the common model for the whole group in one call. The result
carries `common_model`, the per-subject `transformation_matrix` list, the `transformed` data, and an
`isc` value for how well the model captures shared response:

```python
from nltools.algorithms import align

model = align([s.data for s in subjects], method="deterministic_srm", n_features=10)
model["common_model"].shape       # (n_samples, n_features)
```

Once you have a common model, a held-out subject joins it by aligning *to the model array* rather
than to another subject:

```python
projected = new_subject.align(model["common_model"], method="deterministic_srm")
```

Cross-validation depends on that order: fit the common model on training subjects only, then project
test subjects into it. Fitting the model on everyone and then decoding across subjects leaks.

## Estimators and local alignment

```python
from nltools.algorithms import DetSRM, HyperAlignment, LocalAlignment, procrustes

srm = DetSRM(features=10, n_iter=5)
srm.fit([s.data.T for s in subjects])          # each array is voxels x samples
srm.transform([s.data.T for s in subjects])

mtx1, mtx2, disparity, R, scale = procrustes(subjects[0].data, subjects[1].data)

local = LocalAlignment(spatial_scale="searchlight", radius_mm=12.0, n_iter=2, n_jobs=2)
aligned = local.fit_transform([s.data.T for s in subjects], mask)
```

Note the transpose. `BrainData.data` is `(n_samples, n_voxels)`, but the sklearn-style estimators
(`SRM`, `DetSRM`, `LocalAlignment`) take `(n_voxels, n_samples)`, the shape convention from the
original SRM implementations. `BrainData.align` and `nltools.algorithms.align` handle this for you;
the estimators do not.

## Gotchas

- The alignment subsystem keeps legacy kwarg names at its boundary: `parallel=` instead of
  `device=`, `n_iter=` for solver iterations. The class facades translate.
- `LocalAlignment` fits one model per neighborhood. On a whole brain that is tens of thousands of
  models, so start with an ROI mask and budget time before running `spatial_scale='searchlight'`.
- SRM subjects may differ in sample count; shorter ones are zero-padded within each neighborhood.

Next: [Intersubject correlation](intersubject.md), the usual reason to align.
