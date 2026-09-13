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
[`align`](../api/tasks/alignment.md#nltools.algorithms.align)`(..., method='procrustes')` | Building a common model from a group, iteratively | Same voxel count out
`align(..., method='probabilistic_srm')` | You want a low-dimensional shared response and a noise model | Probabilistic, slower; `n_features=` sets the shared dimensionality
`align(..., method='deterministic_srm')` | Same, without the probabilistic machinery | Faster and deterministic; the usual default

Goal | Use | Notes
--- | --- | ---
Align one subject to another | [`BrainData.align`](../api/data/brain_data.md#nltools.data.braindata.BrainData.align)`(target, method='procrustes')` | Returns `transformed`, `transformation_matrix`, `common_model`, `disparity`, `scale`
Build a group common model | [`align`](../api/tasks/alignment.md#nltools.algorithms.align)`(list_of_arrays, method=)` | `'procrustes'`, `'probabilistic_srm'`, `'deterministic_srm'`
Project a new subject in | `BrainData.align(common_model, method='deterministic_srm')` | The target is the fitted model array, not a subject; `transformed` comes back as an array on the model's feature axis
Per-ROI | `BrainData.align(..., spatial_scale='roi')` | Needs `roi_mask=`
Raw matrix superposition | [`procrustes`](../api/tasks/alignment.md#nltools.algorithms.alignment.procrustes.procrustes) | Returns `(mtx1, mtx2, disparity, R, scale)`
Test two matrices' similarity | [`procrustes_distance`](../api/tasks/alignment.md#nltools.algorithms.procrustes_distance) | Permutation test on the Procrustes disparity
Match state maps across groups | [`align_states`](../api/tasks/alignment.md#nltools.algorithms.align_states) | For comparing decompositions, not timeseries

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
projected["transformed"]                  # array, (n_samples, n_features)
projected["transformation_matrix"]        # BrainData, n_features voxel maps
```

The SRM `transformed` and `common_model` are plain arrays because they live on the model's
feature axis, not on voxels. Only `transformation_matrix` is a `BrainData` — it is the stack of
`n_features` voxel maps, so `projected["transformed"] @ projected["transformation_matrix"].data`
puts the aligned data back in the subject's voxel space. Procrustes has no feature axis, so
there every value is a `BrainData`.

Cross-validation depends on that order: fit the common model on training subjects only, then project
test subjects into it. Fitting the model on everyone and then decoding across subjects leaks.

## Estimators

```python
from nltools.algorithms import DetSRM, procrustes

srm = DetSRM(n_features=10, n_iter=5)
srm.fit([s.data.T for s in subjects])          # each array is voxels x samples
srm.transform([s.data.T for s in subjects])

mtx1, mtx2, disparity, R, scale = procrustes(subjects[0].data, subjects[1].data)
```

Note the transpose. `BrainData.data` is `(n_samples, n_voxels)`, but the sklearn-style estimators
(`SRM`, `DetSRM`) take `(n_voxels, n_samples)`, the shape convention from the original SRM
implementations. `BrainData.align` and `nltools.algorithms.align` handle this for you; the
estimators do not.

## Gotchas

- On the estimators, `n_iter=` names solver iterations (EM steps or coordinate-descent rounds),
  not permutations.
- Procrustes back-projection is `transformed @ transformation_matrix.T` on every entry point.
- SRM subjects must all have the same number of samples.

Next: [Intersubject correlation](intersubject.md), the usual reason to align.
