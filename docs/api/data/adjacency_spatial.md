---
title: data.adjacency.spatial
label: data-adjacency-spatial
---

Spatial-scale provenance for stacked Adjacency matrices.

When a stack of Adjacency matrices comes from a per-parcel or per-searchlight
operation on a BrainData, attaching a `SpatialScale` records the atlas,
the parcel labels in stack order, and the source mask — enough to project
per-matrix reductions back to a voxel-space `BrainData` via
``Adjacency.to_brain()``.

See `Adjacency` for the optional ``spatial_scale`` attribute, and
`BrainData.distance` (with ``spatial_scale='roi'|'searchlight'``) for
the canonical producer.

**Classes:**

Name | Description
---- | -----------
[`SpatialScale`](#data-adjacency-spatial-spatialscale) | Record provenance for a per-parcel or per-searchlight Adjacency stack.



## Classes

(data-adjacency-spatial-spatialscale)=
### `SpatialScale`

```python
SpatialScale(atlas: BrainData, roi_labels: np.ndarray, source_mask: Nifti1Image, kind: Literal['roi', 'searchlight'] = 'roi')
```

Record provenance for a per-parcel or per-searchlight Adjacency stack.

The stack comes from a per-parcel or per-searchlight operation on a
`BrainData`.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`atlas` | <code>[BrainData](#data-brain-data)</code> | Labeled volume indicating parcel membership (or searchlight centers). One matrix in the stack per unique label.
`roi_labels` | <code>ndarray</code> | Integer atlas IDs in stack order. ``len(roi_labels)`` must equal the number of matrices in the stack.
`source_mask` | <code>Nifti1Image</code> | The brain mask the atlas/values live in. Used as the target space for back-projection in ``Adjacency.to_brain()``.
`kind` | <code>Literal['roi', 'searchlight']</code> | Which spatial scale produced this stack — ``'roi'`` or ``'searchlight'``.
