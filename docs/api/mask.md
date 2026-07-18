(mask-mask)=
## `mask`

Utilities for creating and manipulating brain masks.

**Methods:**

Name | Description
---- | -----------
[`collapse_mask`](#mask-collapse-mask) | Collapse separate masks into one integer-labeled mask.
[`create_sphere`](#mask-create-sphere) | Generate spheres in brain-mask space.
[`expand_mask`](#mask-expand-mask) | Expand an integer-labeled mask into separate binary masks.
[`roi_to_brain`](#mask-roi-to-brain) | Populate an expanded binary ROI mask with a vector or matrix of per-ROI values.
[`roi_to_brain_from_atlas`](#mask-roi-to-brain-from-atlas) | Paint per-parcel values onto voxel space using a labeled atlas.



### Methods

(mask-collapse-mask)=
#### `collapse_mask`

```python
collapse_mask(mask, auto_label = True, custom_mask = None)
```

Collapse separate masks into one integer-labeled mask.

Overlapping areas are ignored.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask` |  | nibabel or BrainData instance holding 2+ separate masks (stacked along the first axis). | *required*
`auto_label` |  | If True (default), label the collapsed regions with sequential integers (1, 2, 3, …) in mask order. If False, keep each mask's own values as its label. | <code>True</code>
`custom_mask` |  | nibabel instance or string to file path; optional. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | BrainData instance of a mask with different integers indicating different masks.

(mask-create-sphere)=
#### `create_sphere`

```python
create_sphere(coordinates, radius = 5, mask = None)
```

Generate spheres in brain-mask space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`coordinates` |  | a vector of sphere centers of the form `[px, py, pz]` or `[[px1, py1, pz1], ..., [pxn, pyn, pzn]]` | *required*
`radius` |  | radius of the sphere(s). A scalar creates one sphere per center; a vector creates multiple spheres if `len(radius) > 1` | <code>5</code>
`mask` |  | `Nifti1Image` (or path to a mask file) defining the brain space. Defaults to the package brain-space mask when None. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Nifti1Image` |  | A binary image with the requested spheres in mask space.

(mask-expand-mask)=
#### `expand_mask`

```python
expand_mask(mask, custom_mask = None)
```

Expand an integer-labeled mask into separate binary masks.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask` |  | nibabel or BrainData instance | *required*
`custom_mask` |  | nibabel instance or string to file path; optional | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | BrainData instance of multiple binary masks

(mask-roi-to-brain)=
#### `roi_to_brain`

```python
roi_to_brain(data, mask_x)
```

Populate an expanded binary ROI mask with a vector or matrix of per-ROI values.

Accepts lists, numpy arrays, polars DataFrame/Series, or pandas
DataFrame/Series. Internally coerces to a numpy array and operates on
it — 1-D input produces a single BrainData image; 2-D input (ROIs by
observations) produces a stack of BrainData images, one per
observation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | ROI values. 1-D length must equal len(mask_x); 2-D shape must be (n_rois, n_obs) or (n_obs, n_rois). | *required*
`mask_x` |  | An expanded binary mask (BrainData) with one row per ROI. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | A BrainData instance with each ROI populated by the
 |  | provided value(s).

(mask-roi-to-brain-from-atlas)=
#### `roi_to_brain_from_atlas`

```python
roi_to_brain_from_atlas(values, atlas, source_mask, *, roi_labels = None, fill: float = np.nan)
```

Paint per-parcel values onto voxel space using a labeled atlas.

Sibling of `roi_to_brain`, but accepts a *labeled* atlas (one
integer label per voxel — the form carried by
`SpatialScale`), not an expanded mask
with one binary row per ROI. Voxels whose atlas label is not in
``roi_labels`` (or whose label is 0) receive ``fill``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`values` |  | Per-parcel scalars, either 1-D `(n_parcels,)` for a single image or 2-D `(n_images, n_parcels)` for a stack of images. The trailing (parcel) axis must match `len(roi_labels)` (or the number of unique non-zero atlas labels when `roi_labels` is None). | *required*
`atlas` |  | Labeled image — ``BrainData``, ``Nifti1Image``, or path-like. Resampled to ``source_mask`` (nearest-neighbor) if shapes/affines differ. | *required*
`source_mask` |  | ``Nifti1Image`` (or path) defining the output voxel grid. The returned ``BrainData`` is masked to this image. | *required*
`roi_labels` |  | Integer atlas IDs in the same order as ``values``. If None, defaults to ``np.unique`` of the atlas with 0 stripped (sorted ascending). | <code>None</code>
`fill` | <code>[float](#float)</code> | Value for voxels not in any provided ROI. Default ``np.nan``. | <code>[nan](#numpy.nan)</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Masked to `source_mask`, with each in-atlas voxel set to its
 |  | parcel's scalar from `values`. Holds a single image when `values` is
 |  | 1-D, or `n_images` images when `values` is 2-D `(n_images, n_parcels)`.

**Examples:**

```pycon
>>> from nltools.mask import roi_to_brain_from_atlas
>>> brain_map = roi_to_brain_from_atlas(
...     values=accuracies,
...     atlas=atlas_img,
...     source_mask=brain_mask,
...     roi_labels=[1, 2, 3],
... )
```

