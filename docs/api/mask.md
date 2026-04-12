## `mask`

NeuroLearn Mask Classes
=======================

Classes to represent masks

**Methods:**

Name | Description
---- | -----------
[`collapse_mask`](#collapse_mask) | collapse separate masks into one mask with multiple integers
[`create_sphere`](#create_sphere) | Generate a set of spheres in the brain mask space
[`expand_mask`](#expand_mask) | expand a mask with multiple integers into separate binary masks
[`roi_to_brain`](#roi_to_brain) | Populate an expanded binary ROI mask with a vector or matrix of per-ROI values.



### Methods

#### `collapse_mask`

```python
collapse_mask(mask, auto_label = True, custom_mask = None)
```

collapse separate masks into one mask with multiple integers
    overlapping areas are ignored

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask` |  | nibabel or BrainData instance | *required*
`custom_mask` |  | nibabel instance or string to file path; optional | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | BrainData instance of a mask with different integers indicating different masks

#### `create_sphere`

```python
create_sphere(coordinates, radius = 5, mask = None)
```

Generate a set of spheres in the brain mask space

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`radius` |  | vector of radius.  Will create multiple spheres if     len(radius) > 1 | <code>5</code>
`centers` |  | a vector of sphere centers of the form [px, py, pz] or     [[px1, py1, pz1], ..., [pxn, pyn, pzn]] | *required*

#### `expand_mask`

```python
expand_mask(mask, custom_mask = None)
```

expand a mask with multiple integers into separate binary masks

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask` |  | nibabel or BrainData instance | *required*
`custom_mask` |  | nibabel instance or string to file path; optional | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | BrainData instance of multiple binary masks

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

