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
[`roi_to_brain`](#roi_to_brain) | This function will create convert an expanded binary mask of ROIs



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

This function will create convert an expanded binary mask of ROIs
(see expand_mask) based on a vector of of values. The dataframe of values
must correspond to ROI numbers.

This is useful for populating a parcellation scheme by a vector of Values

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | Pandas series, dataframe, list, np.array of ROI by observation | *required*
`mask_x` |  | an expanded binary mask | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (BrainData) BrainData instance where each ROI is now populated  with a value

