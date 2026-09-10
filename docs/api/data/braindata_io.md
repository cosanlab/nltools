---
title: data.braindata.io
label: page-data-braindata-io
---

Loading, resampling, writing, and uploading for `BrainData`.

Functions that resolve a mask, load data (from files, lists, URLs, HDF5, or other
`BrainData` objects), resample to a target grid, write NIfTI/HDF5, and upload to
NeuroVault. `BrainData` methods delegate here.

**Functions:**

Name | Description
---- | -----------
[`check_space_match`](#data-braindata-io-check-space-match) | Check if data and mask are in same space.
[`detect_and_update_mask`](#data-braindata-io-detect-and-update-mask) | Detect best matching template from data and update mask if mask was None.
[`detect_space`](#data-braindata-io-detect-space) | Detect if mask is in MNI space or native space.
[`get_interpolation`](#data-braindata-io-get-interpolation) | Get the interpolation method to use for a given image.
[`initialize_mask`](#data-braindata-io-initialize-mask) | Initialize the mask image.
[`load_from_brain_data`](#data-braindata-io-load-from-brain-data) | Load data from another BrainData object.
[`load_from_file`](#data-braindata-io-load-from-file) | Load data from file path or nibabel object.
[`load_from_h5`](#data-braindata-io-load-from-h5) | Load data from HDF5 file.
[`load_from_list`](#data-braindata-io-load-from-list) | Load data from a list of BrainData objects or file paths.
[`load_from_url`](#data-braindata-io-load-from-url) | Load data from URL.
[`mask_images`](#data-braindata-io-mask-images) | Mask a list of space-aligned images with a single fitted masker.
[`resample`](#data-braindata-io-resample) | Resample BrainData onto a new voxel grid.
[`to_nifti`](#data-braindata-io-to-nifti) | Convert BrainData instance to a nibabel NIfTI image.
[`upload_neurovault`](#data-braindata-io-upload-neurovault) | Upload data to NeuroVault.
[`warn_if_resampling`](#data-braindata-io-warn-if-resampling) | Emit a `ResamplingWarning` if ``verbose=True`` and ``resample=True``.
[`write_brain_data`](#data-braindata-io-write-brain-data) | Write out BrainData object to Nifti or HDF5 File.

## Functions

(data-braindata-io-check-space-match)=
### `check_space_match`

```python
check_space_match(data_img, mask_img)
```

Check if data and mask are in same space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data_img` | <code>Nifti1Image</code> | Data image. | *required*
`mask_img` | <code>Nifti1Image</code> | Mask image. | *required*

**Returns:**

Type | Description
---- | -----------
<code>bool</code> | True if affines and spatial shapes match (no resampling needed).

(data-braindata-io-detect-and-update-mask)=
### `detect_and_update_mask`

```python
detect_and_update_mask(bd, data_img)
```

Detect best matching template from data and update mask if mask was None.

Also handles resampling if needed based on the resample kwarg.

This function is called during data loading to auto-detect template when mask=None.
After detecting or falling back to a template, it checks if resampling is needed
and resamples the data_img accordingly.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Instance whose mask may be updated. | *required*
`data_img` | <code>Nifti1Image</code> | Image from which to detect the template. | *required*

**Returns:**

Type | Description
---- | -----------
<code>Nifti1Image</code> | The input image, resampled to the mask grid if needed.

(data-braindata-io-detect-space)=
### `detect_space`

```python
detect_space(mask)
```

Detect if mask is in MNI space or native space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask` | <code>Nifti1Image</code> | Mask image to classify. | *required*

**Returns:**

Type | Description
---- | -----------
<code>str</code> | 'mni' if the mask matches the MNI template, 'native' otherwise.

(data-braindata-io-get-interpolation)=
### `get_interpolation`

```python
get_interpolation(bd, img)
```

Get the interpolation method to use for a given image.

Resolves 'auto' to either 'nearest' or 'continuous' based on data type.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Instance whose interpolation setting is consulted. | *required*
`img` | <code>Nifti1Image</code> | Image to inspect when the setting is 'auto'. | *required*

**Returns:**

Type | Description
---- | -----------
<code>str</code> | Interpolation method. When the instance setting is 'auto', resolves     to 'nearest' or 'continuous' based on data type; otherwise the     instance's configured interpolation setting.

(data-braindata-io-initialize-mask)=
### `initialize_mask`

```python
initialize_mask(bd, mask)
```

Initialize the mask image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Instance whose mask is being set. | *required*
`mask` | <code>Nifti1Image \| str \| Path \| None</code> | Brain mask as a nibabel image, file path, template name string, or None. Template name strings follow `'{res}mm-MNI152-2009{version}'` (e.g. `'2mm-MNI152-2009c'`, `'3mm-MNI152-2009a'`, `'2mm-MNI152-2009fsl'`). | *required*

(data-braindata-io-load-from-brain-data)=
### `load_from_brain_data`

```python
load_from_brain_data(bd, brain_data, mask = None)
```

Load data from another BrainData object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Instance to populate. | *required*
`brain_data` | <code>[BrainData](#page-data-brain-data)</code> | Object to copy from. | *required*
`mask` | <code>Nifti1Image \| str \| Path \| None</code> | Mask to use. If None, uses the mask from `brain_data`. | <code>None</code>

(data-braindata-io-load-from-file)=
### `load_from_file`

```python
load_from_file(bd, data)
```

Load data from file path or nibabel object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Instance to populate. | *required*
`data` | <code>str \| Path \| Nifti1Image</code> | File path or nibabel image. | *required*

(data-braindata-io-load-from-h5)=
### `load_from_h5`

```python
load_from_h5(bd, file_path, mask)
```

Load data from HDF5 file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Instance to populate. | *required*
`file_path` | <code>str \| Path</code> | Path to the HDF5 file. | *required*
`mask` | <code>Nifti1Image \| str \| Path \| None</code> | User-specified mask; when None the mask stored in the file is used. | *required*

(data-braindata-io-load-from-list)=
### `load_from_list`

```python
load_from_list(bd, data_list)
```

Load data from a list of BrainData objects or file paths.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Instance to populate. | *required*
`data_list` | <code>list[[BrainData](#page-data-brain-data)] \| list[str \| Path \| Nifti1Image]</code> | Items to load and stack. | *required*

(data-braindata-io-load-from-url)=
### `load_from_url`

```python
load_from_url(bd, url)
```

Load data from URL.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Instance to populate. | *required*
`url` | <code>str</code> | URL of a NIfTI file to download. | *required*

(data-braindata-io-mask-images)=
### `mask_images`

```python
mask_images(mask, imgs)
```

Mask a list of space-aligned images with a single fitted masker.

Validates ``mask`` exactly ONCE — one ``load_mask_img`` — and reuses the
binarized mask across every image in ``imgs``, instead of re-running
nilearn's costly ``load_mask_img`` (binarization checks + ``safe_get_data``,
which each trigger nilearn's forced ``gc.collect``) per image.

``nilearn.masking.apply_mask`` is exactly ``load_mask_img`` (validate) ->
``new_img_like`` (build binary mask) -> ``apply_mask_fmri`` (extract), with
``dtype='f'``, ``smoothing_fwhm=None``, ``ensure_finite=True``. This hoists
the first two out of the per-image loop and calls the lower-level
``apply_mask_fmri`` (which "assumes mask_img contains only two different
values") per image, so the result is byte-equivalent to
``np.vstack([apply_mask(im, mask) for im in imgs])`` for space-aligned data.

Images must already share ``mask``'s space (callers resample first); no
resampling is done here. Falls back to the per-image functional
``apply_mask`` if the fast path raises for any reason.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask` | <code>Nifti1Image</code> | Boolean/binary mask image. | *required*
`imgs` | <code>list[Nifti1Image]</code> | Space-aligned images to mask. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Masked data of shape ``(len(imgs), n_voxels)``.

(data-braindata-io-resample)=
### `resample`

```python
resample(bd, *, img = None, resolution = None, interpolation = None)
```

Resample BrainData onto a new voxel grid.

Exactly one of `img` or `resolution` must be given. An `img` supplies only
the target grid; its intensity values never define the output mask. The
source mask is resampled onto that grid with nearest-neighbor interpolation
and installed on the result, which preserves row-aligned `X` and `Y` and
carries no fitted state.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Instance to resample. | *required*
`img` | <code>Nifti1Image \| str \| Path \| None</code> | Target image whose grid to match, as a nibabel image or a path to a `.nii`/`.nii.gz` file. | <code>None</code>
`resolution` | <code>float \| int \| None</code> | Target isotropic voxel size in mm (e.g. `2.0` for 2 mm³ voxels). | <code>None</code>
`interpolation` | <code>str \| None</code> | Interpolation method for the data: `'nearest'` (atlases, masks, labels), `'linear'`, or `'continuous'` (higher-order spline, for stat maps). None uses the instance's interpolation setting. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | New instance with resampled data and mask.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If both `img` and `resolution` are None, both are provided, `resolution` is not positive, or the instance is empty.
<code>TypeError</code> | If `img` is not a valid image type.

(data-braindata-io-to-nifti)=
### `to_nifti`

```python
to_nifti(bd)
```

Convert BrainData instance to a nibabel NIfTI image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Instance to convert. | *required*

**Returns:**

Type | Description
---- | -----------
<code>Nifti1Image</code> | Brain data in volumetric NIfTI format.

(data-braindata-io-upload-neurovault)=
### `upload_neurovault`

```python
upload_neurovault(bd, *, access_token = None, collection_name = None, collection_id = None, img_type = None, img_modality = None, **kwargs)
```

Upload data to NeuroVault.

Adds any columns in `bd.X` to image metadata. Index will be used as image name.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Images to upload. | *required*
`access_token` | <code>str</code> | NeuroVault API access token. Required. | <code>None</code>
`collection_name` | <code>str \| None</code> | Name of a new collection to create. | <code>None</code>
`collection_id` | <code>int \| None</code> | NeuroVault collection ID when adding images to an existing collection. | <code>None</code>
`img_type` | <code>str</code> | NeuroVault map type (e.g. `'Z'`, `'T'`). Required. | <code>None</code>
`img_modality` | <code>str</code> | NeuroVault image modality (e.g. `'fMRI-BOLD'`). Required. | <code>None</code>
`**kwargs` | <code>dict</code> | Additional image metadata forwarded to `pynv.Client.add_image`. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | NeuroVault collection information.

(data-braindata-io-warn-if-resampling)=
### `warn_if_resampling`

```python
warn_if_resampling(bd, context = '')
```

Emit a `ResamplingWarning` if ``verbose=True`` and ``resample=True``.

Sibling of the template-mismatch notice in `match_resolution`: that one
fires when a template is chosen for data at another resolution; this one
fires when the data is actually resampled to the mask's grid.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Instance whose `verbose` and resample settings apply. | *required*
`context` | <code>str</code> | Why the spaces differ, appended to the message. Default: empty string. | <code>''</code>

(data-braindata-io-write-brain-data)=
### `write_brain_data`

```python
write_brain_data(bd, file_name)
```

Write out BrainData object to Nifti or HDF5 File.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Instance to write. | *required*
`file_name` | <code>str \| Path</code> | Output file path. Supports `.nii`/`.nii.gz` (NIfTI) and `.h5`/`.hdf5` (HDF5) formats. | *required*
