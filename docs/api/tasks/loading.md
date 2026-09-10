---
title: Loading, masks & datasets
label: page-tasks-loading
---

Get data into nltools. [BrainData](../data/brain_data.md) and the other data classes load NIfTI files and HDF5 bundles themselves. The functions here cover the rest: example datasets and Neurovault collections, sphere and ROI masks, `concatenate`, and the MNI template every object falls back on when it gets no mask (`set_brainspace`).

**Classes:**

Name | Description
---- | -----------
[`BrainSpaceConfig`](#tasks-loading-brainspaceconfig) | Immutable MNI template configuration.

**Functions:**

Name | Description
---- | -----------
[`load_brain_data_h5`](#tasks-loading-load-brain-data-h5) | Load BrainData contents from an HDF5 file.
[`to_h5`](#tasks-loading-to-h5) | Save BrainData or Adjacency objects to HDF5 files.
[`is_h5_path`](#tasks-loading-is-h5-path) | Check if a file path indicates an HDF5 file.
[`fetch_pain`](#tasks-loading-fetch-pain) | Download and load the pain dataset from the nltools HF dataset.
[`fetch_emotion_ratings`](#tasks-loading-fetch-emotion-ratings) | Download and load the emotion-rating dataset from the nltools HF dataset.
[`fetch_neurovault_collection`](#tasks-loading-fetch-neurovault-collection) | Download images and metadata from a Neurovault collection.
[`load_haxby_example`](#tasks-loading-load-haxby-example) | Load a small synthetic Haxby-like dataset, entirely in-memory.
[`download_nifti`](#tasks-loading-download-nifti) | Download an image from a URL to a nifti file.
[`create_sphere`](#tasks-loading-create-sphere) | Generate spheres in brain-mask space.
[`expand_mask`](#tasks-loading-expand-mask) | Expand an integer-labeled mask into separate binary masks.
[`collapse_mask`](#tasks-loading-collapse-mask) | Collapse separate masks into one integer-labeled mask.
[`collapse_label_stack`](#tasks-loading-collapse-label-stack) | Collapse a stack of binary masks into a single integer label vector.
[`roi_to_brain`](#tasks-loading-roi-to-brain) | Populate an expanded binary ROI mask with a vector or matrix of per-ROI values.
[`roi_to_brain_from_atlas`](#tasks-loading-roi-to-brain-from-atlas) | Paint per-parcel values onto voxel space using a labeled atlas.
[`concatenate`](#tasks-loading-concatenate) | Concatenate a list of `BrainData` or `Adjacency` objects.
[`get_brainspace`](#tasks-loading-get-brainspace) | Return the current global brain-space configuration.
[`set_brainspace`](#tasks-loading-set-brainspace) | Set the global brain-space configuration.
[`reset_brainspace`](#tasks-loading-reset-brainspace) | Reset the global brain-space configuration to defaults.
[`with_brainspace`](#tasks-loading-with-brainspace) | Temporarily change the global brain-space configuration.
[`fetch_resource`](#tasks-loading-fetch-resource) | Return a local path to a file from the `nltools/niftis` HF dataset.
[`list_resources`](#tasks-loading-list-resources) | List files available in the `nltools/niftis` HF dataset.
[`get_bg_image`](#tasks-loading-get-bg-image) | Get a background image path matching a data resolution.
[`is_standard_space`](#tasks-loading-is-standard-space) | Check whether an affine is compatible with our MNI templates.
[`detect_resolution`](#tasks-loading-detect-resolution) | Detect voxel resolution (mm) and isotropy from a NIfTI affine.

## Classes

(tasks-loading-brainspaceconfig)=
### `BrainSpaceConfig`

```python
BrainSpaceConfig(template: TemplateName = 'default', resolution: Resolution = 2)
```

Immutable MNI template configuration.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`template` | <code>str</code> | Template variant (`'default'`, `'nilearn'`, `'fmriprep'`).
`resolution` | <code>int</code> | Resolution in mm (1, 2, or 3).
`mask` | <code>str</code> | Path to the brain mask file.
`brain` | <code>str</code> | Path to the brain-extracted image.
`plot` | <code>str</code> | Path to the full T1 image used for plotting.

## Functions

(tasks-loading-load-brain-data-h5)=
### `load_brain_data_h5`

```python
load_brain_data_h5(file_path, mask = None)
```

Load BrainData contents from an HDF5 file.

Supports the v0.6 layout (`X`/`Y` as Arrow IPC byte datasets) and the legacy
deepdish/PyTables layout written by nltools <= 0.5.1 (`X`/`Y` as flat
datasets with sibling `X_columns`/`X_index` nodes).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_path` | <code>str \| Path</code> | Path to the HDF5 file. | *required*
`mask` | <code>Nifti1Image</code> | Mask to use. If None, the mask stored in the file is loaded when present. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'data'` (np.ndarray), `'X'` and `'Y'` (pl.DataFrame),     `'load_mask'` (bool), and `'mask'` (nibabel.Nifti1Image) when a mask was     loaded from the file.

(tasks-loading-to-h5)=
### `to_h5`

```python
to_h5(obj, file_name, obj_type = 'brain_data', h5_compression = 'gzip')
```

Save BrainData or Adjacency objects to HDF5 files.

Uses h5py for both types; the `X`/`Y` frames (BrainData) and `Y` (Adjacency)
are stored as Arrow IPC byte datasets so every polars dtype round-trips
exactly. A BrainData mask is always stored by value (data + affine
datasets); its filename is stored alongside only when the mask is
file-backed, so in-memory masks serialize without one and round-trip by value.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`obj` | <code>[BrainData](#page-data-brain-data) \| [Adjacency](#page-data-adjacency)</code> | Object to save. | *required*
`file_name` | <code>str \| Path</code> | Path to save the file to. | *required*
`obj_type` | <code>str</code> | `'brain_data'` or `'adjacency'`. | <code>'brain_data'</code>
`h5_compression` | <code>str</code> | Compression filter for h5py datasets. Default `'gzip'`. | <code>'gzip'</code>

(tasks-loading-is-h5-path)=
### `is_h5_path`

```python
is_h5_path(file_name) -> bool
```

Check if a file path indicates an HDF5 file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>str \| Path</code> | Path to check. | *required*

**Returns:**

Type | Description
---- | -----------
<code>bool</code> | True if the file has an HDF5 extension (`.h5` or `.hdf5`).

**Examples:**

```python
is_h5_path("data.h5")  # → True
is_h5_path("data.csv")  # → False
is_h5_path(Path("results.hdf5"))  # → True
```

(tasks-loading-fetch-pain)=
### `fetch_pain`

```python
fetch_pain(verbose = 0)
```

Download and load the pain dataset from the nltools HF dataset.

Loads the Chang et al. (2015) pain-perception study: 28 subjects x 3
stimulus-intensity conditions = 84 whole-brain contrast images, with a
curated metadata table (`SubjectID`, `PainLevel`, `PainIntensity`, `Age`,
`Sex`, provenance `neurovault_id` / `name`).

Data is hosted on the ``nltools/niftis`` Hugging Face dataset and cached
locally on first use, so this works with no extra setup.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`verbose` | <code>int</code> | Verbosity passed to `BrainData` while loading. Default: 0 | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | `BrainData` with the 84 images; `X` holds the metadata table.

<details class="references" open markdown="1">
<summary>References</summary>

Chang, L. J., Gianaros, P. J., Manuck, S. B., Krishnan, A., & Wager, T. D. (2015).
A sensitive and specific neural signature for picture-induced negative affect.
PLoS biology, 13(6), e1002180.

</details>

(tasks-loading-fetch-emotion-ratings)=
### `fetch_emotion_ratings`

```python
fetch_emotion_ratings(verbose = 0)
```

Download and load the emotion-rating dataset from the nltools HF dataset.

Loads the Chang et al. (2015) IAPS emotion-rating study: 679 whole-brain
contrast images across 150 subjects, each rating images 1-5, with a
built-in train/test holdout split. `X` carries the full portable Neurovault
metadata (key columns: `SubjectID`, `Rating`, `Holdout`, `AGE`, `SEX`).

Data is hosted on the ``nltools/niftis`` Hugging Face dataset and cached
locally on first use, so this works with no extra setup.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`verbose` | <code>int</code> | Verbosity passed to `BrainData` while loading. Default: 0 | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | `BrainData` with the 679 images; `X` holds the metadata table.

<details class="references" open markdown="1">
<summary>References</summary>

Chang, L. J., Gianaros, P. J., Manuck, S. B., Krishnan, A., & Wager, T. D. (2015).
A sensitive and specific neural signature for picture-induced negative affect.
PLoS biology, 13(6), e1002180.

</details>

(tasks-loading-fetch-neurovault-collection)=
### `fetch_neurovault_collection`

```python
fetch_neurovault_collection(collection_id, data_dir = None, verbose = 1)
```

Download images and metadata from a Neurovault collection.

This function uses the modern nilearn API to download collections from Neurovault.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`collection_id` | <code>int</code> | Neurovault collection ID | *required*
`data_dir` | <code>str</code> | Directory to store downloaded data. If None, uses nilearn's default data directory. | <code>None</code>
`verbose` | <code>int</code> | Verbosity level. Default: 1 | <code>1</code>

**Returns:**

Type | Description
---- | -----------
<code>tuple[DataFrame, list[str]]</code> | `(metadata, files)` — the image metadata     table and the downloaded image paths.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If collection_id is invalid
<code>RuntimeError</code> | If download fails

(tasks-loading-load-haxby-example)=
### `load_haxby_example`

```python
load_haxby_example(n_runs = 1, random_state = 42)
```

Load a small synthetic Haxby-like dataset, entirely in-memory.

Returns paired lists of `BrainData` and `DesignMatrix`, one entry per
run, generated from a tiny synthetic volume (10 x 10 x 5 = 500 voxels)
with condition-specific signal injected into disjoint voxel clusters.
No network I/O, no disk I/O, no nilearn fetcher dependency. Runs in
well under a second.

Intended for tutorials, documentation examples, and tests where
downloading a real fMRI dataset is impractical. The
eight conditions match the real Haxby 2001 object-recognition experiment
(face, house, cat, bottle, scissors, shoe, chair, scrambledpix), arranged
in a randomized 9-TR block design with TR=2.5s.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_runs` | <code>int</code> | Number of runs to generate. Default 1. | <code>1</code>
`random_state` | <code>int \| None</code> | Seed for reproducible output. Default 42. | <code>42</code>

**Returns:**

Type | Description
---- | -----------
<code>tuple</code> | `(list[BrainData], list[DesignMatrix])`, each of length `n_runs`.     The DesignMatrix columns are the eight condition names suffixed     with `_c0` (HRF-convolved boxcars).

**Examples:**

```python
from nltools.datasets import load_haxby_example

brain_data, design_matrices = load_haxby_example()
data, dm = brain_data[0], design_matrices[0]
data.shape  # → (72, 500)
"face_c0" in dm.columns  # → True
```

(tasks-loading-download-nifti)=
### `download_nifti`

```python
download_nifti(url, data_dir = None)
```

Download an image from a URL to a nifti file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`url` | <code>str</code> | URL of the image to download | *required*
`data_dir` | <code>str</code> | Directory to save the file. If None, uses current directory. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>str</code> | Path to the downloaded file

**Raises:**

Type | Description
---- | -----------
<code>ImportError</code> | If requests is not available
<code>ValueError</code> | If URL is invalid

(tasks-loading-create-sphere)=
### `create_sphere`

```python
create_sphere(coordinates, radius = 5, mask = None)
```

Generate spheres in brain-mask space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`coordinates` | <code>list</code> | Sphere center `[x, y, z]` in voxel coordinates, or one center per sphere `[[x1, y1, z1], ...]`. | *required*
`radius` | <code>int \| float \| list</code> | Radius of the sphere(s) in voxels. A scalar applies to every center; a list gives one radius per center. | <code>5</code>
`mask` | <code>Nifti1Image \| str</code> | Image (or path) defining the brain space. Defaults to the package brain-space mask. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>Nifti1Image</code> | A binary image with the requested spheres in mask space.

(tasks-loading-expand-mask)=
### `expand_mask`

```python
expand_mask(mask, custom_mask = None)
```

Expand an integer-labeled mask into separate binary masks.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask` | <code>Nifti1Image \| [BrainData](#page-data-brain-data)</code> | Integer-labeled mask. | *required*
`custom_mask` | <code>Nifti1Image \| str</code> | Brain mask (or path) used when converting a nibabel `mask` to `BrainData`. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | One binary mask per unique non-zero label.

(tasks-loading-collapse-mask)=
### `collapse_mask`

```python
collapse_mask(mask, auto_label = True, custom_mask = None)
```

Collapse separate masks into one integer-labeled mask.

Overlapping areas are ignored.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask` | <code>Nifti1Image \| [BrainData](#page-data-brain-data)</code> | Two or more separate masks stacked along the first axis. | *required*
`auto_label` | <code>bool</code> | If True (default), label the collapsed regions with sequential integers (1, 2, 3, …) in mask order. If False, keep each mask's own values as its label. | <code>True</code>
`custom_mask` | <code>Nifti1Image \| str</code> | Brain mask (or path) used when converting a nibabel `mask` to `BrainData`. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | A single mask whose integer values identify the source masks.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If `mask` is neither a nibabel nor BrainData instance, or if it holds fewer than 2 masks (nothing to collapse).

(tasks-loading-collapse-label-stack)=
### `collapse_label_stack`

```python
collapse_label_stack(stack)
```

Collapse a stack of binary masks into a single integer label vector.

The array-level inverse of `expand_mask`: row *i* of ``stack`` becomes
label ``i + 1``. Voxels belonging to more than one mask are ambiguous and
are assigned label 0, as are voxels in no mask.

A stacked binary mask carries no label values of its own, so labels are
necessarily sequential in stack order. Round-tripping a 1..n atlas through
`expand_mask` therefore preserves its original labels; an atlas with
non-sequential labels (e.g. 3 and 7) comes back renumbered 1 and 2.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stack` |  | array of shape ``(n_masks, n_voxels)``. Nonzero means membership. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | integer labels of shape ``(n_voxels,)``.

**Examples:**

```python
labels = collapse_label_stack(expand_mask(atlas).data)
```

(tasks-loading-roi-to-brain)=
### `roi_to_brain`

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
`data` | <code>list \| ndarray \| DataFrame \| Series \| DataFrame \| Series</code> | ROI values. 1-D length must equal `len(mask_x)`; 2-D shape must be `(n_rois, n_obs)` or `(n_obs, n_rois)`. | *required*
`mask_x` | <code>[BrainData](#page-data-brain-data)</code> | An expanded binary mask with one row per ROI. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | A BrainData instance with each ROI populated by the     provided value(s).

(tasks-loading-roi-to-brain-from-atlas)=
### `roi_to_brain_from_atlas`

```python
roi_to_brain_from_atlas(values, atlas, source_mask, *, roi_labels = None, fill: float = np.nan)
```

Paint per-parcel values onto voxel space using a labeled atlas.

Sibling of `roi_to_brain`, but accepts a *labeled* atlas (one integer label
per voxel — the form carried by `SpatialScale`), not an expanded mask with
one binary row per ROI. Voxels whose atlas label is not in `roi_labels` (or
whose label is 0) receive `fill`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`values` | <code>ndarray</code> | Per-parcel scalars, either 1-D `(n_parcels,)` for a single image or 2-D `(n_images, n_parcels)` for a stack of images. The trailing (parcel) axis must match `len(roi_labels)` (or the number of unique non-zero atlas labels when `roi_labels` is None). | *required*
`atlas` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image \| str \| Path</code> | Labeled image. Resampled to `source_mask` (nearest-neighbor) if shapes/affines differ. | *required*
`source_mask` | <code>Nifti1Image \| str \| Path</code> | Image (or path) defining the output voxel grid. The returned `BrainData` is masked to this image. | *required*
`roi_labels` | <code>array - like</code> | Integer atlas IDs in the same order as `values`. If None, defaults to `np.unique` of the atlas with 0 stripped (sorted ascending). | <code>None</code>
`fill` | <code>float</code> | Value for voxels not in any provided ROI. Default `np.nan`. | <code>nan</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Masked to `source_mask`, with each in-atlas voxel set to its     parcel's scalar from `values`. Holds a single image when `values` is     1-D, or `n_images` images when `values` is 2-D `(n_images, n_parcels)`.

**Examples:**

```python
from nltools.mask import roi_to_brain_from_atlas

brain_map = roi_to_brain_from_atlas(
    values=accuracies,
    atlas=atlas_img,
    source_mask=brain_mask,
    roi_labels=[1, 2, 3],
)
```

(tasks-loading-concatenate)=
### `concatenate`

```python
concatenate(data)
```

Concatenate a list of `BrainData` or `Adjacency` objects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>list[[BrainData](#page-data-brain-data)] \| list[[Adjacency](#page-data-adjacency)]</code> | Objects to concatenate; all must be of the same class. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data) \| [Adjacency](#page-data-adjacency)</code> | A single object holding every input in order.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If `data` is not a list or mixes classes.

(tasks-loading-get-brainspace)=
### `get_brainspace`

```python
get_brainspace() -> BrainSpaceConfig
```

Return the current global brain-space configuration.

**Returns:**

Type | Description
---- | -----------
<code>[BrainSpaceConfig](#tasks-loading-brainspaceconfig)</code> | The active configuration.

(tasks-loading-set-brainspace)=
### `set_brainspace`

```python
set_brainspace(template: TemplateName | None = None, resolution: Resolution | None = None) -> BrainSpaceConfig
```

Set the global brain-space configuration.

Call with no arguments to return the current config without mutating it.
Call with one or both arguments to mutate the global state; unspecified
fields retain their current value.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`template` | <code>str</code> | Template name to set (`'default'`, `'nilearn'`, `'fmriprep'`). If None, keeps the current value. | <code>None</code>
`resolution` | <code>int</code> | Resolution in mm to set. If None, keeps the current value. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainSpaceConfig](#tasks-loading-brainspaceconfig)</code> | The new (or unchanged) current configuration.

(tasks-loading-reset-brainspace)=
### `reset_brainspace`

```python
reset_brainspace() -> BrainSpaceConfig
```

Reset the global brain-space configuration to defaults.

**Returns:**

Type | Description
---- | -----------
<code>[BrainSpaceConfig](#tasks-loading-brainspaceconfig)</code> | The default configuration (`'default'` template, 2 mm).

(tasks-loading-with-brainspace)=
### `with_brainspace`

```python
with_brainspace(template: TemplateName | None = None, resolution: Resolution | None = None) -> Iterator[BrainSpaceConfig]
```

Temporarily change the global brain-space configuration.

Restores the previous configuration on exit, even if an exception is
raised inside the block.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`template` | <code>str</code> | Template name for the duration of the block. | <code>None</code>
`resolution` | <code>int</code> | Resolution in mm for the duration of the block. | <code>None</code>

**Yields:**

Type | Description
---- | -----------
<code>[BrainSpaceConfig](#tasks-loading-brainspaceconfig)</code> | The configuration active inside the block.

(tasks-loading-fetch-resource)=
### `fetch_resource`

```python
fetch_resource(relpath: str) -> str
```

Return a local path to a file from the `nltools/niftis` HF dataset.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`relpath` | <code>str</code> | Path within the dataset repo, e.g. `'default/2mm-MNI152-2009fsl-mask.nii.gz'` or `'masks/k88_parcel_names.csv'`. Use `list_resources` to enumerate what's available. | *required*

**Returns:**

Type | Description
---- | -----------
<code>str</code> | Absolute path to the cached file on disk. The returned path drops     straight into anything that takes a NIfTI path — nilearn plotting     and masking helpers, `nibabel.load`, and `BrainData(path)`.

<details class="note" open markdown="1">
<summary>Note</summary>

Resolution is memoized per `relpath` for the session — repeated
calls (e.g. every default-mask `BrainData` construction) return the
cached path with no work. A file already in the HF cache is resolved
offline, so only a genuine cache miss touches the network.

</details>

(tasks-loading-list-resources)=
### `list_resources`

```python
list_resources(prefix: str | None = None) -> list[str]
```

List files available in the `nltools/niftis` HF dataset.

Companion to `fetch_resource` — surfaces what's downloadable
without forcing users to remember relpath strings or visit the HF
web UI.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`prefix` | <code>str</code> | Path prefix to filter by (e.g. `'masks/'`, `'default/'`, `'fmriprep/'`). Matches with `str.startswith`. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>list[str]</code> | Sorted relative paths usable with `fetch_resource`.

<details class="note" open markdown="1">
<summary>Note</summary>

Hits the HF API once per session (cached).

</details>

(tasks-loading-get-bg-image)=
### `get_bg_image`

```python
get_bg_image(affine: np.ndarray, img_type: str = 'brain', config: BrainSpaceConfig | None = None) -> str
```

Get a background image path matching a data resolution.

Uses `config` (or the current global brain space) and finds the
matching resolution from the affine. Used by plotting functions to pick
an appropriate background anatomical.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`affine` | <code>ndarray</code> | 4x4 affine matrix from a BrainData's masker. | *required*
`img_type` | <code>str</code> | `'brain'` for the brain-extracted image or `'plot'` for the full T1. Default `'brain'`. | <code>'brain'</code>
`config` | <code>[BrainSpaceConfig](#tasks-loading-brainspaceconfig)</code> | Explicit configuration; defaults to the current global brain space. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>str</code> | Path to the template image file.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If voxels are non-isotropic or `img_type` is invalid.

(tasks-loading-is-standard-space)=
### `is_standard_space`

```python
is_standard_space(affine: np.ndarray, *, config: BrainSpaceConfig | None = None) -> tuple[bool, str | None]
```

Check whether an affine is compatible with our MNI templates.

A "standard space" affine has isotropic voxels at one of the supported
template resolutions (the union of `SUPPORTED_RESOLUTIONS`). Plotting
surfaces (glass brain, flatmap, surface montage) and template-driven
background lookup all assume this — non-isotropic or off-grid data
would render in misleading positions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`affine` | <code>ndarray</code> | 4x4 affine matrix from a NIfTI image (typically `bd.mask.affine`). | *required*
`config` | <code>[BrainSpaceConfig](#tasks-loading-brainspaceconfig)</code> | Explicit configuration; defaults to the current global brain space (only the supported resolution set is consulted). | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>tuple[bool, str \| None]</code> | `(True, None)` if compatible; otherwise     `(False, reason)` with `reason` a one-line human-readable explanation     suitable for embedding in an error message.

(tasks-loading-detect-resolution)=
### `detect_resolution`

```python
detect_resolution(affine: np.ndarray) -> tuple[float, bool]
```

Detect voxel resolution (mm) and isotropy from a NIfTI affine.

Voxels are treated as isotropic when the per-axis sizes agree to within
three decimals. The reported resolution is that shared isotropic size, or
the mean of the per-axis sizes when non-isotropic.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`affine` | <code>ndarray</code> | 4x4 affine matrix from a NIfTI image. | *required*

**Returns:**

Type | Description
---- | -----------
<code>tuple[float, bool]</code> | `(resolution_mm, is_isotropic)`.
