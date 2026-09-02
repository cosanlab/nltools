---
title: templates
label: page-templates
---

Global MNI brain-space configuration for nltools.

This module manages the default MNI template used by `BrainData` and
related classes when no explicit mask is provided. Set it once (e.g., at
the top of a notebook) and all subsequent operations pick it up
automatically.

**Classes:**

Name | Description
---- | -----------
[`BrainSpaceConfig`](#templates-brainspaceconfig) | Immutable MNI template configuration.
[`TemplateMatch`](#templates-templatematch) | Result of matching a data affine to a template.

**Functions:**

Name | Description
---- | -----------
[`detect_resolution`](#templates-detect-resolution) | Detect voxel resolution (mm) and isotropy from a NIfTI affine.
[`fetch_resource`](#templates-fetch-resource) | Return a local path to a file from the `nltools/niftis` HF dataset.
[`get_bg_image`](#templates-get-bg-image) | Get a background image path matching a data resolution.
[`get_brainspace`](#templates-get-brainspace) | Return the current global brain-space configuration.
[`is_standard_space`](#templates-is-standard-space) | Check whether an affine is compatible with our MNI templates.
[`list_resources`](#templates-list-resources) | List files available in the `nltools/niftis` HF dataset.
[`match_resolution`](#templates-match-resolution) | Find the best matching template for a given affine matrix.
[`reset_brainspace`](#templates-reset-brainspace) | Reset the global brain-space configuration to defaults.
[`resolve_paths`](#templates-resolve-paths) | Build mask/brain/plot paths for a template + resolution.
[`resolve_template_name`](#templates-resolve-template-name) | Resolve a template name string to a file path.
[`set_brainspace`](#templates-set-brainspace) | Set the global brain-space configuration.
[`with_brainspace`](#templates-with-brainspace) | Temporarily change the global brain-space configuration.



**Examples:**

Set the global brain space:

```python
import nltools

nltools.set_brainspace(template="fmriprep", resolution=2)
```

Inspect the current configuration:

```python
cfg = nltools.get_brainspace()
print(cfg.mask)
```

Scope a change to a block:

```python
with nltools.with_brainspace(resolution=1):
    brain = BrainData(...)
```

## Classes

(templates-brainspaceconfig)=
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

(templates-templatematch)=
### `TemplateMatch`

```python
TemplateMatch(template: str, resolution: int, mask_path: str, brain_path: str, plot_path: str, match_distance: float)
```

Result of matching a data affine to a template.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`template` | <code>str</code> | Best-matching template name.
`resolution` | <code>int</code> | Best-matching resolution in mm.
`mask_path` | <code>str</code> | Path to the matched mask file.
`brain_path` | <code>str</code> | Path to the matched brain file.
`plot_path` | <code>str</code> | Path to the matched T1/plot file.
`match_distance` | <code>float</code> | Absolute difference in mm between detected data resolution and the selected template resolution (0 for exact).



## Functions

(templates-detect-resolution)=
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

(templates-fetch-resource)=
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

(templates-get-bg-image)=
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

(templates-get-brainspace)=
### `get_brainspace`

```python
get_brainspace() -> BrainSpaceConfig
```

Return the current global brain-space configuration.

**Returns:**

Type | Description
---- | -----------
<code>[BrainSpaceConfig](#tasks-loading-brainspaceconfig)</code> | The active configuration.

(templates-is-standard-space)=
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

(templates-list-resources)=
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

(templates-match-resolution)=
### `match_resolution`

```python
match_resolution(affine: np.ndarray, prefer_exact: bool = True, warn_resample: bool = True) -> TemplateMatch
```

Find the best matching template for a given affine matrix.

Searches available templates by priority and returns the one whose
resolution most closely matches the data's voxel size.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`affine` | <code>ndarray</code> | 4x4 affine matrix from a NIfTI image. | *required*
`prefer_exact` | <code>bool</code> | If True, prefer an exact resolution match. Default True. | <code>True</code>
`warn_resample` | <code>bool</code> | If True, emit a `ResamplingWarning` when the data resolution has no exact template and the closest one is used. Default True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[TemplateMatch](#templates-templatematch)</code> | The selected template, its resolution, and file paths.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If detected resolution is outside a reasonable range.

(templates-reset-brainspace)=
### `reset_brainspace`

```python
reset_brainspace() -> BrainSpaceConfig
```

Reset the global brain-space configuration to defaults.

**Returns:**

Type | Description
---- | -----------
<code>[BrainSpaceConfig](#tasks-loading-brainspaceconfig)</code> | The default configuration (`'default'` template, 2 mm).

(templates-resolve-paths)=
### `resolve_paths`

```python
resolve_paths(template: str, resolution: int) -> dict[str, str]
```

Build mask/brain/plot paths for a template + resolution.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`template` | <code>str</code> | Template name (`'default'`, `'nilearn'`, `'fmriprep'`). | *required*
`resolution` | <code>int</code> | Resolution in mm. | *required*

**Returns:**

Type | Description
---- | -----------
<code>dict[str, str]</code> | Local file paths keyed `'mask'`, `'brain'`, `'plot'`.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If template or resolution is invalid.

(templates-resolve-template-name)=
### `resolve_template_name`

```python
resolve_template_name(template_name: str, file_type: str = 'mask') -> str
```

Resolve a template name string to a file path.

Supports names of the form `'{res}mm-MNI152-2009{version}'`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`template_name` | <code>str</code> | e.g. `'2mm-MNI152-2009c'`, `'3mm-MNI152-2009a'`. | *required*
`file_type` | <code>str</code> | `'mask'`, `'brain'`, or `'T1'`. Default `'mask'`. | <code>'mask'</code>

**Returns:**

Type | Description
---- | -----------
<code>str</code> | Absolute path to the requested template file.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If `file_type` or the template name format is invalid.

(templates-set-brainspace)=
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

(templates-with-brainspace)=
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
