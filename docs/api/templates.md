## `templates`

Global MNI brain-space configuration for nltools.

This module manages the default MNI template used by ``BrainData`` and
related classes when no explicit mask is provided. Set it once (e.g., at
the top of a notebook) and all subsequent operations pick it up
automatically.

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

**Modules:**

Name | Description
---- | -----------
[`config`](#config) | Global brain-space configuration: frozen dataclass + set/get/with API.
[`fetch`](#fetch) | Lazy fetcher for files hosted in the ``nltools/niftis`` HF dataset.
[`matching`](#matching) | Affine-based template matching and background-image selection.
[`paths`](#paths) | Pure path-resolution helpers for MNI template files.
[`registry`](#registry) | Static registry of supported MNI templates.

**Classes:**

Name | Description
---- | -----------
[`BrainSpaceConfig`](#brainspaceconfig) | Immutable MNI template configuration.
`TemplateMatch` | Result of matching a data affine to a template.

**Methods:**

Name | Description
---- | -----------
[`fetch_resource`](#fetch-resource) | Return a local path to a file from the ``nltools/niftis`` HF dataset.
[`get_bg_image`](#get-bg-image) | Get a background image path matching a data resolution.
[`get_brainspace`](#get-brainspace) | Return the current global brain-space configuration.
[`is_standard_space`](#is-standard-space) | Check whether an affine is compatible with our MNI templates.
[`list_resources`](#list-resources) | List files available in the ``nltools/niftis`` HF dataset.
[`match_resolution`](#match-resolution) | Find the best matching template for a given affine matrix.
[`reset_brainspace`](#reset-brainspace) | Reset the global brain-space configuration to defaults.
[`resolve_paths`](#resolve-paths) | Build mask/brain/plot paths for a template + resolution.
[`resolve_template_name`](#resolve-template-name) | Resolve a template name string to a file path.
[`seed_resources`](#seed-resources) | Pre-download dataset files in Pyodide so sync fetches resolve from cache.
[`set_brainspace`](#set-brainspace) | Set the global brain-space configuration.
[`with_brainspace`](#with-brainspace) | Temporarily change the global brain-space configuration.



### Classes

#### `BrainSpaceConfig`

```python
BrainSpaceConfig(template: TemplateName = 'default', resolution: Resolution = 2) -> None
```

Immutable MNI template configuration.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`template` | <code>[TemplateName](#nltools.templates.registry.TemplateName)</code> | Template variant (``'default'``, ``'nilearn'``, ``'fmriprep'``).
[`resolution`](#resolution) | <code>[Resolution](#nltools.templates.registry.Resolution)</code> | Resolution in mm (1, 2, or 3).

### Methods

#### `fetch_resource`

```python
fetch_resource(relpath: str) -> str
```

Return a local path to a file from the ``nltools/niftis`` HF dataset.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`relpath` | <code>[str](#str)</code> | Path within the dataset repo, e.g. ``'default/2mm-MNI152-2009fsl-mask.nii.gz'`` or ``'masks/k88_parcel_names.csv'``. Use `list_resources` to enumerate what's available. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | Absolute path to the cached file on disk. The returned path drops
<code>[str](#str)</code> | straight into anything that takes a NIfTI path — nilearn plotting
<code>[str](#str)</code> | and masking helpers, ``nibabel.load``, and ``BrainData(path)``.

#### `get_bg_image`

```python
get_bg_image(affine: np.ndarray, img_type: str = 'brain', config: BrainSpaceConfig | None = None) -> str
```

Get a background image path matching a data resolution.

Uses ``config`` (or the current global brain space) and finds the
matching resolution from the affine. Used by plotting functions to pick
an appropriate background anatomical.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`affine` | <code>[ndarray](#numpy.ndarray)</code> | 4x4 affine matrix from a BrainData's masker. | *required*
`img_type` | <code>[str](#str)</code> | ``'brain'`` for brain-extracted image or ``'plot'`` for full T1. | <code>'brain'</code>
`config` | <code>[BrainSpaceConfig](#nltools.templates.config.BrainSpaceConfig) \| None</code> | Optional explicit config; defaults to current global. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | Path to the template image file.

#### `get_brainspace`

```python
get_brainspace() -> BrainSpaceConfig
```

Return the current global brain-space configuration.

#### `is_standard_space`

```python
is_standard_space(affine: np.ndarray, *, config: BrainSpaceConfig | None = None) -> tuple[bool, str | None]
```

Check whether an affine is compatible with our MNI templates.

A "standard space" affine has isotropic voxels at one of the supported
template resolutions (the union of ``SUPPORTED_RESOLUTIONS``). Plotting
surfaces (glass brain, flatmap, surface montage) and template-driven
background lookup all assume this — non-isotropic or off-grid data
would render in misleading positions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`affine` | <code>[ndarray](#numpy.ndarray)</code> | 4x4 affine matrix from a NIfTI image (typically ``bd.mask.affine``). | *required*
`config` | <code>[BrainSpaceConfig](#nltools.templates.config.BrainSpaceConfig) \| None</code> | Optional explicit ``BrainSpaceConfig``; defaults to the current global brain space (only the supported resolution set is consulted). | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | ``(True, None)`` if compatible; otherwise ``(False, reason)`` with
<code>[str](#str) \| None</code> | ``reason`` a one-line human-readable explanation suitable for
<code>[tuple](#tuple)[[bool](#bool), [str](#str) \| None]</code> | embedding in an error message.

#### `list_resources`

```python
list_resources(prefix: str | None = None) -> list[str]
```

List files available in the ``nltools/niftis`` HF dataset.

Companion to `fetch_resource` — surfaces what's downloadable
without forcing users to remember relpath strings or visit the HF
web UI.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`prefix` | <code>[str](#str) \| None</code> | Optional path prefix to filter by (e.g., ``'masks/'``, ``'default/'``, ``'fmriprep/'``). Matches with ``str.startswith``. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[str](#str)]</code> | Sorted list of relative paths usable with `fetch_resource`.

<details class="notes" open markdown="1">
<summary>Notes</summary>

Hits the HF API once per session (cached). Not available in
Pyodide — browser-deployed code should know its paths in advance
and pre-seed via `seed_resources`.

</details>

#### `match_resolution`

```python
match_resolution(affine: np.ndarray, prefer_exact: bool = True, warn_resample: bool = True) -> TemplateMatch
```

Find the best matching template for a given affine matrix.

Searches available templates by priority and returns the one whose
resolution most closely matches the data's voxel size.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`affine` | <code>[ndarray](#numpy.ndarray)</code> | 4x4 affine matrix from a NIfTI image. | *required*
`prefer_exact` | <code>[bool](#bool)</code> | If True, prefer an exact resolution match. | <code>True</code>
`warn_resample` | <code>[bool](#bool)</code> | If True, emit a warning when data resolution doesn't exactly match the selected template. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[TemplateMatch](#nltools.templates.matching.TemplateMatch)</code> | A `TemplateMatch`.

#### `reset_brainspace`

```python
reset_brainspace() -> BrainSpaceConfig
```

Reset the global brain-space configuration to defaults.

#### `resolve_paths`

```python
resolve_paths(template: str, resolution: int) -> dict[str, str]
```

Build mask/brain/plot paths for a template + resolution.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`template` | <code>[str](#str)</code> | Template name (``'default'``, ``'nilearn'``, ``'fmriprep'``). | *required*
`resolution` | <code>[int](#int)</code> | Resolution in mm. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [str](#str)]</code> | Dict with keys ``'mask'``, ``'brain'``, ``'plot'``.

#### `resolve_template_name`

```python
resolve_template_name(template_name: str, file_type: str = 'mask') -> str
```

Resolve a template name string to a file path.

Supports names of the form ``'{res}mm-MNI152-2009{version}'``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`template_name` | <code>[str](#str)</code> | e.g. ``'2mm-MNI152-2009c'``, ``'3mm-MNI152-2009a'``. | *required*
`file_type` | <code>[str](#str)</code> | ``'mask'``, ``'brain'``, or ``'T1'``. | <code>'mask'</code>

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | Absolute path to the requested template file.

#### `seed_resources`

```python
seed_resources(relpaths: list[str]) -> None
```

Pre-download dataset files in Pyodide so sync fetches resolve from cache.

No-op outside Pyodide — `fetch_resource` does its own lazy download
via ``huggingface_hub`` there. In Pyodide this must be called (and
awaited) before any code path that calls `fetch_resource`,
`resolve_paths`, or `resolve_template_name` synchronously.

The cache is backed by IndexedDB, so files persist across page reloads.
The first call per session mounts IDBFS and pulls any prior data;
subsequent calls only download files not already cached.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`relpaths` | <code>[list](#list)[[str](#str)]</code> | Paths within the dataset repo to pre-fetch. | *required*

#### `set_brainspace`

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
`template` | <code>[TemplateName](#nltools.templates.registry.TemplateName) \| None</code> | Template name to set. If ``None``, keeps current. | <code>None</code>
`resolution` | <code>[Resolution](#nltools.templates.registry.Resolution) \| None</code> | Resolution to set. If ``None``, keeps current. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainSpaceConfig](#nltools.templates.config.BrainSpaceConfig)</code> | The new (or unchanged) current ``BrainSpaceConfig``.

#### `with_brainspace`

```python
with_brainspace(template: TemplateName | None = None, resolution: Resolution | None = None) -> Iterator[BrainSpaceConfig]
```

Temporarily change the global brain-space configuration.

Restores the previous configuration on exit, even if an exception is
raised inside the block.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`template` | <code>[TemplateName](#nltools.templates.registry.TemplateName) \| None</code> | Template name for the duration of the block. | <code>None</code>
`resolution` | <code>[Resolution](#nltools.templates.registry.Resolution) \| None</code> | Resolution for the duration of the block. | <code>None</code>

**Yields:**

Type | Description
---- | -----------
<code>[BrainSpaceConfig](#nltools.templates.config.BrainSpaceConfig)</code> | The ``BrainSpaceConfig`` active inside the block.



### Modules

#### `config`

Global brain-space configuration: frozen dataclass + set/get/with API.

**Classes:**

Name | Description
---- | -----------
[`BrainSpaceConfig`](#brainspaceconfig) | Immutable MNI template configuration.

**Methods:**

Name | Description
---- | -----------
[`get_brainspace`](#get-brainspace) | Return the current global brain-space configuration.
[`reset_brainspace`](#reset-brainspace) | Reset the global brain-space configuration to defaults.
[`set_brainspace`](#set-brainspace) | Set the global brain-space configuration.
[`with_brainspace`](#with-brainspace) | Temporarily change the global brain-space configuration.

##### Methods

###### `get_brainspace`

```python
get_brainspace() -> BrainSpaceConfig
```

Return the current global brain-space configuration.

###### `reset_brainspace`

```python
reset_brainspace() -> BrainSpaceConfig
```

Reset the global brain-space configuration to defaults.

###### `set_brainspace`

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
`template` | <code>[TemplateName](#nltools.templates.registry.TemplateName) \| None</code> | Template name to set. If ``None``, keeps current. | <code>None</code>
`resolution` | <code>[Resolution](#nltools.templates.registry.Resolution) \| None</code> | Resolution to set. If ``None``, keeps current. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainSpaceConfig](#nltools.templates.config.BrainSpaceConfig)</code> | The new (or unchanged) current ``BrainSpaceConfig``.

###### `with_brainspace`

```python
with_brainspace(template: TemplateName | None = None, resolution: Resolution | None = None) -> Iterator[BrainSpaceConfig]
```

Temporarily change the global brain-space configuration.

Restores the previous configuration on exit, even if an exception is
raised inside the block.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`template` | <code>[TemplateName](#nltools.templates.registry.TemplateName) \| None</code> | Template name for the duration of the block. | <code>None</code>
`resolution` | <code>[Resolution](#nltools.templates.registry.Resolution) \| None</code> | Resolution for the duration of the block. | <code>None</code>

**Yields:**

Type | Description
---- | -----------
<code>[BrainSpaceConfig](#nltools.templates.config.BrainSpaceConfig)</code> | The ``BrainSpaceConfig`` active inside the block.

#### `fetch`

Lazy fetcher for files hosted in the ``nltools/niftis`` HF dataset.

Covers MNI templates, parcellation label maps, the parcel-names CSV, and
any other resources living under huggingface.co/datasets/nltools/niftis.
First call for a given file downloads it into the local HF cache
(``~/.cache/huggingface/hub`` by default); subsequent calls return the
cached path without touching the network.

In Pyodide the synchronous HF cache is unavailable (``huggingface_hub``
isn't installed and sync HTTP from Python is not viable). Consumers must
``await seed_resources([...])`` once at app boot to pre-download the
files they need; subsequent sync ``fetch_resource()`` calls then hit
the IDBFS-backed cache populated by the seed. The cache persists across
page reloads via IndexedDB, so seeding only does network work once per
browser per dataset revision.

**Methods:**

Name | Description
---- | -----------
[`fetch_resource`](#fetch-resource) | Return a local path to a file from the ``nltools/niftis`` HF dataset.
[`list_resources`](#list-resources) | List files available in the ``nltools/niftis`` HF dataset.
[`seed_resources`](#seed-resources) | Pre-download dataset files in Pyodide so sync fetches resolve from cache.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`REPO_ID` |  | 
`REVISION` |  | 

##### Methods

###### `fetch_resource`

```python
fetch_resource(relpath: str) -> str
```

Return a local path to a file from the ``nltools/niftis`` HF dataset.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`relpath` | <code>[str](#str)</code> | Path within the dataset repo, e.g. ``'default/2mm-MNI152-2009fsl-mask.nii.gz'`` or ``'masks/k88_parcel_names.csv'``. Use `list_resources` to enumerate what's available. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | Absolute path to the cached file on disk. The returned path drops
<code>[str](#str)</code> | straight into anything that takes a NIfTI path — nilearn plotting
<code>[str](#str)</code> | and masking helpers, ``nibabel.load``, and ``BrainData(path)``.

###### `list_resources`

```python
list_resources(prefix: str | None = None) -> list[str]
```

List files available in the ``nltools/niftis`` HF dataset.

Companion to `fetch_resource` — surfaces what's downloadable
without forcing users to remember relpath strings or visit the HF
web UI.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`prefix` | <code>[str](#str) \| None</code> | Optional path prefix to filter by (e.g., ``'masks/'``, ``'default/'``, ``'fmriprep/'``). Matches with ``str.startswith``. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[str](#str)]</code> | Sorted list of relative paths usable with `fetch_resource`.

<details class="notes" open markdown="1">
<summary>Notes</summary>

Hits the HF API once per session (cached). Not available in
Pyodide — browser-deployed code should know its paths in advance
and pre-seed via `seed_resources`.

</details>

###### `seed_resources`

```python
seed_resources(relpaths: list[str]) -> None
```

Pre-download dataset files in Pyodide so sync fetches resolve from cache.

No-op outside Pyodide — `fetch_resource` does its own lazy download
via ``huggingface_hub`` there. In Pyodide this must be called (and
awaited) before any code path that calls `fetch_resource`,
`resolve_paths`, or `resolve_template_name` synchronously.

The cache is backed by IndexedDB, so files persist across page reloads.
The first call per session mounts IDBFS and pulls any prior data;
subsequent calls only download files not already cached.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`relpaths` | <code>[list](#list)[[str](#str)]</code> | Paths within the dataset repo to pre-fetch. | *required*

#### `matching`

Affine-based template matching and background-image selection.

**Classes:**

Name | Description
---- | -----------
`TemplateMatch` | Result of matching a data affine to a template.

**Methods:**

Name | Description
---- | -----------
[`get_bg_image`](#get-bg-image) | Get a background image path matching a data resolution.
[`is_standard_space`](#is-standard-space) | Check whether an affine is compatible with our MNI templates.
[`match_resolution`](#match-resolution) | Find the best matching template for a given affine matrix.

##### Methods

###### `get_bg_image`

```python
get_bg_image(affine: np.ndarray, img_type: str = 'brain', config: BrainSpaceConfig | None = None) -> str
```

Get a background image path matching a data resolution.

Uses ``config`` (or the current global brain space) and finds the
matching resolution from the affine. Used by plotting functions to pick
an appropriate background anatomical.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`affine` | <code>[ndarray](#numpy.ndarray)</code> | 4x4 affine matrix from a BrainData's masker. | *required*
`img_type` | <code>[str](#str)</code> | ``'brain'`` for brain-extracted image or ``'plot'`` for full T1. | <code>'brain'</code>
`config` | <code>[BrainSpaceConfig](#nltools.templates.config.BrainSpaceConfig) \| None</code> | Optional explicit config; defaults to current global. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | Path to the template image file.

###### `is_standard_space`

```python
is_standard_space(affine: np.ndarray, *, config: BrainSpaceConfig | None = None) -> tuple[bool, str | None]
```

Check whether an affine is compatible with our MNI templates.

A "standard space" affine has isotropic voxels at one of the supported
template resolutions (the union of ``SUPPORTED_RESOLUTIONS``). Plotting
surfaces (glass brain, flatmap, surface montage) and template-driven
background lookup all assume this — non-isotropic or off-grid data
would render in misleading positions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`affine` | <code>[ndarray](#numpy.ndarray)</code> | 4x4 affine matrix from a NIfTI image (typically ``bd.mask.affine``). | *required*
`config` | <code>[BrainSpaceConfig](#nltools.templates.config.BrainSpaceConfig) \| None</code> | Optional explicit ``BrainSpaceConfig``; defaults to the current global brain space (only the supported resolution set is consulted). | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | ``(True, None)`` if compatible; otherwise ``(False, reason)`` with
<code>[str](#str) \| None</code> | ``reason`` a one-line human-readable explanation suitable for
<code>[tuple](#tuple)[[bool](#bool), [str](#str) \| None]</code> | embedding in an error message.

###### `match_resolution`

```python
match_resolution(affine: np.ndarray, prefer_exact: bool = True, warn_resample: bool = True) -> TemplateMatch
```

Find the best matching template for a given affine matrix.

Searches available templates by priority and returns the one whose
resolution most closely matches the data's voxel size.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`affine` | <code>[ndarray](#numpy.ndarray)</code> | 4x4 affine matrix from a NIfTI image. | *required*
`prefer_exact` | <code>[bool](#bool)</code> | If True, prefer an exact resolution match. | <code>True</code>
`warn_resample` | <code>[bool](#bool)</code> | If True, emit a warning when data resolution doesn't exactly match the selected template. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[TemplateMatch](#nltools.templates.matching.TemplateMatch)</code> | A `TemplateMatch`.

#### `paths`

Pure path-resolution helpers for MNI template files.

Resolves logical (template, resolution, file_type) tuples to local paths.
Files are fetched on first use from the ``nltools/niftis`` HF dataset; see
`nltools.templates.fetch`.

**Methods:**

Name | Description
---- | -----------
[`resolve_paths`](#resolve-paths) | Build mask/brain/plot paths for a template + resolution.
[`resolve_template_name`](#resolve-template-name) | Resolve a template name string to a file path.

##### Methods

###### `resolve_paths`

```python
resolve_paths(template: str, resolution: int) -> dict[str, str]
```

Build mask/brain/plot paths for a template + resolution.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`template` | <code>[str](#str)</code> | Template name (``'default'``, ``'nilearn'``, ``'fmriprep'``). | *required*
`resolution` | <code>[int](#int)</code> | Resolution in mm. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [str](#str)]</code> | Dict with keys ``'mask'``, ``'brain'``, ``'plot'``.

###### `resolve_template_name`

```python
resolve_template_name(template_name: str, file_type: str = 'mask') -> str
```

Resolve a template name string to a file path.

Supports names of the form ``'{res}mm-MNI152-2009{version}'``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`template_name` | <code>[str](#str)</code> | e.g. ``'2mm-MNI152-2009c'``, ``'3mm-MNI152-2009a'``. | *required*
`file_type` | <code>[str](#str)</code> | ``'mask'``, ``'brain'``, or ``'T1'``. | <code>'mask'</code>

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | Absolute path to the requested template file.

#### `registry`

Static registry of supported MNI templates.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`Resolution`](#resolution) |  | 
[`SUPPORTED_RESOLUTIONS`](#supported-resolutions) | <code>[dict](#dict)[[str](#str), [list](#list)[[int](#int)]]</code> | 
[`TEMPLATE_PRIORITY`](#template-priority) | <code>[list](#list)[[str](#str)]</code> | 
[`TemplateName`](#templatename) |  | 
[`VERSION_MAP`](#version-map) | <code>[dict](#dict)[[str](#str), [str](#str)]</code> | 
[`VERSION_TO_TEMPLATE`](#version-to-template) | <code>[dict](#dict)[[str](#str), [str](#str)]</code> | 



##### Attributes

###### `Resolution`

```python
Resolution = Literal[1, 2, 3]
```

###### `SUPPORTED_RESOLUTIONS`

```python
SUPPORTED_RESOLUTIONS: dict[str, list[int]] = {'default': [2, 3], 'nilearn': [1, 2, 3], 'fmriprep': [1, 2]}
```

###### `TEMPLATE_PRIORITY`

```python
TEMPLATE_PRIORITY: list[str] = ['default', 'nilearn', 'fmriprep']
```

###### `TemplateName`

```python
TemplateName = Literal['default', 'nilearn', 'fmriprep']
```

###### `VERSION_MAP`

```python
VERSION_MAP: dict[str, str] = {'default': 'fsl', 'nilearn': 'a', 'fmriprep': 'c'}
```

###### `VERSION_TO_TEMPLATE`

```python
VERSION_TO_TEMPLATE: dict[str, str] = {v: k for k, v in (VERSION_MAP.items())}
```

