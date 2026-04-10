## `templates`

Global MNI brain-space configuration for nltools.

This module manages the default MNI template used by ``BrainData`` and
related classes when no explicit mask is provided. Set it once (e.g., at
the top of a notebook) and all subsequent operations pick it up
automatically.

**Examples:**

Set the global brain space::

    import nltools
    nltools.set_brainspace(template="fmriprep", resolution=2)

Inspect the current configuration::

    cfg = nltools.get_brainspace()
    print(cfg.mask)

Scope a change to a block::

    with nltools.with_brainspace(resolution=1):
        brain = BrainData(...)

**Modules:**

Name | Description
---- | -----------
[`config`](#config) | Global brain-space configuration: frozen dataclass + set/get/with API.
[`matching`](#matching) | Affine-based template matching and background-image selection.
[`paths`](#paths) | Pure path-resolution helpers for MNI template files.
[`registry`](#registry) | Static registry of supported MNI templates.

**Classes:**

Name | Description
---- | -----------
[`BrainSpaceConfig`](#BrainSpaceConfig) | Immutable MNI template configuration.
[`TemplateMatch`](#TemplateMatch) | Result of matching a data affine to a template.

**Methods:**

Name | Description
---- | -----------
[`get_bg_image`](#get_bg_image) | Get a background image path matching a data resolution.
[`get_brainspace`](#get_brainspace) | Return the current global brain-space configuration.
[`match_resolution`](#match_resolution) | Find the best matching template for a given affine matrix.
[`reset_brainspace`](#reset_brainspace) | Reset the global brain-space configuration to defaults.
[`resolve_paths`](#resolve_paths) | Build mask/brain/plot paths for a template + resolution.
[`resolve_template_name`](#resolve_template_name) | Resolve a template name string to a file path.
[`set_brainspace`](#set_brainspace) | Set the global brain-space configuration.
[`with_brainspace`](#with_brainspace) | Temporarily change the global brain-space configuration.



### Classes

#### `BrainSpaceConfig`

```python
BrainSpaceConfig(template: TemplateName = 'default', resolution: Resolution = 2) -> None
```

Immutable MNI template configuration.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`template`](#template) | <code>[TemplateName](#nltools.templates.registry.TemplateName)</code> | Template variant (``'default'``, ``'nilearn'``, ``'fmriprep'``).
[`resolution`](#resolution) | <code>[Resolution](#nltools.templates.registry.Resolution)</code> | Resolution in mm (1, 2, or 3).

### Methods

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

Name | Type | Description
---- | ---- | -----------
`A` | <code>[TemplateMatch](#nltools.templates.matching.TemplateMatch)</code> | class:`TemplateMatch`.

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
[`BrainSpaceConfig`](#BrainSpaceConfig) | Immutable MNI template configuration.

**Methods:**

Name | Description
---- | -----------
[`get_brainspace`](#get_brainspace) | Return the current global brain-space configuration.
[`reset_brainspace`](#reset_brainspace) | Reset the global brain-space configuration to defaults.
[`set_brainspace`](#set_brainspace) | Set the global brain-space configuration.
[`with_brainspace`](#with_brainspace) | Temporarily change the global brain-space configuration.

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

#### `matching`

Affine-based template matching and background-image selection.

**Classes:**

Name | Description
---- | -----------
[`TemplateMatch`](#TemplateMatch) | Result of matching a data affine to a template.

**Methods:**

Name | Description
---- | -----------
[`get_bg_image`](#get_bg_image) | Get a background image path matching a data resolution.
[`match_resolution`](#match_resolution) | Find the best matching template for a given affine matrix.

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

Name | Type | Description
---- | ---- | -----------
`A` | <code>[TemplateMatch](#nltools.templates.matching.TemplateMatch)</code> | class:`TemplateMatch`.

#### `paths`

Pure path-resolution helpers for MNI template files.

**Methods:**

Name | Description
---- | -----------
[`resolve_paths`](#resolve_paths) | Build mask/brain/plot paths for a template + resolution.
[`resolve_template_name`](#resolve_template_name) | Resolve a template name string to a file path.

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
[`Resolution`](#Resolution) |  | 
[`SUPPORTED_RESOLUTIONS`](#SUPPORTED_RESOLUTIONS) | <code>[dict](#dict)[[str](#str), [list](#list)[[int](#int)]]</code> | 
[`TEMPLATE_PRIORITY`](#TEMPLATE_PRIORITY) | <code>[list](#list)[[str](#str)]</code> | 
[`TemplateName`](#TemplateName) |  | 
[`VERSION_MAP`](#VERSION_MAP) | <code>[dict](#dict)[[str](#str), [str](#str)]</code> | 
[`VERSION_TO_TEMPLATE`](#VERSION_TO_TEMPLATE) | <code>[dict](#dict)[[str](#str), [str](#str)]</code> | 



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

