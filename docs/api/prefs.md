## `nltools.prefs`

**Functions:**

Name | Description
---- | -----------
[`resolve_template_name`](#nltools.prefs.resolve_template_name) | Resolve a template name string to a file path.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`MNI_Template`](#nltools.prefs.MNI_Template) |  | 



### Attributes#### `nltools.prefs.MNI_Template`

```python
MNI_Template = MNI_Template_Factory()
```



### Classes

### Functions#### `nltools.prefs.resolve_template_name`

```python
resolve_template_name(template_name: str, file_type: str = 'mask') -> str
```

Resolve a template name string to a file path.

Supports template names in the format ``'{res}mm-MNI152-2009{version}'``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`template_name` | <code>[str](#str)</code> | e.g. ``'2mm-MNI152-2009c'``, ``'3mm-MNI152-2009a'``. | *required*
`file_type` | <code>[str](#str)</code> | ``'mask'``, ``'brain'``, or ``'T1'``. Default: ``'mask'``. | <code>'mask'</code>

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | Full path to the template file.

**Examples:**

```pycon
>>> resolve_template_name('2mm-MNI152-2009c')
'/path/to/nltools/resources/niftis/fmriprep/2mm-MNI152-2009c-mask.nii.gz'
```

