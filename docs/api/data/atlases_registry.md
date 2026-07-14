## `registry`

Static registry of atlases hosted at ``nltools/niftis/atlases``.

Each entry describes an atlas's kind (deterministic vs probabilistic) and
the citation users should cite when they use it. The actual NIfTI + label
files are fetched lazily by `load_atlas` via
`fetch_resource`.

Atlases were sourced from atlasreader (BSD-3-Clause) and are subject to
their original upstream licenses — see ``LICENSES.md`` in the HF dataset.

**Classes:**

Name | Description
---- | -----------
[`AtlasMetadata`](#AtlasMetadata) | Static description of a registered atlas.

**Methods:**

Name | Description
---- | -----------
[`list_atlases`](#list_atlases) | Return the sorted list of registered atlas names.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`ATLASES`](#ATLASES) | <code>[dict](#dict)[[str](#str), [AtlasMetadata](#nltools.data.atlases.registry.AtlasMetadata)]</code> | 
[`AtlasKind`](#AtlasKind) |  | 
[`DEFAULT_ATLASES`](#DEFAULT_ATLASES) | <code>[tuple](#tuple)[[str](#str), ...]</code> | 

### Methods

#### `list_atlases`

```python
list_atlases() -> list[str]
```

Return the sorted list of registered atlas names.

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[str](#str)]</code> | Sorted list of atlas names usable with
<code>[list](#list)[[str](#str)]</code> | `load_atlas`.

