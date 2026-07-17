(data-atlases-registry-registry)=
## `registry`

Static registry of atlases hosted at ``nltools/niftis/atlases``.

Each entry describes an atlas's kind (deterministic vs probabilistic) and
the citation users should cite when they use it. The actual NIfTI + label
files are fetched lazily by `load_atlas` via
`fetch_resource`.

Atlases were sourced from atlasreader (BSD-3-Clause) and are subject to
their original upstream licenses — see ``LICENSES.md`` in the HF dataset.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`ATLASES` | <code>[dict](#dict)[[str](#str), [AtlasMetadata](#nltools.data.atlases.registry.AtlasMetadata)]</code> | 
`AtlasKind` |  | 
`DEFAULT_ATLASES` | <code>[tuple](#tuple)[[str](#str), ...]</code> | 



**Classes:**

Name | Description
---- | -----------
[`AtlasMetadata`](#data-atlases-registry-atlasmetadata) | Static description of a registered atlas.

**Methods:**

Name | Description
---- | -----------
[`list_atlases`](#data-atlases-registry-list-atlases) | Return the sorted list of registered atlas names.

### Classes

(data-atlases-registry-atlasmetadata)=
#### `AtlasMetadata`

```python
AtlasMetadata(kind: AtlasKind, citation: str) -> None
```

Static description of a registered atlas.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`kind` | <code>[AtlasKind](#nltools.data.atlases.registry.AtlasKind)</code> | ``"deterministic"`` (3D integer-labeled) or ``"probabilistic"`` (4D, last axis indexes regions).
`citation` | <code>[str](#str)</code> | Short citation string for the original atlas.

### Methods

(data-atlases-registry-list-atlases)=
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

