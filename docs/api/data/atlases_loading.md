---
title: data.atlases.loading
label: data-atlases-loading
---

Lazy loading of atlas NIfTI + label CSV files from the HF dataset.

**Classes:**

Name | Description
---- | -----------
[`Atlas`](#data-atlases-loading-atlas) | A loaded atlas — image, labels, and metadata.

**Functions:**

Name | Description
---- | -----------
[`load_atlas`](#data-atlases-loading-load-atlas) | Lazy-load an atlas by registry name.



## Classes

(data-atlases-loading-atlas)=
### `Atlas`

```python
Atlas(name: str, image: nb.Nifti1Image, labels: pl.DataFrame, kind: AtlasKind, citation: str)
```

A loaded atlas — image, labels, and metadata.

Constructed by `load_atlas`; users normally don't instantiate
directly.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`name` | <code>str</code> | Registry key (e.g. ``"harvard_oxford"``).
`image` | <code>Nifti1Image</code> | NIfTI volume. 3D for deterministic atlases, 4D for probabilistic ones (last axis indexes regions).
`labels` | <code>DataFrame</code> | Two-column ``index, name`` table. For deterministic atlases ``index`` is the integer voxel value; for probabilistic atlases ``index`` is the region index along the 4th dim of ``image``.
`kind` | <code>AtlasKind</code> | ``"deterministic"`` or ``"probabilistic"``.
`citation` | <code>str</code> | Short citation for the original atlas.



## Functions

(data-atlases-loading-load-atlas)=
### `load_atlas`

```python
load_atlas(name: str) -> Atlas
```

Lazy-load an atlas by registry name.

First call fetches the NIfTI + label CSV from
``huggingface.co/datasets/nltools/niftis`` (cached locally
afterwards). Subsequent calls in the same process are memoized.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`name` | <code>str</code> | Atlas key from `list_atlases`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Atlas](#data-atlases-loading-atlas)</code> | An `Atlas` with image, labels, and metadata loaded.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If ``name`` isn't a registered atlas.
