(data-atlases-loading-loading)=
## `loading`

Lazy loading of atlas NIfTI + label CSV files from the HF dataset.

**Classes:**

Name | Description
---- | -----------
`Atlas` | A loaded atlas — image, labels, and metadata.

**Methods:**

Name | Description
---- | -----------
[`load_atlas`](#data-atlases-loading-load-atlas) | Lazy-load an atlas by registry name.

### Methods

(data-atlases-loading-load-atlas)=
#### `load_atlas`

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
`name` | <code>[str](#str)</code> | Atlas key from `list_atlases`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Atlas](#nltools.data.atlases.loading.Atlas)</code> | An `Atlas` with image, labels, and metadata loaded.

