## `nltools.data.braindata.cache`

Disk-based caching infrastructure for expensive computations.

This module provides a general-purpose caching system for nltools, designed to
be reused across various computationally expensive operations like searchlight
neighborhoods, ISC, and SRM.

<details class="example" open markdown="1">
<summary>Example</summary>

>>> from nltools.data.braindata.cache import CacheManager, hash_mask
>>> import nibabel as nib
>>>
>>> # Hash a mask for cache key generation
>>> mask = nib.load("mask.nii.gz")
>>> mask_hash = hash_mask(mask)
>>>
>>> # Use cache manager for searchlight neighborhoods
>>> cache = CacheManager("searchlight")
>>> if not cache.exists(f"{mask_hash}_10mm"):
...     # Compute expensive operation
...     result = compute_something()
...     cache.save(f"{mask_hash}_10mm", data=result)
>>> else:
...     result = cache.load(f"{mask_hash}_10mm")["data"]

</details>

**Classes:**

Name | Description
---- | -----------
[`CacheManager`](#nltools.data.braindata.cache.CacheManager) | Manages disk-based caching for expensive computations.

**Functions:**

Name | Description
---- | -----------
[`clear_cache`](#nltools.data.braindata.cache.clear_cache) | Clear the nltools cache.
[`get_cache_dir`](#nltools.data.braindata.cache.get_cache_dir) | Get the nltools cache directory.
[`hash_mask`](#nltools.data.braindata.cache.hash_mask) | Compute a stable hash for a NIfTI mask image.



### Classes#### `nltools.data.braindata.cache.CacheManager`

```python
CacheManager(category: str = 'general')
```

Manages disk-based caching for expensive computations.

CacheManager provides a simple key-value interface for caching numpy arrays
to disk. It organizes cached files by category (e.g., "searchlight", "isc")
in separate subdirectories.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`category` | <code>[str](#str)</code> | Category name for organizing cached files (e.g., "searchlight") | <code>'general'</code>

<details class="example" open markdown="1">
<summary>Example</summary>

>>> cache = CacheManager("searchlight")
>>>
>>> # Check if something is cached
>>> if cache.exists("mykey"):
...     data = cache.load("mykey")
... else:
...     result = expensive_computation()
...     cache.save("mykey", adjacency=result, metadata=metadata)
...     data = {"adjacency": result, "metadata": metadata}

</details>

**Functions:**

Name | Description
---- | -----------
[`clear`](#nltools.data.braindata.cache.CacheManager.clear) | Clear all cached files in this category.
[`delete`](#nltools.data.braindata.cache.CacheManager.delete) | Delete a cached file.
[`exists`](#nltools.data.braindata.cache.CacheManager.exists) | Check if a cache key exists.
[`get_path`](#nltools.data.braindata.cache.CacheManager.get_path) | Get the file path for a cache key.
[`list_keys`](#nltools.data.braindata.cache.CacheManager.list_keys) | List all cached keys in this category.
[`load`](#nltools.data.braindata.cache.CacheManager.load) | Load cached data.
[`save`](#nltools.data.braindata.cache.CacheManager.save) | Save arrays to cache.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`cache_dir`](#nltools.data.braindata.cache.CacheManager.cache_dir) |  | 
[`category`](#nltools.data.braindata.cache.CacheManager.category) |  | 



##### Attributes###### `nltools.data.braindata.cache.CacheManager.cache_dir`

```python
cache_dir = get_cache_dir() / category
```

###### `nltools.data.braindata.cache.CacheManager.category`

```python
category = category
```



##### Functions###### `nltools.data.braindata.cache.CacheManager.clear`

```python
clear() -> int
```

Clear all cached files in this category.

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of files deleted

###### `nltools.data.braindata.cache.CacheManager.delete`

```python
delete(key: str, ext: str = '.npz') -> bool
```

Delete a cached file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>[str](#str)</code> | Cache key | *required*
`ext` | <code>[str](#str)</code> | File extension | <code>'.npz'</code>

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if file was deleted, False if it didn't exist

###### `nltools.data.braindata.cache.CacheManager.exists`

```python
exists(key: str, ext: str = '.npz') -> bool
```

Check if a cache key exists.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>[str](#str)</code> | Cache key | *required*
`ext` | <code>[str](#str)</code> | File extension (default: ".npz") | <code>'.npz'</code>

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if cached file exists

###### `nltools.data.braindata.cache.CacheManager.get_path`

```python
get_path(key: str, ext: str = '.npz') -> Path
```

Get the file path for a cache key.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>[str](#str)</code> | Cache key | *required*
`ext` | <code>[str](#str)</code> | File extension (default: ".npz") | <code>'.npz'</code>

**Returns:**

Type | Description
---- | -----------
<code>[Path](#pathlib.Path)</code> | Path to the cache file

###### `nltools.data.braindata.cache.CacheManager.list_keys`

```python
list_keys(ext: str = '.npz') -> list[str]
```

List all cached keys in this category.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`ext` | <code>[str](#str)</code> | File extension to match | <code>'.npz'</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[str](#str)]</code> | List of cache keys (without extension)

###### `nltools.data.braindata.cache.CacheManager.load`

```python
load(key: str) -> dict | None
```

Load cached data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>[str](#str)</code> | Cache key | *required*

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict) \| None</code> | Dictionary of cached arrays, or None if not cached

###### `nltools.data.braindata.cache.CacheManager.save`

```python
save(key: str, compressed: bool = True, **arrays: bool) -> Path
```

Save arrays to cache.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>[str](#str)</code> | Cache key | *required*
`compressed` | <code>[bool](#bool)</code> | If True, use compressed npz format (smaller but slower) | <code>True</code>
`**arrays` |  | Named arrays to cache | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[Path](#pathlib.Path)</code> | Path to saved cache file



### Functions#### `nltools.data.braindata.cache.clear_cache`

```python
clear_cache(category: str | None = None) -> int
```

Clear the nltools cache.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`category` | <code>[str](#str) \| None</code> | If provided, only clear this category. Otherwise clear all. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of files deleted

#### `nltools.data.braindata.cache.get_cache_dir`

```python
get_cache_dir() -> Path
```

Get the nltools cache directory.

Returns ~/.nltools/cache, creating it if necessary.

**Returns:**

Type | Description
---- | -----------
<code>[Path](#pathlib.Path)</code> | Path to cache directory

#### `nltools.data.braindata.cache.hash_mask`

```python
hash_mask(mask_img: 'Nifti1Image') -> str
```

Compute a stable hash for a NIfTI mask image.

The hash is based on the mask's shape, affine transformation, and the
actual voxel positions. This ensures that masks with the same shape but
different voxel locations (or different affines) produce different hashes.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask_img` | <code>'Nifti1Image'</code> | NIfTI image to hash (typically a binary mask) | *required*

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | 16-character hexadecimal hash string

<details class="example" open markdown="1">
<summary>Example</summary>

>>> import nibabel as nib
>>> mask = nib.load("mask.nii.gz")
>>> hash_mask(mask)
'a1b2c3d4e5f6g7h8'

</details>

