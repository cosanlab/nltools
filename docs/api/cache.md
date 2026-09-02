---
title: data.braindata.cache
label: page-cache
---

Disk-based caching infrastructure for expensive computations.

This module provides a general-purpose caching system for nltools, designed to
be reused across various computationally expensive operations like searchlight
neighborhoods, ISC, and SRM.

**Classes:**

Name | Description
---- | -----------
[`CacheManager`](#cache-cachemanager) | Manages disk-based caching for expensive computations.

**Functions:**

Name | Description
---- | -----------
[`clear_cache`](#cache-clear-cache) | Clear the nltools cache.
[`get_cache_dir`](#cache-get-cache-dir) | Get the nltools cache directory.
[`hash_mask`](#cache-hash-mask) | Compute a stable hash for a NIfTI mask image.



**Examples:**

```python
import nibabel as nib
from nltools.data.braindata.cache import CacheManager, hash_mask

# Hash a mask for cache key generation
mask = nib.load("mask.nii.gz")
mask_hash = hash_mask(mask)

# Use cache manager for searchlight neighborhoods
cache = CacheManager("searchlight")
if not cache.exists(f"{mask_hash}_10mm"):
    result = compute_something()  # expensive operation
    cache.save(f"{mask_hash}_10mm", data=result)
else:
    result = cache.load(f"{mask_hash}_10mm")["data"]
```

## Classes

(cache-cachemanager)=
### `CacheManager`

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
`category` | <code>str</code> | Category name for organizing cached files (e.g., "searchlight") | <code>'general'</code>

**Methods:**

Name | Description
---- | -----------
[`clear`](#cache-clear) | Clear all cached files in this category.
[`delete`](#cache-delete) | Delete a cached file.
[`exists`](#cache-exists) | Check if a cache key exists.
[`get_path`](#cache-get-path) | Get the file path for a cache key.
[`list_keys`](#cache-list-keys) | List all cached keys in this category.
[`load`](#cache-load) | Load cached data.
[`save`](#cache-save) | Save arrays to cache.



**Examples:**

```python
cache = CacheManager("searchlight")

# Load from cache if present, otherwise compute and store
if cache.exists("mykey"):
    data = cache.load("mykey")
else:
    result = expensive_computation()
    cache.save("mykey", adjacency=result, metadata=metadata)
    data = {"adjacency": result, "metadata": metadata}
```

#### Methods

(cache-clear)=
##### `clear`

```python
clear() -> int
```

Clear all cached files in this category.

**Returns:**

Type | Description
---- | -----------
<code>int</code> | Number of files deleted

(cache-delete)=
##### `delete`

```python
delete(key: str, ext: str = '.npz') -> bool
```

Delete a cached file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>str</code> | Cache key | *required*
`ext` | <code>str</code> | File extension | <code>'.npz'</code>

**Returns:**

Type | Description
---- | -----------
<code>bool</code> | True if file was deleted, False if it didn't exist

(cache-exists)=
##### `exists`

```python
exists(key: str, ext: str = '.npz') -> bool
```

Check if a cache key exists.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>str</code> | Cache key | *required*
`ext` | <code>str</code> | File extension (default: ".npz") | <code>'.npz'</code>

**Returns:**

Type | Description
---- | -----------
<code>bool</code> | True if cached file exists

(cache-get-path)=
##### `get_path`

```python
get_path(key: str, ext: str = '.npz') -> Path
```

Get the file path for a cache key.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>str</code> | Cache key | *required*
`ext` | <code>str</code> | File extension (default: ".npz") | <code>'.npz'</code>

**Returns:**

Type | Description
---- | -----------
<code>Path</code> | Path to the cache file

(cache-list-keys)=
##### `list_keys`

```python
list_keys(ext: str = '.npz') -> list[str]
```

List all cached keys in this category.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`ext` | <code>str</code> | File extension to match | <code>'.npz'</code>

**Returns:**

Type | Description
---- | -----------
<code>list[str]</code> | List of cache keys (without extension)

(cache-load)=
##### `load`

```python
load(key: str) -> dict | None
```

Load cached data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>str</code> | Cache key | *required*

**Returns:**

Type | Description
---- | -----------
<code>dict \| None</code> | Dictionary of cached arrays, or None if not cached

(cache-save)=
##### `save`

```python
save(key: str, compressed: bool = True, **arrays: bool) -> Path
```

Save arrays to cache.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>str</code> | Cache key | *required*
`compressed` | <code>bool</code> | If True, use compressed npz format (smaller but slower) | <code>True</code>
`**arrays` | <code>ndarray</code> | Named arrays to cache, forwarded to ``np.savez`` / ``np.savez_compressed`` | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>Path</code> | Path to saved cache file



## Functions

(cache-clear-cache)=
### `clear_cache`

```python
clear_cache(category: str | None = None) -> int
```

Clear the nltools cache.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`category` | <code>str \| None</code> | If provided, only clear this category. Otherwise clear all. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>int</code> | Number of files deleted

(cache-get-cache-dir)=
### `get_cache_dir`

```python
get_cache_dir() -> Path
```

Get the nltools cache directory.

Returns ~/.nltools/cache, creating it if necessary.

**Returns:**

Type | Description
---- | -----------
<code>Path</code> | Path to cache directory

(cache-hash-mask)=
### `hash_mask`

```python
hash_mask(mask_img: Nifti1Image) -> str
```

Compute a stable hash for a NIfTI mask image.

The hash is based on the mask's shape, affine transformation, and the
actual voxel positions. This ensures that masks with the same shape but
different voxel locations (or different affines) produce different hashes.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask_img` | <code>Nifti1Image</code> | NIfTI image to hash (typically a binary mask) | *required*

**Returns:**

Type | Description
---- | -----------
<code>str</code> | 16-character hexadecimal hash string

**Examples:**

```python
import nibabel as nib

mask = nib.load("mask.nii.gz")
hash_mask(mask)  # 'a1b2c3d4e5f60789'
```
