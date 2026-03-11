# `nltools.cache`

**Disk Caching**

Persistent caching for expensive computations (e.g., searchlight neighborhoods, alignment transforms).

- `CacheManager` — Context manager for scoped caching
- `get_cache_dir` — Get the cache directory path
- `hash_mask` — Generate a hash key from a brain mask
- `clear_cache` — Remove cached data by category

```{eval-rst}
.. automodule:: nltools.cache
    :members:
    :undoc-members:
    :show-inheritance:
```

## See Also

- {doc}`neighborhoods` — Primary user of the cache system
- {doc}`prefs` — Global preferences configuration
