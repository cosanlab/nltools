# `nltools.neighborhoods`

**Searchlight Neighborhoods**

Compute and cache spherical neighborhoods for searchlight analyses.

- `SphereNeighborhoods` — Container for precomputed voxel neighborhoods
- `compute_searchlight_neighborhoods` — Build neighborhoods from a brain mask

```{eval-rst}
.. autofunction:: nltools.neighborhoods.compute_searchlight_neighborhoods

.. autoclass:: nltools.neighborhoods.SphereNeighborhoods
    :members:
    :show-inheritance:
    :exclude-members: adjacency, mask_hash, radius_mm, n_voxels
```

## See Also

- {doc}`cache` — Caching infrastructure used by neighborhood computation
- {doc}`data/brain_data` — BrainData searchlight integration
