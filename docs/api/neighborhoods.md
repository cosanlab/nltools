---
title: Searchlight neighborhoods
label: page-neighborhoods
---

Spatial neighborhood computation for neuroimaging analyses.

This module provides efficient computation and caching of spatial neighborhoods
(spheres) around brain voxels. It is designed to support searchlight analyses,
ISC, and other operations that require iterating over local brain regions.

The key insight is that for a given mask and radius, the neighborhood structure
is deterministic and can be cached for reuse across analyses.

**Classes:**

Name | Description
---- | -----------
[`SphereNeighborhoods`](#neighborhoods-sphereneighborhoods) | Precomputed sphere neighborhoods for a brain mask.

**Functions:**

Name | Description
---- | -----------
[`compute_searchlight_neighborhoods`](#neighborhoods-compute-searchlight-neighborhoods) | Compute sphere neighborhoods for all voxels in a brain mask.



**Examples:**

```python
import nibabel as nib
from nltools.data.braindata.neighborhoods import compute_searchlight_neighborhoods

mask = nib.load("mask.nii.gz")
neighborhoods = compute_searchlight_neighborhoods(mask, radius=10.0)

# Iterate over all voxels and their neighborhoods
for center_idx, neighbor_indices in neighborhoods.iter_neighborhoods():
    local_data = data[:, neighbor_indices]  # data for the voxels in this sphere
    result[center_idx] = analyze(local_data)
```

## Classes

(neighborhoods-sphereneighborhoods)=
### `SphereNeighborhoods`

```python
SphereNeighborhoods(adjacency: sparse.csr_matrix, mask_hash: str, radius: float, n_voxels: int)
```

Precomputed sphere neighborhoods for a brain mask.

This dataclass stores a sparse adjacency matrix where row i contains True
for all voxels within the specified radius of voxel i. It provides efficient
iteration over neighborhoods for searchlight-style analyses.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`adjacency` | <code>csr_matrix</code> | ``(n_voxels, n_voxels)`` matrix where ``adjacency[i, j]`` is nonzero if voxel ``j`` is within the radius of voxel ``i``.
`mask_hash` | <code>str</code> | Hash of the source mask, for cache validation.
`radius` | <code>float</code> | Radius in millimeters.
`n_voxels` | <code>int</code> | Number of voxels in the mask.
`mean_size` | <code>float</code> | Mean neighborhood size in voxels.
`min_size` | <code>int</code> | Smallest neighborhood size in voxels.
`max_size` | <code>int</code> | Largest neighborhood size in voxels.

**Methods:**

Name | Description
---- | -----------
[`get_neighborhood_size`](#neighborhoods-get-neighborhood-size) | Get the number of voxels in a neighborhood.
[`get_neighbors`](#neighborhoods-get-neighbors) | Get indices of all voxels in the neighborhood of a given voxel.
[`iter_neighborhoods`](#neighborhoods-iter-neighborhoods) | Iterate over all neighborhoods.



**Examples:**

```python
neighborhoods = compute_searchlight_neighborhoods(mask, radius=10.0)
print(f"Mean neighborhood size: {neighborhoods.mean_size:.1f} voxels")

# Get neighbors of a specific voxel
neighbor_idx = neighborhoods.get_neighbors(100)
print(f"Voxel 100 has {len(neighbor_idx)} neighbors")
```

#### Methods

(neighborhoods-get-neighborhood-size)=
##### `get_neighborhood_size`

```python
get_neighborhood_size(voxel_idx: int) -> int
```

Get the number of voxels in a neighborhood.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`voxel_idx` | <code>int</code> | Index of the center voxel | *required*

**Returns:**

Type | Description
---- | -----------
<code>int</code> | Number of voxels in the neighborhood

(neighborhoods-get-neighbors)=
##### `get_neighbors`

```python
get_neighbors(voxel_idx: int) -> np.ndarray
```

Get indices of all voxels in the neighborhood of a given voxel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`voxel_idx` | <code>int</code> | Index of the center voxel (0 to n_voxels-1) | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Array of voxel indices within radius of the center voxel

(neighborhoods-iter-neighborhoods)=
##### `iter_neighborhoods`

```python
iter_neighborhoods(*, progress_bar: bool = False) -> Iterator[tuple[int, np.ndarray]]
```

Iterate over all neighborhoods.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`progress_bar` | <code>bool</code> | If True, wrap the iterator with a tqdm progress bar. | <code>False</code>

**Yields:**

Type | Description
---- | -----------
<code>tuple[int, ndarray]</code> | ``(center_voxel_idx, neighbor_indices)`` for     each voxel.



## Functions

(neighborhoods-compute-searchlight-neighborhoods)=
### `compute_searchlight_neighborhoods`

```python
compute_searchlight_neighborhoods(mask_img: Nifti1Image, radius: float = 10.0, use_cache: bool = True) -> SphereNeighborhoods
```

Compute sphere neighborhoods for all voxels in a brain mask.

For each voxel in the mask, this function identifies all other voxels
within the specified radius (in millimeters). The result is cached to
disk for fast reloading in subsequent analyses.

The algorithm uses sklearn's BallTree for efficient radius queries in
world coordinates (mm), ensuring accurate neighborhoods regardless of
voxel resolution.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask_img` | <code>Nifti1Image</code> | NIfTI mask image defining the brain region | *required*
`radius` | <code>float</code> | Radius of spheres in millimeters (default: 10.0) | <code>10.0</code>
`use_cache` | <code>bool</code> | If True, cache results to ~/.nltools/cache/searchlight/ for fast reloading (default: True) | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[SphereNeighborhoods](#algorithms-sphereneighborhoods)</code> | SphereNeighborhoods with precomputed adjacency matrix

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If mask has no non-zero voxels

**Examples:**

```python
import nibabel as nib

mask = nib.load("brain_mask.nii.gz")

# First call computes and caches (may take a few seconds)
neighborhoods = compute_searchlight_neighborhoods(mask, radius=8.0)

# Subsequent calls load from cache (~50ms)
neighborhoods = compute_searchlight_neighborhoods(mask, radius=8.0)

print(neighborhoods)
# SphereNeighborhoods(n_voxels=50000, radius=8.0mm, mean_size=33.2)
```

<details class="note" open markdown="1">
<summary>Note</summary>

Cache location: ``~/.nltools/cache/searchlight/{mask_hash}_{radius}mm.npz``.
For a typical 2mm MNI mask (~50k voxels) with a 10mm radius the first
run takes ~1-2 seconds; a cached load takes ~50ms.

</details>
