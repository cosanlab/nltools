## `nltools.data.braindata.neighborhoods`

Spatial neighborhood computation for neuroimaging analyses.

This module provides efficient computation and caching of spatial neighborhoods
(spheres) around brain voxels. It is designed to support searchlight analyses,
ISC, and other operations that require iterating over local brain regions.

The key insight is that for a given mask and radius, the neighborhood structure
is deterministic and can be cached for reuse across analyses.

<details class="example" open markdown="1">
<summary>Example</summary>

>>> import nibabel as nib
>>> from nltools.data.braindata.neighborhoods import compute_searchlight_neighborhoods
>>>
>>> mask = nib.load("mask.nii.gz")
>>> neighborhoods = compute_searchlight_neighborhoods(mask, radius_mm=10.0)
>>>
>>> # Iterate over all voxels and their neighborhoods
>>> for center_idx, neighbor_indices in neighborhoods.iter_neighborhoods():
...     # Extract data for these voxels
...     local_data = data[:, neighbor_indices]
...     result[center_idx] = analyze(local_data)

</details>

**Classes:**

Name | Description
---- | -----------
[`SphereNeighborhoods`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods) | Precomputed sphere neighborhoods for a brain mask.

**Functions:**

Name | Description
---- | -----------
[`compute_searchlight_neighborhoods`](#nltools.data.braindata.neighborhoods.compute_searchlight_neighborhoods) | Compute sphere neighborhoods for all voxels in a brain mask.



### Classes#### `nltools.data.braindata.neighborhoods.SphereNeighborhoods`

```python
SphereNeighborhoods(adjacency: sparse.csr_matrix, mask_hash: str, radius_mm: float, n_voxels: int) -> None
```

Precomputed sphere neighborhoods for a brain mask.

This dataclass stores a sparse adjacency matrix where row i contains True
for all voxels within the specified radius of voxel i. It provides efficient
iteration over neighborhoods for searchlight-style analyses.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`adjacency`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.adjacency) | <code>[csr_matrix](#scipy.sparse.csr_matrix)</code> | Sparse CSR matrix (n_voxels, n_voxels) where adjacency[i, j] is True if voxel j is within radius of voxel i
[`mask_hash`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.mask_hash) | <code>[str](#str)</code> | Hash of the source mask for validation
[`radius_mm`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.radius_mm) | <code>[float](#float)</code> | Radius in millimeters
[`n_voxels`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.n_voxels) | <code>[int](#int)</code> | Number of voxels in the mask

<details class="example" open markdown="1">
<summary>Example</summary>

>>> neighborhoods = compute_searchlight_neighborhoods(mask, radius_mm=10.0)
>>> print(f"Mean neighborhood size: {neighborhoods.mean_size:.1f} voxels")
>>>
>>> # Get neighbors of a specific voxel
>>> neighbor_idx = neighborhoods.get_neighbors(100)
>>> print(f"Voxel 100 has {len(neighbor_idx)} neighbors")

</details>

**Functions:**

Name | Description
---- | -----------
[`get_neighborhood_size`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.get_neighborhood_size) | Get the number of voxels in a neighborhood.
[`get_neighbors`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.get_neighbors) | Get indices of all voxels in the neighborhood of a given voxel.
[`iter_neighborhoods`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.iter_neighborhoods) | Iterate over all neighborhoods.



##### Attributes###### `nltools.data.braindata.neighborhoods.SphereNeighborhoods.adjacency`

```python
adjacency: sparse.csr_matrix
```

###### `nltools.data.braindata.neighborhoods.SphereNeighborhoods.mask_hash`

```python
mask_hash: str
```

###### `nltools.data.braindata.neighborhoods.SphereNeighborhoods.max_size`

```python
max_size: int
```

Maximum neighborhood size.

###### `nltools.data.braindata.neighborhoods.SphereNeighborhoods.mean_size`

```python
mean_size: float
```

Mean neighborhood size in voxels.

###### `nltools.data.braindata.neighborhoods.SphereNeighborhoods.min_size`

```python
min_size: int
```

Minimum neighborhood size.

###### `nltools.data.braindata.neighborhoods.SphereNeighborhoods.n_voxels`

```python
n_voxels: int
```

###### `nltools.data.braindata.neighborhoods.SphereNeighborhoods.radius_mm`

```python
radius_mm: float
```



##### Functions###### `nltools.data.braindata.neighborhoods.SphereNeighborhoods.get_neighborhood_size`

```python
get_neighborhood_size(voxel_idx: int) -> int
```

Get the number of voxels in a neighborhood.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`voxel_idx` | <code>[int](#int)</code> | Index of the center voxel | *required*

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of voxels in the neighborhood

###### `nltools.data.braindata.neighborhoods.SphereNeighborhoods.get_neighbors`

```python
get_neighbors(voxel_idx: int) -> np.ndarray
```

Get indices of all voxels in the neighborhood of a given voxel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`voxel_idx` | <code>[int](#int)</code> | Index of the center voxel (0 to n_voxels-1) | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Array of voxel indices within radius of the center voxel

###### `nltools.data.braindata.neighborhoods.SphereNeighborhoods.iter_neighborhoods`

```python
iter_neighborhoods(show_progress: bool = False) -> Iterator[tuple[int, np.ndarray]]
```

Iterate over all neighborhoods.

**Yields:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[int](#int), [ndarray](#numpy.ndarray)]</code> | Tuple of (center_voxel_idx, neighbor_indices) for each voxel

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`show_progress` | <code>[bool](#bool)</code> | If True, wrap iterator with tqdm progress bar | <code>False</code>



### Functions#### `nltools.data.braindata.neighborhoods.compute_searchlight_neighborhoods`

```python
compute_searchlight_neighborhoods(mask_img: 'Nifti1Image', radius_mm: float = 10.0, use_cache: bool = True) -> SphereNeighborhoods
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
`mask_img` | <code>'Nifti1Image'</code> | NIfTI mask image defining the brain region | *required*
`radius_mm` | <code>[float](#float)</code> | Radius of spheres in millimeters (default: 10.0) | <code>10.0</code>
`use_cache` | <code>[bool](#bool)</code> | If True, cache results to ~/.nltools/cache/searchlight/ for fast reloading (default: True) | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[SphereNeighborhoods](#nltools.data.braindata.neighborhoods.SphereNeighborhoods)</code> | SphereNeighborhoods with precomputed adjacency matrix

<details class="example" open markdown="1">
<summary>Example</summary>

>>> import nibabel as nib
>>> mask = nib.load("brain_mask.nii.gz")
>>>
>>> # First call computes and caches (may take a few seconds)
>>> neighborhoods = compute_searchlight_neighborhoods(mask, radius_mm=8.0)
>>>
>>> # Subsequent calls load from cache (~50ms)
>>> neighborhoods = compute_searchlight_neighborhoods(mask, radius_mm=8.0)
>>>
>>> print(neighborhoods)
SphereNeighborhoods(n_voxels=50000, radius=8.0mm, mean_size=33.2)

</details>

<details class="notes" open markdown="1">
<summary>Notes</summary>

Cache location: ~/.nltools/cache/searchlight/{mask_hash}_{radius}mm.npz

For a typical 2mm MNI mask (~50k voxels) with 10mm radius:
- First run: ~1-2 seconds
- Cached load: ~50ms

</details>

