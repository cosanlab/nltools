## `icc`

Voxel-wise Intraclass Correlation Coefficient (ICC) computation.

This module provides GPU-accelerated and CPU-parallel implementations
for computing ICC across many voxels in neuroimaging data.

Typical use case:

- Input: BrainData with shape (n_images, n_voxels)
  where n_images = n_subjects * n_sessions
- Output: ICC map with shape (n_voxels,)
- For typical MNI 2mm space: ~238,955 voxels

Performance:
- GPU: 10-50× speedup for large voxel counts (>50K voxels)
- CPU-parallel: 4-8× speedup on multi-core machines
- Single-threaded: Baseline for small problems

<details class="references" open markdown="1">
<summary>References</summary>

Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
assessing rater reliability. Psychological bulletin, 86(2), 420.

</details>

**Methods:**

Name | Description
---- | -----------
[`compute_icc_voxelwise`](#compute_icc_voxelwise) | Compute voxel-wise ICC across many voxels.

### Methods

#### `compute_icc_voxelwise`

```python
compute_icc_voxelwise(data: np.ndarray, n_subjects: int, n_sessions: int, icc_type: Literal['icc1', 'icc2', 'icc3'] = 'icc2', parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, backend: Backend | None = None) -> np.ndarray
```

Compute voxel-wise ICC across many voxels.

This function computes ICC for each voxel independently, making it
highly parallelizable. Supports GPU acceleration for large voxel counts.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Data array, shape (n_images, n_voxels) where n_images = n_subjects * n_sessions | *required*
`n_subjects` | <code>[int](#int)</code> | Number of subjects | *required*
`n_sessions` | <code>[int](#int)</code> | Number of sessions per subject | *required*
`icc_type` | <code>[Literal](#typing.Literal)['icc1', 'icc2', 'icc3']</code> | Type of ICC to calculate - 'icc1': One-way random effects - 'icc2': Two-way random effects (default) - 'icc3': Two-way mixed effects | <code>'icc2'</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization method - 'cpu': CPU parallelization via joblib (for medium-sized problems, 1K-10K voxels) - 'gpu': GPU acceleration via PyTorch (recommended for large voxel counts >10K, 10-50× speedup) - None: Single-threaded vectorized NumPy (default, memory efficient for all sizes) | <code>'cpu'</code>
`Note` |  | For large voxel counts (>10K), vectorized computation (parallel=None) is | *required*
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores (-1 = all cores) Only used when parallel='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB Only used when parallel='gpu' | <code>4.0</code>
`backend` | <code>[Backend](#nltools.algorithms.backends.Backend) \| None</code> | Backend instance (auto-detected if None) | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | np.ndarray: ICC values, shape (n_voxels,)

**Examples:**

```pycon
>>> # Typical neuroimaging scenario
>>> n_subjects = 20
>>> n_sessions = 3
>>> n_voxels = 238955
>>> data = np.random.randn(n_subjects * n_sessions, n_voxels)
>>> icc_map = compute_icc_voxelwise(
...     data, n_subjects, n_sessions,
...     parallel='gpu',  # GPU for large voxel count
...     icc_type='icc2'
... )
>>> icc_map.shape
(238955,)
>>> np.all((-1 <= icc_map) & (icc_map <= 1))
True
```

