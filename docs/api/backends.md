## `nltools.backends`

Backend abstraction for CPU/GPU operations.

Supports NumPy (CPU-only) and PyTorch (CPU/CUDA/MPS) backends for
linear algebra operations. Enables transparent acceleration while
maintaining NumPy-first development.

**Classes:**

Name | Description
---- | -----------
[`Backend`](#nltools.backends.Backend) | Backend abstraction for numerical operations.

**Functions:**

Name | Description
---- | -----------
[`assert_array_almost_equal`](#nltools.backends.assert_array_almost_equal) | Test array equality with automatic precision adjustment for MPS backend.
[`auto_select_backend`](#nltools.backends.auto_select_backend) | Automatically select backend based on problem size.
[`check_gpu_available`](#nltools.backends.check_gpu_available) | Check if GPU acceleration is available.



### Classes#### `nltools.backends.Backend`

```python
Backend(backend: str = 'numpy')
```

Backend abstraction for numerical operations.

Provides a unified interface for NumPy and PyTorch operations,
enabling transparent GPU acceleration when available.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`backend` | <code>[str](#str)</code> | Backend type: 'numpy', 'torch', or 'auto' - 'numpy': CPU-only using NumPy - 'torch': PyTorch with automatic device detection (cuda/mps/cpu) - 'auto': Automatically select best available backend | <code>'numpy'</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`name`](#nltools.backends.Backend.name) | <code>[str](#str)</code> | Backend identifier (e.g., 'numpy', 'torch-cuda', 'torch-mps')
[`device`](#nltools.backends.Backend.device) | <code>[str](#str)</code> | Device type ('cpu', 'cuda', or 'mps')
[`xp`](#nltools.backends.Backend.xp) | <code>[module](#module)</code> | Array library module (numpy or torch)

**Functions:**

Name | Description
---- | -----------
[`matmul`](#nltools.backends.Backend.matmul) | Matrix multiplication.
[`svd`](#nltools.backends.Backend.svd) | Compute Singular Value Decomposition.
[`to_device`](#nltools.backends.Backend.to_device) | Transfer array to backend device.
[`to_numpy`](#nltools.backends.Backend.to_numpy) | Convert array back to NumPy.



##### Functions###### `nltools.backends.Backend.matmul`

```python
matmul(A, B)
```

Matrix multiplication.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`A` | <code>[array](#array)</code> | First matrix | *required*
`B` | <code>[array](#array)</code> | Second matrix | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`array` |  | Result of A @ B

###### `nltools.backends.Backend.svd`

```python
svd(X, full_matrices = False)
```

Compute Singular Value Decomposition.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[array](#array)</code> | Input matrix (n_samples, n_features) | *required*
`full_matrices` | <code>bool, default=False</code> | If False, returns reduced SVD | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` |  | (U, s, Vt) where: - U (array): Left singular vectors - s (array): Singular values - Vt (array): Right singular vectors (transposed)

###### `nltools.backends.Backend.to_device`

```python
to_device(arr: np.ndarray)
```

Transfer array to backend device.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`arr` | <code>[ndarray](#numpy.ndarray)</code> | Input numpy array | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`array` |  | Array on device (numpy array or torch tensor)

###### `nltools.backends.Backend.to_numpy`

```python
to_numpy(arr)
```

Convert array back to NumPy.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`arr` | <code>[ndarray](#numpy.ndarray) or [Tensor](#torch.Tensor)</code> | Array to convert | *required*

**Returns:**

Type | Description
---- | -----------
 | np.ndarray: NumPy array



### Functions#### `nltools.backends.assert_array_almost_equal`

```python
assert_array_almost_equal(x, y, decimal = 6, err_msg = '', verbose = True, backend = None)
```

Test array equality with automatic precision adjustment for MPS backend.

This utility automatically reduces precision expectations for torch-mps backend
due to float32 precision limitations, preventing test failures while maintaining
realistic precision checks for other backends.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` |  | First array to compare | *required*
`y` |  | Second array to compare | *required*
`decimal` |  | Desired decimal precision (default: 6) | <code>6</code>
`err_msg` |  | Error message prefix | <code>''</code>
`verbose` |  | Whether to print detailed error messages | <code>True</code>
`backend` |  | Backend instance (optional). If None, attempts to detect from x/y. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | None (raises AssertionError if arrays don't match)

#### `nltools.backends.auto_select_backend`

```python
auto_select_backend(n_samples: int, n_features: int, cv: int = 1) -> Backend
```

Automatically select backend based on problem size.

Uses heuristics to decide between NumPy (CPU) and PyTorch (GPU)
based on the computational workload. Small problems use NumPy
to avoid GPU transfer overhead. Large problems prefer GPU when
available.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_samples` | <code>[int](#int)</code> | Number of samples in dataset | *required*
`n_features` | <code>[int](#int)</code> | Number of features in dataset | *required*
`cv` | <code>int, default=1</code> | Number of cross-validation folds (multiplies effective size) | <code>1</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Backend` | <code>[Backend](#nltools.backends.Backend)</code> | Selected backend instance

<details class="notes" open markdown="1">
<summary>Notes</summary>

Selection criteria:
- Small problems (< 10M elements): Use NumPy
- Large problems (> 30M elements): Use GPU if available
- Cross-validation: Prefer GPU even for medium problems

</details>

#### `nltools.backends.check_gpu_available`

```python
check_gpu_available() -> Tuple[bool, Dict[str, Any]]
```

Check if GPU acceleration is available.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` | <code>[Tuple](#typing.Tuple)[[bool](#bool), [Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]]</code> | (available, info) where: - available (bool): True if GPU (CUDA or MPS) is available - info (dict): Dictionary with keys:     - 'backend': 'torch' or 'numpy'     - 'device': 'cpu', 'cuda', or 'mps'     - 'device_name': Human-readable device name

