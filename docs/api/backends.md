## `backends`

Backend abstraction for CPU/GPU operations.

Supports NumPy (CPU-only) and PyTorch (CPU/CUDA/MPS) backends for
linear algebra operations. Enables transparent acceleration while
maintaining NumPy-first development.

**Classes:**

Name | Description
---- | -----------
[`Backend`](#Backend) | Backend abstraction for numerical operations.

**Methods:**

Name | Description
---- | -----------
[`assert_array_almost_equal`](#assert_array_almost_equal) | Test array equality with automatic precision adjustment for MPS backend.
[`auto_select_backend`](#auto_select_backend) | Automatically select backend based on problem size.
[`check_gpu_available`](#check_gpu_available) | Check if GPU acceleration is available.



### Classes

#### `Backend`

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
[`name`](#name) | <code>[str](#str)</code> | Backend identifier (e.g., 'numpy', 'torch-cuda', 'torch-mps')
[`device`](#device) | <code>[str](#str)</code> | Device type ('cpu', 'cuda', or 'mps')
[`xp`](#xp) | <code>[module](#module)</code> | Array library module (numpy or torch)

**Methods:**

Name | Description
---- | -----------
[`asarray`](#asarray) | Convert input to a backend array.
[`asarray_like`](#asarray_like) | Convert *x* to an array matching *ref*'s dtype (and device for torch).
[`check_arrays`](#check_arrays) | Coerce all inputs to the same dtype (and device) as the first.
[`concatenate`](#concatenate) | Concatenate arrays along an axis.
[`copy`](#copy) | Return an independent copy of the array.
[`dtype_to_str`](#dtype_to_str) | Normalize a dtype (numpy, torch, or string) to its string name.
[`expand_dims`](#expand_dims) | Insert a new axis.
[`flatnonzero`](#flatnonzero) | Return indices of non-zero elements in the flattened array.
[`full`](#full) | Create array filled with *fill_value*.
[`full_like`](#full_like) | Create array filled with *fill_value*, optionally with a different shape.
[`matmul`](#matmul) | Matrix multiplication.
[`ones_like`](#ones_like) | Create ones array, optionally with a different shape.
[`sort`](#sort) | Sort along an axis, returning values only.
[`svd`](#svd) | Compute Singular Value Decomposition.
[`to_cpu`](#to_cpu) | Transfer array to CPU. No-op for numpy.
[`to_device`](#to_device) | Transfer array to backend device.
[`to_gpu`](#to_gpu) | Transfer array to GPU. No-op for numpy.
[`to_numpy`](#to_numpy) | Convert array back to NumPy.
[`zeros_like`](#zeros_like) | Create zeros array, optionally with a different shape.

##### Methods

###### `asarray`

```python
asarray(x, dtype = None, device = None)
```

Convert input to a backend array.

Handles numpy arrays, lists, and torch tensors. Places result on
the backend's device (or an explicit *device*).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` |  | Input data (array-like, tensor, list). | *required*
`dtype` |  | Desired dtype as string, numpy, or torch dtype. If None, inferred from input. | <code>None</code>
`device` |  | Target device string (e.g. "cpu", "cuda"). Ignored for numpy backend. If None, uses the backend's default device. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | Backend array (numpy ndarray or torch Tensor).

###### `asarray_like`

```python
asarray_like(x, ref)
```

Convert *x* to an array matching *ref*'s dtype (and device for torch).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` |  | Input data. | *required*
`ref` |  | Reference array whose dtype/device to match. | *required*

**Returns:**

Type | Description
---- | -----------
 | Backend array with same dtype/device as ref.

###### `check_arrays`

```python
check_arrays(*inputs)
```

Coerce all inputs to the same dtype (and device) as the first.

None values are passed through. Lists of arrays are converted
element-wise.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`*inputs` |  | Arrays, lists of arrays, or None. | <code>()</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`list` |  | Converted arrays in the same order as inputs.

###### `concatenate`

```python
concatenate(arrays, axis = 0)
```

Concatenate arrays along an axis.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`arrays` |  | Sequence of arrays. | *required*
`axis` |  | Axis to concatenate along (default 0). | <code>0</code>

###### `copy`

```python
copy(array)
```

Return an independent copy of the array.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Input array. | *required*

###### `dtype_to_str`

```python
dtype_to_str(dtype)
```

Normalize a dtype (numpy, torch, or string) to its string name.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dtype` |  | Data type to convert (str, numpy dtype, torch dtype, or None). | *required*

**Returns:**

Type | Description
---- | -----------
 | str or None: e.g. "float32", "float64", or None if input was None.

###### `expand_dims`

```python
expand_dims(array, axis)
```

Insert a new axis.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Input array. | *required*
`axis` |  | Position of the new axis. | *required*

###### `flatnonzero`

```python
flatnonzero(array)
```

Return indices of non-zero elements in the flattened array.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Input array. | *required*

###### `full`

```python
full(shape, fill_value, dtype = None)
```

Create array filled with *fill_value*.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`shape` |  | Output shape (int or tuple). | *required*
`fill_value` |  | Scalar fill value. | *required*
`dtype` |  | Output dtype. If None, inferred by the backend. | <code>None</code>

###### `full_like`

```python
full_like(array, fill_value, shape = None, dtype = None, device = None)
```

Create array filled with *fill_value*, optionally with a different shape.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Reference array for dtype inference. | *required*
`fill_value` |  | Scalar fill value. | *required*
`shape` |  | Output shape. If None, uses array.shape. | <code>None</code>
`dtype` |  | Output dtype. If None, uses array.dtype. | <code>None</code>
`device` |  | Target device (torch only). If None, uses array's device. | <code>None</code>

###### `matmul`

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

###### `ones_like`

```python
ones_like(array, shape = None, dtype = None, device = None)
```

Create ones array, optionally with a different shape.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Reference array for dtype inference. | *required*
`shape` |  | Output shape. If None, uses array.shape. | <code>None</code>
`dtype` |  | Output dtype. If None, uses array.dtype. | <code>None</code>
`device` |  | Target device (torch only). If None, uses array's device. | <code>None</code>

###### `sort`

```python
sort(array, axis = -1)
```

Sort along an axis, returning values only.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Input array. | *required*
`axis` |  | Axis to sort along (default -1). | <code>-1</code>

###### `svd`

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

###### `to_cpu`

```python
to_cpu(array)
```

Transfer array to CPU. No-op for numpy.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Input array or tensor. | *required*

**Returns:**

Type | Description
---- | -----------
 | Array on CPU.

###### `to_device`

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

###### `to_gpu`

```python
to_gpu(array, device = None)
```

Transfer array to GPU. No-op for numpy.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Input array or tensor. | *required*
`device` |  | Target device (defaults to backend's device). | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | Array on GPU device.

###### `to_numpy`

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

###### `zeros_like`

```python
zeros_like(array, shape = None, dtype = None, device = None)
```

Create zeros array, optionally with a different shape.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Reference array for dtype inference. | *required*
`shape` |  | Output shape. If None, uses array.shape. | <code>None</code>
`dtype` |  | Output dtype. If None, uses array.dtype. | <code>None</code>
`device` |  | Target device (torch only). If None, uses array's device. | <code>None</code>



### Methods

#### `assert_array_almost_equal`

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

#### `auto_select_backend`

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

#### `check_gpu_available`

```python
check_gpu_available() -> Tuple[bool, Dict[str, Any]]
```

Check if GPU acceleration is available.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` | <code>[Tuple](#typing.Tuple)[[bool](#bool), [Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]]</code> | (available, info) where: - available (bool): True if GPU (CUDA or MPS) is available - info (dict): Dictionary with keys:     - 'backend': 'torch' or 'numpy'     - 'device': 'cpu', 'cuda', or 'mps'     - 'device_name': Human-readable device name

