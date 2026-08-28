(backends-backends)=
## `backends`

Backend abstraction for CPU/GPU operations.

Supports NumPy (CPU-only) and PyTorch (CPU/CUDA/MPS) backends for
linear algebra operations. Enables transparent acceleration while
maintaining NumPy-first development.

**Classes:**

Name | Description
---- | -----------
[`Backend`](#backends-backend) | Backend abstraction for numerical operations.

**Methods:**

Name | Description
---- | -----------
[`assert_array_almost_equal`](#backends-assert-array-almost-equal) | Test array equality with automatic precision adjustment for MPS backend.
[`auto_batch_size`](#backends-auto-batch-size) | Split `n_items` into batches that fit a memory budget.
[`auto_n_jobs_for_arrays`](#backends-auto-n-jobs-for-arrays) | Memory-aware joblib worker count for a per-item map over arrays.
[`auto_select_backend`](#backends-auto-select-backend) | Automatically select backend based on problem size.
[`check_gpu_available`](#backends-check-gpu-available) | Check if GPU acceleration is available.
[`compute_oom_safe`](#backends-compute-oom-safe) | Run `fn(*arrays)` with reactive out-of-memory recovery.
[`device_memory_budget`](#backends-device-memory-budget) | Usable memory budget in GB for a backend's device.
[`empty_device_cache`](#backends-empty-device-cache) | Release cached device memory. No-op without torch or a GPU.
[`gb_to_bytes`](#backends-gb-to-bytes) | Convert a GB budget to bytes — the package's one GB↔bytes conversion.
[`is_oom_error`](#backends-is-oom-error) | True if `exc` is a device out-of-memory error (CUDA or MPS).
[`resolve_backend`](#backends-resolve-backend) | Coerce a backend specifier into a `Backend` instance.



### Classes

(backends-backend)=
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
`name` | <code>[str](#str)</code> | Backend identifier (e.g., 'numpy', 'torch-cuda', 'torch-mps')
`device` | <code>[str](#str)</code> | Device type ('cpu', 'cuda', or 'mps')
`xp` | <code>[module](#module)</code> | Array library module (numpy or torch)

**Methods:**

Name | Description
---- | -----------
[`asarray`](#backends-asarray) | Convert input to a backend array.
[`asarray_like`](#backends-asarray-like) | Convert *x* to an array matching *ref*'s dtype (and device for torch).
[`check_arrays`](#backends-check-arrays) | Coerce all inputs to the same dtype (and device) as the first.
[`concatenate`](#backends-concatenate) | Concatenate arrays along an axis.
[`copy`](#backends-copy) | Return an independent copy of the array.
[`dtype_to_str`](#backends-dtype-to-str) | Normalize a dtype (numpy, torch, or string) to its string name.
[`expand_dims`](#backends-expand-dims) | Insert a new axis.
[`flatnonzero`](#backends-flatnonzero) | Return indices of non-zero elements in the flattened array.
[`full`](#backends-full) | Create array filled with *fill_value*.
[`full_like`](#backends-full-like) | Create array filled with *fill_value*, optionally with a different shape.
[`matmul`](#backends-matmul) | Matrix multiplication.
[`ones_like`](#backends-ones-like) | Create ones array, optionally with a different shape.
[`sort`](#backends-sort) | Sort along an axis, returning values only.
[`svd`](#backends-svd) | Compute Singular Value Decomposition.
[`to_cpu`](#backends-to-cpu) | Transfer array to CPU. No-op for numpy.
[`to_device`](#backends-to-device) | Transfer array to backend device.
[`to_gpu`](#backends-to-gpu) | Transfer array to GPU. No-op for numpy.
[`to_numpy`](#backends-to-numpy) | Convert array back to NumPy.
[`zeros_like`](#backends-zeros-like) | Create zeros array, optionally with a different shape.

##### Methods

(backends-asarray)=
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

(backends-asarray-like)=
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

(backends-check-arrays)=
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

(backends-concatenate)=
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

(backends-copy)=
###### `copy`

```python
copy(array)
```

Return an independent copy of the array.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Input array. | *required*

(backends-dtype-to-str)=
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

(backends-expand-dims)=
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

(backends-flatnonzero)=
###### `flatnonzero`

```python
flatnonzero(array)
```

Return indices of non-zero elements in the flattened array.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Input array. | *required*

(backends-full)=
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

(backends-full-like)=
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

(backends-matmul)=
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

(backends-ones-like)=
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

(backends-sort)=
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

(backends-svd)=
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

(backends-to-cpu)=
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

(backends-to-device)=
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

(backends-to-gpu)=
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

(backends-to-numpy)=
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

(backends-zeros-like)=
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

(backends-assert-array-almost-equal)=
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

(backends-auto-batch-size)=
#### `auto_batch_size`

```python
auto_batch_size(n_items: int, bytes_per_item: float, *, budget_gb: float, overhead: float = 1.0, min_batch: int = 1) -> tuple[int, int]
```

Split `n_items` into batches that fit a memory budget.

The one batch calculator for the package. Callers supply only the
per-item working-set estimate (`bytes_per_item`) and an algorithm's
allocation `overhead` factor; the clamp/ceil policy lives here.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_items` | <code>[int](#int)</code> | Total number of items (permutations, targets, ...). | *required*
`bytes_per_item` | <code>[float](#float)</code> | Dominant working-set size of one item in bytes. | *required*
`budget_gb` | <code>[float](#float)</code> | Memory budget from `device_memory_budget`. | *required*
`overhead` | <code>[float](#float)</code> | Multiplier for intermediate allocations (e.g. 3.0 when the computation holds ~3x the input working set). | <code>1.0</code>
`min_batch` | <code>[int](#int)</code> | Smallest batch worth dispatching (amortizes launch and transfer overhead). Never exceeds `n_items`. | <code>1</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | tuple[int, int]: `(batch_size, n_batches)` with
<code>[int](#int)</code> | `batch_size * n_batches >= n_items`.

(backends-auto-n-jobs-for-arrays)=
#### `auto_n_jobs_for_arrays`

```python
auto_n_jobs_for_arrays(arrays, *, max_memory_gb: float | None = None, min_jobs: int = 1) -> int
```

Memory-aware joblib worker count for a per-item map over arrays.

Sizes workers by the largest item (each worker pickles its item), using
the same measured budget as the device batching layer. None entries are
ignored; an empty list returns ``min_jobs``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`arrays` |  | Iterable of numpy arrays (None entries allowed). | *required*
`max_memory_gb` | <code>[float](#float) \| None</code> | Explicit memory budget in GB. None (default) measures available system RAM with headroom via `device_memory_budget`. | <code>None</code>
`min_jobs` | <code>[int](#int)</code> | Minimum number of workers (default: 1). | <code>1</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`int` | <code>[int](#int)</code> | Worker count for ``joblib.Parallel(n_jobs=...)``.

(backends-auto-select-backend)=
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
`Backend` | <code>[Backend](#nltools.algorithms.backends.Backend)</code> | Selected backend instance

<details class="notes" open markdown="1">
<summary>Notes</summary>

Selection criteria:
- Small problems (< 10M elements): Use NumPy
- Large problems (> 30M elements): Use GPU if available
- Cross-validation: Prefer GPU even for medium problems

</details>

(backends-check-gpu-available)=
#### `check_gpu_available`

```python
check_gpu_available() -> tuple[bool, dict[str, Any]]
```

Check if GPU acceleration is available.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` | <code>[tuple](#tuple)[[bool](#bool), [dict](#dict)[[str](#str), [Any](#typing.Any)]]</code> | (available, info) where: - available (bool): True if GPU (CUDA or MPS) is available - info (dict): Dictionary with keys:     - 'backend': 'torch' or 'numpy'     - 'device': 'cpu', 'cuda', or 'mps'     - 'device_name': Human-readable device name

(backends-compute-oom-safe)=
#### `compute_oom_safe`

```python
compute_oom_safe(fn, *arrays, min_chunk: int = 1)
```

Run `fn(*arrays)` with reactive out-of-memory recovery.

All `arrays` must share their axis-0 length, and `fn` must map them to
a numpy array whose axis 0 corresponds row-for-row to its inputs. On a
device OOM the cache is emptied, the arrays are split in half along
axis 0, and the halves are retried recursively; partial results are
concatenated along axis 0.

Because splitting reuses the *already generated* inputs rather than
re-drawing them, recovery never changes which permutations a seeded
result is computed from — RNG-consuming input generation stays outside
this function. For a row-independent `fn` the recovered output matches
the unsplit computation to within floating-point reduction order
(backends may block reductions differently per batch shape; observed
differences are ~1 float32 ulp).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fn` |  | Callable mapping the arrays to a numpy result (axis-0 aligned). | *required*
`*arrays` |  | Input arrays sharing axis-0 length. | <code>()</code>
`min_chunk` | <code>[int](#int)</code> | Chunk size below which an OOM is considered fatal. | <code>1</code>

**Returns:**

Type | Description
---- | -----------
 | np.ndarray: `fn`'s result, possibly assembled from retried chunks.

(backends-device-memory-budget)=
#### `device_memory_budget`

```python
device_memory_budget(backend: Backend | None = None, max_gpu_memory_gb: float | None = None) -> float
```

Usable memory budget in GB for a backend's device.

An explicit `max_gpu_memory_gb` always wins. Otherwise the budget is
measured at call time: free CUDA memory (with headroom) on CUDA
devices; available system RAM (with headroom) for CPU and MPS, which
share unified/system memory. When nothing can be measured the
conservative 4 GB fallback applies.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`backend` | <code>[Backend](#nltools.algorithms.backends.Backend) \| None</code> | Resolved `Backend` whose device the work runs on. None is treated as CPU. | <code>None</code>
`max_gpu_memory_gb` | <code>[float](#float) \| None</code> | Explicit budget override in GB. Must be positive. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`float` | <code>[float](#float)</code> | Budget in GB.

(backends-empty-device-cache)=
#### `empty_device_cache`

```python
empty_device_cache() -> None
```

Release cached device memory. No-op without torch or a GPU.

(backends-gb-to-bytes)=
#### `gb_to_bytes`

```python
gb_to_bytes(gb: float) -> int
```

Convert a GB budget to bytes — the package's one GB↔bytes conversion.

(backends-is-oom-error)=
#### `is_oom_error`

```python
is_oom_error(exc: BaseException) -> bool
```

True if `exc` is a device out-of-memory error (CUDA or MPS).

(backends-resolve-backend)=
#### `resolve_backend`

```python
resolve_backend(parallel)
```

Coerce a backend specifier into a `Backend` instance.

Accepts the values callers typically thread through the algorithms
package (``None``/``"cpu"`` → numpy, ``"gpu"``/``"torch"`` → torch,
``"numpy"``/``"auto"`` → their direct `Backend` constructors).
Existing `Backend` instances are returned unchanged — this is
the main reason to prefer ``resolve_backend`` over constructing a new
``Backend(...)`` at each call site: it avoids repeated device
detection/torch imports when a backend has already been chosen upstream.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`parallel` |  | Backend specifier. One of:<br>- ``None`` or ``"cpu"``: numpy backend. - ``"numpy"``, ``"torch"``, ``"auto"``: forwarded to ``Backend(...)``. - ``"gpu"``: alias for ``"torch"`` (auto-detects cuda/mps/cpu). - An existing `Backend` instance (returned as-is). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Backend` |  | Resolved backend instance.

