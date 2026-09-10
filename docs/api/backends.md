---
title: algorithms.backends
label: page-backends
---

Backend abstraction for CPU/GPU operations.

Supports NumPy (CPU-only) and PyTorch (CPU/CUDA/MPS) backends for linear algebra
operations, so algorithms are written once against NumPy semantics and run on a
GPU when one is available.

This module is also the package's single GPU execution layer: memory budgets
(`device_memory_budget`), batch sizing (`auto_batch_size`), out-of-memory
recovery (`compute_oom_safe`), and CPU worker sizing (`auto_n_jobs_for_arrays`)
live only here. Algorithms supply per-item working-set estimates and never do
their own budget math.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`BATCH_WORKING_SET_CEILING_GB` |  | Saturation ceiling for batch sizing, in GB of per-batch working set.



**Classes:**

Name | Description
---- | -----------
[`Backend`](#backends-backend) | Backend abstraction for numerical operations.

**Functions:**

Name | Description
---- | -----------
[`assert_array_almost_equal`](#backends-assert-array-almost-equal) | Assert two arrays are almost equal, relaxing precision for the MPS backend.
[`auto_batch_size`](#backends-auto-batch-size) | Split `n_items` into batches that fit a memory budget.
[`auto_n_jobs_for_arrays`](#backends-auto-n-jobs-for-arrays) | Memory-aware joblib worker count for a per-item map over arrays.
[`auto_select_backend`](#backends-auto-select-backend) | Select a backend from the problem size.
[`bootstrap_memory_preflight`](#backends-bootstrap-memory-preflight) | Raise before any resampling if the retained output cannot fit the budget.
[`bootstrap_n_jobs_cpu`](#backends-bootstrap-n-jobs-cpu) | CPU worker count for a bootstrap run, capped by `n_jobs` and by memory.
[`bootstrap_output_bytes`](#backends-bootstrap-output-bytes) | Bytes a bootstrap run must hold for its retained output.
[`bootstrap_replicate_window`](#backends-bootstrap-replicate-window) | Replicates a CPU bootstrap may hold in flight before it must aggregate.
[`bootstrap_retained_tail_size`](#backends-bootstrap-retained-tail-size) | Per-element retained tail size for a streaming percentile interval.
[`check_gpu_available`](#backends-check-gpu-available) | Check whether GPU acceleration is available.
[`compute_oom_safe`](#backends-compute-oom-safe) | Run `fn(*arrays)` with reactive out-of-memory recovery.
[`device_memory_budget`](#backends-device-memory-budget) | Usable memory budget in GB for a backend's device.
[`empty_device_cache`](#backends-empty-device-cache) | Release cached device memory.
[`gb_to_bytes`](#backends-gb-to-bytes) | Convert a GB budget to bytes — the package's one GB↔bytes conversion.
[`is_oom_error`](#backends-is-oom-error) | True if `exc` is a device out-of-memory error (CUDA or MPS).
[`resolve_backend`](#backends-resolve-backend) | Coerce a backend specifier into a `Backend` instance.
[`ridge_bootstrap_batch_size`](#backends-ridge-bootstrap-batch-size) | Size a Ridge-bootstrap batch against a memory budget.

## Classes

(backends-backend)=
### `Backend`

```python
Backend(backend: str = 'numpy')
```

Backend abstraction for numerical operations.

Provides a unified interface for NumPy and PyTorch operations, enabling
transparent GPU acceleration when available.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`backend` | <code>str</code> | Backend type. `'numpy'` is CPU-only NumPy; `'torch'` is PyTorch with automatic device detection (cuda, then mps, then cpu); `'auto'` picks `'torch'` when PyTorch is installed and `'numpy'` otherwise. Defaults to `'numpy'`. | <code>'numpy'</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`name` | <code>str</code> | Backend identifier: `'numpy'`, `'torch-cpu'`, `'torch-cuda'`, or `'torch-mps'`.
`device` | <code>str</code> | Device type: `'cpu'`, `'cuda'`, or `'mps'`.
`xp` | <code>module</code> | Array library module (`numpy` or `torch`).
`is_gpu` | <code>bool</code> | True when the device is a GPU (`'cuda'` or `'mps'`).

**Methods:**

Name | Description
---- | -----------
[`asarray`](#backends-asarray) | Convert input to a backend array.
[`asarray_like`](#backends-asarray-like) | Convert `x` to an array matching `ref`'s dtype (and device for torch).
[`check_arrays`](#backends-check-arrays) | Coerce all inputs to the same dtype (and device) as the first.
[`concatenate`](#backends-concatenate) | Concatenate arrays along an axis.
[`copy`](#backends-copy) | Return an independent copy of the array.
[`dtype_to_str`](#backends-dtype-to-str) | Normalize a dtype (numpy, torch, or string) to its string name.
[`expand_dims`](#backends-expand-dims) | Insert a new axis of length one.
[`flatnonzero`](#backends-flatnonzero) | Return indices of the non-zero elements of the flattened array.
[`full`](#backends-full) | Create an array filled with `fill_value` on the backend's device.
[`full_like`](#backends-full-like) | Create an array filled with `fill_value`, optionally with a different shape.
[`matmul`](#backends-matmul) | Matrix multiplication.
[`ones_like`](#backends-ones-like) | Create an array of ones, optionally with a different shape.
[`sort`](#backends-sort) | Sort along an axis, returning values only.
[`svd`](#backends-svd) | Compute the singular value decomposition `X = U @ diag(s) @ Vt`.
[`to_cpu`](#backends-to-cpu) | Transfer an array to the CPU.
[`to_device`](#backends-to-device) | Transfer an array to the backend device as float32.
[`to_gpu`](#backends-to-gpu) | Transfer an array to the GPU.
[`to_numpy`](#backends-to-numpy) | Convert an array back to NumPy.
[`zeros_like`](#backends-zeros-like) | Create an array of zeros, optionally with a different shape.

#### Methods

(backends-asarray)=
##### `asarray`

```python
asarray(x, dtype = None, device = None)
```

Convert input to a backend array.

Handles numpy arrays, lists, and torch tensors. Places the result on
the backend's device (or an explicit `device`). On MPS a float64 dtype
is replaced by float32, which is all the device supports.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` | <code>array - like \| Tensor \| list</code> | Input data. | *required*
`dtype` | <code>str \| dtype \| dtype \| None</code> | Desired dtype. If None, inferred from the input. | <code>None</code>
`device` | <code>str \| device \| None</code> | Target device (e.g. `"cpu"`, `"cuda"`). Ignored for the numpy backend. If None, uses the backend's default device. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray \| Tensor</code> | Backend array.

(backends-asarray-like)=
##### `asarray_like`

```python
asarray_like(x, ref)
```

Convert `x` to an array matching `ref`'s dtype (and device for torch).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` | <code>array - like \| Tensor</code> | Input data. | *required*
`ref` | <code>ndarray \| Tensor</code> | Reference array whose dtype and device to match. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray \| Tensor</code> | Backend array with the same dtype and device as `ref`.

(backends-check-arrays)=
##### `check_arrays`

```python
check_arrays(*inputs)
```

Coerce all inputs to the same dtype (and device) as the first.

None values are passed through. Lists of arrays are converted
element-wise.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`*inputs` | <code>array - like \| list \| None</code> | Arrays, lists of arrays, or None. | <code>()</code>

**Returns:**

Type | Description
---- | -----------
<code>list</code> | Converted arrays in the same order as the inputs.

(backends-concatenate)=
##### `concatenate`

```python
concatenate(arrays, axis = 0)
```

Concatenate arrays along an axis.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`arrays` | <code>Sequence[ndarray \| Tensor]</code> | Arrays to join. | *required*
`axis` | <code>int</code> | Axis to concatenate along. Defaults to 0. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray \| Tensor</code> | Concatenated array.

(backends-copy)=
##### `copy`

```python
copy(array)
```

Return an independent copy of the array.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` | <code>ndarray \| Tensor</code> | Input array. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray \| Tensor</code> | The copy.

(backends-dtype-to-str)=
##### `dtype_to_str`

```python
dtype_to_str(dtype)
```

Normalize a dtype (numpy, torch, or string) to its string name.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dtype` | <code>str \| dtype \| type \| dtype \| None</code> | Data type to convert. | *required*

**Returns:**

Type | Description
---- | -----------
<code>str \| None</code> | The dtype name (e.g. `"float32"`, `"float64"`), or None if     the input was None.

**Raises:**

Type | Description
---- | -----------
<code>NotImplementedError</code> | If the input cannot be interpreted as a dtype.

(backends-expand-dims)=
##### `expand_dims`

```python
expand_dims(array, axis)
```

Insert a new axis of length one.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` | <code>ndarray \| Tensor</code> | Input array. | *required*
`axis` | <code>int</code> | Position of the new axis. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray \| Tensor</code> | View with the added axis.

(backends-flatnonzero)=
##### `flatnonzero`

```python
flatnonzero(array)
```

Return indices of the non-zero elements of the flattened array.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` | <code>ndarray \| Tensor</code> | Input array. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray \| Tensor</code> | 1D integer indices.

(backends-full)=
##### `full`

```python
full(shape, fill_value, dtype = None)
```

Create an array filled with `fill_value` on the backend's device.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`shape` | <code>int \| tuple[int, ...]</code> | Output shape. | *required*
`fill_value` | <code>float \| int \| bool</code> | Scalar fill value. | *required*
`dtype` | <code>dtype \| str \| dtype \| None</code> | Output dtype. If None, inferred by the backend from `fill_value`. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray \| Tensor</code> | Filled array.

(backends-full-like)=
##### `full_like`

```python
full_like(array, fill_value, shape = None, dtype = None, device = None)
```

Create an array filled with `fill_value`, optionally with a different shape.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` | <code>ndarray \| Tensor</code> | Reference array for dtype (and device) inference. | *required*
`fill_value` | <code>float \| int \| bool</code> | Scalar fill value. | *required*
`shape` | <code>int \| tuple[int, ...] \| None</code> | Output shape. If None, uses `array.shape`. | <code>None</code>
`dtype` | <code>dtype \| str \| dtype \| None</code> | Output dtype. If None, uses `array.dtype`. | <code>None</code>
`device` | <code>str \| device \| None</code> | Target device (torch only). If None, uses the reference array's device. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray \| Tensor</code> | Filled array.

(backends-matmul)=
##### `matmul`

```python
matmul(A, B)
```

Matrix multiplication.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`A` | <code>ndarray \| Tensor</code> | First matrix. | *required*
`B` | <code>ndarray \| Tensor</code> | Second matrix. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray \| Tensor</code> | `A @ B`.

(backends-ones-like)=
##### `ones_like`

```python
ones_like(array, shape = None, dtype = None, device = None)
```

Create an array of ones, optionally with a different shape.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` | <code>ndarray \| Tensor</code> | Reference array for dtype (and device) inference. | *required*
`shape` | <code>int \| tuple[int, ...] \| None</code> | Output shape. If None, uses `array.shape`. | <code>None</code>
`dtype` | <code>dtype \| str \| dtype \| None</code> | Output dtype. If None, uses `array.dtype`. | <code>None</code>
`device` | <code>str \| device \| None</code> | Target device (torch only). If None, uses the reference array's device. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray \| Tensor</code> | One-filled array.

(backends-sort)=
##### `sort`

```python
sort(array, axis = -1)
```

Sort along an axis, returning values only.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` | <code>ndarray \| Tensor</code> | Input array. | *required*
`axis` | <code>int</code> | Axis to sort along. Defaults to -1. | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray \| Tensor</code> | Sorted values.

(backends-svd)=
##### `svd`

```python
svd(X, full_matrices = False)
```

Compute the singular value decomposition `X = U @ diag(s) @ Vt`.

The numpy backend also accepts a 3D stack of matrices (SVD of each,
results stacked along axis 0). MPS devices compute the SVD in float64 on
the CPU and return float32 results on the device.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray \| Tensor</code> | Input matrix of shape (n_samples, n_features). | *required*
`full_matrices` | <code>bool</code> | If False, return the reduced SVD. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>tuple[ndarray \| Tensor, ...]</code> | `(U, s, Vt)` — the left singular     vectors, singular values, and transposed right singular vectors.

(backends-to-cpu)=
##### `to_cpu`

```python
to_cpu(array)
```

Transfer an array to the CPU.

No-op for the numpy backend.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` | <code>ndarray \| Tensor</code> | Input array or tensor. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray \| Tensor</code> | Array on the CPU.

(backends-to-device)=
##### `to_device`

```python
to_device(arr: np.ndarray)
```

Transfer an array to the backend device as float32.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`arr` | <code>ndarray</code> | Input numpy array. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray \| Tensor</code> | The array on the device (a numpy array for     the numpy backend, a tensor for torch backends).

(backends-to-gpu)=
##### `to_gpu`

```python
to_gpu(array, device = None)
```

Transfer an array to the GPU.

No-op for the numpy backend.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` | <code>ndarray \| Tensor</code> | Input array or tensor. | *required*
`device` | <code>str \| device \| None</code> | Target device. Defaults to the backend's device. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray \| Tensor</code> | Tensor on the device (the input unchanged     for the numpy backend).

(backends-to-numpy)=
##### `to_numpy`

```python
to_numpy(arr)
```

Convert an array back to NumPy.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`arr` | <code>ndarray \| Tensor</code> | Array to convert. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | The input as a NumPy array.

(backends-zeros-like)=
##### `zeros_like`

```python
zeros_like(array, shape = None, dtype = None, device = None)
```

Create an array of zeros, optionally with a different shape.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` | <code>ndarray \| Tensor</code> | Reference array for dtype (and device) inference. | *required*
`shape` | <code>int \| tuple[int, ...] \| None</code> | Output shape. If None, uses `array.shape`. | <code>None</code>
`dtype` | <code>dtype \| str \| dtype \| None</code> | Output dtype. If None, uses `array.dtype`. | <code>None</code>
`device` | <code>str \| device \| None</code> | Target device (torch only). If None, uses the reference array's device. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray \| Tensor</code> | Zero-filled array.



## Functions

(backends-assert-array-almost-equal)=
### `assert_array_almost_equal`

```python
assert_array_almost_equal(x, y, decimal = 6, err_msg = '', verbose = True, backend = None)
```

Assert two arrays are almost equal, relaxing precision for the MPS backend.

A test helper: on `torch-mps` (float32 only) `decimal` is capped at 2 with a
warning, so the same assertion holds across backends. Torch tensors are
moved to the CPU and converted before comparison.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` | <code>ndarray \| Tensor</code> | First array to compare. | *required*
`y` | <code>ndarray \| Tensor</code> | Second array to compare. | *required*
`decimal` | <code>int</code> | Desired decimal precision. Defaults to 6. | <code>6</code>
`err_msg` | <code>str</code> | Error message prefix. Defaults to `""`. | <code>''</code>
`verbose` | <code>bool</code> | Whether to include the mismatching values in the error. Defaults to True. | <code>True</code>
`backend` | <code>[Backend](#backends-backend) \| None</code> | Backend the arrays came from. If None, an MPS tensor is detected from `x`. | <code>None</code>

**Raises:**

Type | Description
---- | -----------
<code>AssertionError</code> | If the arrays don't match.

(backends-auto-batch-size)=
### `auto_batch_size`

```python
auto_batch_size(n_items: int, bytes_per_item: float, *, budget_gb: float, overhead: float = 1.0) -> tuple[int, int]
```

Split `n_items` into batches that fit a memory budget.

The one batch calculator for the package. Callers supply only the
per-item working-set estimate (`bytes_per_item`) and an algorithm's
allocation `overhead` factor; the clamp/ceil policy lives here.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_items` | <code>int</code> | Total number of items (permutations, targets, ...). | *required*
`bytes_per_item` | <code>float</code> | Dominant working-set size of one item in bytes. | *required*
`budget_gb` | <code>float</code> | Memory budget from `device_memory_budget`. | *required*
`overhead` | <code>float</code> | Multiplier for intermediate allocations (e.g. 3.0 when the computation holds ~3x the input working set). Defaults to 1.0. | <code>1.0</code>

**Returns:**

Type | Description
---- | -----------
<code>tuple[int, int]</code> | `(batch_size, n_batches)` with     `batch_size * n_batches >= n_items`.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If the inputs are invalid or one item exceeds the budget.

(backends-auto-n-jobs-for-arrays)=
### `auto_n_jobs_for_arrays`

```python
auto_n_jobs_for_arrays(arrays, *, max_memory_gb: float | None = None) -> int
```

Memory-aware joblib worker count for a per-item map over arrays.

Sizes workers by the largest item (each worker pickles its item), using
the same measured budget as the device batching layer. None entries are
ignored; an empty list returns one worker.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`arrays` | <code>Iterable[ndarray \| None]</code> | Arrays to map over (None entries allowed). | *required*
`max_memory_gb` | <code>float \| None</code> | Explicit memory budget in GB. None (default) measures available system RAM with headroom via `device_memory_budget`. | <code>None</code>

Returns:
    int: Worker count for `joblib.Parallel(n_jobs=...)`.

(backends-auto-select-backend)=
### `auto_select_backend`

```python
auto_select_backend(n_samples: int, n_features: int, cv: int = 1) -> Backend
```

Select a backend from the problem size.

Small problems stay on NumPy to avoid GPU transfer overhead; large problems
prefer the GPU when one is available. The effective size is
`n_samples * n_features * cv`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_samples` | <code>int</code> | Number of samples in the dataset. | *required*
`n_features` | <code>int</code> | Number of features in the dataset. | *required*
`cv` | <code>int</code> | Number of cross-validation folds, which multiplies the effective size. Defaults to 1. | <code>1</code>

**Returns:**

Type | Description
---- | -----------
<code>[Backend](#backends-backend)</code> | The selected backend.

<details class="note" open markdown="1">
<summary>Note</summary>

Below 10M elements the numpy backend is returned. Above 30M elements the
torch backend is returned when a GPU is available, as it is for any
cross-validated problem (`cv > 1`) with a GPU. Everything else falls to
`Backend('auto')`.

</details>

(backends-bootstrap-memory-preflight)=
### `bootstrap_memory_preflight`

```python
bootstrap_memory_preflight(output_shape: tuple[int, ...], n_samples: int, *, confidence_level: float, return_samples: bool, n_workers: int = 1, memory_budget_gb: float | None = None, backend: Backend | None = None) -> float
```

Raise before any resampling if the retained output cannot fit the budget.

The package's one bootstrap memory gate. It runs against the measured
budget when `memory_budget_gb` is None, so a run that would otherwise die
part-way through fails immediately and says what it needed. It never
weakens the interval, reduces `n_samples`, or disables `return_samples`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`output_shape` | <code>tuple[int, ...]</code> | Shape of one replicate's output. | *required*
`n_samples` | <code>int</code> | Number of bootstrap replicates. | *required*
`confidence_level` | <code>float</code> | Interval confidence level. | *required*
`return_samples` | <code>bool</code> | Whether the complete distribution is retained. | *required*
`n_workers` | <code>int</code> | Planned CPU worker count, which sets the dispatch window. Defaults to 1. | <code>1</code>
`memory_budget_gb` | <code>float \| None</code> | Explicit budget in GB, or None to measure the device. | <code>None</code>
`backend` | <code>[Backend](#backends-backend) \| None</code> | Resolved backend whose device is measured when `memory_budget_gb` is None. None means the CPU. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>float</code> | The required storage in GB.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If the required storage exceeds the budget.

(backends-bootstrap-n-jobs-cpu)=
### `bootstrap_n_jobs_cpu`

```python
bootstrap_n_jobs_cpu(data_size_mb: float, n_samples: int, *, memory_budget_gb: float | None = None, n_jobs: int = -1) -> int
```

CPU worker count for a bootstrap run, capped by `n_jobs` and by memory.

`n_jobs` is the ceiling the caller asked for; this planner may return
fewer when the per-worker copy of the data would not fit the budget. It
never returns zero, because the preflight — not the worker planner — is
where a run that cannot fit is refused.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data_size_mb` | <code>float</code> | Size of the array each worker pickles, in MB. | *required*
`n_samples` | <code>int</code> | Number of bootstrap replicates. | *required*
`memory_budget_gb` | <code>float \| None</code> | Explicit budget in GB, or None to measure available system memory. | <code>None</code>
`n_jobs` | <code>int</code> | Worker ceiling, with joblib's negative convention (`-1` = all cores). | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
<code>int</code> | Worker count for `joblib.Parallel(n_jobs=...)`.

(backends-bootstrap-output-bytes)=
### `bootstrap_output_bytes`

```python
bootstrap_output_bytes(output_shape: tuple[int, ...], n_samples: int, *, confidence_level: float, return_samples: bool, n_workers: int = 1) -> int
```

Bytes a bootstrap run must hold for its retained output.

Charges eight bytes for every output-sized array a run holds at once: the
two bounded tails, the replicates buffered before the next flush and the
two temporaries that flush builds, one dispatch window of in-flight
replicates, every replicate when `return_samples=True`, and the two Welford
accumulators plus the four summary payloads.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`output_shape` | <code>tuple[int, ...]</code> | Shape of one replicate's output. | *required*
`n_samples` | <code>int</code> | Number of bootstrap replicates. | *required*
`confidence_level` | <code>float</code> | Interval confidence level, which sets the retained tail size. | *required*
`return_samples` | <code>bool</code> | Whether the complete distribution is retained. | *required*
`n_workers` | <code>int</code> | Planned CPU worker count, which sets the dispatch window. Defaults to 1 (the GPU driver budgets its own batch through `ridge_bootstrap_batch_size` instead). | <code>1</code>

**Returns:**

Type | Description
---- | -----------
<code>int</code> | Required bytes.

(backends-bootstrap-replicate-window)=
### `bootstrap_replicate_window`

```python
bootstrap_replicate_window(n_samples: int, *, n_workers: int = 1) -> int
```

Replicates a CPU bootstrap may hold in flight before it must aggregate.

`joblib.Parallel` dispatches eagerly and queues finished results, so neither
`pre_dispatch` nor `return_as="generator"` bounds how many replicate arrays
are alive at once. The engines therefore dispatch in windows of this size
and fold each window into the accumulator before opening the next. The
window scales with the worker count — enough to keep every worker busy —
and never with `n_samples`, which is what makes peak memory independent of
the replicate count.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_samples` | <code>int</code> | Total number of bootstrap replicates. | *required*
`n_workers` | <code>int</code> | Planned CPU worker count. | <code>1</code>

**Returns:**

Type | Description
---- | -----------
<code>int</code> | Replicates per dispatch window, at least one.

**Examples:**

```python
bootstrap_replicate_window(5000, n_workers=4)  # → 256
bootstrap_replicate_window(50, n_workers=4)  # → 50
```

(backends-bootstrap-retained-tail-size)=
### `bootstrap_retained_tail_size`

```python
bootstrap_retained_tail_size(n_samples: int, *, confidence_level: float) -> int
```

Per-element retained tail size for a streaming percentile interval.

The streaming accumulator reproduces the complete-distribution percentile
interval by keeping this many of the smallest and largest values seen for
each output element:

```text
k = ceil((B - 1) * (1 - c) / 2) + 1
```

That is exactly the number of order statistics NumPy's linear interpolation
can reach at either end, so nothing the interval needs is discarded.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_samples` | <code>int</code> | Number of bootstrap replicates, `B`. | *required*
`confidence_level` | <code>float</code> | Interval confidence level, `c`, in `(0, 1)`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>int</code> | Values retained per element at each end, never more than     `n_samples`.

**Examples:**

```python
bootstrap_retained_tail_size(1000, confidence_level=0.95)  # → 26
```

(backends-check-gpu-available)=
### `check_gpu_available`

```python
check_gpu_available() -> tuple[bool, dict[str, Any]]
```

Check whether GPU acceleration is available.

**Returns:**

Type | Description
---- | -----------
<code>tuple[bool, dict[str, Any]]</code> | `(available, info)`. `available` is True when     a CUDA or MPS device is usable. `info` has keys `'backend'`     (`'torch'` or `'numpy'`), `'device'` (`'cpu'`, `'cuda'`, or `'mps'`),     and `'device_name'` (human-readable device name).

(backends-compute-oom-safe)=
### `compute_oom_safe`

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
`fn` | <code>Callable..., [ndarray]</code> | Maps the arrays to a numpy result (axis-0 aligned). | *required*
`*arrays` | <code>ndarray</code> | Input arrays sharing their axis-0 length. | <code>()</code>
`min_chunk` | <code>int</code> | Chunk size below which an OOM is considered fatal. Defaults to 1. | <code>1</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | `fn`'s result, possibly assembled from retried chunks.

**Raises:**

Type | Description
---- | -----------
<code>MemoryError</code> | If the device OOMs even at `min_chunk` items.

(backends-device-memory-budget)=
### `device_memory_budget`

```python
device_memory_budget(backend: Backend | None = None, max_gpu_memory_gb: float | None = None, *, cap_for_batching: bool = False) -> float
```

Usable memory budget in GB for a backend's device.

An explicit `max_gpu_memory_gb` always wins, uncapped. Otherwise the
budget is measured at call time: free CUDA memory (with headroom) on
CUDA devices; available system RAM (with headroom) for CPU and MPS,
which share unified/system memory. When nothing can be measured the
conservative 4 GB fallback applies.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`backend` | <code>[Backend](#backends-backend) \| None</code> | Resolved backend whose device the work runs on. None is treated as CPU. | <code>None</code>
`max_gpu_memory_gb` | <code>float \| None</code> | Explicit budget override in GB. Must be positive. | <code>None</code>
`cap_for_batching` | <code>bool</code> | Pass True when the budget sizes batches — a *measured* budget is then capped at `BATCH_WORKING_SET_CEILING_GB`, because working sets beyond the saturation ceiling add allocation cost without throughput gain and starve unified-memory hosts. Never applied to an explicit `max_gpu_memory_gb`; capacity queries (the default) stay uncapped. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>float</code> | Budget in GB.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If `max_gpu_memory_gb` is not positive.

(backends-empty-device-cache)=
### `empty_device_cache`

```python
empty_device_cache() -> None
```

Release cached device memory.

No-op without torch or a GPU.

(backends-gb-to-bytes)=
### `gb_to_bytes`

```python
gb_to_bytes(gb: float) -> int
```

Convert a GB budget to bytes — the package's one GB↔bytes conversion.

(backends-is-oom-error)=
### `is_oom_error`

```python
is_oom_error(exc: BaseException) -> bool
```

True if `exc` is a device out-of-memory error (CUDA or MPS).

(backends-resolve-backend)=
### `resolve_backend`

```python
resolve_backend(parallel)
```

Coerce a backend specifier into a `Backend` instance.

Accepts the values callers thread through the algorithms package. An
existing `Backend` is returned unchanged, which is the reason to prefer
this over constructing `Backend(...)` at each call site: device detection
and the torch import happen once, upstream.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`parallel` | <code>str \| [Backend](#backends-backend) \| None</code> | Backend specifier. `None` or `"cpu"` gives the numpy backend; `"gpu"` requires CUDA or MPS; `"numpy"`, `"torch"`, and `"auto"` are passed to `Backend(...)`; a `Backend` instance is returned as-is. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Backend](#backends-backend)</code> | Resolved backend instance.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If `parallel` is a string outside the accepted set.
<code>RuntimeError</code> | If `parallel="gpu"` and no accelerator is available.

(backends-ridge-bootstrap-batch-size)=
### `ridge_bootstrap_batch_size`

```python
ridge_bootstrap_batch_size(n_bootstrap: int, *, n_samples: int, n_features: int, n_targets: int, output_shape: tuple[int, ...], device_itemsize: int = 4, max_gpu_memory_gb: float | None = None, backend: float | None = None) -> tuple[int, int]
```

Size a Ridge-bootstrap batch against a memory budget.

Models what a batch of replicates actually holds. Each replicate solves on
its own, so device residency is one replicate's resampled design
`(n_samples, n_features)` and response `(n_samples, n_targets)` with an
allowance for the solver's own buffers. What accumulates across a batch is
the host-side result list: `batch_size` float64 arrays of `output_shape`,
which for a prediction bootstrap is sized by an arbitrary `X_test` row count
and can dominate everything else. Both terms are charged per replicate so
the batch cannot outgrow the budget it was given.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_bootstrap` | <code>int</code> | Total number of bootstrap replicates. | *required*
`n_samples` | <code>int</code> | Observations in the training data. | *required*
`n_features` | <code>int</code> | Total feature count across all feature spaces. | *required*
`n_targets` | <code>int</code> | Number of targets (voxels). | *required*
`output_shape` | <code>tuple[int, ...]</code> | Shape of one retained replicate result. | *required*
`device_itemsize` | <code>int</code> | Bytes per element of the solver's working dtype (4 on MPS, 8 elsewhere). Defaults to 4. | <code>4</code>
`max_gpu_memory_gb` | <code>float \| None</code> | Explicit budget in GB, or None to measure the device. | <code>None</code>
`backend` | <code>[Backend](#backends-backend) \| None</code> | Resolved backend, used only to measure the budget when `max_gpu_memory_gb` is None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>tuple[int, int]</code> | `(batch_size, n_batches)`.
