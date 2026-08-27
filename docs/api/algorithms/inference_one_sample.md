(algorithms-inference-one-sample-one-sample)=
## `one_sample`

One-sample permutation test implementations.

This module provides CPU-parallel and GPU-batched implementations
of the one-sample permutation test (sign-flipping test).

**Methods:**

Name | Description
---- | -----------
[`one_sample_permutation_test`](#algorithms-inference-one-sample-one-sample-permutation-test) | One-sample permutation test using sign-flipping.



### Classes

### Methods

(algorithms-inference-one-sample-one-sample-permutation-test)=
#### `one_sample_permutation_test`

```python
one_sample_permutation_test(data: np.ndarray, *, n_permute: int = 5000, tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None, progress_bar: bool = False) -> dict
```

One-sample permutation test using sign-flipping.

Tests whether the mean of data is significantly different from zero
by randomly flipping the sign of each observation. This is the
permutation test equivalent of a one-sample t-test.

Assumption: Symmetric error distribution around zero. For highly skewed
distributions, consider alternative methods (e.g., bootstrap resampling).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Data to test - shape (n_samples,) for single feature - shape (n_samples, n_features) for multi-feature (voxel-wise) | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000) | <code>5000</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type — 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction) - 'two' or 2: Two-tailed test (mean != 0) - 2 | 'two': Two-tailed test (mean != 0) - 1 | 'one': One-tailed (mean > 0; negate the data for the other   direction). The fixed direction keeps MCP correction valid. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Whether to display a progress bar (default: False) | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'mean' (float or np.ndarray): Observed mean(s) - 'p' (float or np.ndarray): P-value(s) - 'null_dist' (np.ndarray): Null distribution (if return_null=True) - 'device' (str): Parallelization method used

**Examples:**

```pycon
>>> # Single feature (default CPU parallelization)
>>> data = np.random.randn(30)
>>> result = one_sample_permutation_test(data, n_permute=5000)
>>> result['p']
0.23
```

```pycon
>>> # Voxel-wise test with GPU
>>> data = np.random.randn(30, 10000)  # 30 subjects, 10K voxels
>>> result = one_sample_permutation_test(data, n_permute=5000, device='gpu')
>>> result['mean'].shape
(10000,)
>>> result['p'].shape
(10000,)
```

```pycon
>>> # Single-threaded (for debugging)
>>> result = one_sample_permutation_test(data, n_permute=5000, device=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (device='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
- Single-threaded (device=None): Use for small problems or debugging
- For voxel-wise tests, each voxel tested independently
- Progress bars show completion for both CPU parallel and GPU batched modes

</details>

