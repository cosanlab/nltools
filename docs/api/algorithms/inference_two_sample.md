## `two_sample`

Two-sample permutation test implementations.

This module provides CPU-parallel and GPU-batched implementations
of the two-sample permutation test (group permutation test).

**Methods:**

Name | Description
---- | -----------
[`two_sample_permutation_test`](#two_sample_permutation_test) | Two-sample permutation test using group label shuffling.



### Classes

### Methods

#### `two_sample_permutation_test`

```python
two_sample_permutation_test(data1: np.ndarray, data2: np.ndarray, n_permute: int = 5000, tail: int | str = 2, return_null: bool = False, parallel: Optional[str] = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: Optional[int] = None) -> dict
```

Two-sample permutation test using group label shuffling.

Tests whether two independent groups have different means by randomly
permuting group labels. This is the permutation test equivalent of an
independent samples t-test.

Assumption: Exchangeability under the null hypothesis (group assignments
are arbitrary). Valid for independent samples from similar distributions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | Group 1 data - shape (n_samples1,) for single feature - shape (n_samples1, n_features) for multi-feature (voxel-wise) | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Group 2 data - shape (n_samples2,) for single feature - shape (n_samples2, n_features) for multi-feature (voxel-wise) | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000) | <code>5000</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type (default: 2) - 'two' or 2: Two-tailed test (mean1 != mean2) - 'upper' or 1: One-tailed upper (mean1 > mean2) - 'lower' or -1: One-tailed lower (mean1 < mean2) For MCP correction (FDR), use 'upper' or 'lower' for consistent direction. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`parallel` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when parallel='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Maximum GPU memory to use in GB (default: 4.0) Controls automatic batching to prevent OOM errors. Only used with parallel='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>4.0</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'mean_diff' (float or np.ndarray): Observed mean difference (data1 - data2) - 'p' (float or np.ndarray): P-value(s) - 'null_dist' (np.ndarray): Null distribution (if return_null=True) - 'parallel' (str): Parallelization method used

**Examples:**

```pycon
>>> # Single feature (default CPU parallelization)
>>> data1 = np.random.randn(20)  # Group 1: 20 subjects
>>> data2 = np.random.randn(25)  # Group 2: 25 subjects
>>> result = two_sample_permutation_test(data1, data2, n_permute=5000)
>>> result['p']
0.45
```

```pycon
>>> # Voxel-wise test with GPU
>>> data1 = np.random.randn(20, 10000)  # 20 subjects, 10K voxels
>>> data2 = np.random.randn(25, 10000)  # 25 subjects, 10K voxels
>>> result = two_sample_permutation_test(data1, data2, n_permute=5000, parallel='gpu')
>>> result['mean_diff'].shape
(10000,)
>>> result['p'].shape
(10000,)
```

```pycon
>>> # Single-threaded (for debugging)
>>> result = two_sample_permutation_test(data1, data2, n_permute=5000, parallel=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (parallel='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
- Single-threaded (parallel=None): Use for small problems or debugging
- For voxel-wise tests, each voxel tested independently
- Group sizes can be unequal

</details>

