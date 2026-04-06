## `nltools.algorithms.inference.correlation`

Correlation permutation test implementations.

This module provides CPU-parallel and GPU-batched implementations
of correlation permutation tests for assessing statistical significance
of correlations.

**Functions:**

Name | Description
---- | -----------
[`correlation_permutation_test`](#nltools.algorithms.inference.correlation.correlation_permutation_test) | Correlation permutation test.



### Attributes

### Classes

### Functions#### `nltools.algorithms.inference.correlation.correlation_permutation_test`

```python
correlation_permutation_test(data1: np.ndarray, data2: np.ndarray, n_permute: int = 5000, metric: str = 'pearson', tail: int | str = 2, return_null: bool = False, parallel: Optional[str] = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: Optional[int] = None) -> dict
```

Correlation permutation test.

Tests whether the correlation between data1 and data2 is significantly
different from zero by randomly permuting data1 and computing correlations.

Assumption: Observations are independent (i.i.d.). For autocorrelated time
series, use timeseries_correlation_permutation_test with circle_shift or
phase_randomize methods instead.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | Data to permute - shape (n_samples,) for single feature - shape (n_samples, n_features) for multi-feature | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Data to correlate with - shape (n_samples,) for single feature - shape (n_samples, n_features) for multi-feature | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000) | <code>5000</code>
`metric` | <code>[str](#str)</code> | Correlation metric (default: 'pearson') - 'pearson': Pearson correlation (linear relationships) - 'spearman': Spearman rank correlation (monotonic relationships) - 'kendall': Kendall tau rank correlation (ordinal association, robust to ties) | <code>'pearson'</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type (default: 2) - 'two' or 2: Two-tailed test (r != 0) - 'upper' or 1: One-tailed upper (r > 0, positive correlation) - 'lower' or -1: One-tailed lower (r < 0, negative correlation) For MCP correction (FDR), use 'upper' or 'lower' for consistent direction. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`parallel` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when parallel='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Maximum GPU memory to use in GB (default: 4.0) Controls automatic batching to prevent OOM errors. Only used with parallel='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>4.0</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'correlation' (float or np.ndarray): Observed correlation(s) - 'p' (float or np.ndarray): P-value(s) - 'null_dist' (np.ndarray): Null distribution (if return_null=True) - 'parallel' (str): Parallelization method used

**Examples:**

```pycon
>>> # Single feature (default CPU parallelization)
>>> x = np.random.randn(100)
>>> y = x + np.random.randn(100) * 0.5  # Correlated
>>> result = correlation_permutation_test(x, y, n_permute=5000)
>>> result['correlation']
0.85
>>> result['p']
0.001
```

```pycon
>>> # Multi-feature (2D arrays)
>>> data1 = np.random.randn(100, 10)  # 100 samples, 10 features
>>> data2 = data1 + np.random.randn(100, 10) * 0.3  # Correlated
>>> result = correlation_permutation_test(data1, data2, n_permute=5000)
>>> result['correlation'].shape
(10,)
>>> result['p'].shape
(10,)
```

```pycon
>>> # GPU acceleration
>>> result = correlation_permutation_test(data1, data2, n_permute=5000, parallel='gpu')
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (parallel='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
    - Pearson correlation: Fully vectorized across all features (5-20× speedup for multi-feature)
    - Spearman/Kendall: Only supported with parallel='cpu' or parallel=None (GPU not yet implemented)
- Single-threaded (parallel=None): Use for small problems or debugging
- For multi-feature data, each feature pair tested independently
- Kendall is O(n^2) complexity, slower than Pearson/Spearman for large samples

</details>

