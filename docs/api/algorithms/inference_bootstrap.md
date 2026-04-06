## `nltools.algorithms.inference.bootstrap`

Bootstrap inference utilities with CPU/GPU support.

**Classes:**

Name | Description
---- | -----------
[`OnlineBootstrapStats`](#nltools.algorithms.inference.bootstrap.OnlineBootstrapStats) | Memory-efficient online statistics aggregator for bootstrap samples.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`FITTED_METHODS`](#nltools.algorithms.inference.bootstrap.FITTED_METHODS) |  | 
[`SIMPLE_METHODS`](#nltools.algorithms.inference.bootstrap.SIMPLE_METHODS) |  | 



### Attributes#### `nltools.algorithms.inference.bootstrap.FITTED_METHODS`

```python
FITTED_METHODS = ['weights', 'predict']
```

#### `nltools.algorithms.inference.bootstrap.SIMPLE_METHODS`

```python
SIMPLE_METHODS = ['mean', 'median', 'std', 'sum', 'min', 'max']
```



### Classes#### `nltools.algorithms.inference.bootstrap.OnlineBootstrapStats`

```python
OnlineBootstrapStats(shape: Tuple[int, ...], save_samples: bool = False, percentiles: Tuple[float, float] = (2.5, 97.5))
```

Memory-efficient online statistics aggregator for bootstrap samples.

Uses Welford's algorithm for numerically stable online computation of
mean and variance. Optionally stores all samples for exact percentile CIs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`shape` | <code>[Tuple](#typing.Tuple)[[int](#int), ...]</code> | Shape of each bootstrap sample. | *required*
`save_samples` | <code>[bool](#bool)</code> | If True, store all samples for exact percentile confidence intervals. If False, use normal approximation (much more memory efficient). Defaults to False. | <code>False</code>
`percentiles` | <code>[Tuple](#typing.Tuple)[[float](#float), [float](#float)]</code> | Percentiles for confidence intervals (e.g., (2.5, 97.5) for 95% CI). Defaults to (2.5, 97.5). | <code>(2.5, 97.5)</code>

**Examples:**

```pycon
>>> stats = OnlineBootstrapStats(shape=(100,), save_samples=False)
>>> for i in range(1000):
...     sample = np.random.randn(100)
...     stats.update(sample)
>>> results = stats.get_results()
>>> print(results.keys())
dict_keys(['mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper'])
```

**Functions:**

Name | Description
---- | -----------
[`get_results`](#nltools.algorithms.inference.bootstrap.OnlineBootstrapStats.get_results) | Compute final bootstrap statistics.
[`update`](#nltools.algorithms.inference.bootstrap.OnlineBootstrapStats.update) | Update statistics with a new bootstrap sample.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`M2`](#nltools.algorithms.inference.bootstrap.OnlineBootstrapStats.M2) |  | 
[`mean`](#nltools.algorithms.inference.bootstrap.OnlineBootstrapStats.mean) |  | 
[`n`](#nltools.algorithms.inference.bootstrap.OnlineBootstrapStats.n) |  | 
[`percentiles`](#nltools.algorithms.inference.bootstrap.OnlineBootstrapStats.percentiles) |  | 
[`samples`](#nltools.algorithms.inference.bootstrap.OnlineBootstrapStats.samples) |  | 
[`save_samples`](#nltools.algorithms.inference.bootstrap.OnlineBootstrapStats.save_samples) |  | 
[`shape`](#nltools.algorithms.inference.bootstrap.OnlineBootstrapStats.shape) |  | 



##### Attributes###### `nltools.algorithms.inference.bootstrap.OnlineBootstrapStats.M2`

```python
M2 = np.zeros(shape, dtype=(np.float64))
```

###### `nltools.algorithms.inference.bootstrap.OnlineBootstrapStats.mean`

```python
mean = np.zeros(shape, dtype=(np.float64))
```

###### `nltools.algorithms.inference.bootstrap.OnlineBootstrapStats.n`

```python
n = 0
```

###### `nltools.algorithms.inference.bootstrap.OnlineBootstrapStats.percentiles`

```python
percentiles = percentiles
```

###### `nltools.algorithms.inference.bootstrap.OnlineBootstrapStats.samples`

```python
samples = [] if save_samples else None
```

###### `nltools.algorithms.inference.bootstrap.OnlineBootstrapStats.save_samples`

```python
save_samples = save_samples
```

###### `nltools.algorithms.inference.bootstrap.OnlineBootstrapStats.shape`

```python
shape = shape
```



##### Functions###### `nltools.algorithms.inference.bootstrap.OnlineBootstrapStats.get_results`

```python
get_results() -> Dict[str, np.ndarray]
```

Compute final bootstrap statistics.

**Returns:**

Type | Description
---- | -----------
<code>[Dict](#typing.Dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | Dictionary containing:
<code>[Dict](#typing.Dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'mean': Bootstrap mean
<code>[Dict](#typing.Dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'std': Bootstrap standard deviation
<code>[Dict](#typing.Dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'Z': Z-scores (mean/std)
<code>[Dict](#typing.Dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'p': Two-tailed p-values
<code>[Dict](#typing.Dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'ci_lower': Lower confidence bound
<code>[Dict](#typing.Dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'ci_upper': Upper confidence bound
<code>[Dict](#typing.Dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'samples': All samples (only if save_samples=True)

Examples:
**Basic usage:**
>>> stats = OnlineBootstrapStats(shape=(100,), save_samples=False)
>>> for i in range(1000):
...     sample = np.random.randn(100)
...     stats.update(sample)
>>> results = stats.get_results()
>>> print(results.keys())
dict_keys(['mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper'])

**Replacing summarize_bootstrap() from nltools.stats:**
The deprecated `summarize_bootstrap()` function can be replaced with this class:

**Old API (deprecated):**
>>> from nltools.stats import summarize_bootstrap
>>> bootstrap_samples = BrainData(list_of_samples)  # Multiple samples
>>> result = summarize_bootstrap(bootstrap_samples, save_weights=False)
>>> # Returns: {'mean': BrainData, 'Z': BrainData, 'p': BrainData}

**New API (recommended):**
>>> from nltools.algorithms.inference.bootstrap import OnlineBootstrapStats
>>> from nltools.data import BrainData
>>>
>>> # Initialize with shape matching your data
>>> stats = OnlineBootstrapStats(
...     shape=(bootstrap_samples.shape[1],),  # Number of voxels/features
...     save_samples=False,  # Set True if you need 'samples' key
...     percentiles=(2.5, 97.5)  # For confidence intervals
... )
>>>
>>> # Update with each bootstrap sample
>>> for sample in bootstrap_samples:  # Iterate over samples
...     stats.update(sample.data)  # Pass 1D array of voxel values
>>>
>>> # Get results (equivalent to summarize_bootstrap output)
>>> result = stats.get_results()
>>> # Returns: {'mean': array, 'std': array, 'Z': array, 'p': array,
>>> #           'ci_lower': array, 'ci_upper': array}
>>>
>>> # Convert to BrainData if needed (reproduce old API format)
>>> mean_brain = bootstrap_samples[0].copy()
>>> mean_brain.data = result['mean']
>>> z_brain = bootstrap_samples[0].copy()
>>> z_brain.data = result['Z']
>>> p_brain = bootstrap_samples[0].copy()
>>> p_brain.data = result['p']
>>>
>>> # Result equivalent to old summarize_bootstrap():
>>> equivalent_result = {
...     'mean': mean_brain,
...     'Z': z_brain,
...     'p': p_brain
... }
>>> # Optionally include samples if save_samples=True:
>>> if 'samples' in result:
...     equivalent_result['samples'] = result['samples']

###### `nltools.algorithms.inference.bootstrap.OnlineBootstrapStats.update`

```python
update(sample: np.ndarray) -> None
```

Update statistics with a new bootstrap sample.

Uses Welford's algorithm for numerical stability.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sample` | <code>[ndarray](#numpy.ndarray)</code> | New bootstrap sample with shape matching self.shape. | *required*



### Functions