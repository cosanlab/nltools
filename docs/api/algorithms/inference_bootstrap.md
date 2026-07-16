(algorithms-inference-bootstrap-bootstrap)=
## `bootstrap`

Bootstrap inference utilities with CPU/GPU support.

**Classes:**

Name | Description
---- | -----------
`OnlineBootstrapStats` | Memory-efficient online statistics aggregator for bootstrap samples.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`FITTED_METHODS` |  | 
`SIMPLE_METHODS` |  | 

##### Methods

(algorithms-inference-bootstrap-get-results)=
###### `get_results`

```python
get_results() -> dict[str, np.ndarray]
```

Compute final bootstrap statistics.

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | Dictionary containing:
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'mean': Bootstrap mean
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'std': Bootstrap standard deviation
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'Z': Z-scores (mean/std)
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'p': Two-tailed p-values
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'ci_lower': Lower confidence bound
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'ci_upper': Upper confidence bound
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'samples': All samples (only if save_samples=True)

Examples:
**Basic usage:**
>>> stats = OnlineBootstrapStats(shape=(100,), save_samples=False)
>>> for i in range(1000):
...     sample = np.random.randn(100)
...     stats.update(sample)
>>> results = stats.get_results()
>>> print(results.keys())
dict_keys(['mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper'])

**Usage:**
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

(algorithms-inference-bootstrap-update)=
###### `update`

```python
update(sample: np.ndarray) -> None
```

Update statistics with a new bootstrap sample.

Uses Welford's algorithm for numerical stability.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sample` | <code>[ndarray](#numpy.ndarray)</code> | New bootstrap sample with shape matching self.shape. | *required*



### Methods