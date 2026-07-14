## `bootstrap`

Bootstrap functions extracted from BrainData methods.

**Methods:**

Name | Description
---- | -----------
[`bootstrap`](#bootstrap) | Bootstrap statistics using efficient online algorithms.
[`convert_bootstrap_results_to_brain_data`](#convert-bootstrap-results-to-brain-data) | Convert bootstrap results dictionary to BrainData format.



### Methods

#### `bootstrap`

```python
bootstrap(bd, stat, n_samples = 5000, save_boots = False, percentiles = (2.5, 97.5), X_test = None, backend = None, max_gpu_memory_gb = 4.0, n_jobs = -1, random_state = None)
```

Bootstrap statistics using efficient online algorithms.

Uses memory-efficient bootstrap infrastructure with CPU parallelization or GPU acceleration.
Supports simple aggregation statistics and fitted model statistics (Ridge).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`stat` |  | (str) Statistic to bootstrap. Options: Simple stats ('mean', 'median', 'std', 'sum', 'min', 'max') or Model stats ('weights' requires fitted Ridge model, 'predict' requires fitted Ridge model + X_test). | *required*
`n_samples` |  | (int) Number of bootstrap iterations. Default: 5000 | <code>5000</code>
`save_boots` |  | (bool) If True, store all bootstrap samples (memory intensive).        Default: False | <code>False</code>
`percentiles` |  | (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5) | <code>(2.5, 97.5)</code>
`X_test` |  | (np.ndarray, optional) Test features for 'predict' bootstrap.    Required if stat='predict' | <code>None</code>
`backend` |  | (str, optional) Backend for Ridge bootstrap: None (CPU), 'torch' (GPU if available), or 'auto' (auto-select). Ignored for simple stats. Default: None | <code>None</code>
`max_gpu_memory_gb` |  | (float) Maximum GPU memory to use when backend is 'torch' or 'auto'. Default: 4.0 | <code>4.0</code>
`n_jobs` |  | (int) Number of CPU cores for parallelization. Default: -1 (all CPUs). | <code>-1</code>
`random_state` |  | (int, optional) Random seed for reproducibility | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or dict: - For simple stats: Returns BrainData with bootstrap mean - For model stats: Returns dict with keys: 'mean', 'std', 'Z', 'p',   'ci_lower', 'ci_upper' (all BrainData objects) - If ``save_boots=True``: Returns dict with 'samples' key containing all samples

**Examples:**

```pycon
>>> # Simple aggregation
>>> boot = brain.bootstrap(stat='mean', n_samples=1000)
>>> assert isinstance(boot, BrainData)
```

```pycon
>>> # Ridge weights bootstrap (CPU)
>>> brain.fit(X=dm, model='ridge', alpha=1.0)
>>> boot = brain.bootstrap(stat='weights', n_samples=1000)
>>> assert 'mean' in boot
>>> assert isinstance(boot['mean'], BrainData)
```

```pycon
>>> # Ridge weights bootstrap (GPU accelerated)
>>> brain.fit(X=dm, model='ridge', alpha=1.0)
>>> boot = brain.bootstrap(stat='weights', n_samples=1000, backend='torch')
>>> assert 'mean' in boot
>>> assert isinstance(boot['mean'], BrainData)
```

```pycon
>>> # Ridge predict bootstrap
>>> brain.fit(X=dm, model='ridge', alpha=1.0)
>>> boot = brain.bootstrap(stat='predict', X_test=X_new, n_samples=1000)
>>> assert 'mean' in boot
>>> assert isinstance(boot['mean'], BrainData)
```

<details class="note" open markdown="1">
<summary>Note</summary>

This method replaces the removed `summarize_bootstrap()` function.

**New API:**
>>> # Option 1: Use BrainData.bootstrap() for generating bootstrap samples
>>> boot = brain.bootstrap(stat='mean', n_samples=1000, save_boots=False)
>>> # Returns BrainData with bootstrap mean
>>> # To get Z and p, use stat='weights' or 'predict' which returns dict

>>> # Option 2: For existing bootstrap samples (BrainData with multiple images),
>>> # use OnlineBootstrapStats directly:
>>> from nltools.algorithms.inference.bootstrap import OnlineBootstrapStats
>>> stats = OnlineBootstrapStats(shape=(brain.shape[1],), save_samples=False)
>>> for sample in bootstrap_samples:  # Iterate over samples
...     stats.update(sample.data)
>>> result = stats.get_results()
>>> # Returns: {'mean': array, 'std': array, 'Z': array, 'p': array,
>>> #           'ci_lower': array, 'ci_upper': array}
>>> # Convert to BrainData if needed:
>>> mean_brain = shallow_copy(brain)
>>> mean_brain.data = result['mean']

</details>

#### `convert_bootstrap_results_to_brain_data`

```python
convert_bootstrap_results_to_brain_data(bd, result, save_boots = False, return_dict = False)
```

Convert bootstrap results dictionary to BrainData format.

Helper method to convert numpy arrays from bootstrap functions into
BrainData objects or dicts of BrainData objects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`result` |  | (dict) Result dictionary from bootstrap function with keys:     'mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper', and optionally 'samples' | *required*
`save_boots` |  | (bool) If True, include 'samples' key in output | <code>False</code>
`return_dict` |  | (bool) If True, always return dict even for simple stats.         If False, return BrainData for simple stats (when save_boots=False) | <code>False</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or dict: - If return_dict=False and save_boots=False: Returns BrainData with mean - Otherwise: Returns dict with BrainData objects for each statistic

