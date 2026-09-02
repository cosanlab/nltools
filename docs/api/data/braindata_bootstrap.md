---
title: data.braindata.bootstrap
---

Bootstrap functions extracted from BrainData methods.

**Methods:**

Name | Description
---- | -----------
[`bootstrap`](#data-braindata-bootstrap-bootstrap) | Bootstrap statistics with CPU parallelization or GPU acceleration.
[`convert_bootstrap_results_to_brain_data`](#data-braindata-bootstrap-convert-bootstrap-results-to-brain-data) | Convert bootstrap results dictionary to BrainData format.



## Methods

(data-braindata-bootstrap-bootstrap)=
### `bootstrap`

```python
bootstrap(bd, stat, *, n_samples = 5000, save_boots = False, percentiles = (2.5, 97.5), X_test = None, device = 'cpu', max_gpu_memory_gb = None, tail = 2, n_jobs = -1, random_state = None, progress_bar = False)
```

Bootstrap statistics with CPU parallelization or GPU acceleration.

Supports simple aggregation statistics and fitted model statistics (Ridge).
Note: the CPU path pre-generates all resample indices and collects every
per-sample result, so peak memory grows with ``n_samples`` (it is not a
streaming/online accumulator).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`stat` |  | (str) Statistic to bootstrap. Options: Simple stats ('mean', 'median', 'std', 'sum', 'min', 'max') or Model stats ('weights' requires fitted Ridge model, 'predict' requires fitted Ridge model + X_test). | *required*
`n_samples` |  | (int) Number of bootstrap iterations. Default: 5000 | <code>5000</code>
`save_boots` |  | (bool) If True, store all bootstrap samples (memory intensive).        Default: False | <code>False</code>
`percentiles` |  | (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5) | <code>(2.5, 97.5)</code>
`X_test` |  | (np.ndarray, optional) Test features for 'predict' bootstrap.    Required if stat='predict' | <code>None</code>
`device` |  | (str) Compute device for Ridge bootstrap: 'cpu' (default), 'gpu' (PyTorch on CUDA/MPS if available), or 'auto' (use a GPU if present, else CPU). Ignored for simple stats. Default: 'cpu' | <code>'cpu'</code>
`max_gpu_memory_gb` |  | (float, optional) Explicit GPU memory budget in GB when device is 'gpu' or 'auto'. None (default) measures the device. | <code>None</code>
`tail` |  | `2` or `'two'` for two-tailed (default); `1` or `'one'` for one-tailed (statistic > 0; negate the data for the other direction). | <code>2</code>
`n_jobs` |  | (int) Number of CPU cores for parallelization. Default: -1 (all CPUs). | <code>-1</code>
`random_state` |  | (int, optional) Random seed for reproducibility | <code>None</code>
`progress_bar` |  | (bool) If True, show a progress bar. Default: False | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#nltools.data.braindata.BrainData) or [dict](#dict)</code> | - For simple stats (with ``save_boots=False``): Returns BrainData       with bootstrap mean     - For model stats: Returns dict with keys: 'mean', 'std', 'Z', 'p',       'ci_lower', 'ci_upper' (all BrainData objects)     - If ``save_boots=True``: Returns a dict (even for simple stats)       with an added 'samples' key holding all samples as a raw ndarray

**Examples:**

```python
# Simple aggregation: returns a BrainData holding the bootstrap mean
boot = brain.bootstrap(stat='mean', n_samples=1000)
assert isinstance(boot, BrainData)

# Ridge weights bootstrap (CPU): returns a dict of BrainData
brain.fit(X=dm, model='ridge', alpha=1.0)
boot = brain.bootstrap(stat='weights', n_samples=1000)
assert isinstance(boot['mean'], BrainData)

# Ridge weights bootstrap (GPU accelerated)
boot = brain.bootstrap(stat='weights', n_samples=1000, device='gpu')

# Ridge predict bootstrap
boot = brain.bootstrap(stat='predict', X_test=X_new, n_samples=1000)

# Summarize pre-existing bootstrap samples (a BrainData with one image per
# sample) with OnlineBootstrapStats instead
from nltools.algorithms.inference.bootstrap import OnlineBootstrapStats

stats = OnlineBootstrapStats(shape=(brain.shape[1],), save_samples=False)
for sample in bootstrap_samples:
    stats.update(sample.data)
result = stats.get_results()  # keys: mean, std, Z, p, ci_lower, ci_upper
mean_brain = shallow_copy(brain)
mean_brain.data = result['mean']
```

<details class="note" open markdown="1">
<summary>Note</summary>

This method replaces the removed `summarize_bootstrap()` function. Use
`stat='mean'` to generate bootstrap samples of an aggregate; use
`stat='weights'` or `stat='predict'` to also get Z and p maps. To summarize
bootstrap samples you already have, feed them to `OnlineBootstrapStats`
directly (see Examples).

</details>

(data-braindata-bootstrap-convert-bootstrap-results-to-brain-data)=
### `convert_bootstrap_results_to_brain_data`

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
<code>[BrainData](#nltools.data.braindata.BrainData) or [dict](#dict)</code> | - If return_dict=False and save_boots=False: Returns BrainData with mean     - Otherwise: Returns dict with BrainData objects for each statistic.       The optional 'samples' entry (when save_boots=True) is a raw       ndarray, not a BrainData.
