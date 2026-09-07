---
title: data.braindata.bootstrap
label: page-data-braindata-bootstrap
---

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
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data to resample. | *required*
`stat` | <code>str</code> | Statistic to bootstrap. Simple aggregates: ``'mean'``, ``'median'``, ``'std'``, ``'sum'``, ``'min'``, ``'max'``. Model statistics (require a fitted Ridge model): ``'weights'``, or ``'predict'`` (also requires ``X_test``). | *required*
`n_samples` | <code>int</code> | Number of bootstrap iterations. Default ``5000``. | <code>5000</code>
`save_boots` | <code>bool</code> | Keep every bootstrap sample (memory intensive). Default ``False``. | <code>False</code>
`percentiles` | <code>tuple[float, float]</code> | Percentiles for the confidence interval. Default ``(2.5, 97.5)``. | <code>(2.5, 97.5)</code>
`X_test` | <code>ndarray \| None</code> | Test features for ``stat='predict'``. | <code>None</code>
`device` | <code>str</code> | Compute device for Ridge bootstraps: ``'cpu'`` (default), ``'gpu'`` (PyTorch on CUDA/MPS; raises if none is available), or ``'auto'`` (a GPU if present, else CPU). Ignored for simple stats. | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>float \| None</code> | Explicit GPU memory budget in GB when ``device`` is ``'gpu'`` or ``'auto'``. ``None`` (default) measures the device. | <code>None</code>
`tail` | <code>int \| str</code> | ``2``/``'two'`` for two-tailed (default); ``1``/``'one'`` for one-tailed (statistic > 0; negate the data for the other direction). | <code>2</code>
`n_jobs` | <code>int</code> | CPU workers for parallelization. Default ``-1`` (all CPUs). | <code>-1</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | Show a progress bar. Default ``False``. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data) \| dict</code> | For simple stats with ``save_boots=False``, a     `BrainData` holding the bootstrap mean. For model stats, a dict with     keys ``'mean'``, ``'std'``, ``'Z'``, ``'p'``, ``'ci_lower'``,     ``'ci_upper'`` (all `BrainData`). With ``save_boots=True``, a dict     (even for simple stats) with an added ``'samples'`` key holding the     raw sample array.

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
mean_brain = BrainData(result['mean'], mask=brain.mask)
```

<details class="note" open markdown="1">
<summary>Note</summary>

Use ``stat='mean'`` to bootstrap an aggregate; use ``stat='weights'`` or
``stat='predict'`` to also get Z and p maps. To summarize bootstrap
samples you already have, feed them to `OnlineBootstrapStats` directly
(see Examples).

</details>
