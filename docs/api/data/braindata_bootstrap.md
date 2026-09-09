---
title: data.braindata.bootstrap
label: page-data-braindata-bootstrap
---

```python
bootstrap(bd, stat, *, X = None, X_test = None, n_samples = 5000, save_boots = False, percentiles = (2.5, 97.5), device = 'cpu', memory_budget_gb = None, tail = 2, n_jobs = -1, random_state = None, progress_bar = False)
```

Bootstrap statistics with CPU parallelization or GPU acceleration.

Supports simple aggregation statistics and fitted `Ridge` statistics. A
Ridge bootstrap resamples the explicitly supplied training `X` together
with `bd.data`, using the same row indices for every feature space, and
refits with the fitted model's selected `alpha_` — and, for a banded model,
its `feature_space_weights_` — held fixed. It never reruns cross-validation
or the banded random search.

Note: the CPU path pre-generates all resample indices and collects every
per-sample result, so peak memory grows with ``n_samples`` (it is not a
streaming/online accumulator).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data to resample. | *required*
`stat` | <code>str</code> | Statistic to bootstrap. Simple aggregates: ``'mean'``, ``'median'``, ``'std'``, ``'sum'``, ``'min'``, ``'max'``. Model statistics (require a fitted `Ridge`): ``'weights'`` or ``'predict'``. | *required*
`X` | <code>ndarray \| Mapping[str, ndarray] \| None</code> | Training features in their original row order, required by both Ridge statistics and rejected by the simple ones. A matrix for ordinary Ridge; a mapping with exactly the fitted feature-space names for banded Ridge. | <code>None</code>
`X_test` | <code>ndarray \| Mapping[str, ndarray] \| None</code> | Evaluation features for ``stat='predict'``, in the same structure as `X`. It may have any row count. | <code>None</code>
`n_samples` | <code>int</code> | Number of bootstrap iterations. Default ``5000``. | <code>5000</code>
`save_boots` | <code>bool</code> | Keep every bootstrap sample (memory intensive). Default ``False``. | <code>False</code>
`percentiles` | <code>tuple[float, float]</code> | Percentiles for the confidence interval. Default ``(2.5, 97.5)``. | <code>(2.5, 97.5)</code>
`device` | <code>str</code> | Compute device for Ridge bootstraps: ``'cpu'`` (default) or ``'gpu'`` (PyTorch on CUDA/MPS; raises if neither is available). | <code>'cpu'</code>
`memory_budget_gb` | <code>float \| None</code> | Working-memory budget in GB used to size GPU batches. ``None`` (default) measures the device. | <code>None</code>
`tail` | <code>int \| str</code> | ``2``/``'two'`` for two-tailed (default); ``1``/``'one'`` for one-tailed (statistic > 0; negate the data for the other direction). | <code>2</code>
`n_jobs` | <code>int</code> | CPU workers for parallelization. Default ``-1`` (all CPUs). | <code>-1</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | Show a progress bar. Default ``False``. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data) \| dict</code> | For simple stats with ``save_boots=False``, a     `BrainData` holding the bootstrap mean. For model stats, a dict with     keys ``'mean'``, ``'std'``, ``'Z'``, ``'p'``, ``'ci_lower'``,     ``'ci_upper'`` (all `BrainData`). With ``save_boots=True``, a dict     (even for simple stats) with an added ``'samples'`` key holding the     raw sample array.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If `stat` is unknown, a simple statistic is given `X` or `X_test`, a Ridge statistic is missing `X` (or `X_test` for ``'predict'``), the fitted model is not a `Ridge`, or `X` does not match the fitted feature structure and observation count.

**Examples:**

```python
# Simple aggregation: returns a BrainData holding the bootstrap mean
boot = brain.bootstrap(stat='mean', n_samples=1000)

# Ridge weights bootstrap: the training features are passed explicitly
brain.fit(model='ridge', X=features, ridge_alpha=1.0)
boot = brain.bootstrap(stat='weights', X=features, n_samples=1000)

# Ridge prediction bootstrap, GPU accelerated
boot = brain.bootstrap(
    stat='predict', X=features, X_test=X_new, n_samples=1000, device='gpu'
)
```

<details class="note" open markdown="1">
<summary>Note</summary>

Fitting retains no hidden copy of the training features, so omitting
`X` raises even when the same features were supplied to `fit`.

</details>
