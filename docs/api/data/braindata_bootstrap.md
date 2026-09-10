---
title: data.braindata.bootstrap
label: page-data-braindata-bootstrap
---

```python
bootstrap(bd, statistic, *, X = None, X_test = None, n_samples = 5000, confidence_level = 0.95, device = 'cpu', memory_budget_gb = None, return_samples = False, n_jobs = -1, random_state = None, progress_bar = False)
```

Bootstrap a statistic and its uncertainty, on CPU workers or a GPU.

Resamples observations with replacement and aggregates the replicates as
they complete, into a running Welford variance plus just enough retained
order statistics per output element to reproduce the exact percentile
interval. What the run holds is that retained tail — about
`(1 - confidence_level)` of the replicates per element — plus one dispatch
window, rather than all `n_samples` maps. This is memory-efficient, not
constant-memory: the tail still grows with `n_samples`, and
`return_samples=True` keeps the whole distribution.

A Ridge bootstrap resamples the explicitly supplied training `X` together
with `bd.data`, using the same row indices for every feature space, and
refits with the fitted model's selected `alpha_` — and, for a banded model,
its `feature_space_weights_` — held fixed. It never reruns cross-validation
or the banded random search.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data to resample. | *required*
`statistic` | <code>str</code> | Statistic to bootstrap. Basic aggregates: ``'mean'``, ``'median'``, ``'std'``, ``'sum'``, ``'min'``, ``'max'``. Model statistics (require a fitted `Ridge`): ``'weights'`` or ``'predict'``. | *required*
`X` | <code>ndarray \| Mapping[str, ndarray] \| None</code> | Training features in their original row order, required by both Ridge statistics and rejected by the basic ones. A matrix for ordinary Ridge; a mapping with exactly the fitted feature-space names for banded Ridge. | <code>None</code>
`X_test` | <code>ndarray \| Mapping[str, ndarray] \| None</code> | Evaluation features for ``statistic='predict'``, in the same structure as `X`. It may have any row count. | <code>None</code>
`n_samples` | <code>int</code> | Number of bootstrap replicates, at least two. Default ``5000``. | <code>5000</code>
`confidence_level` | <code>float</code> | Confidence level of the reported interval, strictly between zero and one. Default ``0.95``. | <code>0.95</code>
`device` | <code>str</code> | Compute device for Ridge refits: ``'cpu'`` (default) or ``'gpu'`` (PyTorch on CUDA/MPS; raises if neither is available). Basic statistics reject ``'gpu'``. | <code>'cpu'</code>
`memory_budget_gb` | <code>float \| None</code> | Working-memory budget in GB. It governs the output preflight and CPU-worker planning for every statistic, and GPU batch sizing for the Ridge ones. ``None`` (default) measures the device. | <code>None</code>
`return_samples` | <code>bool</code> | Retain and return every replicate. Default ``False``. It changes retention only, never interval semantics. | <code>False</code>
`n_jobs` | <code>int</code> | CPU worker ceiling. Default ``-1`` (all cores); the planner may use fewer. | <code>-1</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | Show a progress bar. Default ``False``. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[BootstrapResult](#data-results-bootstrapresult)</code> | `estimate` (the statistic on the unresampled full     sample), `standard_error`, `ci_lower` and `ci_upper` as `BrainData`     maps of identical shape, plus `samples` as a NumPy array with the     bootstrap axis first when ``return_samples=True``.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If `statistic` is unknown, a basic statistic is given `X`, `X_test`, or ``device='gpu'``, a Ridge statistic is missing `X` (or `X_test` for ``'predict'``), the fitted model is not a `Ridge`, `X` does not match the fitted feature structure and observation count, an argument is out of range, or the retained output cannot fit the memory budget.

**Examples:**

```python
boot = brain.bootstrap('mean', n_samples=1000)
boot.estimate.plot()

brain.fit(model='ridge', X=features, ridge_alpha=1.0)
boot = brain.bootstrap('weights', X=features, n_samples=1000)
```

<details class="note" open markdown="1">
<summary>Note</summary>

This is an IID row bootstrap: rows must be exchangeable for the
interval to mean anything. Fitting retains no hidden copy of the
training features, so omitting `X` raises even when the same features
were supplied to `fit`.

</details>
