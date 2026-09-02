---
title: algorithms.inference.bootstrap
---

Bootstrap inference utilities with CPU/GPU support.

**Classes:**

Name | Description
---- | -----------
[`OnlineBootstrapStats`](#algorithms-inference-bootstrap-onlinebootstrapstats) | Memory-efficient online statistics aggregator for bootstrap samples.



## Classes

(algorithms-inference-bootstrap-onlinebootstrapstats)=
### `OnlineBootstrapStats`

```python
OnlineBootstrapStats(shape: tuple[int, ...], save_samples: bool = False, percentiles: tuple[float, float] = (2.5, 97.5))
```

Memory-efficient online statistics aggregator for bootstrap samples.

Uses Welford's algorithm for numerically stable online computation of
mean and variance. Optionally stores all samples for exact percentile CIs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`shape` | <code>[tuple](#tuple)[[int](#int), ...]</code> | Shape of each bootstrap sample. | *required*
`save_samples` | <code>[bool](#bool)</code> | If True, store all samples for exact percentile confidence intervals. If False, use normal approximation (much more memory efficient). Defaults to False. | <code>False</code>
`percentiles` | <code>[tuple](#tuple)[[float](#float), [float](#float)]</code> | Percentiles for confidence intervals (e.g., (2.5, 97.5) for 95% CI). Defaults to (2.5, 97.5). | <code>(2.5, 97.5)</code>

**Methods:**

Name | Description
---- | -----------
[`get_results`](#algorithms-inference-bootstrap-get-results) | Compute final bootstrap statistics.
[`update`](#algorithms-inference-bootstrap-update) | Update statistics with a new bootstrap sample.



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

#### Methods

(algorithms-inference-bootstrap-get-results)=
##### `get_results`

```python
get_results(tail: int | str = 2) -> dict[str, np.ndarray]
```

Compute final bootstrap statistics.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`tail` | <code>[int](#int) \| [str](#str)</code> | `2`/`'two'` (two-tailed, default) or `1`/`'one'` (one-tailed: statistic > 0; negate the data for the other direction). | <code>2</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | Dictionary with keys 'mean' (bootstrap mean), 'std' (bootstrap standard     deviation), 'Z' (z-scores, mean/std), 'p' (p-values per ``tail``),     'ci_lower' and 'ci_upper' (confidence bounds), and 'samples' (all     samples, only if ``save_samples=True``).

**Examples:**

```python
stats = OnlineBootstrapStats(shape=(100,), save_samples=False)
for _ in range(1000):
    stats.update(np.random.randn(100))
results = stats.get_results()
# results.keys() -> mean, std, Z, p, ci_lower, ci_upper
```

(algorithms-inference-bootstrap-update)=
##### `update`

```python
update(sample: np.ndarray) -> None
```

Update statistics with a new bootstrap sample.

Uses Welford's algorithm for numerical stability.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sample` | <code>[ndarray](#numpy.ndarray)</code> | New bootstrap sample with shape matching self.shape. | *required*
