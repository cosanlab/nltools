---
title: data.adjacency.modeling
label: page-data-adjacency-modeling
---

Provide standalone modeling and inference functions for Adjacency matrices.

Each function takes an Adjacency instance as its first argument (`adj`).

**Functions:**

Name | Description
---- | -----------
[`bootstrap`](#data-adjacency-modeling-bootstrap) | Bootstrap statistics using efficient online algorithms.
[`convert_bootstrap_results_to_adjacency`](#data-adjacency-modeling-convert-bootstrap-results-to-adjacency) | Convert bootstrap results dictionary to Adjacency format.
[`generate_permutations`](#data-adjacency-modeling-generate-permutations) | Generate permuted versions of an Adjacency instance lazily.
[`regress`](#data-adjacency-modeling-regress) | Run a regression on an adjacency instance.
[`social_relations_model`](#data-adjacency-modeling-social-relations-model) | Estimate the social relations model from a matrix for a round-robin design.



## Functions

(data-adjacency-modeling-bootstrap)=
### `bootstrap`

```python
bootstrap(adj, stat, *, n_samples = 5000, save_boots = False, percentiles = (2.5, 97.5), tail = 2, n_jobs = -1, random_state = None, progress_bar = False)
```

Bootstrap statistics using efficient online algorithms.

Uses memory-efficient bootstrap infrastructure with CPU parallelization.
Supports simple aggregation statistics (mean, std, median, sum, min, max).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#page-data-adjacency)</code> | Adjacency instance containing multiple matrices. | *required*
`stat` | <code>str</code> | Statistic to bootstrap: `'mean'`, `'median'`, `'std'`, `'sum'`, `'min'`, or `'max'`. | *required*
`n_samples` | <code>int</code> | Number of bootstrap iterations. Default 5000. | <code>5000</code>
`save_boots` | <code>bool</code> | If True, store all bootstrap samples (memory intensive). Default False. | <code>False</code>
`percentiles` | <code>tuple</code> | Percentiles for confidence intervals. Default (2.5, 97.5). | <code>(2.5, 97.5)</code>
`tail` | <code>int \| str</code> | `2`/`'two'` for two-tailed (default); `1`/`'one'` for one-tailed (statistic > 0; negate the data for the other direction). | <code>2</code>
`n_jobs` | <code>int</code> | Number of CPU cores for parallelization. -1 means all CPUs. | <code>-1</code>
`random_state` | <code>int</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. Default False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Dictionary with keys `'Z'`, `'p'`, `'mean'`, `'std'`, `'ci_lower'`,     `'ci_upper'` (all Adjacency objects). If `save_boots=True`, also includes     `'samples'`.

**Examples:**

```python
boot = bootstrap(adj, stat="mean", n_samples=1000)
boot["mean"]  # → Adjacency
```

(data-adjacency-modeling-convert-bootstrap-results-to-adjacency)=
### `convert_bootstrap_results_to_adjacency`

```python
convert_bootstrap_results_to_adjacency(adj, result, save_boots = False)
```

Convert bootstrap results dictionary to Adjacency format.

Helper function to convert numpy arrays from bootstrap functions into
Adjacency objects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#page-data-adjacency)</code> | Adjacency instance (used for `matrix_type` metadata). | *required*
`result` | <code>dict</code> | Result dictionary from a bootstrap function with keys `'mean'`, `'std'`, `'Z'`, `'p'`, `'ci_lower'`, `'ci_upper'`, and optionally `'samples'`. | *required*
`save_boots` | <code>bool</code> | If True, include the `'samples'` key in the output. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Adjacency objects for each statistic.

(data-adjacency-modeling-generate-permutations)=
### `generate_permutations`

```python
generate_permutations(adj, n_permute, random_state = None)
```

Generate permuted versions of an Adjacency instance lazily.

This is useful for iterative comparisons.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#page-data-adjacency)</code> | Adjacency instance. | *required*
`n_permute` | <code>int</code> | Number of permutations. | *required*
`random_state` | <code>int \| RandomState</code> | Random seed for reproducibility. Defaults to None. | <code>None</code>

**Yields:**

Type | Description
---- | -----------
<code>[Adjacency](#page-data-adjacency)</code> | Permuted version of `adj`.

**Examples:**

```python
for perm in generate_permutations(adj, 1000):
    out = neural_distance_mat.similarity(perm)
```

(data-adjacency-modeling-regress)=
### `regress`

```python
regress(adj, X, method = 'ols', tail = 2)
```

Run a regression on an adjacency instance.

Pass an `Adjacency` as `X` to decompose `adj` with other matrices, or a
`DesignMatrix` to regress each cell across a stack of matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#page-data-adjacency)</code> | Adjacency instance. | *required*
`X` | <code>[Adjacency](#page-data-adjacency) \| [DesignMatrix](#page-data-design-matrix)</code> | Design matrix. | *required*
`method` | <code>str</code> | Type of regression; only `'ols'` is currently supported. | <code>'ols'</code>
`tail` | <code>int \| str</code> | `2`/`'two'` for two-tailed (default); `1`/`'one'` for one-tailed (beta > 0; negate a regressor for the other direction). | <code>2</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Adjacency instances keyed `'beta'`, `'sigma'`, `'t'`, `'p'`, `'df'`,     `'residual'`.

(data-adjacency-modeling-social-relations-model)=
### `social_relations_model`

```python
social_relations_model(adj, summarize_results = True, nan_replace = True)
```

Estimate the social relations model from a matrix for a round-robin design.

$$X_{ij} = m + \alpha_i + \beta_j + g_{ij} + \epsilon_{ijl}$$

where $X_{ij}$ is the score for person i rating person j, $m$ is the group mean,
$\alpha_i$ is person i's actor effect, $\beta_j$ is person j's partner effect, $g_{ij}$
is the relationship effect and $\epsilon_{ijl}$ is the error in measure l for actor i and partner j.

This model is primarily concerned with partitioning the variance of the various
effects. The implementation follows Chapter 8 of Kenny, Kashy, & Cook (2006) and
the tests replicate the book's examples. Actor scores are rows (lower triangle)
and partner scores are columns (upper triangle). The minimal sample size to
estimate these effects is 4.

**Model assumptions:** social interactions are exclusively dyadic; people are
randomly sampled from the population; there are no order effects; the effects
combine additively and relationships are linear.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#page-data-adjacency)</code> | A single matrix, or one matrix per group. | *required*
`summarize_results` | <code>bool</code> | If True, print a formatted summary of model results. | <code>True</code>
`nan_replace` | <code>bool</code> | If True, replace NaN values with row and column means. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>Series \| DataFrame</code> | All of the effects estimated using SRM (a Series     for a single matrix, a DataFrame with one row per matrix otherwise).

<details class="references" open markdown="1">
<summary>References</summary>

Kenny, D. A., Kashy, D. A., & Cook, W. L. (2006). *Dyadic data analysis*.
Guilford Press.

</details>
