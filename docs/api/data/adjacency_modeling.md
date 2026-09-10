---
title: data.adjacency.modeling
label: page-data-adjacency-modeling
---

Provide standalone modeling and inference functions for Adjacency matrices.

Each function takes an Adjacency instance as its first argument (`adj`).

**Functions:**

Name | Description
---- | -----------
[`bootstrap`](#data-adjacency-modeling-bootstrap) | Bootstrap an aggregate statistic across a stack of matrices.
[`convert_bootstrap_results_to_adjacency`](#data-adjacency-modeling-convert-bootstrap-results-to-adjacency) | Wrap an engine's arrays as a `BootstrapResult` of single-matrix `Adjacency`.
[`generate_permutations`](#data-adjacency-modeling-generate-permutations) | Generate permuted versions of an Adjacency instance lazily.
[`regress`](#data-adjacency-modeling-regress) | Run a regression on an adjacency instance.
[`social_relations_model`](#data-adjacency-modeling-social-relations-model) | Estimate the social relations model from a matrix for a round-robin design.



## Functions

(data-adjacency-modeling-bootstrap)=
### `bootstrap`

```python
bootstrap(adj, statistic, *, n_samples = 5000, confidence_level = 0.95, memory_budget_gb = None, return_samples = False, n_jobs = -1, random_state = None, progress_bar = False)
```

Bootstrap an aggregate statistic across a stack of matrices.

Resamples matrices with replacement and aggregates the replicates as they
complete, so what the run holds is the retained tail — about
`(1 - confidence_level)` of the replicates per edge — plus one dispatch
window, rather than all `n_samples` matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#page-data-adjacency)</code> | Adjacency instance containing multiple matrices. | *required*
`statistic` | <code>str</code> | Statistic to bootstrap: `'mean'`, `'median'`, `'std'`, `'sum'`, `'min'`, or `'max'` — each the corresponding NumPy reduction over matrices, with `'std'` at `ddof=0`. | *required*
`n_samples` | <code>int</code> | Number of bootstrap replicates, at least two. Default 5000. | <code>5000</code>
`confidence_level` | <code>float</code> | Confidence level of the reported interval, strictly between zero and one. Default 0.95. | <code>0.95</code>
`memory_budget_gb` | <code>float \| None</code> | Working-memory budget in GB governing the output preflight and worker planning. None (default) measures the host. | <code>None</code>
`return_samples` | <code>bool</code> | Retain and return every replicate. Default False. | <code>False</code>
`n_jobs` | <code>int</code> | CPU worker ceiling. -1 (default) means all cores. | <code>-1</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. Default False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[BootstrapResult](#data-results-bootstrapresult)</code> | `estimate`, `standard_error`, `ci_lower` and     `ci_upper` as single-matrix `Adjacency` objects, plus `samples` as     a NumPy array with the bootstrap axis first when     `return_samples=True`.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If `statistic` is unknown, an argument is out of range, or the retained output cannot fit the memory budget.

**Examples:**

```python
boot = bootstrap(adj, "mean", n_samples=1000)
boot.estimate  # → Adjacency
```

(data-adjacency-modeling-convert-bootstrap-results-to-adjacency)=
### `convert_bootstrap_results_to_adjacency`

```python
convert_bootstrap_results_to_adjacency(adj, result)
```

Wrap an engine's arrays as a `BootstrapResult` of single-matrix `Adjacency`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`adj` | <code>[Adjacency](#page-data-adjacency)</code> | Instance supplying matrix kind and node labels. | *required*
`result` | <code>dict</code> | Engine output with `'estimate'`, `'standard_error'`, `'ci_lower'`, `'ci_upper'`, and optionally `'samples'`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BootstrapResult](#data-results-bootstrapresult)</code> | The four summaries as `Adjacency`, and the retained     replicates as a NumPy array when present.

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
<code>dict</code> | Coefficient fields `beta`, `sigma` (coefficient standard error),     `t`, and `p` are predictor Adjacency maps for DesignMatrix input and     native predictor arrays/scalars for Adjacency input. `df` is scalar;     `residual` is an Adjacency retaining the response shape and metadata.

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
