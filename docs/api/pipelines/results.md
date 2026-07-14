## `results`

Result containers for nltools pipelines.

This module provides result classes that hold outputs from pipeline execution,
including cross-validation results and per-fold information.

**Classes:**

Name | Description
---- | -----------
[`CVResult`](#cvresult) | Cross-validation result container.
[`FoldResult`](#foldresult) | Result from a single CV fold.
`ISCResult` | Result from ISC terminal computation.
[`PermutationResult`](#permutationresult) | Result from permutation testing.
[`RSAResult`](#rsaresult) | Result from RSA terminal computation.



### Classes

#### `CVResult`

```python
CVResult(fold_results: list[FoldResult], pipeline: Any) -> None
```

Cross-validation result container.

Aggregates results from all CV folds, providing access to scores,
predictions, and inverse transform capability.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fold_results` | <code>[list](#list)[[FoldResult](#nltools.pipelines.results.FoldResult)]</code> | Results from each CV fold. | *required*
`pipeline` | <code>[Any](#typing.Any)</code> | The pipeline that produced these results. | *required*

Examples:
>>> result = pipeline.predict(y)
>>> print(f"Mean score: {result.mean_score:.4f} (+/- {result.std_score:.4f})")
>>> all_predictions = result.predictions  # In original sample order

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#inverse-transform) | Map predictions back through inverse transforms.
[`summary`](#summary) | Return formatted summary string.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`fold_results` | <code>[list](#list)[[FoldResult](#nltools.pipelines.results.FoldResult)]</code> | 
`is_fully_invertible` | <code>[bool](#bool)</code> | Check if all transform steps are invertible.
`mean_score` | <code>[float](#float)</code> | Mean score across all folds.
`n_folds` | <code>[int](#int)</code> | Number of cross-validation folds.
`pipeline` | <code>[Any](#typing.Any)</code> | 
`predictions` | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | All predictions in original sample order.
`scores` | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | Per-fold prediction scores as a numpy array.
`std_score` | <code>[float](#float)</code> | Standard deviation of scores across folds.

##### Methods

###### `inverse_transform`

```python
inverse_transform(data: NDArray | None = None) -> NDArray
```

Map predictions back through inverse transforms.

Uses the fitted transforms from each fold to inverse transform
predictions back to the original feature space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[NDArray](#numpy.typing.NDArray) \| None</code> | Data to inverse transform. If None, uses self.predictions. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[NDArray](#numpy.typing.NDArray)</code> | Data in original feature space.

<details class="note" open markdown="1">
<summary>Note</summary>

This applies inverse transforms fold-by-fold, using each fold's
fitted parameters. Not all pipelines support full inversion.

</details>

###### `summary`

```python
summary() -> str
```

Return formatted summary string.

#### `FoldResult`

```python
FoldResult(score: float, predictions: NDArray[np.floating], train_idx: NDArray[np.intp], test_idx: NDArray[np.intp], fitted_stack: Any) -> None
```

Result from a single CV fold.

Holds predictions, scores, and fitted transforms for one fold,
enabling result aggregation and inverse transforms.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`score` | <code>[float](#float)</code> | Model score on test set (e.g., R² or accuracy).
`predictions` | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | Model predictions on test set.
`train_idx` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Indices of training samples.
`test_idx` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Indices of test samples.
`fitted_stack` | <code>[Any](#typing.Any)</code> | Stack of fitted transforms for inverse transform support.

##### Methods

###### `summary`

```python
summary() -> str
```

Return formatted summary string.

#### `PermutationResult`

```python
PermutationResult(observed: CVResult, null_distribution: NDArray[np.floating], p_value: float, n_permutations: int) -> None
```

Result from permutation testing.

Contains the observed result from the real data, the null distribution
of scores from permuted data, and the computed p-value.

The p-value is calculated as the proportion of permutation scores
that are greater than or equal to the observed score (for metrics
where higher is better, like R2 or accuracy).

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`observed` | <code>[CVResult](#nltools.pipelines.results.CVResult)</code> | The result from the real (non-permuted) data.
`null_distribution` | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | Array of scores from each permutation.
`p_value` | <code>[float](#float)</code> | Permutation p-value: proportion of null scores >= observed score.
`n_permutations` | <code>[int](#int)</code> | Number of permutations performed.

**Examples:**

```pycon
>>> perm_result = pipeline.permutation_test(y, n_permutations=1000)
>>> print(f"Observed score: {perm_result.observed.mean_score:.4f}")
>>> print(f"p-value: {perm_result.p_value:.4f}")
```

<details class="note" open markdown="1">
<summary>Note</summary>

The p-value uses the formula ``p = (n_exceeding + 1) / (n_permutations + 1)``
to ensure it is never exactly 0 and accounts for the observed value itself.

</details>

**Methods:**

Name | Description
---- | -----------
[`from_scores`](#from-scores) | Create PermutationResult from observed result and null scores.
[`summary`](#summary) | Return formatted summary string.

##### Methods

###### `from_scores`

```python
from_scores(observed: CVResult, null_scores: NDArray[np.floating]) -> PermutationResult
```

Create PermutationResult from observed result and null scores.

Automatically computes the p-value from the null distribution.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`observed` | <code>[CVResult](#nltools.pipelines.results.CVResult)</code> | The result from the real (non-permuted) data. | *required*
`null_scores` | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | Array of scores from each permutation. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[PermutationResult](#nltools.pipelines.results.PermutationResult)</code> | Complete permutation result with computed p-value.

###### `summary`

```python
summary() -> str
```

Return formatted summary string.

#### `RSAResult`

```python
RSAResult(correlation: float, p_value: float, ci: tuple, method: str, n_conditions: int) -> None
```

Result from RSA terminal computation.

Holds representational similarity analysis correlation and p-value.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`correlation` | <code>[float](#float)</code> | Correlation between neural RDM and model RDM.
`p_value` | <code>[float](#float)</code> | P-value from permutation test.
`ci` | <code>[tuple](#tuple)</code> | Confidence interval (lower, upper).
`method` | <code>[str](#str)</code> | Correlation method used (e.g., 'spearman', 'pearson').
`n_conditions` | <code>[int](#int)</code> | Number of conditions/stimuli in the RDM.

**Methods:**

Name | Description
---- | -----------
[`summary`](#summary) | Return formatted summary string.

##### Methods

###### `summary`

```python
summary() -> str
```

Return formatted summary string.

