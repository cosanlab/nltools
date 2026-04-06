## `nltools.pipelines.results`

Result containers for nltools pipelines.

This module provides result classes that hold outputs from pipeline execution,
including cross-validation results and per-fold information.

**Classes:**

Name | Description
---- | -----------
[`CVResult`](#nltools.pipelines.results.CVResult) | Cross-validation result container.
[`FoldResult`](#nltools.pipelines.results.FoldResult) | Result from a single CV fold.
[`ISCResult`](#nltools.pipelines.results.ISCResult) | Result from ISC terminal computation.
[`PermutationResult`](#nltools.pipelines.results.PermutationResult) | Result from permutation testing.
[`RSAResult`](#nltools.pipelines.results.RSAResult) | Result from RSA terminal computation.



### Classes#### `nltools.pipelines.results.CVResult`

```python
CVResult(fold_results: List[FoldResult], pipeline: Any) -> None
```

Cross-validation result container.

Aggregates results from all CV folds, providing access to scores,
predictions, and inverse transform capability.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fold_results` | <code>[List](#typing.List)[[FoldResult](#nltools.pipelines.results.FoldResult)]</code> | Results from each CV fold. | *required*
`pipeline` | <code>[Any](#typing.Any)</code> | The pipeline that produced these results. | *required*

Examples
--------
>>> result = pipeline.predict(y)
>>> print(f"Mean score: {result.mean_score:.4f} (+/- {result.std_score:.4f})")
>>> all_predictions = result.predictions  # In original sample order

**Functions:**

Name | Description
---- | -----------
[`inverse_transform`](#nltools.pipelines.results.CVResult.inverse_transform) | Map predictions back through inverse transforms.
[`summary`](#nltools.pipelines.results.CVResult.summary) | Return formatted summary string.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`fold_results`](#nltools.pipelines.results.CVResult.fold_results) | <code>[List](#typing.List)[[FoldResult](#nltools.pipelines.results.FoldResult)]</code> | 
[`is_fully_invertible`](#nltools.pipelines.results.CVResult.is_fully_invertible) | <code>[bool](#bool)</code> | Check if all transform steps are invertible.
[`mean_score`](#nltools.pipelines.results.CVResult.mean_score) | <code>[float](#float)</code> | Mean score across all folds.
[`n_folds`](#nltools.pipelines.results.CVResult.n_folds) | <code>[int](#int)</code> | Number of cross-validation folds.
[`pipeline`](#nltools.pipelines.results.CVResult.pipeline) | <code>[Any](#typing.Any)</code> | 
[`predictions`](#nltools.pipelines.results.CVResult.predictions) | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | All predictions in original sample order.
[`scores`](#nltools.pipelines.results.CVResult.scores) | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | Per-fold prediction scores as a numpy array.
[`std_score`](#nltools.pipelines.results.CVResult.std_score) | <code>[float](#float)</code> | Standard deviation of scores across folds.



##### Attributes###### `nltools.pipelines.results.CVResult.fold_results`

```python
fold_results: List[FoldResult]
```

###### `nltools.pipelines.results.CVResult.is_fully_invertible`

```python
is_fully_invertible: bool
```

Check if all transform steps are invertible.

###### `nltools.pipelines.results.CVResult.mean_score`

```python
mean_score: float
```

Mean score across all folds.

###### `nltools.pipelines.results.CVResult.n_folds`

```python
n_folds: int
```

Number of cross-validation folds.

###### `nltools.pipelines.results.CVResult.pipeline`

```python
pipeline: Any
```

###### `nltools.pipelines.results.CVResult.predictions`

```python
predictions: NDArray[np.floating]
```

All predictions in original sample order.

Reconstructs predictions array with each sample's prediction
from the fold where it was in the test set.

###### `nltools.pipelines.results.CVResult.scores`

```python
scores: NDArray[np.floating]
```

Per-fold prediction scores as a numpy array.

###### `nltools.pipelines.results.CVResult.std_score`

```python
std_score: float
```

Standard deviation of scores across folds.



##### Functions###### `nltools.pipelines.results.CVResult.inverse_transform`

```python
inverse_transform(data: Optional[NDArray] = None) -> NDArray
```

Map predictions back through inverse transforms.

Uses the fitted transforms from each fold to inverse transform
predictions back to the original feature space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)]</code> | Data to inverse transform. If None, uses self.predictions. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[NDArray](#numpy.typing.NDArray)</code> | Data in original feature space.

<details class="note" open markdown="1">
<summary>Note</summary>

This applies inverse transforms fold-by-fold, using each fold's
fitted parameters. Not all pipelines support full inversion.

</details>

###### `nltools.pipelines.results.CVResult.summary`

```python
summary() -> str
```

Return formatted summary string.

#### `nltools.pipelines.results.FoldResult`

```python
FoldResult(score: float, predictions: NDArray[np.floating], train_idx: NDArray[np.intp], test_idx: NDArray[np.intp], fitted_stack: Any) -> None
```

Result from a single CV fold.

Holds predictions, scores, and fitted transforms for one fold,
enabling result aggregation and inverse transforms.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`score`](#nltools.pipelines.results.FoldResult.score) | <code>[float](#float)</code> | Model score on test set (e.g., R² or accuracy).
[`predictions`](#nltools.pipelines.results.FoldResult.predictions) | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | Model predictions on test set.
[`train_idx`](#nltools.pipelines.results.FoldResult.train_idx) | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Indices of training samples.
[`test_idx`](#nltools.pipelines.results.FoldResult.test_idx) | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Indices of test samples.
[`fitted_stack`](#nltools.pipelines.results.FoldResult.fitted_stack) | <code>[Any](#typing.Any)</code> | Stack of fitted transforms for inverse transform support.



##### Attributes###### `nltools.pipelines.results.FoldResult.fitted_stack`

```python
fitted_stack: Any
```

###### `nltools.pipelines.results.FoldResult.predictions`

```python
predictions: NDArray[np.floating]
```

###### `nltools.pipelines.results.FoldResult.score`

```python
score: float
```

###### `nltools.pipelines.results.FoldResult.test_idx`

```python
test_idx: NDArray[np.intp]
```

###### `nltools.pipelines.results.FoldResult.train_idx`

```python
train_idx: NDArray[np.intp]
```

#### `nltools.pipelines.results.ISCResult`

```python
ISCResult(isc: NDArray[np.floating], p: NDArray[np.floating], ci: tuple, method: str, metric: str, n_subjects: int) -> None
```

Result from ISC terminal computation.

Holds intersubject correlation values, p-values, and confidence intervals
from the ISC permutation test.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`isc`](#nltools.pipelines.results.ISCResult.isc) | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | ISC values. Scalar for single-feature or (n_voxels,) for voxel-wise ISC.
[`p`](#nltools.pipelines.results.ISCResult.p) | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | P-values (Phipson-Smyth corrected).
[`ci`](#nltools.pipelines.results.ISCResult.ci) | <code>[tuple](#tuple)</code> | Confidence interval (lower, upper).
[`method`](#nltools.pipelines.results.ISCResult.method) | <code>[str](#str)</code> | ISC method used ('pairwise' or 'leave-one-out').
[`metric`](#nltools.pipelines.results.ISCResult.metric) | <code>[str](#str)</code> | Summary metric used ('median' or 'mean').
[`n_subjects`](#nltools.pipelines.results.ISCResult.n_subjects) | <code>[int](#int)</code> | Number of subjects in the analysis.

**Functions:**

Name | Description
---- | -----------
[`summary`](#nltools.pipelines.results.ISCResult.summary) | Return formatted summary string.



##### Attributes###### `nltools.pipelines.results.ISCResult.ci`

```python
ci: tuple
```

###### `nltools.pipelines.results.ISCResult.isc`

```python
isc: NDArray[np.floating]
```

###### `nltools.pipelines.results.ISCResult.method`

```python
method: str
```

###### `nltools.pipelines.results.ISCResult.metric`

```python
metric: str
```

###### `nltools.pipelines.results.ISCResult.n_subjects`

```python
n_subjects: int
```

###### `nltools.pipelines.results.ISCResult.p`

```python
p: NDArray[np.floating]
```



##### Functions###### `nltools.pipelines.results.ISCResult.summary`

```python
summary() -> str
```

Return formatted summary string.

#### `nltools.pipelines.results.PermutationResult`

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
[`observed`](#nltools.pipelines.results.PermutationResult.observed) | <code>[CVResult](#nltools.pipelines.results.CVResult)</code> | The result from the real (non-permuted) data.
[`null_distribution`](#nltools.pipelines.results.PermutationResult.null_distribution) | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | Array of scores from each permutation.
[`p_value`](#nltools.pipelines.results.PermutationResult.p_value) | <code>[float](#float)</code> | Permutation p-value: proportion of null scores >= observed score.
[`n_permutations`](#nltools.pipelines.results.PermutationResult.n_permutations) | <code>[int](#int)</code> | Number of permutations performed.

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

**Functions:**

Name | Description
---- | -----------
[`from_scores`](#nltools.pipelines.results.PermutationResult.from_scores) | Create PermutationResult from observed result and null scores.
[`summary`](#nltools.pipelines.results.PermutationResult.summary) | Return formatted summary string.



##### Attributes###### `nltools.pipelines.results.PermutationResult.n_permutations`

```python
n_permutations: int
```

###### `nltools.pipelines.results.PermutationResult.null_distribution`

```python
null_distribution: NDArray[np.floating]
```

###### `nltools.pipelines.results.PermutationResult.null_mean`

```python
null_mean: float
```

Mean of the null distribution.

###### `nltools.pipelines.results.PermutationResult.null_std`

```python
null_std: float
```

Standard deviation of the null distribution.

###### `nltools.pipelines.results.PermutationResult.observed`

```python
observed: CVResult
```

###### `nltools.pipelines.results.PermutationResult.observed_score`

```python
observed_score: float
```

Convenience accessor for observed mean score.

###### `nltools.pipelines.results.PermutationResult.p_value`

```python
p_value: float
```

###### `nltools.pipelines.results.PermutationResult.z_score`

```python
z_score: float
```

Z-score of observed relative to null distribution.



##### Functions###### `nltools.pipelines.results.PermutationResult.from_scores`

```python
from_scores(observed: CVResult, null_scores: NDArray[np.floating]) -> 'PermutationResult'
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
<code>'PermutationResult'</code> | Complete permutation result with computed p-value.

###### `nltools.pipelines.results.PermutationResult.summary`

```python
summary() -> str
```

Return formatted summary string.

#### `nltools.pipelines.results.RSAResult`

```python
RSAResult(correlation: float, p_value: float, ci: tuple, method: str, n_conditions: int) -> None
```

Result from RSA terminal computation.

Holds representational similarity analysis correlation and p-value.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`correlation`](#nltools.pipelines.results.RSAResult.correlation) | <code>[float](#float)</code> | Correlation between neural RDM and model RDM.
[`p_value`](#nltools.pipelines.results.RSAResult.p_value) | <code>[float](#float)</code> | P-value from permutation test.
[`ci`](#nltools.pipelines.results.RSAResult.ci) | <code>[tuple](#tuple)</code> | Confidence interval (lower, upper).
[`method`](#nltools.pipelines.results.RSAResult.method) | <code>[str](#str)</code> | Correlation method used (e.g., 'spearman', 'pearson').
[`n_conditions`](#nltools.pipelines.results.RSAResult.n_conditions) | <code>[int](#int)</code> | Number of conditions/stimuli in the RDM.

**Functions:**

Name | Description
---- | -----------
[`summary`](#nltools.pipelines.results.RSAResult.summary) | Return formatted summary string.



##### Attributes###### `nltools.pipelines.results.RSAResult.ci`

```python
ci: tuple
```

###### `nltools.pipelines.results.RSAResult.correlation`

```python
correlation: float
```

###### `nltools.pipelines.results.RSAResult.method`

```python
method: str
```

###### `nltools.pipelines.results.RSAResult.n_conditions`

```python
n_conditions: int
```

###### `nltools.pipelines.results.RSAResult.p_value`

```python
p_value: float
```



##### Functions###### `nltools.pipelines.results.RSAResult.summary`

```python
summary() -> str
```

Return formatted summary string.

