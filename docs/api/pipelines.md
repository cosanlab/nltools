## `nltools.pipelines`

Pipeline infrastructure for nltools.

This module provides a fluent API for building data processing pipelines
with cross-validation support.

Classes
-------
Pipeline
    Base pipeline for chained transforms with optional CV.
CVScheme
    Cross-validation scheme configuration.
FittedStack
    Collection of fitted transforms for inverse transform support.

Protocols
---------
TransformStep
    Protocol for pipeline transform steps.
FittedTransform
    Protocol for fitted transform objects.
Terminal
    Protocol for terminal operations.

Examples
--------
>>> from nltools.pipelines import Pipeline, CVScheme
>>> cv = CVScheme(scheme='kfold', k=5)
>>> result = (
...     Pipeline(data, cv=cv)
...     .normalize()
...     .reduce(n_components=50)
...     .predict(y)
... )

**Modules:**

Name | Description
---- | -----------
[`base`](#nltools.pipelines.base) | Pipeline base infrastructure for nltools.
[`cv`](#nltools.pipelines.cv) | Cross-validation scheme configuration for nltools pipelines.
[`multi_subject`](#nltools.pipelines.multi_subject) | Multi-subject pipeline for cross-subject analyses.
[`pool`](#nltools.pipelines.pool) | Pool infrastructure for multi-subject aggregation.
[`results`](#nltools.pipelines.results) | Result containers for nltools pipelines.
[`steps`](#nltools.pipelines.steps) | Transform steps for nltools pipelines.
[`terminals`](#nltools.pipelines.terminals) | Terminal operations for nltools pipelines.

**Classes:**

Name | Description
---- | -----------
[`AlignStep`](#nltools.pipelines.AlignStep) | Cross-subject alignment via SRM or HyperAlignment.
[`CVResult`](#nltools.pipelines.CVResult) | Cross-validation result container.
[`CVScheme`](#nltools.pipelines.CVScheme) | Protocol for cross-validation schemes.
[`CVSchemeImpl`](#nltools.pipelines.CVSchemeImpl) | Cross-validation scheme configuration.
[`FittedAlign`](#nltools.pipelines.FittedAlign) | Fitted alignment model.
[`FittedStack`](#nltools.pipelines.FittedStack) | Collection of fitted transforms for inverse transform support.
[`FittedTransform`](#nltools.pipelines.FittedTransform) | Protocol for fitted transform objects.
[`FoldResult`](#nltools.pipelines.FoldResult) | Result from a single CV fold.
[`ISCResult`](#nltools.pipelines.ISCResult) | Result from ISC terminal computation.
[`ISCTerminal`](#nltools.pipelines.ISCTerminal) | ISC terminal for multi-subject pipelines.
[`MultiSubjectPipeline`](#nltools.pipelines.MultiSubjectPipeline) | Pipeline for multi-subject neuroimaging analyses.
[`NestedCVScheme`](#nltools.pipelines.NestedCVScheme) | Nested cross-validation for hyperparameter tuning.
[`NormalizeStep`](#nltools.pipelines.NormalizeStep) | Normalization transform step.
[`PermutationResult`](#nltools.pipelines.PermutationResult) | Result from permutation testing.
[`PipeStep`](#nltools.pipelines.PipeStep) | Wrapper for sklearn-compatible transformers.
[`Pipeline`](#nltools.pipelines.Pipeline) | Base pipeline for chained transforms with optional cross-validation.
[`PooledData`](#nltools.pipelines.PooledData) | Aggregated data from multiple subjects.
[`PredictTerminal`](#nltools.pipelines.PredictTerminal) | Prediction/classification terminal for CV pipelines.
[`RSAResult`](#nltools.pipelines.RSAResult) | Result from RSA terminal computation.
[`RSATerminal`](#nltools.pipelines.RSATerminal) | RSA terminal for multi-subject pipelines.
[`ReduceStep`](#nltools.pipelines.ReduceStep) | Dimensionality reduction step.
[`ResultDict`](#nltools.pipelines.ResultDict) | Dictionary of StatResults, one per contrast.
[`StatResult`](#nltools.pipelines.StatResult) | Result of statistical test.
[`Terminal`](#nltools.pipelines.Terminal) | Protocol for terminal operations that end a pipeline.
[`TransformStep`](#nltools.pipelines.TransformStep) | Protocol for pipeline transform steps.



### Classes#### `nltools.pipelines.AlignStep`

```python
AlignStep(method: str = 'srm', scheme: str = 'global', n_features: int | None = 50, new_subject: str = 'procrustes', n_iter: int = 10, parallel: str | None = 'cpu', n_jobs: int = -1, **kwargs: int)
```

Cross-subject alignment via SRM or HyperAlignment.

Wraps existing SRM and HyperAlignment algorithms for use in pipelines.
Currently supports 'global' scheme only. Searchlight/piecewise schemes
require LocalAlignment (nltools-boll epic).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Alignment method: 'srm' or 'hyperalignment'. Default is 'srm'. | <code>'srm'</code>
`scheme` | <code>[str](#str)</code> | Spatial scheme. Currently only 'global' is supported. 'searchlight' and 'piecewise' require LocalAlignment. | <code>'global'</code>
`n_features` | <code>[int](#int) \| None</code> | Number of features for SRM. None for hyperalignment (full rank). | <code>50</code>
`new_subject` | <code>[str](#str)</code> | Method for aligning held-out subjects in LOSO CV. Default is 'procrustes'. | <code>'procrustes'</code>
`n_iter` | <code>[int](#int)</code> | Number of iterations for SRM (or 2 for hyperalignment). Default is 10. | <code>10</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization: 'cpu', 'gpu', or None. | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of jobs for CPU parallelization. Default is -1. | <code>-1</code>
`**kwargs` |  | Additional arguments passed to the underlying algorithm. For SRM: 'rand_seed'. For HyperAlignment: 'auto_pad'. | <code>{}</code>

Examples
--------
>>> import numpy as np
>>> # Create synthetic multi-subject data
>>> data = [np.random.randn(100, 50) for _ in range(5)]
>>> step = AlignStep(method='srm', n_features=10)
>>> fitted = step.fit(data)
>>> aligned = fitted.transform(data)

**Functions:**

Name | Description
---- | -----------
[`fit`](#nltools.pipelines.AlignStep.fit) | Fit alignment model on list of subject data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#nltools.pipelines.AlignStep.invertible) | <code>[bool](#bool)</code> | Check if alignment is invertible.
[`kwargs`](#nltools.pipelines.AlignStep.kwargs) |  | 
[`method`](#nltools.pipelines.AlignStep.method) |  | 
[`n_features`](#nltools.pipelines.AlignStep.n_features) |  | 
[`n_iter`](#nltools.pipelines.AlignStep.n_iter) |  | 
[`n_jobs`](#nltools.pipelines.AlignStep.n_jobs) |  | 
[`new_subject`](#nltools.pipelines.AlignStep.new_subject) |  | 
[`parallel`](#nltools.pipelines.AlignStep.parallel) |  | 
[`scheme`](#nltools.pipelines.AlignStep.scheme) |  | 



##### Attributes###### `nltools.pipelines.AlignStep.invertible`

```python
invertible: bool
```

Check if alignment is invertible.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if method is hyperalignment (full-rank orthogonal transforms).

###### `nltools.pipelines.AlignStep.kwargs`

```python
kwargs = kwargs
```

###### `nltools.pipelines.AlignStep.method`

```python
method = method
```

###### `nltools.pipelines.AlignStep.n_features`

```python
n_features = n_features
```

###### `nltools.pipelines.AlignStep.n_iter`

```python
n_iter = n_iter
```

###### `nltools.pipelines.AlignStep.n_jobs`

```python
n_jobs = n_jobs
```

###### `nltools.pipelines.AlignStep.new_subject`

```python
new_subject = new_subject
```

###### `nltools.pipelines.AlignStep.parallel`

```python
parallel = parallel
```

###### `nltools.pipelines.AlignStep.scheme`

```python
scheme = scheme
```



##### Functions###### `nltools.pipelines.AlignStep.fit`

```python
fit(data: list[np.ndarray]) -> FittedAlign
```

Fit alignment model on list of subject data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | Each array has shape (n_voxels, n_samples) or (n_samples, n_voxels). Will be transposed if needed to match algorithm expectations. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[FittedAlign](#nltools.pipelines.steps.FittedAlign)</code> | Fitted alignment model.

#### `nltools.pipelines.CVResult`

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
[`inverse_transform`](#nltools.pipelines.CVResult.inverse_transform) | Map predictions back through inverse transforms.
[`summary`](#nltools.pipelines.CVResult.summary) | Return formatted summary string.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`fold_results`](#nltools.pipelines.CVResult.fold_results) | <code>[List](#typing.List)[[FoldResult](#nltools.pipelines.results.FoldResult)]</code> | 
[`is_fully_invertible`](#nltools.pipelines.CVResult.is_fully_invertible) | <code>[bool](#bool)</code> | Check if all transform steps are invertible.
[`mean_score`](#nltools.pipelines.CVResult.mean_score) | <code>[float](#float)</code> | Mean score across all folds.
[`n_folds`](#nltools.pipelines.CVResult.n_folds) | <code>[int](#int)</code> | Number of cross-validation folds.
[`pipeline`](#nltools.pipelines.CVResult.pipeline) | <code>[Any](#typing.Any)</code> | 
[`predictions`](#nltools.pipelines.CVResult.predictions) | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | All predictions in original sample order.
[`scores`](#nltools.pipelines.CVResult.scores) | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | Per-fold prediction scores as a numpy array.
[`std_score`](#nltools.pipelines.CVResult.std_score) | <code>[float](#float)</code> | Standard deviation of scores across folds.



##### Attributes###### `nltools.pipelines.CVResult.fold_results`

```python
fold_results: List[FoldResult]
```

###### `nltools.pipelines.CVResult.is_fully_invertible`

```python
is_fully_invertible: bool
```

Check if all transform steps are invertible.

###### `nltools.pipelines.CVResult.mean_score`

```python
mean_score: float
```

Mean score across all folds.

###### `nltools.pipelines.CVResult.n_folds`

```python
n_folds: int
```

Number of cross-validation folds.

###### `nltools.pipelines.CVResult.pipeline`

```python
pipeline: Any
```

###### `nltools.pipelines.CVResult.predictions`

```python
predictions: NDArray[np.floating]
```

All predictions in original sample order.

Reconstructs predictions array with each sample's prediction
from the fold where it was in the test set.

###### `nltools.pipelines.CVResult.scores`

```python
scores: NDArray[np.floating]
```

Per-fold prediction scores as a numpy array.

###### `nltools.pipelines.CVResult.std_score`

```python
std_score: float
```

Standard deviation of scores across folds.



##### Functions###### `nltools.pipelines.CVResult.inverse_transform`

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

###### `nltools.pipelines.CVResult.summary`

```python
summary() -> str
```

Return formatted summary string.

#### `nltools.pipelines.CVScheme`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for cross-validation schemes.

Compatible with scikit-learn CV splitters and custom implementations.

**Functions:**

Name | Description
---- | -----------
[`split`](#nltools.pipelines.CVScheme.split) | Generate train/test index splits.



##### Functions###### `nltools.pipelines.CVScheme.split`

```python
split(data: Any) -> Any
```

Generate train/test index splits.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (used to determine n_samples). | *required*

**Yields:**

Type | Description
---- | -----------
<code>[Any](#typing.Any)</code> | Tuple of (train_idx, test_idx) arrays of indices for each fold.

#### `nltools.pipelines.CVSchemeImpl`

```python
CVSchemeImpl(k: Optional[int] = None, scheme: CVSchemeType = 'kfold', split_by: Optional[str] = None, n: int = 1000, random_state: Optional[int] = None) -> None
```

Cross-validation scheme configuration.

Supports multiple CV strategies:
- kfold: k-fold cross-validation
- loso: leave-one-subject-out (for multi-subject)
- loro: leave-one-run-out
- bootstrap: bootstrap resampling
- permutation: permutation testing (shuffles targets)

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`k` | <code>[Optional](#typing.Optional)[[int](#int)]</code> | Number of folds (for kfold scheme). Defaults to 5 if scheme is 'kfold'. | <code>None</code>
`scheme` | <code>[CVSchemeType](#nltools.pipelines.cv.CVSchemeType)</code> | CV scheme type. One of 'kfold', 'loso', 'loro', 'bootstrap', or 'permutation'. | <code>'kfold'</code>
`split_by` | <code>[Optional](#typing.Optional)[[str](#str)]</code> | Attribute to split by ('runs', 'subjects', 'sessions'). Used for documentation purposes with loso/loro schemes. | <code>None</code>
`n` | <code>[int](#int)</code> | Number of bootstrap iterations (for bootstrap scheme). Defaults to 1000. | <code>1000</code>
`random_state` | <code>[Optional](#typing.Optional)[[int](#int)]</code> | Random seed for reproducibility. If provided, sets the numpy random seed during initialization. | <code>None</code>

**Examples:**

```pycon
>>> # 5-fold cross-validation
>>> cv = CVScheme(scheme='kfold', k=5)
>>> for train_idx, test_idx in cv.split(data):
...     # train and evaluate model
...     pass
```

```pycon
>>> # Leave-one-subject-out
>>> cv = CVScheme(scheme='loso', split_by='subjects')
>>> for train_idx, test_idx in cv.split(data, groups=subject_ids):
...     pass
```

```pycon
>>> # Bootstrap with 500 iterations
>>> cv = CVScheme(scheme='bootstrap', n=500, random_state=42)
```

```pycon
>>> # Permutation testing with 1000 permutations
>>> cv = CVScheme(scheme='permutation', n=1000, random_state=42)
```

**Functions:**

Name | Description
---- | -----------
[`n_splits`](#nltools.pipelines.CVSchemeImpl.n_splits) | Return number of splits.
[`split`](#nltools.pipelines.CVSchemeImpl.split) | Generate train/test indices for each fold.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`is_loro`](#nltools.pipelines.CVSchemeImpl.is_loro) | <code>[bool](#bool)</code> | Check if this is leave-one-run-out.
[`is_loso`](#nltools.pipelines.CVSchemeImpl.is_loso) | <code>[bool](#bool)</code> | Check if this is leave-one-subject-out.
[`k`](#nltools.pipelines.CVSchemeImpl.k) | <code>[Optional](#typing.Optional)[[int](#int)]</code> | 
[`n`](#nltools.pipelines.CVSchemeImpl.n) | <code>[int](#int)</code> | 
[`random_state`](#nltools.pipelines.CVSchemeImpl.random_state) | <code>[Optional](#typing.Optional)[[int](#int)]</code> | 
[`scheme`](#nltools.pipelines.CVSchemeImpl.scheme) | <code>[CVSchemeType](#nltools.pipelines.cv.CVSchemeType)</code> | 
[`split_by`](#nltools.pipelines.CVSchemeImpl.split_by) | <code>[Optional](#typing.Optional)[[str](#str)]</code> | 



##### Attributes###### `nltools.pipelines.CVSchemeImpl.is_loro`

```python
is_loro: bool
```

Check if this is leave-one-run-out.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if scheme is 'loro', False otherwise.

###### `nltools.pipelines.CVSchemeImpl.is_loso`

```python
is_loso: bool
```

Check if this is leave-one-subject-out.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if scheme is 'loso', False otherwise.

###### `nltools.pipelines.CVSchemeImpl.k`

```python
k: Optional[int] = None
```

###### `nltools.pipelines.CVSchemeImpl.n`

```python
n: int = 1000
```

###### `nltools.pipelines.CVSchemeImpl.random_state`

```python
random_state: Optional[int] = None
```

###### `nltools.pipelines.CVSchemeImpl.scheme`

```python
scheme: CVSchemeType = 'kfold'
```

###### `nltools.pipelines.CVSchemeImpl.split_by`

```python
split_by: Optional[str] = None
```



##### Functions###### `nltools.pipelines.CVSchemeImpl.n_splits`

```python
n_splits(data: Any = None, groups: Optional[NDArray[np.intp]] = None) -> int
```

Return number of splits.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (unused for most schemes, kept for API consistency). | <code>None</code>
`groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for grouped CV. Required for 'loso' and 'loro'. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of splits/folds that will be generated.

###### `nltools.pipelines.CVSchemeImpl.split`

```python
split(data: Any, groups: Optional[NDArray[np.intp]] = None) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]
```

Generate train/test indices for each fold.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (used for length). Can be any object with __len__ or a numpy array with shape attribute. | *required*
`groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for grouped CV (runs, subjects, etc.). Required for 'loso' and 'loro' schemes. | <code>None</code>

**Yields:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Tuple of (train_indices, test_indices) for each fold.

#### `nltools.pipelines.FittedAlign`

```python
FittedAlign(model: Any, method: str, new_subject_method: str = 'procrustes') -> None
```

Fitted alignment model.

Holds a fitted SRM or HyperAlignment model and applies transformations.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`model`](#nltools.pipelines.FittedAlign.model) | <code>[Any](#typing.Any)</code> | Fitted SRM or HyperAlignment instance.
[`method`](#nltools.pipelines.FittedAlign.method) | <code>[str](#str)</code> | The alignment method used ('srm' or 'hyperalignment').
[`new_subject_method`](#nltools.pipelines.FittedAlign.new_subject_method) | <code>[str](#str)</code> | Method for aligning held-out subjects in LOSO CV.

**Functions:**

Name | Description
---- | -----------
[`inverse_transform`](#nltools.pipelines.FittedAlign.inverse_transform) | Reverse alignment (only for full-rank hyperalignment).
[`transform`](#nltools.pipelines.FittedAlign.transform) | Transform subjects that were in training.
[`transform_new_subject`](#nltools.pipelines.FittedAlign.transform_new_subject) | Align a new subject not in training (for LOSO).



##### Attributes###### `nltools.pipelines.FittedAlign.method`

```python
method: str
```

###### `nltools.pipelines.FittedAlign.model`

```python
model: Any
```

###### `nltools.pipelines.FittedAlign.new_subject_method`

```python
new_subject_method: str = 'procrustes'
```



##### Functions###### `nltools.pipelines.FittedAlign.inverse_transform`

```python
inverse_transform(data: list[np.ndarray]) -> list[np.ndarray]
```

Reverse alignment (only for full-rank hyperalignment).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | Aligned data for each subject, shape (n_samples, n_aligned_features). Pipeline convention is (samples, features). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | Data in original subject-specific space, shape (n_samples, n_features).

###### `nltools.pipelines.FittedAlign.transform`

```python
transform(data: list[np.ndarray]) -> list[np.ndarray]
```

Transform subjects that were in training.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | List of subject data arrays, each shape (n_samples, n_features). Pipeline convention is (samples, features). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | Aligned data for each subject, shape (n_samples, n_aligned_features).

###### `nltools.pipelines.FittedAlign.transform_new_subject`

```python
transform_new_subject(data: np.ndarray) -> np.ndarray
```

Align a new subject not in training (for LOSO).

Uses transform_subject() which fits a new transform for the held-out subject.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Data for the new subject, shape (n_samples, n_features). Pipeline convention is (samples, features). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Aligned data for the new subject, shape (n_samples, n_aligned_features).

#### `nltools.pipelines.FittedStack`

```python
FittedStack(steps: List[FittedTransform] = list()) -> None
```

Collection of fitted transforms for inverse transform support.

Maintains the sequence of fitted transforms from a pipeline execution,
enabling inverse transformation back to the original data space.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`steps`](#nltools.pipelines.FittedStack.steps) | <code>[List](#typing.List)[[FittedTransform](#nltools.pipelines.base.FittedTransform)]</code> | Ordered list of fitted transforms.

Examples
--------
>>> stack = FittedStack()
>>> stack.append(fitted_pca)
>>> stack.append(fitted_normalize)
>>> original_space = stack.inverse_transform(predictions)

**Functions:**

Name | Description
---- | -----------
[`append`](#nltools.pipelines.FittedStack.append) | Add a fitted transform to the stack.
[`inverse_transform`](#nltools.pipelines.FittedStack.inverse_transform) | Apply inverse transforms in reverse order.



##### Attributes###### `nltools.pipelines.FittedStack.is_fully_invertible`

```python
is_fully_invertible: bool
```

Check if all steps support inverse transform.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if all steps have callable inverse_transform methods.

###### `nltools.pipelines.FittedStack.steps`

```python
steps: List[FittedTransform] = field(default_factory=list)
```



##### Functions###### `nltools.pipelines.FittedStack.append`

```python
append(fitted_step: FittedTransform) -> None
```

Add a fitted transform to the stack.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fitted_step` | <code>[FittedTransform](#nltools.pipelines.base.FittedTransform)</code> | Fitted transform to append. | *required*

###### `nltools.pipelines.FittedStack.inverse_transform`

```python
inverse_transform(data: Any) -> Any
```

Apply inverse transforms in reverse order.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to inverse transform. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Any](#typing.Any)</code> | Data transformed back toward original space.

<details class="note" open markdown="1">
<summary>Note</summary>

Steps without ``inverse_transform`` are silently skipped.
Use ``is_fully_invertible`` to check if all steps support inversion.

</details>

#### `nltools.pipelines.FittedTransform`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for fitted transform objects.

A fitted transform holds the learned parameters from fitting on training
data and can apply the transformation to new data.

<details class="note" open markdown="1">
<summary>Note</summary>

Not all transforms are invertible. Check the parent TransformStep's
``invertible`` attribute or use ``hasattr`` before calling ``inverse_transform``.

</details>

**Functions:**

Name | Description
---- | -----------
[`inverse_transform`](#nltools.pipelines.FittedTransform.inverse_transform) | Apply the inverse transformation to data.
[`transform`](#nltools.pipelines.FittedTransform.transform) | Apply the learned transformation to data.



##### Functions###### `nltools.pipelines.FittedTransform.inverse_transform`

```python
inverse_transform(data: Any) -> Any
```

Apply the inverse transformation to data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to inverse transform. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Any](#typing.Any)</code> | Data in original space.

###### `nltools.pipelines.FittedTransform.transform`

```python
transform(data: Any) -> Any
```

Apply the learned transformation to data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to transform (typically ndarray or BrainData). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Any](#typing.Any)</code> | Transformed data.

#### `nltools.pipelines.FoldResult`

```python
FoldResult(score: float, predictions: NDArray[np.floating], train_idx: NDArray[np.intp], test_idx: NDArray[np.intp], fitted_stack: Any) -> None
```

Result from a single CV fold.

Holds predictions, scores, and fitted transforms for one fold,
enabling result aggregation and inverse transforms.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`score`](#nltools.pipelines.FoldResult.score) | <code>[float](#float)</code> | Model score on test set (e.g., R² or accuracy).
[`predictions`](#nltools.pipelines.FoldResult.predictions) | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | Model predictions on test set.
[`train_idx`](#nltools.pipelines.FoldResult.train_idx) | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Indices of training samples.
[`test_idx`](#nltools.pipelines.FoldResult.test_idx) | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Indices of test samples.
[`fitted_stack`](#nltools.pipelines.FoldResult.fitted_stack) | <code>[Any](#typing.Any)</code> | Stack of fitted transforms for inverse transform support.



##### Attributes###### `nltools.pipelines.FoldResult.fitted_stack`

```python
fitted_stack: Any
```

###### `nltools.pipelines.FoldResult.predictions`

```python
predictions: NDArray[np.floating]
```

###### `nltools.pipelines.FoldResult.score`

```python
score: float
```

###### `nltools.pipelines.FoldResult.test_idx`

```python
test_idx: NDArray[np.intp]
```

###### `nltools.pipelines.FoldResult.train_idx`

```python
train_idx: NDArray[np.intp]
```

#### `nltools.pipelines.ISCResult`

```python
ISCResult(isc: NDArray[np.floating], p: NDArray[np.floating], ci: tuple, method: str, metric: str, n_subjects: int) -> None
```

Result from ISC terminal computation.

Holds intersubject correlation values, p-values, and confidence intervals
from the ISC permutation test.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`isc`](#nltools.pipelines.ISCResult.isc) | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | ISC values. Scalar for single-feature or (n_voxels,) for voxel-wise ISC.
[`p`](#nltools.pipelines.ISCResult.p) | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | P-values (Phipson-Smyth corrected).
[`ci`](#nltools.pipelines.ISCResult.ci) | <code>[tuple](#tuple)</code> | Confidence interval (lower, upper).
[`method`](#nltools.pipelines.ISCResult.method) | <code>[str](#str)</code> | ISC method used ('pairwise' or 'leave-one-out').
[`metric`](#nltools.pipelines.ISCResult.metric) | <code>[str](#str)</code> | Summary metric used ('median' or 'mean').
[`n_subjects`](#nltools.pipelines.ISCResult.n_subjects) | <code>[int](#int)</code> | Number of subjects in the analysis.

**Functions:**

Name | Description
---- | -----------
[`summary`](#nltools.pipelines.ISCResult.summary) | Return formatted summary string.



##### Attributes###### `nltools.pipelines.ISCResult.ci`

```python
ci: tuple
```

###### `nltools.pipelines.ISCResult.isc`

```python
isc: NDArray[np.floating]
```

###### `nltools.pipelines.ISCResult.method`

```python
method: str
```

###### `nltools.pipelines.ISCResult.metric`

```python
metric: str
```

###### `nltools.pipelines.ISCResult.n_subjects`

```python
n_subjects: int
```

###### `nltools.pipelines.ISCResult.p`

```python
p: NDArray[np.floating]
```



##### Functions###### `nltools.pipelines.ISCResult.summary`

```python
summary() -> str
```

Return formatted summary string.

#### `nltools.pipelines.ISCTerminal`

```python
ISCTerminal(method: str = 'pairwise', metric: str = 'median', n_permute: int = 5000, parallel: str = 'cpu', kwargs: Dict[str, Any] = dict()) -> None
```

ISC terminal for multi-subject pipelines.

Computes inter-subject correlation across subjects in the pipeline.
Uses the ISC permutation test from nltools.algorithms.inference.isc.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ISC computation method: 'pairwise' (default) or 'leave-one-out'. | <code>'pairwise'</code>
`metric` | <code>[str](#str)</code> | Summary statistic: 'median' (default, robust) or 'mean' (Fisher z-transformed). | <code>'median'</code>
`n_permute` | <code>[int](#int)</code> | Number of bootstrap iterations for p-value computation. Default is 5000. | <code>5000</code>
`parallel` | <code>[str](#str)</code> | Parallelization method: 'cpu' (default), 'gpu', or None. | <code>'cpu'</code>
`kwargs` | <code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | Additional arguments passed to isc_permutation_test. | <code>[dict](#dict)()</code>

Examples
--------
>>> terminal = ISCTerminal(method='pairwise', n_permute=1000)
>>> result = terminal.fit_evaluate(data_list)
>>> print(f"ISC: {result.isc:.3f}, p: {result.p:.3f}")

**Functions:**

Name | Description
---- | -----------
[`fit_evaluate`](#nltools.pipelines.ISCTerminal.fit_evaluate) | Compute ISC across subjects.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`kwargs`](#nltools.pipelines.ISCTerminal.kwargs) | <code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | 
[`method`](#nltools.pipelines.ISCTerminal.method) | <code>[str](#str)</code> | 
[`metric`](#nltools.pipelines.ISCTerminal.metric) | <code>[str](#str)</code> | 
[`n_permute`](#nltools.pipelines.ISCTerminal.n_permute) | <code>[int](#int)</code> | 
[`parallel`](#nltools.pipelines.ISCTerminal.parallel) | <code>[str](#str)</code> | 



##### Attributes###### `nltools.pipelines.ISCTerminal.kwargs`

```python
kwargs: Dict[str, Any] = field(default_factory=dict)
```

###### `nltools.pipelines.ISCTerminal.method`

```python
method: str = 'pairwise'
```

###### `nltools.pipelines.ISCTerminal.metric`

```python
metric: str = 'median'
```

###### `nltools.pipelines.ISCTerminal.n_permute`

```python
n_permute: int = 5000
```

###### `nltools.pipelines.ISCTerminal.parallel`

```python
parallel: str = 'cpu'
```



##### Functions###### `nltools.pipelines.ISCTerminal.fit_evaluate`

```python
fit_evaluate(data: list, **kwargs: list) -> 'ISCResult'
```

Compute ISC across subjects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)</code> | List of subject data arrays. Each array should have shape (n_observations, n_features) where n_observations is the same across subjects (e.g., timepoints in fMRI). | *required*

**Returns:**

Type | Description
---- | -----------
<code>'ISCResult'</code> | Result containing ISC values, p-values, and confidence intervals.

#### `nltools.pipelines.MultiSubjectPipeline`

```python
MultiSubjectPipeline(data: List[NDArray], cv: Optional[Any] = None, groups: Optional[NDArray[np.intp]] = None, steps: List[Any] = list(), _is_lazy: bool = False) -> None
```

Pipeline for multi-subject neuroimaging analyses.

Operates on a list of subject data arrays, supporting:
- LOSO (leave-one-subject-out): Train on N-1 subjects, test on 1
- Run-based CV: Split runs within each subject
- Pooling across subjects for group analyses

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[List](#typing.List)[[NDArray](#numpy.typing.NDArray)]</code> | List of subject data arrays, each shape (n_obs, n_voxels). | *required*
`cv` | <code>[Optional](#typing.Optional)[[Any](#typing.Any)]</code> | Cross-validation scheme configuration. | <code>None</code>
`groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for CV splits (e.g., run labels). | <code>None</code>
`steps` | <code>[List](#typing.List)[[Any](#typing.Any)]</code> | Transform steps to apply. | <code>[list](#list)()</code>

Examples
--------
>>> # LOSO CV
>>> pipeline = MultiSubjectPipeline(subject_data, cv=CVScheme(scheme='loso'))
>>> result = pipeline.normalize().predict(y, algorithm='svm')

>>> # Run-based CV across subjects
>>> pipeline = MultiSubjectPipeline(subject_data, cv=CVScheme(scheme='loro'), groups=runs)
>>> result = pipeline.predict(y)

**Functions:**

Name | Description
---- | -----------
[`align`](#nltools.pipelines.MultiSubjectPipeline.align) | Add cross-subject alignment step to pipeline.
[`isc`](#nltools.pipelines.MultiSubjectPipeline.isc) | Compute inter-subject correlation across subjects.
[`normalize`](#nltools.pipelines.MultiSubjectPipeline.normalize) | Add normalization step (per-subject).
[`pipe`](#nltools.pipelines.MultiSubjectPipeline.pipe) | Add custom sklearn transformer.
[`predict`](#nltools.pipelines.MultiSubjectPipeline.predict) | Execute pipeline with CV and return prediction results.
[`reduce`](#nltools.pipelines.MultiSubjectPipeline.reduce) | Add dimensionality reduction step.
[`rsa`](#nltools.pipelines.MultiSubjectPipeline.rsa) | Compute representational similarity analysis.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`cv`](#nltools.pipelines.MultiSubjectPipeline.cv) | <code>[Optional](#typing.Optional)[[Any](#typing.Any)]</code> | 
[`data`](#nltools.pipelines.MultiSubjectPipeline.data) | <code>[List](#typing.List)[[NDArray](#numpy.typing.NDArray)]</code> | 
[`groups`](#nltools.pipelines.MultiSubjectPipeline.groups) | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | 
[`n_steps`](#nltools.pipelines.MultiSubjectPipeline.n_steps) | <code>[int](#int)</code> | Number of transform steps.
[`n_subjects`](#nltools.pipelines.MultiSubjectPipeline.n_subjects) | <code>[int](#int)</code> | Number of subjects in the multi-subject dataset.
[`steps`](#nltools.pipelines.MultiSubjectPipeline.steps) | <code>[List](#typing.List)[[Any](#typing.Any)]</code> | 



##### Attributes###### `nltools.pipelines.MultiSubjectPipeline.cv`

```python
cv: Optional[Any] = None
```

###### `nltools.pipelines.MultiSubjectPipeline.data`

```python
data: List[NDArray]
```

###### `nltools.pipelines.MultiSubjectPipeline.groups`

```python
groups: Optional[NDArray[np.intp]] = None
```

###### `nltools.pipelines.MultiSubjectPipeline.n_steps`

```python
n_steps: int
```

Number of transform steps.

###### `nltools.pipelines.MultiSubjectPipeline.n_subjects`

```python
n_subjects: int
```

Number of subjects in the multi-subject dataset.

###### `nltools.pipelines.MultiSubjectPipeline.steps`

```python
steps: List[Any] = field(default_factory=list)
```



##### Functions###### `nltools.pipelines.MultiSubjectPipeline.align`

```python
align(method: str = 'srm', scheme: str = 'global', n_features: int | None = 50, new_subject: str = 'procrustes', **kwargs: str) -> 'MultiSubjectPipeline'
```

Add cross-subject alignment step to pipeline.

Aligns multi-subject data using SRM or HyperAlignment before
downstream analyses like classification or pooling.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Alignment method: 'srm' (Shared Response Model, reduces dimensionality) or 'hyperalignment' (Procrustes-based, preserves dimensionality). Default is 'srm'. | <code>'srm'</code>
`scheme` | <code>[str](#str)</code> | Spatial scheme. Currently only 'global' is supported. 'searchlight' and 'piecewise' require LocalAlignment (nltools-boll). | <code>'global'</code>
`n_features` | <code>[int](#int) \| None</code> | Number of shared features for SRM. Ignored for hyperalignment. | <code>50</code>
`new_subject` | <code>[str](#str)</code> | Method for aligning held-out subjects in LOSO CV. Default is 'procrustes'. | <code>'procrustes'</code>
`**kwargs` |  | Additional arguments passed to alignment algorithm. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'MultiSubjectPipeline'</code> | New pipeline with alignment step added.

Examples
--------
>>> # SRM alignment before classification
>>> result = (
...     MultiSubjectPipeline(data=subjects, cv=CVScheme(scheme='loso'))
...     .align(method='srm', n_features=50)
...     .predict(y=labels, algorithm='svm')
... )

>>> # Hyperalignment before two-stage GLM
>>> result = (
...     bc.cv(scheme='loso')
...     .align(method='hyperalignment')
...     .fit(model='glm', X=designs)
...     .pool(param='beta')
...     .fit(model='ttest', contrast='A-B')
... )

###### `nltools.pipelines.MultiSubjectPipeline.isc`

```python
isc(method: str = 'pairwise', metric: str = 'median', n_permute: int = 5000, parallel: str = 'cpu', **kwargs: str)
```

Compute inter-subject correlation across subjects.

Executes the pipeline and computes ISC using permutation testing.
Data is transformed through all pipeline steps before ISC computation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ISC computation method: 'pairwise' (average all pairwise correlations) or 'leave-one-out' (correlate each subject with mean of others). Default is 'pairwise'. | <code>'pairwise'</code>
`metric` | <code>[str](#str)</code> | Summary statistic: 'median' (robust, default) or 'mean' (Fisher z-transformed). | <code>'median'</code>
`n_permute` | <code>[int](#int)</code> | Number of bootstrap iterations for p-value computation. Default is 5000. | <code>5000</code>
`parallel` | <code>[str](#str)</code> | Parallelization method: 'cpu', 'gpu', or None. Default is 'cpu'. | <code>'cpu'</code>
`**kwargs` |  | Additional arguments passed to ISCTerminal. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | Result containing ISC values, p-values, and confidence intervals.

Examples
--------
>>> result = (
...     MultiSubjectPipeline(data=subjects, cv=None)
...     .normalize()
...     .isc(method='pairwise', n_permute=1000)
... )
>>> print(f"ISC: {result.isc:.3f}, p: {result.p:.3f}")

###### `nltools.pipelines.MultiSubjectPipeline.normalize`

```python
normalize(method: str = 'zscore', **kwargs: str) -> 'MultiSubjectPipeline'
```

Add normalization step (per-subject).

###### `nltools.pipelines.MultiSubjectPipeline.pipe`

```python
pipe(transformer) -> 'MultiSubjectPipeline'
```

Add custom sklearn transformer.

###### `nltools.pipelines.MultiSubjectPipeline.predict`

```python
predict(y, algorithm: str = 'ridge', **kwargs: str)
```

Execute pipeline with CV and return prediction results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` |  | Target variable. For LOSO, should be (n_subjects,). For run-based CV, should match pooled observations. | *required*
`algorithm` | <code>[str](#str)</code> | Prediction algorithm: 'ridge', 'lasso', 'elastic', 'svr' for regression; 'svm', 'logistic', 'rf' for classification. | <code>'ridge'</code>
`**kwargs` |  | Additional arguments passed to sklearn model constructor. For classification (svm, logistic), use ``class_weight='balanced'`` to handle imbalanced classes. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | Cross-validation results.

Examples
--------
Basic regression with LOSO CV::

    result = pipeline.cv('loso').predict(subject_labels, algorithm='ridge')

Classification with balanced classes::

    result = pipeline.cv('loso').predict(
        group_labels, algorithm='svm', class_weight='balanced'
    )

Logistic regression with regularization::

    result = pipeline.cv('loso').predict(
        binary_labels, algorithm='logistic', C=0.1, class_weight='balanced'
    )

###### `nltools.pipelines.MultiSubjectPipeline.reduce`

```python
reduce(method: str = 'pca', n_components: Optional[int] = None, **kwargs: Optional[int]) -> 'MultiSubjectPipeline'
```

Add dimensionality reduction step.

###### `nltools.pipelines.MultiSubjectPipeline.rsa`

```python
rsa(model_rdm: NDArray, method: str = 'spearman', n_permute: int = 5000, **kwargs: int)
```

Compute representational similarity analysis.

Executes the pipeline and computes RSA correlation between neural
and model RDMs using permutation testing.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model_rdm` | <code>[NDArray](#numpy.typing.NDArray)</code> | Model RDM to correlate with neural RDMs. Should be symmetric matrix or upper triangle (condensed form). | *required*
`method` | <code>[str](#str)</code> | Correlation method: 'spearman' (default), 'pearson', or 'kendall'. | <code>'spearman'</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations for p-value computation. Default is 5000. | <code>5000</code>
`**kwargs` |  | Additional arguments passed to RSATerminal. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | Result containing correlation coefficient and p-value.

Examples
--------
>>> model = np.corrcoef(conditions)  # Theoretical model
>>> result = (
...     MultiSubjectPipeline(data=subjects, cv=None)
...     .normalize()
...     .rsa(model_rdm=model, method='spearman')
... )
>>> print(f"r = {result.correlation:.3f}, p = {result.p_value:.3f}")

#### `nltools.pipelines.NestedCVScheme`

```python
NestedCVScheme(outer: CVScheme, inner: CVScheme) -> None
```

Nested cross-validation for hyperparameter tuning.

Combines an outer CV loop for model evaluation with an inner CV loop
for model selection/hyperparameter tuning. This prevents information
leakage by ensuring the test data in the outer loop is never used
for any model selection decisions.

The outer loop evaluates final model performance on held-out data,
while the inner loop is used for hyperparameter tuning or model
selection within each outer training fold.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`outer` | <code>[CVScheme](#nltools.pipelines.cv.CVScheme)</code> | CVScheme for the outer evaluation loop. | *required*
`inner` | <code>[CVScheme](#nltools.pipelines.cv.CVScheme)</code> | CVScheme for the inner model selection loop. | *required*

**Examples:**

```pycon
>>> # LOSO outer with 3-fold inner for hyperparameter tuning
>>> cv = NestedCVScheme(
...     outer=CVScheme(scheme='loso'),
...     inner=CVScheme(scheme='kfold', k=3)
... )
>>> for outer_train, outer_test, inner_iter in cv.split(data, groups):
...     # Inner loop for hyperparameter tuning
...     for inner_train, inner_val in inner_iter:
...         # These are indices into outer_train
...         actual_train = outer_train[inner_train]
...         actual_val = outer_train[inner_val]
...         # Tune hyperparameters...
...     # Final evaluation on outer_test
```

```pycon
>>> # 5-fold outer with bootstrap inner
>>> cv = NestedCVScheme(
...     outer=CVScheme(scheme='kfold', k=5),
...     inner=CVScheme(scheme='bootstrap', n=100)
... )
```

**Functions:**

Name | Description
---- | -----------
[`n_inner_splits`](#nltools.pipelines.NestedCVScheme.n_inner_splits) | Return number of inner splits per outer fold.
[`n_outer_splits`](#nltools.pipelines.NestedCVScheme.n_outer_splits) | Return number of outer splits.
[`split`](#nltools.pipelines.NestedCVScheme.split) | Generate nested cross-validation splits.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`inner`](#nltools.pipelines.NestedCVScheme.inner) | <code>[CVScheme](#nltools.pipelines.cv.CVScheme)</code> | 
[`outer`](#nltools.pipelines.NestedCVScheme.outer) | <code>[CVScheme](#nltools.pipelines.cv.CVScheme)</code> | 



##### Attributes###### `nltools.pipelines.NestedCVScheme.inner`

```python
inner: CVScheme
```

###### `nltools.pipelines.NestedCVScheme.outer`

```python
outer: CVScheme
```



##### Functions###### `nltools.pipelines.NestedCVScheme.n_inner_splits`

```python
n_inner_splits(data: Any = None, groups: Optional[NDArray[np.intp]] = None) -> int
```

Return number of inner splits per outer fold.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (unused for most schemes). | <code>None</code>
`groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for inner loop. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of inner folds per outer fold.

###### `nltools.pipelines.NestedCVScheme.n_outer_splits`

```python
n_outer_splits(data: Any = None, groups: Optional[NDArray[np.intp]] = None) -> int
```

Return number of outer splits.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (unused for most schemes). | <code>None</code>
`groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for outer loop. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of outer folds.

###### `nltools.pipelines.NestedCVScheme.split`

```python
split(data: Any, groups: Optional[NDArray[np.intp]] = None, inner_groups: Optional[NDArray[np.intp]] = None) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp], Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]]]
```

Generate nested cross-validation splits.

For each outer fold, yields the outer train/test indices and an
iterator over inner train/validation splits. The inner splits are
indices into the outer training set, not the original data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (used for length). | *required*
`groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for outer loop (e.g., subject IDs). Required if outer.scheme is 'loso' or 'loro'. | <code>None</code>
`inner_groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for inner loop. If provided, these are indexed by outer_train to get inner groups. | <code>None</code>

**Yields:**

Type | Description
---- | -----------
<code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Tuple of (outer_train_idx, outer_test_idx, inner_splits_iterator)
<code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | where outer_train_idx and outer_test_idx are arrays of sample
<code>[Iterator](#typing.Iterator)[[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]]</code> | indices, and inner_splits_iterator yields (inner_train, inner_val)
<code>[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [Iterator](#typing.Iterator)[[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]]]</code> | tuples with indices relative to outer_train_idx.

<details class="example" open markdown="1">
<summary>Example</summary>

>>> cv = NestedCVScheme(
...     outer=CVScheme(scheme='kfold', k=5),
...     inner=CVScheme(scheme='kfold', k=3)
... )
>>> for outer_train, outer_test, inner_iter in cv.split(data):
...     for inner_train, inner_val in inner_iter:
...         # inner_train/inner_val are indices into outer_train
...         train_data = data[outer_train[inner_train]]
...         val_data = data[outer_train[inner_val]]

</details>

#### `nltools.pipelines.NormalizeStep`

```python
NormalizeStep(method: str = 'zscore', axis: int = 0, invertible: bool = True) -> None
```

Normalization transform step.

Computes normalization parameters from training data and applies
the transformation to new data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Normalization method: 'zscore' (subtract mean, divide by std) or 'minmax' (scale to [0, 1] range). Default is 'zscore'. | <code>'zscore'</code>
`axis` | <code>[int](#int)</code> | Axis along which to compute statistics. Default 0 (per-feature normalization, treating rows as samples). | <code>0</code>

Examples
--------
>>> import numpy as np
>>> data = np.array([[1, 2], [3, 4], [5, 6]])
>>> step = NormalizeStep(method='zscore')
>>> fitted = step.fit(data)
>>> normalized = fitted.transform(data)
>>> restored = fitted.inverse_transform(normalized)
>>> np.allclose(data, restored)
True

**Functions:**

Name | Description
---- | -----------
[`fit`](#nltools.pipelines.NormalizeStep.fit) | Compute normalization parameters from data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`axis`](#nltools.pipelines.NormalizeStep.axis) | <code>[int](#int)</code> | 
[`invertible`](#nltools.pipelines.NormalizeStep.invertible) | <code>[bool](#bool)</code> | 
[`method`](#nltools.pipelines.NormalizeStep.method) | <code>[str](#str)</code> | 



##### Attributes###### `nltools.pipelines.NormalizeStep.axis`

```python
axis: int = 0
```

###### `nltools.pipelines.NormalizeStep.invertible`

```python
invertible: bool = True
```

###### `nltools.pipelines.NormalizeStep.method`

```python
method: str = 'zscore'
```



##### Functions###### `nltools.pipelines.NormalizeStep.fit`

```python
fit(data: np.ndarray) -> FittedNormalize
```

Compute normalization parameters from data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Training data to compute parameters from. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[FittedNormalize](#nltools.pipelines.steps.FittedNormalize)</code> | Fitted transform that can be applied to new data.

#### `nltools.pipelines.PermutationResult`

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
[`observed`](#nltools.pipelines.PermutationResult.observed) | <code>[CVResult](#nltools.pipelines.results.CVResult)</code> | The result from the real (non-permuted) data.
[`null_distribution`](#nltools.pipelines.PermutationResult.null_distribution) | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | Array of scores from each permutation.
[`p_value`](#nltools.pipelines.PermutationResult.p_value) | <code>[float](#float)</code> | Permutation p-value: proportion of null scores >= observed score.
[`n_permutations`](#nltools.pipelines.PermutationResult.n_permutations) | <code>[int](#int)</code> | Number of permutations performed.

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
[`from_scores`](#nltools.pipelines.PermutationResult.from_scores) | Create PermutationResult from observed result and null scores.
[`summary`](#nltools.pipelines.PermutationResult.summary) | Return formatted summary string.



##### Attributes###### `nltools.pipelines.PermutationResult.n_permutations`

```python
n_permutations: int
```

###### `nltools.pipelines.PermutationResult.null_distribution`

```python
null_distribution: NDArray[np.floating]
```

###### `nltools.pipelines.PermutationResult.null_mean`

```python
null_mean: float
```

Mean of the null distribution.

###### `nltools.pipelines.PermutationResult.null_std`

```python
null_std: float
```

Standard deviation of the null distribution.

###### `nltools.pipelines.PermutationResult.observed`

```python
observed: CVResult
```

###### `nltools.pipelines.PermutationResult.observed_score`

```python
observed_score: float
```

Convenience accessor for observed mean score.

###### `nltools.pipelines.PermutationResult.p_value`

```python
p_value: float
```

###### `nltools.pipelines.PermutationResult.z_score`

```python
z_score: float
```

Z-score of observed relative to null distribution.



##### Functions###### `nltools.pipelines.PermutationResult.from_scores`

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

###### `nltools.pipelines.PermutationResult.summary`

```python
summary() -> str
```

Return formatted summary string.

#### `nltools.pipelines.PipeStep`

```python
PipeStep(transformer: Any = None) -> None
```

Wrapper for sklearn-compatible transformers.

Allows any sklearn transformer with a fit/transform interface to be
used as a pipeline step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`transformer` | <code>[Any](#typing.Any)</code> | An sklearn-compatible transformer instance. Must have fit() and transform() methods. The transformer will be cloned before fitting. | <code>None</code>

Examples
--------
>>> from sklearn.preprocessing import StandardScaler
>>> import numpy as np
>>> data = np.random.randn(100, 10)
>>> step = PipeStep(transformer=StandardScaler())
>>> fitted = step.fit(data)
>>> transformed = fitted.transform(data)
>>> restored = fitted.inverse_transform(transformed)
>>> np.allclose(data, restored)
True

**Functions:**

Name | Description
---- | -----------
[`fit`](#nltools.pipelines.PipeStep.fit) | Fit transformer to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#nltools.pipelines.PipeStep.invertible) | <code>[bool](#bool)</code> | Check if the transformer supports inverse_transform.
[`transformer`](#nltools.pipelines.PipeStep.transformer) | <code>[Any](#typing.Any)</code> | 



##### Attributes###### `nltools.pipelines.PipeStep.invertible`

```python
invertible: bool
```

Check if the transformer supports inverse_transform.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if transformer has inverse_transform method.

###### `nltools.pipelines.PipeStep.transformer`

```python
transformer: Any = None
```



##### Functions###### `nltools.pipelines.PipeStep.fit`

```python
fit(data: np.ndarray) -> FittedPipe
```

Fit transformer to data.

The transformer is cloned before fitting to ensure the original
transformer instance is not modified.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Training data. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[FittedPipe](#nltools.pipelines.steps.FittedPipe)</code> | Fitted transform wrapper.

#### `nltools.pipelines.Pipeline`

```python
Pipeline(data: Any, cv: Optional[CVScheme] = None, steps: List[TransformStep] = list(), _is_lazy: bool = False) -> None
```

Base pipeline for chained transforms with optional cross-validation.

Pipelines enable fluent, chainable data transformations that are executed
within a cross-validation framework. Each transform method returns a new
Pipeline instance (immutable pattern), allowing method chaining without
side effects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Input data (typically ndarray or BrainData). | *required*
`cv` | <code>[Optional](#typing.Optional)[[CVScheme](#nltools.pipelines.base.CVScheme)]</code> | Cross-validation scheme. Required for terminal methods like predict(). | <code>None</code>
`steps` | <code>[List](#typing.List)[[TransformStep](#nltools.pipelines.base.TransformStep)]</code> | List of transform steps (typically not set directly). | <code>[list](#list)()</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`_is_lazy`](#nltools.pipelines.Pipeline._is_lazy) | <code>[bool](#bool)</code> | Whether pipeline is in lazy evaluation mode (future feature).

**Examples:**

```pycon
>>> from sklearn.model_selection import KFold
>>> cv = KFold(n_splits=5)
>>> result = (
...     Pipeline(X, cv=cv)
...     .normalize(method='zscore')
...     .reduce(method='pca', n_components=50)
...     .predict(y, algorithm='ridge')
... )
```

<details class="note" open markdown="1">
<summary>Note</summary>

The pipeline uses an immutable pattern: each method returns a new
Pipeline instance rather than modifying in place. This enables safe method
chaining, branching pipelines from intermediate states, and functional
programming patterns.

</details>

**Functions:**

Name | Description
---- | -----------
[`copy`](#nltools.pipelines.Pipeline.copy) | Create a shallow copy of the pipeline.
[`normalize`](#nltools.pipelines.Pipeline.normalize) | Add a normalization step to the pipeline.
[`pipe`](#nltools.pipelines.Pipeline.pipe) | Add a custom transformer to the pipeline.
[`predict`](#nltools.pipelines.Pipeline.predict) | Execute pipeline with cross-validation and return prediction results.
[`reduce`](#nltools.pipelines.Pipeline.reduce) | Add a dimensionality reduction step to the pipeline.



##### Attributes###### `nltools.pipelines.Pipeline.cv`

```python
cv: Optional[CVScheme] = None
```

###### `nltools.pipelines.Pipeline.data`

```python
data: Any
```

###### `nltools.pipelines.Pipeline.n_steps`

```python
n_steps: int
```

Return number of transform steps.

###### `nltools.pipelines.Pipeline.steps`

```python
steps: List[TransformStep] = field(default_factory=list)
```



##### Functions###### `nltools.pipelines.Pipeline.copy`

```python
copy() -> Pipeline
```

Create a shallow copy of the pipeline.

**Returns:**

Type | Description
---- | -----------
<code>[Pipeline](#nltools.pipelines.base.Pipeline)</code> | New pipeline instance with same configuration.

###### `nltools.pipelines.Pipeline.normalize`

```python
normalize(method: str = 'zscore', **kwargs: Any) -> Pipeline
```

Add a normalization step to the pipeline.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Normalization method. Options: 'zscore', 'minmax', 'robust'. Default is 'zscore'. | <code>'zscore'</code>
`**kwargs` | <code>[Any](#typing.Any)</code> | Additional arguments passed to the normalizer. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[Pipeline](#nltools.pipelines.base.Pipeline)</code> | New pipeline with normalization step added.

**Examples:**

```pycon
>>> pipeline.normalize(method='zscore')
>>> pipeline.normalize(method='minmax', feature_range=(0, 1))
```

###### `nltools.pipelines.Pipeline.pipe`

```python
pipe(transformer: Any) -> Pipeline
```

Add a custom transformer to the pipeline.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`transformer` | <code>[Any](#typing.Any)</code> | Custom transformer with fit/transform interface. Must be compatible with sklearn transformers or implement the TransformStep protocol. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Pipeline](#nltools.pipelines.base.Pipeline)</code> | New pipeline with custom step added.

**Examples:**

```pycon
>>> from sklearn.decomposition import FastICA
>>> pipeline.pipe(FastICA(n_components=20))
```

###### `nltools.pipelines.Pipeline.predict`

```python
predict(y: Any, algorithm: str = 'ridge', **kwargs: Any) -> Any
```

Execute pipeline with cross-validation and return prediction results.

This is a terminal method that triggers pipeline execution.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` | <code>[Any](#typing.Any)</code> | Target variable to predict. | *required*
`algorithm` | <code>[str](#str)</code> | Prediction algorithm. Options: 'ridge', 'lasso', 'svr'. Default is 'ridge'. | <code>'ridge'</code>
`**kwargs` | <code>[Any](#typing.Any)</code> | Additional arguments passed to the predictor. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[Any](#typing.Any)</code> | Cross-validation results containing predictions and metrics.

**Examples:**

```pycon
>>> result = pipeline.predict(y, algorithm='ridge', alpha=1.0)
>>> print(result.summary())
```

###### `nltools.pipelines.Pipeline.reduce`

```python
reduce(method: str = 'pca', n_components: Optional[int] = None, **kwargs: Any) -> Pipeline
```

Add a dimensionality reduction step to the pipeline.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Reduction method. Options: 'pca', 'ica', 'nmf', 'srm'. Default is 'pca'. | <code>'pca'</code>
`n_components` | <code>[Optional](#typing.Optional)[[int](#int)]</code> | Number of components to keep. | <code>None</code>
`**kwargs` | <code>[Any](#typing.Any)</code> | Additional arguments passed to the reducer. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[Pipeline](#nltools.pipelines.base.Pipeline)</code> | New pipeline with reduction step added.

**Examples:**

```pycon
>>> pipeline.reduce(method='pca', n_components=50)
>>> pipeline.reduce(method='srm', n_components=20, n_iter=100)
```

#### `nltools.pipelines.PooledData`

```python
PooledData(data: NDArray, param: str, condition_names: Optional[list[str]] = None, subject_ids: Optional[list[str]] = None, mask: Optional[Any] = None, fitted_state: Optional[Any] = None, save_path: Optional[str] = None) -> None
```

Aggregated data from multiple subjects.

PooledData serves as a checkpoint after first-level analyses,
enabling reusable second-level analyses without re-running
the first-level computations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[NDArray](#numpy.typing.NDArray)</code> | Pooled data array. Shape is (n_subjects, n_voxels) for single parameter or (n_subjects, n_conditions, n_voxels) for multi-condition. | *required*
`param` | <code>[str](#str)</code> | Parameter that was pooled (e.g., 'beta', 'residual', 't'). | *required*
`condition_names` | <code>[Optional](#typing.Optional)[[list](#list)[[str](#str)]]</code> | Names of conditions if multi-condition data. | <code>None</code>
`subject_ids` | <code>[Optional](#typing.Optional)[[list](#list)[[str](#str)]]</code> | Subject identifiers. | <code>None</code>
`fitted_state` | <code>[Optional](#typing.Optional)[[Any](#typing.Any)]</code> | Saved fitted models for repool() functionality. | <code>None</code>
`save_path` | <code>[Optional](#typing.Optional)[[str](#str)]</code> | Path where data was saved. | <code>None</code>

Examples
--------
>>> # Two-stage GLM
>>> pool = bc.fit(model='glm', X=designs).pool(param='beta')
>>> result = pool.fit(model='ttest', contrast='face-house')
>>>
>>> # Reuse for multiple contrasts
>>> result1 = pool.fit(model='ttest', contrast='face-house')
>>> result2 = pool.fit(model='ttest', contrast='face-object')

**Functions:**

Name | Description
---- | -----------
[`cv`](#nltools.pipelines.PooledData.cv) | Create CV pipeline on pooled data.
[`fit`](#nltools.pipelines.PooledData.fit) | Fit second-level statistical model.
[`load`](#nltools.pipelines.PooledData.load) | Load pooled data from disk.
[`repool`](#nltools.pipelines.PooledData.repool) | Re-extract different parameter from saved fitted state.
[`save`](#nltools.pipelines.PooledData.save) | Save pooled data to disk.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`condition_names`](#nltools.pipelines.PooledData.condition_names) | <code>[Optional](#typing.Optional)[[list](#list)[[str](#str)]]</code> | 
[`data`](#nltools.pipelines.PooledData.data) | <code>[NDArray](#numpy.typing.NDArray)</code> | 
[`fitted_state`](#nltools.pipelines.PooledData.fitted_state) | <code>[Optional](#typing.Optional)[[Any](#typing.Any)]</code> | 
[`mask`](#nltools.pipelines.PooledData.mask) | <code>[Optional](#typing.Optional)[[Any](#typing.Any)]</code> | 
[`n_conditions`](#nltools.pipelines.PooledData.n_conditions) | <code>[Optional](#typing.Optional)[[int](#int)]</code> | Number of conditions (None if single-condition).
[`n_subjects`](#nltools.pipelines.PooledData.n_subjects) | <code>[int](#int)</code> | Number of subjects in the pooled dataset (first dimension of data).
[`n_voxels`](#nltools.pipelines.PooledData.n_voxels) | <code>[int](#int)</code> | Number of voxels (last dimension of data array).
[`param`](#nltools.pipelines.PooledData.param) | <code>[str](#str)</code> | 
[`save_path`](#nltools.pipelines.PooledData.save_path) | <code>[Optional](#typing.Optional)[[str](#str)]</code> | 
[`shape`](#nltools.pipelines.PooledData.shape) | <code>[tuple](#tuple)</code> | Shape of the pooled data array as (n_subjects[, n_conditions], n_voxels).
[`subject_ids`](#nltools.pipelines.PooledData.subject_ids) | <code>[Optional](#typing.Optional)[[list](#list)[[str](#str)]]</code> | 



##### Attributes###### `nltools.pipelines.PooledData.condition_names`

```python
condition_names: Optional[list[str]] = None
```

###### `nltools.pipelines.PooledData.data`

```python
data: NDArray
```

###### `nltools.pipelines.PooledData.fitted_state`

```python
fitted_state: Optional[Any] = field(default=None, repr=False)
```

###### `nltools.pipelines.PooledData.mask`

```python
mask: Optional[Any] = field(default=None, repr=False)
```

###### `nltools.pipelines.PooledData.n_conditions`

```python
n_conditions: Optional[int]
```

Number of conditions (None if single-condition).

###### `nltools.pipelines.PooledData.n_subjects`

```python
n_subjects: int
```

Number of subjects in the pooled dataset (first dimension of data).

###### `nltools.pipelines.PooledData.n_voxels`

```python
n_voxels: int
```

Number of voxels (last dimension of data array).

###### `nltools.pipelines.PooledData.param`

```python
param: str
```

###### `nltools.pipelines.PooledData.save_path`

```python
save_path: Optional[str] = None
```

###### `nltools.pipelines.PooledData.shape`

```python
shape: tuple
```

Shape of the pooled data array as (n_subjects[, n_conditions], n_voxels).

###### `nltools.pipelines.PooledData.subject_ids`

```python
subject_ids: Optional[list[str]] = None
```



##### Functions###### `nltools.pipelines.PooledData.cv`

```python
cv(k: Optional[int] = None, scheme: CVSchemeType = 'kfold', **kwargs: CVSchemeType) -> 'Pipeline'
```

Create CV pipeline on pooled data.

Useful for classification on pooled betas.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`k` | <code>[Optional](#typing.Optional)[[int](#int)]</code> | Number of folds. | <code>None</code>
`scheme` | <code>[CVSchemeType](#nltools.pipelines.pool.CVSchemeType)</code> | CV scheme ('kfold', 'loso', 'loro', 'bootstrap'). | <code>'kfold'</code>

**Returns:**

Type | Description
---- | -----------
<code>'Pipeline'</code> | Pipeline for classification on pooled data.

###### `nltools.pipelines.PooledData.fit`

```python
fit(model: str, contrast: Optional[str] = None, contrasts: Optional[list[str]] = None, X: Optional[NDArray] = None, **kwargs: Optional[NDArray]) -> Union['StatResult', 'ResultDict']
```

Fit second-level statistical model.

This is a terminal method - executes immediately (eager).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model` | <code>[str](#str)</code> | Statistical model type: 'ttest' (one-sample or two-sample with X), 'paired_ttest', or 'anova'. | *required*
`contrast` | <code>[Optional](#typing.Optional)[[str](#str)]</code> | Single contrast specification (e.g., 'face-house'). | <code>None</code>
`contrasts` | <code>[Optional](#typing.Optional)[[list](#list)[[str](#str)]]</code> | Multiple contrasts - returns ResultDict. | <code>None</code>
`X` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)]</code> | Design matrix for two-sample tests or ANOVA. | <code>None</code>
`**kwargs` |  | Additional arguments for the statistical test. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[Union](#typing.Union)['StatResult', 'ResultDict']</code> | Statistical results. ResultDict if multiple contrasts specified.

Examples
--------
>>> result = pool.fit(model='ttest', contrast='face-house')
>>> result.t_map.max()

>>> results = pool.fit(model='ttest', contrasts=['A-B', 'A-C', 'B-C'])
>>> results['A-B'].threshold(method='fdr')

###### `nltools.pipelines.PooledData.load`

```python
load(path: str) -> 'PooledData'
```

Load pooled data from disk.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`path` | <code>[str](#str)</code> | Path to saved data. | *required*

**Returns:**

Type | Description
---- | -----------
<code>'PooledData'</code> | Loaded pooled data.

###### `nltools.pipelines.PooledData.repool`

```python
repool(param: str) -> 'PooledData'
```

Re-extract different parameter from saved fitted state.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`param` | <code>[str](#str)</code> | Parameter to extract (e.g., 'residual', 't'). | *required*

**Returns:**

Type | Description
---- | -----------
<code>'PooledData'</code> | New PooledData with the requested parameter.

###### `nltools.pipelines.PooledData.save`

```python
save(path: str) -> None
```

Save pooled data to disk.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`path` | <code>[str](#str)</code> | Output path (directory or .npz file). | *required*

#### `nltools.pipelines.PredictTerminal`

```python
PredictTerminal(y: NDArray, algorithm: str = 'ridge', kwargs: Dict[str, Any] = dict()) -> None
```

Prediction/classification terminal for CV pipelines.

Fits a prediction model on training data and evaluates on test data
within each CV fold.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` | <code>[NDArray](#numpy.typing.NDArray)</code> | Target variable to predict (labels or continuous values). | *required*
`algorithm` | <code>[str](#str)</code> | Prediction algorithm. Regression options: 'ridge' (default, L2), 'lasso' (L1), 'elastic' (L1+L2), 'svr' (kernel-based), 'rf' (random forest, auto-detected). Classification options: 'svm' (kernel-based), 'logistic' (linear), 'rf' (auto-detected for discrete y). | <code>'ridge'</code>
`kwargs` | <code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | Additional arguments passed to the sklearn model constructor. Common kwargs: ``class_weight='balanced'`` for imbalanced classification, ``C`` for regularization strength (svm, logistic), ``alpha`` for regularization strength (ridge, lasso, elastic). | <code>[dict](#dict)()</code>

Examples
--------
Basic classification::

    >>> terminal = PredictTerminal(y=labels, algorithm='svm', kwargs={'C': 1.0})

Balanced classification for imbalanced data::

    >>> terminal = PredictTerminal(
    ...     y=imbalanced_labels,
    ...     algorithm='svm',
    ...     kwargs={'class_weight': 'balanced'}
    ... )

Logistic regression with balanced classes::

    >>> terminal = PredictTerminal(
    ...     y=binary_labels,
    ...     algorithm='logistic',
    ...     kwargs={'class_weight': 'balanced', 'C': 0.1}
    ... )

**Functions:**

Name | Description
---- | -----------
[`fit_evaluate`](#nltools.pipelines.PredictTerminal.fit_evaluate) | Fit model on training data and evaluate on test data.
[`with_y`](#nltools.pipelines.PredictTerminal.with_y) | Create copy with different target variable.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`algorithm`](#nltools.pipelines.PredictTerminal.algorithm) | <code>[str](#str)</code> | 
[`kwargs`](#nltools.pipelines.PredictTerminal.kwargs) | <code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | 
[`y`](#nltools.pipelines.PredictTerminal.y) | <code>[NDArray](#numpy.typing.NDArray)</code> | 



##### Attributes###### `nltools.pipelines.PredictTerminal.algorithm`

```python
algorithm: str = 'ridge'
```

###### `nltools.pipelines.PredictTerminal.kwargs`

```python
kwargs: Dict[str, Any] = field(default_factory=dict)
```

###### `nltools.pipelines.PredictTerminal.y`

```python
y: NDArray
```



##### Functions###### `nltools.pipelines.PredictTerminal.fit_evaluate`

```python
fit_evaluate(train_data: NDArray, test_data: NDArray, train_idx: NDArray[np.intp], test_idx: NDArray[np.intp], fitted_stack: Any) -> 'FoldResult'
```

Fit model on training data and evaluate on test data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`train_data` | <code>[NDArray](#numpy.typing.NDArray)</code> | Transformed training features, shape (n_train, n_features). | *required*
`test_data` | <code>[NDArray](#numpy.typing.NDArray)</code> | Transformed test features, shape (n_test, n_features). | *required*
`train_idx` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Original indices of training samples. | *required*
`test_idx` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Original indices of test samples. | *required*
`fitted_stack` | <code>[Any](#typing.Any)</code> | Stack of fitted transforms for this fold. | *required*

**Returns:**

Type | Description
---- | -----------
<code>'FoldResult'</code> | Result containing score, predictions, indices, and fitted stack.

###### `nltools.pipelines.PredictTerminal.with_y`

```python
with_y(new_y: NDArray) -> 'PredictTerminal'
```

Create copy with different target variable.

Useful for permutation testing.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`new_y` | <code>[NDArray](#numpy.typing.NDArray)</code> | New target variable. | *required*

**Returns:**

Type | Description
---- | -----------
<code>'PredictTerminal'</code> | New terminal with updated y.

#### `nltools.pipelines.RSAResult`

```python
RSAResult(correlation: float, p_value: float, ci: tuple, method: str, n_conditions: int) -> None
```

Result from RSA terminal computation.

Holds representational similarity analysis correlation and p-value.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`correlation`](#nltools.pipelines.RSAResult.correlation) | <code>[float](#float)</code> | Correlation between neural RDM and model RDM.
[`p_value`](#nltools.pipelines.RSAResult.p_value) | <code>[float](#float)</code> | P-value from permutation test.
[`ci`](#nltools.pipelines.RSAResult.ci) | <code>[tuple](#tuple)</code> | Confidence interval (lower, upper).
[`method`](#nltools.pipelines.RSAResult.method) | <code>[str](#str)</code> | Correlation method used (e.g., 'spearman', 'pearson').
[`n_conditions`](#nltools.pipelines.RSAResult.n_conditions) | <code>[int](#int)</code> | Number of conditions/stimuli in the RDM.

**Functions:**

Name | Description
---- | -----------
[`summary`](#nltools.pipelines.RSAResult.summary) | Return formatted summary string.



##### Attributes###### `nltools.pipelines.RSAResult.ci`

```python
ci: tuple
```

###### `nltools.pipelines.RSAResult.correlation`

```python
correlation: float
```

###### `nltools.pipelines.RSAResult.method`

```python
method: str
```

###### `nltools.pipelines.RSAResult.n_conditions`

```python
n_conditions: int
```

###### `nltools.pipelines.RSAResult.p_value`

```python
p_value: float
```



##### Functions###### `nltools.pipelines.RSAResult.summary`

```python
summary() -> str
```

Return formatted summary string.

#### `nltools.pipelines.RSATerminal`

```python
RSATerminal(model_rdm: NDArray, method: str = 'spearman', n_permute: int = 5000, kwargs: Dict[str, Any] = dict()) -> None
```

RSA terminal for multi-subject pipelines.

Computes representational similarity analysis by correlating neural RDMs
with a model RDM.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model_rdm` | <code>[NDArray](#numpy.typing.NDArray)</code> | Model RDM to correlate with neural RDMs. Should be a symmetric matrix or upper triangle (condensed form). | *required*
`method` | <code>[str](#str)</code> | Correlation method: 'spearman' (default), 'pearson', or 'kendall'. | <code>'spearman'</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations for p-value computation. Default is 5000. | <code>5000</code>
`kwargs` | <code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | Additional arguments passed to correlation computation. | <code>[dict](#dict)()</code>

Examples
--------
>>> model = np.random.rand(10, 10)  # 10 conditions
>>> model = (model + model.T) / 2  # Make symmetric
>>> terminal = RSATerminal(model_rdm=model, method='spearman')
>>> result = terminal.fit_evaluate(neural_rdm)

**Functions:**

Name | Description
---- | -----------
[`fit_evaluate`](#nltools.pipelines.RSATerminal.fit_evaluate) | Compute RSA correlation between neural and model RDMs.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`kwargs`](#nltools.pipelines.RSATerminal.kwargs) | <code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | 
[`method`](#nltools.pipelines.RSATerminal.method) | <code>[str](#str)</code> | 
[`model_rdm`](#nltools.pipelines.RSATerminal.model_rdm) | <code>[NDArray](#numpy.typing.NDArray)</code> | 
[`n_permute`](#nltools.pipelines.RSATerminal.n_permute) | <code>[int](#int)</code> | 



##### Attributes###### `nltools.pipelines.RSATerminal.kwargs`

```python
kwargs: Dict[str, Any] = field(default_factory=dict)
```

###### `nltools.pipelines.RSATerminal.method`

```python
method: str = 'spearman'
```

###### `nltools.pipelines.RSATerminal.model_rdm`

```python
model_rdm: NDArray
```

###### `nltools.pipelines.RSATerminal.n_permute`

```python
n_permute: int = 5000
```



##### Functions###### `nltools.pipelines.RSATerminal.fit_evaluate`

```python
fit_evaluate(data: NDArray, **kwargs: NDArray) -> 'RSAResult'
```

Compute RSA correlation between neural and model RDMs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[NDArray](#numpy.typing.NDArray)</code> | Neural data to compute RDM from, or pre-computed RDM. If 2D square, treated as RDM (upper triangle extracted). If 1D, treated as condensed RDM. If 2D non-square (n_conditions, n_features), RDM is computed using correlation distance. | *required*

**Returns:**

Type | Description
---- | -----------
<code>'RSAResult'</code> | Result containing correlation coefficient and p-value.

#### `nltools.pipelines.ReduceStep`

```python
ReduceStep(method: str = 'pca', n_components: Optional[int] = None, random_state: Optional[int] = None) -> None
```

Dimensionality reduction step.

Fits a dimensionality reduction model to training data and transforms
new data to the reduced space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Reduction method: 'pca' (Principal Component Analysis, invertible) or 'ica' (Independent Component Analysis, not invertible). Default is 'pca'. | <code>'pca'</code>
`n_components` | <code>[Optional](#typing.Optional)[[int](#int)]</code> | Number of components to keep. If None, keeps all components. | <code>None</code>
`random_state` | <code>[Optional](#typing.Optional)[[int](#int)]</code> | Random seed for reproducibility. | <code>None</code>

Examples
--------
>>> import numpy as np
>>> data = np.random.randn(100, 50)
>>> step = ReduceStep(method='pca', n_components=10)
>>> fitted = step.fit(data)
>>> reduced = fitted.transform(data)
>>> reduced.shape
(100, 10)

**Functions:**

Name | Description
---- | -----------
[`fit`](#nltools.pipelines.ReduceStep.fit) | Fit reduction model to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#nltools.pipelines.ReduceStep.invertible) | <code>[bool](#bool)</code> | Check if the reduction method supports inverse transform.
[`method`](#nltools.pipelines.ReduceStep.method) | <code>[str](#str)</code> | 
[`n_components`](#nltools.pipelines.ReduceStep.n_components) | <code>[Optional](#typing.Optional)[[int](#int)]</code> | 
[`random_state`](#nltools.pipelines.ReduceStep.random_state) | <code>[Optional](#typing.Optional)[[int](#int)]</code> | 



##### Attributes###### `nltools.pipelines.ReduceStep.invertible`

```python
invertible: bool
```

Check if the reduction method supports inverse transform.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if method is 'pca', False otherwise.

###### `nltools.pipelines.ReduceStep.method`

```python
method: str = 'pca'
```

###### `nltools.pipelines.ReduceStep.n_components`

```python
n_components: Optional[int] = None
```

###### `nltools.pipelines.ReduceStep.random_state`

```python
random_state: Optional[int] = None
```



##### Functions###### `nltools.pipelines.ReduceStep.fit`

```python
fit(data: np.ndarray) -> FittedReduce
```

Fit reduction model to data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Training data, shape (n_samples, n_features). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[FittedReduce](#nltools.pipelines.steps.FittedReduce)</code> | Fitted transform that can be applied to new data.

#### `nltools.pipelines.ResultDict`

Bases: <code>[dict](#dict)</code>

Dictionary of StatResults, one per contrast.

Provides convenience methods for batch operations.

**Functions:**

Name | Description
---- | -----------
[`threshold_all`](#nltools.pipelines.ResultDict.threshold_all) | Apply thresholding to all results.



##### Functions###### `nltools.pipelines.ResultDict.threshold_all`

```python
threshold_all(method: str = 'fdr', alpha: float = 0.05) -> 'ResultDict'
```

Apply thresholding to all results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Correction method: 'fdr', 'bonferroni', or 'uncorrected'. | <code>'fdr'</code>
`alpha` | <code>[float](#float)</code> | Significance threshold. | <code>0.05</code>

**Returns:**

Type | Description
---- | -----------
<code>'ResultDict'</code> | New dict with thresholded results.

#### `nltools.pipelines.StatResult`

```python
StatResult(t_map: Optional[NDArray] = None, f_map: Optional[NDArray] = None, p_map: Optional[NDArray] = None, contrast: Optional[str] = None, df: Optional[int] = None) -> None
```

Result of statistical test.

Holds statistical maps and provides thresholding utilities.

**Functions:**

Name | Description
---- | -----------
[`threshold`](#nltools.pipelines.StatResult.threshold) | Apply multiple comparison correction.
[`to_nifti`](#nltools.pipelines.StatResult.to_nifti) | Save as NIfTI file.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`contrast`](#nltools.pipelines.StatResult.contrast) | <code>[Optional](#typing.Optional)[[str](#str)]</code> | 
[`df`](#nltools.pipelines.StatResult.df) | <code>[Optional](#typing.Optional)[[int](#int)]</code> | 
[`f_map`](#nltools.pipelines.StatResult.f_map) | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)]</code> | 
[`p_map`](#nltools.pipelines.StatResult.p_map) | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)]</code> | 
[`t_map`](#nltools.pipelines.StatResult.t_map) | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)]</code> | 



##### Attributes###### `nltools.pipelines.StatResult.contrast`

```python
contrast: Optional[str] = None
```

###### `nltools.pipelines.StatResult.df`

```python
df: Optional[int] = None
```

###### `nltools.pipelines.StatResult.f_map`

```python
f_map: Optional[NDArray] = None
```

###### `nltools.pipelines.StatResult.p_map`

```python
p_map: Optional[NDArray] = None
```

###### `nltools.pipelines.StatResult.t_map`

```python
t_map: Optional[NDArray] = None
```



##### Functions###### `nltools.pipelines.StatResult.threshold`

```python
threshold(method: str = 'fdr', alpha: float = 0.05) -> 'StatResult'
```

Apply multiple comparison correction.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Correction method: 'fdr', 'bonferroni', or 'uncorrected'. | <code>'fdr'</code>
`alpha` | <code>[float](#float)</code> | Significance threshold. | <code>0.05</code>

**Returns:**

Type | Description
---- | -----------
<code>'StatResult'</code> | New result with thresholded maps.

###### `nltools.pipelines.StatResult.to_nifti`

```python
to_nifti(path: str, mask: str = None) -> None
```

Save as NIfTI file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`path` | <code>[str](#str)</code> | Output path. | *required*
`mask` |  | Mask to use for reconstruction. | <code>None</code>

#### `nltools.pipelines.Terminal`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for terminal operations that end a pipeline.

Terminals perform the final computation (e.g., prediction, similarity)
and produce results for each CV fold.

**Functions:**

Name | Description
---- | -----------
[`fit_evaluate`](#nltools.pipelines.Terminal.fit_evaluate) | Fit on training data and evaluate on test data.



##### Functions###### `nltools.pipelines.Terminal.fit_evaluate`

```python
fit_evaluate(train_data: Any, test_data: Any, train_idx: NDArray[np.intp], test_idx: NDArray[np.intp], fitted_stack: FittedStack) -> Any
```

Fit on training data and evaluate on test data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`train_data` | <code>[Any](#typing.Any)</code> | Transformed training data. | *required*
`test_data` | <code>[Any](#typing.Any)</code> | Transformed test data. | *required*
`train_idx` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Original training indices. | *required*
`test_idx` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Original test indices. | *required*
`fitted_stack` | <code>[FittedStack](#nltools.pipelines.base.FittedStack)</code> | Stack of fitted transforms for inverse transform support. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Any](#typing.Any)</code> | Fold result (structure depends on terminal type).

#### `nltools.pipelines.TransformStep`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for pipeline transform steps.

A transform step defines a transformation that can be fitted to data.
Steps are added to a Pipeline and executed sequentially during CV.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#nltools.pipelines.TransformStep.invertible) | <code>[bool](#bool)</code> | Whether this transform supports inverse_transform.

Examples
--------
>>> class MyStep:
...     invertible = True
...     def fit(self, data):
...         return MyFittedTransform(learned_params)

**Functions:**

Name | Description
---- | -----------
[`fit`](#nltools.pipelines.TransformStep.fit) | Fit the transform to data.



##### Attributes###### `nltools.pipelines.TransformStep.invertible`

```python
invertible: bool
```



##### Functions###### `nltools.pipelines.TransformStep.fit`

```python
fit(data: Any) -> FittedTransform
```

Fit the transform to data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Training data to fit on. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[FittedTransform](#nltools.pipelines.base.FittedTransform)</code> | Fitted transform object that can transform new data.



### Modules#### `nltools.pipelines.base`

Pipeline base infrastructure for nltools.

This module provides the foundational classes and protocols for building
chainable transform pipelines with optional cross-validation support.

The design follows an immutable pattern where each transform method returns
a new Pipeline instance, enabling fluent method chaining without side effects.

Example
-------
>>> pipeline = (
...     Pipeline(data, cv=kfold)
...     .normalize(method='zscore')
...     .reduce(method='pca', n_components=50)
...     .predict(y, algorithm='ridge')
... )

**Classes:**

Name | Description
---- | -----------
[`CVScheme`](#nltools.pipelines.base.CVScheme) | Protocol for cross-validation schemes.
[`FittedStack`](#nltools.pipelines.base.FittedStack) | Collection of fitted transforms for inverse transform support.
[`FittedTransform`](#nltools.pipelines.base.FittedTransform) | Protocol for fitted transform objects.
[`Pipeline`](#nltools.pipelines.base.Pipeline) | Base pipeline for chained transforms with optional cross-validation.
[`Terminal`](#nltools.pipelines.base.Terminal) | Protocol for terminal operations that end a pipeline.
[`TransformStep`](#nltools.pipelines.base.TransformStep) | Protocol for pipeline transform steps.



##### Attributes

##### Classes###### `nltools.pipelines.base.CVScheme`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for cross-validation schemes.

Compatible with scikit-learn CV splitters and custom implementations.

**Functions:**

Name | Description
---- | -----------
[`split`](#nltools.pipelines.base.CVScheme.split) | Generate train/test index splits.



####### Functions######## `nltools.pipelines.base.CVScheme.split`

```python
split(data: Any) -> Any
```

Generate train/test index splits.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (used to determine n_samples). | *required*

**Yields:**

Type | Description
---- | -----------
<code>[Any](#typing.Any)</code> | Tuple of (train_idx, test_idx) arrays of indices for each fold.

###### `nltools.pipelines.base.FittedStack`

```python
FittedStack(steps: List[FittedTransform] = list()) -> None
```

Collection of fitted transforms for inverse transform support.

Maintains the sequence of fitted transforms from a pipeline execution,
enabling inverse transformation back to the original data space.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`steps`](#nltools.pipelines.base.FittedStack.steps) | <code>[List](#typing.List)[[FittedTransform](#nltools.pipelines.base.FittedTransform)]</code> | Ordered list of fitted transforms.

Examples
--------
>>> stack = FittedStack()
>>> stack.append(fitted_pca)
>>> stack.append(fitted_normalize)
>>> original_space = stack.inverse_transform(predictions)

**Functions:**

Name | Description
---- | -----------
[`append`](#nltools.pipelines.base.FittedStack.append) | Add a fitted transform to the stack.
[`inverse_transform`](#nltools.pipelines.base.FittedStack.inverse_transform) | Apply inverse transforms in reverse order.



####### Attributes######## `nltools.pipelines.base.FittedStack.is_fully_invertible`

```python
is_fully_invertible: bool
```

Check if all steps support inverse transform.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if all steps have callable inverse_transform methods.

######## `nltools.pipelines.base.FittedStack.steps`

```python
steps: List[FittedTransform] = field(default_factory=list)
```



####### Functions######## `nltools.pipelines.base.FittedStack.append`

```python
append(fitted_step: FittedTransform) -> None
```

Add a fitted transform to the stack.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fitted_step` | <code>[FittedTransform](#nltools.pipelines.base.FittedTransform)</code> | Fitted transform to append. | *required*

######## `nltools.pipelines.base.FittedStack.inverse_transform`

```python
inverse_transform(data: Any) -> Any
```

Apply inverse transforms in reverse order.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to inverse transform. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Any](#typing.Any)</code> | Data transformed back toward original space.

<details class="note" open markdown="1">
<summary>Note</summary>

Steps without ``inverse_transform`` are silently skipped.
Use ``is_fully_invertible`` to check if all steps support inversion.

</details>

###### `nltools.pipelines.base.FittedTransform`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for fitted transform objects.

A fitted transform holds the learned parameters from fitting on training
data and can apply the transformation to new data.

<details class="note" open markdown="1">
<summary>Note</summary>

Not all transforms are invertible. Check the parent TransformStep's
``invertible`` attribute or use ``hasattr`` before calling ``inverse_transform``.

</details>

**Functions:**

Name | Description
---- | -----------
[`inverse_transform`](#nltools.pipelines.base.FittedTransform.inverse_transform) | Apply the inverse transformation to data.
[`transform`](#nltools.pipelines.base.FittedTransform.transform) | Apply the learned transformation to data.



####### Functions######## `nltools.pipelines.base.FittedTransform.inverse_transform`

```python
inverse_transform(data: Any) -> Any
```

Apply the inverse transformation to data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to inverse transform. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Any](#typing.Any)</code> | Data in original space.

######## `nltools.pipelines.base.FittedTransform.transform`

```python
transform(data: Any) -> Any
```

Apply the learned transformation to data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to transform (typically ndarray or BrainData). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Any](#typing.Any)</code> | Transformed data.

###### `nltools.pipelines.base.Pipeline`

```python
Pipeline(data: Any, cv: Optional[CVScheme] = None, steps: List[TransformStep] = list(), _is_lazy: bool = False) -> None
```

Base pipeline for chained transforms with optional cross-validation.

Pipelines enable fluent, chainable data transformations that are executed
within a cross-validation framework. Each transform method returns a new
Pipeline instance (immutable pattern), allowing method chaining without
side effects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Input data (typically ndarray or BrainData). | *required*
`cv` | <code>[Optional](#typing.Optional)[[CVScheme](#nltools.pipelines.base.CVScheme)]</code> | Cross-validation scheme. Required for terminal methods like predict(). | <code>None</code>
`steps` | <code>[List](#typing.List)[[TransformStep](#nltools.pipelines.base.TransformStep)]</code> | List of transform steps (typically not set directly). | <code>[list](#list)()</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`_is_lazy`](#nltools.pipelines.base.Pipeline._is_lazy) | <code>[bool](#bool)</code> | Whether pipeline is in lazy evaluation mode (future feature).

**Examples:**

```pycon
>>> from sklearn.model_selection import KFold
>>> cv = KFold(n_splits=5)
>>> result = (
...     Pipeline(X, cv=cv)
...     .normalize(method='zscore')
...     .reduce(method='pca', n_components=50)
...     .predict(y, algorithm='ridge')
... )
```

<details class="note" open markdown="1">
<summary>Note</summary>

The pipeline uses an immutable pattern: each method returns a new
Pipeline instance rather than modifying in place. This enables safe method
chaining, branching pipelines from intermediate states, and functional
programming patterns.

</details>

**Functions:**

Name | Description
---- | -----------
[`copy`](#nltools.pipelines.base.Pipeline.copy) | Create a shallow copy of the pipeline.
[`normalize`](#nltools.pipelines.base.Pipeline.normalize) | Add a normalization step to the pipeline.
[`pipe`](#nltools.pipelines.base.Pipeline.pipe) | Add a custom transformer to the pipeline.
[`predict`](#nltools.pipelines.base.Pipeline.predict) | Execute pipeline with cross-validation and return prediction results.
[`reduce`](#nltools.pipelines.base.Pipeline.reduce) | Add a dimensionality reduction step to the pipeline.



####### Attributes######## `nltools.pipelines.base.Pipeline.cv`

```python
cv: Optional[CVScheme] = None
```

######## `nltools.pipelines.base.Pipeline.data`

```python
data: Any
```

######## `nltools.pipelines.base.Pipeline.n_steps`

```python
n_steps: int
```

Return number of transform steps.

######## `nltools.pipelines.base.Pipeline.steps`

```python
steps: List[TransformStep] = field(default_factory=list)
```



####### Functions######## `nltools.pipelines.base.Pipeline.copy`

```python
copy() -> Pipeline
```

Create a shallow copy of the pipeline.

**Returns:**

Type | Description
---- | -----------
<code>[Pipeline](#nltools.pipelines.base.Pipeline)</code> | New pipeline instance with same configuration.

######## `nltools.pipelines.base.Pipeline.normalize`

```python
normalize(method: str = 'zscore', **kwargs: Any) -> Pipeline
```

Add a normalization step to the pipeline.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Normalization method. Options: 'zscore', 'minmax', 'robust'. Default is 'zscore'. | <code>'zscore'</code>
`**kwargs` | <code>[Any](#typing.Any)</code> | Additional arguments passed to the normalizer. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[Pipeline](#nltools.pipelines.base.Pipeline)</code> | New pipeline with normalization step added.

**Examples:**

```pycon
>>> pipeline.normalize(method='zscore')
>>> pipeline.normalize(method='minmax', feature_range=(0, 1))
```

######## `nltools.pipelines.base.Pipeline.pipe`

```python
pipe(transformer: Any) -> Pipeline
```

Add a custom transformer to the pipeline.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`transformer` | <code>[Any](#typing.Any)</code> | Custom transformer with fit/transform interface. Must be compatible with sklearn transformers or implement the TransformStep protocol. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Pipeline](#nltools.pipelines.base.Pipeline)</code> | New pipeline with custom step added.

**Examples:**

```pycon
>>> from sklearn.decomposition import FastICA
>>> pipeline.pipe(FastICA(n_components=20))
```

######## `nltools.pipelines.base.Pipeline.predict`

```python
predict(y: Any, algorithm: str = 'ridge', **kwargs: Any) -> Any
```

Execute pipeline with cross-validation and return prediction results.

This is a terminal method that triggers pipeline execution.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` | <code>[Any](#typing.Any)</code> | Target variable to predict. | *required*
`algorithm` | <code>[str](#str)</code> | Prediction algorithm. Options: 'ridge', 'lasso', 'svr'. Default is 'ridge'. | <code>'ridge'</code>
`**kwargs` | <code>[Any](#typing.Any)</code> | Additional arguments passed to the predictor. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[Any](#typing.Any)</code> | Cross-validation results containing predictions and metrics.

**Examples:**

```pycon
>>> result = pipeline.predict(y, algorithm='ridge', alpha=1.0)
>>> print(result.summary())
```

######## `nltools.pipelines.base.Pipeline.reduce`

```python
reduce(method: str = 'pca', n_components: Optional[int] = None, **kwargs: Any) -> Pipeline
```

Add a dimensionality reduction step to the pipeline.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Reduction method. Options: 'pca', 'ica', 'nmf', 'srm'. Default is 'pca'. | <code>'pca'</code>
`n_components` | <code>[Optional](#typing.Optional)[[int](#int)]</code> | Number of components to keep. | <code>None</code>
`**kwargs` | <code>[Any](#typing.Any)</code> | Additional arguments passed to the reducer. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[Pipeline](#nltools.pipelines.base.Pipeline)</code> | New pipeline with reduction step added.

**Examples:**

```pycon
>>> pipeline.reduce(method='pca', n_components=50)
>>> pipeline.reduce(method='srm', n_components=20, n_iter=100)
```

###### `nltools.pipelines.base.Terminal`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for terminal operations that end a pipeline.

Terminals perform the final computation (e.g., prediction, similarity)
and produce results for each CV fold.

**Functions:**

Name | Description
---- | -----------
[`fit_evaluate`](#nltools.pipelines.base.Terminal.fit_evaluate) | Fit on training data and evaluate on test data.



####### Functions######## `nltools.pipelines.base.Terminal.fit_evaluate`

```python
fit_evaluate(train_data: Any, test_data: Any, train_idx: NDArray[np.intp], test_idx: NDArray[np.intp], fitted_stack: FittedStack) -> Any
```

Fit on training data and evaluate on test data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`train_data` | <code>[Any](#typing.Any)</code> | Transformed training data. | *required*
`test_data` | <code>[Any](#typing.Any)</code> | Transformed test data. | *required*
`train_idx` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Original training indices. | *required*
`test_idx` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Original test indices. | *required*
`fitted_stack` | <code>[FittedStack](#nltools.pipelines.base.FittedStack)</code> | Stack of fitted transforms for inverse transform support. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[Any](#typing.Any)</code> | Fold result (structure depends on terminal type).

###### `nltools.pipelines.base.TransformStep`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for pipeline transform steps.

A transform step defines a transformation that can be fitted to data.
Steps are added to a Pipeline and executed sequentially during CV.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#nltools.pipelines.base.TransformStep.invertible) | <code>[bool](#bool)</code> | Whether this transform supports inverse_transform.

Examples
--------
>>> class MyStep:
...     invertible = True
...     def fit(self, data):
...         return MyFittedTransform(learned_params)

**Functions:**

Name | Description
---- | -----------
[`fit`](#nltools.pipelines.base.TransformStep.fit) | Fit the transform to data.



####### Attributes######## `nltools.pipelines.base.TransformStep.invertible`

```python
invertible: bool
```



####### Functions######## `nltools.pipelines.base.TransformStep.fit`

```python
fit(data: Any) -> FittedTransform
```

Fit the transform to data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Training data to fit on. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[FittedTransform](#nltools.pipelines.base.FittedTransform)</code> | Fitted transform object that can transform new data.

#### `nltools.pipelines.cv`

Cross-validation scheme configuration for nltools pipelines.

This module provides a unified interface for configuring cross-validation
strategies used across nltools analysis pipelines.

**Classes:**

Name | Description
---- | -----------
[`CVScheme`](#nltools.pipelines.cv.CVScheme) | Cross-validation scheme configuration.
[`NestedCVScheme`](#nltools.pipelines.cv.NestedCVScheme) | Nested cross-validation for hyperparameter tuning.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`CVSchemeType`](#nltools.pipelines.cv.CVSchemeType) |  | 



##### Attributes###### `nltools.pipelines.cv.CVSchemeType`

```python
CVSchemeType = Literal['kfold', 'loso', 'loro', 'bootstrap', 'permutation']
```



##### Classes###### `nltools.pipelines.cv.CVScheme`

```python
CVScheme(k: Optional[int] = None, scheme: CVSchemeType = 'kfold', split_by: Optional[str] = None, n: int = 1000, random_state: Optional[int] = None) -> None
```

Cross-validation scheme configuration.

Supports multiple CV strategies:
- kfold: k-fold cross-validation
- loso: leave-one-subject-out (for multi-subject)
- loro: leave-one-run-out
- bootstrap: bootstrap resampling
- permutation: permutation testing (shuffles targets)

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`k` | <code>[Optional](#typing.Optional)[[int](#int)]</code> | Number of folds (for kfold scheme). Defaults to 5 if scheme is 'kfold'. | <code>None</code>
`scheme` | <code>[CVSchemeType](#nltools.pipelines.cv.CVSchemeType)</code> | CV scheme type. One of 'kfold', 'loso', 'loro', 'bootstrap', or 'permutation'. | <code>'kfold'</code>
`split_by` | <code>[Optional](#typing.Optional)[[str](#str)]</code> | Attribute to split by ('runs', 'subjects', 'sessions'). Used for documentation purposes with loso/loro schemes. | <code>None</code>
`n` | <code>[int](#int)</code> | Number of bootstrap iterations (for bootstrap scheme). Defaults to 1000. | <code>1000</code>
`random_state` | <code>[Optional](#typing.Optional)[[int](#int)]</code> | Random seed for reproducibility. If provided, sets the numpy random seed during initialization. | <code>None</code>

**Examples:**

```pycon
>>> # 5-fold cross-validation
>>> cv = CVScheme(scheme='kfold', k=5)
>>> for train_idx, test_idx in cv.split(data):
...     # train and evaluate model
...     pass
```

```pycon
>>> # Leave-one-subject-out
>>> cv = CVScheme(scheme='loso', split_by='subjects')
>>> for train_idx, test_idx in cv.split(data, groups=subject_ids):
...     pass
```

```pycon
>>> # Bootstrap with 500 iterations
>>> cv = CVScheme(scheme='bootstrap', n=500, random_state=42)
```

```pycon
>>> # Permutation testing with 1000 permutations
>>> cv = CVScheme(scheme='permutation', n=1000, random_state=42)
```

**Functions:**

Name | Description
---- | -----------
[`n_splits`](#nltools.pipelines.cv.CVScheme.n_splits) | Return number of splits.
[`split`](#nltools.pipelines.cv.CVScheme.split) | Generate train/test indices for each fold.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`is_loro`](#nltools.pipelines.cv.CVScheme.is_loro) | <code>[bool](#bool)</code> | Check if this is leave-one-run-out.
[`is_loso`](#nltools.pipelines.cv.CVScheme.is_loso) | <code>[bool](#bool)</code> | Check if this is leave-one-subject-out.
[`k`](#nltools.pipelines.cv.CVScheme.k) | <code>[Optional](#typing.Optional)[[int](#int)]</code> | 
[`n`](#nltools.pipelines.cv.CVScheme.n) | <code>[int](#int)</code> | 
[`random_state`](#nltools.pipelines.cv.CVScheme.random_state) | <code>[Optional](#typing.Optional)[[int](#int)]</code> | 
[`scheme`](#nltools.pipelines.cv.CVScheme.scheme) | <code>[CVSchemeType](#nltools.pipelines.cv.CVSchemeType)</code> | 
[`split_by`](#nltools.pipelines.cv.CVScheme.split_by) | <code>[Optional](#typing.Optional)[[str](#str)]</code> | 



####### Attributes######## `nltools.pipelines.cv.CVScheme.is_loro`

```python
is_loro: bool
```

Check if this is leave-one-run-out.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if scheme is 'loro', False otherwise.

######## `nltools.pipelines.cv.CVScheme.is_loso`

```python
is_loso: bool
```

Check if this is leave-one-subject-out.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if scheme is 'loso', False otherwise.

######## `nltools.pipelines.cv.CVScheme.k`

```python
k: Optional[int] = None
```

######## `nltools.pipelines.cv.CVScheme.n`

```python
n: int = 1000
```

######## `nltools.pipelines.cv.CVScheme.random_state`

```python
random_state: Optional[int] = None
```

######## `nltools.pipelines.cv.CVScheme.scheme`

```python
scheme: CVSchemeType = 'kfold'
```

######## `nltools.pipelines.cv.CVScheme.split_by`

```python
split_by: Optional[str] = None
```



####### Functions######## `nltools.pipelines.cv.CVScheme.n_splits`

```python
n_splits(data: Any = None, groups: Optional[NDArray[np.intp]] = None) -> int
```

Return number of splits.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (unused for most schemes, kept for API consistency). | <code>None</code>
`groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for grouped CV. Required for 'loso' and 'loro'. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of splits/folds that will be generated.

######## `nltools.pipelines.cv.CVScheme.split`

```python
split(data: Any, groups: Optional[NDArray[np.intp]] = None) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]
```

Generate train/test indices for each fold.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (used for length). Can be any object with __len__ or a numpy array with shape attribute. | *required*
`groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for grouped CV (runs, subjects, etc.). Required for 'loso' and 'loro' schemes. | <code>None</code>

**Yields:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Tuple of (train_indices, test_indices) for each fold.

###### `nltools.pipelines.cv.NestedCVScheme`

```python
NestedCVScheme(outer: CVScheme, inner: CVScheme) -> None
```

Nested cross-validation for hyperparameter tuning.

Combines an outer CV loop for model evaluation with an inner CV loop
for model selection/hyperparameter tuning. This prevents information
leakage by ensuring the test data in the outer loop is never used
for any model selection decisions.

The outer loop evaluates final model performance on held-out data,
while the inner loop is used for hyperparameter tuning or model
selection within each outer training fold.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`outer` | <code>[CVScheme](#nltools.pipelines.cv.CVScheme)</code> | CVScheme for the outer evaluation loop. | *required*
`inner` | <code>[CVScheme](#nltools.pipelines.cv.CVScheme)</code> | CVScheme for the inner model selection loop. | *required*

**Examples:**

```pycon
>>> # LOSO outer with 3-fold inner for hyperparameter tuning
>>> cv = NestedCVScheme(
...     outer=CVScheme(scheme='loso'),
...     inner=CVScheme(scheme='kfold', k=3)
... )
>>> for outer_train, outer_test, inner_iter in cv.split(data, groups):
...     # Inner loop for hyperparameter tuning
...     for inner_train, inner_val in inner_iter:
...         # These are indices into outer_train
...         actual_train = outer_train[inner_train]
...         actual_val = outer_train[inner_val]
...         # Tune hyperparameters...
...     # Final evaluation on outer_test
```

```pycon
>>> # 5-fold outer with bootstrap inner
>>> cv = NestedCVScheme(
...     outer=CVScheme(scheme='kfold', k=5),
...     inner=CVScheme(scheme='bootstrap', n=100)
... )
```

**Functions:**

Name | Description
---- | -----------
[`n_inner_splits`](#nltools.pipelines.cv.NestedCVScheme.n_inner_splits) | Return number of inner splits per outer fold.
[`n_outer_splits`](#nltools.pipelines.cv.NestedCVScheme.n_outer_splits) | Return number of outer splits.
[`split`](#nltools.pipelines.cv.NestedCVScheme.split) | Generate nested cross-validation splits.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`inner`](#nltools.pipelines.cv.NestedCVScheme.inner) | <code>[CVScheme](#nltools.pipelines.cv.CVScheme)</code> | 
[`outer`](#nltools.pipelines.cv.NestedCVScheme.outer) | <code>[CVScheme](#nltools.pipelines.cv.CVScheme)</code> | 



####### Attributes######## `nltools.pipelines.cv.NestedCVScheme.inner`

```python
inner: CVScheme
```

######## `nltools.pipelines.cv.NestedCVScheme.outer`

```python
outer: CVScheme
```



####### Functions######## `nltools.pipelines.cv.NestedCVScheme.n_inner_splits`

```python
n_inner_splits(data: Any = None, groups: Optional[NDArray[np.intp]] = None) -> int
```

Return number of inner splits per outer fold.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (unused for most schemes). | <code>None</code>
`groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for inner loop. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of inner folds per outer fold.

######## `nltools.pipelines.cv.NestedCVScheme.n_outer_splits`

```python
n_outer_splits(data: Any = None, groups: Optional[NDArray[np.intp]] = None) -> int
```

Return number of outer splits.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (unused for most schemes). | <code>None</code>
`groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for outer loop. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of outer folds.

######## `nltools.pipelines.cv.NestedCVScheme.split`

```python
split(data: Any, groups: Optional[NDArray[np.intp]] = None, inner_groups: Optional[NDArray[np.intp]] = None) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp], Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]]]
```

Generate nested cross-validation splits.

For each outer fold, yields the outer train/test indices and an
iterator over inner train/validation splits. The inner splits are
indices into the outer training set, not the original data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (used for length). | *required*
`groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for outer loop (e.g., subject IDs). Required if outer.scheme is 'loso' or 'loro'. | <code>None</code>
`inner_groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for inner loop. If provided, these are indexed by outer_train to get inner groups. | <code>None</code>

**Yields:**

Type | Description
---- | -----------
<code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Tuple of (outer_train_idx, outer_test_idx, inner_splits_iterator)
<code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | where outer_train_idx and outer_test_idx are arrays of sample
<code>[Iterator](#typing.Iterator)[[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]]</code> | indices, and inner_splits_iterator yields (inner_train, inner_val)
<code>[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [Iterator](#typing.Iterator)[[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]]]</code> | tuples with indices relative to outer_train_idx.

<details class="example" open markdown="1">
<summary>Example</summary>

>>> cv = NestedCVScheme(
...     outer=CVScheme(scheme='kfold', k=5),
...     inner=CVScheme(scheme='kfold', k=3)
... )
>>> for outer_train, outer_test, inner_iter in cv.split(data):
...     for inner_train, inner_val in inner_iter:
...         # inner_train/inner_val are indices into outer_train
...         train_data = data[outer_train[inner_train]]
...         val_data = data[outer_train[inner_val]]

</details>

#### `nltools.pipelines.multi_subject`

Multi-subject pipeline for cross-subject analyses.

This module extends the base Pipeline to handle multi-subject data,
supporting leave-one-subject-out (LOSO) and run-based CV schemes.

**Classes:**

Name | Description
---- | -----------
[`MultiSubjectPipeline`](#nltools.pipelines.multi_subject.MultiSubjectPipeline) | Pipeline for multi-subject neuroimaging analyses.



##### Classes###### `nltools.pipelines.multi_subject.MultiSubjectPipeline`

```python
MultiSubjectPipeline(data: List[NDArray], cv: Optional[Any] = None, groups: Optional[NDArray[np.intp]] = None, steps: List[Any] = list(), _is_lazy: bool = False) -> None
```

Pipeline for multi-subject neuroimaging analyses.

Operates on a list of subject data arrays, supporting:
- LOSO (leave-one-subject-out): Train on N-1 subjects, test on 1
- Run-based CV: Split runs within each subject
- Pooling across subjects for group analyses

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[List](#typing.List)[[NDArray](#numpy.typing.NDArray)]</code> | List of subject data arrays, each shape (n_obs, n_voxels). | *required*
`cv` | <code>[Optional](#typing.Optional)[[Any](#typing.Any)]</code> | Cross-validation scheme configuration. | <code>None</code>
`groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for CV splits (e.g., run labels). | <code>None</code>
`steps` | <code>[List](#typing.List)[[Any](#typing.Any)]</code> | Transform steps to apply. | <code>[list](#list)()</code>

Examples
--------
>>> # LOSO CV
>>> pipeline = MultiSubjectPipeline(subject_data, cv=CVScheme(scheme='loso'))
>>> result = pipeline.normalize().predict(y, algorithm='svm')

>>> # Run-based CV across subjects
>>> pipeline = MultiSubjectPipeline(subject_data, cv=CVScheme(scheme='loro'), groups=runs)
>>> result = pipeline.predict(y)

**Functions:**

Name | Description
---- | -----------
[`align`](#nltools.pipelines.multi_subject.MultiSubjectPipeline.align) | Add cross-subject alignment step to pipeline.
[`isc`](#nltools.pipelines.multi_subject.MultiSubjectPipeline.isc) | Compute inter-subject correlation across subjects.
[`normalize`](#nltools.pipelines.multi_subject.MultiSubjectPipeline.normalize) | Add normalization step (per-subject).
[`pipe`](#nltools.pipelines.multi_subject.MultiSubjectPipeline.pipe) | Add custom sklearn transformer.
[`predict`](#nltools.pipelines.multi_subject.MultiSubjectPipeline.predict) | Execute pipeline with CV and return prediction results.
[`reduce`](#nltools.pipelines.multi_subject.MultiSubjectPipeline.reduce) | Add dimensionality reduction step.
[`rsa`](#nltools.pipelines.multi_subject.MultiSubjectPipeline.rsa) | Compute representational similarity analysis.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`cv`](#nltools.pipelines.multi_subject.MultiSubjectPipeline.cv) | <code>[Optional](#typing.Optional)[[Any](#typing.Any)]</code> | 
[`data`](#nltools.pipelines.multi_subject.MultiSubjectPipeline.data) | <code>[List](#typing.List)[[NDArray](#numpy.typing.NDArray)]</code> | 
[`groups`](#nltools.pipelines.multi_subject.MultiSubjectPipeline.groups) | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | 
[`n_steps`](#nltools.pipelines.multi_subject.MultiSubjectPipeline.n_steps) | <code>[int](#int)</code> | Number of transform steps.
[`n_subjects`](#nltools.pipelines.multi_subject.MultiSubjectPipeline.n_subjects) | <code>[int](#int)</code> | Number of subjects in the multi-subject dataset.
[`steps`](#nltools.pipelines.multi_subject.MultiSubjectPipeline.steps) | <code>[List](#typing.List)[[Any](#typing.Any)]</code> | 



####### Attributes######## `nltools.pipelines.multi_subject.MultiSubjectPipeline.cv`

```python
cv: Optional[Any] = None
```

######## `nltools.pipelines.multi_subject.MultiSubjectPipeline.data`

```python
data: List[NDArray]
```

######## `nltools.pipelines.multi_subject.MultiSubjectPipeline.groups`

```python
groups: Optional[NDArray[np.intp]] = None
```

######## `nltools.pipelines.multi_subject.MultiSubjectPipeline.n_steps`

```python
n_steps: int
```

Number of transform steps.

######## `nltools.pipelines.multi_subject.MultiSubjectPipeline.n_subjects`

```python
n_subjects: int
```

Number of subjects in the multi-subject dataset.

######## `nltools.pipelines.multi_subject.MultiSubjectPipeline.steps`

```python
steps: List[Any] = field(default_factory=list)
```



####### Functions######## `nltools.pipelines.multi_subject.MultiSubjectPipeline.align`

```python
align(method: str = 'srm', scheme: str = 'global', n_features: int | None = 50, new_subject: str = 'procrustes', **kwargs: str) -> 'MultiSubjectPipeline'
```

Add cross-subject alignment step to pipeline.

Aligns multi-subject data using SRM or HyperAlignment before
downstream analyses like classification or pooling.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Alignment method: 'srm' (Shared Response Model, reduces dimensionality) or 'hyperalignment' (Procrustes-based, preserves dimensionality). Default is 'srm'. | <code>'srm'</code>
`scheme` | <code>[str](#str)</code> | Spatial scheme. Currently only 'global' is supported. 'searchlight' and 'piecewise' require LocalAlignment (nltools-boll). | <code>'global'</code>
`n_features` | <code>[int](#int) \| None</code> | Number of shared features for SRM. Ignored for hyperalignment. | <code>50</code>
`new_subject` | <code>[str](#str)</code> | Method for aligning held-out subjects in LOSO CV. Default is 'procrustes'. | <code>'procrustes'</code>
`**kwargs` |  | Additional arguments passed to alignment algorithm. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'MultiSubjectPipeline'</code> | New pipeline with alignment step added.

Examples
--------
>>> # SRM alignment before classification
>>> result = (
...     MultiSubjectPipeline(data=subjects, cv=CVScheme(scheme='loso'))
...     .align(method='srm', n_features=50)
...     .predict(y=labels, algorithm='svm')
... )

>>> # Hyperalignment before two-stage GLM
>>> result = (
...     bc.cv(scheme='loso')
...     .align(method='hyperalignment')
...     .fit(model='glm', X=designs)
...     .pool(param='beta')
...     .fit(model='ttest', contrast='A-B')
... )

######## `nltools.pipelines.multi_subject.MultiSubjectPipeline.isc`

```python
isc(method: str = 'pairwise', metric: str = 'median', n_permute: int = 5000, parallel: str = 'cpu', **kwargs: str)
```

Compute inter-subject correlation across subjects.

Executes the pipeline and computes ISC using permutation testing.
Data is transformed through all pipeline steps before ISC computation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ISC computation method: 'pairwise' (average all pairwise correlations) or 'leave-one-out' (correlate each subject with mean of others). Default is 'pairwise'. | <code>'pairwise'</code>
`metric` | <code>[str](#str)</code> | Summary statistic: 'median' (robust, default) or 'mean' (Fisher z-transformed). | <code>'median'</code>
`n_permute` | <code>[int](#int)</code> | Number of bootstrap iterations for p-value computation. Default is 5000. | <code>5000</code>
`parallel` | <code>[str](#str)</code> | Parallelization method: 'cpu', 'gpu', or None. Default is 'cpu'. | <code>'cpu'</code>
`**kwargs` |  | Additional arguments passed to ISCTerminal. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | Result containing ISC values, p-values, and confidence intervals.

Examples
--------
>>> result = (
...     MultiSubjectPipeline(data=subjects, cv=None)
...     .normalize()
...     .isc(method='pairwise', n_permute=1000)
... )
>>> print(f"ISC: {result.isc:.3f}, p: {result.p:.3f}")

######## `nltools.pipelines.multi_subject.MultiSubjectPipeline.normalize`

```python
normalize(method: str = 'zscore', **kwargs: str) -> 'MultiSubjectPipeline'
```

Add normalization step (per-subject).

######## `nltools.pipelines.multi_subject.MultiSubjectPipeline.pipe`

```python
pipe(transformer) -> 'MultiSubjectPipeline'
```

Add custom sklearn transformer.

######## `nltools.pipelines.multi_subject.MultiSubjectPipeline.predict`

```python
predict(y, algorithm: str = 'ridge', **kwargs: str)
```

Execute pipeline with CV and return prediction results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` |  | Target variable. For LOSO, should be (n_subjects,). For run-based CV, should match pooled observations. | *required*
`algorithm` | <code>[str](#str)</code> | Prediction algorithm: 'ridge', 'lasso', 'elastic', 'svr' for regression; 'svm', 'logistic', 'rf' for classification. | <code>'ridge'</code>
`**kwargs` |  | Additional arguments passed to sklearn model constructor. For classification (svm, logistic), use ``class_weight='balanced'`` to handle imbalanced classes. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | Cross-validation results.

Examples
--------
Basic regression with LOSO CV::

    result = pipeline.cv('loso').predict(subject_labels, algorithm='ridge')

Classification with balanced classes::

    result = pipeline.cv('loso').predict(
        group_labels, algorithm='svm', class_weight='balanced'
    )

Logistic regression with regularization::

    result = pipeline.cv('loso').predict(
        binary_labels, algorithm='logistic', C=0.1, class_weight='balanced'
    )

######## `nltools.pipelines.multi_subject.MultiSubjectPipeline.reduce`

```python
reduce(method: str = 'pca', n_components: Optional[int] = None, **kwargs: Optional[int]) -> 'MultiSubjectPipeline'
```

Add dimensionality reduction step.

######## `nltools.pipelines.multi_subject.MultiSubjectPipeline.rsa`

```python
rsa(model_rdm: NDArray, method: str = 'spearman', n_permute: int = 5000, **kwargs: int)
```

Compute representational similarity analysis.

Executes the pipeline and computes RSA correlation between neural
and model RDMs using permutation testing.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model_rdm` | <code>[NDArray](#numpy.typing.NDArray)</code> | Model RDM to correlate with neural RDMs. Should be symmetric matrix or upper triangle (condensed form). | *required*
`method` | <code>[str](#str)</code> | Correlation method: 'spearman' (default), 'pearson', or 'kendall'. | <code>'spearman'</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations for p-value computation. Default is 5000. | <code>5000</code>
`**kwargs` |  | Additional arguments passed to RSATerminal. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | Result containing correlation coefficient and p-value.

Examples
--------
>>> model = np.corrcoef(conditions)  # Theoretical model
>>> result = (
...     MultiSubjectPipeline(data=subjects, cv=None)
...     .normalize()
...     .rsa(model_rdm=model, method='spearman')
... )
>>> print(f"r = {result.correlation:.3f}, p = {result.p_value:.3f}")

#### `nltools.pipelines.pool`

Pool infrastructure for multi-subject aggregation.

This module provides classes for pooling data across subjects and
enabling two-stage analyses (e.g., first-level GLM -> group t-test).

The pool() method serves as an execution boundary - everything before
it is executed lazily, and pool() triggers execution and aggregation.

**Classes:**

Name | Description
---- | -----------
[`PooledData`](#nltools.pipelines.pool.PooledData) | Aggregated data from multiple subjects.
[`ResultDict`](#nltools.pipelines.pool.ResultDict) | Dictionary of StatResults, one per contrast.
[`StatResult`](#nltools.pipelines.pool.StatResult) | Result of statistical test.



##### Attributes

##### Classes###### `nltools.pipelines.pool.PooledData`

```python
PooledData(data: NDArray, param: str, condition_names: Optional[list[str]] = None, subject_ids: Optional[list[str]] = None, mask: Optional[Any] = None, fitted_state: Optional[Any] = None, save_path: Optional[str] = None) -> None
```

Aggregated data from multiple subjects.

PooledData serves as a checkpoint after first-level analyses,
enabling reusable second-level analyses without re-running
the first-level computations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[NDArray](#numpy.typing.NDArray)</code> | Pooled data array. Shape is (n_subjects, n_voxels) for single parameter or (n_subjects, n_conditions, n_voxels) for multi-condition. | *required*
`param` | <code>[str](#str)</code> | Parameter that was pooled (e.g., 'beta', 'residual', 't'). | *required*
`condition_names` | <code>[Optional](#typing.Optional)[[list](#list)[[str](#str)]]</code> | Names of conditions if multi-condition data. | <code>None</code>
`subject_ids` | <code>[Optional](#typing.Optional)[[list](#list)[[str](#str)]]</code> | Subject identifiers. | <code>None</code>
`fitted_state` | <code>[Optional](#typing.Optional)[[Any](#typing.Any)]</code> | Saved fitted models for repool() functionality. | <code>None</code>
`save_path` | <code>[Optional](#typing.Optional)[[str](#str)]</code> | Path where data was saved. | <code>None</code>

Examples
--------
>>> # Two-stage GLM
>>> pool = bc.fit(model='glm', X=designs).pool(param='beta')
>>> result = pool.fit(model='ttest', contrast='face-house')
>>>
>>> # Reuse for multiple contrasts
>>> result1 = pool.fit(model='ttest', contrast='face-house')
>>> result2 = pool.fit(model='ttest', contrast='face-object')

**Functions:**

Name | Description
---- | -----------
[`cv`](#nltools.pipelines.pool.PooledData.cv) | Create CV pipeline on pooled data.
[`fit`](#nltools.pipelines.pool.PooledData.fit) | Fit second-level statistical model.
[`load`](#nltools.pipelines.pool.PooledData.load) | Load pooled data from disk.
[`repool`](#nltools.pipelines.pool.PooledData.repool) | Re-extract different parameter from saved fitted state.
[`save`](#nltools.pipelines.pool.PooledData.save) | Save pooled data to disk.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`condition_names`](#nltools.pipelines.pool.PooledData.condition_names) | <code>[Optional](#typing.Optional)[[list](#list)[[str](#str)]]</code> | 
[`data`](#nltools.pipelines.pool.PooledData.data) | <code>[NDArray](#numpy.typing.NDArray)</code> | 
[`fitted_state`](#nltools.pipelines.pool.PooledData.fitted_state) | <code>[Optional](#typing.Optional)[[Any](#typing.Any)]</code> | 
[`mask`](#nltools.pipelines.pool.PooledData.mask) | <code>[Optional](#typing.Optional)[[Any](#typing.Any)]</code> | 
[`n_conditions`](#nltools.pipelines.pool.PooledData.n_conditions) | <code>[Optional](#typing.Optional)[[int](#int)]</code> | Number of conditions (None if single-condition).
[`n_subjects`](#nltools.pipelines.pool.PooledData.n_subjects) | <code>[int](#int)</code> | Number of subjects in the pooled dataset (first dimension of data).
[`n_voxels`](#nltools.pipelines.pool.PooledData.n_voxels) | <code>[int](#int)</code> | Number of voxels (last dimension of data array).
[`param`](#nltools.pipelines.pool.PooledData.param) | <code>[str](#str)</code> | 
[`save_path`](#nltools.pipelines.pool.PooledData.save_path) | <code>[Optional](#typing.Optional)[[str](#str)]</code> | 
[`shape`](#nltools.pipelines.pool.PooledData.shape) | <code>[tuple](#tuple)</code> | Shape of the pooled data array as (n_subjects[, n_conditions], n_voxels).
[`subject_ids`](#nltools.pipelines.pool.PooledData.subject_ids) | <code>[Optional](#typing.Optional)[[list](#list)[[str](#str)]]</code> | 



####### Attributes######## `nltools.pipelines.pool.PooledData.condition_names`

```python
condition_names: Optional[list[str]] = None
```

######## `nltools.pipelines.pool.PooledData.data`

```python
data: NDArray
```

######## `nltools.pipelines.pool.PooledData.fitted_state`

```python
fitted_state: Optional[Any] = field(default=None, repr=False)
```

######## `nltools.pipelines.pool.PooledData.mask`

```python
mask: Optional[Any] = field(default=None, repr=False)
```

######## `nltools.pipelines.pool.PooledData.n_conditions`

```python
n_conditions: Optional[int]
```

Number of conditions (None if single-condition).

######## `nltools.pipelines.pool.PooledData.n_subjects`

```python
n_subjects: int
```

Number of subjects in the pooled dataset (first dimension of data).

######## `nltools.pipelines.pool.PooledData.n_voxels`

```python
n_voxels: int
```

Number of voxels (last dimension of data array).

######## `nltools.pipelines.pool.PooledData.param`

```python
param: str
```

######## `nltools.pipelines.pool.PooledData.save_path`

```python
save_path: Optional[str] = None
```

######## `nltools.pipelines.pool.PooledData.shape`

```python
shape: tuple
```

Shape of the pooled data array as (n_subjects[, n_conditions], n_voxels).

######## `nltools.pipelines.pool.PooledData.subject_ids`

```python
subject_ids: Optional[list[str]] = None
```



####### Functions######## `nltools.pipelines.pool.PooledData.cv`

```python
cv(k: Optional[int] = None, scheme: CVSchemeType = 'kfold', **kwargs: CVSchemeType) -> 'Pipeline'
```

Create CV pipeline on pooled data.

Useful for classification on pooled betas.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`k` | <code>[Optional](#typing.Optional)[[int](#int)]</code> | Number of folds. | <code>None</code>
`scheme` | <code>[CVSchemeType](#nltools.pipelines.pool.CVSchemeType)</code> | CV scheme ('kfold', 'loso', 'loro', 'bootstrap'). | <code>'kfold'</code>

**Returns:**

Type | Description
---- | -----------
<code>'Pipeline'</code> | Pipeline for classification on pooled data.

######## `nltools.pipelines.pool.PooledData.fit`

```python
fit(model: str, contrast: Optional[str] = None, contrasts: Optional[list[str]] = None, X: Optional[NDArray] = None, **kwargs: Optional[NDArray]) -> Union['StatResult', 'ResultDict']
```

Fit second-level statistical model.

This is a terminal method - executes immediately (eager).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model` | <code>[str](#str)</code> | Statistical model type: 'ttest' (one-sample or two-sample with X), 'paired_ttest', or 'anova'. | *required*
`contrast` | <code>[Optional](#typing.Optional)[[str](#str)]</code> | Single contrast specification (e.g., 'face-house'). | <code>None</code>
`contrasts` | <code>[Optional](#typing.Optional)[[list](#list)[[str](#str)]]</code> | Multiple contrasts - returns ResultDict. | <code>None</code>
`X` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)]</code> | Design matrix for two-sample tests or ANOVA. | <code>None</code>
`**kwargs` |  | Additional arguments for the statistical test. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[Union](#typing.Union)['StatResult', 'ResultDict']</code> | Statistical results. ResultDict if multiple contrasts specified.

Examples
--------
>>> result = pool.fit(model='ttest', contrast='face-house')
>>> result.t_map.max()

>>> results = pool.fit(model='ttest', contrasts=['A-B', 'A-C', 'B-C'])
>>> results['A-B'].threshold(method='fdr')

######## `nltools.pipelines.pool.PooledData.load`

```python
load(path: str) -> 'PooledData'
```

Load pooled data from disk.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`path` | <code>[str](#str)</code> | Path to saved data. | *required*

**Returns:**

Type | Description
---- | -----------
<code>'PooledData'</code> | Loaded pooled data.

######## `nltools.pipelines.pool.PooledData.repool`

```python
repool(param: str) -> 'PooledData'
```

Re-extract different parameter from saved fitted state.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`param` | <code>[str](#str)</code> | Parameter to extract (e.g., 'residual', 't'). | *required*

**Returns:**

Type | Description
---- | -----------
<code>'PooledData'</code> | New PooledData with the requested parameter.

######## `nltools.pipelines.pool.PooledData.save`

```python
save(path: str) -> None
```

Save pooled data to disk.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`path` | <code>[str](#str)</code> | Output path (directory or .npz file). | *required*

###### `nltools.pipelines.pool.ResultDict`

Bases: <code>[dict](#dict)</code>

Dictionary of StatResults, one per contrast.

Provides convenience methods for batch operations.

**Functions:**

Name | Description
---- | -----------
[`threshold_all`](#nltools.pipelines.pool.ResultDict.threshold_all) | Apply thresholding to all results.



####### Functions######## `nltools.pipelines.pool.ResultDict.threshold_all`

```python
threshold_all(method: str = 'fdr', alpha: float = 0.05) -> 'ResultDict'
```

Apply thresholding to all results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Correction method: 'fdr', 'bonferroni', or 'uncorrected'. | <code>'fdr'</code>
`alpha` | <code>[float](#float)</code> | Significance threshold. | <code>0.05</code>

**Returns:**

Type | Description
---- | -----------
<code>'ResultDict'</code> | New dict with thresholded results.

###### `nltools.pipelines.pool.StatResult`

```python
StatResult(t_map: Optional[NDArray] = None, f_map: Optional[NDArray] = None, p_map: Optional[NDArray] = None, contrast: Optional[str] = None, df: Optional[int] = None) -> None
```

Result of statistical test.

Holds statistical maps and provides thresholding utilities.

**Functions:**

Name | Description
---- | -----------
[`threshold`](#nltools.pipelines.pool.StatResult.threshold) | Apply multiple comparison correction.
[`to_nifti`](#nltools.pipelines.pool.StatResult.to_nifti) | Save as NIfTI file.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`contrast`](#nltools.pipelines.pool.StatResult.contrast) | <code>[Optional](#typing.Optional)[[str](#str)]</code> | 
[`df`](#nltools.pipelines.pool.StatResult.df) | <code>[Optional](#typing.Optional)[[int](#int)]</code> | 
[`f_map`](#nltools.pipelines.pool.StatResult.f_map) | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)]</code> | 
[`p_map`](#nltools.pipelines.pool.StatResult.p_map) | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)]</code> | 
[`t_map`](#nltools.pipelines.pool.StatResult.t_map) | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)]</code> | 



####### Attributes######## `nltools.pipelines.pool.StatResult.contrast`

```python
contrast: Optional[str] = None
```

######## `nltools.pipelines.pool.StatResult.df`

```python
df: Optional[int] = None
```

######## `nltools.pipelines.pool.StatResult.f_map`

```python
f_map: Optional[NDArray] = None
```

######## `nltools.pipelines.pool.StatResult.p_map`

```python
p_map: Optional[NDArray] = None
```

######## `nltools.pipelines.pool.StatResult.t_map`

```python
t_map: Optional[NDArray] = None
```



####### Functions######## `nltools.pipelines.pool.StatResult.threshold`

```python
threshold(method: str = 'fdr', alpha: float = 0.05) -> 'StatResult'
```

Apply multiple comparison correction.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Correction method: 'fdr', 'bonferroni', or 'uncorrected'. | <code>'fdr'</code>
`alpha` | <code>[float](#float)</code> | Significance threshold. | <code>0.05</code>

**Returns:**

Type | Description
---- | -----------
<code>'StatResult'</code> | New result with thresholded maps.

######## `nltools.pipelines.pool.StatResult.to_nifti`

```python
to_nifti(path: str, mask: str = None) -> None
```

Save as NIfTI file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`path` | <code>[str](#str)</code> | Output path. | *required*
`mask` |  | Mask to use for reconstruction. | <code>None</code>

#### `nltools.pipelines.results`

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



##### Classes###### `nltools.pipelines.results.CVResult`

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



####### Attributes######## `nltools.pipelines.results.CVResult.fold_results`

```python
fold_results: List[FoldResult]
```

######## `nltools.pipelines.results.CVResult.is_fully_invertible`

```python
is_fully_invertible: bool
```

Check if all transform steps are invertible.

######## `nltools.pipelines.results.CVResult.mean_score`

```python
mean_score: float
```

Mean score across all folds.

######## `nltools.pipelines.results.CVResult.n_folds`

```python
n_folds: int
```

Number of cross-validation folds.

######## `nltools.pipelines.results.CVResult.pipeline`

```python
pipeline: Any
```

######## `nltools.pipelines.results.CVResult.predictions`

```python
predictions: NDArray[np.floating]
```

All predictions in original sample order.

Reconstructs predictions array with each sample's prediction
from the fold where it was in the test set.

######## `nltools.pipelines.results.CVResult.scores`

```python
scores: NDArray[np.floating]
```

Per-fold prediction scores as a numpy array.

######## `nltools.pipelines.results.CVResult.std_score`

```python
std_score: float
```

Standard deviation of scores across folds.



####### Functions######## `nltools.pipelines.results.CVResult.inverse_transform`

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

######## `nltools.pipelines.results.CVResult.summary`

```python
summary() -> str
```

Return formatted summary string.

###### `nltools.pipelines.results.FoldResult`

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



####### Attributes######## `nltools.pipelines.results.FoldResult.fitted_stack`

```python
fitted_stack: Any
```

######## `nltools.pipelines.results.FoldResult.predictions`

```python
predictions: NDArray[np.floating]
```

######## `nltools.pipelines.results.FoldResult.score`

```python
score: float
```

######## `nltools.pipelines.results.FoldResult.test_idx`

```python
test_idx: NDArray[np.intp]
```

######## `nltools.pipelines.results.FoldResult.train_idx`

```python
train_idx: NDArray[np.intp]
```

###### `nltools.pipelines.results.ISCResult`

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



####### Attributes######## `nltools.pipelines.results.ISCResult.ci`

```python
ci: tuple
```

######## `nltools.pipelines.results.ISCResult.isc`

```python
isc: NDArray[np.floating]
```

######## `nltools.pipelines.results.ISCResult.method`

```python
method: str
```

######## `nltools.pipelines.results.ISCResult.metric`

```python
metric: str
```

######## `nltools.pipelines.results.ISCResult.n_subjects`

```python
n_subjects: int
```

######## `nltools.pipelines.results.ISCResult.p`

```python
p: NDArray[np.floating]
```



####### Functions######## `nltools.pipelines.results.ISCResult.summary`

```python
summary() -> str
```

Return formatted summary string.

###### `nltools.pipelines.results.PermutationResult`

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



####### Attributes######## `nltools.pipelines.results.PermutationResult.n_permutations`

```python
n_permutations: int
```

######## `nltools.pipelines.results.PermutationResult.null_distribution`

```python
null_distribution: NDArray[np.floating]
```

######## `nltools.pipelines.results.PermutationResult.null_mean`

```python
null_mean: float
```

Mean of the null distribution.

######## `nltools.pipelines.results.PermutationResult.null_std`

```python
null_std: float
```

Standard deviation of the null distribution.

######## `nltools.pipelines.results.PermutationResult.observed`

```python
observed: CVResult
```

######## `nltools.pipelines.results.PermutationResult.observed_score`

```python
observed_score: float
```

Convenience accessor for observed mean score.

######## `nltools.pipelines.results.PermutationResult.p_value`

```python
p_value: float
```

######## `nltools.pipelines.results.PermutationResult.z_score`

```python
z_score: float
```

Z-score of observed relative to null distribution.



####### Functions######## `nltools.pipelines.results.PermutationResult.from_scores`

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

######## `nltools.pipelines.results.PermutationResult.summary`

```python
summary() -> str
```

Return formatted summary string.

###### `nltools.pipelines.results.RSAResult`

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



####### Attributes######## `nltools.pipelines.results.RSAResult.ci`

```python
ci: tuple
```

######## `nltools.pipelines.results.RSAResult.correlation`

```python
correlation: float
```

######## `nltools.pipelines.results.RSAResult.method`

```python
method: str
```

######## `nltools.pipelines.results.RSAResult.n_conditions`

```python
n_conditions: int
```

######## `nltools.pipelines.results.RSAResult.p_value`

```python
p_value: float
```



####### Functions######## `nltools.pipelines.results.RSAResult.summary`

```python
summary() -> str
```

Return formatted summary string.

#### `nltools.pipelines.steps`

Transform steps for nltools pipelines.

This module provides reusable transform steps that can be added to pipelines.
Steps implement the TransformStep protocol and can be chained together.

Each step follows the fit/transform pattern:
- `step.fit(data)` returns a FittedX object that holds learned parameters
- `fitted.transform(data)` applies the transformation
- `fitted.inverse_transform(data)` reverses the transformation (if invertible)

**Classes:**

Name | Description
---- | -----------
[`AlignStep`](#nltools.pipelines.steps.AlignStep) | Cross-subject alignment via SRM or HyperAlignment.
[`FittedAlign`](#nltools.pipelines.steps.FittedAlign) | Fitted alignment model.
[`FittedNormalize`](#nltools.pipelines.steps.FittedNormalize) | Fitted normalization transform.
[`FittedPipe`](#nltools.pipelines.steps.FittedPipe) | Fitted sklearn transformer wrapper.
[`FittedReduce`](#nltools.pipelines.steps.FittedReduce) | Fitted dimensionality reduction transform.
[`NormalizeStep`](#nltools.pipelines.steps.NormalizeStep) | Normalization transform step.
[`PipeStep`](#nltools.pipelines.steps.PipeStep) | Wrapper for sklearn-compatible transformers.
[`ReduceStep`](#nltools.pipelines.steps.ReduceStep) | Dimensionality reduction step.



##### Classes###### `nltools.pipelines.steps.AlignStep`

```python
AlignStep(method: str = 'srm', scheme: str = 'global', n_features: int | None = 50, new_subject: str = 'procrustes', n_iter: int = 10, parallel: str | None = 'cpu', n_jobs: int = -1, **kwargs: int)
```

Cross-subject alignment via SRM or HyperAlignment.

Wraps existing SRM and HyperAlignment algorithms for use in pipelines.
Currently supports 'global' scheme only. Searchlight/piecewise schemes
require LocalAlignment (nltools-boll epic).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Alignment method: 'srm' or 'hyperalignment'. Default is 'srm'. | <code>'srm'</code>
`scheme` | <code>[str](#str)</code> | Spatial scheme. Currently only 'global' is supported. 'searchlight' and 'piecewise' require LocalAlignment. | <code>'global'</code>
`n_features` | <code>[int](#int) \| None</code> | Number of features for SRM. None for hyperalignment (full rank). | <code>50</code>
`new_subject` | <code>[str](#str)</code> | Method for aligning held-out subjects in LOSO CV. Default is 'procrustes'. | <code>'procrustes'</code>
`n_iter` | <code>[int](#int)</code> | Number of iterations for SRM (or 2 for hyperalignment). Default is 10. | <code>10</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization: 'cpu', 'gpu', or None. | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of jobs for CPU parallelization. Default is -1. | <code>-1</code>
`**kwargs` |  | Additional arguments passed to the underlying algorithm. For SRM: 'rand_seed'. For HyperAlignment: 'auto_pad'. | <code>{}</code>

Examples
--------
>>> import numpy as np
>>> # Create synthetic multi-subject data
>>> data = [np.random.randn(100, 50) for _ in range(5)]
>>> step = AlignStep(method='srm', n_features=10)
>>> fitted = step.fit(data)
>>> aligned = fitted.transform(data)

**Functions:**

Name | Description
---- | -----------
[`fit`](#nltools.pipelines.steps.AlignStep.fit) | Fit alignment model on list of subject data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#nltools.pipelines.steps.AlignStep.invertible) | <code>[bool](#bool)</code> | Check if alignment is invertible.
[`kwargs`](#nltools.pipelines.steps.AlignStep.kwargs) |  | 
[`method`](#nltools.pipelines.steps.AlignStep.method) |  | 
[`n_features`](#nltools.pipelines.steps.AlignStep.n_features) |  | 
[`n_iter`](#nltools.pipelines.steps.AlignStep.n_iter) |  | 
[`n_jobs`](#nltools.pipelines.steps.AlignStep.n_jobs) |  | 
[`new_subject`](#nltools.pipelines.steps.AlignStep.new_subject) |  | 
[`parallel`](#nltools.pipelines.steps.AlignStep.parallel) |  | 
[`scheme`](#nltools.pipelines.steps.AlignStep.scheme) |  | 



####### Attributes######## `nltools.pipelines.steps.AlignStep.invertible`

```python
invertible: bool
```

Check if alignment is invertible.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if method is hyperalignment (full-rank orthogonal transforms).

######## `nltools.pipelines.steps.AlignStep.kwargs`

```python
kwargs = kwargs
```

######## `nltools.pipelines.steps.AlignStep.method`

```python
method = method
```

######## `nltools.pipelines.steps.AlignStep.n_features`

```python
n_features = n_features
```

######## `nltools.pipelines.steps.AlignStep.n_iter`

```python
n_iter = n_iter
```

######## `nltools.pipelines.steps.AlignStep.n_jobs`

```python
n_jobs = n_jobs
```

######## `nltools.pipelines.steps.AlignStep.new_subject`

```python
new_subject = new_subject
```

######## `nltools.pipelines.steps.AlignStep.parallel`

```python
parallel = parallel
```

######## `nltools.pipelines.steps.AlignStep.scheme`

```python
scheme = scheme
```



####### Functions######## `nltools.pipelines.steps.AlignStep.fit`

```python
fit(data: list[np.ndarray]) -> FittedAlign
```

Fit alignment model on list of subject data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | Each array has shape (n_voxels, n_samples) or (n_samples, n_voxels). Will be transposed if needed to match algorithm expectations. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[FittedAlign](#nltools.pipelines.steps.FittedAlign)</code> | Fitted alignment model.

###### `nltools.pipelines.steps.FittedAlign`

```python
FittedAlign(model: Any, method: str, new_subject_method: str = 'procrustes') -> None
```

Fitted alignment model.

Holds a fitted SRM or HyperAlignment model and applies transformations.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`model`](#nltools.pipelines.steps.FittedAlign.model) | <code>[Any](#typing.Any)</code> | Fitted SRM or HyperAlignment instance.
[`method`](#nltools.pipelines.steps.FittedAlign.method) | <code>[str](#str)</code> | The alignment method used ('srm' or 'hyperalignment').
[`new_subject_method`](#nltools.pipelines.steps.FittedAlign.new_subject_method) | <code>[str](#str)</code> | Method for aligning held-out subjects in LOSO CV.

**Functions:**

Name | Description
---- | -----------
[`inverse_transform`](#nltools.pipelines.steps.FittedAlign.inverse_transform) | Reverse alignment (only for full-rank hyperalignment).
[`transform`](#nltools.pipelines.steps.FittedAlign.transform) | Transform subjects that were in training.
[`transform_new_subject`](#nltools.pipelines.steps.FittedAlign.transform_new_subject) | Align a new subject not in training (for LOSO).



####### Attributes######## `nltools.pipelines.steps.FittedAlign.method`

```python
method: str
```

######## `nltools.pipelines.steps.FittedAlign.model`

```python
model: Any
```

######## `nltools.pipelines.steps.FittedAlign.new_subject_method`

```python
new_subject_method: str = 'procrustes'
```



####### Functions######## `nltools.pipelines.steps.FittedAlign.inverse_transform`

```python
inverse_transform(data: list[np.ndarray]) -> list[np.ndarray]
```

Reverse alignment (only for full-rank hyperalignment).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | Aligned data for each subject, shape (n_samples, n_aligned_features). Pipeline convention is (samples, features). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | Data in original subject-specific space, shape (n_samples, n_features).

######## `nltools.pipelines.steps.FittedAlign.transform`

```python
transform(data: list[np.ndarray]) -> list[np.ndarray]
```

Transform subjects that were in training.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | List of subject data arrays, each shape (n_samples, n_features). Pipeline convention is (samples, features). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | Aligned data for each subject, shape (n_samples, n_aligned_features).

######## `nltools.pipelines.steps.FittedAlign.transform_new_subject`

```python
transform_new_subject(data: np.ndarray) -> np.ndarray
```

Align a new subject not in training (for LOSO).

Uses transform_subject() which fits a new transform for the held-out subject.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Data for the new subject, shape (n_samples, n_features). Pipeline convention is (samples, features). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Aligned data for the new subject, shape (n_samples, n_aligned_features).

###### `nltools.pipelines.steps.FittedNormalize`

```python
FittedNormalize(mean: np.ndarray, std: np.ndarray, method: str) -> None
```

Fitted normalization transform.

Holds the learned parameters (mean, std or min, range) and applies
the transformation to new data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`mean`](#nltools.pipelines.steps.FittedNormalize.mean) | <code>[ndarray](#numpy.ndarray)</code> | For zscore: the mean. For minmax: the min value.
[`std`](#nltools.pipelines.steps.FittedNormalize.std) | <code>[ndarray](#numpy.ndarray)</code> | For zscore: the standard deviation. For minmax: the range (max - min).
[`method`](#nltools.pipelines.steps.FittedNormalize.method) | <code>[str](#str)</code> | The normalization method ('zscore' or 'minmax').

**Functions:**

Name | Description
---- | -----------
[`inverse_transform`](#nltools.pipelines.steps.FittedNormalize.inverse_transform) | Reverse normalization.
[`transform`](#nltools.pipelines.steps.FittedNormalize.transform) | Apply normalization to data.



####### Attributes######## `nltools.pipelines.steps.FittedNormalize.mean`

```python
mean: np.ndarray
```

######## `nltools.pipelines.steps.FittedNormalize.method`

```python
method: str
```

######## `nltools.pipelines.steps.FittedNormalize.std`

```python
std: np.ndarray
```



####### Functions######## `nltools.pipelines.steps.FittedNormalize.inverse_transform`

```python
inverse_transform(data: np.ndarray) -> np.ndarray
```

Reverse normalization.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Normalized data. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Data in original scale.

######## `nltools.pipelines.steps.FittedNormalize.transform`

```python
transform(data: np.ndarray) -> np.ndarray
```

Apply normalization to data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Data to normalize. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Normalized data.

###### `nltools.pipelines.steps.FittedPipe`

```python
FittedPipe(transformer: Any) -> None
```

Fitted sklearn transformer wrapper.

Holds a fitted sklearn transformer and delegates transform calls to it.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`transformer`](#nltools.pipelines.steps.FittedPipe.transformer) | <code>[Any](#typing.Any)</code> | Fitted sklearn-compatible transformer.

**Functions:**

Name | Description
---- | -----------
[`inverse_transform`](#nltools.pipelines.steps.FittedPipe.inverse_transform) | Apply inverse transform if supported.
[`transform`](#nltools.pipelines.steps.FittedPipe.transform) | Apply the fitted transformer.



####### Attributes######## `nltools.pipelines.steps.FittedPipe.transformer`

```python
transformer: Any
```



####### Functions######## `nltools.pipelines.steps.FittedPipe.inverse_transform`

```python
inverse_transform(data: np.ndarray) -> np.ndarray
```

Apply inverse transform if supported.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Transformed data. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Data in original space.

######## `nltools.pipelines.steps.FittedPipe.transform`

```python
transform(data: np.ndarray) -> np.ndarray
```

Apply the fitted transformer.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Data to transform. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Transformed data.

###### `nltools.pipelines.steps.FittedReduce`

```python
FittedReduce(model: Any, method: str) -> None
```

Fitted dimensionality reduction transform.

Holds the fitted sklearn model and applies transformations.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`model`](#nltools.pipelines.steps.FittedReduce.model) | <code>[Any](#typing.Any)</code> | Fitted sklearn decomposition model (PCA, FastICA, etc.).
[`method`](#nltools.pipelines.steps.FittedReduce.method) | <code>[str](#str)</code> | The reduction method used.

**Functions:**

Name | Description
---- | -----------
[`inverse_transform`](#nltools.pipelines.steps.FittedReduce.inverse_transform) | Reverse dimensionality reduction (reconstruct original space).
[`transform`](#nltools.pipelines.steps.FittedReduce.transform) | Apply dimensionality reduction.



####### Attributes######## `nltools.pipelines.steps.FittedReduce.method`

```python
method: str
```

######## `nltools.pipelines.steps.FittedReduce.model`

```python
model: Any
```



####### Functions######## `nltools.pipelines.steps.FittedReduce.inverse_transform`

```python
inverse_transform(data: np.ndarray) -> np.ndarray
```

Reverse dimensionality reduction (reconstruct original space).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Reduced data, shape (n_samples, n_components). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Reconstructed data, shape (n_samples, n_features).

######## `nltools.pipelines.steps.FittedReduce.transform`

```python
transform(data: np.ndarray) -> np.ndarray
```

Apply dimensionality reduction.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Data to reduce, shape (n_samples, n_features). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Reduced data, shape (n_samples, n_components).

###### `nltools.pipelines.steps.NormalizeStep`

```python
NormalizeStep(method: str = 'zscore', axis: int = 0, invertible: bool = True) -> None
```

Normalization transform step.

Computes normalization parameters from training data and applies
the transformation to new data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Normalization method: 'zscore' (subtract mean, divide by std) or 'minmax' (scale to [0, 1] range). Default is 'zscore'. | <code>'zscore'</code>
`axis` | <code>[int](#int)</code> | Axis along which to compute statistics. Default 0 (per-feature normalization, treating rows as samples). | <code>0</code>

Examples
--------
>>> import numpy as np
>>> data = np.array([[1, 2], [3, 4], [5, 6]])
>>> step = NormalizeStep(method='zscore')
>>> fitted = step.fit(data)
>>> normalized = fitted.transform(data)
>>> restored = fitted.inverse_transform(normalized)
>>> np.allclose(data, restored)
True

**Functions:**

Name | Description
---- | -----------
[`fit`](#nltools.pipelines.steps.NormalizeStep.fit) | Compute normalization parameters from data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`axis`](#nltools.pipelines.steps.NormalizeStep.axis) | <code>[int](#int)</code> | 
[`invertible`](#nltools.pipelines.steps.NormalizeStep.invertible) | <code>[bool](#bool)</code> | 
[`method`](#nltools.pipelines.steps.NormalizeStep.method) | <code>[str](#str)</code> | 



####### Attributes######## `nltools.pipelines.steps.NormalizeStep.axis`

```python
axis: int = 0
```

######## `nltools.pipelines.steps.NormalizeStep.invertible`

```python
invertible: bool = True
```

######## `nltools.pipelines.steps.NormalizeStep.method`

```python
method: str = 'zscore'
```



####### Functions######## `nltools.pipelines.steps.NormalizeStep.fit`

```python
fit(data: np.ndarray) -> FittedNormalize
```

Compute normalization parameters from data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Training data to compute parameters from. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[FittedNormalize](#nltools.pipelines.steps.FittedNormalize)</code> | Fitted transform that can be applied to new data.

###### `nltools.pipelines.steps.PipeStep`

```python
PipeStep(transformer: Any = None) -> None
```

Wrapper for sklearn-compatible transformers.

Allows any sklearn transformer with a fit/transform interface to be
used as a pipeline step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`transformer` | <code>[Any](#typing.Any)</code> | An sklearn-compatible transformer instance. Must have fit() and transform() methods. The transformer will be cloned before fitting. | <code>None</code>

Examples
--------
>>> from sklearn.preprocessing import StandardScaler
>>> import numpy as np
>>> data = np.random.randn(100, 10)
>>> step = PipeStep(transformer=StandardScaler())
>>> fitted = step.fit(data)
>>> transformed = fitted.transform(data)
>>> restored = fitted.inverse_transform(transformed)
>>> np.allclose(data, restored)
True

**Functions:**

Name | Description
---- | -----------
[`fit`](#nltools.pipelines.steps.PipeStep.fit) | Fit transformer to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#nltools.pipelines.steps.PipeStep.invertible) | <code>[bool](#bool)</code> | Check if the transformer supports inverse_transform.
[`transformer`](#nltools.pipelines.steps.PipeStep.transformer) | <code>[Any](#typing.Any)</code> | 



####### Attributes######## `nltools.pipelines.steps.PipeStep.invertible`

```python
invertible: bool
```

Check if the transformer supports inverse_transform.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if transformer has inverse_transform method.

######## `nltools.pipelines.steps.PipeStep.transformer`

```python
transformer: Any = None
```



####### Functions######## `nltools.pipelines.steps.PipeStep.fit`

```python
fit(data: np.ndarray) -> FittedPipe
```

Fit transformer to data.

The transformer is cloned before fitting to ensure the original
transformer instance is not modified.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Training data. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[FittedPipe](#nltools.pipelines.steps.FittedPipe)</code> | Fitted transform wrapper.

###### `nltools.pipelines.steps.ReduceStep`

```python
ReduceStep(method: str = 'pca', n_components: Optional[int] = None, random_state: Optional[int] = None) -> None
```

Dimensionality reduction step.

Fits a dimensionality reduction model to training data and transforms
new data to the reduced space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Reduction method: 'pca' (Principal Component Analysis, invertible) or 'ica' (Independent Component Analysis, not invertible). Default is 'pca'. | <code>'pca'</code>
`n_components` | <code>[Optional](#typing.Optional)[[int](#int)]</code> | Number of components to keep. If None, keeps all components. | <code>None</code>
`random_state` | <code>[Optional](#typing.Optional)[[int](#int)]</code> | Random seed for reproducibility. | <code>None</code>

Examples
--------
>>> import numpy as np
>>> data = np.random.randn(100, 50)
>>> step = ReduceStep(method='pca', n_components=10)
>>> fitted = step.fit(data)
>>> reduced = fitted.transform(data)
>>> reduced.shape
(100, 10)

**Functions:**

Name | Description
---- | -----------
[`fit`](#nltools.pipelines.steps.ReduceStep.fit) | Fit reduction model to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#nltools.pipelines.steps.ReduceStep.invertible) | <code>[bool](#bool)</code> | Check if the reduction method supports inverse transform.
[`method`](#nltools.pipelines.steps.ReduceStep.method) | <code>[str](#str)</code> | 
[`n_components`](#nltools.pipelines.steps.ReduceStep.n_components) | <code>[Optional](#typing.Optional)[[int](#int)]</code> | 
[`random_state`](#nltools.pipelines.steps.ReduceStep.random_state) | <code>[Optional](#typing.Optional)[[int](#int)]</code> | 



####### Attributes######## `nltools.pipelines.steps.ReduceStep.invertible`

```python
invertible: bool
```

Check if the reduction method supports inverse transform.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if method is 'pca', False otherwise.

######## `nltools.pipelines.steps.ReduceStep.method`

```python
method: str = 'pca'
```

######## `nltools.pipelines.steps.ReduceStep.n_components`

```python
n_components: Optional[int] = None
```

######## `nltools.pipelines.steps.ReduceStep.random_state`

```python
random_state: Optional[int] = None
```



####### Functions######## `nltools.pipelines.steps.ReduceStep.fit`

```python
fit(data: np.ndarray) -> FittedReduce
```

Fit reduction model to data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Training data, shape (n_samples, n_features). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[FittedReduce](#nltools.pipelines.steps.FittedReduce)</code> | Fitted transform that can be applied to new data.

#### `nltools.pipelines.terminals`

Terminal operations for nltools pipelines.

Terminals are the final step in a pipeline that produce results.
They execute prediction, classification, or other evaluation tasks
within cross-validation folds.

**Classes:**

Name | Description
---- | -----------
[`ISCTerminal`](#nltools.pipelines.terminals.ISCTerminal) | ISC terminal for multi-subject pipelines.
[`PredictTerminal`](#nltools.pipelines.terminals.PredictTerminal) | Prediction/classification terminal for CV pipelines.
[`RSATerminal`](#nltools.pipelines.terminals.RSATerminal) | RSA terminal for multi-subject pipelines.



##### Classes###### `nltools.pipelines.terminals.ISCTerminal`

```python
ISCTerminal(method: str = 'pairwise', metric: str = 'median', n_permute: int = 5000, parallel: str = 'cpu', kwargs: Dict[str, Any] = dict()) -> None
```

ISC terminal for multi-subject pipelines.

Computes inter-subject correlation across subjects in the pipeline.
Uses the ISC permutation test from nltools.algorithms.inference.isc.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ISC computation method: 'pairwise' (default) or 'leave-one-out'. | <code>'pairwise'</code>
`metric` | <code>[str](#str)</code> | Summary statistic: 'median' (default, robust) or 'mean' (Fisher z-transformed). | <code>'median'</code>
`n_permute` | <code>[int](#int)</code> | Number of bootstrap iterations for p-value computation. Default is 5000. | <code>5000</code>
`parallel` | <code>[str](#str)</code> | Parallelization method: 'cpu' (default), 'gpu', or None. | <code>'cpu'</code>
`kwargs` | <code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | Additional arguments passed to isc_permutation_test. | <code>[dict](#dict)()</code>

Examples
--------
>>> terminal = ISCTerminal(method='pairwise', n_permute=1000)
>>> result = terminal.fit_evaluate(data_list)
>>> print(f"ISC: {result.isc:.3f}, p: {result.p:.3f}")

**Functions:**

Name | Description
---- | -----------
[`fit_evaluate`](#nltools.pipelines.terminals.ISCTerminal.fit_evaluate) | Compute ISC across subjects.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`kwargs`](#nltools.pipelines.terminals.ISCTerminal.kwargs) | <code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | 
[`method`](#nltools.pipelines.terminals.ISCTerminal.method) | <code>[str](#str)</code> | 
[`metric`](#nltools.pipelines.terminals.ISCTerminal.metric) | <code>[str](#str)</code> | 
[`n_permute`](#nltools.pipelines.terminals.ISCTerminal.n_permute) | <code>[int](#int)</code> | 
[`parallel`](#nltools.pipelines.terminals.ISCTerminal.parallel) | <code>[str](#str)</code> | 



####### Attributes######## `nltools.pipelines.terminals.ISCTerminal.kwargs`

```python
kwargs: Dict[str, Any] = field(default_factory=dict)
```

######## `nltools.pipelines.terminals.ISCTerminal.method`

```python
method: str = 'pairwise'
```

######## `nltools.pipelines.terminals.ISCTerminal.metric`

```python
metric: str = 'median'
```

######## `nltools.pipelines.terminals.ISCTerminal.n_permute`

```python
n_permute: int = 5000
```

######## `nltools.pipelines.terminals.ISCTerminal.parallel`

```python
parallel: str = 'cpu'
```



####### Functions######## `nltools.pipelines.terminals.ISCTerminal.fit_evaluate`

```python
fit_evaluate(data: list, **kwargs: list) -> 'ISCResult'
```

Compute ISC across subjects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)</code> | List of subject data arrays. Each array should have shape (n_observations, n_features) where n_observations is the same across subjects (e.g., timepoints in fMRI). | *required*

**Returns:**

Type | Description
---- | -----------
<code>'ISCResult'</code> | Result containing ISC values, p-values, and confidence intervals.

###### `nltools.pipelines.terminals.PredictTerminal`

```python
PredictTerminal(y: NDArray, algorithm: str = 'ridge', kwargs: Dict[str, Any] = dict()) -> None
```

Prediction/classification terminal for CV pipelines.

Fits a prediction model on training data and evaluates on test data
within each CV fold.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` | <code>[NDArray](#numpy.typing.NDArray)</code> | Target variable to predict (labels or continuous values). | *required*
`algorithm` | <code>[str](#str)</code> | Prediction algorithm. Regression options: 'ridge' (default, L2), 'lasso' (L1), 'elastic' (L1+L2), 'svr' (kernel-based), 'rf' (random forest, auto-detected). Classification options: 'svm' (kernel-based), 'logistic' (linear), 'rf' (auto-detected for discrete y). | <code>'ridge'</code>
`kwargs` | <code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | Additional arguments passed to the sklearn model constructor. Common kwargs: ``class_weight='balanced'`` for imbalanced classification, ``C`` for regularization strength (svm, logistic), ``alpha`` for regularization strength (ridge, lasso, elastic). | <code>[dict](#dict)()</code>

Examples
--------
Basic classification::

    >>> terminal = PredictTerminal(y=labels, algorithm='svm', kwargs={'C': 1.0})

Balanced classification for imbalanced data::

    >>> terminal = PredictTerminal(
    ...     y=imbalanced_labels,
    ...     algorithm='svm',
    ...     kwargs={'class_weight': 'balanced'}
    ... )

Logistic regression with balanced classes::

    >>> terminal = PredictTerminal(
    ...     y=binary_labels,
    ...     algorithm='logistic',
    ...     kwargs={'class_weight': 'balanced', 'C': 0.1}
    ... )

**Functions:**

Name | Description
---- | -----------
[`fit_evaluate`](#nltools.pipelines.terminals.PredictTerminal.fit_evaluate) | Fit model on training data and evaluate on test data.
[`with_y`](#nltools.pipelines.terminals.PredictTerminal.with_y) | Create copy with different target variable.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`algorithm`](#nltools.pipelines.terminals.PredictTerminal.algorithm) | <code>[str](#str)</code> | 
[`kwargs`](#nltools.pipelines.terminals.PredictTerminal.kwargs) | <code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | 
[`y`](#nltools.pipelines.terminals.PredictTerminal.y) | <code>[NDArray](#numpy.typing.NDArray)</code> | 



####### Attributes######## `nltools.pipelines.terminals.PredictTerminal.algorithm`

```python
algorithm: str = 'ridge'
```

######## `nltools.pipelines.terminals.PredictTerminal.kwargs`

```python
kwargs: Dict[str, Any] = field(default_factory=dict)
```

######## `nltools.pipelines.terminals.PredictTerminal.y`

```python
y: NDArray
```



####### Functions######## `nltools.pipelines.terminals.PredictTerminal.fit_evaluate`

```python
fit_evaluate(train_data: NDArray, test_data: NDArray, train_idx: NDArray[np.intp], test_idx: NDArray[np.intp], fitted_stack: Any) -> 'FoldResult'
```

Fit model on training data and evaluate on test data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`train_data` | <code>[NDArray](#numpy.typing.NDArray)</code> | Transformed training features, shape (n_train, n_features). | *required*
`test_data` | <code>[NDArray](#numpy.typing.NDArray)</code> | Transformed test features, shape (n_test, n_features). | *required*
`train_idx` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Original indices of training samples. | *required*
`test_idx` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Original indices of test samples. | *required*
`fitted_stack` | <code>[Any](#typing.Any)</code> | Stack of fitted transforms for this fold. | *required*

**Returns:**

Type | Description
---- | -----------
<code>'FoldResult'</code> | Result containing score, predictions, indices, and fitted stack.

######## `nltools.pipelines.terminals.PredictTerminal.with_y`

```python
with_y(new_y: NDArray) -> 'PredictTerminal'
```

Create copy with different target variable.

Useful for permutation testing.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`new_y` | <code>[NDArray](#numpy.typing.NDArray)</code> | New target variable. | *required*

**Returns:**

Type | Description
---- | -----------
<code>'PredictTerminal'</code> | New terminal with updated y.

###### `nltools.pipelines.terminals.RSATerminal`

```python
RSATerminal(model_rdm: NDArray, method: str = 'spearman', n_permute: int = 5000, kwargs: Dict[str, Any] = dict()) -> None
```

RSA terminal for multi-subject pipelines.

Computes representational similarity analysis by correlating neural RDMs
with a model RDM.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model_rdm` | <code>[NDArray](#numpy.typing.NDArray)</code> | Model RDM to correlate with neural RDMs. Should be a symmetric matrix or upper triangle (condensed form). | *required*
`method` | <code>[str](#str)</code> | Correlation method: 'spearman' (default), 'pearson', or 'kendall'. | <code>'spearman'</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations for p-value computation. Default is 5000. | <code>5000</code>
`kwargs` | <code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | Additional arguments passed to correlation computation. | <code>[dict](#dict)()</code>

Examples
--------
>>> model = np.random.rand(10, 10)  # 10 conditions
>>> model = (model + model.T) / 2  # Make symmetric
>>> terminal = RSATerminal(model_rdm=model, method='spearman')
>>> result = terminal.fit_evaluate(neural_rdm)

**Functions:**

Name | Description
---- | -----------
[`fit_evaluate`](#nltools.pipelines.terminals.RSATerminal.fit_evaluate) | Compute RSA correlation between neural and model RDMs.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`kwargs`](#nltools.pipelines.terminals.RSATerminal.kwargs) | <code>[Dict](#typing.Dict)[[str](#str), [Any](#typing.Any)]</code> | 
[`method`](#nltools.pipelines.terminals.RSATerminal.method) | <code>[str](#str)</code> | 
[`model_rdm`](#nltools.pipelines.terminals.RSATerminal.model_rdm) | <code>[NDArray](#numpy.typing.NDArray)</code> | 
[`n_permute`](#nltools.pipelines.terminals.RSATerminal.n_permute) | <code>[int](#int)</code> | 



####### Attributes######## `nltools.pipelines.terminals.RSATerminal.kwargs`

```python
kwargs: Dict[str, Any] = field(default_factory=dict)
```

######## `nltools.pipelines.terminals.RSATerminal.method`

```python
method: str = 'spearman'
```

######## `nltools.pipelines.terminals.RSATerminal.model_rdm`

```python
model_rdm: NDArray
```

######## `nltools.pipelines.terminals.RSATerminal.n_permute`

```python
n_permute: int = 5000
```



####### Functions######## `nltools.pipelines.terminals.RSATerminal.fit_evaluate`

```python
fit_evaluate(data: NDArray, **kwargs: NDArray) -> 'RSAResult'
```

Compute RSA correlation between neural and model RDMs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[NDArray](#numpy.typing.NDArray)</code> | Neural data to compute RDM from, or pre-computed RDM. If 2D square, treated as RDM (upper triangle extracted). If 1D, treated as condensed RDM. If 2D non-square (n_conditions, n_features), RDM is computed using correlation distance. | *required*

**Returns:**

Type | Description
---- | -----------
<code>'RSAResult'</code> | Result containing correlation coefficient and p-value.

