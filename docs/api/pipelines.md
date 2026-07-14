## `pipelines`

Pipeline infrastructure for nltools.

This module provides a fluent API for building data processing pipelines
with cross-validation support.

Pipeline
    Base pipeline for chained transforms with optional CV.
CVScheme
    Cross-validation scheme configuration.
FittedStack
    Collection of fitted transforms for inverse transform support.

Protocols:
TransformStep
    Protocol for pipeline transform steps.
FittedTransform
    Protocol for fitted transform objects.
Terminal
    Protocol for terminal operations.

Examples:
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
[`base`](#base) | Pipeline base infrastructure for nltools.
[`cv`](#cv) | Cross-validation scheme configuration for nltools pipelines.
[`multi_subject`](#multi-subject) | Multi-subject pipeline for cross-subject analyses.
[`pool`](#pool) | Pool infrastructure for multi-subject aggregation.
[`results`](#results) | Result containers for nltools pipelines.
[`steps`](#steps) | Transform steps for nltools pipelines.
[`terminals`](#terminals) | Terminal operations for nltools pipelines.

**Classes:**

Name | Description
---- | -----------
[`AlignStep`](#alignstep) | Cross-subject alignment via SRM or HyperAlignment.
[`CVResult`](#cvresult) | Cross-validation result container.
[`CVScheme`](#cvscheme) | Protocol for cross-validation schemes.
[`CVSchemeImpl`](#cvschemeimpl) | Cross-validation scheme configuration.
[`FittedAlign`](#fittedalign) | Fitted alignment model.
[`FittedStack`](#fittedstack) | Collection of fitted transforms for inverse transform support.
[`FittedTransform`](#fittedtransform) | Protocol for fitted transform objects.
[`FoldResult`](#foldresult) | Result from a single CV fold.
[`ISCResult`](#iscresult) | Result from ISC terminal computation.
[`ISCTerminal`](#iscterminal) | ISC terminal for multi-subject pipelines.
[`MultiSubjectPipeline`](#multisubjectpipeline) | Pipeline for multi-subject neuroimaging analyses.
[`NestedCVScheme`](#nestedcvscheme) | Nested cross-validation for hyperparameter tuning.
[`NormalizeStep`](#normalizestep) | Normalization transform step.
[`PermutationResult`](#permutationresult) | Result from permutation testing.
[`PipeStep`](#pipestep) | Wrapper for sklearn-compatible transformers.
[`Pipeline`](#pipeline) | Base pipeline for chained transforms with optional cross-validation.
[`PooledData`](#pooleddata) | Aggregated data from multiple subjects.
[`PredictTerminal`](#predictterminal) | Prediction/classification terminal for CV pipelines.
[`RSAResult`](#rsaresult) | Result from RSA terminal computation.
[`RSATerminal`](#rsaterminal) | RSA terminal for multi-subject pipelines.
[`ReduceStep`](#reducestep) | Dimensionality reduction step.
[`ResultDict`](#resultdict) | Dictionary of StatResults, one per contrast.
[`StatResult`](#statresult) | Result of statistical test.
[`Terminal`](#terminal) | Protocol for terminal operations that end a pipeline.
[`TransformStep`](#transformstep) | Protocol for pipeline transform steps.



### Classes

#### `AlignStep`

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

Examples:
>>> import numpy as np
>>> # Create synthetic multi-subject data
>>> data = [np.random.randn(100, 50) for _ in range(5)]
>>> step = AlignStep(method='srm', n_features=10)
>>> fitted = step.fit(data)
>>> aligned = fitted.transform(data)

**Methods:**

Name | Description
---- | -----------
[`fit`](#fit) | Fit alignment model on list of subject data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#invertible) | <code>[bool](#bool)</code> | Check if alignment is invertible.
[`kwargs`](#kwargs) |  | 
[`method`](#method) |  | 
`n_features` |  | 
`n_iter` |  | 
`n_jobs` |  | 
`new_subject` |  | 
`parallel` |  | 
`scheme` |  | 

##### Methods

###### `fit`

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
[`fold_results`](#fold-results) | <code>[list](#list)[[FoldResult](#nltools.pipelines.results.FoldResult)]</code> | 
[`is_fully_invertible`](#is-fully-invertible) | <code>[bool](#bool)</code> | Check if all transform steps are invertible.
`mean_score` | <code>[float](#float)</code> | Mean score across all folds.
`n_folds` | <code>[int](#int)</code> | Number of cross-validation folds.
[`pipeline`](#pipeline) | <code>[Any](#typing.Any)</code> | 
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

#### `CVScheme`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for cross-validation schemes.

Compatible with scikit-learn CV splitters and custom implementations.

**Methods:**

Name | Description
---- | -----------
[`split`](#split) | Generate train/test index splits.



##### Methods

###### `split`

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

#### `CVSchemeImpl`

```python
CVSchemeImpl(k: int | None = None, scheme: CVSchemeType = 'kfold', split_by: str | None = None, n: int = 1000, random_state: int | None = None) -> None
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
`k` | <code>[int](#int) \| None</code> | Number of folds (for kfold scheme). Defaults to 5 if scheme is 'kfold'. | <code>None</code>
`scheme` | <code>[CVSchemeType](#nltools.pipelines.cv.CVSchemeType)</code> | CV scheme type. One of 'kfold', 'loso', 'loro', 'bootstrap', or 'permutation'. | <code>'kfold'</code>
`split_by` | <code>[str](#str) \| None</code> | Attribute to split by ('runs', 'subjects', 'sessions'). Used for documentation purposes with loso/loro schemes. | <code>None</code>
`n` | <code>[int](#int)</code> | Number of bootstrap iterations (for bootstrap scheme). Defaults to 1000. | <code>1000</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. If provided, sets the numpy random seed during initialization. | <code>None</code>

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

**Methods:**

Name | Description
---- | -----------
[`n_splits`](#n-splits) | Return number of splits.
[`split`](#split) | Generate train/test indices for each fold.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`is_loro`](#is-loro) | <code>[bool](#bool)</code> | Check if this is leave-one-run-out.
`is_loso` | <code>[bool](#bool)</code> | Check if this is leave-one-subject-out.
`k` | <code>[int](#int) \| None</code> | 
`n` | <code>[int](#int)</code> | 
`random_state` | <code>[int](#int) \| None</code> | 
`scheme` | <code>[CVSchemeType](#nltools.pipelines.cv.CVSchemeType)</code> | 
`split_by` | <code>[str](#str) \| None</code> | 

##### Methods

###### `n_splits`

```python
n_splits(data: Any = None, groups: NDArray[np.intp] | None = None) -> int
```

Return number of splits.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (unused for most schemes, kept for API consistency). | <code>None</code>
`groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | Group labels for grouped CV. Required for 'loso' and 'loro'. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of splits/folds that will be generated.

###### `split`

```python
split(data: Any, groups: NDArray[np.intp] | None = None) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]
```

Generate train/test indices for each fold.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (used for length). Can be any object with __len__ or a numpy array with shape attribute. | *required*
`groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | Group labels for grouped CV (runs, subjects, etc.). Required for 'loso' and 'loro' schemes. | <code>None</code>

**Yields:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Tuple of (train_indices, test_indices) for each fold.

#### `FittedAlign`

```python
FittedAlign(model: Any, method: str, new_subject_method: str = 'procrustes') -> None
```

Fitted alignment model.

Holds a fitted SRM or HyperAlignment model and applies transformations.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`model` | <code>[Any](#typing.Any)</code> | Fitted SRM or HyperAlignment instance.
[`method`](#method) | <code>[str](#str)</code> | The alignment method used ('srm' or 'hyperalignment').
`new_subject_method` | <code>[str](#str)</code> | Method for aligning held-out subjects in LOSO CV.

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#inverse-transform) | Reverse alignment (only for full-rank hyperalignment).
[`transform`](#transform) | Transform subjects that were in training.
[`transform_new_subject`](#transform-new-subject) | Align a new subject not in training (for LOSO).

##### Methods

###### `inverse_transform`

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

###### `transform`

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

###### `transform_new_subject`

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

#### `FittedStack`

```python
FittedStack(steps: list[FittedTransform] = list()) -> None
```

Collection of fitted transforms for inverse transform support.

Maintains the sequence of fitted transforms from a pipeline execution,
enabling inverse transformation back to the original data space.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`steps`](#steps) | <code>[list](#list)[[FittedTransform](#nltools.pipelines.base.FittedTransform)]</code> | Ordered list of fitted transforms.

Examples:
>>> stack = FittedStack()
>>> stack.append(fitted_pca)
>>> stack.append(fitted_normalize)
>>> original_space = stack.inverse_transform(predictions)

**Methods:**

Name | Description
---- | -----------
[`append`](#append) | Add a fitted transform to the stack.
[`inverse_transform`](#inverse-transform) | Apply inverse transforms in reverse order.

##### Methods

###### `append`

```python
append(fitted_step: FittedTransform) -> None
```

Add a fitted transform to the stack.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fitted_step` | <code>[FittedTransform](#nltools.pipelines.base.FittedTransform)</code> | Fitted transform to append. | *required*

###### `inverse_transform`

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

#### `FittedTransform`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for fitted transform objects.

A fitted transform holds the learned parameters from fitting on training
data and can apply the transformation to new data.

<details class="note" open markdown="1">
<summary>Note</summary>

Not all transforms are invertible. Check the parent TransformStep's
``invertible`` attribute or use ``hasattr`` before calling ``inverse_transform``.

</details>

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#inverse-transform) | Apply the inverse transformation to data.
[`transform`](#transform) | Apply the learned transformation to data.



##### Methods

###### `inverse_transform`

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

###### `transform`

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
[`fitted_stack`](#fitted-stack) | <code>[Any](#typing.Any)</code> | Stack of fitted transforms for inverse transform support.

##### Methods

###### `summary`

```python
summary() -> str
```

Return formatted summary string.

#### `ISCTerminal`

```python
ISCTerminal(method: str = 'pairwise', metric: str = 'median', n_permute: int = 5000, parallel: str = 'cpu', kwargs: dict[str, Any] = dict()) -> None
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
`kwargs` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Additional arguments passed to isc_permutation_test. | <code>[dict](#dict)()</code>

Examples:
>>> terminal = ISCTerminal(method='pairwise', n_permute=1000)
>>> result = terminal.fit_evaluate(data_list)
>>> print(f"ISC: {result.isc:.3f}, p: {result.p:.3f}")

**Methods:**

Name | Description
---- | -----------
[`fit_evaluate`](#fit-evaluate) | Compute ISC across subjects.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`kwargs`](#kwargs) | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | 
[`method`](#method) | <code>[str](#str)</code> | 
`metric` | <code>[str](#str)</code> | 
`n_permute` | <code>[int](#int)</code> | 
`parallel` | <code>[str](#str)</code> | 

##### Methods

###### `fit_evaluate`

```python
fit_evaluate(data: list, **kwargs: list) -> ISCResult
```

Compute ISC across subjects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)</code> | List of subject data arrays. Each array should have shape (n_observations, n_features) where n_observations is the same across subjects (e.g., timepoints in fMRI). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ISCResult](#nltools.pipelines.results.ISCResult)</code> | Result containing ISC values, p-values, and confidence intervals.

#### `MultiSubjectPipeline`

```python
MultiSubjectPipeline(data: list[NDArray], cv: Any | None = None, groups: NDArray[np.intp] | None = None, steps: list[Any] = list(), _is_lazy: bool = False) -> None
```

Pipeline for multi-subject neuroimaging analyses.

Operates on a list of subject data arrays, supporting:
- LOSO (leave-one-subject-out): Train on N-1 subjects, test on 1
- Run-based CV: Split runs within each subject
- Pooling across subjects for group analyses

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)[[NDArray](#numpy.typing.NDArray)]</code> | List of subject data arrays, each shape (n_obs, n_voxels). | *required*
`cv` | <code>[Any](#typing.Any) \| None</code> | Cross-validation scheme configuration. | <code>None</code>
`groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | Group labels for CV splits (e.g., run labels). | <code>None</code>
`steps` | <code>[list](#list)[[Any](#typing.Any)]</code> | Transform steps to apply. | <code>[list](#list)()</code>

Examples:
>>> # LOSO CV
>>> pipeline = MultiSubjectPipeline(subject_data, cv=CVScheme(scheme='loso'))
>>> result = pipeline.normalize().predict(y, algorithm='svm')

>>> # Run-based CV across subjects
>>> pipeline = MultiSubjectPipeline(subject_data, cv=CVScheme(scheme='loro'), groups=runs)
>>> result = pipeline.predict(y)

**Methods:**

Name | Description
---- | -----------
[`align`](#align) | Add cross-subject alignment step to pipeline.
[`isc`](#isc) | Compute inter-subject correlation across subjects.
[`normalize`](#normalize) | Add normalization step (per-subject).
[`pipe`](#pipe) | Add custom sklearn transformer.
[`predict`](#predict) | Execute pipeline with CV and return prediction results.
[`reduce`](#reduce) | Add dimensionality reduction step.
[`rsa`](#rsa) | Compute representational similarity analysis.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`cv`](#cv) | <code>[Any](#typing.Any) \| None</code> | 
`data` | <code>[list](#list)[[NDArray](#numpy.typing.NDArray)]</code> | 
`groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | 
`n_steps` | <code>[int](#int)</code> | Number of transform steps.
`n_subjects` | <code>[int](#int)</code> | Number of subjects in the multi-subject dataset.
[`steps`](#steps) | <code>[list](#list)[[Any](#typing.Any)]</code> | 

##### Methods

###### `align`

```python
align(method: str = 'srm', scheme: str = 'global', n_features: int | None = 50, new_subject: str = 'procrustes', **kwargs: str) -> MultiSubjectPipeline
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
<code>[MultiSubjectPipeline](#nltools.pipelines.multi_subject.MultiSubjectPipeline)</code> | New pipeline with alignment step added.

Examples:
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

###### `isc`

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

Examples:
>>> result = (
...     MultiSubjectPipeline(data=subjects, cv=None)
...     .normalize()
...     .isc(method='pairwise', n_permute=1000)
... )
>>> print(f"ISC: {result.isc:.3f}, p: {result.p:.3f}")

###### `normalize`

```python
normalize(method: str = 'zscore', **kwargs: str) -> MultiSubjectPipeline
```

Add normalization step (per-subject).

###### `pipe`

```python
pipe(transformer) -> MultiSubjectPipeline
```

Add custom sklearn transformer.

###### `predict`

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

**Examples:**

Basic regression with LOSO CV:

```python
result = pipeline.cv('loso').predict(subject_labels, algorithm='ridge')
```
Classification with balanced classes:

```python
result = pipeline.cv('loso').predict(
    group_labels, algorithm='svm', class_weight='balanced'
)
```
Logistic regression with regularization:

```python
result = pipeline.cv('loso').predict(
    binary_labels, algorithm='logistic', C=0.1, class_weight='balanced'
)
```

###### `reduce`

```python
reduce(method: str = 'pca', n_components: int | None = None, **kwargs: int | None) -> MultiSubjectPipeline
```

Add dimensionality reduction step.

###### `rsa`

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

Examples:
>>> model = np.corrcoef(conditions)  # Theoretical model
>>> result = (
...     MultiSubjectPipeline(data=subjects, cv=None)
...     .normalize()
...     .rsa(model_rdm=model, method='spearman')
... )
>>> print(f"r = {result.correlation:.3f}, p = {result.p_value:.3f}")

#### `NestedCVScheme`

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

**Methods:**

Name | Description
---- | -----------
[`n_inner_splits`](#n-inner-splits) | Return number of inner splits per outer fold.
[`n_outer_splits`](#n-outer-splits) | Return number of outer splits.
[`split`](#split) | Generate nested cross-validation splits.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`inner`](#inner) | <code>[CVScheme](#nltools.pipelines.cv.CVScheme)</code> | 
`outer` | <code>[CVScheme](#nltools.pipelines.cv.CVScheme)</code> | 

##### Methods

###### `n_inner_splits`

```python
n_inner_splits(data: Any = None, groups: NDArray[np.intp] | None = None) -> int
```

Return number of inner splits per outer fold.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (unused for most schemes). | <code>None</code>
`groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | Group labels for inner loop. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of inner folds per outer fold.

###### `n_outer_splits`

```python
n_outer_splits(data: Any = None, groups: NDArray[np.intp] | None = None) -> int
```

Return number of outer splits.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (unused for most schemes). | <code>None</code>
`groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | Group labels for outer loop. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of outer folds.

###### `split`

```python
split(data: Any, groups: NDArray[np.intp] | None = None, inner_groups: NDArray[np.intp] | None = None) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp], Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]]]
```

Generate nested cross-validation splits.

For each outer fold, yields the outer train/test indices and an
iterator over inner train/validation splits. The inner splits are
indices into the outer training set, not the original data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (used for length). | *required*
`groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | Group labels for outer loop (e.g., subject IDs). Required if outer.scheme is 'loso' or 'loro'. | <code>None</code>
`inner_groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | Group labels for inner loop. If provided, these are indexed by outer_train to get inner groups. | <code>None</code>

**Yields:**

Type | Description
---- | -----------
<code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Tuple of (outer_train_idx, outer_test_idx, inner_splits_iterator)
<code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | where outer_train_idx and outer_test_idx are arrays of sample
<code>[Iterator](#collections.abc.Iterator)[[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]]</code> | indices, and inner_splits_iterator yields (inner_train, inner_val)
<code>[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [Iterator](#collections.abc.Iterator)[[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]]]</code> | tuples with indices relative to outer_train_idx.

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

#### `NormalizeStep`

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

Examples:
>>> import numpy as np
>>> data = np.array([[1, 2], [3, 4], [5, 6]])
>>> step = NormalizeStep(method='zscore')
>>> fitted = step.fit(data)
>>> normalized = fitted.transform(data)
>>> restored = fitted.inverse_transform(normalized)
>>> np.allclose(data, restored)
True

**Methods:**

Name | Description
---- | -----------
[`fit`](#fit) | Compute normalization parameters from data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`axis`](#axis) | <code>[int](#int)</code> | 
[`invertible`](#invertible) | <code>[bool](#bool)</code> | 
[`method`](#method) | <code>[str](#str)</code> | 

##### Methods

###### `fit`

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
[`n_permutations`](#n-permutations) | <code>[int](#int)</code> | Number of permutations performed.

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

#### `PipeStep`

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

Examples:
>>> from sklearn.preprocessing import StandardScaler
>>> import numpy as np
>>> data = np.random.randn(100, 10)
>>> step = PipeStep(transformer=StandardScaler())
>>> fitted = step.fit(data)
>>> transformed = fitted.transform(data)
>>> restored = fitted.inverse_transform(transformed)
>>> np.allclose(data, restored)
True

**Methods:**

Name | Description
---- | -----------
[`fit`](#fit) | Fit transformer to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#invertible) | <code>[bool](#bool)</code> | Check if the transformer supports inverse_transform.
[`transformer`](#transformer) | <code>[Any](#typing.Any)</code> | 

##### Methods

###### `fit`

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

#### `Pipeline`

```python
Pipeline(data: Any, cv: CVScheme | None = None, steps: list[TransformStep] = list(), _is_lazy: bool = False) -> None
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
`cv` | <code>[CVScheme](#nltools.pipelines.base.CVScheme) \| None</code> | Cross-validation scheme. Required for terminal methods like predict(). | <code>None</code>
`steps` | <code>[list](#list)[[TransformStep](#nltools.pipelines.base.TransformStep)]</code> | List of transform steps (typically not set directly). | <code>[list](#list)()</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`_is_lazy` | <code>[bool](#bool)</code> | Whether pipeline is in lazy evaluation mode (future feature).

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

**Methods:**

Name | Description
---- | -----------
[`copy`](#copy) | Create a shallow copy of the pipeline.
[`normalize`](#normalize) | Add a normalization step to the pipeline.
[`pipe`](#pipe) | Add a custom transformer to the pipeline.
[`predict`](#predict) | Execute pipeline with cross-validation and return prediction results.
[`reduce`](#reduce) | Add a dimensionality reduction step to the pipeline.

##### Methods

###### `copy`

```python
copy() -> Pipeline
```

Create a shallow copy of the pipeline.

**Returns:**

Type | Description
---- | -----------
<code>[Pipeline](#nltools.pipelines.base.Pipeline)</code> | New pipeline instance with same configuration.

###### `normalize`

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

###### `pipe`

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

###### `predict`

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

###### `reduce`

```python
reduce(method: str = 'pca', n_components: int | None = None, **kwargs: Any) -> Pipeline
```

Add a dimensionality reduction step to the pipeline.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Reduction method. Options: 'pca', 'ica', 'nmf', 'srm'. Default is 'pca'. | <code>'pca'</code>
`n_components` | <code>[int](#int) \| None</code> | Number of components to keep. | <code>None</code>
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

#### `PooledData`

```python
PooledData(data: NDArray, param: str, condition_names: list[str] | None = None, subject_ids: list[str] | None = None, mask: Any | None = None, fitted_state: Any | None = None, save_path: str | None = None) -> None
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
`condition_names` | <code>[list](#list)[[str](#str)] \| None</code> | Names of conditions if multi-condition data. | <code>None</code>
`subject_ids` | <code>[list](#list)[[str](#str)] \| None</code> | Subject identifiers. | <code>None</code>
`fitted_state` | <code>[Any](#typing.Any) \| None</code> | Saved fitted models for repool() functionality. | <code>None</code>
`save_path` | <code>[str](#str) \| None</code> | Path where data was saved. | <code>None</code>

Examples:
>>> # Two-stage GLM
>>> pool = bc.fit(model='glm', X=designs).pool(param='beta')
>>> result = pool.fit(model='ttest', contrast='face-house')
>>>
>>> # Reuse for multiple contrasts
>>> result1 = pool.fit(model='ttest', contrast='face-house')
>>> result2 = pool.fit(model='ttest', contrast='face-object')

**Methods:**

Name | Description
---- | -----------
[`cv`](#cv) | Create CV pipeline on pooled data.
[`fit`](#fit) | Fit second-level statistical model.
[`load`](#load) | Load pooled data from disk.
[`repool`](#repool) | Re-extract different parameter from saved fitted state.
[`save`](#save) | Save pooled data to disk.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`condition_names`](#condition-names) | <code>[list](#list)[[str](#str)] \| None</code> | 
`data` | <code>[NDArray](#numpy.typing.NDArray)</code> | 
`fitted_state` | <code>[Any](#typing.Any) \| None</code> | 
`mask` | <code>[Any](#typing.Any) \| None</code> | 
`n_conditions` | <code>[int](#int) \| None</code> | Number of conditions (None if single-condition).
`n_subjects` | <code>[int](#int)</code> | Number of subjects in the pooled dataset (first dimension of data).
`n_voxels` | <code>[int](#int)</code> | Number of voxels (last dimension of data array).
`param` | <code>[str](#str)</code> | 
`save_path` | <code>[str](#str) \| None</code> | 
`shape` | <code>[tuple](#tuple)</code> | Shape of the pooled data array as (n_subjects[, n_conditions], n_voxels).
`subject_ids` | <code>[list](#list)[[str](#str)] \| None</code> | 

##### Methods

###### `cv`

```python
cv(k: int | None = None, scheme: CVSchemeType = 'kfold', **kwargs: CVSchemeType) -> Pipeline
```

Create CV pipeline on pooled data.

Useful for classification on pooled betas.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`k` | <code>[int](#int) \| None</code> | Number of folds. | <code>None</code>
`scheme` | <code>[CVSchemeType](#nltools.pipelines.pool.CVSchemeType)</code> | CV scheme ('kfold', 'loso', 'loro', 'bootstrap'). | <code>'kfold'</code>

**Returns:**

Type | Description
---- | -----------
<code>[Pipeline](#nltools.pipelines.base.Pipeline)</code> | Pipeline for classification on pooled data.

###### `fit`

```python
fit(model: str, contrast: str | None = None, contrasts: list[str] | None = None, X: NDArray | None = None, **kwargs: NDArray | None) -> StatResult | ResultDict
```

Fit second-level statistical model.

This is a terminal method - executes immediately (eager).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model` | <code>[str](#str)</code> | Statistical model type: 'ttest' (one-sample or two-sample with X), 'paired_ttest', or 'anova'. | *required*
`contrast` | <code>[str](#str) \| None</code> | Single contrast specification (e.g., 'face-house'). | <code>None</code>
`contrasts` | <code>[list](#list)[[str](#str)] \| None</code> | Multiple contrasts - returns ResultDict. | <code>None</code>
`X` | <code>[NDArray](#numpy.typing.NDArray) \| None</code> | Design matrix for two-sample tests or ANOVA. | <code>None</code>
`**kwargs` |  | Additional arguments for the statistical test. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[StatResult](#nltools.pipelines.pool.StatResult) \| [ResultDict](#nltools.pipelines.pool.ResultDict)</code> | Statistical results. ResultDict if multiple contrasts specified.

Examples:
>>> result = pool.fit(model='ttest', contrast='face-house')
>>> result.t_map.max()

>>> results = pool.fit(model='ttest', contrasts=['A-B', 'A-C', 'B-C'])
>>> results['A-B'].threshold(method='fdr')

###### `load`

```python
load(path: str) -> PooledData
```

Load pooled data from disk.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`path` | <code>[str](#str)</code> | Path to saved data. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[PooledData](#nltools.pipelines.pool.PooledData)</code> | Loaded pooled data.

###### `repool`

```python
repool(param: str) -> PooledData
```

Re-extract different parameter from saved fitted state.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`param` | <code>[str](#str)</code> | Parameter to extract (e.g., 'residual', 't'). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[PooledData](#nltools.pipelines.pool.PooledData)</code> | New PooledData with the requested parameter.

###### `save`

```python
save(path: str) -> None
```

Save pooled data to disk.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`path` | <code>[str](#str)</code> | Output path (directory or .npz file). | *required*

#### `PredictTerminal`

```python
PredictTerminal(y: NDArray, algorithm: str = 'ridge', kwargs: dict[str, Any] = dict()) -> None
```

Prediction/classification terminal for CV pipelines.

Fits a prediction model on training data and evaluates on test data
within each CV fold.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` | <code>[NDArray](#numpy.typing.NDArray)</code> | Target variable to predict (labels or continuous values). | *required*
`algorithm` | <code>[str](#str)</code> | Prediction algorithm. Regression options: 'ridge' (default, L2), 'lasso' (L1), 'elastic' (L1+L2), 'svr' (kernel-based), 'rf' (random forest, auto-detected). Classification options: 'svm' (kernel-based), 'logistic' (linear), 'rf' (auto-detected for discrete y). | <code>'ridge'</code>
`kwargs` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Additional arguments passed to the sklearn model constructor. Common kwargs: ``class_weight='balanced'`` for imbalanced classification, ``C`` for regularization strength (svm, logistic), ``alpha`` for regularization strength (ridge, lasso, elastic). | <code>[dict](#dict)()</code>

**Examples:**

Basic classification:

```python
>>> terminal = PredictTerminal(y=labels, algorithm='svm', kwargs={'C': 1.0})
```
Balanced classification for imbalanced data:

```python
>>> terminal = PredictTerminal(
...     y=imbalanced_labels,
...     algorithm='svm',
...     kwargs={'class_weight': 'balanced'}
... )
```
Logistic regression with balanced classes:

```python
>>> terminal = PredictTerminal(
...     y=binary_labels,
...     algorithm='logistic',
...     kwargs={'class_weight': 'balanced', 'C': 0.1}
... )
```

**Methods:**

Name | Description
---- | -----------
[`fit_evaluate`](#fit-evaluate) | Fit model on training data and evaluate on test data.
[`with_y`](#with-y) | Create copy with different target variable.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`algorithm`](#algorithm) | <code>[str](#str)</code> | 
[`kwargs`](#kwargs) | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | 
`y` | <code>[NDArray](#numpy.typing.NDArray)</code> | 

##### Methods

###### `fit_evaluate`

```python
fit_evaluate(train_data: NDArray, test_data: NDArray, train_idx: NDArray[np.intp], test_idx: NDArray[np.intp], fitted_stack: Any) -> FoldResult
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
<code>[FoldResult](#nltools.pipelines.results.FoldResult)</code> | Result containing score, predictions, indices, and fitted stack.

###### `with_y`

```python
with_y(new_y: NDArray) -> PredictTerminal
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
<code>[PredictTerminal](#nltools.pipelines.terminals.PredictTerminal)</code> | New terminal with updated y.

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
[`ci`](#ci) | <code>[tuple](#tuple)</code> | Confidence interval (lower, upper).
[`method`](#method) | <code>[str](#str)</code> | Correlation method used (e.g., 'spearman', 'pearson').
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

#### `RSATerminal`

```python
RSATerminal(model_rdm: NDArray, method: str = 'spearman', n_permute: int = 5000, kwargs: dict[str, Any] = dict()) -> None
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
`kwargs` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Additional arguments passed to correlation computation. | <code>[dict](#dict)()</code>

Examples:
>>> model = np.random.rand(10, 10)  # 10 conditions
>>> model = (model + model.T) / 2  # Make symmetric
>>> terminal = RSATerminal(model_rdm=model, method='spearman')
>>> result = terminal.fit_evaluate(neural_rdm)

**Methods:**

Name | Description
---- | -----------
[`fit_evaluate`](#fit-evaluate) | Compute RSA correlation between neural and model RDMs.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`kwargs`](#kwargs) | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | 
[`method`](#method) | <code>[str](#str)</code> | 
`model_rdm` | <code>[NDArray](#numpy.typing.NDArray)</code> | 
`n_permute` | <code>[int](#int)</code> | 

##### Methods

###### `fit_evaluate`

```python
fit_evaluate(data: NDArray, **kwargs: NDArray) -> RSAResult
```

Compute RSA correlation between neural and model RDMs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[NDArray](#numpy.typing.NDArray)</code> | Neural data to compute RDM from, or pre-computed RDM. If 2D square, treated as RDM (upper triangle extracted). If 1D, treated as condensed RDM. If 2D non-square (n_conditions, n_features), RDM is computed using correlation distance. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[RSAResult](#nltools.pipelines.results.RSAResult)</code> | Result containing correlation coefficient and p-value.

#### `ReduceStep`

```python
ReduceStep(method: str = 'pca', n_components: int | None = None, random_state: int | None = None) -> None
```

Dimensionality reduction step.

Fits a dimensionality reduction model to training data and transforms
new data to the reduced space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Reduction method: 'pca' (Principal Component Analysis, invertible) or 'ica' (Independent Component Analysis, not invertible). Default is 'pca'. | <code>'pca'</code>
`n_components` | <code>[int](#int) \| None</code> | Number of components to keep. If None, keeps all components. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>

Examples:
>>> import numpy as np
>>> data = np.random.randn(100, 50)
>>> step = ReduceStep(method='pca', n_components=10)
>>> fitted = step.fit(data)
>>> reduced = fitted.transform(data)
>>> reduced.shape
(100, 10)

**Methods:**

Name | Description
---- | -----------
[`fit`](#fit) | Fit reduction model to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#invertible) | <code>[bool](#bool)</code> | Check if the reduction method supports inverse transform.
[`method`](#method) | <code>[str](#str)</code> | 
`n_components` | <code>[int](#int) \| None</code> | 
`random_state` | <code>[int](#int) \| None</code> | 

##### Methods

###### `fit`

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

#### `ResultDict`

Bases: <code>[dict](#dict)</code>

Dictionary of StatResults, one per contrast.

Provides convenience methods for batch operations.

**Methods:**

Name | Description
---- | -----------
[`threshold_all`](#threshold-all) | Apply thresholding to all results.



##### Methods

###### `threshold_all`

```python
threshold_all(method: str = 'fdr', alpha: float = 0.05) -> ResultDict
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
<code>[ResultDict](#nltools.pipelines.pool.ResultDict)</code> | New dict with thresholded results.

#### `StatResult`

```python
StatResult(t_map: NDArray | None = None, f_map: NDArray | None = None, p_map: NDArray | None = None, contrast: str | None = None, df: int | None = None) -> None
```

Result of statistical test.

Holds statistical maps and provides thresholding utilities.

**Methods:**

Name | Description
---- | -----------
[`threshold`](#threshold) | Apply multiple comparison correction.
[`to_nifti`](#to-nifti) | Save as NIfTI file.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`contrast`](#contrast) | <code>[str](#str) \| None</code> | 
`df` | <code>[int](#int) \| None</code> | 
`f_map` | <code>[NDArray](#numpy.typing.NDArray) \| None</code> | 
`p_map` | <code>[NDArray](#numpy.typing.NDArray) \| None</code> | 
`t_map` | <code>[NDArray](#numpy.typing.NDArray) \| None</code> | 

##### Methods

###### `threshold`

```python
threshold(method: str = 'fdr', alpha: float = 0.05) -> StatResult
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
<code>[StatResult](#nltools.pipelines.pool.StatResult)</code> | New result with thresholded maps.

###### `to_nifti`

```python
to_nifti(path: str, mask: str = None) -> None
```

Save as NIfTI file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`path` | <code>[str](#str)</code> | Output path. | *required*
`mask` |  | Mask to use for reconstruction. | <code>None</code>

#### `Terminal`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for terminal operations that end a pipeline.

Terminals perform the final computation (e.g., prediction, similarity)
and produce results for each CV fold.

**Methods:**

Name | Description
---- | -----------
[`fit_evaluate`](#fit-evaluate) | Fit on training data and evaluate on test data.



##### Methods

###### `fit_evaluate`

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

#### `TransformStep`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for pipeline transform steps.

A transform step defines a transformation that can be fitted to data.
Steps are added to a Pipeline and executed sequentially during CV.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#invertible) | <code>[bool](#bool)</code> | Whether this transform supports inverse_transform.

Examples:
>>> class MyStep:
...     invertible = True
...     def fit(self, data):
...         return MyFittedTransform(learned_params)

**Methods:**

Name | Description
---- | -----------
[`fit`](#fit) | Fit the transform to data.

##### Methods

###### `fit`

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



### Modules

#### `base`

Pipeline base infrastructure for nltools.

This module provides the foundational classes and protocols for building
chainable transform pipelines with optional cross-validation support.

The design follows an immutable pattern where each transform method returns
a new Pipeline instance, enabling fluent method chaining without side effects.

Example:
>>> pipeline = (
...     Pipeline(data, cv=kfold)
...     .normalize(method='zscore')
...     .reduce(method='pca', n_components=50)
...     .predict(y, algorithm='ridge')
... )

**Classes:**

Name | Description
---- | -----------
[`CVScheme`](#cvscheme) | Protocol for cross-validation schemes.
[`FittedStack`](#fittedstack) | Collection of fitted transforms for inverse transform support.
[`FittedTransform`](#fittedtransform) | Protocol for fitted transform objects.
[`Pipeline`](#pipeline) | Base pipeline for chained transforms with optional cross-validation.
[`Terminal`](#terminal) | Protocol for terminal operations that end a pipeline.
[`TransformStep`](#transformstep) | Protocol for pipeline transform steps.



##### Attributes

##### Classes

###### `CVScheme`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for cross-validation schemes.

Compatible with scikit-learn CV splitters and custom implementations.

**Methods:**

Name | Description
---- | -----------
[`split`](#split) | Generate train/test index splits.



####### Functions##

###### `split`

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

###### `FittedStack`

```python
FittedStack(steps: list[FittedTransform] = list()) -> None
```

Collection of fitted transforms for inverse transform support.

Maintains the sequence of fitted transforms from a pipeline execution,
enabling inverse transformation back to the original data space.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`steps`](#steps) | <code>[list](#list)[[FittedTransform](#nltools.pipelines.base.FittedTransform)]</code> | Ordered list of fitted transforms.

Examples:
>>> stack = FittedStack()
>>> stack.append(fitted_pca)
>>> stack.append(fitted_normalize)
>>> original_space = stack.inverse_transform(predictions)

**Methods:**

Name | Description
---- | -----------
[`append`](#append) | Add a fitted transform to the stack.
[`inverse_transform`](#inverse-transform) | Apply inverse transforms in reverse order.



####### Attributes##

###### `is_fully_invertible`

```python
is_fully_invertible: bool
```

Check if all steps support inverse transform.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if all steps have callable inverse_transform methods.

######## `steps`

```python
steps: list[FittedTransform] = field(default_factory=list)
```



####### Functions##

###### `append`

```python
append(fitted_step: FittedTransform) -> None
```

Add a fitted transform to the stack.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fitted_step` | <code>[FittedTransform](#nltools.pipelines.base.FittedTransform)</code> | Fitted transform to append. | *required*

######## `inverse_transform`

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

###### `FittedTransform`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for fitted transform objects.

A fitted transform holds the learned parameters from fitting on training
data and can apply the transformation to new data.

<details class="note" open markdown="1">
<summary>Note</summary>

Not all transforms are invertible. Check the parent TransformStep's
``invertible`` attribute or use ``hasattr`` before calling ``inverse_transform``.

</details>

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#inverse-transform) | Apply the inverse transformation to data.
[`transform`](#transform) | Apply the learned transformation to data.



####### Functions##

###### `inverse_transform`

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

######## `transform`

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

###### `Pipeline`

```python
Pipeline(data: Any, cv: CVScheme | None = None, steps: list[TransformStep] = list(), _is_lazy: bool = False) -> None
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
`cv` | <code>[CVScheme](#nltools.pipelines.base.CVScheme) \| None</code> | Cross-validation scheme. Required for terminal methods like predict(). | <code>None</code>
`steps` | <code>[list](#list)[[TransformStep](#nltools.pipelines.base.TransformStep)]</code> | List of transform steps (typically not set directly). | <code>[list](#list)()</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`_is_lazy` | <code>[bool](#bool)</code> | Whether pipeline is in lazy evaluation mode (future feature).

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

**Methods:**

Name | Description
---- | -----------
[`copy`](#copy) | Create a shallow copy of the pipeline.
[`normalize`](#normalize) | Add a normalization step to the pipeline.
[`pipe`](#pipe) | Add a custom transformer to the pipeline.
[`predict`](#predict) | Execute pipeline with cross-validation and return prediction results.
[`reduce`](#reduce) | Add a dimensionality reduction step to the pipeline.



####### Attributes##

###### `cv`

```python
cv: CVScheme | None = None
```

######## `data`

```python
data: Any
```

######## `n_steps`

```python
n_steps: int
```

Return number of transform steps.

######## `steps`

```python
steps: list[TransformStep] = field(default_factory=list)
```



####### Functions##

###### `copy`

```python
copy() -> Pipeline
```

Create a shallow copy of the pipeline.

**Returns:**

Type | Description
---- | -----------
<code>[Pipeline](#nltools.pipelines.base.Pipeline)</code> | New pipeline instance with same configuration.

######## `normalize`

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

######## `pipe`

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

######## `predict`

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

######## `reduce`

```python
reduce(method: str = 'pca', n_components: int | None = None, **kwargs: Any) -> Pipeline
```

Add a dimensionality reduction step to the pipeline.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Reduction method. Options: 'pca', 'ica', 'nmf', 'srm'. Default is 'pca'. | <code>'pca'</code>
`n_components` | <code>[int](#int) \| None</code> | Number of components to keep. | <code>None</code>
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

###### `Terminal`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for terminal operations that end a pipeline.

Terminals perform the final computation (e.g., prediction, similarity)
and produce results for each CV fold.

**Methods:**

Name | Description
---- | -----------
[`fit_evaluate`](#fit-evaluate) | Fit on training data and evaluate on test data.



####### Functions##

###### `fit_evaluate`

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

###### `TransformStep`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for pipeline transform steps.

A transform step defines a transformation that can be fitted to data.
Steps are added to a Pipeline and executed sequentially during CV.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#invertible) | <code>[bool](#bool)</code> | Whether this transform supports inverse_transform.

Examples:
>>> class MyStep:
...     invertible = True
...     def fit(self, data):
...         return MyFittedTransform(learned_params)

**Methods:**

Name | Description
---- | -----------
[`fit`](#fit) | Fit the transform to data.



####### Attributes##

###### `invertible`

```python
invertible: bool
```



####### Functions##

###### `fit`

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

#### `cv`

Cross-validation scheme configuration for nltools pipelines.

This module provides a unified interface for configuring cross-validation
strategies used across nltools analysis pipelines.

**Classes:**

Name | Description
---- | -----------
[`CVScheme`](#cvscheme) | Cross-validation scheme configuration.
[`NestedCVScheme`](#nestedcvscheme) | Nested cross-validation for hyperparameter tuning.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`CVSchemeType`](#cvschemetype) |  | 



##### Attributes

###### `CVSchemeType`

```python
CVSchemeType = Literal['kfold', 'loso', 'loro', 'bootstrap', 'permutation']
```



##### Classes

###### `CVScheme`

```python
CVScheme(k: int | None = None, scheme: CVSchemeType = 'kfold', split_by: str | None = None, n: int = 1000, random_state: int | None = None) -> None
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
`k` | <code>[int](#int) \| None</code> | Number of folds (for kfold scheme). Defaults to 5 if scheme is 'kfold'. | <code>None</code>
`scheme` | <code>[CVSchemeType](#nltools.pipelines.cv.CVSchemeType)</code> | CV scheme type. One of 'kfold', 'loso', 'loro', 'bootstrap', or 'permutation'. | <code>'kfold'</code>
`split_by` | <code>[str](#str) \| None</code> | Attribute to split by ('runs', 'subjects', 'sessions'). Used for documentation purposes with loso/loro schemes. | <code>None</code>
`n` | <code>[int](#int)</code> | Number of bootstrap iterations (for bootstrap scheme). Defaults to 1000. | <code>1000</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. If provided, sets the numpy random seed during initialization. | <code>None</code>

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

**Methods:**

Name | Description
---- | -----------
[`n_splits`](#n-splits) | Return number of splits.
[`split`](#split) | Generate train/test indices for each fold.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`is_loro`](#is-loro) | <code>[bool](#bool)</code> | Check if this is leave-one-run-out.
`is_loso` | <code>[bool](#bool)</code> | Check if this is leave-one-subject-out.
`k` | <code>[int](#int) \| None</code> | 
`n` | <code>[int](#int)</code> | 
`random_state` | <code>[int](#int) \| None</code> | 
`scheme` | <code>[CVSchemeType](#nltools.pipelines.cv.CVSchemeType)</code> | 
`split_by` | <code>[str](#str) \| None</code> | 



####### Attributes##

###### `is_loro`

```python
is_loro: bool
```

Check if this is leave-one-run-out.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if scheme is 'loro', False otherwise.

######## `is_loso`

```python
is_loso: bool
```

Check if this is leave-one-subject-out.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if scheme is 'loso', False otherwise.

######## `k`

```python
k: int | None = None
```

######## `n`

```python
n: int = 1000
```

######## `random_state`

```python
random_state: int | None = None
```

######## `scheme`

```python
scheme: CVSchemeType = 'kfold'
```

######## `split_by`

```python
split_by: str | None = None
```



####### Functions##

###### `n_splits`

```python
n_splits(data: Any = None, groups: NDArray[np.intp] | None = None) -> int
```

Return number of splits.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (unused for most schemes, kept for API consistency). | <code>None</code>
`groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | Group labels for grouped CV. Required for 'loso' and 'loro'. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of splits/folds that will be generated.

######## `split`

```python
split(data: Any, groups: NDArray[np.intp] | None = None) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]
```

Generate train/test indices for each fold.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (used for length). Can be any object with __len__ or a numpy array with shape attribute. | *required*
`groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | Group labels for grouped CV (runs, subjects, etc.). Required for 'loso' and 'loro' schemes. | <code>None</code>

**Yields:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Tuple of (train_indices, test_indices) for each fold.

###### `NestedCVScheme`

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

**Methods:**

Name | Description
---- | -----------
[`n_inner_splits`](#n-inner-splits) | Return number of inner splits per outer fold.
[`n_outer_splits`](#n-outer-splits) | Return number of outer splits.
[`split`](#split) | Generate nested cross-validation splits.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`inner`](#inner) | <code>[CVScheme](#nltools.pipelines.cv.CVScheme)</code> | 
`outer` | <code>[CVScheme](#nltools.pipelines.cv.CVScheme)</code> | 



####### Attributes##

###### `inner`

```python
inner: CVScheme
```

######## `outer`

```python
outer: CVScheme
```



####### Functions##

###### `n_inner_splits`

```python
n_inner_splits(data: Any = None, groups: NDArray[np.intp] | None = None) -> int
```

Return number of inner splits per outer fold.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (unused for most schemes). | <code>None</code>
`groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | Group labels for inner loop. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of inner folds per outer fold.

######## `n_outer_splits`

```python
n_outer_splits(data: Any = None, groups: NDArray[np.intp] | None = None) -> int
```

Return number of outer splits.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (unused for most schemes). | <code>None</code>
`groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | Group labels for outer loop. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of outer folds.

######## `split`

```python
split(data: Any, groups: NDArray[np.intp] | None = None, inner_groups: NDArray[np.intp] | None = None) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp], Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]]]
```

Generate nested cross-validation splits.

For each outer fold, yields the outer train/test indices and an
iterator over inner train/validation splits. The inner splits are
indices into the outer training set, not the original data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (used for length). | *required*
`groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | Group labels for outer loop (e.g., subject IDs). Required if outer.scheme is 'loso' or 'loro'. | <code>None</code>
`inner_groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | Group labels for inner loop. If provided, these are indexed by outer_train to get inner groups. | <code>None</code>

**Yields:**

Type | Description
---- | -----------
<code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Tuple of (outer_train_idx, outer_test_idx, inner_splits_iterator)
<code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | where outer_train_idx and outer_test_idx are arrays of sample
<code>[Iterator](#collections.abc.Iterator)[[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]]</code> | indices, and inner_splits_iterator yields (inner_train, inner_val)
<code>[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [Iterator](#collections.abc.Iterator)[[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]]]</code> | tuples with indices relative to outer_train_idx.

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

#### `multi_subject`

Multi-subject pipeline for cross-subject analyses.

This module extends the base Pipeline to handle multi-subject data,
supporting leave-one-subject-out (LOSO) and run-based CV schemes.

**Classes:**

Name | Description
---- | -----------
[`MultiSubjectPipeline`](#multisubjectpipeline) | Pipeline for multi-subject neuroimaging analyses.



##### Classes

###### `MultiSubjectPipeline`

```python
MultiSubjectPipeline(data: list[NDArray], cv: Any | None = None, groups: NDArray[np.intp] | None = None, steps: list[Any] = list(), _is_lazy: bool = False) -> None
```

Pipeline for multi-subject neuroimaging analyses.

Operates on a list of subject data arrays, supporting:
- LOSO (leave-one-subject-out): Train on N-1 subjects, test on 1
- Run-based CV: Split runs within each subject
- Pooling across subjects for group analyses

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)[[NDArray](#numpy.typing.NDArray)]</code> | List of subject data arrays, each shape (n_obs, n_voxels). | *required*
`cv` | <code>[Any](#typing.Any) \| None</code> | Cross-validation scheme configuration. | <code>None</code>
`groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | Group labels for CV splits (e.g., run labels). | <code>None</code>
`steps` | <code>[list](#list)[[Any](#typing.Any)]</code> | Transform steps to apply. | <code>[list](#list)()</code>

Examples:
>>> # LOSO CV
>>> pipeline = MultiSubjectPipeline(subject_data, cv=CVScheme(scheme='loso'))
>>> result = pipeline.normalize().predict(y, algorithm='svm')

>>> # Run-based CV across subjects
>>> pipeline = MultiSubjectPipeline(subject_data, cv=CVScheme(scheme='loro'), groups=runs)
>>> result = pipeline.predict(y)

**Methods:**

Name | Description
---- | -----------
[`align`](#align) | Add cross-subject alignment step to pipeline.
[`isc`](#isc) | Compute inter-subject correlation across subjects.
[`normalize`](#normalize) | Add normalization step (per-subject).
[`pipe`](#pipe) | Add custom sklearn transformer.
[`predict`](#predict) | Execute pipeline with CV and return prediction results.
[`reduce`](#reduce) | Add dimensionality reduction step.
[`rsa`](#rsa) | Compute representational similarity analysis.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`cv`](#cv) | <code>[Any](#typing.Any) \| None</code> | 
`data` | <code>[list](#list)[[NDArray](#numpy.typing.NDArray)]</code> | 
`groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | 
`n_steps` | <code>[int](#int)</code> | Number of transform steps.
`n_subjects` | <code>[int](#int)</code> | Number of subjects in the multi-subject dataset.
[`steps`](#steps) | <code>[list](#list)[[Any](#typing.Any)]</code> | 



####### Attributes##

###### `cv`

```python
cv: Any | None = None
```

######## `data`

```python
data: list[NDArray]
```

######## `groups`

```python
groups: NDArray[np.intp] | None = None
```

######## `n_steps`

```python
n_steps: int
```

Number of transform steps.

######## `n_subjects`

```python
n_subjects: int
```

Number of subjects in the multi-subject dataset.

######## `steps`

```python
steps: list[Any] = field(default_factory=list)
```



####### Functions##

###### `align`

```python
align(method: str = 'srm', scheme: str = 'global', n_features: int | None = 50, new_subject: str = 'procrustes', **kwargs: str) -> MultiSubjectPipeline
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
<code>[MultiSubjectPipeline](#nltools.pipelines.multi_subject.MultiSubjectPipeline)</code> | New pipeline with alignment step added.

Examples:
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

######## `isc`

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

Examples:
>>> result = (
...     MultiSubjectPipeline(data=subjects, cv=None)
...     .normalize()
...     .isc(method='pairwise', n_permute=1000)
... )
>>> print(f"ISC: {result.isc:.3f}, p: {result.p:.3f}")

######## `normalize`

```python
normalize(method: str = 'zscore', **kwargs: str) -> MultiSubjectPipeline
```

Add normalization step (per-subject).

######## `pipe`

```python
pipe(transformer) -> MultiSubjectPipeline
```

Add custom sklearn transformer.

######## `predict`

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

**Examples:**

Basic regression with LOSO CV:

```python
result = pipeline.cv('loso').predict(subject_labels, algorithm='ridge')
```
Classification with balanced classes:

```python
result = pipeline.cv('loso').predict(
    group_labels, algorithm='svm', class_weight='balanced'
)
```
Logistic regression with regularization:

```python
result = pipeline.cv('loso').predict(
    binary_labels, algorithm='logistic', C=0.1, class_weight='balanced'
)
```

######## `reduce`

```python
reduce(method: str = 'pca', n_components: int | None = None, **kwargs: int | None) -> MultiSubjectPipeline
```

Add dimensionality reduction step.

######## `rsa`

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

Examples:
>>> model = np.corrcoef(conditions)  # Theoretical model
>>> result = (
...     MultiSubjectPipeline(data=subjects, cv=None)
...     .normalize()
...     .rsa(model_rdm=model, method='spearman')
... )
>>> print(f"r = {result.correlation:.3f}, p = {result.p_value:.3f}")

#### `pool`

Pool infrastructure for multi-subject aggregation.

This module provides classes for pooling data across subjects and
enabling two-stage analyses (e.g., first-level GLM -> group t-test).

The pool() method serves as an execution boundary - everything before
it is executed lazily, and pool() triggers execution and aggregation.

**Classes:**

Name | Description
---- | -----------
[`PooledData`](#pooleddata) | Aggregated data from multiple subjects.
[`ResultDict`](#resultdict) | Dictionary of StatResults, one per contrast.
[`StatResult`](#statresult) | Result of statistical test.



##### Attributes

##### Classes

###### `PooledData`

```python
PooledData(data: NDArray, param: str, condition_names: list[str] | None = None, subject_ids: list[str] | None = None, mask: Any | None = None, fitted_state: Any | None = None, save_path: str | None = None) -> None
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
`condition_names` | <code>[list](#list)[[str](#str)] \| None</code> | Names of conditions if multi-condition data. | <code>None</code>
`subject_ids` | <code>[list](#list)[[str](#str)] \| None</code> | Subject identifiers. | <code>None</code>
`fitted_state` | <code>[Any](#typing.Any) \| None</code> | Saved fitted models for repool() functionality. | <code>None</code>
`save_path` | <code>[str](#str) \| None</code> | Path where data was saved. | <code>None</code>

Examples:
>>> # Two-stage GLM
>>> pool = bc.fit(model='glm', X=designs).pool(param='beta')
>>> result = pool.fit(model='ttest', contrast='face-house')
>>>
>>> # Reuse for multiple contrasts
>>> result1 = pool.fit(model='ttest', contrast='face-house')
>>> result2 = pool.fit(model='ttest', contrast='face-object')

**Methods:**

Name | Description
---- | -----------
[`cv`](#cv) | Create CV pipeline on pooled data.
[`fit`](#fit) | Fit second-level statistical model.
[`load`](#load) | Load pooled data from disk.
[`repool`](#repool) | Re-extract different parameter from saved fitted state.
[`save`](#save) | Save pooled data to disk.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`condition_names`](#condition-names) | <code>[list](#list)[[str](#str)] \| None</code> | 
`data` | <code>[NDArray](#numpy.typing.NDArray)</code> | 
`fitted_state` | <code>[Any](#typing.Any) \| None</code> | 
`mask` | <code>[Any](#typing.Any) \| None</code> | 
`n_conditions` | <code>[int](#int) \| None</code> | Number of conditions (None if single-condition).
`n_subjects` | <code>[int](#int)</code> | Number of subjects in the pooled dataset (first dimension of data).
`n_voxels` | <code>[int](#int)</code> | Number of voxels (last dimension of data array).
`param` | <code>[str](#str)</code> | 
`save_path` | <code>[str](#str) \| None</code> | 
`shape` | <code>[tuple](#tuple)</code> | Shape of the pooled data array as (n_subjects[, n_conditions], n_voxels).
`subject_ids` | <code>[list](#list)[[str](#str)] \| None</code> | 



####### Attributes##

###### `condition_names`

```python
condition_names: list[str] | None = None
```

######## `data`

```python
data: NDArray
```

######## `fitted_state`

```python
fitted_state: Any | None = field(default=None, repr=False)
```

######## `mask`

```python
mask: Any | None = field(default=None, repr=False)
```

######## `n_conditions`

```python
n_conditions: int | None
```

Number of conditions (None if single-condition).

######## `n_subjects`

```python
n_subjects: int
```

Number of subjects in the pooled dataset (first dimension of data).

######## `n_voxels`

```python
n_voxels: int
```

Number of voxels (last dimension of data array).

######## `param`

```python
param: str
```

######## `save_path`

```python
save_path: str | None = None
```

######## `shape`

```python
shape: tuple
```

Shape of the pooled data array as (n_subjects[, n_conditions], n_voxels).

######## `subject_ids`

```python
subject_ids: list[str] | None = None
```



####### Functions##

###### `cv`

```python
cv(k: int | None = None, scheme: CVSchemeType = 'kfold', **kwargs: CVSchemeType) -> Pipeline
```

Create CV pipeline on pooled data.

Useful for classification on pooled betas.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`k` | <code>[int](#int) \| None</code> | Number of folds. | <code>None</code>
`scheme` | <code>[CVSchemeType](#nltools.pipelines.pool.CVSchemeType)</code> | CV scheme ('kfold', 'loso', 'loro', 'bootstrap'). | <code>'kfold'</code>

**Returns:**

Type | Description
---- | -----------
<code>[Pipeline](#nltools.pipelines.base.Pipeline)</code> | Pipeline for classification on pooled data.

######## `fit`

```python
fit(model: str, contrast: str | None = None, contrasts: list[str] | None = None, X: NDArray | None = None, **kwargs: NDArray | None) -> StatResult | ResultDict
```

Fit second-level statistical model.

This is a terminal method - executes immediately (eager).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model` | <code>[str](#str)</code> | Statistical model type: 'ttest' (one-sample or two-sample with X), 'paired_ttest', or 'anova'. | *required*
`contrast` | <code>[str](#str) \| None</code> | Single contrast specification (e.g., 'face-house'). | <code>None</code>
`contrasts` | <code>[list](#list)[[str](#str)] \| None</code> | Multiple contrasts - returns ResultDict. | <code>None</code>
`X` | <code>[NDArray](#numpy.typing.NDArray) \| None</code> | Design matrix for two-sample tests or ANOVA. | <code>None</code>
`**kwargs` |  | Additional arguments for the statistical test. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[StatResult](#nltools.pipelines.pool.StatResult) \| [ResultDict](#nltools.pipelines.pool.ResultDict)</code> | Statistical results. ResultDict if multiple contrasts specified.

Examples:
>>> result = pool.fit(model='ttest', contrast='face-house')
>>> result.t_map.max()

>>> results = pool.fit(model='ttest', contrasts=['A-B', 'A-C', 'B-C'])
>>> results['A-B'].threshold(method='fdr')

######## `load`

```python
load(path: str) -> PooledData
```

Load pooled data from disk.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`path` | <code>[str](#str)</code> | Path to saved data. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[PooledData](#nltools.pipelines.pool.PooledData)</code> | Loaded pooled data.

######## `repool`

```python
repool(param: str) -> PooledData
```

Re-extract different parameter from saved fitted state.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`param` | <code>[str](#str)</code> | Parameter to extract (e.g., 'residual', 't'). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[PooledData](#nltools.pipelines.pool.PooledData)</code> | New PooledData with the requested parameter.

######## `save`

```python
save(path: str) -> None
```

Save pooled data to disk.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`path` | <code>[str](#str)</code> | Output path (directory or .npz file). | *required*

###### `ResultDict`

Bases: <code>[dict](#dict)</code>

Dictionary of StatResults, one per contrast.

Provides convenience methods for batch operations.

**Methods:**

Name | Description
---- | -----------
[`threshold_all`](#threshold-all) | Apply thresholding to all results.



####### Functions##

###### `threshold_all`

```python
threshold_all(method: str = 'fdr', alpha: float = 0.05) -> ResultDict
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
<code>[ResultDict](#nltools.pipelines.pool.ResultDict)</code> | New dict with thresholded results.

###### `StatResult`

```python
StatResult(t_map: NDArray | None = None, f_map: NDArray | None = None, p_map: NDArray | None = None, contrast: str | None = None, df: int | None = None) -> None
```

Result of statistical test.

Holds statistical maps and provides thresholding utilities.

**Methods:**

Name | Description
---- | -----------
[`threshold`](#threshold) | Apply multiple comparison correction.
[`to_nifti`](#to-nifti) | Save as NIfTI file.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`contrast`](#contrast) | <code>[str](#str) \| None</code> | 
`df` | <code>[int](#int) \| None</code> | 
`f_map` | <code>[NDArray](#numpy.typing.NDArray) \| None</code> | 
`p_map` | <code>[NDArray](#numpy.typing.NDArray) \| None</code> | 
`t_map` | <code>[NDArray](#numpy.typing.NDArray) \| None</code> | 



####### Attributes##

###### `contrast`

```python
contrast: str | None = None
```

######## `df`

```python
df: int | None = None
```

######## `f_map`

```python
f_map: NDArray | None = None
```

######## `p_map`

```python
p_map: NDArray | None = None
```

######## `t_map`

```python
t_map: NDArray | None = None
```



####### Functions##

###### `threshold`

```python
threshold(method: str = 'fdr', alpha: float = 0.05) -> StatResult
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
<code>[StatResult](#nltools.pipelines.pool.StatResult)</code> | New result with thresholded maps.

######## `to_nifti`

```python
to_nifti(path: str, mask: str = None) -> None
```

Save as NIfTI file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`path` | <code>[str](#str)</code> | Output path. | *required*
`mask` |  | Mask to use for reconstruction. | <code>None</code>

#### `results`

Result containers for nltools pipelines.

This module provides result classes that hold outputs from pipeline execution,
including cross-validation results and per-fold information.

**Classes:**

Name | Description
---- | -----------
[`CVResult`](#cvresult) | Cross-validation result container.
[`FoldResult`](#foldresult) | Result from a single CV fold.
[`ISCResult`](#iscresult) | Result from ISC terminal computation.
[`PermutationResult`](#permutationresult) | Result from permutation testing.
[`RSAResult`](#rsaresult) | Result from RSA terminal computation.



##### Classes

###### `CVResult`

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
[`fold_results`](#fold-results) | <code>[list](#list)[[FoldResult](#nltools.pipelines.results.FoldResult)]</code> | 
[`is_fully_invertible`](#is-fully-invertible) | <code>[bool](#bool)</code> | Check if all transform steps are invertible.
`mean_score` | <code>[float](#float)</code> | Mean score across all folds.
`n_folds` | <code>[int](#int)</code> | Number of cross-validation folds.
[`pipeline`](#pipeline) | <code>[Any](#typing.Any)</code> | 
`predictions` | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | All predictions in original sample order.
`scores` | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | Per-fold prediction scores as a numpy array.
`std_score` | <code>[float](#float)</code> | Standard deviation of scores across folds.



####### Attributes##

###### `fold_results`

```python
fold_results: list[FoldResult]
```

######## `is_fully_invertible`

```python
is_fully_invertible: bool
```

Check if all transform steps are invertible.

######## `mean_score`

```python
mean_score: float
```

Mean score across all folds.

######## `n_folds`

```python
n_folds: int
```

Number of cross-validation folds.

######## `pipeline`

```python
pipeline: Any
```

######## `predictions`

```python
predictions: NDArray[np.floating]
```

All predictions in original sample order.

Reconstructs predictions array with each sample's prediction
from the fold where it was in the test set.

######## `scores`

```python
scores: NDArray[np.floating]
```

Per-fold prediction scores as a numpy array.

######## `std_score`

```python
std_score: float
```

Standard deviation of scores across folds.



####### Functions##

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

######## `summary`

```python
summary() -> str
```

Return formatted summary string.

###### `FoldResult`

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
[`fitted_stack`](#fitted-stack) | <code>[Any](#typing.Any)</code> | Stack of fitted transforms for inverse transform support.



####### Attributes##

###### `fitted_stack`

```python
fitted_stack: Any
```

######## `predictions`

```python
predictions: NDArray[np.floating]
```

######## `score`

```python
score: float
```

######## `test_idx`

```python
test_idx: NDArray[np.intp]
```

######## `train_idx`

```python
train_idx: NDArray[np.intp]
```

###### `ISCResult`

```python
ISCResult(isc: NDArray[np.floating], p: NDArray[np.floating], ci: tuple, method: str, metric: str, n_subjects: int) -> None
```

Result from ISC terminal computation.

Holds intersubject correlation values, p-values, and confidence intervals
from the ISC permutation test.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`isc`](#isc) | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | ISC values. Scalar for single-feature or (n_voxels,) for voxel-wise ISC.
`p` | <code>[NDArray](#numpy.typing.NDArray)[[floating](#numpy.floating)]</code> | P-values (Phipson-Smyth corrected).
[`ci`](#ci) | <code>[tuple](#tuple)</code> | Confidence interval (lower, upper).
[`method`](#method) | <code>[str](#str)</code> | ISC method used ('pairwise' or 'leave-one-out').
`metric` | <code>[str](#str)</code> | Summary metric used ('median' or 'mean').
`n_subjects` | <code>[int](#int)</code> | Number of subjects in the analysis.

**Methods:**

Name | Description
---- | -----------
[`summary`](#summary) | Return formatted summary string.



####### Attributes##

###### `ci`

```python
ci: tuple
```

######## `isc`

```python
isc: NDArray[np.floating]
```

######## `method`

```python
method: str
```

######## `metric`

```python
metric: str
```

######## `n_subjects`

```python
n_subjects: int
```

######## `p`

```python
p: NDArray[np.floating]
```



####### Functions##

###### `summary`

```python
summary() -> str
```

Return formatted summary string.

###### `PermutationResult`

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
[`n_permutations`](#n-permutations) | <code>[int](#int)</code> | Number of permutations performed.

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



####### Attributes##

###### `n_permutations`

```python
n_permutations: int
```

######## `null_distribution`

```python
null_distribution: NDArray[np.floating]
```

######## `null_mean`

```python
null_mean: float
```

Mean of the null distribution.

######## `null_std`

```python
null_std: float
```

Standard deviation of the null distribution.

######## `observed`

```python
observed: CVResult
```

######## `observed_score`

```python
observed_score: float
```

Convenience accessor for observed mean score.

######## `p_value`

```python
p_value: float
```

######## `z_score`

```python
z_score: float
```

Z-score of observed relative to null distribution.



####### Functions##

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

######## `summary`

```python
summary() -> str
```

Return formatted summary string.

###### `RSAResult`

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
[`ci`](#ci) | <code>[tuple](#tuple)</code> | Confidence interval (lower, upper).
[`method`](#method) | <code>[str](#str)</code> | Correlation method used (e.g., 'spearman', 'pearson').
`n_conditions` | <code>[int](#int)</code> | Number of conditions/stimuli in the RDM.

**Methods:**

Name | Description
---- | -----------
[`summary`](#summary) | Return formatted summary string.



####### Attributes##

###### `ci`

```python
ci: tuple
```

######## `correlation`

```python
correlation: float
```

######## `method`

```python
method: str
```

######## `n_conditions`

```python
n_conditions: int
```

######## `p_value`

```python
p_value: float
```



####### Functions##

###### `summary`

```python
summary() -> str
```

Return formatted summary string.

#### `steps`

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
[`AlignStep`](#alignstep) | Cross-subject alignment via SRM or HyperAlignment.
[`FittedAlign`](#fittedalign) | Fitted alignment model.
[`FittedNormalize`](#fittednormalize) | Fitted normalization transform.
[`FittedPipe`](#fittedpipe) | Fitted sklearn transformer wrapper.
[`FittedReduce`](#fittedreduce) | Fitted dimensionality reduction transform.
[`NormalizeStep`](#normalizestep) | Normalization transform step.
[`PipeStep`](#pipestep) | Wrapper for sklearn-compatible transformers.
[`ReduceStep`](#reducestep) | Dimensionality reduction step.



##### Classes

###### `AlignStep`

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

Examples:
>>> import numpy as np
>>> # Create synthetic multi-subject data
>>> data = [np.random.randn(100, 50) for _ in range(5)]
>>> step = AlignStep(method='srm', n_features=10)
>>> fitted = step.fit(data)
>>> aligned = fitted.transform(data)

**Methods:**

Name | Description
---- | -----------
[`fit`](#fit) | Fit alignment model on list of subject data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#invertible) | <code>[bool](#bool)</code> | Check if alignment is invertible.
[`kwargs`](#kwargs) |  | 
[`method`](#method) |  | 
`n_features` |  | 
`n_iter` |  | 
`n_jobs` |  | 
`new_subject` |  | 
`parallel` |  | 
`scheme` |  | 



####### Attributes##

###### `invertible`

```python
invertible: bool
```

Check if alignment is invertible.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if method is hyperalignment (full-rank orthogonal transforms).

######## `kwargs`

```python
kwargs = kwargs
```

######## `method`

```python
method = method
```

######## `n_features`

```python
n_features = n_features
```

######## `n_iter`

```python
n_iter = n_iter
```

######## `n_jobs`

```python
n_jobs = n_jobs
```

######## `new_subject`

```python
new_subject = new_subject
```

######## `parallel`

```python
parallel = parallel
```

######## `scheme`

```python
scheme = scheme
```



####### Functions##

###### `fit`

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

###### `FittedAlign`

```python
FittedAlign(model: Any, method: str, new_subject_method: str = 'procrustes') -> None
```

Fitted alignment model.

Holds a fitted SRM or HyperAlignment model and applies transformations.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`model` | <code>[Any](#typing.Any)</code> | Fitted SRM or HyperAlignment instance.
[`method`](#method) | <code>[str](#str)</code> | The alignment method used ('srm' or 'hyperalignment').
`new_subject_method` | <code>[str](#str)</code> | Method for aligning held-out subjects in LOSO CV.

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#inverse-transform) | Reverse alignment (only for full-rank hyperalignment).
[`transform`](#transform) | Transform subjects that were in training.
[`transform_new_subject`](#transform-new-subject) | Align a new subject not in training (for LOSO).



####### Attributes##

###### `method`

```python
method: str
```

######## `model`

```python
model: Any
```

######## `new_subject_method`

```python
new_subject_method: str = 'procrustes'
```



####### Functions##

###### `inverse_transform`

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

######## `transform`

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

######## `transform_new_subject`

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

###### `FittedNormalize`

```python
FittedNormalize(mean: np.ndarray, std: np.ndarray, method: str) -> None
```

Fitted normalization transform.

Holds the learned parameters (mean, std or min, range) and applies
the transformation to new data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`mean`](#mean) | <code>[ndarray](#numpy.ndarray)</code> | For zscore: the mean. For minmax: the min value.
`std` | <code>[ndarray](#numpy.ndarray)</code> | For zscore: the standard deviation. For minmax: the range (max - min).
[`method`](#method) | <code>[str](#str)</code> | The normalization method ('zscore' or 'minmax').

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#inverse-transform) | Reverse normalization.
[`transform`](#transform) | Apply normalization to data.



####### Attributes##

###### `mean`

```python
mean: np.ndarray
```

######## `method`

```python
method: str
```

######## `std`

```python
std: np.ndarray
```



####### Functions##

###### `inverse_transform`

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

######## `transform`

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

###### `FittedPipe`

```python
FittedPipe(transformer: Any) -> None
```

Fitted sklearn transformer wrapper.

Holds a fitted sklearn transformer and delegates transform calls to it.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`transformer`](#transformer) | <code>[Any](#typing.Any)</code> | Fitted sklearn-compatible transformer.

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#inverse-transform) | Apply inverse transform if supported.
[`transform`](#transform) | Apply the fitted transformer.



####### Attributes##

###### `transformer`

```python
transformer: Any
```



####### Functions##

###### `inverse_transform`

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

######## `transform`

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

###### `FittedReduce`

```python
FittedReduce(model: Any, method: str) -> None
```

Fitted dimensionality reduction transform.

Holds the fitted sklearn model and applies transformations.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`model` | <code>[Any](#typing.Any)</code> | Fitted sklearn decomposition model (PCA, FastICA, etc.).
[`method`](#method) | <code>[str](#str)</code> | The reduction method used.

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#inverse-transform) | Reverse dimensionality reduction (reconstruct original space).
[`transform`](#transform) | Apply dimensionality reduction.



####### Attributes##

###### `method`

```python
method: str
```

######## `model`

```python
model: Any
```



####### Functions##

###### `inverse_transform`

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

######## `transform`

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

###### `NormalizeStep`

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

Examples:
>>> import numpy as np
>>> data = np.array([[1, 2], [3, 4], [5, 6]])
>>> step = NormalizeStep(method='zscore')
>>> fitted = step.fit(data)
>>> normalized = fitted.transform(data)
>>> restored = fitted.inverse_transform(normalized)
>>> np.allclose(data, restored)
True

**Methods:**

Name | Description
---- | -----------
[`fit`](#fit) | Compute normalization parameters from data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`axis`](#axis) | <code>[int](#int)</code> | 
[`invertible`](#invertible) | <code>[bool](#bool)</code> | 
[`method`](#method) | <code>[str](#str)</code> | 



####### Attributes##

###### `axis`

```python
axis: int = 0
```

######## `invertible`

```python
invertible: bool = True
```

######## `method`

```python
method: str = 'zscore'
```



####### Functions##

###### `fit`

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

###### `PipeStep`

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

Examples:
>>> from sklearn.preprocessing import StandardScaler
>>> import numpy as np
>>> data = np.random.randn(100, 10)
>>> step = PipeStep(transformer=StandardScaler())
>>> fitted = step.fit(data)
>>> transformed = fitted.transform(data)
>>> restored = fitted.inverse_transform(transformed)
>>> np.allclose(data, restored)
True

**Methods:**

Name | Description
---- | -----------
[`fit`](#fit) | Fit transformer to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#invertible) | <code>[bool](#bool)</code> | Check if the transformer supports inverse_transform.
[`transformer`](#transformer) | <code>[Any](#typing.Any)</code> | 



####### Attributes##

###### `invertible`

```python
invertible: bool
```

Check if the transformer supports inverse_transform.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if transformer has inverse_transform method.

######## `transformer`

```python
transformer: Any = None
```



####### Functions##

###### `fit`

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

###### `ReduceStep`

```python
ReduceStep(method: str = 'pca', n_components: int | None = None, random_state: int | None = None) -> None
```

Dimensionality reduction step.

Fits a dimensionality reduction model to training data and transforms
new data to the reduced space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Reduction method: 'pca' (Principal Component Analysis, invertible) or 'ica' (Independent Component Analysis, not invertible). Default is 'pca'. | <code>'pca'</code>
`n_components` | <code>[int](#int) \| None</code> | Number of components to keep. If None, keeps all components. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>

Examples:
>>> import numpy as np
>>> data = np.random.randn(100, 50)
>>> step = ReduceStep(method='pca', n_components=10)
>>> fitted = step.fit(data)
>>> reduced = fitted.transform(data)
>>> reduced.shape
(100, 10)

**Methods:**

Name | Description
---- | -----------
[`fit`](#fit) | Fit reduction model to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#invertible) | <code>[bool](#bool)</code> | Check if the reduction method supports inverse transform.
[`method`](#method) | <code>[str](#str)</code> | 
`n_components` | <code>[int](#int) \| None</code> | 
`random_state` | <code>[int](#int) \| None</code> | 



####### Attributes##

###### `invertible`

```python
invertible: bool
```

Check if the reduction method supports inverse transform.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if method is 'pca', False otherwise.

######## `method`

```python
method: str = 'pca'
```

######## `n_components`

```python
n_components: int | None = None
```

######## `random_state`

```python
random_state: int | None = None
```



####### Functions##

###### `fit`

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

#### `terminals`

Terminal operations for nltools pipelines.

Terminals are the final step in a pipeline that produce results.
They execute prediction, classification, or other evaluation tasks
within cross-validation folds.

**Classes:**

Name | Description
---- | -----------
[`ISCTerminal`](#iscterminal) | ISC terminal for multi-subject pipelines.
[`PredictTerminal`](#predictterminal) | Prediction/classification terminal for CV pipelines.
[`RSATerminal`](#rsaterminal) | RSA terminal for multi-subject pipelines.



##### Classes

###### `ISCTerminal`

```python
ISCTerminal(method: str = 'pairwise', metric: str = 'median', n_permute: int = 5000, parallel: str = 'cpu', kwargs: dict[str, Any] = dict()) -> None
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
`kwargs` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Additional arguments passed to isc_permutation_test. | <code>[dict](#dict)()</code>

Examples:
>>> terminal = ISCTerminal(method='pairwise', n_permute=1000)
>>> result = terminal.fit_evaluate(data_list)
>>> print(f"ISC: {result.isc:.3f}, p: {result.p:.3f}")

**Methods:**

Name | Description
---- | -----------
[`fit_evaluate`](#fit-evaluate) | Compute ISC across subjects.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`kwargs`](#kwargs) | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | 
[`method`](#method) | <code>[str](#str)</code> | 
`metric` | <code>[str](#str)</code> | 
`n_permute` | <code>[int](#int)</code> | 
`parallel` | <code>[str](#str)</code> | 



####### Attributes##

###### `kwargs`

```python
kwargs: dict[str, Any] = field(default_factory=dict)
```

######## `method`

```python
method: str = 'pairwise'
```

######## `metric`

```python
metric: str = 'median'
```

######## `n_permute`

```python
n_permute: int = 5000
```

######## `parallel`

```python
parallel: str = 'cpu'
```



####### Functions##

###### `fit_evaluate`

```python
fit_evaluate(data: list, **kwargs: list) -> ISCResult
```

Compute ISC across subjects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)</code> | List of subject data arrays. Each array should have shape (n_observations, n_features) where n_observations is the same across subjects (e.g., timepoints in fMRI). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ISCResult](#nltools.pipelines.results.ISCResult)</code> | Result containing ISC values, p-values, and confidence intervals.

###### `PredictTerminal`

```python
PredictTerminal(y: NDArray, algorithm: str = 'ridge', kwargs: dict[str, Any] = dict()) -> None
```

Prediction/classification terminal for CV pipelines.

Fits a prediction model on training data and evaluates on test data
within each CV fold.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` | <code>[NDArray](#numpy.typing.NDArray)</code> | Target variable to predict (labels or continuous values). | *required*
`algorithm` | <code>[str](#str)</code> | Prediction algorithm. Regression options: 'ridge' (default, L2), 'lasso' (L1), 'elastic' (L1+L2), 'svr' (kernel-based), 'rf' (random forest, auto-detected). Classification options: 'svm' (kernel-based), 'logistic' (linear), 'rf' (auto-detected for discrete y). | <code>'ridge'</code>
`kwargs` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Additional arguments passed to the sklearn model constructor. Common kwargs: ``class_weight='balanced'`` for imbalanced classification, ``C`` for regularization strength (svm, logistic), ``alpha`` for regularization strength (ridge, lasso, elastic). | <code>[dict](#dict)()</code>

**Examples:**

Basic classification:

```python
>>> terminal = PredictTerminal(y=labels, algorithm='svm', kwargs={'C': 1.0})
```
Balanced classification for imbalanced data:

```python
>>> terminal = PredictTerminal(
...     y=imbalanced_labels,
...     algorithm='svm',
...     kwargs={'class_weight': 'balanced'}
... )
```
Logistic regression with balanced classes:

```python
>>> terminal = PredictTerminal(
...     y=binary_labels,
...     algorithm='logistic',
...     kwargs={'class_weight': 'balanced', 'C': 0.1}
... )
```

**Methods:**

Name | Description
---- | -----------
[`fit_evaluate`](#fit-evaluate) | Fit model on training data and evaluate on test data.
[`with_y`](#with-y) | Create copy with different target variable.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`algorithm`](#algorithm) | <code>[str](#str)</code> | 
[`kwargs`](#kwargs) | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | 
`y` | <code>[NDArray](#numpy.typing.NDArray)</code> | 



####### Attributes##

###### `algorithm`

```python
algorithm: str = 'ridge'
```

######## `kwargs`

```python
kwargs: dict[str, Any] = field(default_factory=dict)
```

######## `y`

```python
y: NDArray
```



####### Functions##

###### `fit_evaluate`

```python
fit_evaluate(train_data: NDArray, test_data: NDArray, train_idx: NDArray[np.intp], test_idx: NDArray[np.intp], fitted_stack: Any) -> FoldResult
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
<code>[FoldResult](#nltools.pipelines.results.FoldResult)</code> | Result containing score, predictions, indices, and fitted stack.

######## `with_y`

```python
with_y(new_y: NDArray) -> PredictTerminal
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
<code>[PredictTerminal](#nltools.pipelines.terminals.PredictTerminal)</code> | New terminal with updated y.

###### `RSATerminal`

```python
RSATerminal(model_rdm: NDArray, method: str = 'spearman', n_permute: int = 5000, kwargs: dict[str, Any] = dict()) -> None
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
`kwargs` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Additional arguments passed to correlation computation. | <code>[dict](#dict)()</code>

Examples:
>>> model = np.random.rand(10, 10)  # 10 conditions
>>> model = (model + model.T) / 2  # Make symmetric
>>> terminal = RSATerminal(model_rdm=model, method='spearman')
>>> result = terminal.fit_evaluate(neural_rdm)

**Methods:**

Name | Description
---- | -----------
[`fit_evaluate`](#fit-evaluate) | Compute RSA correlation between neural and model RDMs.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`kwargs`](#kwargs) | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | 
[`method`](#method) | <code>[str](#str)</code> | 
`model_rdm` | <code>[NDArray](#numpy.typing.NDArray)</code> | 
`n_permute` | <code>[int](#int)</code> | 



####### Attributes##

###### `kwargs`

```python
kwargs: dict[str, Any] = field(default_factory=dict)
```

######## `method`

```python
method: str = 'spearman'
```

######## `model_rdm`

```python
model_rdm: NDArray
```

######## `n_permute`

```python
n_permute: int = 5000
```



####### Functions##

###### `fit_evaluate`

```python
fit_evaluate(data: NDArray, **kwargs: NDArray) -> RSAResult
```

Compute RSA correlation between neural and model RDMs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[NDArray](#numpy.typing.NDArray)</code> | Neural data to compute RDM from, or pre-computed RDM. If 2D square, treated as RDM (upper triangle extracted). If 1D, treated as condensed RDM. If 2D non-square (n_conditions, n_features), RDM is computed using correlation distance. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[RSAResult](#nltools.pipelines.results.RSAResult)</code> | Result containing correlation coefficient and p-value.

