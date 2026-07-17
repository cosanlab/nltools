(data-collection-pipesteps-pipesteps)=
## `pipesteps`

Low-level pipeline primitives used by `BrainCollection`.

These are the building blocks that back `BrainCollectionPipeline`: transform
steps (`NormalizeStep`, `ReduceStep`, `PipeStep`), the fitted-stack container
(`FittedStack`), the cross-validation scheme (`CVScheme`), and the transform
protocols. This package is internal; the standalone fluent `Pipeline` /
`MultiSubjectPipeline` orchestration was removed in v0.6.0 — multi-subject CV
now lives on `BrainCollection` (`.cv().standardize().reduce().predict()`) and
custom single-dataset preprocessing uses `model=make_pipeline(...)` on
`BrainData.predict`.

**Modules:**

Name | Description
---- | -----------
[`base`](#data-collection-pipesteps-base) | Low-level pipeline primitives for nltools.
[`cv`](#data-collection-pipesteps-cv) | Cross-validation scheme configuration for nltools pipelines.
[`steps`](#data-collection-pipesteps-steps) | Transform steps for nltools pipelines.

**Classes:**

Name | Description
---- | -----------
[`CVScheme`](#data-collection-pipesteps-cvscheme) | Cross-validation scheme configuration.
[`FittedStack`](#data-collection-pipesteps-fittedstack) | Collection of fitted transforms for inverse transform support.
[`FittedTransform`](#data-collection-pipesteps-fittedtransform) | Protocol for fitted transform objects.
[`NormalizeStep`](#data-collection-pipesteps-normalizestep) | Normalization transform step.
[`PipeStep`](#data-collection-pipesteps-pipestep) | Wrapper for sklearn-compatible transformers.
[`ReduceStep`](#data-collection-pipesteps-reducestep) | Dimensionality reduction step.
[`TransformStep`](#data-collection-pipesteps-transformstep) | Protocol for pipeline transform steps.



### Classes

(data-collection-pipesteps-cvscheme)=
#### `CVScheme`

```python
CVScheme(k: int | None = None, scheme: CVSchemeType = 'kfold', split_by: str | None = None, n: int = 1000, random_state: int | None = None) -> None
```

Cross-validation scheme configuration.

Supports multiple CV strategies:
- kfold: k-fold cross-validation
- loso: leave-one-subject-out (for multi-subject)
- loro: leave-one-run-out
- bootstrap: bootstrap resampling

For the label-permutation accuracy null (the classic MVPA permutation
test), use ``BrainCollectionPipeline.predict(n_permute=...)`` — it is a
dedicated outer loop over shuffled targets, not a train/test split.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`k` | <code>[int](#int) \| None</code> | Number of folds (for kfold scheme). Defaults to 5 if scheme is 'kfold'. | <code>None</code>
`scheme` | <code>[CVSchemeType](#nltools.data.collection.pipesteps.cv.CVSchemeType)</code> | CV scheme type. One of 'kfold', 'loso', 'loro', or 'bootstrap'. | <code>'kfold'</code>
`split_by` | <code>[str](#str) \| None</code> | Attribute to split by ('runs', 'subjects', 'sessions'). Used for documentation purposes with loso/loro schemes. | <code>None</code>
`n` | <code>[int](#int)</code> | Number of resampling iterations (bootstrap draws or permutations). Defaults to 1000. | <code>1000</code>
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

**Methods:**

Name | Description
---- | -----------
[`n_splits`](#data-collection-pipesteps-n-splits) | Return number of splits.
[`split`](#data-collection-pipesteps-split) | Generate train/test indices for each fold.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`is_loro`](#data-collection-pipesteps-is-loro) | <code>[bool](#bool)</code> | Check if this is leave-one-run-out.
`is_loso` | <code>[bool](#bool)</code> | Check if this is leave-one-subject-out.
`k` | <code>[int](#int) \| None</code> | 
`n` | <code>[int](#int)</code> | 
`random_state` | <code>[int](#int) \| None</code> | 
`scheme` | <code>[CVSchemeType](#nltools.data.collection.pipesteps.cv.CVSchemeType)</code> | 
`split_by` | <code>[str](#str) \| None</code> | 

##### Methods

(data-collection-pipesteps-n-splits)=
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

(data-collection-pipesteps-split)=
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

(data-collection-pipesteps-fittedstack)=
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
[`steps`](#data-collection-pipesteps-steps) | <code>[list](#list)[[FittedTransform](#nltools.data.collection.pipesteps.base.FittedTransform)]</code> | Ordered list of fitted transforms.

Examples:
>>> stack = FittedStack()
>>> stack.append(fitted_pca)
>>> stack.append(fitted_normalize)
>>> original_space = stack.inverse_transform(predictions)

**Methods:**

Name | Description
---- | -----------
[`append`](#data-collection-pipesteps-append) | Add a fitted transform to the stack.
[`inverse_transform`](#data-collection-pipesteps-inverse-transform) | Apply inverse transforms in reverse order.

##### Methods

(data-collection-pipesteps-append)=
###### `append`

```python
append(fitted_step: FittedTransform) -> None
```

Add a fitted transform to the stack.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fitted_step` | <code>[FittedTransform](#nltools.data.collection.pipesteps.base.FittedTransform)</code> | Fitted transform to append. | *required*

(data-collection-pipesteps-inverse-transform)=
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

(data-collection-pipesteps-fittedtransform)=
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
[`inverse_transform`](#data-collection-pipesteps-inverse-transform) | Apply the inverse transformation to data.
[`transform`](#data-collection-pipesteps-transform) | Apply the learned transformation to data.



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

(data-collection-pipesteps-transform)=
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

(data-collection-pipesteps-normalizestep)=
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
[`fit`](#data-collection-pipesteps-fit) | Compute normalization parameters from data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`axis`](#data-collection-pipesteps-axis) | <code>[int](#int)</code> | 
[`invertible`](#data-collection-pipesteps-invertible) | <code>[bool](#bool)</code> | 
[`method`](#data-collection-pipesteps-method) | <code>[str](#str)</code> | 

##### Methods

(data-collection-pipesteps-fit)=
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
<code>[FittedNormalize](#nltools.data.collection.pipesteps.steps.FittedNormalize)</code> | Fitted transform that can be applied to new data.

(data-collection-pipesteps-pipestep)=
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
[`fit`](#data-collection-pipesteps-fit) | Fit transformer to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#data-collection-pipesteps-invertible) | <code>[bool](#bool)</code> | Check if the transformer supports inverse_transform.
[`transformer`](#data-collection-pipesteps-transformer) | <code>[Any](#typing.Any)</code> | 

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
<code>[FittedPipe](#nltools.data.collection.pipesteps.steps.FittedPipe)</code> | Fitted transform wrapper.

(data-collection-pipesteps-reducestep)=
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
[`fit`](#data-collection-pipesteps-fit) | Fit reduction model to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#data-collection-pipesteps-invertible) | <code>[bool](#bool)</code> | Check if the reduction method supports inverse transform.
[`method`](#data-collection-pipesteps-method) | <code>[str](#str)</code> | 
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
<code>[FittedReduce](#nltools.data.collection.pipesteps.steps.FittedReduce)</code> | Fitted transform that can be applied to new data.

(data-collection-pipesteps-transformstep)=
#### `TransformStep`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for pipeline transform steps.

A transform step defines a transformation that can be fitted to data.
Steps are added to a Pipeline and executed sequentially during CV.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#data-collection-pipesteps-invertible) | <code>[bool](#bool)</code> | Whether this transform supports inverse_transform.

Examples:
>>> class MyStep:
...     invertible = True
...     def fit(self, data):
...         return MyFittedTransform(learned_params)

**Methods:**

Name | Description
---- | -----------
[`fit`](#data-collection-pipesteps-fit) | Fit the transform to data.

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
<code>[FittedTransform](#nltools.data.collection.pipesteps.base.FittedTransform)</code> | Fitted transform object that can transform new data.



### Modules

(data-collection-pipesteps-base)=
#### `base`

Low-level pipeline primitives for nltools.

Defines the transform protocols (`TransformStep`, `FittedTransform`) and
`FittedStack`, the container that records fitted transforms so a stack can be
inverted. These primitives back `BrainCollectionPipeline`; the standalone fluent
`Pipeline` orchestrator was removed in v0.6.0 in favor of `BrainCollection`'s
native `.cv().standardize().reduce().predict()`.

**Classes:**

Name | Description
---- | -----------
[`FittedStack`](#data-collection-pipesteps-fittedstack) | Collection of fitted transforms for inverse transform support.
[`FittedTransform`](#data-collection-pipesteps-fittedtransform) | Protocol for fitted transform objects.
[`TransformStep`](#data-collection-pipesteps-transformstep) | Protocol for pipeline transform steps.



##### Classes

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
[`steps`](#data-collection-pipesteps-steps) | <code>[list](#list)[[FittedTransform](#nltools.data.collection.pipesteps.base.FittedTransform)]</code> | Ordered list of fitted transforms.

Examples:
>>> stack = FittedStack()
>>> stack.append(fitted_pca)
>>> stack.append(fitted_normalize)
>>> original_space = stack.inverse_transform(predictions)

**Methods:**

Name | Description
---- | -----------
[`append`](#data-collection-pipesteps-append) | Add a fitted transform to the stack.
[`inverse_transform`](#data-collection-pipesteps-inverse-transform) | Apply inverse transforms in reverse order.



####### Attributes##

(data-collection-pipesteps-is-fully-invertible)=
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
`fitted_step` | <code>[FittedTransform](#nltools.data.collection.pipesteps.base.FittedTransform)</code> | Fitted transform to append. | *required*

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
[`inverse_transform`](#data-collection-pipesteps-inverse-transform) | Apply the inverse transformation to data.
[`transform`](#data-collection-pipesteps-transform) | Apply the learned transformation to data.



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

###### `TransformStep`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for pipeline transform steps.

A transform step defines a transformation that can be fitted to data.
Steps are added to a Pipeline and executed sequentially during CV.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#data-collection-pipesteps-invertible) | <code>[bool](#bool)</code> | Whether this transform supports inverse_transform.

Examples:
>>> class MyStep:
...     invertible = True
...     def fit(self, data):
...         return MyFittedTransform(learned_params)

**Methods:**

Name | Description
---- | -----------
[`fit`](#data-collection-pipesteps-fit) | Fit the transform to data.



####### Attributes##

(data-collection-pipesteps-invertible)=
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
<code>[FittedTransform](#nltools.data.collection.pipesteps.base.FittedTransform)</code> | Fitted transform object that can transform new data.

(data-collection-pipesteps-cv)=
#### `cv`

Cross-validation scheme configuration for nltools pipelines.

This module provides a unified interface for configuring cross-validation
strategies used across nltools analysis pipelines.

**Classes:**

Name | Description
---- | -----------
[`CVScheme`](#data-collection-pipesteps-cvscheme) | Cross-validation scheme configuration.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`CVSchemeType` |  | 

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

For the label-permutation accuracy null (the classic MVPA permutation
test), use ``BrainCollectionPipeline.predict(n_permute=...)`` — it is a
dedicated outer loop over shuffled targets, not a train/test split.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`k` | <code>[int](#int) \| None</code> | Number of folds (for kfold scheme). Defaults to 5 if scheme is 'kfold'. | <code>None</code>
`scheme` | <code>[CVSchemeType](#nltools.data.collection.pipesteps.cv.CVSchemeType)</code> | CV scheme type. One of 'kfold', 'loso', 'loro', or 'bootstrap'. | <code>'kfold'</code>
`split_by` | <code>[str](#str) \| None</code> | Attribute to split by ('runs', 'subjects', 'sessions'). Used for documentation purposes with loso/loro schemes. | <code>None</code>
`n` | <code>[int](#int)</code> | Number of resampling iterations (bootstrap draws or permutations). Defaults to 1000. | <code>1000</code>
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

**Methods:**

Name | Description
---- | -----------
[`n_splits`](#data-collection-pipesteps-n-splits) | Return number of splits.
[`split`](#data-collection-pipesteps-split) | Generate train/test indices for each fold.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`is_loro`](#data-collection-pipesteps-is-loro) | <code>[bool](#bool)</code> | Check if this is leave-one-run-out.
`is_loso` | <code>[bool](#bool)</code> | Check if this is leave-one-subject-out.
`k` | <code>[int](#int) \| None</code> | 
`n` | <code>[int](#int)</code> | 
`random_state` | <code>[int](#int) \| None</code> | 
`scheme` | <code>[CVSchemeType](#nltools.data.collection.pipesteps.cv.CVSchemeType)</code> | 
`split_by` | <code>[str](#str) \| None</code> | 



####### Attributes##

(data-collection-pipesteps-is-loro)=
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

(data-collection-pipesteps-steps)=
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
[`FittedNormalize`](#data-collection-pipesteps-fittednormalize) | Fitted normalization transform.
[`FittedPipe`](#data-collection-pipesteps-fittedpipe) | Fitted sklearn transformer wrapper.
[`FittedReduce`](#data-collection-pipesteps-fittedreduce) | Fitted dimensionality reduction transform.
[`NormalizeStep`](#data-collection-pipesteps-normalizestep) | Normalization transform step.
[`PipeStep`](#data-collection-pipesteps-pipestep) | Wrapper for sklearn-compatible transformers.
[`ReduceStep`](#data-collection-pipesteps-reducestep) | Dimensionality reduction step.



##### Classes

(data-collection-pipesteps-fittednormalize)=
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
[`mean`](#data-collection-pipesteps-mean) | <code>[ndarray](#numpy.ndarray)</code> | For zscore: the mean. For minmax: the min value.
`std` | <code>[ndarray](#numpy.ndarray)</code> | For zscore: the standard deviation. For minmax: the range (max - min).
[`method`](#data-collection-pipesteps-method) | <code>[str](#str)</code> | The normalization method ('zscore' or 'minmax').

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#data-collection-pipesteps-inverse-transform) | Reverse normalization.
[`transform`](#data-collection-pipesteps-transform) | Apply normalization to data.



####### Attributes##

(data-collection-pipesteps-mean)=
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

(data-collection-pipesteps-fittedpipe)=
###### `FittedPipe`

```python
FittedPipe(transformer: Any) -> None
```

Fitted sklearn transformer wrapper.

Holds a fitted sklearn transformer and delegates transform calls to it.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`transformer`](#data-collection-pipesteps-transformer) | <code>[Any](#typing.Any)</code> | Fitted sklearn-compatible transformer.

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#data-collection-pipesteps-inverse-transform) | Apply inverse transform if supported.
[`transform`](#data-collection-pipesteps-transform) | Apply the fitted transformer.



####### Attributes##

(data-collection-pipesteps-transformer)=
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

(data-collection-pipesteps-fittedreduce)=
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
[`method`](#data-collection-pipesteps-method) | <code>[str](#str)</code> | The reduction method used.

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#data-collection-pipesteps-inverse-transform) | Reverse dimensionality reduction (reconstruct original space).
[`transform`](#data-collection-pipesteps-transform) | Apply dimensionality reduction.



####### Attributes##

(data-collection-pipesteps-method)=
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
[`fit`](#data-collection-pipesteps-fit) | Compute normalization parameters from data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`axis`](#data-collection-pipesteps-axis) | <code>[int](#int)</code> | 
[`invertible`](#data-collection-pipesteps-invertible) | <code>[bool](#bool)</code> | 
[`method`](#data-collection-pipesteps-method) | <code>[str](#str)</code> | 



####### Attributes##

(data-collection-pipesteps-axis)=
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
<code>[FittedNormalize](#nltools.data.collection.pipesteps.steps.FittedNormalize)</code> | Fitted transform that can be applied to new data.

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
[`fit`](#data-collection-pipesteps-fit) | Fit transformer to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#data-collection-pipesteps-invertible) | <code>[bool](#bool)</code> | Check if the transformer supports inverse_transform.
[`transformer`](#data-collection-pipesteps-transformer) | <code>[Any](#typing.Any)</code> | 



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
<code>[FittedPipe](#nltools.data.collection.pipesteps.steps.FittedPipe)</code> | Fitted transform wrapper.

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
[`fit`](#data-collection-pipesteps-fit) | Fit reduction model to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#data-collection-pipesteps-invertible) | <code>[bool](#bool)</code> | Check if the reduction method supports inverse transform.
[`method`](#data-collection-pipesteps-method) | <code>[str](#str)</code> | 
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
<code>[FittedReduce](#nltools.data.collection.pipesteps.steps.FittedReduce)</code> | Fitted transform that can be applied to new data.

