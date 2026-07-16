(pipelines-pipeline-base)=
## `base`

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
`CVScheme` | Protocol for cross-validation schemes.
[`FittedStack`](#pipelines-pipeline-fittedstack) | Collection of fitted transforms for inverse transform support.
[`FittedTransform`](#pipelines-pipeline-fittedtransform) | Protocol for fitted transform objects.
[`Pipeline`](#pipelines-pipeline-pipeline) | Base pipeline for chained transforms with optional cross-validation.
[`Terminal`](#pipelines-pipeline-terminal) | Protocol for terminal operations that end a pipeline.
[`TransformStep`](#pipelines-pipeline-transformstep) | Protocol for pipeline transform steps.

##### Methods

(pipelines-pipeline-split)=
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

(pipelines-pipeline-fittedstack)=
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
`steps` | <code>[list](#list)[[FittedTransform](#nltools.pipelines.base.FittedTransform)]</code> | Ordered list of fitted transforms.

Examples:
>>> stack = FittedStack()
>>> stack.append(fitted_pca)
>>> stack.append(fitted_normalize)
>>> original_space = stack.inverse_transform(predictions)

**Methods:**

Name | Description
---- | -----------
[`append`](#pipelines-pipeline-append) | Add a fitted transform to the stack.
[`inverse_transform`](#pipelines-pipeline-inverse-transform) | Apply inverse transforms in reverse order.

##### Methods

(pipelines-pipeline-append)=
###### `append`

```python
append(fitted_step: FittedTransform) -> None
```

Add a fitted transform to the stack.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fitted_step` | <code>[FittedTransform](#nltools.pipelines.base.FittedTransform)</code> | Fitted transform to append. | *required*

(pipelines-pipeline-inverse-transform)=
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

(pipelines-pipeline-fittedtransform)=
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
[`inverse_transform`](#pipelines-pipeline-inverse-transform) | Apply the inverse transformation to data.
[`transform`](#pipelines-pipeline-transform) | Apply the learned transformation to data.



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

(pipelines-pipeline-transform)=
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

(pipelines-pipeline-pipeline)=
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
[`copy`](#pipelines-pipeline-copy) | Create a shallow copy of the pipeline.
[`normalize`](#pipelines-pipeline-normalize) | Add a normalization step to the pipeline.
[`pipe`](#pipelines-pipeline-pipe) | Add a custom transformer to the pipeline.
[`predict`](#pipelines-pipeline-predict) | Execute pipeline with cross-validation and return prediction results.
[`reduce`](#pipelines-pipeline-reduce) | Add a dimensionality reduction step to the pipeline.

##### Methods

(pipelines-pipeline-copy)=
###### `copy`

```python
copy() -> Pipeline
```

Create a shallow copy of the pipeline.

**Returns:**

Type | Description
---- | -----------
<code>[Pipeline](#nltools.pipelines.base.Pipeline)</code> | New pipeline instance with same configuration.

(pipelines-pipeline-normalize)=
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

(pipelines-pipeline-pipe)=
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

(pipelines-pipeline-predict)=
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

(pipelines-pipeline-reduce)=
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

(pipelines-pipeline-terminal)=
#### `Terminal`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for terminal operations that end a pipeline.

Terminals perform the final computation (e.g., prediction, similarity)
and produce results for each CV fold.

**Methods:**

Name | Description
---- | -----------
[`fit_evaluate`](#pipelines-pipeline-fit-evaluate) | Fit on training data and evaluate on test data.



##### Methods

(pipelines-pipeline-fit-evaluate)=
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

(pipelines-pipeline-transformstep)=
#### `TransformStep`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for pipeline transform steps.

A transform step defines a transformation that can be fitted to data.
Steps are added to a Pipeline and executed sequentially during CV.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`invertible` | <code>[bool](#bool)</code> | Whether this transform supports inverse_transform.

Examples:
>>> class MyStep:
...     invertible = True
...     def fit(self, data):
...         return MyFittedTransform(learned_params)

**Methods:**

Name | Description
---- | -----------
[`fit`](#pipelines-pipeline-fit) | Fit the transform to data.

##### Methods

(pipelines-pipeline-fit)=
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

