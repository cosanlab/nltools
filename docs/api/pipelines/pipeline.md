(pipelines-pipeline-base)=
## `base`

Low-level pipeline primitives for nltools.

Defines the transform protocols (`TransformStep`, `FittedTransform`, `CVScheme`,
`Terminal`) and `FittedStack`, the container that records fitted transforms so a
stack can be inverted. These primitives back `BrainCollectionPipeline`; the
standalone fluent `Pipeline` orchestrator was removed in v0.6.0 in favor of
`BrainCollection`'s native `.cv().standardize().reduce().predict()`.

**Classes:**

Name | Description
---- | -----------
`CVScheme` | Protocol for cross-validation schemes.
[`FittedStack`](#pipelines-pipeline-fittedstack) | Collection of fitted transforms for inverse transform support.
[`FittedTransform`](#pipelines-pipeline-fittedtransform) | Protocol for fitted transform objects.
[`Terminal`](#pipelines-pipeline-terminal) | Protocol for terminal operations that end a pipeline.
[`TransformStep`](#pipelines-pipeline-transformstep) | Protocol for pipeline transform steps.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`DataType` |  | 
`T` |  | 

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

