(data-collection-pipesteps-base-base)=
## `base`

Low-level pipeline primitives for nltools.

Defines the transform protocols (`TransformStep`, `FittedTransform`) and
`FittedStack`, the container that records fitted transforms so a stack can be
inverted. These primitives back `BrainCollectionPipeline`; the standalone fluent
`Pipeline` orchestrator was removed in v0.6.0 in favor of `BrainCollection`'s
native `.cv().standardize().reduce().predict()`.

**Classes:**

Name | Description
---- | -----------
[`FittedStack`](#data-collection-pipesteps-base-fittedstack) | Collection of fitted transforms for inverse transform support.
[`FittedTransform`](#data-collection-pipesteps-base-fittedtransform) | Protocol for fitted transform objects.
[`TransformStep`](#data-collection-pipesteps-base-transformstep) | Protocol for pipeline transform steps.



### Classes

(data-collection-pipesteps-base-fittedstack)=
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
`steps` | <code>[list](#list)[[FittedTransform](#nltools.data.collection.pipesteps.base.FittedTransform)]</code> | Ordered list of fitted transforms.

Examples:
>>> stack = FittedStack()
>>> stack.append(fitted_pca)
>>> stack.append(fitted_normalize)
>>> original_space = stack.inverse_transform(predictions)

**Methods:**

Name | Description
---- | -----------
[`append`](#data-collection-pipesteps-base-append) | Add a fitted transform to the stack.
[`inverse_transform`](#data-collection-pipesteps-base-inverse-transform) | Apply inverse transforms in reverse order.

##### Methods

(data-collection-pipesteps-base-append)=
###### `append`

```python
append(fitted_step: FittedTransform) -> None
```

Add a fitted transform to the stack.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fitted_step` | <code>[FittedTransform](#nltools.data.collection.pipesteps.base.FittedTransform)</code> | Fitted transform to append. | *required*

(data-collection-pipesteps-base-inverse-transform)=
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

(data-collection-pipesteps-base-fittedtransform)=
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
[`inverse_transform`](#data-collection-pipesteps-base-inverse-transform) | Apply the inverse transformation to data.
[`transform`](#data-collection-pipesteps-base-transform) | Apply the learned transformation to data.



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

(data-collection-pipesteps-base-transform)=
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

(data-collection-pipesteps-base-transformstep)=
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
[`fit`](#data-collection-pipesteps-base-fit) | Fit the transform to data.

##### Methods

(data-collection-pipesteps-base-fit)=
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

