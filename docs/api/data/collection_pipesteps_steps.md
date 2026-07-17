(data-collection-pipesteps-steps-steps)=
## `steps`

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
[`FittedNormalize`](#data-collection-pipesteps-steps-fittednormalize) | Fitted normalization transform.
[`FittedPipe`](#data-collection-pipesteps-steps-fittedpipe) | Fitted sklearn transformer wrapper.
[`FittedReduce`](#data-collection-pipesteps-steps-fittedreduce) | Fitted dimensionality reduction transform.
[`NormalizeStep`](#data-collection-pipesteps-steps-normalizestep) | Normalization transform step.
[`PipeStep`](#data-collection-pipesteps-steps-pipestep) | Wrapper for sklearn-compatible transformers.
[`ReduceStep`](#data-collection-pipesteps-steps-reducestep) | Dimensionality reduction step.



### Classes

(data-collection-pipesteps-steps-fittednormalize)=
#### `FittedNormalize`

```python
FittedNormalize(mean: np.ndarray, std: np.ndarray, method: str) -> None
```

Fitted normalization transform.

Holds the learned parameters (mean, std or min, range) and applies
the transformation to new data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`mean` | <code>[ndarray](#numpy.ndarray)</code> | For zscore: the mean. For minmax: the min value.
`std` | <code>[ndarray](#numpy.ndarray)</code> | For zscore: the standard deviation. For minmax: the range (max - min).
`method` | <code>[str](#str)</code> | The normalization method ('zscore' or 'minmax').

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#data-collection-pipesteps-steps-inverse-transform) | Reverse normalization.
[`transform`](#data-collection-pipesteps-steps-transform) | Apply normalization to data.

##### Methods

(data-collection-pipesteps-steps-inverse-transform)=
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

(data-collection-pipesteps-steps-transform)=
###### `transform`

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

(data-collection-pipesteps-steps-fittedpipe)=
#### `FittedPipe`

```python
FittedPipe(transformer: Any) -> None
```

Fitted sklearn transformer wrapper.

Holds a fitted sklearn transformer and delegates transform calls to it.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`transformer` | <code>[Any](#typing.Any)</code> | Fitted sklearn-compatible transformer.

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#data-collection-pipesteps-steps-inverse-transform) | Apply inverse transform if supported.
[`transform`](#data-collection-pipesteps-steps-transform) | Apply the fitted transformer.

##### Methods

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

###### `transform`

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

(data-collection-pipesteps-steps-fittedreduce)=
#### `FittedReduce`

```python
FittedReduce(model: Any, method: str) -> None
```

Fitted dimensionality reduction transform.

Holds the fitted sklearn model and applies transformations.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`model` | <code>[Any](#typing.Any)</code> | Fitted sklearn decomposition model (PCA, FastICA, etc.).
`method` | <code>[str](#str)</code> | The reduction method used.

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#data-collection-pipesteps-steps-inverse-transform) | Reverse dimensionality reduction (reconstruct original space).
[`transform`](#data-collection-pipesteps-steps-transform) | Apply dimensionality reduction.

##### Methods

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

###### `transform`

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

(data-collection-pipesteps-steps-normalizestep)=
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

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`axis` | <code>[int](#int)</code> | 
`invertible` | <code>[bool](#bool)</code> | 
`method` | <code>[str](#str)</code> | 



**Methods:**

Name | Description
---- | -----------
[`fit`](#data-collection-pipesteps-steps-fit) | Compute normalization parameters from data.

##### Methods

(data-collection-pipesteps-steps-fit)=
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

(data-collection-pipesteps-steps-pipestep)=
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

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`invertible` | <code>[bool](#bool)</code> | Check if the transformer supports inverse_transform.
`transformer` | <code>[Any](#typing.Any)</code> | 



**Methods:**

Name | Description
---- | -----------
[`fit`](#data-collection-pipesteps-steps-fit) | Fit transformer to data.

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

(data-collection-pipesteps-steps-reducestep)=
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

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`invertible` | <code>[bool](#bool)</code> | Check if the reduction method supports inverse transform.
`method` | <code>[str](#str)</code> | 
`n_components` | <code>[int](#int) \| None</code> | 
`random_state` | <code>[int](#int) \| None</code> | 



**Methods:**

Name | Description
---- | -----------
[`fit`](#data-collection-pipesteps-steps-fit) | Fit reduction model to data.

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

