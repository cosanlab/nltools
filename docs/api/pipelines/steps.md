(pipelines-steps-steps)=
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
[`AlignStep`](#pipelines-steps-alignstep) | Cross-subject alignment via SRM or HyperAlignment.
[`FittedAlign`](#pipelines-steps-fittedalign) | Fitted alignment model.
[`FittedNormalize`](#pipelines-steps-fittednormalize) | Fitted normalization transform.
[`FittedPipe`](#pipelines-steps-fittedpipe) | Fitted sklearn transformer wrapper.
[`FittedReduce`](#pipelines-steps-fittedreduce) | Fitted dimensionality reduction transform.
[`NormalizeStep`](#pipelines-steps-normalizestep) | Normalization transform step.
[`PipeStep`](#pipelines-steps-pipestep) | Wrapper for sklearn-compatible transformers.
[`ReduceStep`](#pipelines-steps-reducestep) | Dimensionality reduction step.



### Classes

(pipelines-steps-alignstep)=
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
[`fit`](#pipelines-steps-fit) | Fit alignment model on list of subject data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`invertible` | <code>[bool](#bool)</code> | Check if alignment is invertible.
`kwargs` |  | 
`method` |  | 
`n_features` |  | 
`n_iter` |  | 
`n_jobs` |  | 
`new_subject` |  | 
`parallel` |  | 
`scheme` |  | 

##### Methods

(pipelines-steps-fit)=
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

(pipelines-steps-fittedalign)=
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
`method` | <code>[str](#str)</code> | The alignment method used ('srm' or 'hyperalignment').
`new_subject_method` | <code>[str](#str)</code> | Method for aligning held-out subjects in LOSO CV.

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#pipelines-steps-inverse-transform) | Reverse alignment (only for full-rank hyperalignment).
[`transform`](#pipelines-steps-transform) | Transform subjects that were in training.
[`transform_new_subject`](#pipelines-steps-transform-new-subject) | Align a new subject not in training (for LOSO).

##### Methods

(pipelines-steps-inverse-transform)=
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

(pipelines-steps-transform)=
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

(pipelines-steps-transform-new-subject)=
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

(pipelines-steps-fittednormalize)=
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
[`inverse_transform`](#pipelines-steps-inverse-transform) | Reverse normalization.
[`transform`](#pipelines-steps-transform) | Apply normalization to data.

##### Methods

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

(pipelines-steps-fittedpipe)=
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
[`inverse_transform`](#pipelines-steps-inverse-transform) | Apply inverse transform if supported.
[`transform`](#pipelines-steps-transform) | Apply the fitted transformer.

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

(pipelines-steps-fittedreduce)=
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
[`inverse_transform`](#pipelines-steps-inverse-transform) | Reverse dimensionality reduction (reconstruct original space).
[`transform`](#pipelines-steps-transform) | Apply dimensionality reduction.

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

(pipelines-steps-normalizestep)=
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
[`fit`](#pipelines-steps-fit) | Compute normalization parameters from data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`axis` | <code>[int](#int)</code> | 
`invertible` | <code>[bool](#bool)</code> | 
`method` | <code>[str](#str)</code> | 

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

(pipelines-steps-pipestep)=
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
[`fit`](#pipelines-steps-fit) | Fit transformer to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`invertible` | <code>[bool](#bool)</code> | Check if the transformer supports inverse_transform.
`transformer` | <code>[Any](#typing.Any)</code> | 

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

(pipelines-steps-reducestep)=
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
[`fit`](#pipelines-steps-fit) | Fit reduction model to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`invertible` | <code>[bool](#bool)</code> | Check if the reduction method supports inverse transform.
`method` | <code>[str](#str)</code> | 
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

