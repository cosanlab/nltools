## `nltools.pipelines.steps`

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



### Classes#### `nltools.pipelines.steps.AlignStep`

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



##### Attributes###### `nltools.pipelines.steps.AlignStep.invertible`

```python
invertible: bool
```

Check if alignment is invertible.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if method is hyperalignment (full-rank orthogonal transforms).

###### `nltools.pipelines.steps.AlignStep.kwargs`

```python
kwargs = kwargs
```

###### `nltools.pipelines.steps.AlignStep.method`

```python
method = method
```

###### `nltools.pipelines.steps.AlignStep.n_features`

```python
n_features = n_features
```

###### `nltools.pipelines.steps.AlignStep.n_iter`

```python
n_iter = n_iter
```

###### `nltools.pipelines.steps.AlignStep.n_jobs`

```python
n_jobs = n_jobs
```

###### `nltools.pipelines.steps.AlignStep.new_subject`

```python
new_subject = new_subject
```

###### `nltools.pipelines.steps.AlignStep.parallel`

```python
parallel = parallel
```

###### `nltools.pipelines.steps.AlignStep.scheme`

```python
scheme = scheme
```



##### Functions###### `nltools.pipelines.steps.AlignStep.fit`

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

#### `nltools.pipelines.steps.FittedAlign`

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



##### Attributes###### `nltools.pipelines.steps.FittedAlign.method`

```python
method: str
```

###### `nltools.pipelines.steps.FittedAlign.model`

```python
model: Any
```

###### `nltools.pipelines.steps.FittedAlign.new_subject_method`

```python
new_subject_method: str = 'procrustes'
```



##### Functions###### `nltools.pipelines.steps.FittedAlign.inverse_transform`

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

###### `nltools.pipelines.steps.FittedAlign.transform`

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

###### `nltools.pipelines.steps.FittedAlign.transform_new_subject`

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

#### `nltools.pipelines.steps.FittedNormalize`

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



##### Attributes###### `nltools.pipelines.steps.FittedNormalize.mean`

```python
mean: np.ndarray
```

###### `nltools.pipelines.steps.FittedNormalize.method`

```python
method: str
```

###### `nltools.pipelines.steps.FittedNormalize.std`

```python
std: np.ndarray
```



##### Functions###### `nltools.pipelines.steps.FittedNormalize.inverse_transform`

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

###### `nltools.pipelines.steps.FittedNormalize.transform`

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

#### `nltools.pipelines.steps.FittedPipe`

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



##### Attributes###### `nltools.pipelines.steps.FittedPipe.transformer`

```python
transformer: Any
```



##### Functions###### `nltools.pipelines.steps.FittedPipe.inverse_transform`

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

###### `nltools.pipelines.steps.FittedPipe.transform`

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

#### `nltools.pipelines.steps.FittedReduce`

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



##### Attributes###### `nltools.pipelines.steps.FittedReduce.method`

```python
method: str
```

###### `nltools.pipelines.steps.FittedReduce.model`

```python
model: Any
```



##### Functions###### `nltools.pipelines.steps.FittedReduce.inverse_transform`

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

###### `nltools.pipelines.steps.FittedReduce.transform`

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

#### `nltools.pipelines.steps.NormalizeStep`

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



##### Attributes###### `nltools.pipelines.steps.NormalizeStep.axis`

```python
axis: int = 0
```

###### `nltools.pipelines.steps.NormalizeStep.invertible`

```python
invertible: bool = True
```

###### `nltools.pipelines.steps.NormalizeStep.method`

```python
method: str = 'zscore'
```



##### Functions###### `nltools.pipelines.steps.NormalizeStep.fit`

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

#### `nltools.pipelines.steps.PipeStep`

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



##### Attributes###### `nltools.pipelines.steps.PipeStep.invertible`

```python
invertible: bool
```

Check if the transformer supports inverse_transform.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if transformer has inverse_transform method.

###### `nltools.pipelines.steps.PipeStep.transformer`

```python
transformer: Any = None
```



##### Functions###### `nltools.pipelines.steps.PipeStep.fit`

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

#### `nltools.pipelines.steps.ReduceStep`

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



##### Attributes###### `nltools.pipelines.steps.ReduceStep.invertible`

```python
invertible: bool
```

Check if the reduction method supports inverse transform.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if method is 'pca', False otherwise.

###### `nltools.pipelines.steps.ReduceStep.method`

```python
method: str = 'pca'
```

###### `nltools.pipelines.steps.ReduceStep.n_components`

```python
n_components: Optional[int] = None
```

###### `nltools.pipelines.steps.ReduceStep.random_state`

```python
random_state: Optional[int] = None
```



##### Functions###### `nltools.pipelines.steps.ReduceStep.fit`

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

