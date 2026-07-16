(pipelines-pipelines)=
## `pipelines`

Low-level pipeline primitives used by `BrainCollection`.

These are the building blocks that back `BrainCollectionPipeline`: transform
steps (`NormalizeStep`, `ReduceStep`, `PipeStep`, `AlignStep`), the fitted-stack
container (`FittedStack`), the pooled-data aggregator (`PooledData`), and the
transform protocols. The standalone fluent `Pipeline` / `MultiSubjectPipeline`
orchestration was removed in v0.6.0 — multi-subject CV now lives on
`BrainCollection` (`.cv().standardize().reduce().predict()`) and custom
single-dataset preprocessing uses `model=make_pipeline(...)` on `BrainData.predict`.

**Modules:**

Name | Description
---- | -----------
[`base`](#pipelines-base) | Low-level pipeline primitives for nltools.
[`cv`](#pipelines-cv) | Cross-validation scheme configuration for nltools pipelines.
[`pool`](#pipelines-pool) | Pool infrastructure for multi-subject aggregation.
[`steps`](#pipelines-steps) | Transform steps for nltools pipelines.

**Classes:**

Name | Description
---- | -----------
[`AlignStep`](#pipelines-alignstep) | Cross-subject alignment via SRM or HyperAlignment.
[`CVScheme`](#pipelines-cvscheme) | Protocol for cross-validation schemes.
[`FittedAlign`](#pipelines-fittedalign) | Fitted alignment model.
[`FittedStack`](#pipelines-fittedstack) | Collection of fitted transforms for inverse transform support.
[`FittedTransform`](#pipelines-fittedtransform) | Protocol for fitted transform objects.
[`NormalizeStep`](#pipelines-normalizestep) | Normalization transform step.
[`PipeStep`](#pipelines-pipestep) | Wrapper for sklearn-compatible transformers.
[`PooledData`](#pipelines-pooleddata) | Aggregated data from multiple subjects.
[`ReduceStep`](#pipelines-reducestep) | Dimensionality reduction step.
[`ResultDict`](#pipelines-resultdict) | Dictionary of StatResults, one per contrast.
[`StatResult`](#pipelines-statresult) | Result of statistical test.
[`Terminal`](#pipelines-terminal) | Protocol for terminal operations that end a pipeline.
[`TransformStep`](#pipelines-transformstep) | Protocol for pipeline transform steps.



### Classes

(pipelines-alignstep)=
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
[`fit`](#pipelines-fit) | Fit alignment model on list of subject data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#pipelines-invertible) | <code>[bool](#bool)</code> | Check if alignment is invertible.
`kwargs` |  | 
[`method`](#pipelines-method) |  | 
`n_features` |  | 
`n_iter` |  | 
`n_jobs` |  | 
`new_subject` |  | 
`parallel` |  | 
`scheme` |  | 

##### Methods

(pipelines-fit)=
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

(pipelines-cvscheme)=
#### `CVScheme`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for cross-validation schemes.

Compatible with scikit-learn CV splitters and custom implementations.

**Methods:**

Name | Description
---- | -----------
[`split`](#pipelines-split) | Generate train/test index splits.



##### Methods

(pipelines-split)=
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

(pipelines-fittedalign)=
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
[`method`](#pipelines-method) | <code>[str](#str)</code> | The alignment method used ('srm' or 'hyperalignment').
`new_subject_method` | <code>[str](#str)</code> | Method for aligning held-out subjects in LOSO CV.

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#pipelines-inverse-transform) | Reverse alignment (only for full-rank hyperalignment).
[`transform`](#pipelines-transform) | Transform subjects that were in training.
[`transform_new_subject`](#pipelines-transform-new-subject) | Align a new subject not in training (for LOSO).

##### Methods

(pipelines-inverse-transform)=
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

(pipelines-transform)=
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

(pipelines-transform-new-subject)=
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

(pipelines-fittedstack)=
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
[`steps`](#pipelines-steps) | <code>[list](#list)[[FittedTransform](#nltools.pipelines.base.FittedTransform)]</code> | Ordered list of fitted transforms.

Examples:
>>> stack = FittedStack()
>>> stack.append(fitted_pca)
>>> stack.append(fitted_normalize)
>>> original_space = stack.inverse_transform(predictions)

**Methods:**

Name | Description
---- | -----------
[`append`](#pipelines-append) | Add a fitted transform to the stack.
[`inverse_transform`](#pipelines-inverse-transform) | Apply inverse transforms in reverse order.

##### Methods

(pipelines-append)=
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

(pipelines-fittedtransform)=
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
[`inverse_transform`](#pipelines-inverse-transform) | Apply the inverse transformation to data.
[`transform`](#pipelines-transform) | Apply the learned transformation to data.



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

(pipelines-normalizestep)=
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
[`fit`](#pipelines-fit) | Compute normalization parameters from data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`axis`](#pipelines-axis) | <code>[int](#int)</code> | 
[`invertible`](#pipelines-invertible) | <code>[bool](#bool)</code> | 
[`method`](#pipelines-method) | <code>[str](#str)</code> | 

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

(pipelines-pipestep)=
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
[`fit`](#pipelines-fit) | Fit transformer to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#pipelines-invertible) | <code>[bool](#bool)</code> | Check if the transformer supports inverse_transform.
[`transformer`](#pipelines-transformer) | <code>[Any](#typing.Any)</code> | 

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

(pipelines-pooleddata)=
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
[`fit`](#pipelines-fit) | Fit second-level statistical model.
[`load`](#pipelines-load) | Load pooled data from disk.
[`repool`](#pipelines-repool) | Re-extract different parameter from saved fitted state.
[`save`](#pipelines-save) | Save pooled data to disk.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`condition_names`](#pipelines-condition-names) | <code>[list](#list)[[str](#str)] \| None</code> | 
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

(pipelines-load)=
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

(pipelines-repool)=
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

(pipelines-save)=
###### `save`

```python
save(path: str) -> None
```

Save pooled data to disk.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`path` | <code>[str](#str)</code> | Output path (directory or .npz file). | *required*

(pipelines-reducestep)=
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
[`fit`](#pipelines-fit) | Fit reduction model to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#pipelines-invertible) | <code>[bool](#bool)</code> | Check if the reduction method supports inverse transform.
[`method`](#pipelines-method) | <code>[str](#str)</code> | 
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

(pipelines-resultdict)=
#### `ResultDict`

Bases: <code>[dict](#dict)</code>

Dictionary of StatResults, one per contrast.

Provides convenience methods for batch operations.

**Methods:**

Name | Description
---- | -----------
[`threshold_all`](#pipelines-threshold-all) | Apply thresholding to all results.



##### Methods

(pipelines-threshold-all)=
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

(pipelines-statresult)=
#### `StatResult`

```python
StatResult(t_map: NDArray | None = None, f_map: NDArray | None = None, p_map: NDArray | None = None, contrast: str | None = None, df: int | None = None) -> None
```

Result of statistical test.

Holds statistical maps and provides thresholding utilities.

**Methods:**

Name | Description
---- | -----------
[`threshold`](#pipelines-threshold) | Apply multiple comparison correction.
[`to_nifti`](#pipelines-to-nifti) | Save as NIfTI file.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`contrast`](#pipelines-contrast) | <code>[str](#str) \| None</code> | 
`df` | <code>[int](#int) \| None</code> | 
`f_map` | <code>[NDArray](#numpy.typing.NDArray) \| None</code> | 
`p_map` | <code>[NDArray](#numpy.typing.NDArray) \| None</code> | 
`t_map` | <code>[NDArray](#numpy.typing.NDArray) \| None</code> | 

##### Methods

(pipelines-threshold)=
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

(pipelines-to-nifti)=
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

(pipelines-terminal)=
#### `Terminal`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for terminal operations that end a pipeline.

Terminals perform the final computation (e.g., prediction, similarity)
and produce results for each CV fold.

**Methods:**

Name | Description
---- | -----------
[`fit_evaluate`](#pipelines-fit-evaluate) | Fit on training data and evaluate on test data.



##### Methods

(pipelines-fit-evaluate)=
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

(pipelines-transformstep)=
#### `TransformStep`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for pipeline transform steps.

A transform step defines a transformation that can be fitted to data.
Steps are added to a Pipeline and executed sequentially during CV.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#pipelines-invertible) | <code>[bool](#bool)</code> | Whether this transform supports inverse_transform.

Examples:
>>> class MyStep:
...     invertible = True
...     def fit(self, data):
...         return MyFittedTransform(learned_params)

**Methods:**

Name | Description
---- | -----------
[`fit`](#pipelines-fit) | Fit the transform to data.

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

(pipelines-base)=
#### `base`

Low-level pipeline primitives for nltools.

Defines the transform protocols (`TransformStep`, `FittedTransform`, `CVScheme`,
`Terminal`) and `FittedStack`, the container that records fitted transforms so a
stack can be inverted. These primitives back `BrainCollectionPipeline`; the
standalone fluent `Pipeline` orchestrator was removed in v0.6.0 in favor of
`BrainCollection`'s native `.cv().standardize().reduce().predict()`.

**Classes:**

Name | Description
---- | -----------
[`CVScheme`](#pipelines-cvscheme) | Protocol for cross-validation schemes.
[`FittedStack`](#pipelines-fittedstack) | Collection of fitted transforms for inverse transform support.
[`FittedTransform`](#pipelines-fittedtransform) | Protocol for fitted transform objects.
[`Terminal`](#pipelines-terminal) | Protocol for terminal operations that end a pipeline.
[`TransformStep`](#pipelines-transformstep) | Protocol for pipeline transform steps.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`DataType` |  | 
`T` |  | 

##### Classes

###### `CVScheme`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for cross-validation schemes.

Compatible with scikit-learn CV splitters and custom implementations.

**Methods:**

Name | Description
---- | -----------
[`split`](#pipelines-split) | Generate train/test index splits.



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
[`steps`](#pipelines-steps) | <code>[list](#list)[[FittedTransform](#nltools.pipelines.base.FittedTransform)]</code> | Ordered list of fitted transforms.

Examples:
>>> stack = FittedStack()
>>> stack.append(fitted_pca)
>>> stack.append(fitted_normalize)
>>> original_space = stack.inverse_transform(predictions)

**Methods:**

Name | Description
---- | -----------
[`append`](#pipelines-append) | Add a fitted transform to the stack.
[`inverse_transform`](#pipelines-inverse-transform) | Apply inverse transforms in reverse order.



####### Attributes##

(pipelines-is-fully-invertible)=
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
[`inverse_transform`](#pipelines-inverse-transform) | Apply the inverse transformation to data.
[`transform`](#pipelines-transform) | Apply the learned transformation to data.



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

###### `Terminal`

Bases: <code>[Protocol](#typing.Protocol)</code>

Protocol for terminal operations that end a pipeline.

Terminals perform the final computation (e.g., prediction, similarity)
and produce results for each CV fold.

**Methods:**

Name | Description
---- | -----------
[`fit_evaluate`](#pipelines-fit-evaluate) | Fit on training data and evaluate on test data.



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
[`invertible`](#pipelines-invertible) | <code>[bool](#bool)</code> | Whether this transform supports inverse_transform.

Examples:
>>> class MyStep:
...     invertible = True
...     def fit(self, data):
...         return MyFittedTransform(learned_params)

**Methods:**

Name | Description
---- | -----------
[`fit`](#pipelines-fit) | Fit the transform to data.



####### Attributes##

(pipelines-invertible)=
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

(pipelines-cv)=
#### `cv`

Cross-validation scheme configuration for nltools pipelines.

This module provides a unified interface for configuring cross-validation
strategies used across nltools analysis pipelines.

**Classes:**

Name | Description
---- | -----------
[`CVScheme`](#pipelines-cvscheme) | Cross-validation scheme configuration.

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
[`n_splits`](#pipelines-n-splits) | Return number of splits.
[`split`](#pipelines-split) | Generate train/test indices for each fold.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`is_loro`](#pipelines-is-loro) | <code>[bool](#bool)</code> | Check if this is leave-one-run-out.
`is_loso` | <code>[bool](#bool)</code> | Check if this is leave-one-subject-out.
`k` | <code>[int](#int) \| None</code> | 
`n` | <code>[int](#int)</code> | 
`random_state` | <code>[int](#int) \| None</code> | 
`scheme` | <code>[CVSchemeType](#nltools.pipelines.cv.CVSchemeType)</code> | 
`split_by` | <code>[str](#str) \| None</code> | 



####### Attributes##

(pipelines-is-loro)=
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

(pipelines-n-splits)=
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

(pipelines-pool)=
#### `pool`

Pool infrastructure for multi-subject aggregation.

This module provides classes for pooling data across subjects and
enabling two-stage analyses (e.g., first-level GLM -> group t-test).

The pool() method serves as an execution boundary - everything before
it is executed lazily, and pool() triggers execution and aggregation.

**Classes:**

Name | Description
---- | -----------
[`PooledData`](#pipelines-pooleddata) | Aggregated data from multiple subjects.
[`ResultDict`](#pipelines-resultdict) | Dictionary of StatResults, one per contrast.
[`StatResult`](#pipelines-statresult) | Result of statistical test.



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
[`fit`](#pipelines-fit) | Fit second-level statistical model.
[`load`](#pipelines-load) | Load pooled data from disk.
[`repool`](#pipelines-repool) | Re-extract different parameter from saved fitted state.
[`save`](#pipelines-save) | Save pooled data to disk.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`condition_names`](#pipelines-condition-names) | <code>[list](#list)[[str](#str)] \| None</code> | 
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

(pipelines-condition-names)=
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
[`threshold_all`](#pipelines-threshold-all) | Apply thresholding to all results.



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
[`threshold`](#pipelines-threshold) | Apply multiple comparison correction.
[`to_nifti`](#pipelines-to-nifti) | Save as NIfTI file.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`contrast`](#pipelines-contrast) | <code>[str](#str) \| None</code> | 
`df` | <code>[int](#int) \| None</code> | 
`f_map` | <code>[NDArray](#numpy.typing.NDArray) \| None</code> | 
`p_map` | <code>[NDArray](#numpy.typing.NDArray) \| None</code> | 
`t_map` | <code>[NDArray](#numpy.typing.NDArray) \| None</code> | 



####### Attributes##

(pipelines-contrast)=
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

(pipelines-steps)=
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
[`AlignStep`](#pipelines-alignstep) | Cross-subject alignment via SRM or HyperAlignment.
[`FittedAlign`](#pipelines-fittedalign) | Fitted alignment model.
[`FittedNormalize`](#pipelines-fittednormalize) | Fitted normalization transform.
[`FittedPipe`](#pipelines-fittedpipe) | Fitted sklearn transformer wrapper.
[`FittedReduce`](#pipelines-fittedreduce) | Fitted dimensionality reduction transform.
[`NormalizeStep`](#pipelines-normalizestep) | Normalization transform step.
[`PipeStep`](#pipelines-pipestep) | Wrapper for sklearn-compatible transformers.
[`ReduceStep`](#pipelines-reducestep) | Dimensionality reduction step.



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
[`fit`](#pipelines-fit) | Fit alignment model on list of subject data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#pipelines-invertible) | <code>[bool](#bool)</code> | Check if alignment is invertible.
`kwargs` |  | 
[`method`](#pipelines-method) |  | 
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
[`method`](#pipelines-method) | <code>[str](#str)</code> | The alignment method used ('srm' or 'hyperalignment').
`new_subject_method` | <code>[str](#str)</code> | Method for aligning held-out subjects in LOSO CV.

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#pipelines-inverse-transform) | Reverse alignment (only for full-rank hyperalignment).
[`transform`](#pipelines-transform) | Transform subjects that were in training.
[`transform_new_subject`](#pipelines-transform-new-subject) | Align a new subject not in training (for LOSO).



####### Attributes##

(pipelines-method)=
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

(pipelines-fittednormalize)=
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
[`mean`](#pipelines-mean) | <code>[ndarray](#numpy.ndarray)</code> | For zscore: the mean. For minmax: the min value.
`std` | <code>[ndarray](#numpy.ndarray)</code> | For zscore: the standard deviation. For minmax: the range (max - min).
[`method`](#pipelines-method) | <code>[str](#str)</code> | The normalization method ('zscore' or 'minmax').

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#pipelines-inverse-transform) | Reverse normalization.
[`transform`](#pipelines-transform) | Apply normalization to data.



####### Attributes##

(pipelines-mean)=
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

(pipelines-fittedpipe)=
###### `FittedPipe`

```python
FittedPipe(transformer: Any) -> None
```

Fitted sklearn transformer wrapper.

Holds a fitted sklearn transformer and delegates transform calls to it.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`transformer`](#pipelines-transformer) | <code>[Any](#typing.Any)</code> | Fitted sklearn-compatible transformer.

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#pipelines-inverse-transform) | Apply inverse transform if supported.
[`transform`](#pipelines-transform) | Apply the fitted transformer.



####### Attributes##

(pipelines-transformer)=
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

(pipelines-fittedreduce)=
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
[`method`](#pipelines-method) | <code>[str](#str)</code> | The reduction method used.

**Methods:**

Name | Description
---- | -----------
[`inverse_transform`](#pipelines-inverse-transform) | Reverse dimensionality reduction (reconstruct original space).
[`transform`](#pipelines-transform) | Apply dimensionality reduction.



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
[`fit`](#pipelines-fit) | Compute normalization parameters from data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`axis`](#pipelines-axis) | <code>[int](#int)</code> | 
[`invertible`](#pipelines-invertible) | <code>[bool](#bool)</code> | 
[`method`](#pipelines-method) | <code>[str](#str)</code> | 



####### Attributes##

(pipelines-axis)=
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
[`fit`](#pipelines-fit) | Fit transformer to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#pipelines-invertible) | <code>[bool](#bool)</code> | Check if the transformer supports inverse_transform.
[`transformer`](#pipelines-transformer) | <code>[Any](#typing.Any)</code> | 



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
[`fit`](#pipelines-fit) | Fit reduction model to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`invertible`](#pipelines-invertible) | <code>[bool](#bool)</code> | Check if the reduction method supports inverse transform.
[`method`](#pipelines-method) | <code>[str](#str)</code> | 
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

