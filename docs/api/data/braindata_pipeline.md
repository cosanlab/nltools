## `nltools.data.braindata.pipeline`

BrainData pipeline and cross-validation result classes.

**Classes:**

Name | Description
---- | -----------
[`BrainDataCVResult`](#nltools.data.braindata.pipeline.BrainDataCVResult) | Cross-validation results for BrainData pipelines.
[`BrainDataPipeline`](#nltools.data.braindata.pipeline.BrainDataPipeline) | Pipeline specialized for BrainData with CV support.



### Classes#### `nltools.data.braindata.pipeline.BrainDataCVResult`

```python
BrainDataCVResult(fold_results: list, pipeline: list)
```

Cross-validation results for BrainData pipelines.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`fold_results`](#nltools.data.braindata.pipeline.BrainDataCVResult.fold_results) |  | 
[`mean_score`](#nltools.data.braindata.pipeline.BrainDataCVResult.mean_score) | <code>[float](#float)</code> | Mean score across folds.
[`pipeline`](#nltools.data.braindata.pipeline.BrainDataCVResult.pipeline) |  | 
[`predictions`](#nltools.data.braindata.pipeline.BrainDataCVResult.predictions) | <code>[ndarray](#numpy.ndarray)</code> | All predictions in original sample order.
[`scores`](#nltools.data.braindata.pipeline.BrainDataCVResult.scores) | <code>[ndarray](#numpy.ndarray)</code> | Per-fold prediction scores as a numpy array.
[`std_score`](#nltools.data.braindata.pipeline.BrainDataCVResult.std_score) | <code>[float](#float)</code> | Standard deviation of scores.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fold_results` | <code>[list](#list)</code> | List of dicts, one per fold, each containing 'score', 'predictions', 'train_idx', 'test_idx', and 'fitted_stack'. | *required*
`pipeline` | <code>[BrainDataPipeline](#nltools.data.braindata.pipeline.BrainDataPipeline)</code> | The pipeline that produced these results. | *required*



##### Attributes###### `nltools.data.braindata.pipeline.BrainDataCVResult.fold_results`

```python
fold_results = fold_results
```

###### `nltools.data.braindata.pipeline.BrainDataCVResult.mean_score`

```python
mean_score: float
```

Mean score across folds.

###### `nltools.data.braindata.pipeline.BrainDataCVResult.pipeline`

```python
pipeline = pipeline
```

###### `nltools.data.braindata.pipeline.BrainDataCVResult.predictions`

```python
predictions: np.ndarray
```

All predictions in original sample order.

###### `nltools.data.braindata.pipeline.BrainDataCVResult.scores`

```python
scores: np.ndarray
```

Per-fold prediction scores as a numpy array.

###### `nltools.data.braindata.pipeline.BrainDataCVResult.std_score`

```python
std_score: float
```

Standard deviation of scores.

#### `nltools.data.braindata.pipeline.BrainDataPipeline`

```python
BrainDataPipeline(brain_data, cv = None, groups = None)
```

Pipeline specialized for BrainData with CV support.

Wraps the base Pipeline to handle BrainData-specific operations
like splitting by samples and accessing the underlying data array.

**Functions:**

Name | Description
---- | -----------
[`normalize`](#nltools.data.braindata.pipeline.BrainDataPipeline.normalize) | Add normalization step.
[`pipe`](#nltools.data.braindata.pipeline.BrainDataPipeline.pipe) | Add custom sklearn transformer.
[`predict`](#nltools.data.braindata.pipeline.BrainDataPipeline.predict) | Execute pipeline with CV and return prediction results.
[`reduce`](#nltools.data.braindata.pipeline.BrainDataPipeline.reduce) | Add dimensionality reduction step.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`cv`](#nltools.data.braindata.pipeline.BrainDataPipeline.cv) |  | Cross-validation splitter for this pipeline.
[`data`](#nltools.data.braindata.pipeline.BrainDataPipeline.data) |  | Get underlying data array.
[`n_steps`](#nltools.data.braindata.pipeline.BrainDataPipeline.n_steps) | <code>[int](#int)</code> | Number of processing steps in this pipeline.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_data` |  | BrainData instance to build the pipeline on. | *required*
`cv` |  | Cross-validation splitter (e.g., CVScheme instance) or None. | <code>None</code>
`groups` | <code>[array](#array) - [like](#like)</code> | Group labels for CV splits (e.g., run IDs for leave-one-run-out). | <code>None</code>



##### Attributes###### `nltools.data.braindata.pipeline.BrainDataPipeline.cv`

```python
cv
```

Cross-validation splitter for this pipeline.

**Returns:**

Type | Description
---- | -----------
 | sklearn cross-validator or None: The cross-validation strategy
 | set for this pipeline, or None if not configured.

###### `nltools.data.braindata.pipeline.BrainDataPipeline.data`

```python
data
```

Get underlying data array.

###### `nltools.data.braindata.pipeline.BrainDataPipeline.n_steps`

```python
n_steps: int
```

Number of processing steps in this pipeline.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`int` | <code>[int](#int)</code> | The count of steps added to this pipeline.



##### Functions###### `nltools.data.braindata.pipeline.BrainDataPipeline.normalize`

```python
normalize(method: str = 'zscore', **kwargs: str) -> BrainDataPipeline
```

Add normalization step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Normalization method. Default: 'zscore'. | <code>'zscore'</code>
`**kwargs` |  | Additional arguments passed to NormalizeStep. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainDataPipeline` | <code>[BrainDataPipeline](#nltools.data.braindata.pipeline.BrainDataPipeline)</code> | New pipeline with the normalization step appended.

###### `nltools.data.braindata.pipeline.BrainDataPipeline.pipe`

```python
pipe(transformer) -> BrainDataPipeline
```

Add custom sklearn transformer.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`transformer` |  | An sklearn-compatible transformer with fit/transform methods. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainDataPipeline` | <code>[BrainDataPipeline](#nltools.data.braindata.pipeline.BrainDataPipeline)</code> | New pipeline with the custom step appended.

###### `nltools.data.braindata.pipeline.BrainDataPipeline.predict`

```python
predict(y, algorithm: str = 'ridge', **kwargs: str)
```

Execute pipeline with CV and return prediction results.

This is a terminal method that executes the full pipeline.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` |  | Target variable (labels or continuous values). | *required*
`algorithm` | <code>[str](#str)</code> | Prediction algorithm. Options: - 'ridge': Ridge regression (continuous targets) - 'lasso': Lasso regression (continuous targets) - 'svr': Support Vector Regression (continuous targets) - 'svm': Support Vector Classification (categorical targets) | <code>'ridge'</code>
`**kwargs` |  | Additional arguments passed to sklearn model constructor. For classification (svm), use class_weight='balanced' to handle imbalanced classes. See sklearn documentation for all options. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | BrainDataCVResult with scores, predictions, and fold information.

**Examples:**

Basic regression::

    result = brain.cv(5).predict(continuous_y, algorithm='ridge', alpha=1.0)

Classification with balanced classes::

    result = brain.cv(5).predict(labels, algorithm='svm', class_weight='balanced')

###### `nltools.data.braindata.pipeline.BrainDataPipeline.reduce`

```python
reduce(method: str = 'pca', n_components: int | None = None, **kwargs: int | None) -> BrainDataPipeline
```

Add dimensionality reduction step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Reduction method (e.g., 'pca'). Default: 'pca'. | <code>'pca'</code>
`n_components` | <code>[int](#int)</code> | Number of components to keep. | <code>None</code>
`**kwargs` |  | Additional arguments passed to ReduceStep. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainDataPipeline` | <code>[BrainDataPipeline](#nltools.data.braindata.pipeline.BrainDataPipeline)</code> | New pipeline with the reduction step appended.

