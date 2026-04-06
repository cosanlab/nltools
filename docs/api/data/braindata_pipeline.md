## `pipeline`

BrainData pipeline and cross-validation result classes.

**Classes:**

Name | Description
---- | -----------
[`BrainDataCVResult`](#BrainDataCVResult) | Cross-validation results for BrainData pipelines.
[`BrainDataPipeline`](#BrainDataPipeline) | Pipeline specialized for BrainData with CV support.



### Classes

#### `BrainDataCVResult`

```python
BrainDataCVResult(fold_results: list, pipeline: list)
```

Cross-validation results for BrainData pipelines.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`fold_results`](#fold_results) |  | 
[`mean_score`](#mean_score) | <code>[float](#float)</code> | Mean score across folds.
[`pipeline`](#pipeline) |  | 
[`predictions`](#predictions) | <code>[ndarray](#numpy.ndarray)</code> | All predictions in original sample order.
[`scores`](#scores) | <code>[ndarray](#numpy.ndarray)</code> | Per-fold prediction scores as a numpy array.
[`std_score`](#std_score) | <code>[float](#float)</code> | Standard deviation of scores.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fold_results` | <code>[list](#list)</code> | List of dicts, one per fold, each containing 'score', 'predictions', 'train_idx', 'test_idx', and 'fitted_stack'. | *required*
`pipeline` | <code>[BrainDataPipeline](#nltools.data.braindata.pipeline.BrainDataPipeline)</code> | The pipeline that produced these results. | *required*

##### Methods

###### `normalize`

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

###### `pipe`

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

###### `predict`

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

###### `reduce`

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

