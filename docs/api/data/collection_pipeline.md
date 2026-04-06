## `nltools.data.collection.pipeline`

Pipeline classes for BrainCollection.

Provides BrainCollectionPipeline for fluent pipeline API with cross-validation,
BrainCollectionCVResult for storing CV results, and FittedBrainCollection for
chaining pool() after fit().

**Classes:**

Name | Description
---- | -----------
[`BrainCollectionCVResult`](#nltools.data.collection.pipeline.BrainCollectionCVResult) | Cross-validation results for BrainCollection pipelines.
[`BrainCollectionPipeline`](#nltools.data.collection.pipeline.BrainCollectionPipeline) | Pipeline for BrainCollection with multi-subject CV support.
[`FittedBrainCollection`](#nltools.data.collection.pipeline.FittedBrainCollection) | Wrapper for fitted BrainCollection enabling pool() chaining.



### Classes#### `nltools.data.collection.pipeline.BrainCollectionCVResult`

```python
BrainCollectionCVResult(fold_results: list, pipeline: BrainCollectionPipeline)
```

Cross-validation results for BrainCollection pipelines.

Contains fold-by-fold results from cross-validated prediction,
with convenience properties for accessing scores and predictions.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`fold_results`](#nltools.data.collection.pipeline.BrainCollectionCVResult.fold_results) |  | List of dictionaries with per-fold results.
[`pipeline`](#nltools.data.collection.pipeline.BrainCollectionCVResult.pipeline) |  | The pipeline that generated these results.
[`scores`](#nltools.data.collection.pipeline.BrainCollectionCVResult.scores) | <code>[ndarray](#numpy.ndarray)</code> | Per-fold prediction scores.
[`mean_score`](#nltools.data.collection.pipeline.BrainCollectionCVResult.mean_score) | <code>[float](#float)</code> | Mean score across all folds.
[`std_score`](#nltools.data.collection.pipeline.BrainCollectionCVResult.std_score) | <code>[float](#float)</code> | Standard deviation of scores.
[`n_folds`](#nltools.data.collection.pipeline.BrainCollectionCVResult.n_folds) | <code>[int](#int)</code> | Number of CV folds.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fold_results` | <code>[list](#list)</code> | List of fold result dictionaries. | *required*
`pipeline` | <code>[BrainCollectionPipeline](#nltools.data.collection.pipeline.BrainCollectionPipeline)</code> | The pipeline that generated these results. | *required*



##### Attributes###### `nltools.data.collection.pipeline.BrainCollectionCVResult.fold_results`

```python
fold_results = fold_results
```

###### `nltools.data.collection.pipeline.BrainCollectionCVResult.mean_score`

```python
mean_score: float
```

Mean score across folds.

###### `nltools.data.collection.pipeline.BrainCollectionCVResult.n_folds`

```python
n_folds: int
```

Number of cross-validation folds.

###### `nltools.data.collection.pipeline.BrainCollectionCVResult.pipeline`

```python
pipeline = pipeline
```

###### `nltools.data.collection.pipeline.BrainCollectionCVResult.scores`

```python
scores: np.ndarray
```

Per-fold prediction scores as a numpy array.

###### `nltools.data.collection.pipeline.BrainCollectionCVResult.std_score`

```python
std_score: float
```

Standard deviation of scores.

#### `nltools.data.collection.pipeline.BrainCollectionPipeline`

```python
BrainCollectionPipeline(brain_collection: 'BrainCollection', cv: 'BrainCollection' = None, groups: np.ndarray | None = None)
```

Pipeline for BrainCollection with multi-subject CV support.

Wraps BrainCollection to provide fluent pipeline API with LOSO
and run-based cross-validation.

This class enables method chaining for preprocessing and prediction
with proper cross-validation semantics for multi-subject neuroimaging
analyses.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`n_subjects`](#nltools.data.collection.pipeline.BrainCollectionPipeline.n_subjects) | <code>[int](#int)</code> | Number of subjects/images in the collection.
[`cv`](#nltools.data.collection.pipeline.BrainCollectionPipeline.cv) |  | The cross-validation scheme configuration.
[`n_steps`](#nltools.data.collection.pipeline.BrainCollectionPipeline.n_steps) | <code>[int](#int)</code> | Number of transform steps in the pipeline.

**Examples:**

```pycon
>>> # Leave-one-subject-out with preprocessing
>>> result = (bc
...     .cv(scheme='loso')
...     .normalize()
...     .reduce(n_components=50)
...     .predict(labels, algorithm='svm'))
>>> print(f"Mean accuracy: {result.mean_score:.2%}")
```

**Functions:**

Name | Description
---- | -----------
[`normalize`](#nltools.data.collection.pipeline.BrainCollectionPipeline.normalize) | Add normalization step.
[`pipe`](#nltools.data.collection.pipeline.BrainCollectionPipeline.pipe) | Add custom sklearn transformer.
[`predict`](#nltools.data.collection.pipeline.BrainCollectionPipeline.predict) | Execute pipeline with CV and return prediction results.
[`reduce`](#nltools.data.collection.pipeline.BrainCollectionPipeline.reduce) | Add dimensionality reduction step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_collection` | <code>'BrainCollection'</code> | BrainCollection to wrap. | *required*
`cv` |  | CVScheme configuration. | <code>None</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Group labels for CV splits. | <code>None</code>



##### Attributes###### `nltools.data.collection.pipeline.BrainCollectionPipeline.cv`

```python
cv
```

Cross-validation scheme.

###### `nltools.data.collection.pipeline.BrainCollectionPipeline.n_steps`

```python
n_steps: int
```

Number of transform steps.

###### `nltools.data.collection.pipeline.BrainCollectionPipeline.n_subjects`

```python
n_subjects: int
```

Number of subjects/images.



##### Functions###### `nltools.data.collection.pipeline.BrainCollectionPipeline.normalize`

```python
normalize(method: str = 'zscore', **kwargs: str) -> 'BrainCollectionPipeline'
```

Add normalization step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Normalization method ('zscore', 'minmax'). | <code>'zscore'</code>
`**kwargs` |  | Additional arguments for NormalizeStep. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollectionPipeline'</code> | New pipeline with normalization step added.

###### `nltools.data.collection.pipeline.BrainCollectionPipeline.pipe`

```python
pipe(transformer) -> 'BrainCollectionPipeline'
```

Add custom sklearn transformer.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`transformer` |  | sklearn-compatible transformer with fit/transform interface. | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollectionPipeline'</code> | New pipeline with custom step added.

###### `nltools.data.collection.pipeline.BrainCollectionPipeline.predict`

```python
predict(y, algorithm: str = 'ridge', **kwargs: str) -> 'BrainCollectionCVResult'
```

Execute pipeline with CV and return prediction results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` |  | Target variable. For LOSO, shape should be (n_subjects,). | *required*
`algorithm` | <code>[str](#str)</code> | Prediction algorithm ('ridge', 'svm', 'logistic', etc.) | <code>'ridge'</code>
`**kwargs` |  | Passed to model constructor. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollectionCVResult'</code> | Cross-validation results with scores and predictions.

###### `nltools.data.collection.pipeline.BrainCollectionPipeline.reduce`

```python
reduce(method: str = 'pca', n_components: int | None = None, **kwargs: int | None) -> 'BrainCollectionPipeline'
```

Add dimensionality reduction step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Reduction method ('pca', 'ica'). | <code>'pca'</code>
`n_components` | <code>[int](#int) \| None</code> | Number of components to keep. | <code>None</code>
`**kwargs` |  | Additional arguments for ReduceStep. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollectionPipeline'</code> | New pipeline with reduction step added.

#### `nltools.data.collection.pipeline.FittedBrainCollection`

```python
FittedBrainCollection(brain_collection: 'BrainCollection', fitted_results: 'BrainCollection | dict[str, BrainCollection]', model: str, condition_names: list[str] | None = None)
```

Wrapper for fitted BrainCollection enabling pool() chaining.

This class wraps the results of bc.fit() and provides the .pool()
method for aggregating across subjects.

The execution model:
- fit() executes immediately (eager)
- pool() aggregates the fitted parameters
- pool() returns PooledData for second-level analysis

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_collection` | <code>'BrainCollection'</code> | The original collection that was fitted. | *required*
`fitted_results` | <code>'BrainCollection \| dict[str, BrainCollection]'</code> | The fitted results. Can be a BrainCollection (betas or scores) or a dict mapping stat names to BrainCollections (e.g., {'betas': ..., 't': ...}). | *required*
`model` | <code>[str](#str)</code> | The model type that was fitted ('glm' or 'ridge'). | *required*
`condition_names` | <code>[list](#list)[[str](#str)] \| None</code> | Names of conditions/regressors from the design matrix. | <code>None</code>

Examples
--------
>>> fitted = bc.fit(model='glm', X=dm)
>>> pool = fitted.pool(param='beta')
>>> result = pool.fit(model='ttest', contrast='A-B')

**Functions:**

Name | Description
---- | -----------
[`pool`](#nltools.data.collection.pipeline.FittedBrainCollection.pool) | Pool fitted parameters across subjects.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`betas`](#nltools.data.collection.pipeline.FittedBrainCollection.betas) | <code>'BrainCollection'</code> | Convenience accessor for beta coefficients from a GLM fit.
[`n_subjects`](#nltools.data.collection.pipeline.FittedBrainCollection.n_subjects) | <code>[int](#int)</code> | Number of subjects in the fitted collection.
[`results`](#nltools.data.collection.pipeline.FittedBrainCollection.results) | <code>'BrainCollection \| dict[str, BrainCollection]'</code> | Access the fitted results directly.



##### Attributes###### `nltools.data.collection.pipeline.FittedBrainCollection.betas`

```python
betas: 'BrainCollection'
```

Convenience accessor for beta coefficients from a GLM fit.

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | Beta coefficients from GLM fit.

###### `nltools.data.collection.pipeline.FittedBrainCollection.n_subjects`

```python
n_subjects: int
```

Number of subjects in the fitted collection.

###### `nltools.data.collection.pipeline.FittedBrainCollection.results`

```python
results: 'BrainCollection | dict[str, BrainCollection]'
```

Access the fitted results directly.

Returns the underlying BrainCollection or dict of BrainCollections.
Use this for backward compatibility or when pool() is not needed.



##### Functions###### `nltools.data.collection.pipeline.FittedBrainCollection.pool`

```python
pool(param: str = 'beta', contrast: str | None = None, save: str | None = None, save_fitted: bool = False)
```

Pool fitted parameters across subjects.

Aggregates per-subject fitted results for group-level analysis.
Returns a PooledData object that can be passed to second-level
statistical tests.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`param` | <code>[str](#str)</code> | Parameter to pool. GLM options: 'beta', 't', 'r2', 'p', 'se', 'residual'. Ridge options: 'scores', 'weights'. Default is 'beta'. | <code>'beta'</code>
`contrast` | <code>[str](#str) \| None</code> | Apply contrast before pooling. Format: 'A-B' or 'A+B'. Requires condition_names to be available. | <code>None</code>
`save` | <code>[str](#str) \| None</code> | Path template to save per-subject results before pooling. Supports {subject}, {idx} placeholders. | <code>None</code>
`save_fitted` | <code>[bool](#bool)</code> | If True, save full fitted state for later repool(). | <code>False</code>

**Returns:**

Type | Description
---- | -----------
 | Pooled data ready for second-level analysis.

Examples
--------
>>> pool = bc.fit(model='glm', X=designs).pool(param='beta')
>>> result = pool.fit(model='ttest', contrast='face-house')

>>> # Pool t-statistics instead of betas
>>> pool = bc.fit(model='glm', X=dm, return_stats=['t']).pool(param='t')

