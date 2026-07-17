(data-collection-pipeline-pipeline)=
## `pipeline`

Pipeline classes for BrainCollection.

Provides BrainCollectionPipeline for a fluent pipeline API with
cross-validation. CV-aware ``predict()`` returns a ``BrainData`` with CV
attributes attached (``cv_scores``, ``cv_predictions``, ``mean_score``,
``std_score``, ``fold_results``, ``cv_pipeline``).

**Classes:**

Name | Description
---- | -----------
[`BrainCollectionPipeline`](#data-collection-pipeline-braincollectionpipeline) | Pipeline for BrainCollection with multi-subject CV support.



### Classes

(data-collection-pipeline-braincollectionpipeline)=
#### `BrainCollectionPipeline`

```python
BrainCollectionPipeline(brain_collection: BrainCollection, cv: BrainCollection = None, groups: np.ndarray | None = None)
```

Pipeline for BrainCollection with multi-subject CV support.

Wraps BrainCollection to provide fluent pipeline API with LOSO
and run-based cross-validation.

This class enables method chaining for preprocessing and prediction
with proper cross-validation semantics for multi-subject neuroimaging
analyses.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_collection` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | BrainCollection to wrap. | *required*
`cv` |  | CVScheme configuration. | <code>None</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Group labels for CV splits. | <code>None</code>



**Attributes:**

Name | Type | Description
---- | ---- | -----------
`n_subjects` | <code>[int](#int)</code> | Number of subjects/images in the collection.
`cv` |  | The cross-validation scheme configuration.
`n_steps` | <code>[int](#int)</code> | Number of transform steps in the pipeline.

**Methods:**

Name | Description
---- | -----------
[`pipe`](#data-collection-pipeline-pipe) | Add custom sklearn transformer.
[`predict`](#data-collection-pipeline-predict) | Execute pipeline with CV and return prediction results.
[`reduce`](#data-collection-pipeline-reduce) | Add dimensionality reduction step.
[`standardize`](#data-collection-pipeline-standardize) | Add standardization step.

**Examples:**

```pycon
>>> # Leave-one-subject-out with preprocessing
>>> result = (bc
...     .cv(method='loso')
...     .standardize()
...     .reduce(n_components=50)
...     .predict(labels, method='svm'))
>>> print(f"Mean accuracy: {result.mean_score:.2%}")
```

##### Methods

(data-collection-pipeline-pipe)=
###### `pipe`

```python
pipe(transformer) -> BrainCollectionPipeline
```

Add custom sklearn transformer.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`transformer` |  | sklearn-compatible transformer with fit/transform interface. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollectionPipeline](#nltools.data.collection.pipeline.BrainCollectionPipeline)</code> | New pipeline with custom step added.

(data-collection-pipeline-predict)=
###### `predict`

```python
predict(y, method: str = 'ridge', *, n_permute: int = 0, random_state: int = None, **kwargs: int)
```

Execute pipeline with CV and return prediction results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` |  | Target variable. For LOSO, shape should be (n_subjects,). | *required*
`method` | <code>[str](#str)</code> | Prediction algorithm ('ridge', 'svm', 'logistic', etc.) | <code>'ridge'</code>
`n_permute` | <code>[int](#int)</code> | If ``> 0``, also build a label-permutation null of the CV score — the classic MVPA permutation test. Each iteration shuffles ``y``, re-runs the *same* cross-validation, and records the mean score; the result gets ``permutation_scores`` (the null array) and ``permutation_pvalue`` attached. Default 0 (no null). | <code>0</code>
`random_state` |  | Seed for the label shuffling (permutation null only). | <code>None</code>
`**kwargs` |  | Passed to model constructor. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | ``BrainData`` carrying out-of-fold predictions plus CV attributes
 | (``cv_scores``, ``cv_predictions``, ``mean_score``, ``std_score``,
 | ``fold_results``, ``cv_pipeline``). When ``n_permute > 0`` it also
 | carries ``permutation_scores`` and ``permutation_pvalue``.

(data-collection-pipeline-reduce)=
###### `reduce`

```python
reduce(method: str = 'pca', n_components: int | None = None, **kwargs: int | None) -> BrainCollectionPipeline
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
<code>[BrainCollectionPipeline](#nltools.data.collection.pipeline.BrainCollectionPipeline)</code> | New pipeline with reduction step added.

(data-collection-pipeline-standardize)=
###### `standardize`

```python
standardize(method: str = 'zscore', **kwargs: str) -> BrainCollectionPipeline
```

Add standardization step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Standardization method ('zscore', 'minmax'). | <code>'zscore'</code>
`**kwargs` |  | Additional arguments for NormalizeStep. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollectionPipeline](#nltools.data.collection.pipeline.BrainCollectionPipeline)</code> | New pipeline with standardization step added.

