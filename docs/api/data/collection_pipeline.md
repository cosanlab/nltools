(data-collection-pipeline-pipeline)=
## `pipeline`

Pipeline classes for BrainCollection.

Provides BrainCollectionPipeline for fluent pipeline API with cross-validation,
and FittedBrainCollection for chaining pool() after fit(). CV-aware
``predict()`` returns a ``BrainData`` with CV attributes attached
(``cv_scores``, ``cv_predictions``, ``mean_score``, ``std_score``,
``fold_results``, ``cv_pipeline``).

**Classes:**

Name | Description
---- | -----------
[`BrainCollectionPipeline`](#data-collection-pipeline-braincollectionpipeline) | Pipeline for BrainCollection with multi-subject CV support.
[`FittedBrainCollection`](#data-collection-pipeline-fittedbraincollection) | Wrapper for fitted BrainCollection enabling pool() chaining.



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

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`n_subjects` | <code>[int](#int)</code> | Number of subjects/images in the collection.
`cv` |  | The cross-validation scheme configuration.
`n_steps` | <code>[int](#int)</code> | Number of transform steps in the pipeline.

**Examples:**

```pycon
>>> # Leave-one-subject-out with preprocessing
>>> result = (bc
...     .cv(scheme='loso')
...     .standardize()
...     .reduce(n_components=50)
...     .predict(labels, method='svm'))
>>> print(f"Mean accuracy: {result.mean_score:.2%}")
```

**Methods:**

Name | Description
---- | -----------
[`pipe`](#data-collection-pipeline-pipe) | Add custom sklearn transformer.
[`predict`](#data-collection-pipeline-predict) | Execute pipeline with CV and return prediction results.
[`reduce`](#data-collection-pipeline-reduce) | Add dimensionality reduction step.
[`standardize`](#data-collection-pipeline-standardize) | Add standardization step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_collection` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | BrainCollection to wrap. | *required*
`cv` |  | CVScheme configuration. | <code>None</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Group labels for CV splits. | <code>None</code>

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

(data-collection-pipeline-fittedbraincollection)=
#### `FittedBrainCollection`

```python
FittedBrainCollection(brain_collection: BrainCollection, fitted_results: BrainCollection | dict[str, BrainCollection], model: str, condition_names: list[str] | None = None)
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
`brain_collection` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | The original collection that was fitted. | *required*
`fitted_results` | <code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | The fitted results. Can be a BrainCollection (betas or scores) or a dict mapping stat names to BrainCollections (e.g., {'betas': ..., 't': ...}). | *required*
`model` | <code>[str](#str)</code> | The model type that was fitted ('glm' or 'ridge'). | *required*
`condition_names` | <code>[list](#list)[[str](#str)] \| None</code> | Names of conditions/regressors from the design matrix. | <code>None</code>

Examples
--------
>>> fitted = bc.fit(model='glm', X=dm)
>>> pool = fitted.pool(param='beta')
>>> result = pool.fit(model='ttest', contrast='A-B')

**Methods:**

Name | Description
---- | -----------
[`pool`](#data-collection-pipeline-pool) | Pool fitted parameters across subjects.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`betas` | <code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | Convenience accessor for beta coefficients from a GLM fit.
`n_subjects` | <code>[int](#int)</code> | Number of subjects in the fitted collection.
`results` | <code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | Access the fitted results directly.

##### Methods

(data-collection-pipeline-pool)=
###### `pool`

```python
pool(*, param: str = 'beta', contrast: str | None = None, save: str | None = None, save_fitted: bool = False)
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

