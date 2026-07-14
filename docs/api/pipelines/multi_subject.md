## `multi_subject`

Multi-subject pipeline for cross-subject analyses.

This module extends the base Pipeline to handle multi-subject data,
supporting leave-one-subject-out (LOSO) and run-based CV schemes.

**Classes:**

Name | Description
---- | -----------
[`MultiSubjectPipeline`](#multisubjectpipeline) | Pipeline for multi-subject neuroimaging analyses.



### Classes

#### `MultiSubjectPipeline`

```python
MultiSubjectPipeline(data: list[NDArray], cv: Any | None = None, groups: NDArray[np.intp] | None = None, steps: list[Any] = list(), _is_lazy: bool = False) -> None
```

Pipeline for multi-subject neuroimaging analyses.

Operates on a list of subject data arrays, supporting:
- LOSO (leave-one-subject-out): Train on N-1 subjects, test on 1
- Run-based CV: Split runs within each subject
- Pooling across subjects for group analyses

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)[[NDArray](#numpy.typing.NDArray)]</code> | List of subject data arrays, each shape (n_obs, n_voxels). | *required*
`cv` | <code>[Any](#typing.Any) \| None</code> | Cross-validation scheme configuration. | <code>None</code>
`groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | Group labels for CV splits (e.g., run labels). | <code>None</code>
`steps` | <code>[list](#list)[[Any](#typing.Any)]</code> | Transform steps to apply. | <code>[list](#list)()</code>

Examples:
>>> # LOSO CV
>>> pipeline = MultiSubjectPipeline(subject_data, cv=CVScheme(scheme='loso'))
>>> result = pipeline.normalize().predict(y, algorithm='svm')

>>> # Run-based CV across subjects
>>> pipeline = MultiSubjectPipeline(subject_data, cv=CVScheme(scheme='loro'), groups=runs)
>>> result = pipeline.predict(y)

**Methods:**

Name | Description
---- | -----------
[`align`](#align) | Add cross-subject alignment step to pipeline.
[`isc`](#isc) | Compute inter-subject correlation across subjects.
[`normalize`](#normalize) | Add normalization step (per-subject).
[`pipe`](#pipe) | Add custom sklearn transformer.
[`predict`](#predict) | Execute pipeline with CV and return prediction results.
[`reduce`](#reduce) | Add dimensionality reduction step.
[`rsa`](#rsa) | Compute representational similarity analysis.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`cv` | <code>[Any](#typing.Any) \| None</code> | 
`data` | <code>[list](#list)[[NDArray](#numpy.typing.NDArray)]</code> | 
`groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | 
`n_steps` | <code>[int](#int)</code> | Number of transform steps.
`n_subjects` | <code>[int](#int)</code> | Number of subjects in the multi-subject dataset.
`steps` | <code>[list](#list)[[Any](#typing.Any)]</code> | 

##### Methods

###### `align`

```python
align(method: str = 'srm', scheme: str = 'global', n_features: int | None = 50, new_subject: str = 'procrustes', **kwargs: str) -> MultiSubjectPipeline
```

Add cross-subject alignment step to pipeline.

Aligns multi-subject data using SRM or HyperAlignment before
downstream analyses like classification or pooling.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Alignment method: 'srm' (Shared Response Model, reduces dimensionality) or 'hyperalignment' (Procrustes-based, preserves dimensionality). Default is 'srm'. | <code>'srm'</code>
`scheme` | <code>[str](#str)</code> | Spatial scheme. Currently only 'global' is supported. 'searchlight' and 'piecewise' require LocalAlignment (nltools-boll). | <code>'global'</code>
`n_features` | <code>[int](#int) \| None</code> | Number of shared features for SRM. Ignored for hyperalignment. | <code>50</code>
`new_subject` | <code>[str](#str)</code> | Method for aligning held-out subjects in LOSO CV. Default is 'procrustes'. | <code>'procrustes'</code>
`**kwargs` |  | Additional arguments passed to alignment algorithm. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[MultiSubjectPipeline](#nltools.pipelines.multi_subject.MultiSubjectPipeline)</code> | New pipeline with alignment step added.

Examples:
>>> # SRM alignment before classification
>>> result = (
...     MultiSubjectPipeline(data=subjects, cv=CVScheme(scheme='loso'))
...     .align(method='srm', n_features=50)
...     .predict(y=labels, algorithm='svm')
... )

>>> # Hyperalignment before two-stage GLM
>>> result = (
...     bc.cv(scheme='loso')
...     .align(method='hyperalignment')
...     .fit(model='glm', X=designs)
...     .pool(param='beta')
...     .fit(model='ttest', contrast='A-B')
... )

###### `isc`

```python
isc(method: str = 'pairwise', metric: str = 'median', n_permute: int = 5000, parallel: str = 'cpu', **kwargs: str)
```

Compute inter-subject correlation across subjects.

Executes the pipeline and computes ISC using permutation testing.
Data is transformed through all pipeline steps before ISC computation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ISC computation method: 'pairwise' (average all pairwise correlations) or 'leave-one-out' (correlate each subject with mean of others). Default is 'pairwise'. | <code>'pairwise'</code>
`metric` | <code>[str](#str)</code> | Summary statistic: 'median' (robust, default) or 'mean' (Fisher z-transformed). | <code>'median'</code>
`n_permute` | <code>[int](#int)</code> | Number of bootstrap iterations for p-value computation. Default is 5000. | <code>5000</code>
`parallel` | <code>[str](#str)</code> | Parallelization method: 'cpu', 'gpu', or None. Default is 'cpu'. | <code>'cpu'</code>
`**kwargs` |  | Additional arguments passed to ISCTerminal. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | Result containing ISC values, p-values, and confidence intervals.

Examples:
>>> result = (
...     MultiSubjectPipeline(data=subjects, cv=None)
...     .normalize()
...     .isc(method='pairwise', n_permute=1000)
... )
>>> print(f"ISC: {result.isc:.3f}, p: {result.p:.3f}")

###### `normalize`

```python
normalize(method: str = 'zscore', **kwargs: str) -> MultiSubjectPipeline
```

Add normalization step (per-subject).

###### `pipe`

```python
pipe(transformer) -> MultiSubjectPipeline
```

Add custom sklearn transformer.

###### `predict`

```python
predict(y, algorithm: str = 'ridge', **kwargs: str)
```

Execute pipeline with CV and return prediction results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` |  | Target variable. For LOSO, should be (n_subjects,). For run-based CV, should match pooled observations. | *required*
`algorithm` | <code>[str](#str)</code> | Prediction algorithm: 'ridge', 'lasso', 'elastic', 'svr' for regression; 'svm', 'logistic', 'rf' for classification. | <code>'ridge'</code>
`**kwargs` |  | Additional arguments passed to sklearn model constructor. For classification (svm, logistic), use ``class_weight='balanced'`` to handle imbalanced classes. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | Cross-validation results.

**Examples:**

Basic regression with LOSO CV:

```python
result = pipeline.cv('loso').predict(subject_labels, algorithm='ridge')
```
Classification with balanced classes:

```python
result = pipeline.cv('loso').predict(
    group_labels, algorithm='svm', class_weight='balanced'
)
```
Logistic regression with regularization:

```python
result = pipeline.cv('loso').predict(
    binary_labels, algorithm='logistic', C=0.1, class_weight='balanced'
)
```

###### `reduce`

```python
reduce(method: str = 'pca', n_components: int | None = None, **kwargs: int | None) -> MultiSubjectPipeline
```

Add dimensionality reduction step.

###### `rsa`

```python
rsa(model_rdm: NDArray, method: str = 'spearman', n_permute: int = 5000, **kwargs: int)
```

Compute representational similarity analysis.

Executes the pipeline and computes RSA correlation between neural
and model RDMs using permutation testing.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model_rdm` | <code>[NDArray](#numpy.typing.NDArray)</code> | Model RDM to correlate with neural RDMs. Should be symmetric matrix or upper triangle (condensed form). | *required*
`method` | <code>[str](#str)</code> | Correlation method: 'spearman' (default), 'pearson', or 'kendall'. | <code>'spearman'</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations for p-value computation. Default is 5000. | <code>5000</code>
`**kwargs` |  | Additional arguments passed to RSATerminal. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | Result containing correlation coefficient and p-value.

Examples:
>>> model = np.corrcoef(conditions)  # Theoretical model
>>> result = (
...     MultiSubjectPipeline(data=subjects, cv=None)
...     .normalize()
...     .rsa(model_rdm=model, method='spearman')
... )
>>> print(f"r = {result.correlation:.3f}, p = {result.p_value:.3f}")

