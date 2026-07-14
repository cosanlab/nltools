## `terminals`

Terminal operations for nltools pipelines.

Terminals are the final step in a pipeline that produce results.
They execute prediction, classification, or other evaluation tasks
within cross-validation folds.

**Classes:**

Name | Description
---- | -----------
[`ISCTerminal`](#iscterminal) | ISC terminal for multi-subject pipelines.
[`PredictTerminal`](#predictterminal) | Prediction/classification terminal for CV pipelines.
[`RSATerminal`](#rsaterminal) | RSA terminal for multi-subject pipelines.



### Classes

#### `ISCTerminal`

```python
ISCTerminal(method: str = 'pairwise', metric: str = 'median', n_permute: int = 5000, parallel: str = 'cpu', kwargs: dict[str, Any] = dict()) -> None
```

ISC terminal for multi-subject pipelines.

Computes inter-subject correlation across subjects in the pipeline.
Uses the ISC permutation test from nltools.algorithms.inference.isc.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ISC computation method: 'pairwise' (default) or 'leave-one-out'. | <code>'pairwise'</code>
`metric` | <code>[str](#str)</code> | Summary statistic: 'median' (default, robust) or 'mean' (Fisher z-transformed). | <code>'median'</code>
`n_permute` | <code>[int](#int)</code> | Number of bootstrap iterations for p-value computation. Default is 5000. | <code>5000</code>
`parallel` | <code>[str](#str)</code> | Parallelization method: 'cpu' (default), 'gpu', or None. | <code>'cpu'</code>
`kwargs` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Additional arguments passed to isc_permutation_test. | <code>[dict](#dict)()</code>

Examples:
>>> terminal = ISCTerminal(method='pairwise', n_permute=1000)
>>> result = terminal.fit_evaluate(data_list)
>>> print(f"ISC: {result.isc:.3f}, p: {result.p:.3f}")

**Methods:**

Name | Description
---- | -----------
[`fit_evaluate`](#fit-evaluate) | Compute ISC across subjects.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`kwargs` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | 
`method` | <code>[str](#str)</code> | 
`metric` | <code>[str](#str)</code> | 
`n_permute` | <code>[int](#int)</code> | 
`parallel` | <code>[str](#str)</code> | 

##### Methods

###### `fit_evaluate`

```python
fit_evaluate(data: list, **kwargs: list) -> ISCResult
```

Compute ISC across subjects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)</code> | List of subject data arrays. Each array should have shape (n_observations, n_features) where n_observations is the same across subjects (e.g., timepoints in fMRI). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ISCResult](#nltools.pipelines.results.ISCResult)</code> | Result containing ISC values, p-values, and confidence intervals.

#### `PredictTerminal`

```python
PredictTerminal(y: NDArray, algorithm: str = 'ridge', kwargs: dict[str, Any] = dict()) -> None
```

Prediction/classification terminal for CV pipelines.

Fits a prediction model on training data and evaluates on test data
within each CV fold.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` | <code>[NDArray](#numpy.typing.NDArray)</code> | Target variable to predict (labels or continuous values). | *required*
`algorithm` | <code>[str](#str)</code> | Prediction algorithm. Regression options: 'ridge' (default, L2), 'lasso' (L1), 'elastic' (L1+L2), 'svr' (kernel-based), 'rf' (random forest, auto-detected). Classification options: 'svm' (kernel-based), 'logistic' (linear), 'rf' (auto-detected for discrete y). | <code>'ridge'</code>
`kwargs` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Additional arguments passed to the sklearn model constructor. Common kwargs: ``class_weight='balanced'`` for imbalanced classification, ``C`` for regularization strength (svm, logistic), ``alpha`` for regularization strength (ridge, lasso, elastic). | <code>[dict](#dict)()</code>

**Examples:**

Basic classification:

```python
>>> terminal = PredictTerminal(y=labels, algorithm='svm', kwargs={'C': 1.0})
```
Balanced classification for imbalanced data:

```python
>>> terminal = PredictTerminal(
...     y=imbalanced_labels,
...     algorithm='svm',
...     kwargs={'class_weight': 'balanced'}
... )
```
Logistic regression with balanced classes:

```python
>>> terminal = PredictTerminal(
...     y=binary_labels,
...     algorithm='logistic',
...     kwargs={'class_weight': 'balanced', 'C': 0.1}
... )
```

**Methods:**

Name | Description
---- | -----------
[`fit_evaluate`](#fit-evaluate) | Fit model on training data and evaluate on test data.
[`with_y`](#with-y) | Create copy with different target variable.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`algorithm` | <code>[str](#str)</code> | 
`kwargs` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | 
`y` | <code>[NDArray](#numpy.typing.NDArray)</code> | 

##### Methods

###### `fit_evaluate`

```python
fit_evaluate(train_data: NDArray, test_data: NDArray, train_idx: NDArray[np.intp], test_idx: NDArray[np.intp], fitted_stack: Any) -> FoldResult
```

Fit model on training data and evaluate on test data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`train_data` | <code>[NDArray](#numpy.typing.NDArray)</code> | Transformed training features, shape (n_train, n_features). | *required*
`test_data` | <code>[NDArray](#numpy.typing.NDArray)</code> | Transformed test features, shape (n_test, n_features). | *required*
`train_idx` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Original indices of training samples. | *required*
`test_idx` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Original indices of test samples. | *required*
`fitted_stack` | <code>[Any](#typing.Any)</code> | Stack of fitted transforms for this fold. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[FoldResult](#nltools.pipelines.results.FoldResult)</code> | Result containing score, predictions, indices, and fitted stack.

###### `with_y`

```python
with_y(new_y: NDArray) -> PredictTerminal
```

Create copy with different target variable.

Useful for permutation testing.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`new_y` | <code>[NDArray](#numpy.typing.NDArray)</code> | New target variable. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[PredictTerminal](#nltools.pipelines.terminals.PredictTerminal)</code> | New terminal with updated y.

#### `RSATerminal`

```python
RSATerminal(model_rdm: NDArray, method: str = 'spearman', n_permute: int = 5000, kwargs: dict[str, Any] = dict()) -> None
```

RSA terminal for multi-subject pipelines.

Computes representational similarity analysis by correlating neural RDMs
with a model RDM.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model_rdm` | <code>[NDArray](#numpy.typing.NDArray)</code> | Model RDM to correlate with neural RDMs. Should be a symmetric matrix or upper triangle (condensed form). | *required*
`method` | <code>[str](#str)</code> | Correlation method: 'spearman' (default), 'pearson', or 'kendall'. | <code>'spearman'</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations for p-value computation. Default is 5000. | <code>5000</code>
`kwargs` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Additional arguments passed to correlation computation. | <code>[dict](#dict)()</code>

Examples:
>>> model = np.random.rand(10, 10)  # 10 conditions
>>> model = (model + model.T) / 2  # Make symmetric
>>> terminal = RSATerminal(model_rdm=model, method='spearman')
>>> result = terminal.fit_evaluate(neural_rdm)

**Methods:**

Name | Description
---- | -----------
[`fit_evaluate`](#fit-evaluate) | Compute RSA correlation between neural and model RDMs.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`kwargs` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | 
`method` | <code>[str](#str)</code> | 
`model_rdm` | <code>[NDArray](#numpy.typing.NDArray)</code> | 
`n_permute` | <code>[int](#int)</code> | 

##### Methods

###### `fit_evaluate`

```python
fit_evaluate(data: NDArray, **kwargs: NDArray) -> RSAResult
```

Compute RSA correlation between neural and model RDMs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[NDArray](#numpy.typing.NDArray)</code> | Neural data to compute RDM from, or pre-computed RDM. If 2D square, treated as RDM (upper triangle extracted). If 1D, treated as condensed RDM. If 2D non-square (n_conditions, n_features), RDM is computed using correlation distance. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[RSAResult](#nltools.pipelines.results.RSAResult)</code> | Result containing correlation coefficient and p-value.

