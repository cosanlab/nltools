## `nltools.pipelines.cv`

Cross-validation scheme configuration for nltools pipelines.

This module provides a unified interface for configuring cross-validation
strategies used across nltools analysis pipelines.

**Classes:**

Name | Description
---- | -----------
[`CVScheme`](#nltools.pipelines.cv.CVScheme) | Cross-validation scheme configuration.
[`NestedCVScheme`](#nltools.pipelines.cv.NestedCVScheme) | Nested cross-validation for hyperparameter tuning.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`CVSchemeType`](#nltools.pipelines.cv.CVSchemeType) |  | 



### Attributes#### `nltools.pipelines.cv.CVSchemeType`

```python
CVSchemeType = Literal['kfold', 'loso', 'loro', 'bootstrap', 'permutation']
```



### Classes#### `nltools.pipelines.cv.CVScheme`

```python
CVScheme(k: Optional[int] = None, scheme: CVSchemeType = 'kfold', split_by: Optional[str] = None, n: int = 1000, random_state: Optional[int] = None) -> None
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
`k` | <code>[Optional](#typing.Optional)[[int](#int)]</code> | Number of folds (for kfold scheme). Defaults to 5 if scheme is 'kfold'. | <code>None</code>
`scheme` | <code>[CVSchemeType](#nltools.pipelines.cv.CVSchemeType)</code> | CV scheme type. One of 'kfold', 'loso', 'loro', 'bootstrap', or 'permutation'. | <code>'kfold'</code>
`split_by` | <code>[Optional](#typing.Optional)[[str](#str)]</code> | Attribute to split by ('runs', 'subjects', 'sessions'). Used for documentation purposes with loso/loro schemes. | <code>None</code>
`n` | <code>[int](#int)</code> | Number of bootstrap iterations (for bootstrap scheme). Defaults to 1000. | <code>1000</code>
`random_state` | <code>[Optional](#typing.Optional)[[int](#int)]</code> | Random seed for reproducibility. If provided, sets the numpy random seed during initialization. | <code>None</code>

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

**Functions:**

Name | Description
---- | -----------
[`n_splits`](#nltools.pipelines.cv.CVScheme.n_splits) | Return number of splits.
[`split`](#nltools.pipelines.cv.CVScheme.split) | Generate train/test indices for each fold.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`is_loro`](#nltools.pipelines.cv.CVScheme.is_loro) | <code>[bool](#bool)</code> | Check if this is leave-one-run-out.
[`is_loso`](#nltools.pipelines.cv.CVScheme.is_loso) | <code>[bool](#bool)</code> | Check if this is leave-one-subject-out.
[`k`](#nltools.pipelines.cv.CVScheme.k) | <code>[Optional](#typing.Optional)[[int](#int)]</code> | 
[`n`](#nltools.pipelines.cv.CVScheme.n) | <code>[int](#int)</code> | 
[`random_state`](#nltools.pipelines.cv.CVScheme.random_state) | <code>[Optional](#typing.Optional)[[int](#int)]</code> | 
[`scheme`](#nltools.pipelines.cv.CVScheme.scheme) | <code>[CVSchemeType](#nltools.pipelines.cv.CVSchemeType)</code> | 
[`split_by`](#nltools.pipelines.cv.CVScheme.split_by) | <code>[Optional](#typing.Optional)[[str](#str)]</code> | 



##### Attributes###### `nltools.pipelines.cv.CVScheme.is_loro`

```python
is_loro: bool
```

Check if this is leave-one-run-out.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if scheme is 'loro', False otherwise.

###### `nltools.pipelines.cv.CVScheme.is_loso`

```python
is_loso: bool
```

Check if this is leave-one-subject-out.

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if scheme is 'loso', False otherwise.

###### `nltools.pipelines.cv.CVScheme.k`

```python
k: Optional[int] = None
```

###### `nltools.pipelines.cv.CVScheme.n`

```python
n: int = 1000
```

###### `nltools.pipelines.cv.CVScheme.random_state`

```python
random_state: Optional[int] = None
```

###### `nltools.pipelines.cv.CVScheme.scheme`

```python
scheme: CVSchemeType = 'kfold'
```

###### `nltools.pipelines.cv.CVScheme.split_by`

```python
split_by: Optional[str] = None
```



##### Functions###### `nltools.pipelines.cv.CVScheme.n_splits`

```python
n_splits(data: Any = None, groups: Optional[NDArray[np.intp]] = None) -> int
```

Return number of splits.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (unused for most schemes, kept for API consistency). | <code>None</code>
`groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for grouped CV. Required for 'loso' and 'loro'. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of splits/folds that will be generated.

###### `nltools.pipelines.cv.CVScheme.split`

```python
split(data: Any, groups: Optional[NDArray[np.intp]] = None) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]
```

Generate train/test indices for each fold.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (used for length). Can be any object with __len__ or a numpy array with shape attribute. | *required*
`groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for grouped CV (runs, subjects, etc.). Required for 'loso' and 'loro' schemes. | <code>None</code>

**Yields:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Tuple of (train_indices, test_indices) for each fold.

#### `nltools.pipelines.cv.NestedCVScheme`

```python
NestedCVScheme(outer: CVScheme, inner: CVScheme) -> None
```

Nested cross-validation for hyperparameter tuning.

Combines an outer CV loop for model evaluation with an inner CV loop
for model selection/hyperparameter tuning. This prevents information
leakage by ensuring the test data in the outer loop is never used
for any model selection decisions.

The outer loop evaluates final model performance on held-out data,
while the inner loop is used for hyperparameter tuning or model
selection within each outer training fold.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`outer` | <code>[CVScheme](#nltools.pipelines.cv.CVScheme)</code> | CVScheme for the outer evaluation loop. | *required*
`inner` | <code>[CVScheme](#nltools.pipelines.cv.CVScheme)</code> | CVScheme for the inner model selection loop. | *required*

**Examples:**

```pycon
>>> # LOSO outer with 3-fold inner for hyperparameter tuning
>>> cv = NestedCVScheme(
...     outer=CVScheme(scheme='loso'),
...     inner=CVScheme(scheme='kfold', k=3)
... )
>>> for outer_train, outer_test, inner_iter in cv.split(data, groups):
...     # Inner loop for hyperparameter tuning
...     for inner_train, inner_val in inner_iter:
...         # These are indices into outer_train
...         actual_train = outer_train[inner_train]
...         actual_val = outer_train[inner_val]
...         # Tune hyperparameters...
...     # Final evaluation on outer_test
```

```pycon
>>> # 5-fold outer with bootstrap inner
>>> cv = NestedCVScheme(
...     outer=CVScheme(scheme='kfold', k=5),
...     inner=CVScheme(scheme='bootstrap', n=100)
... )
```

**Functions:**

Name | Description
---- | -----------
[`n_inner_splits`](#nltools.pipelines.cv.NestedCVScheme.n_inner_splits) | Return number of inner splits per outer fold.
[`n_outer_splits`](#nltools.pipelines.cv.NestedCVScheme.n_outer_splits) | Return number of outer splits.
[`split`](#nltools.pipelines.cv.NestedCVScheme.split) | Generate nested cross-validation splits.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`inner`](#nltools.pipelines.cv.NestedCVScheme.inner) | <code>[CVScheme](#nltools.pipelines.cv.CVScheme)</code> | 
[`outer`](#nltools.pipelines.cv.NestedCVScheme.outer) | <code>[CVScheme](#nltools.pipelines.cv.CVScheme)</code> | 



##### Attributes###### `nltools.pipelines.cv.NestedCVScheme.inner`

```python
inner: CVScheme
```

###### `nltools.pipelines.cv.NestedCVScheme.outer`

```python
outer: CVScheme
```



##### Functions###### `nltools.pipelines.cv.NestedCVScheme.n_inner_splits`

```python
n_inner_splits(data: Any = None, groups: Optional[NDArray[np.intp]] = None) -> int
```

Return number of inner splits per outer fold.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (unused for most schemes). | <code>None</code>
`groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for inner loop. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of inner folds per outer fold.

###### `nltools.pipelines.cv.NestedCVScheme.n_outer_splits`

```python
n_outer_splits(data: Any = None, groups: Optional[NDArray[np.intp]] = None) -> int
```

Return number of outer splits.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (unused for most schemes). | <code>None</code>
`groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for outer loop. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of outer folds.

###### `nltools.pipelines.cv.NestedCVScheme.split`

```python
split(data: Any, groups: Optional[NDArray[np.intp]] = None, inner_groups: Optional[NDArray[np.intp]] = None) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp], Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]]]
```

Generate nested cross-validation splits.

For each outer fold, yields the outer train/test indices and an
iterator over inner train/validation splits. The inner splits are
indices into the outer training set, not the original data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (used for length). | *required*
`groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for outer loop (e.g., subject IDs). Required if outer.scheme is 'loso' or 'loro'. | <code>None</code>
`inner_groups` | <code>[Optional](#typing.Optional)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]</code> | Group labels for inner loop. If provided, these are indexed by outer_train to get inner groups. | <code>None</code>

**Yields:**

Type | Description
---- | -----------
<code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Tuple of (outer_train_idx, outer_test_idx, inner_splits_iterator)
<code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | where outer_train_idx and outer_test_idx are arrays of sample
<code>[Iterator](#typing.Iterator)[[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]]</code> | indices, and inner_splits_iterator yields (inner_train, inner_val)
<code>[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [Iterator](#typing.Iterator)[[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]]]</code> | tuples with indices relative to outer_train_idx.

<details class="example" open markdown="1">
<summary>Example</summary>

>>> cv = NestedCVScheme(
...     outer=CVScheme(scheme='kfold', k=5),
...     inner=CVScheme(scheme='kfold', k=3)
... )
>>> for outer_train, outer_test, inner_iter in cv.split(data):
...     for inner_train, inner_val in inner_iter:
...         # inner_train/inner_val are indices into outer_train
...         train_data = data[outer_train[inner_train]]
...         val_data = data[outer_train[inner_val]]

</details>

