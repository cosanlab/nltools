(pipelines-cv-cv)=
## `cv`

Cross-validation scheme configuration for nltools pipelines.

This module provides a unified interface for configuring cross-validation
strategies used across nltools analysis pipelines.

**Classes:**

Name | Description
---- | -----------
`CVScheme` | Cross-validation scheme configuration.
[`NestedCVScheme`](#pipelines-cv-nestedcvscheme) | Nested cross-validation for hyperparameter tuning.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`CVSchemeType` |  | 

##### Methods

(pipelines-cv-n-splits)=
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

(pipelines-cv-split)=
###### `split`

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

(pipelines-cv-nestedcvscheme)=
#### `NestedCVScheme`

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

**Methods:**

Name | Description
---- | -----------
[`n_inner_splits`](#pipelines-cv-n-inner-splits) | Return number of inner splits per outer fold.
[`n_outer_splits`](#pipelines-cv-n-outer-splits) | Return number of outer splits.
[`split`](#pipelines-cv-split) | Generate nested cross-validation splits.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`inner` | <code>[CVScheme](#nltools.pipelines.cv.CVScheme)</code> | 
`outer` | <code>[CVScheme](#nltools.pipelines.cv.CVScheme)</code> | 

##### Methods

(pipelines-cv-n-inner-splits)=
###### `n_inner_splits`

```python
n_inner_splits(data: Any = None, groups: NDArray[np.intp] | None = None) -> int
```

Return number of inner splits per outer fold.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (unused for most schemes). | <code>None</code>
`groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | Group labels for inner loop. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of inner folds per outer fold.

(pipelines-cv-n-outer-splits)=
###### `n_outer_splits`

```python
n_outer_splits(data: Any = None, groups: NDArray[np.intp] | None = None) -> int
```

Return number of outer splits.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (unused for most schemes). | <code>None</code>
`groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | Group labels for outer loop. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of outer folds.

###### `split`

```python
split(data: Any, groups: NDArray[np.intp] | None = None, inner_groups: NDArray[np.intp] | None = None) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp], Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]]]
```

Generate nested cross-validation splits.

For each outer fold, yields the outer train/test indices and an
iterator over inner train/validation splits. The inner splits are
indices into the outer training set, not the original data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[Any](#typing.Any)</code> | Data to split (used for length). | *required*
`groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | Group labels for outer loop (e.g., subject IDs). Required if outer.scheme is 'loso' or 'loro'. | <code>None</code>
`inner_groups` | <code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)] \| None</code> | Group labels for inner loop. If provided, these are indexed by outer_train to get inner groups. | <code>None</code>

**Yields:**

Type | Description
---- | -----------
<code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | Tuple of (outer_train_idx, outer_test_idx, inner_splits_iterator)
<code>[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]</code> | where outer_train_idx and outer_test_idx are arrays of sample
<code>[Iterator](#collections.abc.Iterator)[[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]]</code> | indices, and inner_splits_iterator yields (inner_train, inner_val)
<code>[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [Iterator](#collections.abc.Iterator)[[tuple](#tuple)[[NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)], [NDArray](#numpy.typing.NDArray)[[intp](#numpy.intp)]]]]</code> | tuples with indices relative to outer_train_idx.

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

