(pipelines-cv-cv)=
## `cv`

Cross-validation scheme configuration for nltools pipelines.

This module provides a unified interface for configuring cross-validation
strategies used across nltools analysis pipelines.

**Classes:**

Name | Description
---- | -----------
[`CVScheme`](#pipelines-cv-cvscheme) | Cross-validation scheme configuration.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`CVSchemeType` |  | 

### Classes

(pipelines-cv-cvscheme)=
#### `CVScheme`

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
[`n_splits`](#pipelines-cv-n-splits) | Return number of splits.
[`split`](#pipelines-cv-split) | Generate train/test indices for each fold.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`is_loro` | <code>[bool](#bool)</code> | Check if this is leave-one-run-out.
`is_loso` | <code>[bool](#bool)</code> | Check if this is leave-one-subject-out.
`k` | <code>[int](#int) \| None</code> | 
`n` | <code>[int](#int)</code> | 
`random_state` | <code>[int](#int) \| None</code> | 
`scheme` | <code>[CVSchemeType](#nltools.pipelines.cv.CVSchemeType)</code> | 
`split_by` | <code>[str](#str) \| None</code> | 

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

