(pipelines-cv-cv)=
## `cv`

Cross-validation scheme configuration for nltools pipelines.

This module provides a unified interface for configuring cross-validation
strategies used across nltools analysis pipelines.

**Classes:**

Name | Description
---- | -----------
`CVScheme` | Cross-validation scheme configuration.

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

