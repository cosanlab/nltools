(crossval-cross-validation)=
## `cross_validation`

Provide cross-validation data classes.

Scikit-learn compatible classes for performing various
types of cross-validation

**Classes:**

Name | Description
---- | -----------
[`KFoldStratified`](#crossval-kfoldstratified) | Stratify continuous targets across K-fold cross-validation.



### Classes

(crossval-kfoldstratified)=
#### `KFoldStratified`

```python
KFoldStratified(n_splits = 3, shuffle = False, random_state = None)
```

Bases: <code>[_BaseKFold](#sklearn.model_selection._split._BaseKFold)</code>

Stratify continuous targets across K-fold cross-validation.

Unlike the scikit-learn equivalent, this iterator stratifies continuous data.

Provides train/test indices to split data in train test sets. Split
dataset into k consecutive folds while ensuring that same subject is
held out within each fold.  Each fold is then used a validation set
once while the k - 1 remaining folds form the training set.
Extension of KFold from scikit-learn cross_validation model

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_splits` |  | int, default=3 Number of folds. Must be at least 2. | <code>3</code>
`shuffle` |  | boolean, optional Whether to shuffle the data before splitting into batches. | <code>False</code>
`random_state` |  | None, int or RandomState Pseudo-random number generator state used for random sampling. If None, use default numpy RNG for shuffling | <code>None</code>

**Methods:**

Name | Description
---- | -----------
[`split`](#crossval-split) | Generate indices to split data into training and test set.



##### Methods

(crossval-split)=
###### `split`

```python
split(X, y = None, groups = None)
```

Generate indices to split data into training and test set.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X ` |  | array-like, shape (n_samples, n_features) Training data, where n_samples is the number of samples and n_features is the number of features. Note that providing ``y`` is sufficient to generate the splits and hence ``np.zeros(n_samples)`` may be used as a placeholder for ``X`` instead of actual training data. | *required*
`y ` |  | array-like, shape (n_samples,) The target variable for supervised learning problems. Stratification is done based on the y labels. | *required*
`groups ` |  | (object) Always ignored, exists for compatibility. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`train` |  | (ndarray) The training set indices for that split.
`test` |  | (ndarray) The testing set indices for that split.

