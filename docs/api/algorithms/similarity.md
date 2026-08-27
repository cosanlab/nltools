(algorithms-similarity-similarity)=
## `similarity`

Similarity metrics and correlation.

**Methods:**

Name | Description
---- | -----------
[`compute_multivariate_similarity`](#algorithms-similarity-compute-multivariate-similarity) | Compute multivariate similarity via OLS regression.
[`compute_similarity`](#algorithms-similarity-compute-similarity) | Compute similarity between two data arrays.
[`fisher_r_to_z`](#algorithms-similarity-fisher-r-to-z) | Use Fisher transformation to convert correlation to z score.
[`fisher_z_to_r`](#algorithms-similarity-fisher-z-to-r) | Convert Fisher z back to a correlation coefficient.
[`transform_pairwise`](#algorithms-similarity-transform-pairwise) | Transform data into pairs with balanced labels for ranking.



### Methods

(algorithms-similarity-compute-multivariate-similarity)=
#### `compute_multivariate_similarity`

```python
compute_multivariate_similarity(y, X, method = 'ols', tail = 2)
```

Compute multivariate similarity via OLS regression.

This is the functional core implementation for multivariate similarity computation.
Used by BrainData.multivariate_similarity() to delegate computation to the functional core.

Predicts spatial distribution of y from linear combination of X columns.
Computes OLS regression statistics including beta coefficients, t-statistics,
p-values, and residuals.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` | <code>[ndarray](#numpy.ndarray)</code> | Target data, shape (n_features,) - single image | *required*
`X` | <code>[ndarray](#numpy.ndarray)</code> | Predictor data, shape (n_features, n_predictors) where first column should be intercept (ones) if intercept is desired. If X does not include intercept, an intercept will be added automatically. | *required*
`method` | <code>[str](#str)</code> | Regression method (currently only 'ols' supported) | <code>'ols'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | Dictionary with keys: - 'beta': Regression coefficients including intercept, shape (n_predictors+1,) - 't': t-statistics, shape (n_predictors+1,) - 'p': p-values, shape (n_predictors+1,) - 'df': Degrees of freedom (int) - 'sigma': Residual standard deviation (float) - 'residual': Residuals, shape (n_features,)

**Examples:**

```pycon
>>> y = np.random.randn(100)
>>> X = np.random.randn(100, 5)
>>> result = compute_multivariate_similarity(y, X, method='ols')
>>> 'beta' in result
True
>>> result['beta'].shape
(6,)  # 5 predictors + intercept
```

(algorithms-similarity-compute-similarity)=
#### `compute_similarity`

```python
compute_similarity(data1, data2, metric = 'correlation')
```

Compute similarity between two data arrays.

This is the functional core implementation for similarity computation.
Used by BrainData.similarity() to delegate computation to the functional core.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | First data array, shape (n_samples1, n_features) | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Second data array, shape (n_samples2, n_features) | *required*
`metric` | <code>[str](#str)</code> | Type of similarity metric - 'correlation' or 'pearson': Pearson correlation - 'spearman' or 'rank_correlation': Spearman rank correlation - 'dot_product': Dot product - 'cosine': Cosine similarity | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
 | np.ndarray: Similarity matrix or vector - If data1.shape[0] == 1 and data2.shape[0] == 1: scalar - If data1.shape[0] == 1 or data2.shape[0] == 1: 1D array - Otherwise: 2D array shape (n_samples1, n_samples2)

**Examples:**

```pycon
>>> data1 = np.random.randn(10, 100)
>>> data2 = np.random.randn(5, 100)
>>> sim = compute_similarity(data1, data2, metric='correlation')
>>> sim.shape
(10, 5)
```

(algorithms-similarity-fisher-r-to-z)=
#### `fisher_r_to_z`

```python
fisher_r_to_z(r)
```

Use Fisher transformation to convert correlation to z score.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`r` |  | correlation coefficient(s) | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`z` |  | Fisher z-transformed correlation(s)

(algorithms-similarity-fisher-z-to-r)=
#### `fisher_z_to_r`

```python
fisher_z_to_r(z)
```

Convert Fisher z back to a correlation coefficient.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`z` |  | Fisher z-transformed value(s) | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`r` |  | correlation coefficient(s)

(algorithms-similarity-transform-pairwise)=
#### `transform_pairwise`

```python
transform_pairwise(X, y)
```

Transform data into pairs with balanced labels for ranking.

Transforms a n-class ranking problem into a two-class classification
problem. Subclasses implementing particular strategies for choosing
pairs should override this method.
In this method, all pairs are choosen, except for those that have the
same target value. The output is an array of balanced classes, i.e.
there are the same number of -1 as +1

Reference: "Large Margin Rank Boundaries for Ordinal Regression",
R. Herbrich, T. Graepel, K. Obermayer. Authors: Fabian Pedregosa
<fabian@fseoane.net> Alexandre Gramfort <alexandre.gramfort@inria.fr>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` |  | (np.array), shape (n_samples, n_features) The data | *required*
`y` |  | (np.array), shape (n_samples,) or (n_samples, 2) Target labels. If it's a 2D array, the second column represents the grouping of samples, i.e., samples with different groups will not be considered. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`X_trans` |  | (np.array), shape (k, n_features) Data as pairs, where k = n_samples * (n_samples-1)) / 2 if grouping values were not passed. If grouping variables exist, then returns values computed for each group.
`y_trans` |  | (np.array), shape (k,) Output class labels, where classes have values {-1, +1} If y was shape (n_samples, 2), then returns (k, 2) with groups on the second dimension.

