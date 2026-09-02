---
title: algorithms.alignment.procrustes
---

Data alignment — SRM, Procrustes, and state alignment.

**Methods:**

Name | Description
---- | -----------
[`align`](#algorithms-alignment-procrustes-align) | Align subject data into a common response model.
[`align_states`](#algorithms-alignment-procrustes-align-states) | Align state weight maps by minimizing pairwise distance between group states.
[`procrustes`](#algorithms-alignment-procrustes-procrustes) | Perform a Procrustes similarity analysis on two data sets.
[`procrustes_distance`](#algorithms-alignment-procrustes-procrustes-distance) | Test matrix similarity using Procrustes superposition.

## Methods

(algorithms-alignment-procrustes-align)=
### `align`

```python
align(data, method = 'deterministic_srm', n_features = None, axis = 0, *args, **kwargs)
```

Align subject data into a common response model.

This function is a convenience wrapper around `HyperAlignment` and `SRM` classes.

Can be used to hyperalign source data to target data using
Hyperalignment from Dartmouth (i.e., procrustes transformation; see
nltools.algorithms.procrustes) or Shared Response Model from Princeton (see
nltools.algorithms.srm). (see nltools.data.BrainData.align for aligning
a single Brain object to another). Common Model is shared response
model or centered target data. Transformed data can be back projected to
original data using Tranformation matrix. Inputs must be a list of BrainData
instances or numpy arrays (observations by features).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (list) A list of BrainData objects | *required*
`method` |  | (str) alignment method to use ['probabilistic_srm','deterministic_srm','procrustes'] | <code>'deterministic_srm'</code>
`n_features` |  | (int) number of features to align to common space. If None then will select number of voxels | <code>None</code>
`axis` |  | (int) axis to align on | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | A dictionary containing a list of transformed subject matrices, a     list of transformation matrices, the shared response matrix, and the     intersubject correlation of the shared responses.

**Examples:**

```python
# Hyperalign using procrustes transform
out = align(data, method='procrustes')

# Align using shared response model
out = align(data, method='probabilistic_srm', n_features=None)

# Project aligned data back into original data space
original_data = [
    np.dot(t.data, tm.T)
    for t, tm in zip(out['transformed'], out['transformation_matrix'])
]
```

(algorithms-alignment-procrustes-align-states)=
### `align_states`

```python
align_states(reference, target, *, metric = 'correlation', return_index = False, replace_zero_variance = False)
```

Align state weight maps by minimizing pairwise distance between group states.

This function uses the Hungarian algorithm for state alignment, which is
different from aligning multiple subjects' data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`reference` |  | (np.array) reference pattern x state matrix | *required*
`target` |  | (np.array) target pattern x state matrix to align to reference | *required*
`metric` |  | (str) distance metric to use | <code>'correlation'</code>
`return_index` |  | (bool) return index if True, return remapped data if False | <code>False</code>
`replace_zero_variance` |  | (bool) transform a vector with zero variance to random numbers from a uniform distribution. Useful when using correlation as a distance metric to avoid NaNs. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | If `return_index=False` (default), `target[:, remapping]` — the     target's columns reordered to match the reference, oriented pattern x     state (same shape as `target`). If `return_index=True`, the remapping     index array that reorders the target's state columns.

(algorithms-alignment-procrustes-procrustes)=
### `procrustes`

```python
procrustes(data1, data2)
```

Perform a Procrustes similarity analysis on two data sets.

For more comprehensive Procrustes-based alignment tasks, use
`HyperAlignment` and `align()` instead.

Each input matrix is a set of points or vectors (the rows of the matrix).
The dimension of the space is the number of columns of each matrix. Given
two identically sized matrices, procrustes standardizes both such that:
- $tr(AA^{T}) = 1$.
- Both sets of points are centered around the origin.
Procrustes then applies the optimal transform to the second
matrix (including scaling/dilation, rotations, and reflections) to minimize
$M^{2}=\sum(data1-data2)^{2}$, or the sum of the squares of the
pointwise differences between the two input datasets.
This function was not designed to handle datasets with different numbers of
datapoints (rows).  If two data sets have different dimensionality
(different number of columns), this function will add columns of zeros to
the smaller of the two.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` |  | Matrix whose n rows represent points in k (columns) space. `data1` is the reference data; after it is standardized, the data from `data2` will be transformed to fit the pattern in `data1` (must have >1 unique points). | *required*
`data2` |  | n rows of data in k space to be fit to `data1`. Must be the same shape `(numrows, numcols)` as `data1` (must have >1 unique points). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray), [float](#float), [ndarray](#numpy.ndarray), [float](#float)]</code> | `(mtx1, mtx2,     disparity, R, scale)` — `mtx1` is a standardized version of `data1`;     `mtx2` is the orientation of `data2` that best fits `data1` (centered,     but not necessarily $tr(AA^{T}) = 1$); `disparity` is $M^{2}$ as defined     above; `R` is the `(N, N)` matrix solution of the orthogonal Procrustes     problem, minimizing the Frobenius norm of `dot(data1, R) - data2` subject     to `dot(R.T, R) == I`; `scale` is the sum of the singular values of     `dot(data1.T, data2)`.

(algorithms-alignment-procrustes-procrustes-distance)=
### `procrustes_distance`

```python
procrustes_distance(mat1, mat2, *, n_permute = 5000, tail = 2, n_jobs = -1, random_state = None)
```

Test matrix similarity using Procrustes superposition.

Matrices need to match in size on their first dimension only, as the smaller
matrix on the second dimension will be padded with zeros. After aligning two
matrices using the Procrustes transformation, use the computed disparity
between them (sum of squared error of elements) as a similarity metric.
Shuffle the rows of one of the matrices and recompute the disparity to perform
inference (Peres-Neto & Jackson, 2001).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mat1` | <code>[ndarray](#ndarray)</code> | 2d numpy array; must have same number of rows as mat2 | *required*
`mat2` | <code>[ndarray](#ndarray)</code> | 1d or 2d numpy array; must have same number of rows as mat1 | *required*
`n_permute` | <code>[int](#int)</code> | number of permutation iterations to perform | <code>5000</code>
`tail` | <code>[int](#int) or [str](#str)</code> | `2` or `'two'` for two-tailed (default); `1` or `'one'` for one-tailed (similarity > chance) | <code>2</code>
`n_jobs` | <code>[int](#int)</code> | The number of CPUs to use to do permutation; default -1 (all) | <code>-1</code>
`random_state` | <code>int, np.random.RandomState, or None</code> | seed or generator for the permutation shuffling; default None | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | results with keys `similarity` (float in [0, 1]) and `p` (permuted p-value)
