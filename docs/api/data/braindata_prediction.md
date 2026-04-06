## `nltools.data.braindata.prediction`

BrainData prediction functions.

Standalone functions extracted from BrainData class methods for timeseries
prediction (encoding models) and MVPA decoding (pattern classification).

**Functions:**

Name | Description
---- | -----------
[`mvpa_roi`](#nltools.data.braindata.prediction.mvpa_roi) | ROI-based MVPA - accuracy per ROI.
[`mvpa_searchlight`](#nltools.data.braindata.prediction.mvpa_searchlight) | Searchlight MVPA - accuracy per voxel neighborhood.
[`mvpa_whole_brain`](#nltools.data.braindata.prediction.mvpa_whole_brain) | Whole-brain MVPA - single accuracy across all voxels.
[`mvpa_whole_brain_pipeline`](#nltools.data.braindata.prediction.mvpa_whole_brain_pipeline) | Whole-brain MVPA using Pipeline infrastructure.
[`predict`](#nltools.data.braindata.prediction.predict) | Generate predictions using fitted model OR classify patterns (MVPA).
[`predict_mvpa`](#nltools.data.braindata.prediction.predict_mvpa) | Perform MVPA decoding using cross-validation.
[`predict_timeseries`](#nltools.data.braindata.prediction.predict_timeseries) | Generate timeseries predictions using fitted model.
[`resolve_estimator`](#nltools.data.braindata.prediction.resolve_estimator) | Resolve string shortcut to sklearn estimator.



### Functions#### `nltools.data.braindata.prediction.mvpa_roi`

```python
mvpa_roi(bd, X, y, pipe, cv, groups, scoring, roi_mask, n_jobs, show_progress)
```

ROI-based MVPA - accuracy per ROI.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`X` |  | Feature matrix, shape (n_samples, n_voxels). | *required*
`y` |  | Labels, shape (n_samples,). | *required*
`pipe` |  | sklearn pipeline or estimator. | *required*
`cv` |  | Cross-validation splitter. | *required*
`groups` |  | Group labels for CV. | *required*
`scoring` |  | Scoring metric string. | *required*
`roi_mask` |  | Atlas/parcellation image or path. | *required*
`n_jobs` |  | Number of parallel jobs. | *required*
`show_progress` |  | Whether to show progress bar. | *required*

**Returns:**

Type | Description
---- | -----------
 | np.ndarray of accuracy values per ROI.

#### `nltools.data.braindata.prediction.mvpa_searchlight`

```python
mvpa_searchlight(bd, X, y, pipe, cv, groups, scoring, radius, n_jobs, show_progress)
```

Searchlight MVPA - accuracy per voxel neighborhood.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`X` |  | Feature matrix, shape (n_samples, n_voxels). | *required*
`y` |  | Labels, shape (n_samples,). | *required*
`pipe` |  | sklearn pipeline or estimator. | *required*
`cv` |  | Cross-validation splitter. | *required*
`groups` |  | Group labels for CV. | *required*
`scoring` |  | Scoring metric string. | *required*
`radius` |  | Searchlight radius in mm. | *required*
`n_jobs` |  | Number of parallel jobs. | *required*
`show_progress` |  | Whether to show progress bar. | *required*

**Returns:**

Type | Description
---- | -----------
 | np.ndarray of accuracy values per voxel.

#### `nltools.data.braindata.prediction.mvpa_whole_brain`

```python
mvpa_whole_brain(bd, X, y, pipe, cv, groups, scoring)
```

Whole-brain MVPA - single accuracy across all voxels.

Legacy implementation using sklearn cross_val_score directly.
Kept for searchlight/ROI methods that still use sklearn pipelines.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance (unused, kept for API consistency). | *required*
`X` |  | Feature matrix, shape (n_samples, n_voxels). | *required*
`y` |  | Labels, shape (n_samples,). | *required*
`pipe` |  | sklearn pipeline or estimator. | *required*
`cv` |  | Cross-validation splitter. | *required*
`groups` |  | Group labels for CV. | *required*
`scoring` |  | Scoring metric string. | *required*

**Returns:**

Type | Description
---- | -----------
 | np.ndarray with single mean accuracy value.

#### `nltools.data.braindata.prediction.mvpa_whole_brain_pipeline`

```python
mvpa_whole_brain_pipeline(bd, y, estimator, cv, groups, standardize)
```

Whole-brain MVPA using Pipeline infrastructure.

Delegates to the fluent pipeline API for whole-brain classification,
then extracts mean accuracy for backward compatibility.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`y` |  | Labels to predict. | *required*
`estimator` |  | Estimator name ('svm', 'logistic', etc.). | *required*
`cv` |  | Cross-validation splitter or int. | *required*
`groups` |  | Group labels for CV. | *required*
`standardize` |  | Whether to z-score features. | *required*

**Returns:**

Type | Description
---- | -----------
 | np.ndarray with single mean accuracy value.

#### `nltools.data.braindata.prediction.predict`

```python
predict(bd, X = None, y = None, method = 'whole_brain', estimator = 'svm', cv = 5, groups = None, roi_mask = None, radius = 10.0, scoring = 'accuracy', standardize = True, n_jobs = -1, show_progress = True)
```

Generate predictions using fitted model OR classify patterns (MVPA).

This method supports two prediction modes determined by which parameter
is provided:

1. **Timeseries prediction** (X provided): Use fitted ridge model to
   predict voxel responses for new feature data.

2. **MVPA decoding** (y provided): Train a classifier to predict labels
   from brain patterns using cross-validation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`X` | <code>[array](#array) - [like](#like)</code> | Features for timeseries prediction, shape (n_samples, n_features). If None and y is None, uses training data from fit(). | <code>None</code>
`y` | <code>[array](#array) - [like](#like)</code> | Labels for MVPA decoding, shape (n_samples,). If provided, performs pattern classification instead of timeseries prediction. | <code>None</code>
`MVPA-specific parameters` | <code>only used when y is provided</code> |  | *required*
`method` | <code>[str](#str)</code> | Decoding method - 'whole_brain', 'searchlight', or 'roi'. | <code>'whole_brain'</code>
`estimator` | <code>str or sklearn estimator</code> | Classifier to use. Can be: - 'svm': LinearSVC (default) - 'logistic': LogisticRegression - 'ridge': RidgeClassifier - 'lda': LinearDiscriminantAnalysis - Any sklearn-compatible estimator with fit/predict | <code>'svm'</code>
`cv` | <code>int or sklearn CV splitter</code> | Cross-validation specification. Int for k-fold or sklearn CV object. | <code>5</code>
`groups` | <code>[array](#array) - [like](#like)</code> | Group labels for CV (e.g., run IDs for leave-one-run-out). | <code>None</code>
`roi_mask` | <code>[Nifti1Image](#Nifti1Image) or [str](#str)</code> | Atlas/parcellation for ROI-based decoding. | <code>None</code>
`radius` | <code>[float](#float)</code> | Searchlight radius in mm. Default: 10.0. | <code>10.0</code>
`scoring` | <code>[str](#str)</code> | Metric for evaluation ('accuracy', 'balanced_accuracy', 'roc_auc'). | <code>'accuracy'</code>
`standardize` | <code>[bool](#bool)</code> | Z-score features before classification. Default: True. | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs for searchlight (-1 = all cores). | <code>-1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar for searchlight. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | For timeseries prediction, shape (n_samples, n_voxels). For MVPA, shape (1, n_voxels) with accuracy per voxel/ROI.

**Examples:**

```pycon
>>> # Timeseries prediction (encoding model)
>>> brain_data.fit(model='ridge', X=features)
>>> predictions = brain_data.predict(X=new_features)
```

```pycon
>>> # MVPA decoding (pattern classification)
>>> # brain_data.data has shape (n_trials, n_voxels)
>>> accuracy = brain_data.predict(y=labels, method='searchlight')
>>> print(accuracy.shape)  # (1, n_voxels)
```

#### `nltools.data.braindata.prediction.predict_mvpa`

```python
predict_mvpa(bd, y, method = 'whole_brain', estimator = 'svm', cv = 5, groups = None, roi_mask = None, radius = 10.0, scoring = 'accuracy', standardize = True, n_jobs = -1, show_progress = True)
```

Perform MVPA decoding using cross-validation.

Internal function for pattern classification.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`y` | <code>[array](#array) - [like](#like)</code> | Labels to predict, shape (n_samples,). | *required*
`method` | <code>[str](#str)</code> | 'whole_brain', 'searchlight', or 'roi'. | <code>'whole_brain'</code>
`estimator` | <code>str or sklearn estimator</code> | Classifier (string shortcut or sklearn estimator). | <code>'svm'</code>
`cv` | <code>int or sklearn CV splitter</code> | Cross-validation specification. | <code>5</code>
`groups` | <code>[array](#array) - [like](#like)</code> | Group labels for CV. | <code>None</code>
`roi_mask` | <code>[Nifti1Image](#Nifti1Image) or [str](#str)</code> | Atlas for ROI-based decoding. | <code>None</code>
`radius` | <code>[float](#float)</code> | Searchlight radius in mm. | <code>10.0</code>
`scoring` | <code>[str](#str)</code> | Scoring metric. | <code>'accuracy'</code>
`standardize` | <code>[bool](#bool)</code> | Whether to z-score features. | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Parallel jobs for searchlight. | <code>-1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData with accuracy values.

#### `nltools.data.braindata.prediction.predict_timeseries`

```python
predict_timeseries(bd, X = None)
```

Generate timeseries predictions using fitted model.

Internal function for encoding model prediction.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`X` | <code>[array](#array) - [like](#like)</code> | Features to predict on. If None, uses training data. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData with predicted timeseries.

#### `nltools.data.braindata.prediction.resolve_estimator`

```python
resolve_estimator(bd, estimator)
```

Resolve string shortcut to sklearn estimator.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance (unused, kept for API consistency). | *required*
`estimator` |  | String shortcut or sklearn estimator object. | *required*

**Returns:**

Type | Description
---- | -----------
 | Instantiated sklearn estimator.

