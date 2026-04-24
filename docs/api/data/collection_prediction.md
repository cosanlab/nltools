## `prediction`

Prediction functions extracted from BrainCollection.

Contains predict, compute_contrasts, select_feature, and related helpers.
All BrainCollection methods converted to functions taking `bc` as first argument.

**Methods:**

Name | Description
---- | -----------
[`compute_contrasts`](#compute_contrasts) | Compute contrasts from fitted GLM beta coefficients.
[`compute_single_contrast`](#compute_single_contrast) | Compute a single contrast across all subjects.
[`parse_contrast_string`](#parse_contrast_string) | Parse a contrast string into a numeric contrast vector.
[`predict`](#predict) | Generate predictions for each subject in collection.
[`select_feature`](#select_feature) | Select a single feature's weights across all subjects.



### Classes

### Methods

#### `compute_contrasts`

```python
compute_contrasts(bc, contrasts: str | dict | np.ndarray | list) -> BrainCollection | dict[str, BrainCollection]
```

Compute contrasts from fitted GLM beta coefficients.

Applies contrast weights to each subject's betas and returns a
BrainCollection of contrast values suitable for group-level analysis.

Must be called on a BrainCollection created by fit_glm() which has
the _design_columns attribute set.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`contrasts` | <code>[str](#str) \| [dict](#dict) \| [ndarray](#numpy.ndarray) \| [list](#list)</code> | Can be: - str: Contrast string using column names, e.g., "face - house" - dict: Multiple contrasts, e.g., {"main": "face - house", "avg": [0.5, 0.5]} - array/list: Numeric contrast vector, e.g., [1, -1] | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | BrainCollection where each BrainData has shape (n_voxels,) containing
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | the contrast values. If dict input, returns dict of BrainCollections.

**Examples:**

```pycon
>>> # Fit GLM and compute contrast
>>> betas = bc.fit_glm(events=events_df, t_r=2.0)
>>> contrast = betas.compute_contrasts("face - house")
>>> # Group t-test on contrast
>>> group_result = contrast.ttest()
```

```pycon
>>> # Multiple contrasts
>>> contrasts = betas.compute_contrasts({
...     "face_vs_house": "face - house",
...     "face_vs_baseline": "face",
... })
>>> face_vs_house_ttest = contrasts["face_vs_house"].ttest()
```

#### `compute_single_contrast`

```python
compute_single_contrast(bc, contrast: str | np.ndarray | list, design_columns: list[str]) -> BrainCollection
```

Compute a single contrast across all subjects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`contrast` | <code>[str](#str) \| [ndarray](#numpy.ndarray) \| [list](#list)</code> | Contrast specification (string, array, or list) | *required*
`design_columns` | <code>[list](#list)[[str](#str)]</code> | List of regressor names from fit_glm | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | BrainCollection with contrast values for each subject

#### `parse_contrast_string`

```python
parse_contrast_string(bc, contrast_str: str, design_columns: list[str]) -> np.ndarray
```

Parse a contrast string into a numeric contrast vector.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance (unused, kept for API consistency). | *required*
`contrast_str` | <code>[str](#str)</code> | Contrast string like "A - B" or "2*A - B" | *required*
`design_columns` | <code>[list](#list)[[str](#str)]</code> | List of regressor column names | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Numeric contrast vector

#### `predict`

```python
predict(bc, X: np.ndarray | str | list | None = None, y: np.ndarray | None = None, method: str = 'whole_brain', estimator: str = 'svm', cv: str = 5, groups: np.ndarray | None = None, roi_mask: np.ndarray | None = None, radius_mm: float = 10.0, scoring: str = 'accuracy', standardize: bool = True, n_jobs: int = -1, progress_bar: bool = False) -> BrainCollection
```

Generate predictions for each subject in collection.

This method supports two prediction modes determined by which parameter
is provided:

1. **Timeseries prediction** (X provided): Use fitted ridge model to
   predict voxel responses for new feature data.

2. **MVPA decoding** (y provided): Train a classifier to predict labels
   from brain patterns using cross-validation.

For MVPA, if this collection was created with by_run=True, you can
use y=None to infer labels from _condition_labels and groups from
_run_labels (leave-one-run-out CV).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`X` | <code>[ndarray](#numpy.ndarray) \| [str](#str) \| [list](#list) \| None</code> | Features for timeseries prediction. Can be: - np.ndarray: Shared features (same for all subjects) - str: Metadata column with per-subject feature paths - list: Per-subject feature arrays | <code>None</code>
`y` | <code>[ndarray](#numpy.ndarray) \| None</code> | Labels for MVPA decoding. If None and _condition_labels exists, will use stored condition labels (from fit_glm with by_run=True). | <code>None</code>
`method` | <code>[str](#str)</code> | MVPA method - 'whole_brain', 'searchlight', or 'roi'. | <code>'whole_brain'</code>
`estimator` |  | Classifier - 'svm', 'logistic', 'ridge', 'lda' or sklearn estimator instance. | <code>'svm'</code>
`cv` |  | Cross-validation strategy. If None and _run_labels exists, uses leave-one-group-out with run labels. | <code>5</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Group labels for GroupKFold/LeaveOneGroupOut. If None and _run_labels exists, uses stored run labels. | <code>None</code>
`roi_mask` |  | Mask for ROI-based MVPA. Required if method='roi'. | <code>None</code>
`radius_mm` | <code>[float](#float)</code> | searchlight radius in mm (default 10.0). | <code>10.0</code>
`scoring` | <code>[str](#str)</code> | Scoring metric (default 'accuracy'). | <code>'accuracy'</code>
`standardize` | <code>[bool](#bool)</code> | If True, standardize features before classification. | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Parallel jobs for searchlight (-1 = all cores). | <code>-1</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | BrainCollection with prediction results:
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | - For timeseries: (n_timepoints, n_voxels) predicted responses
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | - For MVPA: (1, n_voxels) accuracy values

**Examples:**

```pycon
>>> # MVPA workflow with run-level betas
>>> betas = bc.fit_glm(events=events, t_r=2.0, by_run=True)
>>> accuracy = betas.predict(y=None, method='whole_brain')
>>> # y and groups inferred from _condition_labels, _run_labels
```

```pycon
>>> # Explicit labels
>>> accuracy = betas.predict(y=labels, method='searchlight')
```

```pycon
>>> # Timeseries prediction with ridge weights
>>> weights = bc.fit_ridge(X=features, output='weights')
>>> predictions = weights.predict(X=new_features)
```

#### `select_feature`

```python
select_feature(bc, feature: int | str) -> BrainCollection
```

Select a single feature's weights across all subjects.

Used after fit_ridge() to extract weights for a specific feature
for group-level analysis (e.g., t-test on feature weights).

Must be called on a BrainCollection created by fit_ridge() where
each subject has shape (n_features, n_voxels).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`feature` | <code>[int](#int) \| [str](#str)</code> | Feature to select. Can be: - int: Feature index (0-based) - str: Feature name (requires _feature_names attribute) | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | BrainCollection where each BrainData has shape (n_voxels,)
<code>[BrainCollection](#nltools.data.collection.BrainCollection)</code> | containing the weights for the specified feature.

**Examples:**

```pycon
>>> # Fit ridge and select feature
>>> weights = bc.fit_ridge(X=features, alpha=1.0)
>>> feature_0 = weights.select_feature(0)
>>> # Group t-test on first feature's weights
>>> group_result = feature_0.ttest()
```

```pycon
>>> # By name (if features had column names)
>>> face_weights = weights.select_feature("face_response")
```

