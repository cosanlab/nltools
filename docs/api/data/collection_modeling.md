## `modeling`

Modeling functions extracted from BrainCollection.

Contains GLM fitting, Ridge fitting, design matrix building, and related helpers.
All BrainCollection methods converted to functions taking `bc` as first argument.

**Methods:**

Name | Description
---- | -----------
[`cv`](#cv) | Create a cross-validation pipeline for multi-subject analysis.
[`fit`](#fit) | Fit a model to each subject in the collection.
[`fit_from_events`](#fit_from_events) | Build design matrices from events and fit GLM to each subject.
[`fit_glm`](#fit_glm) | Fit GLM to each subject in collection.
[`fit_glm_internal`](#fit_glm_internal) | Internal GLM fitting with design matrix input.
[`fit_ridge`](#fit_ridge) | Fit ridge regression to each subject in collection.
[`load_design_matrix`](#load_design_matrix) | Load design matrix from a file path.
[`load_features`](#load_features) | Load features from a file path.
[`resolve_X`](#resolve_X) | Resolve design/feature matrix X to per-subject list.
[`resolve_confounds`](#resolve_confounds) | Resolve confounds argument to per-subject list.



### Classes

### Methods

#### `cv`

```python
cv(bc, k: int | None = None, method: str = 'kfold', split_by: str | None = None, groups: np.ndarray | None = None, n: int = 1000, random_state: int | None = None) -> BrainCollectionPipeline
```

Create a cross-validation pipeline for multi-subject analysis.

Returns a pipeline object that enables fluent, chainable transforms
with cross-validation across subjects or runs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`k` | <code>[int](#int) \| None</code> | Number of folds (for kfold method). Defaults to 5. | <code>None</code>
`method` | <code>[str](#str)</code> | CV scheme type. Options: - 'kfold': k-fold cross-validation on pooled data - 'loso': leave-one-subject-out (one image held out per fold) - 'loro': leave-one-run-out (requires groups) | <code>'kfold'</code>
`split_by` | <code>[str](#str) \| None</code> | Metadata column for group splits. If provided and groups is None, gets groups from bc.metadata[split_by]. | <code>None</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Explicit group labels for CV splits. | <code>None</code>
`n` | <code>[int](#int)</code> | Number of iterations for bootstrap/permutation methods. Default 1000. | <code>1000</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainCollectionPipeline` | <code>[BrainCollectionPipeline](#nltools.data.collection.pipeline.BrainCollectionPipeline)</code> | Pipeline for method chaining.

**Examples:**

```pycon
>>> # Leave-one-subject-out classification
>>> result = bc.cv(method='loso').normalize().predict(subject_labels, algorithm='svm')
>>> print(f"Mean accuracy: {result.mean_score:.2%}")
```

```pycon
>>> # With preprocessing
>>> result = (bc
...     .cv(method='loso')
...     .normalize()
...     .reduce(n_components=50)
...     .predict(labels))
```

```pycon
>>> # Run-based CV with metadata
>>> result = bc.cv(method='loro', split_by='run').predict(y)
```

<details class="see-also" open markdown="1">
<summary>See Also</summary>

BrainCollectionPipeline: For available transforms and terminals.
CVScheme: For CV scheme configuration details.

</details>

#### `fit`

```python
fit(bc, model: str, X: pd.DataFrame | np.ndarray | str | list, cv: int | None = None, scale: bool = True, scale_value: float = 100.0, progress_bar: bool = False, **kwargs: bool) -> FittedBrainCollection
```

Fit a model to each subject in the collection.

Unified fitting method that shadows BrainData.fit() API for multi-subject
analysis. Dispatches to model-specific implementations based on the
model parameter.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`model` | <code>[str](#str)</code> | Model type - 'glm' or 'ridge' | *required*
`X` | <code>[DataFrame](#pandas.DataFrame) \| [ndarray](#numpy.ndarray) \| [str](#str) \| [list](#list)</code> | Design/feature matrix. Can be: - pd.DataFrame/DesignMatrix: Shared (used for all subjects) - np.ndarray: Shared array (used for all subjects) - str: Column name in metadata pointing to file paths - list: Per-subject list of DataFrames/arrays/paths | *required*
`cv` | <code>[int](#int) \| None</code> | Cross-validation folds (Ridge only). Default is None for GLM, 5 for Ridge when output='scores'. | <code>None</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>False</code>
`**kwargs` |  | Model-specific arguments passed to _fit_glm or _fit_ridge: - GLM: return_stats, save - Ridge: alpha, output, save, backend, random_state | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[FittedBrainCollection](#nltools.data.collection.pipeline.FittedBrainCollection)</code> | FittedBrainCollection wrapping the fitted results. Supports:
<code>[FittedBrainCollection](#nltools.data.collection.pipeline.FittedBrainCollection)</code> | - ``.results``: Access underlying BrainCollection(s) directly
<code>[FittedBrainCollection](#nltools.data.collection.pipeline.FittedBrainCollection)</code> | - ``.betas``: Convenience accessor for beta coefficients (GLM)
<code>[FittedBrainCollection](#nltools.data.collection.pipeline.FittedBrainCollection)</code> | - ``.pool()``: Aggregate across subjects for group analysis
<code>[FittedBrainCollection](#nltools.data.collection.pipeline.FittedBrainCollection)</code> | The underlying results contain:
<code>[FittedBrainCollection](#nltools.data.collection.pipeline.FittedBrainCollection)</code> | - GLM: Beta coefficients (n_regressors, n_voxels) per subject
<code>[FittedBrainCollection](#nltools.data.collection.pipeline.FittedBrainCollection)</code> | - Ridge: Scores or weights depending on 'output' kwarg
<code>[FittedBrainCollection](#nltools.data.collection.pipeline.FittedBrainCollection)</code> | If return_stats (GLM) or output='both' (Ridge), results is a dict.

**Examples:**

```pycon
>>> # GLM with shared design matrix
>>> fitted = bc.fit(model='glm', X=dm)
>>> betas = fitted.results  # Access BrainCollection directly
>>>
>>> # Two-stage analysis with pool()
>>> pool = bc.fit(model='glm', X=dm).pool(param='beta')
>>> t_map = pool.fit(model='ttest', contrast='A-B')
>>>
>>> # GLM with per-subject design matrices
>>> fitted = bc.fit(model='glm', X=[dm1, dm2, dm3])
>>>
>>> # Ridge encoding model with CV scores
>>> fitted = bc.fit(model='ridge', X=features, cv=5)
>>> scores = fitted.results
```

<details class="see-also" open markdown="1">
<summary>See Also</summary>

fit_from_events: Convenience method for event-based GLM workflows
fit_glm: Legacy GLM fitting (use fit_from_events instead)
fit_ridge: Legacy Ridge fitting (use fit(..., model='ridge') instead)

</details>

#### `fit_from_events`

```python
fit_from_events(bc, events: pd.DataFrame, t_r: float, confounds: str | list[pd.DataFrame | Path | str] | None = None, confound_columns: list[str] | None = None, hrf_model: str = 'spm', drift_model: str = 'cosine', high_pass: float = 0.01, scale: bool = True, scale_value: float = 100.0, return_stats: list[str] | None = None, return_residuals: bool = False, save: dict[str, str] | None = None, progress_bar: bool = False, by_run: bool = False, run_column: str = 'run', run_lengths: int | list[int] | None = None) -> BrainCollection | dict[str, BrainCollection]
```

Build design matrices from events and fit GLM to each subject.

Convenience method for event-based experimental designs. Builds
nilearn-compatible design matrices from the events DataFrame and
fits a GLM to each subject in the collection.

This is the recommended method for typical task-based fMRI analysis
where you have event timing information. For more control, use
fit(model='glm', X=design_matrices) with pre-built design matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`events` | <code>[DataFrame](#pandas.DataFrame)</code> | Task events DataFrame with onset, duration, trial_type columns. This is shared across all subjects (same experimental paradigm). If by_run=True, must also have a run column. | *required*
`t_r` | <code>[float](#float)</code> | Repetition time (TR) in seconds. | *required*
`confounds` | <code>[str](#str) \| [list](#list)[[DataFrame](#pandas.DataFrame) \| [Path](#pathlib.Path) \| [str](#str)] \| None</code> | Subject-specific confounds. Can be: - str: Column name in metadata pointing to confound file paths - list: List of DataFrames or paths, one per subject - None: No confounds (only task + drift terms) | <code>None</code>
`confound_columns` | <code>[list](#list)[[str](#str)] \| None</code> | Columns to extract from confound files. If None and confounds provided, uses all columns. | <code>None</code>
`hrf_model` | <code>[str](#str)</code> | HRF model for convolution ('spm', 'glover', 'fir', etc.) | <code>'spm'</code>
`drift_model` | <code>[str](#str)</code> | Drift model ('cosine', 'polynomial', None) | <code>'cosine'</code>
`high_pass` | <code>[float](#float)</code> | High-pass filter cutoff in Hz (default 0.01) | <code>0.01</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`return_stats` | <code>[list](#list)[[str](#str)] \| None</code> | Optional list of statistics to return as separate BrainCollections. Options: 't', 'r2', 'p', 'se', 'residual'. | <code>None</code>
`return_residuals` | <code>[bool](#bool)</code> | If True, return residuals (same as return_stats=['residual']). | <code>False</code>
`save` | <code>[dict](#dict)[[str](#str), [str](#str)] \| None</code> | Dict mapping output type to path template. | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>False</code>
`by_run` | <code>[bool](#bool)</code> | If True, fit GLM separately per run and return run-level betas. This enables MVPA decoding with leave-one-run-out CV. | <code>False</code>
`run_column` | <code>[str](#str)</code> | Column name in events identifying runs (default 'run'). | <code>'run'</code>
`run_lengths` | <code>[int](#int) \| [list](#list)[[int](#int)] \| None</code> | Number of TRs per run. Required when by_run=True. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | BrainCollection of beta coefficients for task regressors.
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | If return_stats specified, returns dict with keys 'betas', 't', etc.

**Examples:**

```pycon
>>> # Basic GLM fit from events
>>> betas = bc.fit_from_events(events=events_df, t_r=2.0)
>>> group_t = betas.ttest()
>>>
>>> # With confounds from metadata column
>>> betas = bc.fit_from_events(
...     events=events_df,
...     t_r=2.0,
...     confounds='confound_file',
...     confound_columns=['trans_x', 'trans_y', 'trans_z']
... )
>>>
>>> # Run-level betas for MVPA
>>> betas = bc.fit_from_events(events=events_df, t_r=2.0, by_run=True)
```

<details class="see-also" open markdown="1">
<summary>See Also</summary>

fit: Unified fit method that accepts pre-built design matrices
_fit_glm: Internal method for design matrix-based fitting

</details>

#### `fit_glm`

```python
fit_glm(bc, events: pd.DataFrame, t_r: float, confounds: str | list[pd.DataFrame | Path | str] | None = None, confound_columns: list[str] | None = None, hrf_model: str = 'spm', drift_model: str = 'cosine', high_pass: float = 0.01, scale: bool = True, scale_value: float = 100.0, return_stats: list[str] | None = None, return_residuals: bool = False, save: dict[str, str] | None = None, progress_bar: bool = False, by_run: bool = False, run_column: str = 'run', run_lengths: int | list[int] | None = None) -> BrainCollection | dict[str, BrainCollection]
```

Fit GLM to each subject in collection.

Memory-efficient first-level GLM analysis that processes subjects
one at a time. Returns a BrainCollection of beta coefficients for
task regressors (confounds and drift terms are fit but not returned).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`events` | <code>[DataFrame](#pandas.DataFrame)</code> | Task events DataFrame with onset, duration, trial_type columns. This is shared across all subjects (same experimental paradigm). If by_run=True, must also have a run column. | *required*
`t_r` | <code>[float](#float)</code> | Repetition time (TR) in seconds. | *required*
`confounds` | <code>[str](#str) \| [list](#list)[[DataFrame](#pandas.DataFrame) \| [Path](#pathlib.Path) \| [str](#str)] \| None</code> | Subject-specific confounds. Can be: - str: Column name in metadata pointing to confound file paths - list: List of DataFrames or paths, one per subject - None: No confounds (only task + drift terms) | <code>None</code>
`confound_columns` | <code>[list](#list)[[str](#str)] \| None</code> | Columns to extract from confound files. If None and confounds provided, uses all columns. | <code>None</code>
`hrf_model` | <code>[str](#str)</code> | HRF model for convolution ('spm', 'glover', 'fir', etc.) | <code>'spm'</code>
`drift_model` | <code>[str](#str)</code> | Drift model ('cosine', 'polynomial', None) | <code>'cosine'</code>
`high_pass` | <code>[float](#float)</code> | High-pass filter cutoff in Hz (default 0.01) | <code>0.01</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`return_stats` | <code>[list](#list)[[str](#str)] \| None</code> | Optional list of statistics to return as separate BrainCollections. Options: 't', 'r2', 'p', 'se', 'residual'. | <code>None</code>
`return_residuals` | <code>[bool](#bool)</code> | If True, return residuals (same as return_stats=['residual']). | <code>False</code>
`save` | <code>[dict](#dict)[[str](#str), [str](#str)] \| None</code> | Dict mapping output type to path template, e.g. ``{'betas': 'output/{subject}_betas.nii.gz', 't': 'output/{subject}_tstat.nii.gz'}``. Supports {subject}, {session}, {idx}, and other metadata columns. | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>False</code>
`by_run` | <code>[bool](#bool)</code> | If True, fit GLM separately per run and return run-level betas. This enables MVPA decoding with leave-one-run-out CV. Each subject will have (n_runs * n_conditions, n_voxels) betas. | <code>False</code>
`run_column` | <code>[str](#str)</code> | Column name in events identifying runs (default 'run'). | <code>'run'</code>
`run_lengths` | <code>[int](#int) \| [list](#list)[[int](#int)] \| None</code> | Number of TRs per run. Required when by_run=True.<br>- int: All runs have same length - list of int: Different length per run - None: Will attempt to infer equal-length runs from total scans | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | BrainCollection where each BrainData has shape:
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | - (n_task_regressors, n_voxels) if by_run=False (default)
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | - (n_runs * n_task_regressors, n_voxels) if by_run=True
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | The ``._design_columns`` attribute stores task regressor names.
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | If by_run=True, also stores ``._condition_labels`` and ``._run_labels``.
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | If return_stats specified, returns dict with keys 'betas', 't', etc.

**Examples:**

```pycon
>>> # Basic GLM fit
>>> betas = bc.fit_glm(events=events_df, t_r=2.0)
>>> # Group t-test on first regressor
>>> group_t = betas[:, 0].ttest()
```

```pycon
>>> # Run-level betas for MVPA decoding
>>> betas = bc.fit_glm(events=events_df, t_r=2.0, by_run=True)
>>> # betas._condition_labels = ['face', 'house', 'face', 'house', ...]
>>> # betas._run_labels = [1, 1, 2, 2, 3, 3, ...]
>>> accuracy = betas.predict(y=None, method='searchlight')
```

```pycon
>>> # With confounds from metadata column
>>> betas = bc.fit_glm(
...     events=events_df,
...     t_r=2.0,
...     confounds='confound_file',  # column name in metadata
...     confound_columns=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
... )
```

#### `fit_glm_internal`

```python
fit_glm_internal(bc, X: pd.DataFrame | np.ndarray | str | list, scale: bool = True, scale_value: float = 100.0, return_stats: list[str] | None = None, save: dict[str, str] | None = None, progress_bar: bool = False) -> BrainCollection | dict[str, BrainCollection]
```

Internal GLM fitting with design matrix input.

Core implementation that accepts DesignMatrix/DataFrame directly.
Called by fit(model='glm') and fit_from_events().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`X` | <code>[DataFrame](#pandas.DataFrame) \| [ndarray](#numpy.ndarray) \| [str](#str) \| [list](#list)</code> | Design matrix. Can be: - pd.DataFrame/DesignMatrix: Shared (used for all subjects) - np.ndarray: Shared array (converted to DataFrame internally) - str: Column name in metadata pointing to file paths - list: Per-subject list of DataFrames/arrays/paths | *required*
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`return_stats` | <code>[list](#list)[[str](#str)] \| None</code> | Optional list of statistics to return as separate BrainCollections. Options: 't', 'r2', 'p', 'se', 'residual'. | <code>None</code>
`save` | <code>[dict](#dict)[[str](#str), [str](#str)] \| None</code> | Dict mapping output type to path template. | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | BrainCollection of betas, or dict with betas + requested stats.

#### `fit_ridge`

```python
fit_ridge(bc, X: np.ndarray | str | list, alpha: float | str = 1.0, cv: int | None = 5, scale: bool = True, scale_value: float = 100.0, output: str = 'scores', save: dict[str, str] | None = None, progress_bar: bool = False, **ridge_kwargs: bool) -> BrainCollection | dict[str, BrainCollection]
```

Fit ridge regression to each subject in collection.

Memory-efficient encoding model fitting that processes subjects one at a
time. Default behavior returns cross-validated R-squared scores per voxel,
suitable for group-level inference on encoding model performance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`X` | <code>[ndarray](#numpy.ndarray) \| [str](#str) \| [list](#list)</code> | Feature matrix. Can be: - np.ndarray: Shared features (n_samples, n_features) used for all subjects - str: Column name in metadata pointing to feature file paths - list: List of arrays/DataFrames, one per subject | *required*
`alpha` | <code>[float](#float) \| [str](#str)</code> | Ridge regularization parameter. Can be: - float: Fixed regularization strength - 'auto': Use cross-validation to select optimal alpha | <code>1.0</code>
`cv` | <code>[int](#int) \| None</code> | Cross-validation folds for computing scores. Default is 5. Required when output='scores' or 'both'. Set to None only when output='weights'. | <code>5</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`output` | <code>[str](#str)</code> | What to return. Options: - 'scores': CV R-squared scores per voxel (default, for encoding workflow) - 'weights': Model weights (n_features, n_voxels) - 'both': Dict with both 'scores' and 'weights' | <code>'scores'</code>
`save` | <code>[dict](#dict)[[str](#str), [str](#str)] \| None</code> | Dict mapping output type to path template, e.g. ``{'weights': 'output/{subject}_weights.nii.gz', 'scores': 'output/{subject}_scores.nii.gz'}``. Supports {subject}, {session}, {idx}, and other metadata columns. | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>False</code>
`**ridge_kwargs` |  | Additional arguments passed to Ridge model (e.g., backend='torch', random_state=42). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | BrainCollection of scores or weights, or dict with both if output='both'.
<code>[BrainCollection](#nltools.data.collection.BrainCollection) \| [dict](#dict)[[str](#str), [BrainCollection](#nltools.data.collection.BrainCollection)]</code> | Each BrainData will have ``cv_results_`` attribute when cv is used.

**Examples:**

```pycon
>>> # Encoding model workflow: get CV scores for group analysis
>>> scores = bc.fit_ridge(X=features, alpha=1.0)
>>> group_ttest = scores.ttest()  # Test encoding accuracy vs chance
```

```pycon
>>> # Get both scores and weights
>>> results = bc.fit_ridge(X=features, alpha=1.0, output='both')
>>> scores = results['scores']
>>> weights = results['weights']
```

```pycon
>>> # Auto-select alpha with CV
>>> scores = bc.fit_ridge(X=features, alpha='auto', cv=5)
```

```pycon
>>> # Get weights only (no CV needed)
>>> weights = bc.fit_ridge(X=features, alpha=1.0, output='weights', cv=None)
```

#### `load_design_matrix`

```python
load_design_matrix(bc, path: str | Path) -> pd.DataFrame
```

Load design matrix from a file path.

Supports common formats: .csv, .tsv, .txt

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance (unused, kept for API consistency). | *required*
`path` | <code>[str](#str) \| [Path](#pathlib.Path)</code> | Path to design matrix file. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#pandas.DataFrame)</code> | DataFrame with design matrix contents.

#### `load_features`

```python
load_features(bc, path: str | Path) -> np.ndarray
```

Load features from a file path.

Supports common formats: .npy, .csv, .tsv, .txt

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance (unused, kept for API consistency). | *required*
`path` | <code>[str](#str) \| [Path](#pathlib.Path)</code> | Path to feature file. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | NumPy array of feature values.

#### `resolve_X`

```python
resolve_X(bc, X: np.ndarray | pd.DataFrame | str | list | None) -> list | None
```

Resolve design/feature matrix X to per-subject list.

Unified helper for resolving X parameter across fit methods. Supports
three input patterns:
1. Shared matrix (array/DataFrame/DesignMatrix): Same X for all subjects
2. Per-subject list: List of matrices, one per subject
3. Metadata column: String column name pointing to file paths

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`X` | <code>[ndarray](#numpy.ndarray) \| [DataFrame](#pandas.DataFrame) \| [str](#str) \| [list](#list) \| None</code> | Design/feature matrix. Can be: - np.ndarray: Shared array (used for all subjects) - pd.DataFrame: Shared DataFrame/DesignMatrix (used for all subjects) - str: Column name in metadata containing file paths - list: Per-subject list of arrays/DataFrames/paths - None: Error | *required*

**Returns:**

Type | Description
---- | -----------
<code>[list](#list) \| None</code> | list | None: Per-subject list if X varies by subject, None if shared. Caller should use: `X_subj = X_list[i] if X_list else X`

#### `resolve_confounds`

```python
resolve_confounds(bc, confounds: str | list[pd.DataFrame | Path | str] | None) -> list[pd.DataFrame | Path | str] | None
```

Resolve confounds argument to per-subject list.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`confounds` | <code>[str](#str) \| [list](#list)[[DataFrame](#pandas.DataFrame) \| [Path](#pathlib.Path) \| [str](#str)] \| None</code> | Either: - str: Column name in metadata containing confound paths - list: Already per-subject list of DataFrames or paths - None: No confounds | *required*

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[DataFrame](#pandas.DataFrame) \| [Path](#pathlib.Path) \| [str](#str)] \| None</code> | List of confounds (one per subject) or None

