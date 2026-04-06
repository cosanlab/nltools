## `nltools.data.collection`

BrainCollection: Multi-subject brain data container.

Provides tensor-like semantics for efficient group analyses with lazy loading
and memory-efficient operations.

<details class="shape-semantics" open markdown="1">
<summary>(n_images, n_observations, n_voxels)</summary>

- axis 0: images (subjects, runs, etc.)
- axis 1: observations (timepoints, TRs)
- axis 2: voxels (spatial)

</details>

**Modules:**

Name | Description
---- | -----------
[`constructors`](#nltools.data.collection.constructors) | Constructor functions for BrainCollection.
[`inference`](#nltools.data.collection.inference) | BrainCollection inference functions.
[`io`](#nltools.data.collection.io) | I/O functions for BrainCollection.
[`modeling`](#nltools.data.collection.modeling) | Modeling functions extracted from BrainCollection.
[`pipeline`](#nltools.data.collection.pipeline) | Pipeline classes for BrainCollection.
[`prediction`](#nltools.data.collection.prediction) | Prediction functions extracted from BrainCollection.
[`transforms`](#nltools.data.collection.transforms) | BrainCollection transform functions.

**Classes:**

Name | Description
---- | -----------
[`BrainCollection`](#nltools.data.collection.BrainCollection) | Collection of brain images with tensor-like operations.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`T`](#nltools.data.collection.T) |  | 
[`tqdm`](#nltools.data.collection.tqdm) |  | 



### Attributes#### `nltools.data.collection.T`

```python
T = TypeVar('T')
```

#### `nltools.data.collection.tqdm`

```python
tqdm = attempt_to_import('tqdm', 'tqdm')
```



### Classes#### `nltools.data.collection.BrainCollection`

```python
BrainCollection(items: list[Path | str | 'BrainData'], mask: nib.Nifti1Image | Path | str, metadata: pd.DataFrame | None = None, lazy: bool = True)
```

Collection of brain images with tensor-like operations.

BrainCollection provides a container for multiple brain images (e.g., multiple
subjects or runs) with numpy-style indexing and axis operations. It supports
lazy loading for memory efficiency and integrates with pybids for BIDS datasets.

<details class="shape-semantics" open markdown="1">
<summary>(n_images, n_observations, n_voxels)</summary>

- axis 0: images (subjects, runs, etc.)
- axis 1: observations (timepoints, TRs)
- axis 2: voxels (spatial locations)

</details>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`items` | <code>[list](#list)[[Path](#pathlib.Path) \| [str](#str) \| 'BrainData']</code> | List of file paths, BrainData objects, or mix of both. Paths are loaded lazily by default. | *required*
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str)</code> | Brain mask. Required. Can be: - nibabel Nifti1Image - Path to mask file - Template name (e.g., '2mm-MNI152-2009c') | *required*
`metadata` | <code>[DataFrame](#pandas.DataFrame) \| None</code> | Optional DataFrame with per-image metadata (subject, session, etc.). Index should match items order. | <code>None</code>
`lazy` | <code>[bool](#bool)</code> | If True (default), paths are not loaded until accessed. | <code>True</code>

**Examples:**

```pycon
>>> # Create from paths (lazy loading)
>>> bc = BrainCollection(
...     ['/data/sub-01.nii.gz', '/data/sub-02.nii.gz'],
...     mask='2mm-MNI152-2009c'
... )
>>> bc.shape
(2, 100, 228453)
```

```pycon
>>> # NumPy-style indexing
>>> bc[0]  # First subject -> BrainData
>>> bc[:, 0]  # First timepoint across all subjects -> BrainCollection
>>> bc[0:5, 10:20]  # 5 subjects, timepoints 10-20 -> BrainCollection
```

```pycon
>>> # Axis operations
>>> bc.mean(axis=0)  # Mean across subjects -> BrainData
>>> bc.mean(axis=1)  # Mean across time per subject -> BrainCollection
```

```pycon
>>> # From BIDS dataset
>>> bc = BrainCollection.from_bids('/data/bids', task='rest', mask=mask)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- All images must share the same mask/space. Heterogeneous masks are not
  supported; data is resampled to mask space on load.
- Some operations (e.g., to_tensor) require uniform observation counts
  across all images.

</details>

**Functions:**

Name | Description
---- | -----------
[`align`](#nltools.data.collection.BrainCollection.align) | Align subjects using local functional alignment.
[`anova`](#nltools.data.collection.BrainCollection.anova) | One-way ANOVA across groups defined by metadata.
[`compute_contrasts`](#nltools.data.collection.BrainCollection.compute_contrasts) | Compute contrasts from fitted GLM beta coefficients.
[`cv`](#nltools.data.collection.BrainCollection.cv) | Create a cross-validation pipeline for multi-subject analysis.
[`detrend`](#nltools.data.collection.BrainCollection.detrend) | Remove trend from each image.
[`filter`](#nltools.data.collection.BrainCollection.filter) | Filter collection by predicate.
[`fit`](#nltools.data.collection.BrainCollection.fit) | Fit a model to each subject in the collection.
[`fit_from_events`](#nltools.data.collection.BrainCollection.fit_from_events) | Build design matrices from events and fit GLM to each subject.
[`fit_glm`](#nltools.data.collection.BrainCollection.fit_glm) | Fit GLM to each subject in collection.
[`fit_ridge`](#nltools.data.collection.BrainCollection.fit_ridge) | Fit ridge regression to each subject in collection.
[`from_bids`](#nltools.data.collection.BrainCollection.from_bids) | Create BrainCollection from a BIDS dataset.
[`from_glob`](#nltools.data.collection.BrainCollection.from_glob) | Create BrainCollection from glob pattern.
[`from_stacked`](#nltools.data.collection.BrainCollection.from_stacked) | Create BrainCollection by splitting a stacked BrainData.
[`isc`](#nltools.data.collection.BrainCollection.isc) | Compute intersubject correlation (ISC) across the collection.
[`isc_test`](#nltools.data.collection.BrainCollection.isc_test) | Compute ISC with permutation testing for statistical inference.
[`iter_batches`](#nltools.data.collection.BrainCollection.iter_batches) | Iterate in batches along axis.
[`load`](#nltools.data.collection.BrainCollection.load) | Load specified images into memory.
[`map`](#nltools.data.collection.BrainCollection.map) | Apply function across specified axis.
[`max`](#nltools.data.collection.BrainCollection.max) | Compute maximum along axis. See mean() for details.
[`mean`](#nltools.data.collection.BrainCollection.mean) | Compute mean along axis.
[`median`](#nltools.data.collection.BrainCollection.median) | Compute median along axis. See mean() for details.
[`memory_estimate`](#nltools.data.collection.BrainCollection.memory_estimate) | Estimate memory usage for loading all images.
[`min`](#nltools.data.collection.BrainCollection.min) | Compute minimum along axis. See mean() for details.
[`permutation_test`](#nltools.data.collection.BrainCollection.permutation_test) | One-sample permutation test across images (sign-flipping).
[`permutation_test2`](#nltools.data.collection.BrainCollection.permutation_test2) | Two-sample permutation test between collections.
[`predict`](#nltools.data.collection.BrainCollection.predict) | Generate predictions for each subject in collection.
[`select_feature`](#nltools.data.collection.BrainCollection.select_feature) | Select a single feature's weights across all subjects.
[`smooth`](#nltools.data.collection.BrainCollection.smooth) | Spatially smooth each image.
[`standardize`](#nltools.data.collection.BrainCollection.standardize) | Standardize each image.
[`std`](#nltools.data.collection.BrainCollection.std) | Compute standard deviation along axis. See mean() for details.
[`sum`](#nltools.data.collection.BrainCollection.sum) | Compute sum along axis. See mean() for details.
[`threshold`](#nltools.data.collection.BrainCollection.threshold) | Threshold each image.
[`to_list`](#nltools.data.collection.BrainCollection.to_list) | Return list of BrainData objects.
[`to_stacked`](#nltools.data.collection.BrainCollection.to_stacked) | Stack all into single BrainData (n_total_obs, n_voxels).
[`to_tensor`](#nltools.data.collection.BrainCollection.to_tensor) | Convert to numpy array (n_images, n_obs, n_voxels).
[`ttest`](#nltools.data.collection.BrainCollection.ttest) | One-sample t-test across images.
[`ttest2`](#nltools.data.collection.BrainCollection.ttest2) | Two-sample t-test between collections.
[`unload`](#nltools.data.collection.BrainCollection.unload) | Free memory for specified images (keep paths for reloading).
[`var`](#nltools.data.collection.BrainCollection.var) | Compute variance along axis. See mean() for details.
[`write`](#nltools.data.collection.BrainCollection.write) | Write all images in collection to files.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`is_loaded`](#nltools.data.collection.BrainCollection.is_loaded) | <code>[list](#list)[[bool](#bool)]</code> | List indicating which images are currently in memory.
[`mask`](#nltools.data.collection.BrainCollection.mask) | <code>[Nifti1Image](#nibabel.Nifti1Image)</code> | Shared NIfTI brain mask image used to define the voxel space for the collection.
[`metadata`](#nltools.data.collection.BrainCollection.metadata) | <code>[DataFrame](#pandas.DataFrame)</code> | Per-image metadata DataFrame.
[`n_images`](#nltools.data.collection.BrainCollection.n_images) | <code>[int](#int)</code> | Number of images in collection.
[`n_voxels`](#nltools.data.collection.BrainCollection.n_voxels) | <code>[int](#int)</code> | Number of voxels (from mask).
[`shape`](#nltools.data.collection.BrainCollection.shape) | <code>[tuple](#tuple)[[int](#int), [int](#int) \| None, [int](#int)]</code> | Shape as (n_images, n_observations, n_voxels).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`items` | <code>[list](#list)[[Path](#pathlib.Path) \| [str](#str) \| 'BrainData']</code> | List of paths or BrainData objects. | *required*
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str)</code> | Shared mask (required). Path, nibabel image, or template name. | *required*
`metadata` | <code>[DataFrame](#pandas.DataFrame) \| None</code> | Optional per-image metadata DataFrame. | <code>None</code>
`lazy` | <code>[bool](#bool)</code> | If True, paths are loaded on demand. | <code>True</code>



##### Attributes###### `nltools.data.collection.BrainCollection.is_loaded`

```python
is_loaded: list[bool]
```

List indicating which images are currently in memory.

###### `nltools.data.collection.BrainCollection.mask`

```python
mask: nib.Nifti1Image
```

Shared NIfTI brain mask image used to define the voxel space for the collection.

###### `nltools.data.collection.BrainCollection.metadata`

```python
metadata: pd.DataFrame
```

Per-image metadata DataFrame.

###### `nltools.data.collection.BrainCollection.n_images`

```python
n_images: int
```

Number of images in collection.

###### `nltools.data.collection.BrainCollection.n_voxels`

```python
n_voxels: int
```

Number of voxels (from mask).

###### `nltools.data.collection.BrainCollection.shape`

```python
shape: tuple[int, int | None, int]
```

Shape as (n_images, n_observations, n_voxels).

n_observations is None if images have variable counts or not all are loaded.



##### Functions###### `nltools.data.collection.BrainCollection.align`

```python
align(method: str = 'procrustes', scheme: str = 'searchlight', radius_mm: float = 10.0, parcellation: 'nib.Nifti1Image | None' = None, n_features: int | None = None, n_iter: int = 3, parallel: str | None = 'cpu', n_jobs: int = -1, return_model: bool = False, show_progress: bool = True) -> 'BrainCollection | tuple[BrainCollection, object]'
```

Align subjects using local functional alignment.

Performs neighborhood-based functional alignment across subjects using
LocalAlignment. Each subject's data is aligned to a common template space
using local transforms learned within searchlight spheres or parcels.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Alignment method. Options: - 'procrustes': Orthogonal Procrustes (default, preserves dimensions) - 'srm': Shared Response Model (dimensionality reduction) - 'hyperalignment': Hyperalignment (iterative Procrustes) | <code>'procrustes'</code>
`scheme` | <code>[str](#str)</code> | Spatial scheme. Options: - 'searchlight': Overlapping spheres with center-only aggregation - 'piecewise': Non-overlapping parcels (requires parcellation) | <code>'searchlight'</code>
`radius_mm` | <code>[float](#float)</code> | Sphere radius in millimeters for searchlight scheme. | <code>10.0</code>
`parcellation` | <code>'nib.Nifti1Image \| None'</code> | Parcellation image for piecewise scheme (required if scheme='piecewise'). | <code>None</code>
`n_features` | <code>[int](#int) \| None</code> | Number of features for SRM. None uses full dimensions. | <code>None</code>
`n_iter` | <code>[int](#int)</code> | Number of iterations for alignment refinement. | <code>3</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization mode. Options: - None: Single-threaded - 'cpu': CPU parallelization with joblib - 'gpu': GPU acceleration via PyTorch | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs for CPU mode (-1 = auto). | <code>-1</code>
`return_model` | <code>[bool](#bool)</code> | If True, return (aligned_collection, model) tuple for fit/transform workflow with new data. | <code>False</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| tuple[BrainCollection, object]'</code> | BrainCollection with aligned data. If return_model=True, returns
<code>'BrainCollection \| tuple[BrainCollection, object]'</code> | tuple of (aligned_collection, LocalAlignment_model).

**Examples:**

```pycon
>>> # Basic searchlight alignment
>>> aligned_bc = bc.align(method='procrustes', radius_mm=10.0)
```

```pycon
>>> # Piecewise alignment with parcellation
>>> aligned_bc = bc.align(
...     scheme='piecewise',
...     parcellation=parcellation_img,
...     method='srm',
...     n_features=50
... )
```

```pycon
>>> # Fit/transform workflow for train/test split
>>> aligned_train, model = train_bc.align(return_model=True)
>>> aligned_test = model.transform(test_data_list)
```

```pycon
>>> # GPU-accelerated alignment
>>> aligned_bc = bc.align(parallel='gpu')
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

Based on Bazeille et al. 2021 "An empirical evaluation of functional
alignment using inter-subject decoding". Center-only aggregation is
used for searchlight to preserve local orthogonality of transforms.

</details>

<details class="see-also" open markdown="1">
<summary>See Also</summary>

nltools.algorithms.alignment.LocalAlignment: Underlying alignment class.

</details>

###### `nltools.data.collection.BrainCollection.anova`

```python
anova(groups: str | list | np.ndarray) -> tuple['BrainData', 'BrainData']
```

One-way ANOVA across groups defined by metadata.

Tests whether group means differ significantly. This is the
voxel-wise equivalent of scipy.stats.f_oneway.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`groups` | <code>[str](#str) \| [list](#list) \| [ndarray](#numpy.ndarray)</code> | Group assignment for each image. Can be: - str: Column name in metadata - list/array: Group labels of length n_images | *required*

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)['BrainData', 'BrainData']</code> | Tuple of (F_stat, p_value) as BrainData objects.

**Examples:**

```pycon
>>> # Groups from metadata column
>>> f_stat, p_val = bc.anova('condition')
```

```pycon
>>> # Explicit group labels
>>> groups = ['control'] * 10 + ['patient'] * 15
>>> f_stat, p_val = bc.anova(groups)
```

###### `nltools.data.collection.BrainCollection.compute_contrasts`

```python
compute_contrasts(contrasts: 'str | dict | np.ndarray | list') -> 'BrainCollection | dict[str, BrainCollection]'
```

Compute contrasts from fitted GLM beta coefficients.

Applies contrast weights to each subject's betas and returns a
BrainCollection of contrast values suitable for group-level analysis.

Must be called on a BrainCollection created by fit_glm() which has
the _design_columns attribute set.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`contrasts` | <code>'str \| dict \| np.ndarray \| list'</code> | Can be: - str: Contrast string using column names, e.g., "face - house" - dict: Multiple contrasts, e.g., {"main": "face - house", "avg": [0.5, 0.5]} - array/list: Numeric contrast vector, e.g., [1, -1] | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection where each BrainData has shape (n_voxels,) containing
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | the contrast values. If dict input, returns dict of BrainCollections.

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

###### `nltools.data.collection.BrainCollection.cv`

```python
cv(k: int | None = None, scheme: str = 'kfold', split_by: str | None = None, groups: np.ndarray | None = None, random_state: int | None = None, **kwargs: int | None) -> 'BrainCollectionPipeline'
```

Create a cross-validation pipeline for multi-subject analysis.

Returns a pipeline object that enables fluent, chainable transforms
with cross-validation across subjects or runs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`k` | <code>[int](#int) \| None</code> | Number of folds (for kfold scheme). Defaults to 5. | <code>None</code>
`scheme` | <code>[str](#str)</code> | CV scheme type. Options: - 'kfold': k-fold cross-validation on pooled data - 'loso': leave-one-subject-out (one image held out per fold) - 'loro': leave-one-run-out (requires groups) | <code>'kfold'</code>
`split_by` | <code>[str](#str) \| None</code> | Metadata column for group splits. If provided and groups is None, gets groups from self.metadata[split_by]. | <code>None</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Explicit group labels for CV splits. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`**kwargs` |  | Additional arguments passed to CVScheme. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainCollectionPipeline` | <code>'BrainCollectionPipeline'</code> | Pipeline for method chaining.

**Examples:**

```pycon
>>> # Leave-one-subject-out classification
>>> result = bc.cv(scheme='loso').normalize().predict(subject_labels, algorithm='svm')
>>> print(f"Mean accuracy: {result.mean_score:.2%}")
```

```pycon
>>> # With preprocessing
>>> result = (bc
...     .cv(scheme='loso')
...     .normalize()
...     .reduce(n_components=50)
...     .predict(labels))
```

```pycon
>>> # Run-based CV with metadata
>>> result = bc.cv(scheme='loro', split_by='run').predict(y)
```

<details class="see-also" open markdown="1">
<summary>See Also</summary>

BrainCollectionPipeline: For available transforms and terminals.
CVScheme: For CV scheme configuration details.

</details>

###### `nltools.data.collection.BrainCollection.detrend`

```python
detrend(method: str = 'linear', n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Remove trend from each image.

Delegates to BrainData.detrend() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | 'linear' or 'constant'. | <code>'linear'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with detrended images.

**Examples:**

```pycon
>>> bc.detrend()  # Remove linear trend
>>> bc.detrend(method='constant')  # Remove mean only
```

###### `nltools.data.collection.BrainCollection.filter`

```python
filter(predicate: Callable | list | np.ndarray | 'pd.Series') -> 'BrainCollection'
```

Filter collection by predicate.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`predicate` | <code>[Callable](#collections.abc.Callable) \| [list](#list) \| [ndarray](#numpy.ndarray) \| 'pd.Series'</code> | Filter condition. Can be: - callable: fn(BrainData) → bool - list/ndarray: Boolean mask of length n_images - pd.Series: Boolean series (index ignored) | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with subset of images matching predicate.

**Examples:**

```pycon
>>> # Filter by callable
>>> bc.filter(lambda bd: bd.data.mean() > 0)
```

```pycon
>>> # Filter by boolean mask
>>> mask = [True, False, True]
>>> bc.filter(mask)
```

```pycon
>>> # Filter by metadata condition
>>> bc.filter(bc.metadata['group'] == 'control')
```

###### `nltools.data.collection.BrainCollection.fit`

```python
fit(model: str, X: 'pd.DataFrame | np.ndarray | str | list', cv: int | None = None, scale: bool = True, scale_value: float = 100.0, show_progress: bool = True, **kwargs: bool) -> 'FittedBrainCollection'
```

Fit a model to each subject in the collection.

Unified fitting method that shadows BrainData.fit() API for multi-subject
analysis. Dispatches to model-specific implementations based on the
model parameter.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model` | <code>[str](#str)</code> | Model type - 'glm' or 'ridge' | *required*
`X` | <code>'pd.DataFrame \| np.ndarray \| str \| list'</code> | Design/feature matrix. Can be: - pd.DataFrame/DesignMatrix: Shared (used for all subjects) - np.ndarray: Shared array (used for all subjects) - str: Column name in metadata pointing to file paths - list: Per-subject list of DataFrames/arrays/paths | *required*
`cv` | <code>[int](#int) \| None</code> | Cross-validation folds (Ridge only). Default is None for GLM, 5 for Ridge when output='scores'. | <code>None</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`**kwargs` |  | Model-specific arguments passed to _fit_glm or _fit_ridge: - GLM: return_stats, save - Ridge: alpha, output, save, backend, random_state | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'FittedBrainCollection'</code> | FittedBrainCollection wrapping the fitted results. Supports:
<code>'FittedBrainCollection'</code> | - ``.results``: Access underlying BrainCollection(s) directly
<code>'FittedBrainCollection'</code> | - ``.betas``: Convenience accessor for beta coefficients (GLM)
<code>'FittedBrainCollection'</code> | - ``.pool()``: Aggregate across subjects for group analysis
<code>'FittedBrainCollection'</code> | The underlying results contain:
<code>'FittedBrainCollection'</code> | - GLM: Beta coefficients (n_regressors, n_voxels) per subject
<code>'FittedBrainCollection'</code> | - Ridge: Scores or weights depending on 'output' kwarg
<code>'FittedBrainCollection'</code> | If return_stats (GLM) or output='both' (Ridge), results is a dict.

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

###### `nltools.data.collection.BrainCollection.fit_from_events`

```python
fit_from_events(events: pd.DataFrame, t_r: float, confounds: str | list[pd.DataFrame | Path | str] | None = None, confound_columns: list[str] | None = None, hrf_model: str = 'spm', drift_model: str = 'cosine', high_pass: float = 0.01, scale: bool = True, scale_value: float = 100.0, return_stats: list[str] | None = None, return_residuals: bool = False, save: dict[str, str] | None = None, show_progress: bool = True, by_run: bool = False, run_column: str = 'run', run_lengths: int | list[int] | None = None) -> 'BrainCollection | dict[str, BrainCollection]'
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
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`by_run` | <code>[bool](#bool)</code> | If True, fit GLM separately per run and return run-level betas. This enables MVPA decoding with leave-one-run-out CV. | <code>False</code>
`run_column` | <code>[str](#str)</code> | Column name in events identifying runs (default 'run'). | <code>'run'</code>
`run_lengths` | <code>[int](#int) \| [list](#list)[[int](#int)] \| None</code> | Number of TRs per run. Required when by_run=True. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection of beta coefficients for task regressors.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | If return_stats specified, returns dict with keys 'betas', 't', etc.

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

###### `nltools.data.collection.BrainCollection.fit_glm`

```python
fit_glm(events: pd.DataFrame, t_r: float, confounds: str | list[pd.DataFrame | Path | str] | None = None, confound_columns: list[str] | None = None, hrf_model: str = 'spm', drift_model: str = 'cosine', high_pass: float = 0.01, scale: bool = True, scale_value: float = 100.0, return_stats: list[str] | None = None, return_residuals: bool = False, save: dict[str, str] | None = None, show_progress: bool = True, by_run: bool = False, run_column: str = 'run', run_lengths: int | list[int] | None = None) -> 'BrainCollection | dict[str, BrainCollection]'
```

Fit GLM to each subject in collection.

Memory-efficient first-level GLM analysis that processes subjects
one at a time. Returns a BrainCollection of beta coefficients for
task regressors (confounds and drift terms are fit but not returned).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
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
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`by_run` | <code>[bool](#bool)</code> | If True, fit GLM separately per run and return run-level betas. This enables MVPA decoding with leave-one-run-out CV. Each subject will have (n_runs * n_conditions, n_voxels) betas. | <code>False</code>
`run_column` | <code>[str](#str)</code> | Column name in events identifying runs (default 'run'). | <code>'run'</code>
`run_lengths` | <code>[int](#int) \| [list](#list)[[int](#int)] \| None</code> | Number of TRs per run. Required when by_run=True.<br>- int: All runs have same length - list of int: Different length per run - None: Will attempt to infer equal-length runs from total scans | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection where each BrainData has shape:
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | - (n_task_regressors, n_voxels) if by_run=False (default)
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | - (n_runs * n_task_regressors, n_voxels) if by_run=True
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | The ``._design_columns`` attribute stores task regressor names.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | If by_run=True, also stores ``._condition_labels`` and ``._run_labels``.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | If return_stats specified, returns dict with keys 'betas', 't', etc.

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

###### `nltools.data.collection.BrainCollection.fit_ridge`

```python
fit_ridge(X: 'np.ndarray | str | list', alpha: float | str = 1.0, cv: int | None = 5, scale: bool = True, scale_value: float = 100.0, output: str = 'scores', save: dict[str, str] | None = None, show_progress: bool = True, **ridge_kwargs: bool) -> 'BrainCollection | dict[str, BrainCollection]'
```

Fit ridge regression to each subject in collection.

Memory-efficient encoding model fitting that processes subjects one at a
time. Default behavior returns cross-validated R² scores per voxel,
suitable for group-level inference on encoding model performance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>'np.ndarray \| str \| list'</code> | Feature matrix. Can be: - np.ndarray: Shared features (n_samples, n_features) used for all subjects - str: Column name in metadata pointing to feature file paths - list: List of arrays/DataFrames, one per subject | *required*
`alpha` | <code>[float](#float) \| [str](#str)</code> | Ridge regularization parameter. Can be: - float: Fixed regularization strength - 'auto': Use cross-validation to select optimal alpha | <code>1.0</code>
`cv` | <code>[int](#int) \| None</code> | Cross-validation folds for computing scores. Default is 5. Required when output='scores' or 'both'. Set to None only when output='weights'. | <code>5</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`output` | <code>[str](#str)</code> | What to return. Options: - 'scores': CV R² scores per voxel (default, for encoding workflow) - 'weights': Model weights (n_features, n_voxels) - 'both': Dict with both 'scores' and 'weights' | <code>'scores'</code>
`save` | <code>[dict](#dict)[[str](#str), [str](#str)] \| None</code> | Dict mapping output type to path template, e.g. ``{'weights': 'output/{subject}_weights.nii.gz', 'scores': 'output/{subject}_scores.nii.gz'}``. Supports {subject}, {session}, {idx}, and other metadata columns. | <code>None</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`**ridge_kwargs` |  | Additional arguments passed to Ridge model (e.g., backend='torch', random_state=42). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection of scores or weights, or dict with both if output='both'.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | Each BrainData will have ``cv_results_`` attribute when cv is used.

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

###### `nltools.data.collection.BrainCollection.from_bids`

```python
from_bids(layout: Any, mask: nib.Nifti1Image | Path | str, *, task: str | None = None, subject: str | list[str] | None = None, session: str | list[str] | None = None, run: int | list[int] | None = None, space: str | None = None, suffix: str = 'bold', extension: str = 'nii.gz', **bids_filters: str) -> 'BrainCollection'
```

Create BrainCollection from a BIDS dataset.

Requires pybids to be installed: `pip install pybids`

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`layout` | <code>[Any](#typing.Any)</code> | pybids BIDSLayout object or path to BIDS dataset. | *required*
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str)</code> | Shared mask (required). | *required*
`task` | <code>[str](#str) \| None</code> | BIDS task filter. | <code>None</code>
`subject` | <code>[str](#str) \| [list](#list)[[str](#str)] \| None</code> | Subject ID(s) to include. | <code>None</code>
`session` | <code>[str](#str) \| [list](#list)[[str](#str)] \| None</code> | Session ID(s) to include. | <code>None</code>
`run` | <code>[int](#int) \| [list](#list)[[int](#int)] \| None</code> | Run number(s) to include. | <code>None</code>
`space` | <code>[str](#str) \| None</code> | BIDS space filter (e.g., 'MNI152NLin2009cAsym'). | <code>None</code>
`suffix` | <code>[str](#str)</code> | BIDS suffix (default 'bold'). | <code>'bold'</code>
`extension` | <code>[str](#str)</code> | File extension (default 'nii.gz'). | <code>'nii.gz'</code>
`**bids_filters` |  | Additional BIDS entity filters. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with metadata extracted from BIDS entities.

**Examples:**

```pycon
>>> bc = BrainCollection.from_bids(
...     '/data/bids_dataset',
...     mask='2mm-MNI152-2009c',
...     task='rest',
...     space='MNI152NLin2009cAsym'
... )
```

###### `nltools.data.collection.BrainCollection.from_glob`

```python
from_glob(pattern: str, mask: nib.Nifti1Image | Path | str, *, pattern_groups: dict[str, int] | str | None = None, sort: bool = True) -> 'BrainCollection'
```

Create BrainCollection from glob pattern.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`pattern` | <code>[str](#str)</code> | Glob pattern (e.g., ``'/data/*/func/*_bold.nii.gz'``). | *required*
`mask` | <code>[Nifti1Image](#nibabel.Nifti1Image) \| [Path](#pathlib.Path) \| [str](#str)</code> | Shared mask (required). | *required*
`pattern_groups` | <code>[dict](#dict)[[str](#str), [int](#int)] \| [str](#str) \| None</code> | Regex pattern with named groups for metadata extraction. Example: ``r'sub-(?P<subject>\w+)/.*run-(?P<run>\d+)'`` | <code>None</code>
`sort` | <code>[bool](#bool)</code> | Sort files alphabetically (default True). | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with optional metadata from pattern groups.

**Examples:**

```pycon
>>> bc = BrainCollection.from_glob(
...     '/data/sub-*/func/*_bold.nii.gz',
...     mask=mask,
...     pattern_groups=r'sub-(?P<subject>\w+)'
... )
```

###### `nltools.data.collection.BrainCollection.from_stacked`

```python
from_stacked(brain_data: 'BrainData', splits: list[int] | None = None, n_images: int | None = None) -> 'BrainCollection'
```

Create BrainCollection by splitting a stacked BrainData.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_data` | <code>'BrainData'</code> | BrainData with shape (n_total_obs, n_voxels). | *required*
`splits` | <code>[list](#list)[[int](#int)] \| None</code> | List of observation counts per image. Must sum to n_total_obs. | <code>None</code>
`n_images` | <code>[int](#int) \| None</code> | Number of images (splits evenly). Mutually exclusive with splits. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with data split according to specification.

**Examples:**

```pycon
>>> # Split evenly into 3 images
>>> bc = BrainCollection.from_stacked(bd, n_images=3)
```

```pycon
>>> # Split with explicit counts
>>> bc = BrainCollection.from_stacked(bd, splits=[100, 100, 150])
```

###### `nltools.data.collection.BrainCollection.isc`

```python
isc(method: str = 'loo', roi_mask: 'nib.Nifti1Image | Path | str | None' = None, radius: float | None = 6.0, metric: str = 'median', parallel: str = 'cpu', n_jobs: int = -1, show_progress: bool = True) -> dict
```

Compute intersubject correlation (ISC) across the collection.

ISC measures the similarity of brain responses across subjects,
computed by correlating each subject's timeseries with others.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ISC computation method: - 'loo': Leave-one-out (correlate each subject with mean of others) - 'pairwise': All pairwise correlations between subjects | <code>'loo'</code>
`roi_mask` | <code>'nib.Nifti1Image \| Path \| str \| None'</code> | If provided, compute ISC per ROI. Can be: - NIfTI image with integer labels (atlas/parcellation) - Path to parcellation file | <code>None</code>
`radius` | <code>[float](#float) \| None</code> | Searchlight radius in mm. If None, use voxelwise mode. Ignored if roi_mask is provided. | <code>6.0</code>
`metric` | <code>[str](#str)</code> | Summary statistic for aggregating ISC values: - 'median': Robust to outliers (default) - 'mean': Fisher z-transformed mean | <code>'median'</code>
`parallel` | <code>[str](#str)</code> | Parallelization method ('cpu', 'gpu', or None). | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs (-1 = all cores). | <code>-1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during extraction. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with: - 'isc': BrainData with ISC values - 'method': ISC method used ('loo' or 'pairwise') - 'extraction': Extraction mode ('roi', 'searchlight', 'voxelwise') - 'n_subjects': Number of subjects - 'extraction_info': Dict with extraction metadata

**Examples:**

```pycon
>>> # ROI-based ISC using atlas
>>> result = bc.isc(roi_mask="atlas.nii.gz")
>>> result['isc'].plot()
```

```pycon
>>> # Searchlight ISC
>>> result = bc.isc(radius=10.0)
```

```pycon
>>> # Voxelwise ISC
>>> result = bc.isc(radius=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

For permutation testing, see BrainCollection.isc_test() (requires
discussion of statistical methodology first).

</details>

###### `nltools.data.collection.BrainCollection.isc_test`

```python
isc_test(method: str = 'loo', roi_mask: 'nib.Nifti1Image | Path | str | None' = None, radius: float | None = 6.0, n_permute: int = 5000, permutation_method: str = 'bootstrap', metric: str = 'median', tail: int = 2, ci_percentile: float = 95, parallel: str = 'cpu', n_jobs: int = -1, random_state: int | None = None, return_null: bool = False, show_progress: bool = True) -> dict
```

Compute ISC with permutation testing for statistical inference.

This method combines ISC computation with permutation testing to
determine statistical significance. It uses the same extraction
pipeline as isc() and wraps isc_permutation_test().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | ISC computation method: - 'loo': Leave-one-out (correlate each subject with mean of others) - 'pairwise': All pairwise correlations between subjects | <code>'loo'</code>
`roi_mask` | <code>'nib.Nifti1Image \| Path \| str \| None'</code> | If provided, compute ISC per ROI. Can be: - NIfTI image with integer labels (atlas/parcellation) - Path to parcellation file | <code>None</code>
`radius` | <code>[float](#float) \| None</code> | Searchlight radius in mm. If None, use voxelwise mode. Ignored if roi_mask is provided. | <code>6.0</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations/bootstrap iterations. Default 5000. | <code>5000</code>
`permutation_method` | <code>[str](#str)</code> | Method for null distribution:<br>- 'bootstrap': Subject-wise bootstrap (default, Chen et al. 2016).   Tests whether observed ISC differs from random groupings. - 'circle_shift': Circular time-shift (preserves autocorrelation).   Tests for temporally-locked shared signal. - 'phase_randomize': FFT phase randomization (preserves power spectrum).   Tests for nonlinear temporal coupling. | <code>'bootstrap'</code>
`metric` | <code>[str](#str)</code> | Summary statistic for aggregating ISC values: - 'median': Robust to outliers (default) - 'mean': Fisher z-transformed mean | <code>'median'</code>
`tail` | <code>[int](#int)</code> | One-tailed (1) or two-tailed (2) test. Default 2. | <code>2</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95). Default 95. | <code>95</code>
`parallel` | <code>[str](#str)</code> | Parallelization method ('cpu', 'gpu', or None). | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs (-1 = all cores). | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, include null distribution in results. | <code>False</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during extraction and permutation. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with: - 'isc': BrainData with ISC values - 'p': BrainData with p-values (Phipson-Smyth corrected) - 'ci': Tuple of (lower, upper) BrainData confidence intervals - 'method': ISC method used ('loo' or 'pairwise') - 'permutation_method': Permutation method used - 'extraction': Extraction mode ('roi', 'searchlight', 'voxelwise') - 'n_subjects': Number of subjects - 'n_permute': Number of permutations - 'null_dist': (optional) Null distribution array if return_null=True

**Examples:**

```pycon
>>> # ROI-based ISC with permutation testing
>>> result = bc.isc_test(roi_mask="atlas.nii.gz", n_permute=5000)
>>> sig_mask = result['p'].data < 0.05
>>> print(f"Significant ROIs: {sig_mask.sum()}")
```

```pycon
>>> # Searchlight ISC testing
>>> result = bc.isc_test(radius=10.0)
>>> result['isc'].plot()  # Show ISC values
>>> result['p'].plot()    # Show p-values
```

```pycon
>>> # Voxelwise with phase randomization (tests temporal coupling)
>>> result = bc.isc_test(
...     radius=None,
...     permutation_method='phase_randomize',
...     parallel='gpu'
... )
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Bootstrap (default) is recommended for standard ISC inference
  (Chen et al. 2016). It tests whether ISC is significant at
  the group level.
- Circle_shift and phase_randomize are more specialized - they
  test for temporally-structured shared signal beyond what's
  explained by autocorrelation or spectral structure alone.
- For large voxelwise analyses, bootstrap is much faster as it
  resamples pre-computed values rather than recomputing ISC.

</details>

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., et al. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

###### `nltools.data.collection.BrainCollection.iter_batches`

```python
iter_batches(batch_size: int, axis: int = 0, show_progress: bool = True) -> Generator['BrainCollection', None, None]
```

Iterate in batches along axis.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`batch_size` | <code>[int](#int)</code> | Number of items per batch. | *required*
`axis` | <code>[int](#int)</code> | Axis to batch along: - 0: Batches of images (default) - 1: Batches of timepoints (within each image) | <code>0</code>
`show_progress` | <code>[bool](#bool)</code> | Show tqdm progress bar. | <code>True</code>

**Yields:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection for each batch.

**Examples:**

```pycon
>>> # Batch over images
>>> for batch in bc.iter_batches(batch_size=5):
...     process(batch)  # batch is BrainCollection with 5 images
```

```pycon
>>> # Batch over time
>>> for batch in bc.iter_batches(batch_size=10, axis=1):
...     process(batch)  # batch has 10 timepoints per image
```

###### `nltools.data.collection.BrainCollection.load`

```python
load(indices: list[int] | None = None) -> 'BrainCollection'
```

Load specified images into memory.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`indices` | <code>[list](#list)[[int](#int)] \| None</code> | List of indices to load. If None, loads all. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | self (for chaining)

###### `nltools.data.collection.BrainCollection.map`

```python
map(fn: Callable, axis: int | str = 0, n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Apply function across specified axis.

This is the general-purpose transformation method. For common operations,
use convenience methods like standardize(), smooth(), etc.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fn` | <code>[Callable](#collections.abc.Callable)</code> | Function to apply. Signature depends on axis: - axis=0: fn(BrainData) → BrainData (per image) - axis=1: fn(BrainData) → BrainData (per timepoint slice) - axis=2: fn(ndarray[n_obs]) → ndarray (per voxel timeseries) | *required*
`axis` | <code>[int](#int) \| [str](#str)</code> | Axis to iterate over: - 0 or 'images': Apply fn to each image independently - 1 or 'time': Apply fn to each timepoint across images - 2 or 'voxels': Apply fn to each voxel timeseries per image | <code>0</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. -1 for all cores. Default 1. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show tqdm progress bar. Default True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with transformed data.

**Examples:**

```pycon
>>> # Per-image operation
>>> bc.map(lambda bd: bd.standardize())
```

```pycon
>>> # Per-voxel timeseries (e.g., detrend each voxel)
>>> from scipy.signal import detrend
>>> bc.map(detrend, axis=2)
```

```pycon
>>> # Parallel processing
>>> bc.map(expensive_fn, n_jobs=-1)
```

###### `nltools.data.collection.BrainCollection.max`

```python
max(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute maximum along axis. See mean() for details.

###### `nltools.data.collection.BrainCollection.mean`

```python
mean(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute mean along axis.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>[int](#int) \| [str](#str) \| [tuple](#tuple)[[int](#int), ...]</code> | Axis or axes to aggregate: - 0 or 'images': Mean across images -> BrainData (n_obs, n_voxels) - 1 or 'time': Mean across time -> BrainCollection (n_images, n_voxels) - 2 or 'voxels': Mean across voxels -> np.ndarray (n_images, n_obs) - (0, 1): Mean across images and time -> BrainData (n_voxels,) | <code>0</code>
`batch_size` | <code>[int](#int) \| None</code> | Number of images to process at once (for memory efficiency). If None, uses streaming algorithm. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainData \| BrainCollection \| np.ndarray'</code> | BrainData, BrainCollection, or np.ndarray depending on axis.

**Examples:**

```pycon
>>> bc.mean(axis=0)  # Mean across subjects
>>> bc.mean(axis='images')  # Same as above
>>> bc.mean(axis=1)  # Mean across time per subject
>>> bc.mean(axis=(0, 1))  # Grand mean
```

###### `nltools.data.collection.BrainCollection.median`

```python
median(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute median along axis. See mean() for details.

###### `nltools.data.collection.BrainCollection.memory_estimate`

```python
memory_estimate() -> str
```

Estimate memory usage for loading all images.

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | Human-readable string like "12.4 GB total (1.2 GB per image avg)"

###### `nltools.data.collection.BrainCollection.min`

```python
min(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute minimum along axis. See mean() for details.

###### `nltools.data.collection.BrainCollection.permutation_test`

```python
permutation_test(n_permute: int = 5000, tail: int = 2, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None, return_null: bool = False) -> dict
```

One-sample permutation test across images (sign-flipping).

Tests whether the mean across images is significantly different from
zero using sign-flipping permutation. More robust than parametric
t-test for non-normal distributions.

This is a collection-level interface to
nltools.algorithms.inference.one_sample_permutation_test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000). | <code>5000</code>
`tail` | <code>[int](#int)</code> | Test type - 1 for one-tailed, 2 for two-tailed (default). | <code>2</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization method: - 'cpu': CPU parallelization via joblib (default) - 'gpu': GPU acceleration via PyTorch - None: Single-threaded (for debugging) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores (default: -1 = all cores). | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget (default: 4.0 GB). | <code>4.0</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, include null distribution in result. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | dict with keys: - 'mean': BrainData with observed mean across images - 'p': BrainData with p-values - 'null_dist': np.ndarray (if return_null=True) - 'parallel': parallelization method used

**Examples:**

```pycon
>>> result = bc.permutation_test(n_permute=5000)
>>> mean_bd, p_bd = result['mean'], result['p']
```

```pycon
>>> # With GPU acceleration
>>> result = bc.permutation_test(parallel='gpu')
```

###### `nltools.data.collection.BrainCollection.permutation_test2`

```python
permutation_test2(other: 'BrainCollection', n_permute: int = 5000, tail: int = 2, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None, return_null: bool = False) -> dict
```

Two-sample permutation test between collections.

Tests whether two collections have different means using group
label permutation. More robust than parametric t-test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` | <code>'BrainCollection'</code> | Another BrainCollection to compare against. | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000). | <code>5000</code>
`tail` | <code>[int](#int)</code> | Test type - 1 for one-tailed, 2 for two-tailed (default). | <code>2</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization method ('cpu', 'gpu', or None). | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores (default: -1 = all cores). | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget (default: 4.0 GB). | <code>4.0</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, include null distribution in result. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | dict with keys: - 'mean_diff': BrainData with observed mean difference - 'p': BrainData with p-values - 'null_dist': np.ndarray (if return_null=True) - 'parallel': parallelization method used

**Examples:**

```pycon
>>> result = patients.permutation_test2(controls)
>>> diff_bd, p_bd = result['mean_diff'], result['p']
```

###### `nltools.data.collection.BrainCollection.predict`

```python
predict(X: 'np.ndarray | str | list | None' = None, y: 'np.ndarray | None' = None, method: str = 'whole_brain', estimator: str = 'svm', cv: str = 5, groups: 'np.ndarray | None' = None, roi_mask: 'np.ndarray | None' = None, radius: float = 10.0, scoring: str = 'accuracy', standardize: bool = True, n_jobs: int = -1, show_progress: bool = True) -> 'BrainCollection'
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
`X` | <code>'np.ndarray \| str \| list \| None'</code> | Features for timeseries prediction. Can be: - np.ndarray: Shared features (same for all subjects) - str: Metadata column with per-subject feature paths - list: Per-subject feature arrays | <code>None</code>
`y` | <code>'np.ndarray \| None'</code> | Labels for MVPA decoding. If None and _condition_labels exists, will use stored condition labels (from fit_glm with by_run=True). | <code>None</code>
`method` | <code>[str](#str)</code> | MVPA method - 'whole_brain', 'searchlight', or 'roi'. | <code>'whole_brain'</code>
`estimator` |  | Classifier - 'svm', 'logistic', 'ridge', 'lda' or sklearn estimator instance. | <code>'svm'</code>
`cv` |  | Cross-validation strategy. If None and _run_labels exists, uses leave-one-group-out with run labels. | <code>5</code>
`groups` | <code>'np.ndarray \| None'</code> | Group labels for GroupKFold/LeaveOneGroupOut. If None and _run_labels exists, uses stored run labels. | <code>None</code>
`roi_mask` |  | Mask for ROI-based MVPA. Required if method='roi'. | <code>None</code>
`radius` | <code>[float](#float)</code> | Searchlight radius in mm (default 10.0). | <code>10.0</code>
`scoring` | <code>[str](#str)</code> | Scoring metric (default 'accuracy'). | <code>'accuracy'</code>
`standardize` | <code>[bool](#bool)</code> | If True, standardize features before classification. | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Parallel jobs for searchlight (-1 = all cores). | <code>-1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with prediction results:
<code>'BrainCollection'</code> | - For timeseries: (n_timepoints, n_voxels) predicted responses
<code>'BrainCollection'</code> | - For MVPA: (1, n_voxels) accuracy values

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

###### `nltools.data.collection.BrainCollection.select_feature`

```python
select_feature(feature: 'int | str') -> 'BrainCollection'
```

Select a single feature's weights across all subjects.

Used after fit_ridge() to extract weights for a specific feature
for group-level analysis (e.g., t-test on feature weights).

Must be called on a BrainCollection created by fit_ridge() where
each subject has shape (n_features, n_voxels).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`feature` | <code>'int \| str'</code> | Feature to select. Can be: - int: Feature index (0-based) - str: Feature name (requires _feature_names attribute) | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection where each BrainData has shape (n_voxels,)
<code>'BrainCollection'</code> | containing the weights for the specified feature.

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

###### `nltools.data.collection.BrainCollection.smooth`

```python
smooth(fwhm: float, n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Spatially smooth each image.

Delegates to BrainData.smooth() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fwhm` | <code>[float](#float)</code> | Full width at half maximum of Gaussian kernel in mm. | *required*
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with smoothed images.

**Examples:**

```pycon
>>> bc.smooth(fwhm=6)  # 6mm FWHM smoothing
```

###### `nltools.data.collection.BrainCollection.standardize`

```python
standardize(axis: int = 0, method: str = 'center', n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Standardize each image.

Delegates to BrainData.standardize() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>[int](#int)</code> | Axis for standardization within each image: - 0: Standardize across observations (time) per voxel - 1: Standardize across voxels per observation | <code>0</code>
`method` | <code>[str](#str)</code> | 'center' (subtract mean) or 'zscore' (subtract mean, divide std) | <code>'center'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with standardized images.

**Examples:**

```pycon
>>> bc.standardize()  # Center each image across time
>>> bc.standardize(method='zscore')  # Z-score each image
>>> bc.standardize(axis=1)  # Standardize across voxels
```

###### `nltools.data.collection.BrainCollection.std`

```python
std(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute standard deviation along axis. See mean() for details.

###### `nltools.data.collection.BrainCollection.sum`

```python
sum(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute sum along axis. See mean() for details.

###### `nltools.data.collection.BrainCollection.threshold`

```python
threshold(upper: float | str | None = None, lower: float | str | None = None, binarize: bool = False, coerce_nan: bool = True, n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Threshold each image.

Delegates to BrainData.threshold() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`upper` | <code>[float](#float) \| [str](#str) \| None</code> | Upper cutoff. String interpreted as percentile. | <code>None</code>
`lower` | <code>[float](#float) \| [str](#str) \| None</code> | Lower cutoff. String interpreted as percentile. | <code>None</code>
`binarize` | <code>[bool](#bool)</code> | Return binary mask. | <code>False</code>
`coerce_nan` | <code>[bool](#bool)</code> | Replace NaN with 0. | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with thresholded images.

**Examples:**

```pycon
>>> bc.threshold(lower=0)  # Zero out negative values
>>> bc.threshold(upper='95%')  # Keep top 5%
>>> bc.threshold(lower=2, binarize=True)  # Binary mask
```

###### `nltools.data.collection.BrainCollection.to_list`

```python
to_list() -> list['BrainData']
```

Return list of BrainData objects.

Loads all items if not already loaded.

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)['BrainData']</code> | List of BrainData objects.

###### `nltools.data.collection.BrainCollection.to_stacked`

```python
to_stacked() -> 'BrainData'
```

Stack all into single BrainData (n_total_obs, n_voxels).

**Returns:**

Type | Description
---- | -----------
<code>'BrainData'</code> | Single BrainData with all observations concatenated.

**Examples:**

```pycon
>>> bc = BrainCollection([bd1, bd2, bd3], mask=mask)
>>> stacked = bc.to_stacked()
>>> stacked.shape
(300, 50000)  # 3 images * 100 obs each
```

###### `nltools.data.collection.BrainCollection.to_tensor`

```python
to_tensor(batch_size: int | None = None) -> np.ndarray | Generator[np.ndarray, None, None]
```

Convert to numpy array (n_images, n_obs, n_voxels).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`batch_size` | <code>[int](#int) \| None</code> | If specified, returns generator yielding batches. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray) \| [Generator](#typing.Generator)[[ndarray](#numpy.ndarray), None, None]</code> | Full tensor if batch_size is None, otherwise generator.

**Examples:**

```pycon
>>> tensor = bc.to_tensor()  # Full array
>>> tensor.shape
(3, 100, 50000)
```

```pycon
>>> # Batched iteration
>>> for batch in bc.to_tensor(batch_size=10):
...     process(batch)  # batch.shape = (10, 100, 50000)
```

###### `nltools.data.collection.BrainCollection.ttest`

```python
ttest(popmean: float = 0.0, axis: int | str = 0) -> tuple['BrainData', 'BrainData']
```

One-sample t-test across images.

Tests whether the mean across images is significantly different from
a population mean (default: 0). This is the voxel-wise equivalent of
scipy.stats.ttest_1samp.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`popmean` | <code>[float](#float)</code> | Population mean to test against (default: 0). | <code>0.0</code>
`axis` | <code>[int](#int) \| [str](#str)</code> | Axis to test across. Only axis=0 (images) supported. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainData'</code> | Tuple of (t_stat, p_value) as BrainData objects.
<code>'BrainData'</code> | Both have shape (n_obs, n_voxels) if uniform obs counts.

**Examples:**

```pycon
>>> t_stat, p_val = bc.ttest()  # Test mean != 0
>>> t_stat, p_val = bc.ttest(popmean=0.5)  # Test mean != 0.5
```

```pycon
>>> # Threshold significant voxels
>>> sig_mask = p_val.data < 0.05
```

###### `nltools.data.collection.BrainCollection.ttest2`

```python
ttest2(other: 'BrainCollection', equal_var: bool = True) -> tuple['BrainData', 'BrainData']
```

Two-sample t-test between collections.

Tests whether two collections have different means. This is the
voxel-wise equivalent of scipy.stats.ttest_ind.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` | <code>'BrainCollection'</code> | Another BrainCollection to compare against. | *required*
`equal_var` | <code>[bool](#bool)</code> | If True (default), perform standard t-test assuming equal variances. If False, use Welch's t-test. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)['BrainData', 'BrainData']</code> | Tuple of (t_stat, p_value) as BrainData objects.

**Examples:**

```pycon
>>> t_stat, p_val = patients.ttest2(controls)
>>> t_stat, p_val = group1.ttest2(group2, equal_var=False)  # Welch's
```

###### `nltools.data.collection.BrainCollection.unload`

```python
unload(indices: list[int] | None = None) -> 'BrainCollection'
```

Free memory for specified images (keep paths for reloading).

Only works for items that were originally loaded from paths.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`indices` | <code>[list](#list)[[int](#int)] \| None</code> | List of indices to unload. If None, unloads all possible. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | self (for chaining)

###### `nltools.data.collection.BrainCollection.var`

```python
var(axis: int | str | tuple[int, ...] = 0, batch_size: int | None = None) -> 'BrainData | BrainCollection | np.ndarray'
```

Compute variance along axis. See mean() for details.

###### `nltools.data.collection.BrainCollection.write`

```python
write(directory: str | Path, pattern: str = 'image_{i:04d}.nii.gz', metadata_file: str | None = 'metadata.csv') -> list[Path]
```

Write all images in collection to files.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`directory` | <code>[str](#str) \| [Path](#pathlib.Path)</code> | Output directory path. Will be created if it doesn't exist. | *required*
`pattern` | <code>[str](#str)</code> | Filename pattern with {i} placeholder for image index. Default: "image_{i:04d}.nii.gz" produces image_0000.nii.gz, etc. | <code>'image_{i:04d}.nii.gz'</code>
`metadata_file` | <code>[str](#str) \| None</code> | Optional filename for metadata CSV. Set to None to skip. Default: "metadata.csv" | <code>'metadata.csv'</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[Path](#pathlib.Path)]</code> | List of paths to written files.

**Examples:**

```pycon
>>> bc = BrainCollection([bd1, bd2, bd3], mask=mask)
>>> paths = bc.write("output/")
>>> # Creates: output/image_0000.nii.gz, image_0001.nii.gz, etc.
```

```pycon
>>> # Custom pattern
>>> bc.write("output/", pattern="sub-{i:02d}_bold.nii.gz")
>>> # Creates: output/sub-00_bold.nii.gz, sub-01_bold.nii.gz, etc.
```

```pycon
>>> # With BIDS-style naming using metadata
>>> bc.metadata["filename"] = [f"sub-{s}_bold.nii.gz" for s in subjects]
>>> for i, bd in enumerate(bc):
...     bd.write(f"output/{bc.metadata.loc[i, 'filename']}")
```



### Functions

### Modules#### `nltools.data.collection.constructors`

Constructor functions for BrainCollection.

Standalone functions that create BrainCollection instances from various sources
(BIDS datasets, glob patterns, stacked BrainData).

**Functions:**

Name | Description
---- | -----------
[`from_bids`](#nltools.data.collection.constructors.from_bids) | Create BrainCollection from a BIDS dataset.
[`from_glob`](#nltools.data.collection.constructors.from_glob) | Create BrainCollection from glob pattern.
[`from_stacked`](#nltools.data.collection.constructors.from_stacked) | Create BrainCollection by splitting a stacked BrainData.



##### Classes

##### Functions###### `nltools.data.collection.constructors.from_bids`

```python
from_bids(layout: Any, mask: 'nib.Nifti1Image | Path | str', *, task: str | None = None, subject: str | list[str] | None = None, session: str | list[str] | None = None, run: int | list[int] | None = None, space: str | None = None, suffix: str = 'bold', extension: str = 'nii.gz', **bids_filters: str) -> 'BrainCollection'
```

Create BrainCollection from a BIDS dataset.

Requires pybids to be installed: `pip install pybids`

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`layout` | <code>[Any](#typing.Any)</code> | pybids BIDSLayout object or path to BIDS dataset. | *required*
`mask` | <code>'nib.Nifti1Image \| Path \| str'</code> | Shared mask (required). | *required*
`task` | <code>[str](#str) \| None</code> | BIDS task filter. | <code>None</code>
`subject` | <code>[str](#str) \| [list](#list)[[str](#str)] \| None</code> | Subject ID(s) to include. | <code>None</code>
`session` | <code>[str](#str) \| [list](#list)[[str](#str)] \| None</code> | Session ID(s) to include. | <code>None</code>
`run` | <code>[int](#int) \| [list](#list)[[int](#int)] \| None</code> | Run number(s) to include. | <code>None</code>
`space` | <code>[str](#str) \| None</code> | BIDS space filter (e.g., 'MNI152NLin2009cAsym'). | <code>None</code>
`suffix` | <code>[str](#str)</code> | BIDS suffix (default 'bold'). | <code>'bold'</code>
`extension` | <code>[str](#str)</code> | File extension (default 'nii.gz'). | <code>'nii.gz'</code>
`**bids_filters` |  | Additional BIDS entity filters. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with metadata extracted from BIDS entities.

**Examples:**

```pycon
>>> bc = from_bids(
...     '/data/bids_dataset',
...     mask='2mm-MNI152-2009c',
...     task='rest',
...     space='MNI152NLin2009cAsym'
... )
```

###### `nltools.data.collection.constructors.from_glob`

```python
from_glob(pattern: str, mask: 'nib.Nifti1Image | Path | str', *, pattern_groups: 'dict[str, int] | str | None' = None, sort: bool = True) -> 'BrainCollection'
```

Create BrainCollection from glob pattern.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`pattern` | <code>[str](#str)</code> | Glob pattern (e.g., ``'/data/*/func/*_bold.nii.gz'``). | *required*
`mask` | <code>'nib.Nifti1Image \| Path \| str'</code> | Shared mask (required). | *required*
`pattern_groups` | <code>'dict[str, int] \| str \| None'</code> | Regex pattern with named groups for metadata extraction. Example: ``r'sub-(?P<subject>\w+)/.*run-(?P<run>\d+)'`` | <code>None</code>
`sort` | <code>[bool](#bool)</code> | Sort files alphabetically (default True). | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with optional metadata from pattern groups.

**Examples:**

```pycon
>>> bc = from_glob(
...     '/data/sub-*/func/*_bold.nii.gz',
...     mask=mask,
...     pattern_groups=r'sub-(?P<subject>\w+)'
... )
```

###### `nltools.data.collection.constructors.from_stacked`

```python
from_stacked(brain_data: 'BrainData', splits: list[int] | None = None, n_images: int | None = None) -> 'BrainCollection'
```

Create BrainCollection by splitting a stacked BrainData.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_data` | <code>'BrainData'</code> | BrainData with shape (n_total_obs, n_voxels). | *required*
`splits` | <code>[list](#list)[[int](#int)] \| None</code> | List of observation counts per image. Must sum to n_total_obs. | <code>None</code>
`n_images` | <code>[int](#int) \| None</code> | Number of images (splits evenly). Mutually exclusive with splits. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with data split according to specification.

**Examples:**

```pycon
>>> # Split evenly into 3 images
>>> bc = from_stacked(bd, n_images=3)
```

```pycon
>>> # Split with explicit counts
>>> bc = from_stacked(bd, splits=[100, 100, 150])
```

#### `nltools.data.collection.inference`

BrainCollection inference functions.

Extracted from BrainCollection methods — each function takes a BrainCollection
as its first argument (``bc``) instead of ``self``.

**Functions:**

Name | Description
---- | -----------
[`anova`](#nltools.data.collection.inference.anova) | One-way ANOVA across groups defined by metadata.
[`extract_for_isc`](#nltools.data.collection.inference.extract_for_isc) | Extract data for ISC computation.
[`extract_roi`](#nltools.data.collection.inference.extract_roi) | Extract mean signal per ROI.
[`extract_searchlight`](#nltools.data.collection.inference.extract_searchlight) | Extract mean signal per searchlight sphere.
[`extract_voxelwise`](#nltools.data.collection.inference.extract_voxelwise) | Extract raw voxel data.
[`isc`](#nltools.data.collection.inference.isc) | Compute intersubject correlation (ISC) across the collection.
[`isc_test`](#nltools.data.collection.inference.isc_test) | Compute ISC with permutation testing for statistical inference.
[`permutation_test`](#nltools.data.collection.inference.permutation_test) | One-sample permutation test across images (sign-flipping).
[`permutation_test2`](#nltools.data.collection.inference.permutation_test2) | Two-sample permutation test between collections.
[`project_to_brain`](#nltools.data.collection.inference.project_to_brain) | Project ISC values back to brain space.
[`ttest`](#nltools.data.collection.inference.ttest) | One-sample t-test across images.
[`ttest2`](#nltools.data.collection.inference.ttest2) | Two-sample t-test between collections.



##### Classes

##### Functions###### `nltools.data.collection.inference.anova`

```python
anova(bc: 'BrainCollection', groups: str | list | np.ndarray) -> tuple
```

One-way ANOVA across groups defined by metadata.

Tests whether group means differ significantly. This is the
voxel-wise equivalent of scipy.stats.f_oneway.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to test. | *required*
`groups` | <code>[str](#str) \| [list](#list) \| [ndarray](#numpy.ndarray)</code> | Group assignment for each image. Can be: - str: Column name in metadata - list/array: Group labels of length n_images | *required*

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)</code> | Tuple of (F_stat, p_value) as BrainData objects.

**Examples:**

```pycon
>>> # Groups from metadata column
>>> f_stat, p_val = bc.anova('condition')
```

```pycon
>>> # Explicit group labels
>>> groups = ['control'] * 10 + ['patient'] * 15
>>> f_stat, p_val = bc.anova(groups)
```

###### `nltools.data.collection.inference.extract_for_isc`

```python
extract_for_isc(bc: 'BrainCollection', roi_mask: 'nib.Nifti1Image | Path | str | None' = None, radius: float | None = 6.0, show_progress: bool = True) -> tuple[np.ndarray, dict]
```

Extract data for ISC computation.

Memory-efficient extraction that processes one subject at a time.
Returns data in ISC-compatible format: (n_obs, n_subjects, n_features).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to extract from. | *required*
`roi_mask` | <code>'nib.Nifti1Image \| Path \| str \| None'</code> | If provided, extract mean per ROI. Can be: - NIfTI image with integer labels (atlas/parcellation) - Path to parcellation file | <code>None</code>
`radius` | <code>[float](#float) \| None</code> | Searchlight radius in mm. If None, use voxelwise mode. Ignored if roi_mask is provided. | <code>6.0</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during extraction. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[ndarray](#numpy.ndarray), [dict](#dict)]</code> | Tuple of: - extracted_data: Array of shape (n_obs, n_subjects, n_features) - extraction_info: Dict with metadata for projection back:     - 'mode': 'roi', 'searchlight', or 'voxelwise'     - 'n_features': Number of features     - 'roi_mask': ROI mask if mode='roi'     - 'neighborhoods': SphereNeighborhoods if mode='searchlight'

###### `nltools.data.collection.inference.extract_roi`

```python
extract_roi(bc: 'BrainCollection', roi_mask: 'nib.Nifti1Image | Path | str', show_progress: bool = True) -> tuple[np.ndarray, dict]
```

Extract mean signal per ROI.

###### `nltools.data.collection.inference.extract_searchlight`

```python
extract_searchlight(bc: 'BrainCollection', radius: float, show_progress: bool = True) -> tuple[np.ndarray, dict]
```

Extract mean signal per searchlight sphere.

###### `nltools.data.collection.inference.extract_voxelwise`

```python
extract_voxelwise(bc: 'BrainCollection', show_progress: bool = True) -> tuple[np.ndarray, dict]
```

Extract raw voxel data.

###### `nltools.data.collection.inference.isc`

```python
isc(bc: 'BrainCollection', method: str = 'loo', roi_mask: 'nib.Nifti1Image | Path | str | None' = None, radius: float | None = 6.0, metric: str = 'median', parallel: str = 'cpu', n_jobs: int = -1, show_progress: bool = True) -> dict
```

Compute intersubject correlation (ISC) across the collection.

ISC measures the similarity of brain responses across subjects,
computed by correlating each subject's timeseries with others.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to compute ISC on. | *required*
`method` | <code>[str](#str)</code> | ISC computation method: - 'loo': Leave-one-out (correlate each subject with mean of others) - 'pairwise': All pairwise correlations between subjects | <code>'loo'</code>
`roi_mask` | <code>'nib.Nifti1Image \| Path \| str \| None'</code> | If provided, compute ISC per ROI. Can be: - NIfTI image with integer labels (atlas/parcellation) - Path to parcellation file | <code>None</code>
`radius` | <code>[float](#float) \| None</code> | Searchlight radius in mm. If None, use voxelwise mode. Ignored if roi_mask is provided. | <code>6.0</code>
`metric` | <code>[str](#str)</code> | Summary statistic for aggregating ISC values: - 'median': Robust to outliers (default) - 'mean': Fisher z-transformed mean | <code>'median'</code>
`parallel` | <code>[str](#str)</code> | Parallelization method ('cpu', 'gpu', or None). | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs (-1 = all cores). | <code>-1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during extraction. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with: - 'isc': BrainData with ISC values - 'method': ISC method used ('loo' or 'pairwise') - 'extraction': Extraction mode ('roi', 'searchlight', 'voxelwise') - 'n_subjects': Number of subjects - 'extraction_info': Dict with extraction metadata

**Examples:**

```pycon
>>> # ROI-based ISC using atlas
>>> result = bc.isc(roi_mask="atlas.nii.gz")
>>> result['isc'].plot()
```

```pycon
>>> # Searchlight ISC
>>> result = bc.isc(radius=10.0)
```

```pycon
>>> # Voxelwise ISC
>>> result = bc.isc(radius=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

For permutation testing, see BrainCollection.isc_test() (requires
discussion of statistical methodology first).

</details>

###### `nltools.data.collection.inference.isc_test`

```python
isc_test(bc: 'BrainCollection', method: str = 'loo', roi_mask: 'nib.Nifti1Image | Path | str | None' = None, radius: float | None = 6.0, n_permute: int = 5000, permutation_method: str = 'bootstrap', metric: str = 'median', tail: int = 2, ci_percentile: float = 95, parallel: str = 'cpu', n_jobs: int = -1, random_state: int | None = None, return_null: bool = False, show_progress: bool = True) -> dict
```

Compute ISC with permutation testing for statistical inference.

This method combines ISC computation with permutation testing to
determine statistical significance. It uses the same extraction
pipeline as isc() and wraps isc_permutation_test().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to test. | *required*
`method` | <code>[str](#str)</code> | ISC computation method: - 'loo': Leave-one-out (correlate each subject with mean of others) - 'pairwise': All pairwise correlations between subjects | <code>'loo'</code>
`roi_mask` | <code>'nib.Nifti1Image \| Path \| str \| None'</code> | If provided, compute ISC per ROI. Can be: - NIfTI image with integer labels (atlas/parcellation) - Path to parcellation file | <code>None</code>
`radius` | <code>[float](#float) \| None</code> | Searchlight radius in mm. If None, use voxelwise mode. Ignored if roi_mask is provided. | <code>6.0</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations/bootstrap iterations. Default 5000. | <code>5000</code>
`permutation_method` | <code>[str](#str)</code> | Method for null distribution:<br>- 'bootstrap': Subject-wise bootstrap (default, Chen et al. 2016).   Tests whether observed ISC differs from random groupings. - 'circle_shift': Circular time-shift (preserves autocorrelation).   Tests for temporally-locked shared signal. - 'phase_randomize': FFT phase randomization (preserves power spectrum).   Tests for nonlinear temporal coupling. | <code>'bootstrap'</code>
`metric` | <code>[str](#str)</code> | Summary statistic for aggregating ISC values: - 'median': Robust to outliers (default) - 'mean': Fisher z-transformed mean | <code>'median'</code>
`tail` | <code>[int](#int)</code> | One-tailed (1) or two-tailed (2) test. Default 2. | <code>2</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95). Default 95. | <code>95</code>
`parallel` | <code>[str](#str)</code> | Parallelization method ('cpu', 'gpu', or None). | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs (-1 = all cores). | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, include null distribution in results. | <code>False</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during extraction and permutation. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with: - 'isc': BrainData with ISC values - 'p': BrainData with p-values (Phipson-Smyth corrected) - 'ci': Tuple of (lower, upper) BrainData confidence intervals - 'method': ISC method used ('loo' or 'pairwise') - 'permutation_method': Permutation method used - 'extraction': Extraction mode ('roi', 'searchlight', 'voxelwise') - 'n_subjects': Number of subjects - 'n_permute': Number of permutations - 'null_dist': (optional) Null distribution array if return_null=True

**Examples:**

```pycon
>>> # ROI-based ISC with permutation testing
>>> result = bc.isc_test(roi_mask="atlas.nii.gz", n_permute=5000)
>>> sig_mask = result['p'].data < 0.05
>>> print(f"Significant ROIs: {sig_mask.sum()}")
```

```pycon
>>> # Searchlight ISC testing
>>> result = bc.isc_test(radius=10.0)
>>> result['isc'].plot()  # Show ISC values
>>> result['p'].plot()    # Show p-values
```

```pycon
>>> # Voxelwise with phase randomization (tests temporal coupling)
>>> result = bc.isc_test(
...     radius=None,
...     permutation_method='phase_randomize',
...     parallel='gpu'
... )
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Bootstrap (default) is recommended for standard ISC inference
  (Chen et al. 2016). It tests whether ISC is significant at
  the group level.
- Circle_shift and phase_randomize are more specialized - they
  test for temporally-structured shared signal beyond what's
  explained by autocorrelation or spectral structure alone.
- For large voxelwise analyses, bootstrap is much faster as it
  resamples pre-computed values rather than recomputing ISC.

</details>

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., et al. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

###### `nltools.data.collection.inference.permutation_test`

```python
permutation_test(bc: 'BrainCollection', n_permute: int = 5000, tail: int = 2, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None, return_null: bool = False) -> dict
```

One-sample permutation test across images (sign-flipping).

Tests whether the mean across images is significantly different from
zero using sign-flipping permutation. More robust than parametric
t-test for non-normal distributions.

This is a collection-level interface to
nltools.algorithms.inference.one_sample_permutation_test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to test. | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000). | <code>5000</code>
`tail` | <code>[int](#int)</code> | Test type - 1 for one-tailed, 2 for two-tailed (default). | <code>2</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization method: - 'cpu': CPU parallelization via joblib (default) - 'gpu': GPU acceleration via PyTorch - None: Single-threaded (for debugging) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores (default: -1 = all cores). | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget (default: 4.0 GB). | <code>4.0</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, include null distribution in result. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | dict with keys: - 'mean': BrainData with observed mean across images - 'p': BrainData with p-values - 'null_dist': np.ndarray (if return_null=True) - 'parallel': parallelization method used

**Examples:**

```pycon
>>> result = bc.permutation_test(n_permute=5000)
>>> mean_bd, p_bd = result['mean'], result['p']
```

```pycon
>>> # With GPU acceleration
>>> result = bc.permutation_test(parallel='gpu')
```

###### `nltools.data.collection.inference.permutation_test2`

```python
permutation_test2(bc: 'BrainCollection', other: 'BrainCollection', n_permute: int = 5000, tail: int = 2, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None, return_null: bool = False) -> dict
```

Two-sample permutation test between collections.

Tests whether two collections have different means using group
label permutation. More robust than parametric t-test.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | First BrainCollection. | *required*
`other` | <code>'BrainCollection'</code> | Another BrainCollection to compare against. | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000). | <code>5000</code>
`tail` | <code>[int](#int)</code> | Test type - 1 for one-tailed, 2 for two-tailed (default). | <code>2</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization method ('cpu', 'gpu', or None). | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores (default: -1 = all cores). | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget (default: 4.0 GB). | <code>4.0</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, include null distribution in result. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | dict with keys: - 'mean_diff': BrainData with observed mean difference - 'p': BrainData with p-values - 'null_dist': np.ndarray (if return_null=True) - 'parallel': parallelization method used

**Examples:**

```pycon
>>> result = patients.permutation_test2(controls)
>>> diff_bd, p_bd = result['mean_diff'], result['p']
```

###### `nltools.data.collection.inference.project_to_brain`

```python
project_to_brain(bc: 'BrainCollection', values: np.ndarray, extraction_info: dict)
```

Project ISC values back to brain space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection (used for mask). | *required*
`values` | <code>[ndarray](#numpy.ndarray)</code> | ISC values, shape depends on extraction mode: - ROI mode: (n_rois,) - Searchlight/voxelwise: (n_voxels,) | *required*
`extraction_info` | <code>[dict](#dict)</code> | Dict from extract_for_isc with mode info. | *required*

**Returns:**

Type | Description
---- | -----------
 | BrainData with ISC values in brain space.

###### `nltools.data.collection.inference.ttest`

```python
ttest(bc: 'BrainCollection', popmean: float = 0.0, axis: int | str = 0) -> tuple
```

One-sample t-test across images.

Tests whether the mean across images is significantly different from
a population mean (default: 0). This is the voxel-wise equivalent of
scipy.stats.ttest_1samp.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to test. | *required*
`popmean` | <code>[float](#float)</code> | Population mean to test against (default: 0). | <code>0.0</code>
`axis` | <code>[int](#int) \| [str](#str)</code> | Axis to test across. Only axis=0 (images) supported. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)</code> | Tuple of (t_stat, p_value) as BrainData objects.
<code>[tuple](#tuple)</code> | Both have shape (n_obs, n_voxels) if uniform obs counts.

**Examples:**

```pycon
>>> t_stat, p_val = bc.ttest()  # Test mean != 0
>>> t_stat, p_val = bc.ttest(popmean=0.5)  # Test mean != 0.5
```

```pycon
>>> # Threshold significant voxels
>>> sig_mask = p_val.data < 0.05
```

###### `nltools.data.collection.inference.ttest2`

```python
ttest2(bc: 'BrainCollection', other: 'BrainCollection', equal_var: bool = True) -> tuple
```

Two-sample t-test between collections.

Tests whether two collections have different means. This is the
voxel-wise equivalent of scipy.stats.ttest_ind.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | First BrainCollection. | *required*
`other` | <code>'BrainCollection'</code> | Another BrainCollection to compare against. | *required*
`equal_var` | <code>[bool](#bool)</code> | If True (default), perform standard t-test assuming equal variances. If False, use Welch's t-test. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)</code> | Tuple of (t_stat, p_value) as BrainData objects.

**Examples:**

```pycon
>>> t_stat, p_val = patients.ttest2(controls)
>>> t_stat, p_val = group1.ttest2(group2, equal_var=False)  # Welch's
```

#### `nltools.data.collection.io`

I/O functions for BrainCollection.

Provides save path resolution and write functionality extracted from BrainCollection.

**Functions:**

Name | Description
---- | -----------
[`write`](#nltools.data.collection.io.write) | Write all images in collection to files.



##### Classes

##### Functions###### `nltools.data.collection.io.write`

```python
write(bc: 'BrainCollection', directory: 'str | Path', pattern: str = 'image_{i:04d}.nii.gz', metadata_file: str | None = 'metadata.csv') -> list['Path']
```

Write all images in collection to files.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to write. | *required*
`directory` | <code>'str \| Path'</code> | Output directory path. Will be created if it doesn't exist. | *required*
`pattern` | <code>[str](#str)</code> | Filename pattern with {i} placeholder for image index. Default: "image_{i:04d}.nii.gz" produces image_0000.nii.gz, etc. | <code>'image_{i:04d}.nii.gz'</code>
`metadata_file` | <code>[str](#str) \| None</code> | Optional filename for metadata CSV. Set to None to skip. Default: "metadata.csv" | <code>'metadata.csv'</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)['Path']</code> | List of paths to written files.

**Examples:**

```pycon
>>> bc = BrainCollection([bd1, bd2, bd3], mask=mask)
>>> paths = write(bc, "output/")
>>> # Creates: output/image_0000.nii.gz, image_0001.nii.gz, etc.
```

```pycon
>>> # Custom pattern
>>> write(bc, "output/", pattern="sub-{i:02d}_bold.nii.gz")
>>> # Creates: output/sub-00_bold.nii.gz, sub-01_bold.nii.gz, etc.
```

```pycon
>>> # With BIDS-style naming using metadata
>>> bc.metadata["filename"] = [f"sub-{s}_bold.nii.gz" for s in subjects]
>>> for i, bd in enumerate(bc):
...     bd.write(f"output/{bc.metadata.loc[i, 'filename']}")
```

#### `nltools.data.collection.modeling`

Modeling functions extracted from BrainCollection.

Contains GLM fitting, Ridge fitting, design matrix building, and related helpers.
All BrainCollection methods converted to functions taking `bc` as first argument.

**Functions:**

Name | Description
---- | -----------
[`cv`](#nltools.data.collection.modeling.cv) | Create a cross-validation pipeline for multi-subject analysis.
[`fit`](#nltools.data.collection.modeling.fit) | Fit a model to each subject in the collection.
[`fit_from_events`](#nltools.data.collection.modeling.fit_from_events) | Build design matrices from events and fit GLM to each subject.
[`fit_glm`](#nltools.data.collection.modeling.fit_glm) | Fit GLM to each subject in collection.
[`fit_glm_internal`](#nltools.data.collection.modeling.fit_glm_internal) | Internal GLM fitting with design matrix input.
[`fit_ridge`](#nltools.data.collection.modeling.fit_ridge) | Fit ridge regression to each subject in collection.
[`load_design_matrix`](#nltools.data.collection.modeling.load_design_matrix) | Load design matrix from a file path.
[`load_features`](#nltools.data.collection.modeling.load_features) | Load features from a file path.
[`resolve_X`](#nltools.data.collection.modeling.resolve_X) | Resolve design/feature matrix X to per-subject list.
[`resolve_confounds`](#nltools.data.collection.modeling.resolve_confounds) | Resolve confounds argument to per-subject list.



##### Classes

##### Functions###### `nltools.data.collection.modeling.cv`

```python
cv(bc, k: int | None = None, scheme: str = 'kfold', split_by: str | None = None, groups: np.ndarray | None = None, random_state: int | None = None, **kwargs: int | None) -> 'BrainCollectionPipeline'
```

Create a cross-validation pipeline for multi-subject analysis.

Returns a pipeline object that enables fluent, chainable transforms
with cross-validation across subjects or runs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`k` | <code>[int](#int) \| None</code> | Number of folds (for kfold scheme). Defaults to 5. | <code>None</code>
`scheme` | <code>[str](#str)</code> | CV scheme type. Options: - 'kfold': k-fold cross-validation on pooled data - 'loso': leave-one-subject-out (one image held out per fold) - 'loro': leave-one-run-out (requires groups) | <code>'kfold'</code>
`split_by` | <code>[str](#str) \| None</code> | Metadata column for group splits. If provided and groups is None, gets groups from bc.metadata[split_by]. | <code>None</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Explicit group labels for CV splits. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`**kwargs` |  | Additional arguments passed to CVScheme. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainCollectionPipeline` | <code>'BrainCollectionPipeline'</code> | Pipeline for method chaining.

**Examples:**

```pycon
>>> # Leave-one-subject-out classification
>>> result = bc.cv(scheme='loso').normalize().predict(subject_labels, algorithm='svm')
>>> print(f"Mean accuracy: {result.mean_score:.2%}")
```

```pycon
>>> # With preprocessing
>>> result = (bc
...     .cv(scheme='loso')
...     .normalize()
...     .reduce(n_components=50)
...     .predict(labels))
```

```pycon
>>> # Run-based CV with metadata
>>> result = bc.cv(scheme='loro', split_by='run').predict(y)
```

<details class="see-also" open markdown="1">
<summary>See Also</summary>

BrainCollectionPipeline: For available transforms and terminals.
CVScheme: For CV scheme configuration details.

</details>

###### `nltools.data.collection.modeling.fit`

```python
fit(bc, model: str, X: 'pd.DataFrame | np.ndarray | str | list', cv: int | None = None, scale: bool = True, scale_value: float = 100.0, show_progress: bool = True, **kwargs: bool) -> 'FittedBrainCollection'
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
`X` | <code>'pd.DataFrame \| np.ndarray \| str \| list'</code> | Design/feature matrix. Can be: - pd.DataFrame/DesignMatrix: Shared (used for all subjects) - np.ndarray: Shared array (used for all subjects) - str: Column name in metadata pointing to file paths - list: Per-subject list of DataFrames/arrays/paths | *required*
`cv` | <code>[int](#int) \| None</code> | Cross-validation folds (Ridge only). Default is None for GLM, 5 for Ridge when output='scores'. | <code>None</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`**kwargs` |  | Model-specific arguments passed to _fit_glm or _fit_ridge: - GLM: return_stats, save - Ridge: alpha, output, save, backend, random_state | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'FittedBrainCollection'</code> | FittedBrainCollection wrapping the fitted results. Supports:
<code>'FittedBrainCollection'</code> | - ``.results``: Access underlying BrainCollection(s) directly
<code>'FittedBrainCollection'</code> | - ``.betas``: Convenience accessor for beta coefficients (GLM)
<code>'FittedBrainCollection'</code> | - ``.pool()``: Aggregate across subjects for group analysis
<code>'FittedBrainCollection'</code> | The underlying results contain:
<code>'FittedBrainCollection'</code> | - GLM: Beta coefficients (n_regressors, n_voxels) per subject
<code>'FittedBrainCollection'</code> | - Ridge: Scores or weights depending on 'output' kwarg
<code>'FittedBrainCollection'</code> | If return_stats (GLM) or output='both' (Ridge), results is a dict.

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

###### `nltools.data.collection.modeling.fit_from_events`

```python
fit_from_events(bc, events: pd.DataFrame, t_r: float, confounds: str | list[pd.DataFrame | Path | str] | None = None, confound_columns: list[str] | None = None, hrf_model: str = 'spm', drift_model: str = 'cosine', high_pass: float = 0.01, scale: bool = True, scale_value: float = 100.0, return_stats: list[str] | None = None, return_residuals: bool = False, save: dict[str, str] | None = None, show_progress: bool = True, by_run: bool = False, run_column: str = 'run', run_lengths: int | list[int] | None = None) -> 'BrainCollection | dict[str, BrainCollection]'
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
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`by_run` | <code>[bool](#bool)</code> | If True, fit GLM separately per run and return run-level betas. This enables MVPA decoding with leave-one-run-out CV. | <code>False</code>
`run_column` | <code>[str](#str)</code> | Column name in events identifying runs (default 'run'). | <code>'run'</code>
`run_lengths` | <code>[int](#int) \| [list](#list)[[int](#int)] \| None</code> | Number of TRs per run. Required when by_run=True. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection of beta coefficients for task regressors.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | If return_stats specified, returns dict with keys 'betas', 't', etc.

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

###### `nltools.data.collection.modeling.fit_glm`

```python
fit_glm(bc, events: pd.DataFrame, t_r: float, confounds: str | list[pd.DataFrame | Path | str] | None = None, confound_columns: list[str] | None = None, hrf_model: str = 'spm', drift_model: str = 'cosine', high_pass: float = 0.01, scale: bool = True, scale_value: float = 100.0, return_stats: list[str] | None = None, return_residuals: bool = False, save: dict[str, str] | None = None, show_progress: bool = True, by_run: bool = False, run_column: str = 'run', run_lengths: int | list[int] | None = None) -> 'BrainCollection | dict[str, BrainCollection]'
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
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`by_run` | <code>[bool](#bool)</code> | If True, fit GLM separately per run and return run-level betas. This enables MVPA decoding with leave-one-run-out CV. Each subject will have (n_runs * n_conditions, n_voxels) betas. | <code>False</code>
`run_column` | <code>[str](#str)</code> | Column name in events identifying runs (default 'run'). | <code>'run'</code>
`run_lengths` | <code>[int](#int) \| [list](#list)[[int](#int)] \| None</code> | Number of TRs per run. Required when by_run=True.<br>- int: All runs have same length - list of int: Different length per run - None: Will attempt to infer equal-length runs from total scans | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection where each BrainData has shape:
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | - (n_task_regressors, n_voxels) if by_run=False (default)
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | - (n_runs * n_task_regressors, n_voxels) if by_run=True
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | The ``._design_columns`` attribute stores task regressor names.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | If by_run=True, also stores ``._condition_labels`` and ``._run_labels``.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | If return_stats specified, returns dict with keys 'betas', 't', etc.

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

###### `nltools.data.collection.modeling.fit_glm_internal`

```python
fit_glm_internal(bc, X: 'pd.DataFrame | np.ndarray | str | list', scale: bool = True, scale_value: float = 100.0, return_stats: list[str] | None = None, save: dict[str, str] | None = None, show_progress: bool = True) -> 'BrainCollection | dict[str, BrainCollection]'
```

Internal GLM fitting with design matrix input.

Core implementation that accepts DesignMatrix/DataFrame directly.
Called by fit(model='glm') and fit_from_events().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`X` | <code>'pd.DataFrame \| np.ndarray \| str \| list'</code> | Design matrix. Can be: - pd.DataFrame/DesignMatrix: Shared (used for all subjects) - np.ndarray: Shared array (converted to DataFrame internally) - str: Column name in metadata pointing to file paths - list: Per-subject list of DataFrames/arrays/paths | *required*
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`return_stats` | <code>[list](#list)[[str](#str)] \| None</code> | Optional list of statistics to return as separate BrainCollections. Options: 't', 'r2', 'p', 'se', 'residual'. | <code>None</code>
`save` | <code>[dict](#dict)[[str](#str), [str](#str)] \| None</code> | Dict mapping output type to path template. | <code>None</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection of betas, or dict with betas + requested stats.

###### `nltools.data.collection.modeling.fit_ridge`

```python
fit_ridge(bc, X: 'np.ndarray | str | list', alpha: float | str = 1.0, cv: int | None = 5, scale: bool = True, scale_value: float = 100.0, output: str = 'scores', save: dict[str, str] | None = None, show_progress: bool = True, **ridge_kwargs: bool) -> 'BrainCollection | dict[str, BrainCollection]'
```

Fit ridge regression to each subject in collection.

Memory-efficient encoding model fitting that processes subjects one at a
time. Default behavior returns cross-validated R-squared scores per voxel,
suitable for group-level inference on encoding model performance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`X` | <code>'np.ndarray \| str \| list'</code> | Feature matrix. Can be: - np.ndarray: Shared features (n_samples, n_features) used for all subjects - str: Column name in metadata pointing to feature file paths - list: List of arrays/DataFrames, one per subject | *required*
`alpha` | <code>[float](#float) \| [str](#str)</code> | Ridge regularization parameter. Can be: - float: Fixed regularization strength - 'auto': Use cross-validation to select optimal alpha | <code>1.0</code>
`cv` | <code>[int](#int) \| None</code> | Cross-validation folds for computing scores. Default is 5. Required when output='scores' or 'both'. Set to None only when output='weights'. | <code>5</code>
`scale` | <code>[bool](#bool)</code> | If True, apply percent signal change scaling before fitting. | <code>True</code>
`scale_value` | <code>[float](#float)</code> | Scaling value (default 100.0 for percent signal change). | <code>100.0</code>
`output` | <code>[str](#str)</code> | What to return. Options: - 'scores': CV R-squared scores per voxel (default, for encoding workflow) - 'weights': Model weights (n_features, n_voxels) - 'both': Dict with both 'scores' and 'weights' | <code>'scores'</code>
`save` | <code>[dict](#dict)[[str](#str), [str](#str)] \| None</code> | Dict mapping output type to path template, e.g. ``{'weights': 'output/{subject}_weights.nii.gz', 'scores': 'output/{subject}_scores.nii.gz'}``. Supports {subject}, {session}, {idx}, and other metadata columns. | <code>None</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>
`**ridge_kwargs` |  | Additional arguments passed to Ridge model (e.g., backend='torch', random_state=42). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection of scores or weights, or dict with both if output='both'.
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | Each BrainData will have ``cv_results_`` attribute when cv is used.

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

###### `nltools.data.collection.modeling.load_design_matrix`

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

###### `nltools.data.collection.modeling.load_features`

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

###### `nltools.data.collection.modeling.resolve_X`

```python
resolve_X(bc, X: 'np.ndarray | pd.DataFrame | str | list | None') -> list | None
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
`X` | <code>'np.ndarray \| pd.DataFrame \| str \| list \| None'</code> | Design/feature matrix. Can be: - np.ndarray: Shared array (used for all subjects) - pd.DataFrame: Shared DataFrame/DesignMatrix (used for all subjects) - str: Column name in metadata containing file paths - list: Per-subject list of arrays/DataFrames/paths - None: Error | *required*

**Returns:**

Type | Description
---- | -----------
<code>[list](#list) \| None</code> | list | None: Per-subject list if X varies by subject, None if shared. Caller should use: `X_subj = X_list[i] if X_list else X`

###### `nltools.data.collection.modeling.resolve_confounds`

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

#### `nltools.data.collection.pipeline`

Pipeline classes for BrainCollection.

Provides BrainCollectionPipeline for fluent pipeline API with cross-validation,
BrainCollectionCVResult for storing CV results, and FittedBrainCollection for
chaining pool() after fit().

**Classes:**

Name | Description
---- | -----------
[`BrainCollectionCVResult`](#nltools.data.collection.pipeline.BrainCollectionCVResult) | Cross-validation results for BrainCollection pipelines.
[`BrainCollectionPipeline`](#nltools.data.collection.pipeline.BrainCollectionPipeline) | Pipeline for BrainCollection with multi-subject CV support.
[`FittedBrainCollection`](#nltools.data.collection.pipeline.FittedBrainCollection) | Wrapper for fitted BrainCollection enabling pool() chaining.



##### Classes###### `nltools.data.collection.pipeline.BrainCollectionCVResult`

```python
BrainCollectionCVResult(fold_results: list, pipeline: BrainCollectionPipeline)
```

Cross-validation results for BrainCollection pipelines.

Contains fold-by-fold results from cross-validated prediction,
with convenience properties for accessing scores and predictions.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`fold_results`](#nltools.data.collection.pipeline.BrainCollectionCVResult.fold_results) |  | List of dictionaries with per-fold results.
[`pipeline`](#nltools.data.collection.pipeline.BrainCollectionCVResult.pipeline) |  | The pipeline that generated these results.
[`scores`](#nltools.data.collection.pipeline.BrainCollectionCVResult.scores) | <code>[ndarray](#numpy.ndarray)</code> | Per-fold prediction scores.
[`mean_score`](#nltools.data.collection.pipeline.BrainCollectionCVResult.mean_score) | <code>[float](#float)</code> | Mean score across all folds.
[`std_score`](#nltools.data.collection.pipeline.BrainCollectionCVResult.std_score) | <code>[float](#float)</code> | Standard deviation of scores.
[`n_folds`](#nltools.data.collection.pipeline.BrainCollectionCVResult.n_folds) | <code>[int](#int)</code> | Number of CV folds.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fold_results` | <code>[list](#list)</code> | List of fold result dictionaries. | *required*
`pipeline` | <code>[BrainCollectionPipeline](#nltools.data.collection.pipeline.BrainCollectionPipeline)</code> | The pipeline that generated these results. | *required*



####### Attributes######## `nltools.data.collection.pipeline.BrainCollectionCVResult.fold_results`

```python
fold_results = fold_results
```

######## `nltools.data.collection.pipeline.BrainCollectionCVResult.mean_score`

```python
mean_score: float
```

Mean score across folds.

######## `nltools.data.collection.pipeline.BrainCollectionCVResult.n_folds`

```python
n_folds: int
```

Number of cross-validation folds.

######## `nltools.data.collection.pipeline.BrainCollectionCVResult.pipeline`

```python
pipeline = pipeline
```

######## `nltools.data.collection.pipeline.BrainCollectionCVResult.scores`

```python
scores: np.ndarray
```

Per-fold prediction scores as a numpy array.

######## `nltools.data.collection.pipeline.BrainCollectionCVResult.std_score`

```python
std_score: float
```

Standard deviation of scores.

###### `nltools.data.collection.pipeline.BrainCollectionPipeline`

```python
BrainCollectionPipeline(brain_collection: 'BrainCollection', cv: 'BrainCollection' = None, groups: np.ndarray | None = None)
```

Pipeline for BrainCollection with multi-subject CV support.

Wraps BrainCollection to provide fluent pipeline API with LOSO
and run-based cross-validation.

This class enables method chaining for preprocessing and prediction
with proper cross-validation semantics for multi-subject neuroimaging
analyses.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`n_subjects`](#nltools.data.collection.pipeline.BrainCollectionPipeline.n_subjects) | <code>[int](#int)</code> | Number of subjects/images in the collection.
[`cv`](#nltools.data.collection.pipeline.BrainCollectionPipeline.cv) |  | The cross-validation scheme configuration.
[`n_steps`](#nltools.data.collection.pipeline.BrainCollectionPipeline.n_steps) | <code>[int](#int)</code> | Number of transform steps in the pipeline.

**Examples:**

```pycon
>>> # Leave-one-subject-out with preprocessing
>>> result = (bc
...     .cv(scheme='loso')
...     .normalize()
...     .reduce(n_components=50)
...     .predict(labels, algorithm='svm'))
>>> print(f"Mean accuracy: {result.mean_score:.2%}")
```

**Functions:**

Name | Description
---- | -----------
[`normalize`](#nltools.data.collection.pipeline.BrainCollectionPipeline.normalize) | Add normalization step.
[`pipe`](#nltools.data.collection.pipeline.BrainCollectionPipeline.pipe) | Add custom sklearn transformer.
[`predict`](#nltools.data.collection.pipeline.BrainCollectionPipeline.predict) | Execute pipeline with CV and return prediction results.
[`reduce`](#nltools.data.collection.pipeline.BrainCollectionPipeline.reduce) | Add dimensionality reduction step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_collection` | <code>'BrainCollection'</code> | BrainCollection to wrap. | *required*
`cv` |  | CVScheme configuration. | <code>None</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Group labels for CV splits. | <code>None</code>



####### Attributes######## `nltools.data.collection.pipeline.BrainCollectionPipeline.cv`

```python
cv
```

Cross-validation scheme.

######## `nltools.data.collection.pipeline.BrainCollectionPipeline.n_steps`

```python
n_steps: int
```

Number of transform steps.

######## `nltools.data.collection.pipeline.BrainCollectionPipeline.n_subjects`

```python
n_subjects: int
```

Number of subjects/images.



####### Functions######## `nltools.data.collection.pipeline.BrainCollectionPipeline.normalize`

```python
normalize(method: str = 'zscore', **kwargs: str) -> 'BrainCollectionPipeline'
```

Add normalization step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Normalization method ('zscore', 'minmax'). | <code>'zscore'</code>
`**kwargs` |  | Additional arguments for NormalizeStep. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollectionPipeline'</code> | New pipeline with normalization step added.

######## `nltools.data.collection.pipeline.BrainCollectionPipeline.pipe`

```python
pipe(transformer) -> 'BrainCollectionPipeline'
```

Add custom sklearn transformer.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`transformer` |  | sklearn-compatible transformer with fit/transform interface. | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollectionPipeline'</code> | New pipeline with custom step added.

######## `nltools.data.collection.pipeline.BrainCollectionPipeline.predict`

```python
predict(y, algorithm: str = 'ridge', **kwargs: str) -> 'BrainCollectionCVResult'
```

Execute pipeline with CV and return prediction results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` |  | Target variable. For LOSO, shape should be (n_subjects,). | *required*
`algorithm` | <code>[str](#str)</code> | Prediction algorithm ('ridge', 'svm', 'logistic', etc.) | <code>'ridge'</code>
`**kwargs` |  | Passed to model constructor. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollectionCVResult'</code> | Cross-validation results with scores and predictions.

######## `nltools.data.collection.pipeline.BrainCollectionPipeline.reduce`

```python
reduce(method: str = 'pca', n_components: int | None = None, **kwargs: int | None) -> 'BrainCollectionPipeline'
```

Add dimensionality reduction step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Reduction method ('pca', 'ica'). | <code>'pca'</code>
`n_components` | <code>[int](#int) \| None</code> | Number of components to keep. | <code>None</code>
`**kwargs` |  | Additional arguments for ReduceStep. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollectionPipeline'</code> | New pipeline with reduction step added.

###### `nltools.data.collection.pipeline.FittedBrainCollection`

```python
FittedBrainCollection(brain_collection: 'BrainCollection', fitted_results: 'BrainCollection | dict[str, BrainCollection]', model: str, condition_names: list[str] | None = None)
```

Wrapper for fitted BrainCollection enabling pool() chaining.

This class wraps the results of bc.fit() and provides the .pool()
method for aggregating across subjects.

The execution model:
- fit() executes immediately (eager)
- pool() aggregates the fitted parameters
- pool() returns PooledData for second-level analysis

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_collection` | <code>'BrainCollection'</code> | The original collection that was fitted. | *required*
`fitted_results` | <code>'BrainCollection \| dict[str, BrainCollection]'</code> | The fitted results. Can be a BrainCollection (betas or scores) or a dict mapping stat names to BrainCollections (e.g., {'betas': ..., 't': ...}). | *required*
`model` | <code>[str](#str)</code> | The model type that was fitted ('glm' or 'ridge'). | *required*
`condition_names` | <code>[list](#list)[[str](#str)] \| None</code> | Names of conditions/regressors from the design matrix. | <code>None</code>

Examples
--------
>>> fitted = bc.fit(model='glm', X=dm)
>>> pool = fitted.pool(param='beta')
>>> result = pool.fit(model='ttest', contrast='A-B')

**Functions:**

Name | Description
---- | -----------
[`pool`](#nltools.data.collection.pipeline.FittedBrainCollection.pool) | Pool fitted parameters across subjects.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`betas`](#nltools.data.collection.pipeline.FittedBrainCollection.betas) | <code>'BrainCollection'</code> | Convenience accessor for beta coefficients from a GLM fit.
[`n_subjects`](#nltools.data.collection.pipeline.FittedBrainCollection.n_subjects) | <code>[int](#int)</code> | Number of subjects in the fitted collection.
[`results`](#nltools.data.collection.pipeline.FittedBrainCollection.results) | <code>'BrainCollection \| dict[str, BrainCollection]'</code> | Access the fitted results directly.



####### Attributes######## `nltools.data.collection.pipeline.FittedBrainCollection.betas`

```python
betas: 'BrainCollection'
```

Convenience accessor for beta coefficients from a GLM fit.

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | Beta coefficients from GLM fit.

######## `nltools.data.collection.pipeline.FittedBrainCollection.n_subjects`

```python
n_subjects: int
```

Number of subjects in the fitted collection.

######## `nltools.data.collection.pipeline.FittedBrainCollection.results`

```python
results: 'BrainCollection | dict[str, BrainCollection]'
```

Access the fitted results directly.

Returns the underlying BrainCollection or dict of BrainCollections.
Use this for backward compatibility or when pool() is not needed.



####### Functions######## `nltools.data.collection.pipeline.FittedBrainCollection.pool`

```python
pool(param: str = 'beta', contrast: str | None = None, save: str | None = None, save_fitted: bool = False)
```

Pool fitted parameters across subjects.

Aggregates per-subject fitted results for group-level analysis.
Returns a PooledData object that can be passed to second-level
statistical tests.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`param` | <code>[str](#str)</code> | Parameter to pool. GLM options: 'beta', 't', 'r2', 'p', 'se', 'residual'. Ridge options: 'scores', 'weights'. Default is 'beta'. | <code>'beta'</code>
`contrast` | <code>[str](#str) \| None</code> | Apply contrast before pooling. Format: 'A-B' or 'A+B'. Requires condition_names to be available. | <code>None</code>
`save` | <code>[str](#str) \| None</code> | Path template to save per-subject results before pooling. Supports {subject}, {idx} placeholders. | <code>None</code>
`save_fitted` | <code>[bool](#bool)</code> | If True, save full fitted state for later repool(). | <code>False</code>

**Returns:**

Type | Description
---- | -----------
 | Pooled data ready for second-level analysis.

Examples
--------
>>> pool = bc.fit(model='glm', X=designs).pool(param='beta')
>>> result = pool.fit(model='ttest', contrast='face-house')

>>> # Pool t-statistics instead of betas
>>> pool = bc.fit(model='glm', X=dm, return_stats=['t']).pool(param='t')

#### `nltools.data.collection.prediction`

Prediction functions extracted from BrainCollection.

Contains predict, compute_contrasts, select_feature, and related helpers.
All BrainCollection methods converted to functions taking `bc` as first argument.

**Functions:**

Name | Description
---- | -----------
[`compute_contrasts`](#nltools.data.collection.prediction.compute_contrasts) | Compute contrasts from fitted GLM beta coefficients.
[`compute_single_contrast`](#nltools.data.collection.prediction.compute_single_contrast) | Compute a single contrast across all subjects.
[`parse_contrast_string`](#nltools.data.collection.prediction.parse_contrast_string) | Parse a contrast string into a numeric contrast vector.
[`predict`](#nltools.data.collection.prediction.predict) | Generate predictions for each subject in collection.
[`select_feature`](#nltools.data.collection.prediction.select_feature) | Select a single feature's weights across all subjects.



##### Classes

##### Functions###### `nltools.data.collection.prediction.compute_contrasts`

```python
compute_contrasts(bc, contrasts: 'str | dict | np.ndarray | list') -> 'BrainCollection | dict[str, BrainCollection]'
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
`contrasts` | <code>'str \| dict \| np.ndarray \| list'</code> | Can be: - str: Contrast string using column names, e.g., "face - house" - dict: Multiple contrasts, e.g., {"main": "face - house", "avg": [0.5, 0.5]} - array/list: Numeric contrast vector, e.g., [1, -1] | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | BrainCollection where each BrainData has shape (n_voxels,) containing
<code>'BrainCollection \| dict[str, BrainCollection]'</code> | the contrast values. If dict input, returns dict of BrainCollections.

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

###### `nltools.data.collection.prediction.compute_single_contrast`

```python
compute_single_contrast(bc, contrast: 'str | np.ndarray | list', design_columns: list[str]) -> 'BrainCollection'
```

Compute a single contrast across all subjects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` |  | BrainCollection instance. | *required*
`contrast` | <code>'str \| np.ndarray \| list'</code> | Contrast specification (string, array, or list) | *required*
`design_columns` | <code>[list](#list)[[str](#str)]</code> | List of regressor names from fit_glm | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with contrast values for each subject

###### `nltools.data.collection.prediction.parse_contrast_string`

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

###### `nltools.data.collection.prediction.predict`

```python
predict(bc, X: 'np.ndarray | str | list | None' = None, y: 'np.ndarray | None' = None, method: str = 'whole_brain', estimator: str = 'svm', cv: str = 5, groups: 'np.ndarray | None' = None, roi_mask: 'np.ndarray | None' = None, radius: float = 10.0, scoring: str = 'accuracy', standardize: bool = True, n_jobs: int = -1, show_progress: bool = True) -> 'BrainCollection'
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
`X` | <code>'np.ndarray \| str \| list \| None'</code> | Features for timeseries prediction. Can be: - np.ndarray: Shared features (same for all subjects) - str: Metadata column with per-subject feature paths - list: Per-subject feature arrays | <code>None</code>
`y` | <code>'np.ndarray \| None'</code> | Labels for MVPA decoding. If None and _condition_labels exists, will use stored condition labels (from fit_glm with by_run=True). | <code>None</code>
`method` | <code>[str](#str)</code> | MVPA method - 'whole_brain', 'searchlight', or 'roi'. | <code>'whole_brain'</code>
`estimator` |  | Classifier - 'svm', 'logistic', 'ridge', 'lda' or sklearn estimator instance. | <code>'svm'</code>
`cv` |  | Cross-validation strategy. If None and _run_labels exists, uses leave-one-group-out with run labels. | <code>5</code>
`groups` | <code>'np.ndarray \| None'</code> | Group labels for GroupKFold/LeaveOneGroupOut. If None and _run_labels exists, uses stored run labels. | <code>None</code>
`roi_mask` |  | Mask for ROI-based MVPA. Required if method='roi'. | <code>None</code>
`radius` | <code>[float](#float)</code> | Searchlight radius in mm (default 10.0). | <code>10.0</code>
`scoring` | <code>[str](#str)</code> | Scoring metric (default 'accuracy'). | <code>'accuracy'</code>
`standardize` | <code>[bool](#bool)</code> | If True, standardize features before classification. | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Parallel jobs for searchlight (-1 = all cores). | <code>-1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with prediction results:
<code>'BrainCollection'</code> | - For timeseries: (n_timepoints, n_voxels) predicted responses
<code>'BrainCollection'</code> | - For MVPA: (1, n_voxels) accuracy values

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

###### `nltools.data.collection.prediction.select_feature`

```python
select_feature(bc, feature: 'int | str') -> 'BrainCollection'
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
`feature` | <code>'int \| str'</code> | Feature to select. Can be: - int: Feature index (0-based) - str: Feature name (requires _feature_names attribute) | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection where each BrainData has shape (n_voxels,)
<code>'BrainCollection'</code> | containing the weights for the specified feature.

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

#### `nltools.data.collection.transforms`

BrainCollection transform functions.

Extracted from BrainCollection methods — each function takes a BrainCollection
as its first argument (``bc``) instead of ``self``.

**Functions:**

Name | Description
---- | -----------
[`align`](#nltools.data.collection.transforms.align) | Align subjects using local functional alignment.
[`detrend`](#nltools.data.collection.transforms.detrend) | Remove trend from each image.
[`filter_collection`](#nltools.data.collection.transforms.filter_collection) | Filter collection by predicate.
[`map_axis0`](#nltools.data.collection.transforms.map_axis0) | Map function over images (axis=0).
[`map_axis1`](#nltools.data.collection.transforms.map_axis1) | Map function over timepoints (axis=1).
[`map_axis2`](#nltools.data.collection.transforms.map_axis2) | Map function over voxels (axis=2) per image.
[`map_collection`](#nltools.data.collection.transforms.map_collection) | Apply function across specified axis.
[`smooth`](#nltools.data.collection.transforms.smooth) | Spatially smooth each image.
[`standardize`](#nltools.data.collection.transforms.standardize) | Standardize each image.
[`threshold`](#nltools.data.collection.transforms.threshold) | Threshold each image.



##### Classes

##### Functions###### `nltools.data.collection.transforms.align`

```python
align(bc: 'BrainCollection', method: str = 'procrustes', scheme: str = 'searchlight', radius_mm: float = 10.0, parcellation: 'nib.Nifti1Image | None' = None, n_features: int | None = None, n_iter: int = 3, parallel: str | None = 'cpu', n_jobs: int = -1, return_model: bool = False, show_progress: bool = True) -> 'BrainCollection | tuple[BrainCollection, object]'
```

Align subjects using local functional alignment.

Performs neighborhood-based functional alignment across subjects using
LocalAlignment. Each subject's data is aligned to a common template space
using local transforms learned within searchlight spheres or parcels.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to align. | *required*
`method` | <code>[str](#str)</code> | Alignment method. Options: - 'procrustes': Orthogonal Procrustes (default, preserves dimensions) - 'srm': Shared Response Model (dimensionality reduction) - 'hyperalignment': Hyperalignment (iterative Procrustes) | <code>'procrustes'</code>
`scheme` | <code>[str](#str)</code> | Spatial scheme. Options: - 'searchlight': Overlapping spheres with center-only aggregation - 'piecewise': Non-overlapping parcels (requires parcellation) | <code>'searchlight'</code>
`radius_mm` | <code>[float](#float)</code> | Sphere radius in millimeters for searchlight scheme. | <code>10.0</code>
`parcellation` | <code>'nib.Nifti1Image \| None'</code> | Parcellation image for piecewise scheme (required if scheme='piecewise'). | <code>None</code>
`n_features` | <code>[int](#int) \| None</code> | Number of features for SRM. None uses full dimensions. | <code>None</code>
`n_iter` | <code>[int](#int)</code> | Number of iterations for alignment refinement. | <code>3</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization mode. Options: - None: Single-threaded - 'cpu': CPU parallelization with joblib - 'gpu': GPU acceleration via PyTorch | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs for CPU mode (-1 = auto). | <code>-1</code>
`return_model` | <code>[bool](#bool)</code> | If True, return (aligned_collection, model) tuple for fit/transform workflow with new data. | <code>False</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar during fitting. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection \| tuple[BrainCollection, object]'</code> | BrainCollection with aligned data. If return_model=True, returns
<code>'BrainCollection \| tuple[BrainCollection, object]'</code> | tuple of (aligned_collection, LocalAlignment_model).

**Examples:**

```pycon
>>> # Basic searchlight alignment
>>> aligned_bc = bc.align(method='procrustes', radius_mm=10.0)
```

```pycon
>>> # Piecewise alignment with parcellation
>>> aligned_bc = bc.align(
...     scheme='piecewise',
...     parcellation=parcellation_img,
...     method='srm',
...     n_features=50
... )
```

```pycon
>>> # Fit/transform workflow for train/test split
>>> aligned_train, model = train_bc.align(return_model=True)
>>> aligned_test = model.transform(test_data_list)
```

```pycon
>>> # GPU-accelerated alignment
>>> aligned_bc = bc.align(parallel='gpu')
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

Based on Bazeille et al. 2021 "An empirical evaluation of functional
alignment using inter-subject decoding". Center-only aggregation is
used for searchlight to preserve local orthogonality of transforms.

</details>

<details class="see-also" open markdown="1">
<summary>See Also</summary>

nltools.algorithms.alignment.LocalAlignment: Underlying alignment class.

</details>

###### `nltools.data.collection.transforms.detrend`

```python
detrend(bc: 'BrainCollection', method: str = 'linear', n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Remove trend from each image.

Delegates to BrainData.detrend() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to detrend. | *required*
`method` | <code>[str](#str)</code> | 'linear' or 'constant'. | <code>'linear'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with detrended images.

**Examples:**

```pycon
>>> bc.detrend()  # Remove linear trend
>>> bc.detrend(method='constant')  # Remove mean only
```

###### `nltools.data.collection.transforms.filter_collection`

```python
filter_collection(bc: 'BrainCollection', predicate: 'Callable | list | np.ndarray') -> 'BrainCollection'
```

Filter collection by predicate.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to filter. | *required*
`predicate` | <code>'Callable \| list \| np.ndarray'</code> | Filter condition. Can be: - callable: fn(BrainData) -> bool - list/ndarray: Boolean mask of length n_images - pd.Series: Boolean series (index ignored) | *required*

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with subset of images matching predicate.

**Examples:**

```pycon
>>> # Filter by callable
>>> bc.filter(lambda bd: bd.data.mean() > 0)
```

```pycon
>>> # Filter by boolean mask
>>> mask = [True, False, True]
>>> bc.filter(mask)
```

```pycon
>>> # Filter by metadata condition
>>> bc.filter(bc.metadata['group'] == 'control')
```

###### `nltools.data.collection.transforms.map_axis0`

```python
map_axis0(bc: 'BrainCollection', fn: Callable, n_jobs: int, show_progress: bool) -> 'BrainCollection'
```

Map function over images (axis=0).

###### `nltools.data.collection.transforms.map_axis1`

```python
map_axis1(bc: 'BrainCollection', fn: Callable, n_jobs: int, show_progress: bool) -> 'BrainCollection'
```

Map function over timepoints (axis=1).

###### `nltools.data.collection.transforms.map_axis2`

```python
map_axis2(bc: 'BrainCollection', fn: Callable, n_jobs: int, show_progress: bool) -> 'BrainCollection'
```

Map function over voxels (axis=2) per image.

###### `nltools.data.collection.transforms.map_collection`

```python
map_collection(bc: 'BrainCollection', fn: Callable, axis: int | str = 0, n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Apply function across specified axis.

This is the general-purpose transformation method. For common operations,
use convenience methods like standardize(), smooth(), etc.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to transform. | *required*
`fn` | <code>[Callable](#collections.abc.Callable)</code> | Function to apply. Signature depends on axis: - axis=0: fn(BrainData) -> BrainData (per image) - axis=1: fn(BrainData) -> BrainData (per timepoint slice) - axis=2: fn(ndarray[n_obs]) -> ndarray (per voxel timeseries) | *required*
`axis` | <code>[int](#int) \| [str](#str)</code> | Axis to iterate over: - 0 or 'images': Apply fn to each image independently - 1 or 'time': Apply fn to each timepoint across images - 2 or 'voxels': Apply fn to each voxel timeseries per image | <code>0</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. -1 for all cores. Default 1. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show tqdm progress bar. Default True. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with transformed data.

**Examples:**

```pycon
>>> # Per-image operation
>>> bc.map(lambda bd: bd.standardize())
```

```pycon
>>> # Per-voxel timeseries (e.g., detrend each voxel)
>>> from scipy.signal import detrend
>>> bc.map(detrend, axis=2)
```

```pycon
>>> # Parallel processing
>>> bc.map(expensive_fn, n_jobs=-1)
```

###### `nltools.data.collection.transforms.smooth`

```python
smooth(bc: 'BrainCollection', fwhm: float, n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Spatially smooth each image.

Delegates to BrainData.smooth() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to smooth. | *required*
`fwhm` | <code>[float](#float)</code> | Full width at half maximum of Gaussian kernel in mm. | *required*
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with smoothed images.

**Examples:**

```pycon
>>> bc.smooth(fwhm=6)  # 6mm FWHM smoothing
```

###### `nltools.data.collection.transforms.standardize`

```python
standardize(bc: 'BrainCollection', axis: int = 0, method: str = 'center', n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Standardize each image.

Delegates to BrainData.standardize() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to standardize. | *required*
`axis` | <code>[int](#int)</code> | Axis for standardization within each image: - 0: Standardize across observations (time) per voxel - 1: Standardize across voxels per observation | <code>0</code>
`method` | <code>[str](#str)</code> | 'center' (subtract mean) or 'zscore' (subtract mean, divide std) | <code>'center'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with standardized images.

**Examples:**

```pycon
>>> bc.standardize()  # Center each image across time
>>> bc.standardize(method='zscore')  # Z-score each image
>>> bc.standardize(axis=1)  # Standardize across voxels
```

###### `nltools.data.collection.transforms.threshold`

```python
threshold(bc: 'BrainCollection', upper: float | str | None = None, lower: float | str | None = None, binarize: bool = False, coerce_nan: bool = True, n_jobs: int = 1, show_progress: bool = True) -> 'BrainCollection'
```

Threshold each image.

Delegates to BrainData.threshold() for each image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bc` | <code>'BrainCollection'</code> | BrainCollection to threshold. | *required*
`upper` | <code>[float](#float) \| [str](#str) \| None</code> | Upper cutoff. String interpreted as percentile. | <code>None</code>
`lower` | <code>[float](#float) \| [str](#str) \| None</code> | Lower cutoff. String interpreted as percentile. | <code>None</code>
`binarize` | <code>[bool](#bool)</code> | Return binary mask. | <code>False</code>
`coerce_nan` | <code>[bool](#bool)</code> | Replace NaN with 0. | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs. | <code>1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>'BrainCollection'</code> | BrainCollection with thresholded images.

**Examples:**

```pycon
>>> bc.threshold(lower=0)  # Zero out negative values
>>> bc.threshold(upper='95%')  # Keep top 5%
>>> bc.threshold(lower=2, binarize=True)  # Binary mask
```

