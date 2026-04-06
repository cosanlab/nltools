## `nltools.data.braindata`

NeuroLearn Brain Data
=====================

Classes to represent brain image data.

**Modules:**

Name | Description
---- | -----------
[`analysis`](#nltools.data.braindata.analysis) | BrainData analysis functions.
[`bootstrap`](#nltools.data.braindata.bootstrap) | Bootstrap functions extracted from BrainData methods.
[`cache`](#nltools.data.braindata.cache) | Disk-based caching infrastructure for expensive computations.
[`io`](#nltools.data.braindata.io) | BrainData I/O and loading functions.
[`modeling`](#nltools.data.braindata.modeling) | BrainData modeling functions.
[`neighborhoods`](#nltools.data.braindata.neighborhoods) | Spatial neighborhood computation for neuroimaging analyses.
[`pipeline`](#nltools.data.braindata.pipeline) | BrainData pipeline and cross-validation result classes.
[`plotting`](#nltools.data.braindata.plotting) | BrainData plotting functions.
[`prediction`](#nltools.data.braindata.prediction) | BrainData prediction functions.
[`utils`](#nltools.data.braindata.utils) | Shared helpers for BrainData submodules.
[`validation`](#nltools.data.braindata.validation) | Validation utilities for BrainData class.

**Classes:**

Name | Description
---- | -----------
[`BrainData`](#nltools.data.braindata.BrainData) | BrainData is a class to represent neuroimaging data in python as a vector



### Attributes

### Classes#### `nltools.data.braindata.BrainData`

```python
BrainData(data = None, Y = None, X = None, mask = None, masker = None, **kwargs)
```

Bases: <code>[object](#object)</code>

BrainData is a class to represent neuroimaging data in python as a vector
rather than a 3-dimensional matrix. This makes it easier to perform data
manipulation and analyses.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | nibabel data instance or list of files | <code>None</code>
`Y` |  | Pandas DataFrame of training labels | <code>None</code>
`X` |  | Pandas DataFrame Design Matrix for running univariate models | <code>None</code>
`mask` |  | binary nifti file to mask brain data | <code>None</code>
`masker` |  | nilearn masker object (e.g., ROI or searchlight extractor). Default uses voxel-level masking. | <code>None</code>
`**kwargs` |  | Additional keyword arguments passed to NiftiMasker | <code>{}</code>

**Functions:**

Name | Description
---- | -----------
[`align`](#nltools.data.braindata.BrainData.align) | Align BrainData instance to target object using functional alignment.
[`append`](#nltools.data.braindata.BrainData.append) | Append data to BrainData instance.
[`apply_mask`](#nltools.data.braindata.BrainData.apply_mask) | Mask BrainData instance using nilearn functionality.
[`astype`](#nltools.data.braindata.BrainData.astype) | Cast BrainData.data as type.
[`bootstrap`](#nltools.data.braindata.BrainData.bootstrap) | Bootstrap statistics using efficient online algorithms.
[`compute_contrasts`](#nltools.data.braindata.BrainData.compute_contrasts) | Compute contrasts from fitted GLM results.
[`copy`](#nltools.data.braindata.BrainData.copy) | Create a deep copy of a BrainData instance.
[`create_empty`](#nltools.data.braindata.BrainData.create_empty) | Create a copy of BrainData with empty data array.
[`cv`](#nltools.data.braindata.BrainData.cv) | Create a cross-validation pipeline for this BrainData.
[`decompose`](#nltools.data.braindata.BrainData.decompose) | Decompose BrainData object.
[`detrend`](#nltools.data.braindata.BrainData.detrend) | Remove linear trend from each voxel.
[`distance`](#nltools.data.braindata.BrainData.distance) | Calculate distance between images within a BrainData() instance.
[`extract_roi`](#nltools.data.braindata.BrainData.extract_roi) | Extract activity from mask or ROI atlas using NiftiLabelsMasker.
[`filter`](#nltools.data.braindata.BrainData.filter) | Apply butterworth filter to data. Wraps nilearn.signal.clean.
[`find_spikes`](#nltools.data.braindata.BrainData.find_spikes) | Identify spikes from Time Series Data.
[`fit`](#nltools.data.braindata.BrainData.fit) | Fit a model to brain imaging data.
[`icc`](#nltools.data.braindata.BrainData.icc) | Calculate voxel-wise intraclass correlation coefficient.
[`mean`](#nltools.data.braindata.BrainData.mean) | Get mean of each voxel or image.
[`median`](#nltools.data.braindata.BrainData.median) | Get median of each voxel or image.
[`multivariate_similarity`](#nltools.data.braindata.BrainData.multivariate_similarity) | Predict spatial distribution of BrainData() instance from linear
[`plot`](#nltools.data.braindata.BrainData.plot) | Plot BrainData instance using nilearn visualization or matplotlib.
[`plot_flatmap`](#nltools.data.braindata.BrainData.plot_flatmap) | Plot brain data on cortical flatmap.
[`predict`](#nltools.data.braindata.BrainData.predict) | Generate predictions using fitted model OR classify patterns (MVPA).
[`r_to_z`](#nltools.data.braindata.BrainData.r_to_z) | Apply Fisher's r to z transformation to each element of the data
[`regions`](#nltools.data.braindata.BrainData.regions) | Extract brain connected regions into separate regions.
[`regress`](#nltools.data.braindata.BrainData.regress) | Deprecated: Use fit(model='glm', X=design_matrix) instead.
[`resample_to`](#nltools.data.braindata.BrainData.resample_to) | Resample BrainData to match target image or resolution.
[`scale`](#nltools.data.braindata.BrainData.scale) | Scale data via mean scaling.
[`similarity`](#nltools.data.braindata.BrainData.similarity) | Calculate similarity of BrainData() instance with single
[`smooth`](#nltools.data.braindata.BrainData.smooth) | Apply spatial smoothing using nilearn smooth_img().
[`standardize`](#nltools.data.braindata.BrainData.standardize) | Standardize BrainData() instance.
[`std`](#nltools.data.braindata.BrainData.std) | Get standard deviation of each voxel or image.
[`sum`](#nltools.data.braindata.BrainData.sum) | Get sum of each voxel or image.
[`temporal_resample`](#nltools.data.braindata.BrainData.temporal_resample) | Resample BrainData timeseries to a new target frequency or number of samples.
[`threshold`](#nltools.data.braindata.BrainData.threshold) | Threshold BrainData instance with optional cluster filtering.
[`to_nifti`](#nltools.data.braindata.BrainData.to_nifti) | Convert BrainData Instance into Nifti Object.
[`transform_pairwise`](#nltools.data.braindata.BrainData.transform_pairwise) | Transform data into pairwise comparisons.
[`upload_neurovault`](#nltools.data.braindata.BrainData.upload_neurovault) | Upload Data to Neurovault.  Will add any columns in self.X to image
[`write`](#nltools.data.braindata.BrainData.write) | Write out BrainData object to Nifti or HDF5 File.
[`z_to_r`](#nltools.data.braindata.BrainData.z_to_r) | Convert z score back into r value for each element of data object.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`X`](#nltools.data.braindata.BrainData.X) |  | 
[`Y`](#nltools.data.braindata.BrainData.Y) |  | 
[`data`](#nltools.data.braindata.BrainData.data) |  | 
[`design_matrix`](#nltools.data.braindata.BrainData.design_matrix) |  | 
[`dtype`](#nltools.data.braindata.BrainData.dtype) |  | Get data type of BrainData.data.
[`is_empty`](#nltools.data.braindata.BrainData.is_empty) | <code>[bool](#bool)</code> | Check if BrainData.data is empty.
[`masker`](#nltools.data.braindata.BrainData.masker) |  | 
[`shape`](#nltools.data.braindata.BrainData.shape) |  | Get images by voxels shape.
[`verbose`](#nltools.data.braindata.BrainData.verbose) |  | 

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | Neuroimaging data. Can be: - None (empty BrainData) - BrainData object - List of BrainData objects or file paths - File path (str/Path) to .nii/.nii.gz/.h5/.hdf5 - nibabel Nifti1Image object - URL to download data from | <code>None</code>
`mask` |  | Brain mask. Can be None (uses MNI template), a nibabel Nifti1Image, a file path (str/Path) to a mask file, or a template name string like ``'2mm-MNI152-2009c'`` (version: 'fsl' for default/, 'a' for nilearn/, 'c' for fmriprep/). | <code>None</code>
`masker` |  | nilearn masker object (e.g. ROI or searchlight extractor). Default will load data as voxels. | <code>None</code>
`resample` | <code>bool, default=True</code> | Whether to automatically resample data to mask space. If True, data is resampled to match mask spatial characteristics. If False, data must already be in mask space. Default True preserves backward compatibility with v0.5.1. | *required*
`interpolation` | <code>str, default='auto'</code> | Interpolation method for resampling. Options: 'auto' (detect based on data type; uses 'nearest' for discrete data like atlases/masks and 'continuous' for stat maps), 'nearest' (nearest-neighbor, preserves discrete values), 'linear' (linear interpolation), 'continuous' (higher-order spline, use for stat maps). | *required*
`**kwargs` |  | Additional arguments passed to NiftiMasker. | <code>{}</code>



##### Attributes###### `nltools.data.braindata.BrainData.X`

```python
X = X
```

###### `nltools.data.braindata.BrainData.Y`

```python
Y = Y
```

###### `nltools.data.braindata.BrainData.data`

```python
data = np.array([])
```

###### `nltools.data.braindata.BrainData.design_matrix`

```python
design_matrix = None
```

###### `nltools.data.braindata.BrainData.dtype`

```python
dtype
```

Get data type of BrainData.data.

###### `nltools.data.braindata.BrainData.is_empty`

```python
is_empty: bool
```

Check if BrainData.data is empty.

###### `nltools.data.braindata.BrainData.masker`

```python
masker = masker
```

###### `nltools.data.braindata.BrainData.shape`

```python
shape
```

Get images by voxels shape.

###### `nltools.data.braindata.BrainData.verbose`

```python
verbose = kwargs.pop('verbose', False)
```



##### Functions###### `nltools.data.braindata.BrainData.align`

```python
align(target, method = 'procrustes', axis = 0, *args, **kwargs)
```

Align BrainData instance to target object using functional alignment.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` |  | (BrainData) object to align to. | *required*
`method` |  | (str) alignment method to use ['probabilistic_srm','deterministic_srm','procrustes'] | <code>'procrustes'</code>
`axis` |  | (int) axis to align on | <code>0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (dict) a dictionary containing transformed object, transformation matrix, and the shared response matrix

**Examples:**

```pycon
>>> out = data.align(target, method='procrustes')
>>> out = data.align(target, method='probabilistic_srm', n_features=None)
```

###### `nltools.data.braindata.BrainData.append`

```python
append(data, ignore_attrs = False, **kwargs)
```

Append data to BrainData instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | BrainData instance to append. | *required*
`ignore_attrs` |  | (bool) If True, skip concatenation of X and Y     attributes. Useful when appending images where .X or .Y     have different column counts. Default False. | <code>False</code>
`kwargs` |  | Optional arguments passed to pandas concat for X/Y. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | New appended BrainData instance.

###### `nltools.data.braindata.BrainData.apply_mask`

```python
apply_mask(mask, resample_mask_to_brain = False)
```

Mask BrainData instance using nilearn functionality.

Note target data will be resampled into the same space as the mask. If you would like the mask
resampled into the BrainData space, then set resample_mask_to_brain=True.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask` |  | (BrainData or nifti object) mask to apply to BrainData object. | *required*
`resample_mask_to_brain` |  | (bool) Will resample mask to brain space before applying mask (default=False). | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`masked` |  | (BrainData) masked BrainData object

###### `nltools.data.braindata.BrainData.astype`

```python
astype(dtype)
```

Cast BrainData.data as type.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dtype` |  | datatype to convert | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with new datatype

###### `nltools.data.braindata.BrainData.bootstrap`

```python
bootstrap(stat, n_samples = 5000, save_boots = False, n_jobs = -1, random_state = None, percentiles = (2.5, 97.5), X_test = None, **kwargs)
```

Bootstrap statistics using efficient online algorithms.

Uses memory-efficient bootstrap infrastructure with CPU parallelization or GPU acceleration.
Supports simple aggregation statistics and fitted model statistics (Ridge).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stat` |  | (str) Statistic to bootstrap. Options: Simple stats ('mean', 'median', 'std', 'sum', 'min', 'max') or Model stats ('weights' requires fitted Ridge model, 'predict' requires fitted Ridge model + X_test). | *required*
`n_samples` |  | (int) Number of bootstrap iterations. Default: 5000 | <code>5000</code>
`save_boots` |  | (bool) If True, store all bootstrap samples. Default: False | <code>False</code>
`n_jobs` |  | (int) Number of CPU cores for parallelization. -1 means all CPUs. | <code>-1</code>
`random_state` |  | (int, optional) Random seed for reproducibility | <code>None</code>
`percentiles` |  | (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5) | <code>(2.5, 97.5)</code>
`X_test` |  | (np.ndarray, optional) Test features for 'predict' bootstrap. | <code>None</code>
`**kwargs` |  | Additional parameters (backend, max_gpu_memory_gb, etc.) | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or dict: - For simple stats: Returns BrainData with bootstrap mean - For model stats: Returns dict with keys: 'mean', 'std', 'Z', 'p',   'ci_lower', 'ci_upper' (all BrainData objects) - If ``save_boots=True``: Returns dict with 'samples' key containing all samples

**Examples:**

```pycon
>>> boot = brain.bootstrap(stat='mean', n_samples=1000)
>>> brain.fit(X=dm, model='ridge', alpha=1.0)
>>> boot = brain.bootstrap(stat='weights', n_samples=1000)
```

###### `nltools.data.braindata.BrainData.compute_contrasts`

```python
compute_contrasts(contrasts, contrast_type = 't')
```

Compute contrasts from fitted GLM results.

This method computes contrasts as linear combinations of the GLM beta coefficients.
Must be called after .fit(model='glm', X=design_matrix) has been run.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`contrasts` |  | Can be:<br>- str: A string specifying the contrast using column names   e.g., "conditionA - conditionB" or "2*conditionA - conditionB - conditionC" - dict: Dictionary with contrast names as keys and contrast strings/vectors as values   e.g., {"main_effect": "conditionA - conditionB", "interaction": [1, -1, -1, 1]} - array: Numeric contrast vector matching the number of regressors   e.g., [1, -1, 0, 0] for a 4-regressor model | *required*
`contrast_type` | <code>[str](#str)</code> | Type of contrast statistic ('t' or 'F'). Default: 't' Note: Currently only 't' contrasts are supported. | <code>'t'</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or dict: If single contrast, returns BrainData object with contrast map.                If multiple contrasts (dict input), returns dict of BrainData objects.

**Examples:**

```pycon
>>> brain.fit(model='glm', X=design_matrix)
>>> contrast1 = brain.compute_contrasts([0, 1, -1])
>>> contrast2 = brain.compute_contrasts("conditionA - conditionB")
>>> results = brain.compute_contrasts({
...     "A_vs_B": "conditionA - conditionB",
...     "avg_effect": [0, 0.5, 0.5],
... })
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- String contrasts support coefficients: "2*A - B" or "0.5*A + 0.5*B"
- Column names must match design matrix columns exactly (case-sensitive)
- Contrast weights should sum to zero for proper inference in most cases

</details>

###### `nltools.data.braindata.BrainData.copy`

```python
copy()
```

Create a deep copy of a BrainData instance.

All attributes including data, fitted models, and results are deep copied.
Use this when you need a complete independent copy.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Deep copied instance

###### `nltools.data.braindata.BrainData.create_empty`

```python
create_empty()
```

Create a copy of BrainData with empty data array.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | A copy of this object with an empty data array.

###### `nltools.data.braindata.BrainData.cv`

```python
cv(k: int | None = None, scheme: str = 'kfold', split_by: str | None = None, groups: np.ndarray | None = None, random_state: int | None = None, **kwargs: int | None) -> BrainDataPipeline
```

Create a cross-validation pipeline for this BrainData.

Returns a Pipeline object that enables fluent, chainable transforms
with cross-validation. Terminal methods like .predict() execute the
pipeline and return results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`k` | <code>[int](#int) \| None</code> | Number of folds (for kfold scheme). Defaults to 5. | <code>None</code>
`scheme` | <code>[str](#str)</code> | CV scheme type. Options: - 'kfold': k-fold cross-validation (default) - 'loro': leave-one-run-out (requires split_by='runs' or groups) - 'bootstrap': bootstrap with out-of-bag test sets | <code>'kfold'</code>
`split_by` | <code>[str](#str) \| None</code> | Attribute name for group splits (e.g., 'runs'). | <code>None</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Explicit group labels for CV splits. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`**kwargs` |  | Additional arguments passed to CVScheme. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainDataPipeline` | <code>[BrainDataPipeline](#nltools.data.braindata.pipeline.BrainDataPipeline)</code> | A pipeline object for method chaining.

**Examples:**

```pycon
>>> result = brain.cv(k=5).predict(y, algorithm='ridge')
>>> result = brain.cv(scheme='loro', groups=run_labels).predict(y)
```

###### `nltools.data.braindata.BrainData.decompose`

```python
decompose(algorithm = 'pca', axis = 'voxels', n_components = None, *args, **kwargs)
```

Decompose BrainData object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`algorithm` |  | (str) Algorithm to perform decomposition         types=['pca','ica','nnmf','fa','dictionary','kernelpca'] | <code>'pca'</code>
`axis` |  | dimension to decompose ['voxels','images'] | <code>'voxels'</code>
`n_components` |  | (int) number of components. If None then retain         as many as possible. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`output` |  | a dictionary of decomposition parameters

###### `nltools.data.braindata.BrainData.detrend`

```python
detrend(method = 'linear')
```

Remove linear trend from each voxel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` |  | ('linear','constant', optional) type of detrending | <code>'linear'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (BrainData) detrended BrainData instance

###### `nltools.data.braindata.BrainData.distance`

```python
distance(metric = 'euclidean', **kwargs)
```

Calculate distance between images within a BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` |  | (str) type of distance metric (can use any scipy.spatial.distance     metric supported by cdist) | <code>'euclidean'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | Pairwise distance matrix.

###### `nltools.data.braindata.BrainData.extract_roi`

```python
extract_roi(mask, metric = 'mean', n_components = None)
```

Extract activity from mask or ROI atlas using NiftiLabelsMasker.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask` |  | BrainData, nibabel image, or file path. Can be:<br>  - Binary mask (extracts from single ROI)   - Labeled atlas (extracts from multiple ROIs) | *required*
`metric` |  | Extraction method ('mean', 'median', 'pca'). Default: 'mean' | <code>'mean'</code>
`n_components` |  | If metric='pca', number of components to return | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | For binary mask: scalar or 1D array.
 | For labeled atlas: 1D or 2D array, or PCA components.

**Examples:**

```pycon
>>> roi_values = brain.extract_roi(binary_mask)
>>> atlas_values = brain.extract_roi(atlas_mask)
>>> components = brain.extract_roi(mask, metric='pca', n_components=5)
```

###### `nltools.data.braindata.BrainData.filter`

```python
filter(sampling_freq = None, high_pass = None, low_pass = None, **kwargs)
```

Apply butterworth filter to data. Wraps nilearn.signal.clean.

<details class="note" open markdown="1">
<summary>Note</summary>

Unlike nilearn's default, does not detrend or standardize. Pass
detrend=True or standardize=True via kwargs to enable.

</details>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sampling_freq` |  | Sampling freq in hertz (i.e. 1 / TR) | <code>None</code>
`high_pass` |  | High pass cutoff frequency | <code>None</code>
`low_pass` |  | Low pass cutoff frequency | <code>None</code>
`**kwargs` |  | Additional arguments passed to nilearn.signal.clean | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Filtered BrainData instance

###### `nltools.data.braindata.BrainData.find_spikes`

```python
find_spikes(global_spike_cutoff = 3, diff_spike_cutoff = 3)
```

Identify spikes from Time Series Data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`global_spike_cutoff` | <code>[int](#int) or None</code> | cutoff to identify spikes in global signal in standard deviations, or None to skip. | <code>3</code>
`diff_spike_cutoff` | <code>[int](#int) or None</code> | cutoff to identify spikes in average frame difference in standard deviations, or None to skip. | <code>3</code>

**Returns:**

Type | Description
---- | -----------
 | pandas dataframe with spikes as indicator variables

###### `nltools.data.braindata.BrainData.fit`

```python
fit(model = None, X = None, cv = None, inplace = True, progress_bar = None, scale = True, scale_value = 100.0, **kwargs)
```

Fit a model to brain imaging data.

Creates and fits a model from string specification. The brain data
(self.data) is always used as the target variable. Model and results
are stored for later use with predict().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model` | <code>[str](#str)</code> | Model type: 'ridge', 'glm', or future model names | <code>None</code>
`X` | <code>[array](#array) - [like](#like) or [DataFrame](#DataFrame)</code> | Design matrix or feature matrix | <code>None</code>
`cv` | <code>int, 'auto', or sklearn CV splitter</code> | Cross-validation specification (Ridge only) | <code>None</code>
`inplace` | <code>bool, default=True</code> | If True, mutate self and return self. If False, return Fit dataclass with results (self unchanged). | <code>True</code>
`progress_bar` | <code>[bool](#bool)</code> | Display progress bar during fitting. | <code>None</code>
`scale` | <code>bool, default=True</code> | Apply grand-mean scaling before fitting. | <code>True</code>
`scale_value` | <code>float, default=100.0</code> | Target value for mean after scaling. | <code>100.0</code>
`**kwargs` | <code>[dict](#dict)</code> | Additional arguments passed to model constructor | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or Fit: If ``inplace=True``, returns self (fitted BrainData). If ``inplace=False``, returns Fit dataclass with results.

**Examples:**

```pycon
>>> brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
>>> fit = brain_data.fit(model='ridge', alpha=1.0, X=features, inplace=False)
```

###### `nltools.data.braindata.BrainData.icc`

```python
icc(n_subjects, n_sessions, icc_type = 'icc2', parallel = None, n_jobs = -1, max_gpu_memory_gb = 4.0)
```

Calculate voxel-wise intraclass correlation coefficient.

ICC Formulas based on Shrout & Fleiss (1979).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_subjects` |  | Number of subjects in the data | *required*
`n_sessions` |  | Number of sessions per subject | *required*
`icc_type` |  | Type of ICC ('icc1', 'icc2', 'icc3'). Default: 'icc2' | <code>'icc2'</code>
`parallel` |  | Parallelization method (None, 'cpu', 'gpu') | <code>None</code>
`n_jobs` |  | Number of CPU cores (-1 = all cores) | <code>-1</code>
`max_gpu_memory_gb` |  | GPU memory budget in GB | <code>4.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with ICC map (shape: (1, n_voxels))

**Examples:**

```pycon
>>> icc_map = data.icc(n_subjects=20, n_sessions=3, icc_type='icc2')
```

###### `nltools.data.braindata.BrainData.mean`

```python
mean(axis = 0)
```

Get mean of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array) | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Mean values.

###### `nltools.data.braindata.BrainData.median`

```python
median(axis = 0)
```

Get median of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array) | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Median values.

###### `nltools.data.braindata.BrainData.multivariate_similarity`

```python
multivariate_similarity(images, method = 'ols')
```

Predict spatial distribution of BrainData() instance from linear
combination of other BrainData() instances or Nibabel images.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`images` |  | BrainData instance of weight map | *required*
`method` | <code>[str](#str)</code> | Regression method. Default: 'ols'. | <code>'ols'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | dictionary of regression statistics in BrainData instances {'beta','t','p','df','residual'}

###### `nltools.data.braindata.BrainData.plot`

```python
plot(kind = 'glass', thr_upper = None, thr_lower = None, threshold = None, cut_coords = None, cmap = None, bg_img = None, ax = None, title = None, colorbar = True, save = None, stat = 'mean', **kwargs)
```

Plot BrainData instance using nilearn visualization or matplotlib.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`kind` | <code>[str](#str)</code> | Visualization type: 'glass', 'slices', 'timeseries', 'histogram' | <code>'glass'</code>
`thr_upper` | <code>[str](#str) / [float](#float)</code> | Upper threshold. | <code>None</code>
`thr_lower` | <code>[str](#str) / [float](#float)</code> | Lower threshold. | <code>None</code>
`threshold` | <code>[float](#float)</code> | Convenience parameter for thresholding. | <code>None</code>
`cut_coords` | <code>[list](#list)</code> | Cut coordinates for multi-slice views. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Colormap name. | <code>None</code>
`bg_img` | <code>str/nibabel image</code> | Background image. | <code>None</code>
`ax` | <code>[Axes](#matplotlib.axes.Axes)</code> | Matplotlib axis. | <code>None</code>
`title` | <code>[str](#str)</code> | Plot title. | <code>None</code>
`colorbar` | <code>[bool](#bool)</code> | Whether to show colorbar. Default: True. | <code>True</code>
`save` | <code>[str](#str)</code> | Path to save figure(s). | <code>None</code>
`stat` | <code>[str](#str)</code> | Statistic for timeseries plots. Default: 'mean'. | <code>'mean'</code>
`**kwargs` |  | Additional arguments passed to nilearn plot functions. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | Display or matplotlib Figure.

###### `nltools.data.braindata.BrainData.plot_flatmap`

```python
plot_flatmap(threshold = None, cmap = 'RdBu_r', vmax = None, vmin = None, template = 'fsaverage5', with_curvature = True, curvature_contrast = 0.5, curvature_brightness = 0.5, colorbar = True, colorbar_orientation = 'horizontal', figsize = (12, 6), title = None, radius = 3.0, interpolation = 'linear', axes = None, save = None)
```

Plot brain data on cortical flatmap.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`threshold` | <code>[float](#float)</code> | Values below this absolute threshold are masked. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Matplotlib colormap. Default: 'RdBu_r'. | <code>'RdBu_r'</code>
`vmax` | <code>[float](#float)</code> | Maximum value for colormap. | <code>None</code>
`vmin` | <code>[float](#float)</code> | Minimum value for colormap. | <code>None</code>
`template` | <code>[str](#str)</code> | Freesurfer surface resolution. Default: 'fsaverage5'. | <code>'fsaverage5'</code>
`with_curvature` | <code>[bool](#bool)</code> | Show sulcal/gyral pattern. Default: True. | <code>True</code>
`curvature_contrast` | <code>[float](#float)</code> | Contrast of curvature overlay. Default: 0.5. | <code>0.5</code>
`curvature_brightness` | <code>[float](#float)</code> | Mean brightness of curvature overlay. Default: 0.5. | <code>0.5</code>
`colorbar` | <code>[bool](#bool)</code> | Show colorbar. Default: True. | <code>True</code>
`colorbar_orientation` | <code>[str](#str)</code> | 'horizontal' or 'vertical'. Default: 'horizontal'. | <code>'horizontal'</code>
`figsize` | <code>[tuple](#tuple)</code> | Figure size as (width, height). Default: (12, 6). | <code>(12, 6)</code>
`title` | <code>[str](#str)</code> | Figure title. | <code>None</code>
`radius` | <code>[float](#float)</code> | Sampling radius in mm. Default: 3.0. | <code>3.0</code>
`interpolation` | <code>[str](#str)</code> | Interpolation method. Default: 'linear'. | <code>'linear'</code>
`axes` | <code>[Axes](#matplotlib.axes.Axes)</code> | Existing axes to plot on. | <code>None</code>
`save` | <code>[str](#str)</code> | File path to save figure. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

###### `nltools.data.braindata.BrainData.predict`

```python
predict(X: np.ndarray | None = None, y: np.ndarray | None = None, method: str = 'whole_brain', estimator: str = 'svm', cv: str = 5, groups: np.ndarray | None = None, roi_mask: np.ndarray | None = None, radius: float = 10.0, scoring: str = 'accuracy', standardize: bool = True, n_jobs: int = -1, show_progress: bool = True)
```

Generate predictions using fitted model OR classify patterns (MVPA).

Two modes:
1. **Timeseries prediction** (X provided): Use fitted ridge model to predict voxel responses.
2. **MVPA decoding** (y provided): Train a classifier to predict labels from brain patterns.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray) \| None</code> | Features for timeseries prediction, shape (n_samples, n_features). | <code>None</code>
`y` | <code>[ndarray](#numpy.ndarray) \| None</code> | Labels for MVPA decoding, shape (n_samples,). | <code>None</code>
`method` | <code>[str](#str)</code> | Decoding method - 'whole_brain', 'searchlight', or 'roi'. | <code>'whole_brain'</code>
`estimator` |  | Classifier ('svm', 'logistic', 'ridge', 'lda', or sklearn estimator). | <code>'svm'</code>
`cv` |  | Cross-validation specification. | <code>5</code>
`groups` | <code>[ndarray](#numpy.ndarray) \| None</code> | Group labels for CV. | <code>None</code>
`roi_mask` |  | Atlas/parcellation for ROI-based decoding. | <code>None</code>
`radius` | <code>[float](#float)</code> | Searchlight radius in mm (default 10.0). | <code>10.0</code>
`scoring` | <code>[str](#str)</code> | Metric for evaluation. | <code>'accuracy'</code>
`standardize` | <code>[bool](#bool)</code> | Z-score features before classification (default True). | <code>True</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs (-1 = all cores). | <code>-1</code>
`show_progress` | <code>[bool](#bool)</code> | Show progress bar for searchlight. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Predicted timeseries or accuracy map.

**Examples:**

```pycon
>>> brain_data.fit(model='ridge', X=features)
>>> predictions = brain_data.predict(X=new_features)
>>> accuracy = brain_data.predict(y=labels, method='searchlight')
```

###### `nltools.data.braindata.BrainData.r_to_z`

```python
r_to_z()
```

Apply Fisher's r to z transformation to each element of the data
object.

###### `nltools.data.braindata.BrainData.regions`

```python
regions(min_region_size = 1350, extract_type = 'local_regions', smoothing_fwhm = 6, is_mask = False)
```

Extract brain connected regions into separate regions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`min_region_size` | <code>[int](#int)</code> | Minimum volume in mm3 for a region to be kept. | <code>1350</code>
`extract_type` | <code>[str](#str)</code> | Type of extraction method                 ['connected_components', 'local_regions']. | <code>'local_regions'</code>
`smoothing_fwhm` | <code>[scalar](#scalar)</code> | Smooth an image to extract more sparser regions. | <code>6</code>
`is_mask` | <code>[bool](#bool)</code> | Whether to treat as boolean mask. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with extracted ROIs as data.

###### `nltools.data.braindata.BrainData.regress`

```python
regress(design_matrix = None, noise_model = 'ols', mode = None, **kwargs)
```

Deprecated: Use fit(model='glm', X=design_matrix) instead.

.. deprecated:: 0.6.0
    Use :meth:`fit` with ``model='glm'`` instead.

###### `nltools.data.braindata.BrainData.resample_to`

```python
resample_to(img = None, resolution = None, interpolation = None)
```

Resample BrainData to match target image or resolution.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`img` |  | Target image for resampling (nibabel Nifti1Image, str/Path, or None). | <code>None</code>
`resolution` |  | Target voxel size in mm (float/int for isotropic, or None). | <code>None</code>
`interpolation` |  | Interpolation method ('nearest', 'linear', 'continuous', or None). | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | New BrainData instance with resampled data

###### `nltools.data.braindata.BrainData.scale`

```python
scale(scale_val = 100.0, axis = None)
```

Scale data via mean scaling.

Two scaling modes are available:

- **Grand-mean scaling** (axis=None, default): Divides all values by the
  global mean across all voxels and timepoints.

- **Voxel-wise scaling** (axis=0): Divides each voxel's time-series by
  its own temporal mean.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`scale_val` |  | (int/float) Target value for the mean after scaling. Default 100. | <code>100.0</code>
`axis` |  | (int or None) None for grand-mean scaling (default), 0 for voxel-wise scaling. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | New BrainData instance with scaled data.

###### `nltools.data.braindata.BrainData.similarity`

```python
similarity(image, method = 'correlation')
```

Calculate similarity of BrainData() instance with single
BrainData or Nibabel image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`image` |  | (BrainData, nifti) image to evaluate similarity | *required*
`method` |  | (str) Type of similarity     ['correlation','dot_product','cosine'] | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
 | float or np.ndarray: Similarity value(s).

###### `nltools.data.braindata.BrainData.smooth`

```python
smooth(fwhm)
```

Apply spatial smoothing using nilearn smooth_img().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fwhm` |  | (float) full width half maximum of gaussian spatial filter | *required*

**Returns:**

Type | Description
---- | -----------
 | BrainData instance (copy with smoothed data)

###### `nltools.data.braindata.BrainData.standardize`

```python
standardize(axis = 0, method = 'center')
```

Standardize BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>[int](#int)</code> | 0 standardizes each voxel across observations (default). 1 standardizes each observation across voxels. | <code>0</code>
`method` | <code>[str](#str)</code> | 'center' subtracts the mean (default). 'zscore' subtracts the mean and divides by standard deviation. | <code>'center'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Standardized BrainData instance.

###### `nltools.data.braindata.BrainData.std`

```python
std(axis = 0)
```

Get standard deviation of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array) | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Standard deviation values.

###### `nltools.data.braindata.BrainData.sum`

```python
sum(axis = 0)
```

Get sum of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array) | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Sum values.

###### `nltools.data.braindata.BrainData.temporal_resample`

```python
temporal_resample(sampling_freq = None, target = None, target_type = 'hz')
```

Resample BrainData timeseries to a new target frequency or number of samples.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sampling_freq` |  | (float) sampling frequency of data in hertz | <code>None</code>
`target` |  | (float) upsampling target | <code>None</code>
`target_type` |  | (str) type of target can be [samples,seconds,hz] | <code>'hz'</code>

**Returns:**

Type | Description
---- | -----------
 | upsampled BrainData instance

###### `nltools.data.braindata.BrainData.threshold`

```python
threshold(upper = None, lower = None, binarize = False, coerce_nan = True, cluster_threshold = 0)
```

Threshold BrainData instance with optional cluster filtering.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`upper` |  | (float or str) Upper cutoff for thresholding. | <code>None</code>
`lower` |  | (float or str) Lower cutoff for thresholding. | <code>None</code>
`binarize` | <code>[bool](#bool)</code> | return binarized image. Default False. | <code>False</code>
`coerce_nan` | <code>[bool](#bool)</code> | coerce nan values to 0s. Default True. | <code>True</code>
`cluster_threshold` | <code>[int](#int)</code> | Minimum cluster size in voxels. Default 0. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | Thresholded BrainData object.

###### `nltools.data.braindata.BrainData.to_nifti`

```python
to_nifti()
```

Convert BrainData Instance into Nifti Object.

**Returns:**

Type | Description
---- | -----------
 | nibabel.Nifti1Image: Brain data as a NIfTI image.

###### `nltools.data.braindata.BrainData.transform_pairwise`

```python
transform_pairwise()
```

Transform data into pairwise comparisons.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance transformed into pairwise comparisons

###### `nltools.data.braindata.BrainData.upload_neurovault`

```python
upload_neurovault(access_token = None, collection_name = None, collection_id = None, img_type = None, img_modality = None, **kwargs)
```

Upload Data to Neurovault.  Will add any columns in self.X to image
    metadata. Index will be used as image name.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`access_token` |  | (str, Required) Neurovault api access token | <code>None</code>
`collection_name` |  | (str, Optional) name of new collection to create | <code>None</code>
`collection_id` |  | (int, Optional) neurovault collection_id if adding images             to existing collection | <code>None</code>
`img_type` |  | (str, Required) Neurovault map_type | <code>None</code>
`img_modality` |  | (str, Required) Neurovault image modality | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`collection` |  | (pd.DataFrame) neurovault collection information

###### `nltools.data.braindata.BrainData.write`

```python
write(file_name)
```

Write out BrainData object to Nifti or HDF5 File.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>[str](#str) or [Path](#Path)</code> | Output file path (.nii/.nii.gz for NIfTI, .h5/.hdf5 for HDF5). | *required*

###### `nltools.data.braindata.BrainData.z_to_r`

```python
z_to_r()
```

Convert z score back into r value for each element of data object.



### Functions

### Modules#### `nltools.data.braindata.analysis`

BrainData analysis functions.

Standalone functions extracted from BrainData class methods for similarity,
distance, masking, ROI extraction, ICC, filtering, thresholding, decomposition,
alignment, smoothing, and other analytical operations.

**Functions:**

Name | Description
---- | -----------
[`align`](#nltools.data.braindata.analysis.align) | Align BrainData instance to target object using functional alignment
[`apply_mask`](#nltools.data.braindata.analysis.apply_mask) | Mask BrainData instance using nilearn functionality.
[`check_masks`](#nltools.data.braindata.analysis.check_masks) | Check to make sure masks are the same for each dataset and if not create a union mask
[`decompose`](#nltools.data.braindata.analysis.decompose) | Decompose BrainData object
[`detrend_data`](#nltools.data.braindata.analysis.detrend_data) | Remove linear trend from each voxel
[`distance`](#nltools.data.braindata.analysis.distance) | Calculate distance between images within a BrainData() instance.
[`extract_roi`](#nltools.data.braindata.analysis.extract_roi) | Extract activity from mask or ROI atlas using NiftiLabelsMasker.
[`filter_data`](#nltools.data.braindata.analysis.filter_data) | Apply butterworth filter to data. Wraps nilearn.signal.clean.
[`find_spikes_data`](#nltools.data.braindata.analysis.find_spikes_data) | Function to identify spikes from Time Series Data
[`icc`](#nltools.data.braindata.analysis.icc) | Calculate voxel-wise intraclass correlation coefficient for data within
[`multivariate_similarity`](#nltools.data.braindata.analysis.multivariate_similarity) | Predict spatial distribution of BrainData() instance from linear
[`r_to_z`](#nltools.data.braindata.analysis.r_to_z) | Apply Fisher's r to z transformation to each element of the data
[`regions`](#nltools.data.braindata.analysis.regions) | Extract brain connected regions into separate regions.
[`scale_data`](#nltools.data.braindata.analysis.scale_data) | Scale data via mean scaling.
[`similarity`](#nltools.data.braindata.analysis.similarity) | Calculate similarity of BrainData() instance with single
[`smooth`](#nltools.data.braindata.analysis.smooth) | Apply spatial smoothing using nilearn smooth_img()
[`standardize`](#nltools.data.braindata.analysis.standardize) | Standardize BrainData() instance.
[`temporal_resample`](#nltools.data.braindata.analysis.temporal_resample) | Resample BrainData timeseries to a new target frequency or number of samples
[`threshold_data`](#nltools.data.braindata.analysis.threshold_data) | Threshold BrainData instance with optional cluster filtering.
[`transform_pairwise_data`](#nltools.data.braindata.analysis.transform_pairwise_data) | Transform BrainData into pairwise comparisons.
[`z_to_r`](#nltools.data.braindata.analysis.z_to_r) | Convert z score back into r value for each element of data object.



##### Functions###### `nltools.data.braindata.analysis.align`

```python
align(bd, target, method = 'procrustes', axis = 0, *args, **kwargs)
```

Align BrainData instance to target object using functional alignment

Alignment type can be hyperalignment or Shared Response Model. When
using hyperalignment, `target` image can be another subject or an
already estimated common model. When using SRM, `target` must be a previously
estimated common model stored as a numpy array. Transformed data can be back
projected to original data using Transformation matrix.

See nltools.stats.align for aligning multiple BrainData instances

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`target` |  | (BrainData) object to align to. | *required*
`method` |  | (str) alignment method to use ['probabilistic_srm','deterministic_srm','procrustes'] | <code>'procrustes'</code>
`axis` |  | (int) axis to align on (default: 0) | <code>0</code>
`**kwargs` |  | Additional keyword arguments passed to the alignment function. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (dict) a dictionary containing transformed object, transformation matrix, and the shared response matrix

**Examples:**

- Hyperalign using procrustes transform:
    >>> out = data.align(target, method='procrustes')
- Align using shared response model:
    >>> out = data.align(target, method='probabilistic_srm', n_features=None)
- Project aligned data into original data:
    >>> original_data = np.dot(out['transformed'].data,out['transformation_matrix'].T)

###### `nltools.data.braindata.analysis.apply_mask`

```python
apply_mask(bd, mask, resample_mask_to_brain = False)
```

Mask BrainData instance using nilearn functionality.

Note target data will be resampled into the same space as the mask. If you would like the mask
resampled into the BrainData space, then set resample_mask_to_brain=True.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`mask` |  | (BrainData or nifti object) mask to apply to BrainData object. | *required*
`resample_mask_to_brain` |  | (bool) Will resample mask to brain space before applying mask (default=False). | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`masked` |  | (BrainData) masked BrainData object

<details class="note" open markdown="1">
<summary>Note</summary>

Uses nilearn.masking.apply_mask for efficient, validated masking.
Simplified from 47-line manual implementation to leverage nilearn's
Cython-optimized code with better validation and memory management.

</details>

###### `nltools.data.braindata.analysis.check_masks`

```python
check_masks(bd, image)
```

Check to make sure masks are the same for each dataset and if not create a union mask

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance | *required*
`image` |  | BrainData instance to compare masks with | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` |  | (data2, image2) arrays with compatible masks

###### `nltools.data.braindata.analysis.decompose`

```python
decompose(bd, algorithm = 'pca', axis = 'voxels', n_components = None, *args, **kwargs)
```

Decompose BrainData object

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`algorithm` |  | (str) Algorithm to perform decomposition         types=['pca','ica','nnmf','fa','dictionary','kernelpca'] | <code>'pca'</code>
`axis` |  | dimension to decompose ['voxels','images'] | <code>'voxels'</code>
`n_components` |  | (int) number of components. If None then retain         as many as possible (default: None). | <code>None</code>
`**kwargs` |  | Additional keyword arguments passed to the decomposition algorithm. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`output` |  | a dictionary of decomposition parameters

###### `nltools.data.braindata.analysis.detrend_data`

```python
detrend_data(bd, method = 'linear')
```

Remove linear trend from each voxel

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`method` |  | ('linear','constant', optional) type of detrending | <code>'linear'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (BrainData) detrended BrainData instance

###### `nltools.data.braindata.analysis.distance`

```python
distance(bd, metric = 'euclidean', **kwargs)
```

Calculate distance between images within a BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`metric` |  | (str) type of distance metric (can use any scipy.spatial.distance     metric supported by cdist, e.g., 'euclidean', 'cityblock', 'cosine',     'correlation', 'hamming', 'jaccard', etc.) | <code>'euclidean'</code>
`**kwargs` |  | Additional arguments passed to scipy.spatial.distance.cdist. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dist` |  | (Adjacency) Outputs a 2D distance matrix.

###### `nltools.data.braindata.analysis.extract_roi`

```python
extract_roi(bd, mask, metric = 'mean', n_components = None)
```

Extract activity from mask or ROI atlas using NiftiLabelsMasker.

This method now uses nilearn's NiftiLabelsMasker for efficient ROI extraction
when dealing with labeled atlases (multiple ROIs).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`mask` |  | BrainData, nibabel image, or file path. Can be:<br>  - Binary mask (extracts from single ROI)   - Labeled atlas (extracts from multiple ROIs) | *required*
`metric` |  | Extraction method ('mean', 'median', 'pca'). Default: 'mean'     Note: 'median' and 'pca' require additional computation after extraction | <code>'mean'</code>
`n_components` |  | If metric='pca', number of components to return | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | For binary mask:<br>- Single image: scalar value - Multiple images: 1D array of values
 | For labeled atlas:<br>- Single image: 1D array (one value per ROI) - Multiple images: 2D array (images x ROIs) - If metric='pca': returns components array

**Examples:**

```pycon
>>> # Extract mean from binary mask
>>> roi_values = brain.extract_roi(binary_mask)
>>> # Extract from atlas
>>> atlas_values = brain.extract_roi(atlas_mask)
>>> # PCA extraction
>>> components = brain.extract_roi(mask, metric='pca', n_components=5)
```

###### `nltools.data.braindata.analysis.filter_data`

```python
filter_data(bd, sampling_freq = None, high_pass = None, low_pass = None, **kwargs)
```

Apply butterworth filter to data. Wraps nilearn.signal.clean.

Does not default to detrending and standardizing like nilearn
implementation, but this can be overridden using kwargs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`sampling_freq` |  | Sampling freq in hertz (i.e. 1 / TR). Default: None. | <code>None</code>
`high_pass` |  | High pass cutoff frequency. Default: None. | <code>None</code>
`low_pass` |  | Low pass cutoff frequency. Default: None. | <code>None</code>
`**kwargs` |  | Additional arguments passed to nilearn.signal.clean       Common options:       - confounds: Confound timeseries to remove       - sample_mask: Volumes to exclude (scrubbing)       - detrend: Enable detrending (default False)       - standardize: Enable standardization (default False)       - ensure_finite: Replace NaN/inf (default False) | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Filtered BrainData instance

<details class="see-also" open markdown="1">
<summary>See Also</summary>

nilearn.signal.clean documentation for all available options

</details>

###### `nltools.data.braindata.analysis.find_spikes_data`

```python
find_spikes_data(bd, global_spike_cutoff = 3, diff_spike_cutoff = 3)
```

Function to identify spikes from Time Series Data

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`global_spike_cutoff` |  | (int,None) cutoff to identify spikes in global signal                  in standard deviations, None indicates do not calculate. | <code>3</code>
`diff_spike_cutoff` |  | (int,None) cutoff to identify spikes in average frame difference                  in standard deviations, None indicates do not calculate. | <code>3</code>

**Returns:**

Type | Description
---- | -----------
 | pd.DataFrame: DataFrame with spikes as indicator variables.

###### `nltools.data.braindata.analysis.icc`

```python
icc(bd, n_subjects, n_sessions, icc_type = 'icc2', parallel = None, n_jobs = -1, max_gpu_memory_gb = 4.0)
```

Calculate voxel-wise intraclass correlation coefficient for data within
    BrainData class.

Computes ICC for each voxel independently, making it highly parallelizable.
Supports GPU acceleration for large voxel counts.

ICC Formulas are based on:
Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
assessing rater reliability. Psychological bulletin, 86(2), 420.

icc1:  x_ij = mu + beta_j + w_ij
icc2/3:  x_ij = mu + alpha_i + beta_j + (ab)_ij + epsilon_ij

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`n_subjects` |  | Number of subjects in the data | *required*
`n_sessions` |  | Number of sessions per subject | *required*
`icc_type` |  | Type of ICC to calculate     - 'icc1': One-way random effects (subjects random, sessions treated as interchangeable)     - 'icc2': Two-way random effects (subjects and sessions random) (default)     - 'icc3': Two-way mixed effects (subjects random, sessions fixed) | <code>'icc2'</code>
`parallel` |  | Parallelization method     - None: Single-threaded vectorized NumPy (default, memory efficient)     - 'cpu': CPU parallelization via joblib (for medium-sized problems, 1K-10K voxels)     - 'gpu': GPU acceleration via PyTorch (recommended for large voxel counts >10K, 10-50x speedup) | <code>None</code>
`n_jobs` |  | Number of CPU cores (-1 = all cores). Only used when parallel='cpu' | <code>-1</code>
`max_gpu_memory_gb` |  | GPU memory budget in GB. Only used when parallel='gpu' | <code>4.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with ICC map (shape: (1, n_voxels))

**Examples:**

```pycon
>>> # Typical test-retest reliability analysis
>>> data = BrainData(...)  # Shape: (60, 238955) = 20 subjects x 3 sessions
>>> icc_map = data.icc(n_subjects=20, n_sessions=3, icc_type='icc2')
>>> icc_map.shape
(1, 238955)
>>> # Visualize ICC map
>>> icc_map.plot()
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

Data must be organized such that n_images = n_subjects * n_sessions.
Images should be ordered as: [subject1_session1, subject1_session2, ...,
subject2_session1, ...]

</details>

###### `nltools.data.braindata.analysis.multivariate_similarity`

```python
multivariate_similarity(bd, images, method = 'ols')
```

Predict spatial distribution of BrainData() instance from linear
combination of other BrainData() instances or Nibabel images

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance of data to be applied | *required*
`images` |  | BrainData instance of weight map | *required*
`method` | <code>[str](#str)</code> | Regression method. Default: 'ols'. | <code>'ols'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | dictionary of regression statistics in BrainData instances {'beta','t','p','df','residual'}

###### `nltools.data.braindata.analysis.r_to_z`

```python
r_to_z(bd)
```

Apply Fisher's r to z transformation to each element of the data
object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Transformed BrainData instance.

###### `nltools.data.braindata.analysis.regions`

```python
regions(bd, min_region_size = 1350, extract_type = 'local_regions', smoothing_fwhm = 6, is_mask = False)
```

Extract brain connected regions into separate regions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`min_region_size` | <code>[int](#int)</code> | Minimum volume in mm3 for a region to be                 kept. | <code>1350</code>
`extract_type` | <code>[str](#str)</code> | Type of extraction method                 ['connected_components', 'local_regions'].                 If 'connected_components', each component/region                 in the image is extracted automatically by                 labelling each region based upon the presence of                 unique features in their respective regions.                 If 'local_regions', each component/region is                 extracted based on their maximum peak value to                 define a seed marker and then using random                 walker segementation algorithm on these                 markers for region separation. | <code>'local_regions'</code>
`smoothing_fwhm` | <code>[scalar](#scalar)</code> | Smooth an image to extract more sparser                 regions. Only works for extract_type                 'local_regions'. | <code>6</code>
`is_mask` | <code>[bool](#bool)</code> | Whether the BrainData instance should be treated             as a boolean mask and if so, calls             connected_label_regions instead. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with extracted ROIs as data.

###### `nltools.data.braindata.analysis.scale_data`

```python
scale_data(bd, scale_val = 100.0, axis = None)
```

Scale data via mean scaling.

Two scaling modes are available:

- **Grand-mean scaling** (axis=None, default): Divides all values by the
  global mean across all voxels and timepoints. This is consistent with
  FSL and SPM behavior. Use scale_val=10000 for FSL-style scaling.

- **Voxel-wise scaling** (axis=0): Divides each voxel's time-series by
  its own temporal mean. This is AFNI-style scaling and can be useful
  when voxels have very different baseline intensities. Voxels with
  zero or near-zero mean are set to zero to avoid NaN/Inf.

When scale_val=100 (default), the result can be interpreted as something
akin to (but not exactly) "percent signal change."

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`scale_val` |  | (int/float) Target value for the mean after scaling. Default 100. | <code>100.0</code>
`axis` |  | (int or None) Axis along which to compute the mean. None for grand-mean scaling (default, FSL/SPM style). 0 for voxel-wise scaling (AFNI style, each voxel scaled by its own temporal mean). | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | New BrainData instance with scaled data.

**Examples:**

```pycon
>>> # Grand-mean scaling (default)
>>> scaled = brain.scale(100.0)
>>>
>>> # Voxel-wise scaling (AFNI style)
>>> scaled = brain.scale(100.0, axis=0)
```

###### `nltools.data.braindata.analysis.similarity`

```python
similarity(bd, image, method = 'correlation')
```

Calculate similarity of BrainData() instance with single
BrainData or Nibabel image

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`image` |  | (BrainData, nifti)  image to evaluate similarity | *required*
`method` |  | (str) Type of similarity     ['correlation', 'pearson', 'rank_correlation', 'spearman', 'dot_product', 'cosine'] | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
 | np.ndarray: Similarity values.

###### `nltools.data.braindata.analysis.smooth`

```python
smooth(bd, fwhm)
```

Apply spatial smoothing using nilearn smooth_img()

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`fwhm` |  | (float) full width half maximum of gaussian spatial filter | *required*

**Returns:**

Type | Description
---- | -----------
 | BrainData instance (copy with smoothed data)

###### `nltools.data.braindata.analysis.standardize`

```python
standardize(bd, axis = 0, method = 'center')
```

Standardize BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`axis` |  | 0 for observations 1 for voxels (default: 0) | <code>0</code>
`method` |  | ['center','zscore'] (default: 'center') | <code>'center'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Standardized BrainData instance.

###### `nltools.data.braindata.analysis.temporal_resample`

```python
temporal_resample(bd, sampling_freq = None, target = None, target_type = 'hz')
```

Resample BrainData timeseries to a new target frequency or number of samples
using Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) interpolation.
This function can up- or down-sample data.

Note: this function can use quite a bit of RAM.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`sampling_freq` |  | (float) sampling frequency of data in hertz (default: None) | <code>None</code>
`target` |  | (float) upsampling target (default: None) | <code>None</code>
`target_type` |  | (str) type of target can be [samples,seconds,hz] (default: 'hz') | <code>'hz'</code>

**Returns:**

Type | Description
---- | -----------
 | upsampled BrainData instance

###### `nltools.data.braindata.analysis.threshold_data`

```python
threshold_data(bd, upper = None, lower = None, binarize = False, coerce_nan = True, cluster_threshold = 0)
```

Threshold BrainData instance with optional cluster filtering.

Provide upper and lower values or percentages to perform two-sided
thresholding. Binarize will return a mask image respecting thresholds
if provided, otherwise respecting every non-zero value.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`upper` |  | (float or str) Upper cutoff for thresholding. If string     will interpret as percentile; can be None for one-sided     thresholding. | <code>None</code>
`lower` |  | (float or str) Lower cutoff for thresholding. If string     will interpret as percentile; can be None for one-sided     thresholding. | <code>None</code>
`bd` |  | BrainData instance. | *required*
`binarize` | <code>[bool](#bool)</code> | return binarized image respecting thresholds if     provided, otherwise binarize on every non-zero value;     default False | <code>False</code>
`coerce_nan` | <code>[bool](#bool)</code> | coerce nan values to 0s; default True | <code>True</code>
`cluster_threshold` | <code>[int](#int)</code> | Minimum cluster size in voxels. If > 0, uses     nilearn.image.threshold_img with cluster filtering.     Band-pass filtering (both upper AND lower) not supported     with cluster thresholding. Default 0 (disabled). | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | Thresholded BrainData object.

<details class="note" open markdown="1">
<summary>Note</summary>

When cluster_threshold=0 (default), uses fast path for basic thresholding.
When cluster_threshold>0, uses nilearn for cluster filtering.
Band-pass filtering (unique nltools feature) preserved when cluster_threshold=0.

</details>

###### `nltools.data.braindata.analysis.transform_pairwise_data`

```python
transform_pairwise_data(bd)
```

Transform BrainData into pairwise comparisons.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance transformed into pairwise comparisons.

###### `nltools.data.braindata.analysis.z_to_r`

```python
z_to_r(bd)
```

Convert z score back into r value for each element of data object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Transformed BrainData instance.

#### `nltools.data.braindata.bootstrap`

Bootstrap functions extracted from BrainData methods.

**Functions:**

Name | Description
---- | -----------
[`bootstrap`](#nltools.data.braindata.bootstrap.bootstrap) | Bootstrap statistics using efficient online algorithms.
[`convert_bootstrap_results_to_brain_data`](#nltools.data.braindata.bootstrap.convert_bootstrap_results_to_brain_data) | Convert bootstrap results dictionary to BrainData format.



##### Functions###### `nltools.data.braindata.bootstrap.bootstrap`

```python
bootstrap(bd, stat, n_samples = 5000, save_boots = False, n_jobs = -1, random_state = None, percentiles = (2.5, 97.5), X_test = None, **kwargs)
```

Bootstrap statistics using efficient online algorithms.

Uses memory-efficient bootstrap infrastructure with CPU parallelization or GPU acceleration.
Supports simple aggregation statistics and fitted model statistics (Ridge).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`stat` |  | (str) Statistic to bootstrap. Options: Simple stats ('mean', 'median', 'std', 'sum', 'min', 'max') or Model stats ('weights' requires fitted Ridge model, 'predict' requires fitted Ridge model + X_test). | *required*
`n_samples` |  | (int) Number of bootstrap iterations. Default: 5000 | <code>5000</code>
`save_boots` |  | (bool) If True, store all bootstrap samples (memory intensive).        Default: False | <code>False</code>
`n_jobs` |  | (int) Number of CPU cores for parallelization. Default: -1 (all CPUs). | <code>-1</code>
`random_state` |  | (int, optional) Random seed for reproducibility | <code>None</code>
`percentiles` |  | (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5) | <code>(2.5, 97.5)</code>
`X_test` |  | (np.ndarray, optional) Test features for 'predict' bootstrap.    Required if stat='predict' | <code>None</code>
`backend` |  | (str, optional) Backend for computation ('numpy', 'torch', 'auto').     If 'torch' and GPU available, uses optimized GPU acceleration with     inline Ridge computation (no CPU round-trips). Default: None (CPU). | *required*
`max_gpu_memory_gb` |  | (float) Maximum GPU memory to use in GB. Default: 4.0 | *required*
`**kwargs` |  | Additional keyword arguments passed to the underlying bootstrap computation functions. Also accepts ``backend`` (str) and ``max_gpu_memory_gb`` (float). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or dict: - For simple stats: Returns BrainData with bootstrap mean - For model stats: Returns dict with keys: 'mean', 'std', 'Z', 'p',   'ci_lower', 'ci_upper' (all BrainData objects) - If ``save_boots=True``: Returns dict with 'samples' key containing all samples

**Examples:**

```pycon
>>> # Simple aggregation
>>> boot = brain.bootstrap(stat='mean', n_samples=1000)
>>> assert isinstance(boot, BrainData)
```

```pycon
>>> # Ridge weights bootstrap (CPU)
>>> brain.fit(X=dm, model='ridge', alpha=1.0)
>>> boot = brain.bootstrap(stat='weights', n_samples=1000)
>>> assert 'mean' in boot
>>> assert isinstance(boot['mean'], BrainData)
```

```pycon
>>> # Ridge weights bootstrap (GPU accelerated)
>>> brain.fit(X=dm, model='ridge', alpha=1.0)
>>> boot = brain.bootstrap(stat='weights', n_samples=1000, backend='torch')
>>> assert 'mean' in boot
>>> assert isinstance(boot['mean'], BrainData)
```

```pycon
>>> # Ridge predict bootstrap
>>> brain.fit(X=dm, model='ridge', alpha=1.0)
>>> boot = brain.bootstrap(stat='predict', X_test=X_new, n_samples=1000)
>>> assert 'mean' in boot
>>> assert isinstance(boot['mean'], BrainData)
```

<details class="note" open markdown="1">
<summary>Note</summary>

This method replaces the deprecated `summarize_bootstrap()` function from
`nltools.stats`. To reproduce `summarize_bootstrap()` functionality:

**Old API (deprecated):**
>>> from nltools.stats import summarize_bootstrap
>>> bootstrap_samples = BrainData(list_of_samples)  # Multiple samples
>>> result = summarize_bootstrap(bootstrap_samples, save_weights=False)
>>> # Returns: {'mean': BrainData, 'Z': BrainData, 'p': BrainData}

**New API (recommended):**
>>> # Option 1: Use BrainData.bootstrap() for generating bootstrap samples
>>> boot = brain.bootstrap(stat='mean', n_samples=1000, save_boots=False)
>>> # Returns BrainData with bootstrap mean
>>> # To get Z and p, use stat='weights' or 'predict' which returns dict

>>> # Option 2: For existing bootstrap samples (BrainData with multiple images),
>>> # use OnlineBootstrapStats directly:
>>> from nltools.algorithms.inference.bootstrap import OnlineBootstrapStats
>>> stats = OnlineBootstrapStats(shape=(brain.shape[1],), save_samples=False)
>>> for sample in bootstrap_samples:  # Iterate over samples
...     stats.update(sample.data)
>>> result = stats.get_results()
>>> # Returns: {'mean': array, 'std': array, 'Z': array, 'p': array,
>>> #           'ci_lower': array, 'ci_upper': array}
>>> # Convert to BrainData if needed:
>>> mean_brain = shallow_copy(brain)
>>> mean_brain.data = result['mean']

</details>

###### `nltools.data.braindata.bootstrap.convert_bootstrap_results_to_brain_data`

```python
convert_bootstrap_results_to_brain_data(bd, result, save_boots = False, return_dict = False)
```

Convert bootstrap results dictionary to BrainData format.

Helper method to convert numpy arrays from bootstrap functions into
BrainData objects or dicts of BrainData objects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`result` |  | (dict) Result dictionary from bootstrap function with keys:     'mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper', and optionally 'samples' | *required*
`save_boots` |  | (bool) If True, include 'samples' key in output | <code>False</code>
`return_dict` |  | (bool) If True, always return dict even for simple stats.         If False, return BrainData for simple stats (when save_boots=False) | <code>False</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or dict: - If return_dict=False and save_boots=False: Returns BrainData with mean - Otherwise: Returns dict with BrainData objects for each statistic

#### `nltools.data.braindata.cache`

Disk-based caching infrastructure for expensive computations.

This module provides a general-purpose caching system for nltools, designed to
be reused across various computationally expensive operations like searchlight
neighborhoods, ISC, and SRM.

<details class="example" open markdown="1">
<summary>Example</summary>

>>> from nltools.data.braindata.cache import CacheManager, hash_mask
>>> import nibabel as nib
>>>
>>> # Hash a mask for cache key generation
>>> mask = nib.load("mask.nii.gz")
>>> mask_hash = hash_mask(mask)
>>>
>>> # Use cache manager for searchlight neighborhoods
>>> cache = CacheManager("searchlight")
>>> if not cache.exists(f"{mask_hash}_10mm"):
...     # Compute expensive operation
...     result = compute_something()
...     cache.save(f"{mask_hash}_10mm", data=result)
>>> else:
...     result = cache.load(f"{mask_hash}_10mm")["data"]

</details>

**Classes:**

Name | Description
---- | -----------
[`CacheManager`](#nltools.data.braindata.cache.CacheManager) | Manages disk-based caching for expensive computations.

**Functions:**

Name | Description
---- | -----------
[`clear_cache`](#nltools.data.braindata.cache.clear_cache) | Clear the nltools cache.
[`get_cache_dir`](#nltools.data.braindata.cache.get_cache_dir) | Get the nltools cache directory.
[`hash_mask`](#nltools.data.braindata.cache.hash_mask) | Compute a stable hash for a NIfTI mask image.



##### Classes###### `nltools.data.braindata.cache.CacheManager`

```python
CacheManager(category: str = 'general')
```

Manages disk-based caching for expensive computations.

CacheManager provides a simple key-value interface for caching numpy arrays
to disk. It organizes cached files by category (e.g., "searchlight", "isc")
in separate subdirectories.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`category` | <code>[str](#str)</code> | Category name for organizing cached files (e.g., "searchlight") | <code>'general'</code>

<details class="example" open markdown="1">
<summary>Example</summary>

>>> cache = CacheManager("searchlight")
>>>
>>> # Check if something is cached
>>> if cache.exists("mykey"):
...     data = cache.load("mykey")
... else:
...     result = expensive_computation()
...     cache.save("mykey", adjacency=result, metadata=metadata)
...     data = {"adjacency": result, "metadata": metadata}

</details>

**Functions:**

Name | Description
---- | -----------
[`clear`](#nltools.data.braindata.cache.CacheManager.clear) | Clear all cached files in this category.
[`delete`](#nltools.data.braindata.cache.CacheManager.delete) | Delete a cached file.
[`exists`](#nltools.data.braindata.cache.CacheManager.exists) | Check if a cache key exists.
[`get_path`](#nltools.data.braindata.cache.CacheManager.get_path) | Get the file path for a cache key.
[`list_keys`](#nltools.data.braindata.cache.CacheManager.list_keys) | List all cached keys in this category.
[`load`](#nltools.data.braindata.cache.CacheManager.load) | Load cached data.
[`save`](#nltools.data.braindata.cache.CacheManager.save) | Save arrays to cache.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`cache_dir`](#nltools.data.braindata.cache.CacheManager.cache_dir) |  | 
[`category`](#nltools.data.braindata.cache.CacheManager.category) |  | 



####### Attributes######## `nltools.data.braindata.cache.CacheManager.cache_dir`

```python
cache_dir = get_cache_dir() / category
```

######## `nltools.data.braindata.cache.CacheManager.category`

```python
category = category
```



####### Functions######## `nltools.data.braindata.cache.CacheManager.clear`

```python
clear() -> int
```

Clear all cached files in this category.

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of files deleted

######## `nltools.data.braindata.cache.CacheManager.delete`

```python
delete(key: str, ext: str = '.npz') -> bool
```

Delete a cached file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>[str](#str)</code> | Cache key | *required*
`ext` | <code>[str](#str)</code> | File extension | <code>'.npz'</code>

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if file was deleted, False if it didn't exist

######## `nltools.data.braindata.cache.CacheManager.exists`

```python
exists(key: str, ext: str = '.npz') -> bool
```

Check if a cache key exists.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>[str](#str)</code> | Cache key | *required*
`ext` | <code>[str](#str)</code> | File extension (default: ".npz") | <code>'.npz'</code>

**Returns:**

Type | Description
---- | -----------
<code>[bool](#bool)</code> | True if cached file exists

######## `nltools.data.braindata.cache.CacheManager.get_path`

```python
get_path(key: str, ext: str = '.npz') -> Path
```

Get the file path for a cache key.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>[str](#str)</code> | Cache key | *required*
`ext` | <code>[str](#str)</code> | File extension (default: ".npz") | <code>'.npz'</code>

**Returns:**

Type | Description
---- | -----------
<code>[Path](#pathlib.Path)</code> | Path to the cache file

######## `nltools.data.braindata.cache.CacheManager.list_keys`

```python
list_keys(ext: str = '.npz') -> list[str]
```

List all cached keys in this category.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`ext` | <code>[str](#str)</code> | File extension to match | <code>'.npz'</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[str](#str)]</code> | List of cache keys (without extension)

######## `nltools.data.braindata.cache.CacheManager.load`

```python
load(key: str) -> dict | None
```

Load cached data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>[str](#str)</code> | Cache key | *required*

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict) \| None</code> | Dictionary of cached arrays, or None if not cached

######## `nltools.data.braindata.cache.CacheManager.save`

```python
save(key: str, compressed: bool = True, **arrays: bool) -> Path
```

Save arrays to cache.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`key` | <code>[str](#str)</code> | Cache key | *required*
`compressed` | <code>[bool](#bool)</code> | If True, use compressed npz format (smaller but slower) | <code>True</code>
`**arrays` |  | Named arrays to cache | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[Path](#pathlib.Path)</code> | Path to saved cache file



##### Functions###### `nltools.data.braindata.cache.clear_cache`

```python
clear_cache(category: str | None = None) -> int
```

Clear the nltools cache.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`category` | <code>[str](#str) \| None</code> | If provided, only clear this category. Otherwise clear all. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of files deleted

###### `nltools.data.braindata.cache.get_cache_dir`

```python
get_cache_dir() -> Path
```

Get the nltools cache directory.

Returns ~/.nltools/cache, creating it if necessary.

**Returns:**

Type | Description
---- | -----------
<code>[Path](#pathlib.Path)</code> | Path to cache directory

###### `nltools.data.braindata.cache.hash_mask`

```python
hash_mask(mask_img: 'Nifti1Image') -> str
```

Compute a stable hash for a NIfTI mask image.

The hash is based on the mask's shape, affine transformation, and the
actual voxel positions. This ensures that masks with the same shape but
different voxel locations (or different affines) produce different hashes.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask_img` | <code>'Nifti1Image'</code> | NIfTI image to hash (typically a binary mask) | *required*

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | 16-character hexadecimal hash string

<details class="example" open markdown="1">
<summary>Example</summary>

>>> import nibabel as nib
>>> mask = nib.load("mask.nii.gz")
>>> hash_mask(mask)
'a1b2c3d4e5f6g7h8'

</details>

#### `nltools.data.braindata.io`

BrainData I/O and loading functions.

Standalone functions extracted from BrainData class methods for mask initialization,
data loading (from files, lists, URLs, HDF5, other BrainData objects), resampling,
writing, and uploading.

**Functions:**

Name | Description
---- | -----------
[`check_space_match`](#nltools.data.braindata.io.check_space_match) | Check if data and mask are in same space.
[`detect_and_update_mask`](#nltools.data.braindata.io.detect_and_update_mask) | Detect best matching template from data and update mask if mask was None.
[`detect_space`](#nltools.data.braindata.io.detect_space) | Detect if mask is in MNI space or native space.
[`get_interpolation`](#nltools.data.braindata.io.get_interpolation) | Get the interpolation method to use for a given image.
[`initialize_mask`](#nltools.data.braindata.io.initialize_mask) | Initialize the mask and NiftiMasker.
[`load_from_brain_data`](#nltools.data.braindata.io.load_from_brain_data) | Load data from another BrainData object.
[`load_from_file`](#nltools.data.braindata.io.load_from_file) | Load data from file path or nibabel object.
[`load_from_h5`](#nltools.data.braindata.io.load_from_h5) | Load data from HDF5 file.
[`load_from_list`](#nltools.data.braindata.io.load_from_list) | Load data from a list of BrainData objects or file paths.
[`load_from_url`](#nltools.data.braindata.io.load_from_url) | Load data from URL.
[`resample_to`](#nltools.data.braindata.io.resample_to) | Resample BrainData to match target image or resolution.
[`to_nifti`](#nltools.data.braindata.io.to_nifti) | Convert BrainData instance to a nibabel NIfTI image.
[`upload_neurovault`](#nltools.data.braindata.io.upload_neurovault) | Upload data to NeuroVault.
[`warn_if_resampling`](#nltools.data.braindata.io.warn_if_resampling) | Warn about resampling if verbose=True and resample=True.
[`write_brain_data`](#nltools.data.braindata.io.write_brain_data) | Write out BrainData object to Nifti or HDF5 File.



##### Functions###### `nltools.data.braindata.io.check_space_match`

```python
check_space_match(data_img, mask_img)
```

Check if data and mask are in same space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data_img` |  | nibabel Nifti1Image object | *required*
`mask_img` |  | nibabel Nifti1Image object (mask) | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`bool` |  | True if spaces match (no resampling needed), False otherwise

###### `nltools.data.braindata.io.detect_and_update_mask`

```python
detect_and_update_mask(bd, data_img)
```

Detect best matching template from data and update mask if mask was None.

Also handles resampling if needed based on the resample kwarg.

This function is called during data loading to auto-detect template when mask=None.
After detecting or falling back to a template, it checks if resampling is needed
and resamples the data_img accordingly.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`data_img` |  | nibabel Nifti1Image object from which to detect template | *required*

**Returns:**

Type | Description
---- | -----------
 | nibabel.Nifti1Image: The data_img, possibly resampled to match the mask

###### `nltools.data.braindata.io.detect_space`

```python
detect_space(bd, mask)
```

Detect if mask is in MNI space or native space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance (unused, kept for API consistency). | *required*
`mask` |  | nibabel Nifti1Image object | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` |  | 'mni' if mask is MNI template, 'native' otherwise

###### `nltools.data.braindata.io.get_interpolation`

```python
get_interpolation(bd, img)
```

Get the interpolation method to use for a given image.

Resolves 'auto' to either 'nearest' or 'continuous' based on data type.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`img` |  | nibabel image to check (used when interpolation='auto') | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` |  | Interpolation method. When 'auto', resolves to 'nearest' or 'continuous' based on data type. Otherwise returns the instance's configured interpolation setting.

###### `nltools.data.braindata.io.initialize_mask`

```python
initialize_mask(bd, mask, **kwargs)
```

Initialize the mask and NiftiMasker.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`mask` |  | Brain mask as nibabel object, file path, template name string, or None. Template name strings supported: '{res}mm-MNI152-2009{version}' (e.g., '2mm-MNI152-2009c', '3mm-MNI152-2009a', '2mm-MNI152-2009fsl') | *required*
`**kwargs` |  | Additional arguments passed to NiftiMasker. | <code>{}</code>

###### `nltools.data.braindata.io.load_from_brain_data`

```python
load_from_brain_data(bd, brain_data, mask = None)
```

Load data from another BrainData object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`brain_data` |  | BrainData object to copy from. | *required*
`mask` |  | Optional mask to use. If None, uses mask from brain_data. | <code>None</code>

###### `nltools.data.braindata.io.load_from_file`

```python
load_from_file(bd, data)
```

Load data from file path or nibabel object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`data` |  | File path or nibabel object. | *required*

###### `nltools.data.braindata.io.load_from_h5`

```python
load_from_h5(bd, file_path, mask)
```

Load data from HDF5 file.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`file_path` |  | Path to HDF5 file. | *required*
`mask` |  | User-specified mask (to determine if we should load mask from file). | *required*

###### `nltools.data.braindata.io.load_from_list`

```python
load_from_list(bd, data_list)
```

Load data from a list of BrainData objects or file paths.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`data_list` |  | List of BrainData objects or file paths. | *required*

###### `nltools.data.braindata.io.load_from_url`

```python
load_from_url(bd, url)
```

Load data from URL.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`url` |  | URL to download data from. | *required*

###### `nltools.data.braindata.io.resample_to`

```python
resample_to(bd, img = None, resolution = None, interpolation = None)
```

Resample BrainData to match target image or resolution.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`img` |  | Target image for resampling. Can be: - nibabel Nifti1Image object - str/Path to .nii/.nii.gz file - None (if using resolution parameter) | <code>None</code>
`resolution` |  | Target voxel size in mm. Can be: - float/int: Isotropic resolution (e.g., 2.0 = 2mm^3) - None (if using img parameter) | <code>None</code>
`interpolation` |  | Interpolation method for resampling. Can be: - None (default): Uses instance's interpolation setting - 'nearest': Nearest-neighbor (for atlases, masks, labels) - 'linear': Linear interpolation - 'continuous': Higher-order spline (for stat maps) | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | New BrainData instance with resampled data

###### `nltools.data.braindata.io.to_nifti`

```python
to_nifti(bd)
```

Convert BrainData instance to a nibabel NIfTI image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*

**Returns:**

Type | Description
---- | -----------
 | nibabel.Nifti1Image: Brain data in volumetric NIfTI format.

###### `nltools.data.braindata.io.upload_neurovault`

```python
upload_neurovault(bd, access_token = None, collection_name = None, collection_id = None, img_type = None, img_modality = None, **kwargs)
```

Upload data to NeuroVault.

Adds any columns in bd.X to image metadata. Index will be used as image name.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`access_token` | <code>[str](#str)</code> | NeuroVault API access token. Required. | <code>None</code>
`collection_name` | <code>[str](#str)</code> | Name of new collection to create. | <code>None</code>
`collection_id` | <code>[int](#int)</code> | NeuroVault collection ID if adding images to an existing collection. | <code>None</code>
`img_type` | <code>[str](#str)</code> | NeuroVault map type. Required. | <code>None</code>
`img_modality` | <code>[str](#str)</code> | NeuroVault image modality. Required. | <code>None</code>
`**kwargs` |  | Additional keyword arguments passed to the NeuroVault API. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | NeuroVault collection information.

###### `nltools.data.braindata.io.warn_if_resampling`

```python
warn_if_resampling(bd, context = '')
```

Warn about resampling if verbose=True and resample=True.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`context` | <code>[str](#str)</code> | Context string to include in warning. Default: empty string. | <code>''</code>

###### `nltools.data.braindata.io.write_brain_data`

```python
write_brain_data(bd, file_name)
```

Write out BrainData object to Nifti or HDF5 File.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`file_name` | <code>[str](#str) or [Path](#pathlib.Path)</code> | Output file path. Supports .nii/.nii.gz (NIfTI) and .h5/.hdf5 (HDF5) formats. | *required*

#### `nltools.data.braindata.modeling`

BrainData modeling functions.

Standalone functions extracted from BrainData class methods for model fitting,
cross-validation, GLM estimation, Ridge regression, and contrast computation.

**Functions:**

Name | Description
---- | -----------
[`compute_contrasts`](#nltools.data.braindata.modeling.compute_contrasts) | Compute contrasts from fitted GLM results.
[`compute_ridge_cv`](#nltools.data.braindata.modeling.compute_ridge_cv) | Compute cross-validation results for Ridge regression.
[`cv`](#nltools.data.braindata.modeling.cv) | Create a cross-validation pipeline for this BrainData.
[`fit`](#nltools.data.braindata.modeling.fit) | Fit a model to brain imaging data.
[`fit_glm`](#nltools.data.braindata.modeling.fit_glm) | Fit GLM model and extract results (same logic as current regress()).
[`fit_ridge`](#nltools.data.braindata.modeling.fit_ridge) | Fit Ridge model and extract results.
[`parse_contrast_string`](#nltools.data.braindata.modeling.parse_contrast_string) | Parse a contrast string into a numeric contrast vector.
[`regress`](#nltools.data.braindata.modeling.regress) | Deprecated: Use fit(model='glm', X=design_matrix) instead.
[`to_fit_dataclass`](#nltools.data.braindata.modeling.to_fit_dataclass) | Convert BrainData fit results to Fit dataclass.



##### Functions###### `nltools.data.braindata.modeling.compute_contrasts`

```python
compute_contrasts(bd, contrasts, contrast_type = 't')
```

Compute contrasts from fitted GLM results.

This method computes contrasts as linear combinations of the GLM beta coefficients.
Must be called after .fit(model='glm', X=design_matrix) has been run.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`contrasts` |  | Can be:<br>- str: A string specifying the contrast using column names   e.g., "conditionA - conditionB" or "2*conditionA - conditionB - conditionC" - dict: Dictionary with contrast names as keys and contrast strings/vectors as values   e.g., {"main_effect": "conditionA - conditionB", "interaction": [1, -1, -1, 1]} - array: Numeric contrast vector matching the number of regressors   e.g., [1, -1, 0, 0] for a 4-regressor model | *required*
`contrast_type` | <code>[str](#str)</code> | Type of contrast statistic ('t' or 'F'). Default: 't' Note: Currently only 't' contrasts are supported. | <code>'t'</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or dict: If single contrast, returns BrainData object with contrast map.                If multiple contrasts (dict input), returns dict of BrainData objects.

**Examples:**

```pycon
>>> # Fit GLM model
>>> design_matrix = pd.DataFrame({
...     'intercept': np.ones(n_samples),
...     'conditionA': signal_a,
...     'conditionB': signal_b
... })
>>> brain.fit(model='glm', X=design_matrix)
>>>
>>> # Simple numeric contrast: A - B
>>> contrast1 = brain.compute_contrasts([0, 1, -1])
>>>
>>> # String-based contrast (more readable)
>>> contrast2 = brain.compute_contrasts("conditionA - conditionB")
>>>
>>> # Multiple contrasts at once
>>> contrasts = {
...     "A_vs_B": "conditionA - conditionB",
...     "avg_effect": [0, 0.5, 0.5],
...     "weighted": "2*conditionA - conditionB"
... }
>>> results = brain.compute_contrasts(contrasts)
>>> # results is a dict: {"A_vs_B": BrainData, "avg_effect": BrainData, ...}
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- String contrasts support coefficients: "2*A - B" or "0.5*A + 0.5*B"
- Column names must match design matrix columns exactly (case-sensitive)
- Contrast weights should sum to zero for proper inference in most cases

</details>

###### `nltools.data.braindata.modeling.compute_ridge_cv`

```python
compute_ridge_cv(bd, X, cv, alpha = None, alphas = None, backend = 'auto', **kwargs)
```

Compute cross-validation results for Ridge regression.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`X` | <code>[ndarray](#ndarray) or [list](#list)</code> | Training features. If ndarray, shape (n_samples, n_features). If list, list of feature spaces for banded ridge. | *required*
`cv` | <code>int, 'auto', or sklearn CV splitter</code> | Cross-validation specification | *required*
`alpha` | <code>[float](#float) or [auto](#auto)</code> | Regularization strength (extracted from model if not provided) | <code>None</code>
`alphas` | <code>[array](#array) - [like](#like)</code> | Alpha values to try for alpha selection | <code>None</code>
`backend` | <code>[str](#str)</code> | Computational backend ('numpy', 'torch', 'auto'). Default: 'auto' | <code>'auto'</code>
`**kwargs` | <code>[dict](#dict)</code> | Additional arguments (currently unused, for future extensibility) | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | Dictionary containing: - 'scores': (n_folds, n_voxels) array of R-squared per fold - 'mean_score': (n_voxels,) array of mean R-squared across folds - 'predictions': BrainData of out-of-fold predictions - 'folds': (n_samples,) array of fold indices - 'best_alpha': Selected alpha (if alpha selection performed) - 'alpha_scores': (n_folds, n_alphas, n_voxels) array (if alpha selection)

###### `nltools.data.braindata.modeling.cv`

```python
cv(bd, k = None, scheme = 'kfold', split_by = None, groups = None, random_state = None, **kwargs)
```

Create a cross-validation pipeline for this BrainData.

Returns a Pipeline object that enables fluent, chainable transforms
with cross-validation. Terminal methods like .predict() execute the
pipeline and return results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`k` |  | Number of folds (for kfold scheme). Defaults to 5. | <code>None</code>
`scheme` |  | CV scheme type. Options: - 'kfold': k-fold cross-validation (default) - 'loro': leave-one-run-out (requires split_by='runs' or groups) - 'bootstrap': bootstrap with out-of-bag test sets | <code>'kfold'</code>
`split_by` |  | Attribute name for group splits (e.g., 'runs'). If provided and groups is None, will try to get groups from bd.X[split_by] if bd.X is a DataFrame. | <code>None</code>
`groups` |  | Explicit group labels for CV splits. | <code>None</code>
`random_state` |  | Random seed for reproducibility. | <code>None</code>
`**kwargs` |  | Additional arguments passed to CVScheme. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainDataPipeline` |  | A pipeline object for method chaining.

**Examples:**

```pycon
>>> # Simple 5-fold CV with prediction
>>> result = brain.cv(k=5).predict(y, algorithm='ridge')
>>> print(f"Mean score: {result.mean_score:.3f}")
```

```pycon
>>> # With preprocessing
>>> result = (brain
...     .cv(k=5)
...     .normalize()
...     .reduce(n_components=50)
...     .predict(y))
```

```pycon
>>> # Leave-one-run-out CV
>>> result = brain.cv(scheme='loro', groups=run_labels).predict(y)
```

<details class="see-also" open markdown="1">
<summary>See Also</summary>

BrainDataPipeline: For available transforms and terminal methods.
CVScheme: For CV scheme configuration details.

</details>

###### `nltools.data.braindata.modeling.fit`

```python
fit(bd, model = None, X = None, cv = None, inplace = True, progress_bar = None, scale = True, scale_value = 100.0, **kwargs)
```

Fit a model to brain imaging data.

Creates and fits a model from string specification. The brain data
(bd.data) is always used as the target variable. Model and results
are stored for later use with predict().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`model` | <code>[str](#str)</code> | Model type: 'ridge', 'glm', or future model names | <code>None</code>
`X` | <code>[array](#array) - [like](#like) or [DataFrame](#DataFrame)</code> | Design matrix or feature matrix, shape (n_samples, n_features) - For GLM: Design matrix with regressors (n_samples must match bd.data) - For Ridge: Feature matrix for prediction (n_samples must match bd.data) | <code>None</code>
`cv` | <code>int, 'auto', or sklearn CV splitter</code> | Cross-validation specification (Ridge only): - int: Number of folds for k-fold CV (returns CV scores) - 'auto': Triggers alpha selection via CV (implies alpha='auto') - sklearn CV object: Custom CV splitter (e.g., KFold(3, shuffle=True)) - None: No CV (default, backward compatible) | <code>None</code>
`inplace` | <code>bool, default=True</code> | If True, mutate bd and return bd (backward compatible). If False, return Fit dataclass with results (bd unchanged). | <code>True</code>
`progress_bar` | <code>[bool](#bool)</code> | Display progress bar during fitting. - If None: Uses bd.verbose (default) - If True: Shows progress bar for long-running operations - If False: No progress bar | <code>None</code>
`scale` | <code>bool, default=True</code> | Apply grand-mean scaling before fitting. Calls bd.scale(scale_value) which divides all values by the global mean and multiplies by scale_value. This puts data in percent signal change units, which is standard for fMRI analysis. | <code>True</code>
`scale_value` | <code>float, default=100.0</code> | Target value for mean after scaling. Only used if scale=True. | <code>100.0</code>
`**kwargs` | <code>[dict](#dict)</code> | Additional arguments passed to model constructor - Ridge: alpha, alphas, backend, random_state - Glm: noise_model, minimize_memory, etc. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or Fit: If ``inplace=True``, returns bd (fitted BrainData). If ``inplace=False``, returns Fit dataclass with results.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`The`](#nltools.data.braindata.modeling.fit.The) | <code>following are set on bd when ``inplace=True``</code> | 
[```model_```](#nltools.data.braindata.modeling.fit.``model_``) | <code>[BaseModel](#BaseModel)</code> | Fitted model instance (Ridge, Glm, etc.)
[```X_```](#nltools.data.braindata.modeling.fit.``X_``) | <code>[ndarray](#ndarray)</code> | Training data X, stored for predict() default
[```cv_results_```](#nltools.data.braindata.modeling.fit.``cv_results_``) | <code>[dict](#dict)</code> | Cross-validation results dict with keys 'scores',
[`glm_betas`](#nltools.data.braindata.modeling.fit.glm_betas) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Beta coefficients (for model='glm')
[`glm_t`](#nltools.data.braindata.modeling.fit.glm_t) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | T-statistics (for model='glm')
[`glm_p`](#nltools.data.braindata.modeling.fit.glm_p) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | P-values (for model='glm')
[`glm_se`](#nltools.data.braindata.modeling.fit.glm_se) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Standard errors (for model='glm')
[`glm_residual`](#nltools.data.braindata.modeling.fit.glm_residual) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Residuals (for model='glm')
[`glm_predicted`](#nltools.data.braindata.modeling.fit.glm_predicted) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Fitted values (for model='glm')
[`glm_r2`](#nltools.data.braindata.modeling.fit.glm_r2) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | R-squared values (for model='glm')
[`ridge_weights`](#nltools.data.braindata.modeling.fit.ridge_weights) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Model coefficients (for model='ridge')
[`ridge_fitted_values`](#nltools.data.braindata.modeling.fit.ridge_fitted_values) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | Fitted values (for model='ridge')
[`ridge_scores`](#nltools.data.braindata.modeling.fit.ridge_scores) | <code>[BrainData](#nltools.data.braindata.BrainData)</code> | R-squared scores (for model='ridge')

**Examples:**

```pycon
>>> # Old behavior (backward compatible): mutate self
>>> brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
>>> print(f"CV R2: {brain_data.cv_results_['mean_score'].mean():.3f}")
>>> weights = brain_data.ridge_weights  # Access as attribute
>>>
>>> # New behavior: return Fit dataclass (self unchanged)
>>> fit = brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features, inplace=False)
>>> assert isinstance(fit, Fit)
>>> assert 'weights' in fit.available()
>>> assert not hasattr(brain_data, 'ridge_weights')  # brain_data unchanged
>>> print(f"CV R2: {fit.cv_mean_score.mean():.3f}")
>>>
>>> # GLM with Fit dataclass
>>> fit_glm = brain_data.fit(model='glm', X=design_matrix, inplace=False)
>>> assert 'betas' in fit_glm.available()
>>> assert 't_stats' in fit_glm.available()
```

###### `nltools.data.braindata.modeling.fit_glm`

```python
fit_glm(bd, X)
```

Fit GLM model and extract results (same logic as current regress()).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`X` |  | Design matrix (DataFrame or DesignMatrix). | *required*

<details class="note" open markdown="1">
<summary>Note</summary>

Sets glm_betas, glm_t, glm_p, glm_se, glm_residual, glm_predicted,
glm_r2, and design_matrix on bd.

</details>

###### `nltools.data.braindata.modeling.fit_ridge`

```python
fit_ridge(bd, X, cv = None, **kwargs)
```

Fit Ridge model and extract results.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`X` | <code>[ndarray](#ndarray)</code> | Training features | *required*
`cv` | <code>int, 'auto', or sklearn CV splitter</code> | Cross-validation specification | <code>None</code>
`**kwargs` | <code>[dict](#dict)</code> | Additional arguments for CV (alpha, alphas, backend, etc.) | <code>{}</code>

<details class="note" open markdown="1">
<summary>Note</summary>

Sets ridge_weights, ridge_fitted_values, ridge_scores, and
cv_results_ (if cv provided) on bd.

</details>

###### `nltools.data.braindata.modeling.parse_contrast_string`

```python
parse_contrast_string(bd, contrast_str)
```

Parse a contrast string into a numeric contrast vector.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`contrast_str` | <code>[str](#str)</code> | Contrast string like "A - B" or "2*A - B - C" | *required*

**Returns:**

Type | Description
---- | -----------
 | np.array: Numeric contrast vector

###### `nltools.data.braindata.modeling.regress`

```python
regress(bd, design_matrix = None, noise_model = 'ols', mode = None, **kwargs)
```

Deprecated: Use fit(model='glm', X=design_matrix) instead.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`design_matrix` |  | Design matrix (unused, raises error). | <code>None</code>
`noise_model` |  | Noise model (unused, raises error). | <code>'ols'</code>
`mode` |  | Mode (unused, raises error). | <code>None</code>
`**kwargs` |  | Additional arguments (unused, raises error). | <code>{}</code>

###### `nltools.data.braindata.modeling.to_fit_dataclass`

```python
to_fit_dataclass(bd, model)
```

Convert BrainData fit results to Fit dataclass.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`model` | <code>[str](#str)</code> | Model type ('ridge' or 'glm') | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Fit` |  | Dataclass containing fit results

#### `nltools.data.braindata.neighborhoods`

Spatial neighborhood computation for neuroimaging analyses.

This module provides efficient computation and caching of spatial neighborhoods
(spheres) around brain voxels. It is designed to support searchlight analyses,
ISC, and other operations that require iterating over local brain regions.

The key insight is that for a given mask and radius, the neighborhood structure
is deterministic and can be cached for reuse across analyses.

<details class="example" open markdown="1">
<summary>Example</summary>

>>> import nibabel as nib
>>> from nltools.data.braindata.neighborhoods import compute_searchlight_neighborhoods
>>>
>>> mask = nib.load("mask.nii.gz")
>>> neighborhoods = compute_searchlight_neighborhoods(mask, radius_mm=10.0)
>>>
>>> # Iterate over all voxels and their neighborhoods
>>> for center_idx, neighbor_indices in neighborhoods.iter_neighborhoods():
...     # Extract data for these voxels
...     local_data = data[:, neighbor_indices]
...     result[center_idx] = analyze(local_data)

</details>

**Classes:**

Name | Description
---- | -----------
[`SphereNeighborhoods`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods) | Precomputed sphere neighborhoods for a brain mask.

**Functions:**

Name | Description
---- | -----------
[`compute_searchlight_neighborhoods`](#nltools.data.braindata.neighborhoods.compute_searchlight_neighborhoods) | Compute sphere neighborhoods for all voxels in a brain mask.



##### Classes###### `nltools.data.braindata.neighborhoods.SphereNeighborhoods`

```python
SphereNeighborhoods(adjacency: sparse.csr_matrix, mask_hash: str, radius_mm: float, n_voxels: int) -> None
```

Precomputed sphere neighborhoods for a brain mask.

This dataclass stores a sparse adjacency matrix where row i contains True
for all voxels within the specified radius of voxel i. It provides efficient
iteration over neighborhoods for searchlight-style analyses.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`adjacency`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.adjacency) | <code>[csr_matrix](#scipy.sparse.csr_matrix)</code> | Sparse CSR matrix (n_voxels, n_voxels) where adjacency[i, j] is True if voxel j is within radius of voxel i
[`mask_hash`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.mask_hash) | <code>[str](#str)</code> | Hash of the source mask for validation
[`radius_mm`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.radius_mm) | <code>[float](#float)</code> | Radius in millimeters
[`n_voxels`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.n_voxels) | <code>[int](#int)</code> | Number of voxels in the mask

<details class="example" open markdown="1">
<summary>Example</summary>

>>> neighborhoods = compute_searchlight_neighborhoods(mask, radius_mm=10.0)
>>> print(f"Mean neighborhood size: {neighborhoods.mean_size:.1f} voxels")
>>>
>>> # Get neighbors of a specific voxel
>>> neighbor_idx = neighborhoods.get_neighbors(100)
>>> print(f"Voxel 100 has {len(neighbor_idx)} neighbors")

</details>

**Functions:**

Name | Description
---- | -----------
[`get_neighborhood_size`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.get_neighborhood_size) | Get the number of voxels in a neighborhood.
[`get_neighbors`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.get_neighbors) | Get indices of all voxels in the neighborhood of a given voxel.
[`iter_neighborhoods`](#nltools.data.braindata.neighborhoods.SphereNeighborhoods.iter_neighborhoods) | Iterate over all neighborhoods.



####### Attributes######## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.adjacency`

```python
adjacency: sparse.csr_matrix
```

######## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.mask_hash`

```python
mask_hash: str
```

######## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.max_size`

```python
max_size: int
```

Maximum neighborhood size.

######## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.mean_size`

```python
mean_size: float
```

Mean neighborhood size in voxels.

######## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.min_size`

```python
min_size: int
```

Minimum neighborhood size.

######## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.n_voxels`

```python
n_voxels: int
```

######## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.radius_mm`

```python
radius_mm: float
```



####### Functions######## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.get_neighborhood_size`

```python
get_neighborhood_size(voxel_idx: int) -> int
```

Get the number of voxels in a neighborhood.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`voxel_idx` | <code>[int](#int)</code> | Index of the center voxel | *required*

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | Number of voxels in the neighborhood

######## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.get_neighbors`

```python
get_neighbors(voxel_idx: int) -> np.ndarray
```

Get indices of all voxels in the neighborhood of a given voxel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`voxel_idx` | <code>[int](#int)</code> | Index of the center voxel (0 to n_voxels-1) | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Array of voxel indices within radius of the center voxel

######## `nltools.data.braindata.neighborhoods.SphereNeighborhoods.iter_neighborhoods`

```python
iter_neighborhoods(show_progress: bool = False) -> Iterator[tuple[int, np.ndarray]]
```

Iterate over all neighborhoods.

**Yields:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[int](#int), [ndarray](#numpy.ndarray)]</code> | Tuple of (center_voxel_idx, neighbor_indices) for each voxel

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`show_progress` | <code>[bool](#bool)</code> | If True, wrap iterator with tqdm progress bar | <code>False</code>



##### Functions###### `nltools.data.braindata.neighborhoods.compute_searchlight_neighborhoods`

```python
compute_searchlight_neighborhoods(mask_img: 'Nifti1Image', radius_mm: float = 10.0, use_cache: bool = True) -> SphereNeighborhoods
```

Compute sphere neighborhoods for all voxels in a brain mask.

For each voxel in the mask, this function identifies all other voxels
within the specified radius (in millimeters). The result is cached to
disk for fast reloading in subsequent analyses.

The algorithm uses sklearn's BallTree for efficient radius queries in
world coordinates (mm), ensuring accurate neighborhoods regardless of
voxel resolution.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask_img` | <code>'Nifti1Image'</code> | NIfTI mask image defining the brain region | *required*
`radius_mm` | <code>[float](#float)</code> | Radius of spheres in millimeters (default: 10.0) | <code>10.0</code>
`use_cache` | <code>[bool](#bool)</code> | If True, cache results to ~/.nltools/cache/searchlight/ for fast reloading (default: True) | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[SphereNeighborhoods](#nltools.data.braindata.neighborhoods.SphereNeighborhoods)</code> | SphereNeighborhoods with precomputed adjacency matrix

<details class="example" open markdown="1">
<summary>Example</summary>

>>> import nibabel as nib
>>> mask = nib.load("brain_mask.nii.gz")
>>>
>>> # First call computes and caches (may take a few seconds)
>>> neighborhoods = compute_searchlight_neighborhoods(mask, radius_mm=8.0)
>>>
>>> # Subsequent calls load from cache (~50ms)
>>> neighborhoods = compute_searchlight_neighborhoods(mask, radius_mm=8.0)
>>>
>>> print(neighborhoods)
SphereNeighborhoods(n_voxels=50000, radius=8.0mm, mean_size=33.2)

</details>

<details class="notes" open markdown="1">
<summary>Notes</summary>

Cache location: ~/.nltools/cache/searchlight/{mask_hash}_{radius}mm.npz

For a typical 2mm MNI mask (~50k voxels) with 10mm radius:
- First run: ~1-2 seconds
- Cached load: ~50ms

</details>

#### `nltools.data.braindata.pipeline`

BrainData pipeline and cross-validation result classes.

**Classes:**

Name | Description
---- | -----------
[`BrainDataCVResult`](#nltools.data.braindata.pipeline.BrainDataCVResult) | Cross-validation results for BrainData pipelines.
[`BrainDataPipeline`](#nltools.data.braindata.pipeline.BrainDataPipeline) | Pipeline specialized for BrainData with CV support.



##### Classes###### `nltools.data.braindata.pipeline.BrainDataCVResult`

```python
BrainDataCVResult(fold_results: list, pipeline: list)
```

Cross-validation results for BrainData pipelines.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`fold_results`](#nltools.data.braindata.pipeline.BrainDataCVResult.fold_results) |  | 
[`mean_score`](#nltools.data.braindata.pipeline.BrainDataCVResult.mean_score) | <code>[float](#float)</code> | Mean score across folds.
[`pipeline`](#nltools.data.braindata.pipeline.BrainDataCVResult.pipeline) |  | 
[`predictions`](#nltools.data.braindata.pipeline.BrainDataCVResult.predictions) | <code>[ndarray](#numpy.ndarray)</code> | All predictions in original sample order.
[`scores`](#nltools.data.braindata.pipeline.BrainDataCVResult.scores) | <code>[ndarray](#numpy.ndarray)</code> | Per-fold prediction scores as a numpy array.
[`std_score`](#nltools.data.braindata.pipeline.BrainDataCVResult.std_score) | <code>[float](#float)</code> | Standard deviation of scores.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fold_results` | <code>[list](#list)</code> | List of dicts, one per fold, each containing 'score', 'predictions', 'train_idx', 'test_idx', and 'fitted_stack'. | *required*
`pipeline` | <code>[BrainDataPipeline](#nltools.data.braindata.pipeline.BrainDataPipeline)</code> | The pipeline that produced these results. | *required*



####### Attributes######## `nltools.data.braindata.pipeline.BrainDataCVResult.fold_results`

```python
fold_results = fold_results
```

######## `nltools.data.braindata.pipeline.BrainDataCVResult.mean_score`

```python
mean_score: float
```

Mean score across folds.

######## `nltools.data.braindata.pipeline.BrainDataCVResult.pipeline`

```python
pipeline = pipeline
```

######## `nltools.data.braindata.pipeline.BrainDataCVResult.predictions`

```python
predictions: np.ndarray
```

All predictions in original sample order.

######## `nltools.data.braindata.pipeline.BrainDataCVResult.scores`

```python
scores: np.ndarray
```

Per-fold prediction scores as a numpy array.

######## `nltools.data.braindata.pipeline.BrainDataCVResult.std_score`

```python
std_score: float
```

Standard deviation of scores.

###### `nltools.data.braindata.pipeline.BrainDataPipeline`

```python
BrainDataPipeline(brain_data, cv = None, groups = None)
```

Pipeline specialized for BrainData with CV support.

Wraps the base Pipeline to handle BrainData-specific operations
like splitting by samples and accessing the underlying data array.

**Functions:**

Name | Description
---- | -----------
[`normalize`](#nltools.data.braindata.pipeline.BrainDataPipeline.normalize) | Add normalization step.
[`pipe`](#nltools.data.braindata.pipeline.BrainDataPipeline.pipe) | Add custom sklearn transformer.
[`predict`](#nltools.data.braindata.pipeline.BrainDataPipeline.predict) | Execute pipeline with CV and return prediction results.
[`reduce`](#nltools.data.braindata.pipeline.BrainDataPipeline.reduce) | Add dimensionality reduction step.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`cv`](#nltools.data.braindata.pipeline.BrainDataPipeline.cv) |  | Cross-validation splitter for this pipeline.
[`data`](#nltools.data.braindata.pipeline.BrainDataPipeline.data) |  | Get underlying data array.
[`n_steps`](#nltools.data.braindata.pipeline.BrainDataPipeline.n_steps) | <code>[int](#int)</code> | Number of processing steps in this pipeline.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_data` |  | BrainData instance to build the pipeline on. | *required*
`cv` |  | Cross-validation splitter (e.g., CVScheme instance) or None. | <code>None</code>
`groups` | <code>[array](#array) - [like](#like)</code> | Group labels for CV splits (e.g., run IDs for leave-one-run-out). | <code>None</code>



####### Attributes######## `nltools.data.braindata.pipeline.BrainDataPipeline.cv`

```python
cv
```

Cross-validation splitter for this pipeline.

**Returns:**

Type | Description
---- | -----------
 | sklearn cross-validator or None: The cross-validation strategy
 | set for this pipeline, or None if not configured.

######## `nltools.data.braindata.pipeline.BrainDataPipeline.data`

```python
data
```

Get underlying data array.

######## `nltools.data.braindata.pipeline.BrainDataPipeline.n_steps`

```python
n_steps: int
```

Number of processing steps in this pipeline.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`int` | <code>[int](#int)</code> | The count of steps added to this pipeline.



####### Functions######## `nltools.data.braindata.pipeline.BrainDataPipeline.normalize`

```python
normalize(method: str = 'zscore', **kwargs: str) -> BrainDataPipeline
```

Add normalization step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Normalization method. Default: 'zscore'. | <code>'zscore'</code>
`**kwargs` |  | Additional arguments passed to NormalizeStep. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainDataPipeline` | <code>[BrainDataPipeline](#nltools.data.braindata.pipeline.BrainDataPipeline)</code> | New pipeline with the normalization step appended.

######## `nltools.data.braindata.pipeline.BrainDataPipeline.pipe`

```python
pipe(transformer) -> BrainDataPipeline
```

Add custom sklearn transformer.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`transformer` |  | An sklearn-compatible transformer with fit/transform methods. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainDataPipeline` | <code>[BrainDataPipeline](#nltools.data.braindata.pipeline.BrainDataPipeline)</code> | New pipeline with the custom step appended.

######## `nltools.data.braindata.pipeline.BrainDataPipeline.predict`

```python
predict(y, algorithm: str = 'ridge', **kwargs: str)
```

Execute pipeline with CV and return prediction results.

This is a terminal method that executes the full pipeline.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` |  | Target variable (labels or continuous values). | *required*
`algorithm` | <code>[str](#str)</code> | Prediction algorithm. Options: - 'ridge': Ridge regression (continuous targets) - 'lasso': Lasso regression (continuous targets) - 'svr': Support Vector Regression (continuous targets) - 'svm': Support Vector Classification (categorical targets) | <code>'ridge'</code>
`**kwargs` |  | Additional arguments passed to sklearn model constructor. For classification (svm), use class_weight='balanced' to handle imbalanced classes. See sklearn documentation for all options. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | BrainDataCVResult with scores, predictions, and fold information.

**Examples:**

Basic regression::

    result = brain.cv(5).predict(continuous_y, algorithm='ridge', alpha=1.0)

Classification with balanced classes::

    result = brain.cv(5).predict(labels, algorithm='svm', class_weight='balanced')

######## `nltools.data.braindata.pipeline.BrainDataPipeline.reduce`

```python
reduce(method: str = 'pca', n_components: int | None = None, **kwargs: int | None) -> BrainDataPipeline
```

Add dimensionality reduction step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Reduction method (e.g., 'pca'). Default: 'pca'. | <code>'pca'</code>
`n_components` | <code>[int](#int)</code> | Number of components to keep. | <code>None</code>
`**kwargs` |  | Additional arguments passed to ReduceStep. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainDataPipeline` | <code>[BrainDataPipeline](#nltools.data.braindata.pipeline.BrainDataPipeline)</code> | New pipeline with the reduction step appended.

#### `nltools.data.braindata.plotting`

BrainData plotting functions.

**Functions:**

Name | Description
---- | -----------
[`auto_select_colormap`](#nltools.data.braindata.plotting.auto_select_colormap) | Auto-select colormap based on data characteristics.
[`plot_brain`](#nltools.data.braindata.plotting.plot_brain) | Plot BrainData instance using nilearn visualization or matplotlib.
[`plot_flatmap_brain`](#nltools.data.braindata.plotting.plot_flatmap_brain) | Plot brain data on cortical flatmap.
[`plot_matplotlib`](#nltools.data.braindata.plotting.plot_matplotlib) | Plot using matplotlib (timeseries or histogram).
[`prepare_save_paths`](#nltools.data.braindata.plotting.prepare_save_paths) | Prepare save paths for multiple plot outputs.



##### Functions###### `nltools.data.braindata.plotting.auto_select_colormap`

```python
auto_select_colormap(data)
```

Auto-select colormap based on data characteristics.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | numpy array of brain data | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` |  | Colormap name

###### `nltools.data.braindata.plotting.plot_brain`

```python
plot_brain(bd, kind = 'glass', thr_upper = None, thr_lower = None, threshold = None, cut_coords = None, cmap = None, bg_img = None, ax = None, title = None, colorbar = True, save = None, stat = 'mean', **kwargs)
```

Plot BrainData instance using nilearn visualization or matplotlib.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`kind` | <code>[str](#str)</code> | Visualization type ('glass', 'slices', 'timeseries', 'histogram'). | <code>'glass'</code>
`thr_upper` | <code>[str](#str) / [float](#float)</code> | Upper threshold. | <code>None</code>
`thr_lower` | <code>[str](#str) / [float](#float)</code> | Lower threshold. | <code>None</code>
`threshold` | <code>[float](#float)</code> | Convenience parameter. If positive, sets thr_upper (shows values above threshold). If negative, sets thr_lower (shows values below threshold). | <code>None</code>
`cut_coords` | <code>[list](#list)</code> | Cut coordinates for multi-slice views. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Colormap name. | <code>None</code>
`bg_img` | <code>[Nifti1Image](#Nifti1Image) or [str](#str)</code> | Background image for slice views. | <code>None</code>
`ax` | <code>[Axes](#matplotlib.axes.Axes)</code> | Matplotlib axis to plot on. | <code>None</code>
`title` | <code>[str](#str)</code> | Plot title. | <code>None</code>
`colorbar` | <code>[bool](#bool)</code> | Whether to show colorbar. Default: True. | <code>True</code>
`save` | <code>[str](#str)</code> | Path to save figure(s). | <code>None</code>
`stat` | <code>[str](#str)</code> | Statistic for timeseries plots. Valid options: 'mean', 'median', 'std'. | <code>'mean'</code>
`**kwargs` |  | Additional arguments passed to nilearn plot functions. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | Display or matplotlib Figure.

###### `nltools.data.braindata.plotting.plot_flatmap_brain`

```python
plot_flatmap_brain(bd, threshold = None, cmap = 'RdBu_r', vmax = None, vmin = None, template = 'fsaverage5', with_curvature = True, curvature_contrast = 0.5, curvature_brightness = 0.5, colorbar = True, colorbar_orientation = 'horizontal', figsize = (12, 6), title = None, radius = 3.0, interpolation = 'linear', axes = None, save = None)
```

Plot brain data on cortical flatmap.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`threshold` | <code>[float](#float)</code> | Values below this absolute threshold are masked. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Matplotlib colormap for data. Default: 'RdBu_r'. | <code>'RdBu_r'</code>
`vmax` | <code>[float](#float)</code> | Maximum value for colormap. | <code>None</code>
`vmin` | <code>[float](#float)</code> | Minimum value for colormap. | <code>None</code>
`template` | <code>[str](#str)</code> | fsaverage resolution. Default: 'fsaverage5'. | <code>'fsaverage5'</code>
`with_curvature` | <code>[bool](#bool)</code> | Show sulcal/gyral pattern. Default: True. | <code>True</code>
`curvature_contrast` | <code>[float](#float)</code> | Contrast of curvature. Default: 0.5. | <code>0.5</code>
`curvature_brightness` | <code>[float](#float)</code> | Mean brightness of curvature. Default: 0.5. | <code>0.5</code>
`colorbar` | <code>[bool](#bool)</code> | Show colorbar. Default: True. | <code>True</code>
`colorbar_orientation` | <code>[str](#str)</code> | 'horizontal' or 'vertical'. Default: 'horizontal'. | <code>'horizontal'</code>
`figsize` | <code>[tuple](#tuple)</code> | Figure size. Default: (12, 6). | <code>(12, 6)</code>
`title` | <code>[str](#str)</code> | Figure title. | <code>None</code>
`radius` | <code>[float](#float)</code> | Sampling radius in mm for vol_to_surf. Default: 3.0. | <code>3.0</code>
`interpolation` | <code>[str](#str)</code> | Interpolation for vol_to_surf. Default: 'linear'. | <code>'linear'</code>
`axes` | <code>[Axes](#matplotlib.axes.Axes)</code> | Existing axes to plot on. | <code>None</code>
`save` | <code>[str](#str)</code> | File path to save figure. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

###### `nltools.data.braindata.plotting.plot_matplotlib`

```python
plot_matplotlib(bd, kind, stat = 'mean', ax = None, title = None, save = None)
```

Plot using matplotlib (timeseries or histogram).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`kind` | <code>[str](#str)</code> | 'timeseries' or 'histogram' | *required*
`stat` | <code>[str](#str)</code> | Statistic for timeseries ('mean', 'median', 'std') | <code>'mean'</code>
`ax` |  | Matplotlib axis. | <code>None</code>
`title` | <code>[str](#str)</code> | Plot title. | <code>None</code>
`save` | <code>[str](#str)</code> | Path to save figure. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

###### `nltools.data.braindata.plotting.prepare_save_paths`

```python
prepare_save_paths(save)
```

Prepare save paths for multiple plot outputs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`save` |  | Base save path (str or Path) | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | Dictionary with 'glass' and 'slices' keys containing save paths

#### `nltools.data.braindata.prediction`

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



##### Functions###### `nltools.data.braindata.prediction.mvpa_roi`

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

###### `nltools.data.braindata.prediction.mvpa_searchlight`

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

###### `nltools.data.braindata.prediction.mvpa_whole_brain`

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

###### `nltools.data.braindata.prediction.mvpa_whole_brain_pipeline`

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

###### `nltools.data.braindata.prediction.predict`

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

###### `nltools.data.braindata.prediction.predict_mvpa`

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

###### `nltools.data.braindata.prediction.predict_timeseries`

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

###### `nltools.data.braindata.prediction.resolve_estimator`

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

#### `nltools.data.braindata.utils`

Shared helpers for BrainData submodules.

These are internal utilities used by the facade and submodules — not part of the
public API.

**Functions:**

Name | Description
---- | -----------
[`apply_func`](#nltools.data.braindata.utils.apply_func) | Apply a statistical function to BrainData's ``.data`` attribute.
[`check_brain_data`](#nltools.data.braindata.utils.check_brain_data) | Check if data is a BrainData Instance, coercing Nifti1Image if needed.
[`check_brain_data_is_single`](#nltools.data.braindata.utils.check_brain_data_is_single) | Logical test if BrainData instance is a single image.
[`perform_arithmetic`](#nltools.data.braindata.utils.perform_arithmetic) | Perform an arithmetic operation with validation.
[`shallow_copy`](#nltools.data.braindata.utils.shallow_copy) | Create a shallow copy of a BrainData for efficient method chaining.



##### Functions###### `nltools.data.braindata.utils.apply_func`

```python
apply_func(bd, stat_func, axis = 0)
```

Apply a statistical function to BrainData's ``.data`` attribute.

If *axis* is 0, returns a BrainData with the statistic computed across
samples (e.g. within a voxel over time).  If *axis* is 1, returns a numpy
array with the statistic computed across features (e.g. across voxels
within a single time-point).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`stat_func` |  | Callable accepting an array and an ``axis`` kwarg. | *required*
`axis` |  | 0 = across images, 1 = within images. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
 | float | np.ndarray | BrainData

###### `nltools.data.braindata.utils.check_brain_data`

```python
check_brain_data(data, mask = None)
```

Check if data is a BrainData Instance, coercing Nifti1Image if needed.

###### `nltools.data.braindata.utils.check_brain_data_is_single`

```python
check_brain_data_is_single(data)
```

Logical test if BrainData instance is a single image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | brain data | *required*

**Returns:**

Type | Description
---- | -----------
 | (bool)

###### `nltools.data.braindata.utils.perform_arithmetic`

```python
perform_arithmetic(bd, other, operation, operation_name, inplace = False, reverse = False)
```

Perform an arithmetic operation with validation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance (left operand unless *reverse* is True). | *required*
`other` |  | The other operand (scalar, BrainData, or array). | *required*
`operation` |  | Numpy ufunc (e.g. ``np.add``, ``np.subtract``). | *required*
`operation_name` |  | Human-readable name for error messages. | *required*
`inplace` |  | If True, mutate *bd* in place. | <code>False</code>
`reverse` |  | If True, reverse operand order (for ``__rsub__`` etc.). | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Result of the operation.

###### `nltools.data.braindata.utils.shallow_copy`

```python
shallow_copy(bd)
```

Create a shallow copy of a BrainData for efficient method chaining.

Creates a new BrainData instance that shares immutable objects (mask,
nifti_masker) but copies mutable attributes.  The data array is NOT
copied — callers should handle data copying as needed.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance to copy. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | New instance with shared/copied attributes.

#### `nltools.data.braindata.validation`

Validation utilities for BrainData class.

This module contains helper functions for validating inputs, shapes, and
compatibility between BrainData objects and other data types.

**Functions:**

Name | Description
---- | -----------
[`validate_append_shapes`](#nltools.data.braindata.validation.validate_append_shapes) | Validate shape compatibility for appending BrainData objects.
[`validate_arithmetic_operand`](#nltools.data.braindata.validation.validate_arithmetic_operand) | Validate operand type for arithmetic operations.
[`validate_brain_data_shapes`](#nltools.data.braindata.validation.validate_brain_data_shapes) | Validate shape compatibility between two BrainData objects.
[`validate_data_type`](#nltools.data.braindata.validation.validate_data_type) | Validate input data type for BrainData initialization.
[`validate_frame`](#nltools.data.braindata.validation.validate_frame) | Validate and process X or Y dataframes for BrainData.
[`validate_index_operations`](#nltools.data.braindata.validation.validate_index_operations) | Validate indexing operations for BrainData.
[`validate_list_data`](#nltools.data.braindata.validation.validate_list_data) | Validate that all items in a list are the same type.



##### Functions###### `nltools.data.braindata.validation.validate_append_shapes`

```python
validate_append_shapes(data1_shape, data2_shape)
```

Validate shape compatibility for appending BrainData objects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1_shape` |  | Shape of first BrainData. | *required*
`data2_shape` |  | Shape of second BrainData to append. | *required*

###### `nltools.data.braindata.validation.validate_arithmetic_operand`

```python
validate_arithmetic_operand(other, operation_name)
```

Validate operand type for arithmetic operations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` |  | The operand to validate. | *required*
`operation_name` |  | Name of operation (e.g., 'add', 'multiply'). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` |  | Type of operand ('scalar', 'brain_data', or 'array').

###### `nltools.data.braindata.validation.validate_brain_data_shapes`

```python
validate_brain_data_shapes(brain1, brain2, operation = 'operation')
```

Validate shape compatibility between two BrainData objects.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain1` |  | First BrainData object. | *required*
`brain2` |  | Second BrainData object. | *required*
`operation` |  | Name of operation for error messages. | <code>'operation'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` |  | (brain1_is_single, brain2_is_single) booleans.

###### `nltools.data.braindata.validation.validate_data_type`

```python
validate_data_type(data)
```

Validate input data type for BrainData initialization.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | Input data to validate. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` |  | Type of data ('brain_data', 'list', 'h5', 'url', 'file', 'nibabel', 'none').

###### `nltools.data.braindata.validation.validate_frame`

```python
validate_frame(frame, data_shape = None, frame_type = 'DataFrame')
```

Validate and process X or Y dataframes for BrainData.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`frame` |  | Input to validate. Can be str, Path, pd.DataFrame, or None. | *required*
`data_shape` |  | Optional tuple of data shape to validate against. | <code>None</code>
`frame_type` |  | Type of frame for error messages (e.g., "X", "Y"). | <code>'DataFrame'</code>

**Returns:**

Type | Description
---- | -----------
 | pd.DataFrame: Validated and processed dataframe.

###### `nltools.data.braindata.validation.validate_index_operations`

```python
validate_index_operations(data_shape, index)
```

Validate indexing operations for BrainData.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data_shape` |  | Shape of the data array. | *required*
`index` |  | Index to validate. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` |  | Type of indexing ('single', 'slice', 'array').

###### `nltools.data.braindata.validation.validate_list_data`

```python
validate_list_data(data_list)
```

Validate that all items in a list are the same type.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data_list` |  | List to validate. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`str` |  | Type of items ('brain_data' or 'file').

