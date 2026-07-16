(data-brain-data-braindata)=
## `BrainData`

```python
BrainData(data = None, *, Y = None, X = None, mask = None, masker = None, h5_compression = 'gzip', verbose = False, resample = True, interpolation = 'auto')
```

Represent neuroimaging data as vectors instead of three-dimensional matrices.

This representation makes it easier to perform data manipulation and analyses.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | Neuroimaging data. Can be: - None (empty BrainData) - BrainData object - List of BrainData objects or file paths - File path (str/Path) to .nii/.nii.gz/.h5/.hdf5 - nibabel Nifti1Image object - URL to download data from - numpy array (1D ``(n_voxels,)`` for a single image or 2D   ``(n_images, n_voxels)`` for a stack). The ``mask`` argument   is required and must define the same number of in-mask voxels. | <code>None</code>
`mask` |  | Brain mask. Can be None (uses MNI template), a nibabel Nifti1Image, a file path (str/Path) to a mask file, or a template name string like ``'2mm-MNI152-2009c'`` (version: 'fsl' for default/, 'a' for nilearn/, 'c' for fmriprep/). | <code>None</code>
`masker` |  | nilearn masker object (e.g. ROI or searchlight extractor). Default will load data as voxels. | <code>None</code>
`resample` | <code>bool, default=True</code> | Whether to automatically resample data to mask space. If True, data is resampled to match mask spatial characteristics. If False, data must already be in mask space. Default True preserves backward compatibility with v0.5.1. | <code>True</code>
`interpolation` | <code>str, default='auto'</code> | Interpolation method for resampling. Options: 'auto' (detect based on data type; uses 'nearest' for discrete data like atlases/masks and 'continuous' for stat maps), 'nearest' (nearest-neighbor, preserves discrete values), 'linear' (linear interpolation), 'continuous' (higher-order spline, use for stat maps). | <code>'auto'</code>

**Methods:**

Name | Description
---- | -----------
[`align`](#data-brain-data-align) | Align BrainData instance to target object using functional alignment.
[`append`](#data-brain-data-append) | Append data to BrainData instance.
[`apply_mask`](#data-brain-data-apply-mask) | Mask BrainData instance using nilearn functionality.
[`astype`](#data-brain-data-astype) | Cast BrainData.data as type.
[`bootstrap`](#data-brain-data-bootstrap) | Bootstrap statistics using efficient online algorithms.
[`cluster_report`](#data-brain-data-cluster-report) | Generate a cluster report with anatomical labels.
[`compute_contrasts`](#data-brain-data-compute-contrasts) | Compute contrasts from fitted GLM results.
[`copy`](#data-brain-data-copy) | Create a deep copy of a BrainData instance.
[`create_empty`](#data-brain-data-create-empty) | Create a copy of BrainData with empty data array.
[`decompose`](#data-brain-data-decompose) | Decompose BrainData object.
[`detrend`](#data-brain-data-detrend) | Remove linear trend from each voxel.
[`distance`](#data-brain-data-distance) | Calculate distance between images within a BrainData() instance.
[`extract_roi`](#data-brain-data-extract-roi) | Extract activity from mask or ROI atlas using NiftiLabelsMasker.
[`filter`](#data-brain-data-filter) | Apply butterworth filter to data. Wraps nilearn.signal.clean.
[`find_spikes`](#data-brain-data-find-spikes) | Identify spikes from Time Series Data.
[`fit`](#data-brain-data-fit) | Fit a model to brain imaging data.
[`icc`](#data-brain-data-icc) | Calculate voxel-wise intraclass correlation coefficient.
[`iplot`](#data-brain-data-iplot) | Interactive WebGL brain viewer powered by niivue (`ipyniivue`).
[`mean`](#data-brain-data-mean) | Get mean of each voxel or image.
[`median`](#data-brain-data-median) | Get median of each voxel or image.
[`multivariate_similarity`](#data-brain-data-multivariate-similarity) | Predict a BrainData spatial distribution from a linear combination.
[`plot`](#data-brain-data-plot) | Plot BrainData instance using nilearn visualization or matplotlib.
[`plot_flatmap`](#data-brain-data-plot-flatmap) | Plot brain data on cortical flatmap.
[`plot_surf`](#data-brain-data-plot-surf) | Render this BrainData on fsaverage surfaces as a tight 2×2 montage.
[`predict`](#data-brain-data-predict) | Predict voxel timeseries (encoding) or decode labels (MVPA).
[`r_to_z`](#data-brain-data-r-to-z) | Apply Fisher's r-to-z transformation to each data element.
[`regions`](#data-brain-data-regions) | Extract brain connected regions into separate regions.
[`resample_to`](#data-brain-data-resample-to) | Resample BrainData to match target image or resolution.
[`scale`](#data-brain-data-scale) | Scale data via mean scaling.
[`similarity`](#data-brain-data-similarity) | Calculate similarity to a single BrainData or nibabel image.
[`smooth`](#data-brain-data-smooth) | Apply spatial smoothing using nilearn smooth_img().
[`standardize`](#data-brain-data-standardize) | Standardize BrainData() instance.
[`std`](#data-brain-data-std) | Get standard deviation of each voxel or image.
[`sum`](#data-brain-data-sum) | Get sum of each voxel or image.
[`temporal_resample`](#data-brain-data-temporal-resample) | Resample BrainData timeseries to a new target frequency or number of samples.
[`threshold`](#data-brain-data-threshold) | Threshold BrainData instance with optional cluster filtering.
[`to_nifti`](#data-brain-data-to-nifti) | Convert BrainData Instance into Nifti Object.
[`transform_pairwise`](#data-brain-data-transform-pairwise) | Transform data into pairwise comparisons.
[`ttest`](#data-brain-data-ttest) | One-sample voxelwise t-test across images (axis 0).
[`ttest2`](#data-brain-data-ttest2) | Two-sample voxelwise t-test between two BrainData stacks.
[`upload_neurovault`](#data-brain-data-upload-neurovault) | Upload BrainData images and metadata to NeuroVault.
[`write`](#data-brain-data-write) | Write out BrainData object to Nifti or HDF5 File.
[`z_to_r`](#data-brain-data-z-to-r) | Convert z score back into r value for each element of data object.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`X` |  | Design matrix / per-image covariates as a polars DataFrame.
`Y` |  | Per-image targets as a polars DataFrame.
`data` |  | 
`design_matrix` |  | 
`dtype` |  | Get data type of BrainData.data.
`is_empty` | <code>[bool](#bool)</code> | Check if BrainData.data is empty.
`masker` |  | 
`shape` |  | Get images by voxels shape.
`size` |  | Total number of elements in BrainData.data (numpy convention).
`verbose` |  | 

### Methods

(data-brain-data-align)=
#### `align`

```python
align(target, method = 'procrustes', axis = 0, *, spatial_scale: str = 'whole_brain', roi_mask: str = None, radius_mm: float = 10.0)
```

Align BrainData instance to target object using functional alignment.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` |  | (BrainData) object to align to. | *required*
`method` |  | (str) alignment method to use ['probabilistic_srm','deterministic_srm','procrustes'] | <code>'procrustes'</code>
`axis` |  | (int) axis to align on | <code>0</code>
`spatial_scale` | <code>[str](#str)</code> | ``'whole_brain'`` (default), ``'roi'``, or ``'searchlight'``. ``'roi'`` / ``'searchlight'`` are not yet implemented (per-parcel transforms + reassembly is a follow-up slice). | <code>'whole_brain'</code>
`roi_mask` |  | Reserved for ``spatial_scale='roi'``. | <code>None</code>
`radius_mm` | <code>[float](#float)</code> | Reserved for ``spatial_scale='searchlight'``. | <code>10.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (dict) a dictionary containing transformed object, transformation matrix, and the shared response matrix

**Examples:**

```pycon
>>> out = data.align(target, method='procrustes')
>>> out = data.align(target, method='probabilistic_srm')
```

(data-brain-data-append)=
#### `append`

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

(data-brain-data-apply-mask)=
#### `apply_mask`

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

(data-brain-data-astype)=
#### `astype`

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

(data-brain-data-bootstrap)=
#### `bootstrap`

```python
bootstrap(stat, n_samples = 5000, save_boots = False, percentiles = (2.5, 97.5), X_test = None, backend = None, max_gpu_memory_gb = 4.0, n_jobs = -1, random_state = None)
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
`percentiles` |  | (tuple) Percentiles for confidence intervals. Default: (2.5, 97.5) | <code>(2.5, 97.5)</code>
`X_test` |  | (np.ndarray, optional) Test features for 'predict' bootstrap. | <code>None</code>
`backend` |  | (str, optional) Backend for Ridge bootstrap: None (CPU), 'torch' (GPU if available), or 'auto' (auto-select). Ignored for simple stats. | <code>None</code>
`max_gpu_memory_gb` |  | (float) Maximum GPU memory to use when backend is 'torch' or 'auto'. Default: 4.0 | <code>4.0</code>
`n_jobs` |  | (int) Number of CPU cores for parallelization. -1 means all CPUs. | <code>-1</code>
`random_state` |  | (int, optional) Random seed for reproducibility | <code>None</code>

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

(data-brain-data-cluster-report)=
#### `cluster_report`

```python
cluster_report(*, stat_threshold: float | None = 3.0, cluster_threshold: int = 10, two_sided: bool = True, min_distance: float = 8.0, atlas: str | Sequence[str] | None = None, prob_threshold: float = 5.0) -> ClusterReport
```

Generate a cluster report with anatomical labels.

Identifies surviving clusters in the stat map (after voxel + extent
thresholding), reports peak coordinates and sub-peaks, and labels
each peak/cluster against one or more atlases.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stat_threshold` | <code>[float](#float) \| None</code> | Voxel-level threshold (e.g. z- or t-cutoff). ``None`` treats ``self`` as already thresholded. | <code>3.0</code>
`cluster_threshold` | <code>[int](#int)</code> | Minimum cluster size in voxels. | <code>10</code>
`two_sided` | <code>[bool](#bool)</code> | Report negative clusters separately. | <code>True</code>
`min_distance` | <code>[float](#float)</code> | Minimum mm between sub-peaks within a cluster. | <code>8.0</code>
`atlas` | <code>[str](#str) \| [Sequence](#collections.abc.Sequence)[[str](#str)] \| None</code> | Atlas name or list of names (see `list_atlases`). Defaults to ``("harvard_oxford", "aal", "schaefer_200")``. | <code>None</code>
`prob_threshold` | <code>[float](#float)</code> | Drop probabilistic-atlas regions below this %. | <code>5.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[ClusterReport](#nltools.data.atlases.ClusterReport)</code> | `ClusterReport` with ``peaks``,
<code>[ClusterReport](#nltools.data.atlases.ClusterReport)</code> | ``clusters`` (polars DataFrames), and ``stat_img`` (BrainData).

(data-brain-data-compute-contrasts)=
#### `compute_contrasts`

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
`contrast_type` | <code>[str](#str)</code> | What to return per contrast. One of `"t"` (default, t-statistic map), `"z"` (z-score), `"p"` (p-value), `"beta"` / `"effect_size"` (effect-size β map — use this when feeding a second-level group analysis), or `"all"` (a bundle dict `{"beta", "t", "z", "p", "se"}` of maps for one contrast). Default: `"t"`. | <code>'t'</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or dict: A single contrast with a scalar `contrast_type` returns a `BrainData` map; with `contrast_type="all"` it returns a flat dict keyed by `"beta"`/`"t"`/`"z"`/`"p"`/`"se"`. A dict of contrasts returns a dict keyed by contrast name (nested under the five keys when `contrast_type="all"`).

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

(data-brain-data-copy)=
#### `copy`

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

(data-brain-data-create-empty)=
#### `create_empty`

```python
create_empty()
```

Create a copy of BrainData with empty data array.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | A copy of this object with an empty data array.

(data-brain-data-decompose)=
#### `decompose`

```python
decompose(*, method = 'pca', axis = 'voxels', n_components = None, **kwargs)
```

Decompose BrainData object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` |  | (str) Algorithm to perform decomposition         types=['pca','ica','nnmf','fa','dictionary','kernelpca'] | <code>'pca'</code>
`axis` |  | dimension to decompose ['voxels','images'] | <code>'voxels'</code>
`n_components` |  | (int) number of components. If None then retain         as many as possible. | <code>None</code>
`**kwargs` |  | forwarded to the underlying sklearn decomposition estimator. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`output` |  | a dictionary of decomposition parameters

(data-brain-data-detrend)=
#### `detrend`

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

(data-brain-data-distance)=
#### `distance`

```python
distance(metric = 'euclidean', *, spatial_scale: str = 'whole_brain', roi_mask: str = None, radius_mm: float = 10.0, **kwargs: float)
```

Calculate distance between images within a BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` |  | (str) type of distance metric (can use any scipy.spatial.distance     metric supported by cdist) | <code>'euclidean'</code>
`spatial_scale` | <code>[str](#str)</code> | One of ``'whole_brain'`` (default), ``'roi'``, or ``'searchlight'``. ``'whole_brain'`` returns a single pairwise distance ``Adjacency`` between images. ``'roi'`` requires ``roi_mask`` and returns a stacked ``Adjacency`` with one RDM per parcel and ``spatial_scale`` provenance attached for back-projection via ``Adjacency.to_brain()``. ``'searchlight'`` requires ``radius_mm`` (and is not yet implemented in this slice). | <code>'whole_brain'</code>
`roi_mask` |  | Atlas image (BrainData / Nifti1Image / path) for ``spatial_scale='roi'``. | <code>None</code>
`radius_mm` | <code>[float](#float)</code> | Searchlight radius in mm. Default 10.0. | <code>10.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Adjacency` |  | Single pairwise distance matrix for ``'whole_brain'``; stacked Adjacency (one matrix per parcel/searchlight) with ``spatial_scale`` set for ``'roi'`` / ``'searchlight'``.

(data-brain-data-extract-roi)=
#### `extract_roi`

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

(data-brain-data-filter)=
#### `filter`

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

(data-brain-data-find-spikes)=
#### `find_spikes`

```python
find_spikes(global_spike_cutoff = 3, diff_spike_cutoff = 3, *, TR: float | None = None, sampling_freq: float | None = None)
```

Identify spikes from Time Series Data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`global_spike_cutoff` | <code>[int](#int) or None</code> | cutoff to identify spikes in global signal in standard deviations, or None to skip. | <code>3</code>
`diff_spike_cutoff` | <code>[int](#int) or None</code> | cutoff to identify spikes in average frame difference in standard deviations, or None to skip. | <code>3</code>
`TR` | <code>[float](#float) \| None</code> | Repetition time in seconds. Sets the returned DesignMatrix's sampling_freq for downstream `.append(...)` / `.convolve()`. Pass exactly one of `TR` or `sampling_freq`. | <code>None</code>
`sampling_freq` | <code>[float](#float) \| None</code> | Sampling frequency in Hz (= 1/TR). See `TR`. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | DesignMatrix with one indicator column per detected spike, with
 | all spike columns pre-marked as confounds.

(data-brain-data-fit)=
#### `fit`

```python
fit(model = 'glm', *, X = None, cv = None, local_alpha = True, fit_intercept = False, inplace = True, scale = True, scale_value = 100.0, progress_bar = None, design_clean = True, design_clean_thresh = 0.95, design_clean_exclude_confounds = False, design_clean_fill_na = 0, **kwargs)
```

Fit a model to brain imaging data.

Creates and fits a model from string specification. The brain data
(self.data) is always used as the target variable. Model and results
are stored for later use with predict().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model` | <code>[str](#str)</code> | Model type: 'ridge', 'glm', or future model names | <code>'glm'</code>
`X` | <code>[array](#array) - [like](#like) or [DataFrame](#DataFrame)</code> | Design matrix or feature matrix | <code>None</code>
`cv` | <code>int or sklearn CV splitter</code> | Cross-validation specification (Ridge only). int → ``KFold(cv)``; pass a splitter object (e.g. ``KFold(5, shuffle=True)``, ``GroupKFold(8)``) for non-contiguous folds. Generators (``splitter.split(X)``) are rejected. | <code>None</code>
`local_alpha` | <code>bool, default=True</code> | Ridge only. If True, select α independently per voxel via ``solve_ridge_cv``. If False, pick a single α shared across all voxels. | <code>True</code>
`fit_intercept` | <code>bool, default=False</code> | Ridge only. Forwarded to the Ridge model — center X and y on the training fold mean per fold and recover the intercept after. | <code>False</code>
`inplace` | <code>bool, default=True</code> | If True, mutate self and return self. If False, return Fit dataclass with results (self unchanged). | <code>True</code>
`scale` | <code>bool, default=True</code> | Apply grand-mean scaling before fitting. | <code>True</code>
`scale_value` | <code>float, default=100.0</code> | Target value for mean after scaling. | <code>100.0</code>
`progress_bar` | <code>[bool](#bool)</code> | Display progress bar during fitting. | <code>None</code>
`design_clean` | <code>bool, default=True</code> | GLM only. Run ``DesignMatrix.clean()`` on ``X`` before fitting to drop highly correlated regressors. Coerces ``X`` to ``DesignMatrix`` if needed. Ignored when ``model='ridge'``. | <code>True</code>
`design_clean_thresh` | <code>float, default=0.95</code> | GLM only. Correlation threshold passed to ``DesignMatrix.clean()`` (drops if ``abs(r) >= thresh``). Ignored when ``model='ridge'``. | <code>0.95</code>
`design_clean_exclude_confounds` | <code>bool, default=False</code> | GLM only. If True, ``DesignMatrix.clean()`` skips confound columns when checking correlations. Ignored when ``model='ridge'``. | <code>False</code>
`design_clean_fill_na` | <code>int, float, or None, default=0</code> | GLM only. Fill value for NaNs before correlation check in ``DesignMatrix.clean()``. Ignored when ``model='ridge'``. | <code>0</code>
`**kwargs` | <code>[dict](#dict)</code> | Additional arguments passed to model constructor | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | BrainData or Fit: If ``inplace=True``, returns self (fitted BrainData). If ``inplace=False``, returns Fit dataclass with results.

<details class="notes" open markdown="1">
<summary>Notes</summary>

After ``model="glm"``, the following per-regressor BrainData
attributes are populated — one map per design-matrix column:

- ``glm_betas``: effect-size (β) maps.
- ``glm_t``: marginal t-statistic for each regressor.
- ``glm_p``: marginal p-value.
- ``glm_se``: standard error of β.
- ``glm_r2``: voxel-wise R².

``glm_t[i]`` is a valid t-map for the trivial one-hot contrast on
regressor ``i`` only. For contrasts across regressors
(``"A - B"``, ``[1, -1, 0, ...]``) use `compute_contrasts` —
you cannot correctly combine these per-regressor maps by hand
because t-statistic arithmetic requires the off-diagonal elements
of the parameter covariance matrix, which are not stored. Pass
``contrast_type="all"`` to get ``β``/``t``/``z``/``p``/``se`` for
one contrast in a single call.

</details>

**Examples:**

```pycon
>>> brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
>>> fit = brain_data.fit(model='ridge', alpha=1.0, X=features, inplace=False)
```

(data-brain-data-icc)=
#### `icc`

```python
icc(n_subjects, n_sessions, method = 'icc2', parallel = None, n_jobs = -1, max_gpu_memory_gb = 4.0)
```

Calculate voxel-wise intraclass correlation coefficient.

ICC Formulas based on Shrout & Fleiss (1979).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_subjects` |  | Number of subjects in the data | *required*
`n_sessions` |  | Number of sessions per subject | *required*
`method` |  | Type of ICC ('icc1', 'icc2', 'icc3'). Default: 'icc2' | <code>'icc2'</code>
`parallel` |  | Parallelization method (None, 'cpu', 'gpu') | <code>None</code>
`n_jobs` |  | Number of CPU cores (-1 = all cores) | <code>-1</code>
`max_gpu_memory_gb` |  | GPU memory budget in GB | <code>4.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with ICC map (shape: (1, n_voxels))

**Examples:**

```pycon
>>> icc_map = data.icc(n_subjects=20, n_sessions=3, method='icc2')
```

(data-brain-data-iplot)=
#### `iplot`

```python
iplot(*, view: str = 'ortho', threshold: float | None = None, lower: float | None = None, upper: float | None = None, cmap: str = 'warm', bg_img: str | bool | None = None, atlas: str | Atlas | None = None, opacity: float = 1.0, outline: float = 0.0, colorbar: bool = True, controls: bool = True, **kwargs: bool)
```

Interactive WebGL brain viewer powered by niivue (`ipyniivue`).

Renders inline in a live kernel (Jupyter, marimo) with live windowing
(right-drag to set the threshold/contrast), slice scrolling, native 4D
frame scrubbing, true 3D rendering, a stat-map colorbar, and optional
nltools-atlas overlays. Static-built docs are not supported; use
`plot` there.

By default (``controls=True``) the return value is an
`ipywidgets.VBox` stacking a threshold slider above the viewer; access
the underlying `NiiVue` via its ``.viewer`` attribute and the slider
via ``.threshold_slider``. Pass ``controls=False`` to get the bare
`NiiVue` widget instead.

Thresholding is a divergent magnitude window: ``cal_min`` is the
display floor (sub-floor voxels render transparent), ``cal_max`` the
saturation point, with the positive limb using ``cmap`` and the
negative limb its mirrored partner. Precedence: ``lower``/``upper``
win; otherwise ``threshold`` sets the floor (ceiling auto);
otherwise the window is fully auto.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`view` | <code>[str](#str)</code> | ``"ortho"`` (default), ``"axial"``, ``"coronal"``, ``"sagittal"``, or ``"render"`` (3D volume render). ``"surface"`` is no longer supported — use ``"render"`` or `plot_flatmap` / `plot_surf`. | <code>'ortho'</code>
`threshold` | <code>[float](#float) \| None</code> | Convenience symmetric magnitude floor (→ ``cal_min``). | <code>None</code>
`lower` | <code>[float](#float) \| None</code> | Window floor (→ ``cal_min``). Overrides ``threshold``. | <code>None</code>
`upper` | <code>[float](#float) \| None</code> | Window ceiling (→ ``cal_max``). Overrides ``threshold``. | <code>None</code>
`cmap` | <code>[str](#str)</code> | niivue colormap for the positive limb (default ``"warm"``). Common matplotlib names are auto-mapped with a warning. | <code>'warm'</code>
`bg_img` | <code>[str](#str) \| [bool](#bool) \| None</code> | ``None``/``True`` auto-loads the matching MNI template when the data is in standard space (else none); ``False`` disables the background; a path string uses that image. | <code>None</code>
`atlas` | <code>[str](#str) \| [Atlas](#nltools.data.atlases.Atlas) \| None</code> | Atlas overlay — a registry name (e.g. ``"aal"``), a loaded `Atlas`, or ``None``. Deterministic atlases only; probabilistic atlases raise. | <code>None</code>
`opacity` | <code>[float](#float)</code> | Stat-map (and filled-atlas) opacity in ``0..1``. | <code>1.0</code>
`outline` | <code>[float](#float)</code> | ``> 0`` draws atlas region boundaries of that width (stat map stays visible); ``0`` draws filled regions. | <code>0.0</code>
`colorbar` | <code>[bool](#bool)</code> | Show the stat-map colorbar (default ``True``). An explicit ``is_colorbar`` kwarg overrides this. | <code>True</code>
`controls` | <code>[bool](#bool)</code> | Wrap the viewer in a `VBox` with an interactive threshold slider (default ``True``). ``False`` returns the bare `NiiVue`. Requires the ``ipywidgets`` optional dependency when ``True``. | <code>True</code>
`**kwargs` |  | Forwarded verbatim to ``ipyniivue.NiiVue(**kwargs)`` (e.g. ``height``, ConfigOptions like ``is_colorbar``). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | ipywidgets.VBox with ``.viewer`` (the `NiiVue`) and
 | ``.threshold_slider`` when ``controls=True``; otherwise the bare
 | ``ipyniivue.NiiVue`` widget.

(data-brain-data-mean)=
#### `mean`

```python
mean(axis = 0, *, spatial_scale: str = 'whole_brain', roi_mask: str = None)
```

Get mean of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array). Ignored when ``spatial_scale='roi'``. | <code>0</code>
`spatial_scale` | <code>[str](#str)</code> | ``'whole_brain'`` (default) preserves existing behavior. ``'roi'`` requires ``roi_mask`` and returns a BrainData of the same shape with each voxel painted with its parcel's mean per image (parcellation smoothing). | <code>'whole_brain'</code>
`roi_mask` |  | Atlas image for ``spatial_scale='roi'``. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Mean values.

(data-brain-data-median)=
#### `median`

```python
median(axis = 0, *, spatial_scale: str = 'whole_brain', roi_mask: str = None)
```

Get median of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array). Ignored when ``spatial_scale='roi'``. | <code>0</code>
`spatial_scale` | <code>[str](#str)</code> | ``'whole_brain'`` (default) or ``'roi'`` (paints each voxel with its parcel's median per image). | <code>'whole_brain'</code>
`roi_mask` |  | Atlas image for ``spatial_scale='roi'``. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Median values.

(data-brain-data-multivariate-similarity)=
#### `multivariate_similarity`

```python
multivariate_similarity(images, method = 'ols')
```

Predict a BrainData spatial distribution from a linear combination.

The predictors may be other BrainData instances or nibabel images.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`images` |  | BrainData instance of weight map | *required*
`method` | <code>[str](#str)</code> | Regression method. Default: 'ols'. | <code>'ols'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | dictionary of regression statistics in BrainData instances {'beta','t','p','df','residual'}

(data-brain-data-plot)=
#### `plot`

```python
plot(method = 'glass', upper = None, lower = None, threshold = None, view = 'z', cut_coords = None, cmap = None, bg_img = None, ax = None, figsize = (8, 6), title = None, colorbar = True, save = None, stat = 'mean', limit = 3, **kwargs)
```

Plot BrainData instance using nilearn visualization or matplotlib.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Visualization type: 'glass', 'slices', 'timeseries', 'histogram' | <code>'glass'</code>
`upper` | <code>[str](#str) / [float](#float)</code> | Upper threshold. | <code>None</code>
`lower` | <code>[str](#str) / [float](#float)</code> | Lower threshold. | <code>None</code>
`threshold` | <code>[float](#float)</code> | Convenience parameter for thresholding. | <code>None</code>
`view` | <code>[str](#str)</code> | For ``method="slices"``, any non-empty combination of ``"x"``, ``"y"``, ``"z"`` (e.g. ``"xyz"``, ``"xz"``, ``"y"``). Default: ``"z"``. | <code>'z'</code>
`cut_coords` | <code>[list](#list) or [dict](#dict)</code> | Cut coordinates for multi-slice views. Takes precedence over ``view``-based defaults. Either a list matching ``len(view)`` or a dict keyed by axis letter. | <code>None</code>
`cmap` | <code>[str](#str)</code> | Colormap name. | <code>None</code>
`bg_img` | <code>str/nibabel image</code> | Background image. | <code>None</code>
`ax` | <code>[Axes](#matplotlib.axes.Axes)</code> | Matplotlib axis. | <code>None</code>
`figsize` | <code>[tuple](#tuple)</code> | default figure size if no axis (8, 6) | <code>(8, 6)</code>
`title` | <code>[str](#str)</code> | Plot title. | <code>None</code>
`colorbar` | <code>[bool](#bool)</code> | Whether to show colorbar. Default: True. | <code>True</code>
`save` | <code>[str](#str)</code> | Path to save figure(s). | <code>None</code>
`stat` | <code>[str](#str)</code> | Statistic for timeseries plots. Default: 'mean'. | <code>'mean'</code>
`limit` | <code>[int](#int)</code> | Maximum number of images to render when this BrainData contains multiple maps and ``method`` is ``"glass"`` or ``"slices"``. Default: 3. Warns when more images exist than ``limit``. | <code>3</code>
`**kwargs` |  | Additional arguments passed to nilearn plot functions. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure or list[matplotlib.figure.Figure]: A
 | single figure for single-image data; a list of figures for
 | multi-image data with ``method`` in ``{"glass", "slices"}``
 | (one per image for glass; one per image-and-view pair for
 | slices).

(data-brain-data-plot-flatmap)=
#### `plot_flatmap`

```python
plot_flatmap(threshold = None, cmap = 'RdBu_r', vmax = None, vmin = None, template = 'fsaverage5', with_curvature = True, curvature_contrast = 0.5, curvature_brightness = 0.5, transparency = 'auto', colorbar = True, colorbar_orientation = 'horizontal', figsize = (12, 6), title = None, radius_mm = 3.0, interpolation = 'linear', axes = None, save = None)
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
`transparency` | <code>BrainData, Nifti1Image, str, or "auto"</code> | Binary mask used to render vertices outside the mask as transparent. ``"auto"`` (default) uses the instance's ``.mask``; pass ``None`` to disable masking. | <code>'auto'</code>
`colorbar` | <code>[bool](#bool)</code> | Show colorbar. Default: True. | <code>True</code>
`colorbar_orientation` | <code>[str](#str)</code> | 'horizontal' or 'vertical'. Default: 'horizontal'. | <code>'horizontal'</code>
`figsize` | <code>[tuple](#tuple)</code> | Figure size as (width, height). Default: (12, 6). | <code>(12, 6)</code>
`title` | <code>[str](#str)</code> | Figure title. | <code>None</code>
`radius_mm` | <code>[float](#float)</code> | Sampling radius in mm. Default: 3.0. | <code>3.0</code>
`interpolation` | <code>[str](#str)</code> | Interpolation method. Default: 'linear'. | <code>'linear'</code>
`axes` | <code>[Axes](#matplotlib.axes.Axes)</code> | Existing axes to plot on. | <code>None</code>
`save` | <code>[str](#str)</code> | File path to save figure. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

(data-brain-data-plot-surf)=
#### `plot_surf`

```python
plot_surf(*, hemi = 'both', view = 'montage', surface = 'pial', template = 'fsaverage5', threshold = None, cmap = 'RdBu_r', vmin = None, vmax = None, transparency = 'auto', bg_on_data = False, colorbar = True, colorbar_orientation = 'horizontal', figsize = (10, 8), title = None, radius_mm = 3.0, interpolation = 'linear', zoom = 1.2, axes = None, save = None)
```

Render this BrainData on fsaverage surfaces as a tight 2×2 montage.

Facade over `plot_surf`. See that function's
docstring for the full argument reference. Notable defaults:
``surface="pial"``, ``zoom=1.2``, ``transparency="auto"`` (uses
this instance's ``.mask``).

**Returns:**

Type | Description
---- | -----------
 | matplotlib.figure.Figure

(data-brain-data-predict)=
#### `predict`

```python
predict(*, y: np.ndarray | None = None, X: np.ndarray | None = None, spatial_scale: str = 'whole_brain', model: str = 'svm', cv: int = 5, standardize: bool = True, reduce: str | None = None, n_components: int | None = None, scoring: str = 'auto', groups: np.ndarray | None = None, roi_mask: np.ndarray | None = None, radius_mm: float = 10.0, inplace: bool = False, n_jobs: int = 1, progress_bar: bool = False)
```

Predict voxel timeseries (encoding) or decode labels (MVPA).

Dispatched by which of ``X`` or ``y`` is provided:

1. **Timeseries prediction** (``X`` provided): use a fitted ridge /
   GLM encoding model on ``self`` to predict voxel responses.
   Returns a fresh ``BrainData`` whose ``.data`` holds the predicted
   timeseries (composes directly with ``.plot()``, ``.standardize()``
   etc.). ``inplace`` has no effect in this mode.
2. **MVPA decoding** (``y`` provided): train a classifier or
   regressor with cross-validation. Returns a `Predict`
   dataclass. Spatial fields (``weight_map``, ``fold_weight_maps``,
   ``final_weight_map``, ``accuracy_map``) are `BrainData`
   objects so ``result.weight_map.plot()`` works directly. Drop down
   to numpy via ``result.weight_map.data``.

Field shapes by ``spatial_scale=``:

- **whole_brain**: ``predictions`` (n_samples,) OOF predictions,
  ``scores`` (n_folds,), ``mean_score`` float, ``std_score`` float,
  ``weight_map`` BrainData (``coef_`` from one fit on the **full**
  ``(X, y)`` — the publishable map), ``fold_weight_maps`` BrainData
  (n_folds, n_voxels) for stability analysis, ``estimator`` the
  fitted all-data sklearn estimator (use for ``.predict()`` on new
  data).
- **roi**: ``scores`` (n_folds, n_rois), ``mean_score`` (n_rois,),
  ``std_score`` (n_rois,), ``roi_labels`` (n_rois,) atlas IDs in
  matching order, ``accuracy_map`` / ``weight_map`` /
  ``fold_weight_maps`` BrainData (per-parcel coefs reassembled to
  voxel space; voxels outside the atlas = NaN), ``estimator`` dict
  keyed by atlas label.
- **searchlight**: ``accuracy_map`` BrainData.

With ``inplace=True``, fields are attached to ``self`` with a
``predict_`` prefix (e.g. ``self.predict_weight_map``,
``self.predict_accuracy_map``), mirroring ``bd.fit()``'s
``glm_*`` / ``ridge_*`` naming.

Why ``weight_map`` is the all-data refit, not the CV mean:
the mean of K per-fold ``coef_`` vectors doesn't correspond to
any actual fitted estimator (each fold saw a different subset).
The all-data refit is a single legitimate model with all the
information used. CV gives the honest *score*; the refit gives
the publishable *map*. The CV-mean is one line away if you want
it: ``result.fold_weight_maps.data.mean(axis=0)``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` | <code>[array](#array) - [like](#like)</code> | Labels (classification) or continuous targets (regression), shape ``(n_samples,)``. Triggers MVPA mode. | <code>None</code>
`X` | <code>[array](#array) - [like](#like)</code> | Features for timeseries prediction, shape ``(n_samples, n_features)``. Triggers encoding mode. | <code>None</code>
`spatial_scale` | <code>[str](#str)</code> | MVPA dispatch — ``'whole_brain'``, ``'searchlight'``, or ``'roi'``. | <code>'whole_brain'</code>
`model` | <code>str or sklearn estimator</code> | Algorithm. String shortcuts:<br>- Classification: ``'svm'`` (LinearSVC), ``'logistic'``,   ``'lda'``, ``'ridge_classifier'``. - Regression: ``'ridge'``, ``'lasso'``, ``'svr'``.<br>Or pass any sklearn estimator / Pipeline (e.g., ``make_pipeline(StandardScaler(), SelectKBest(k=500), LinearSVC())``). When ``model`` is a sklearn ``Pipeline``, ``standardize`` is auto-defaulted to ``False`` (with a warning) so we don't wrap another StandardScaler around your pipeline. Pass ``standardize=True`` explicitly to override. | <code>'svm'</code>
`cv` | <code>int or sklearn CV splitter</code> | ``int`` → KFold (regression) or StratifiedKFold (classification); pass a splitter for custom schemes (e.g., ``GroupKFold``). | <code>5</code>
`standardize` | <code>[bool](#bool)</code> | Z-score features per fold before fitting. Default ``True``. Auto-flipped to ``False`` when ``model`` is a sklearn ``Pipeline`` (see ``model`` above). | <code>True</code>
`reduce` | <code>[str](#str)</code> | Per-fold dimensionality reduction. Currently only ``'pca'`` supported. Default ``None``. Weight maps are back-projected through PCA to voxel space. | <code>None</code>
`n_components` | <code>[int](#int)</code> | PCA components when ``reduce='pca'``. | <code>None</code>
`scoring` | <code>[str](#str)</code> | Sklearn scoring string. Default ``'auto'`` → ``'accuracy'`` if classifier, ``'r2'`` if regressor. | <code>'auto'</code>
`groups` | <code>[array](#array) - [like](#like)</code> | Group labels for CV splitters that need them (e.g., leave-one-run-out). | <code>None</code>
`roi_mask` | <code>[Nifti1Image](#Nifti1Image) or [path](#path) - [like](#like)</code> | Atlas image for ``spatial_scale='roi'``. | <code>None</code>
`radius_mm` | <code>[float](#float)</code> | Searchlight radius in mm. Default ``10.0``. | <code>10.0</code>
`inplace` | <code>[bool](#bool)</code> | If ``True``, populate result fields as ``predict_*`` attributes on ``self`` and return ``self``. Default ``False`` returns a fresh `Predict`. | <code>False</code>
`n_jobs` | <code>[int](#int)</code> | Parallel jobs for searchlight / ROI. Default ``1``; searchlight on a real brain at higher ``n_jobs`` can be memory-heavy. | <code>1</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar for searchlight / ROI. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
 | Predict | BrainData: ``Predict`` dataclass when ``inplace=False``; ``self`` (mutated, with ``predict_*`` attrs) when ``inplace=True``.

**Examples:**

```pycon
>>> result = brain.predict(y=labels, spatial_scale='whole_brain', cv=5)
>>> result.weight_map.plot()       # publishable map (all-data fit)
>>> result.mean_score              # honest CV-derived accuracy
>>> new_pred = result.estimator.predict(new_X)  # apply to new data
```

```pycon
>>> result = brain.predict(y=labels, spatial_scale='searchlight',
...                        radius_mm=8.0, n_jobs=4)
>>> result.accuracy_map.plot()
```

```pycon
>>> result = brain.predict(y=labels, spatial_scale='roi', roi_mask=atlas)
>>> top = result.roi_labels[result.mean_score.argsort()[::-1][:10]]
>>> result.accuracy_map.plot()  # brain-space view of the same map
```

Custom sklearn pipeline as model — standardize auto-defaults to
False because we detect the Pipeline:

```python
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
pipe = make_pipeline(StandardScaler(), SelectKBest(k=500),
                     LinearSVC())
result = brain.predict(y=labels, model=pipe)
```

(data-brain-data-r-to-z)=
#### `r_to_z`

```python
r_to_z()
```

Apply Fisher's r-to-z transformation to each data element.

(data-brain-data-regions)=
#### `regions`

```python
regions(min_region_size = 1350, method = 'local_regions', smoothing_fwhm = 6, is_mask = False)
```

Extract brain connected regions into separate regions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`min_region_size` | <code>[int](#int)</code> | Minimum volume in mm3 for a region to be kept. | <code>1350</code>
`method` | <code>[str](#str)</code> | Type of extraction method                 ['connected_components', 'local_regions']. | <code>'local_regions'</code>
`smoothing_fwhm` | <code>[scalar](#scalar)</code> | Smooth an image to extract more sparser regions. | <code>6</code>
`is_mask` | <code>[bool](#bool)</code> | Whether to treat as boolean mask. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with extracted ROIs as data.

(data-brain-data-resample-to)=
#### `resample_to`

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

(data-brain-data-scale)=
#### `scale`

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

(data-brain-data-similarity)=
#### `similarity`

```python
similarity(image, method = 'correlation')
```

Calculate similarity to a single BrainData or nibabel image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`image` |  | (BrainData, nifti) image to evaluate similarity | *required*
`method` |  | (str) Type of similarity     ['correlation','dot_product','cosine'] | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
 | float or np.ndarray: Similarity value(s).

(data-brain-data-smooth)=
#### `smooth`

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

(data-brain-data-standardize)=
#### `standardize`

```python
standardize(axis = 0, method = 'center', verbose = True)
```

Standardize BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>[int](#int)</code> | 0 standardizes each voxel across observations (default). 1 standardizes each observation across voxels. | <code>0</code>
`method` | <code>[str](#str)</code> | 'center' subtracts the mean (default). 'zscore' subtracts the mean and divides by standard deviation. | <code>'center'</code>
`verbose` | <code>[bool](#bool)</code> | If False, suppress sklearn numerical warnings that occur when voxels have near-zero variance. Default: True. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Standardized BrainData instance.

(data-brain-data-std)=
#### `std`

```python
std(axis = 0, *, spatial_scale: str = 'whole_brain', roi_mask: str = None)
```

Get standard deviation of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` |  | 0 = across images (default, returns BrainData), 1 = within images (returns array). Ignored when ``spatial_scale='roi'``. | <code>0</code>
`spatial_scale` | <code>[str](#str)</code> | ``'whole_brain'`` (default) or ``'roi'`` (paints each voxel with its parcel's std per image). | <code>'whole_brain'</code>
`roi_mask` |  | Atlas image for ``spatial_scale='roi'``. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | float/np.array/BrainData: Standard deviation values.

(data-brain-data-sum)=
#### `sum`

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

(data-brain-data-temporal-resample)=
#### `temporal_resample`

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

(data-brain-data-threshold)=
#### `threshold`

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

(data-brain-data-to-nifti)=
#### `to_nifti`

```python
to_nifti()
```

Convert BrainData Instance into Nifti Object.

**Returns:**

Type | Description
---- | -----------
 | nibabel.Nifti1Image: Brain data as a NIfTI image.

(data-brain-data-transform-pairwise)=
#### `transform_pairwise`

```python
transform_pairwise()
```

Transform data into pairwise comparisons.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance transformed into pairwise comparisons

(data-brain-data-ttest)=
#### `ttest`

```python
ttest(popmean = 0.0, permutation = False, n_permute = 5000, tail = 2, return_null = False, n_jobs = -1, random_state = None)
```

One-sample voxelwise t-test across images (axis 0).

Tests whether the per-voxel mean across images differs from
``popmean``. Operates on a stack of images (e.g. subject-level
contrast maps) with shape ``(n_samples, n_voxels)``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`popmean` |  | Population mean to test against. Default 0.0. | <code>0.0</code>
`permutation` |  | If True, use sign-flip permutation test via `one_sample_permutation_test`. | <code>False</code>
`n_permute` |  | Number of permutations (used only when ``permutation=True``). Default 5000. | <code>5000</code>
`tail` |  | Tail of the test (1 or 2). Default 2. | <code>2</code>
`return_null` |  | If True, also return the null distribution. Default False. | <code>False</code>
`n_jobs` |  | Number of parallel jobs. Default -1 (all cores). | <code>-1</code>
`random_state` |  | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | dict with four BrainData keys:<br>- ``"mean"``: voxelwise mean across images (effect size). - ``"t"``: parametric one-sample t-statistic. - ``"z"``: signed z-score, ``sign(t) * norm.isf(p/2)`` —   matches nilearn's ``output_type='z_score'``. - ``"p"``: parametric p-value, or empirical p when   ``permutation=True``.
 | The effect size is always returned alongside the inferential maps
 | so group-level code never has to recompute the mean.

**Examples:**

```pycon
>>> # Stack of subject-level contrast maps
>>> result = contrast_maps.ttest()
>>> sig = result["p"].data < 0.05
>>> effect = result["mean"]       # for reporting magnitude
>>> z_map = result["z"]           # for nilearn-style thresholding
```

```pycon
>>> # Permutation-based p-values; still reports t/z/mean
>>> result = contrast_maps.ttest(permutation=True, n_permute=5000)
```

(data-brain-data-ttest2)=
#### `ttest2`

```python
ttest2(other, equal_var = True)
```

Two-sample voxelwise t-test between two BrainData stacks.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` |  | BrainData to compare against. Must have the same number of voxels. | *required*
`equal_var` |  | If True (default), standard two-sample t-test. If False, Welch's t-test. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | ``{"t": BrainData, "p": BrainData}``.

(data-brain-data-upload-neurovault)=
#### `upload_neurovault`

```python
upload_neurovault(access_token = None, collection_name = None, collection_id = None, img_type = None, img_modality = None, **kwargs)
```

Upload BrainData images and metadata to NeuroVault.

Adds any columns in ``self.X`` to image metadata. The index is used as
the image name.

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

(data-brain-data-write)=
#### `write`

```python
write(file_name)
```

Write out BrainData object to Nifti or HDF5 File.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>[str](#str) or [Path](#Path)</code> | Output file path (.nii/.nii.gz for NIfTI, .h5/.hdf5 for HDF5). | *required*

(data-brain-data-z-to-r)=
#### `z_to_r`

```python
z_to_r()
```

Convert z score back into r value for each element of data object.

