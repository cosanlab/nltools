(data-braindata-analysis-analysis)=
## `analysis`

BrainData analysis functions.

Standalone functions extracted from BrainData class methods for similarity,
distance, masking, ROI extraction, filtering, thresholding, decomposition,
alignment, smoothing, and other analytical operations.

**Methods:**

Name | Description
---- | -----------
[`align`](#data-braindata-analysis-align) | Align a BrainData instance to a target using functional alignment.
[`align_per_roi`](#data-braindata-analysis-align-per-roi) | Per-parcel functional alignment + voxel-space reassembly.
[`apply_mask`](#data-braindata-analysis-apply-mask) | Mask BrainData instance using nilearn functionality.
[`check_masks`](#data-braindata-analysis-check-masks) | Ensure two datasets use compatible masks, creating a union mask if needed.
[`decompose`](#data-braindata-analysis-decompose) | Decompose a BrainData object.
[`detrend_data`](#data-braindata-analysis-detrend-data) | Remove the linear trend from each voxel.
[`distance`](#data-braindata-analysis-distance) | Calculate distance between images within a BrainData() instance.
[`extract_roi`](#data-braindata-analysis-extract-roi) | Extract activity from mask or ROI atlas using NiftiLabelsMasker.
[`filter_data`](#data-braindata-analysis-filter-data) | Apply butterworth filter to data. Wraps nilearn.signal.clean.
[`find_spikes_data`](#data-braindata-analysis-find-spikes-data) | Identify spikes from time-series data; see `find_spikes`.
[`multivariate_similarity`](#data-braindata-analysis-multivariate-similarity) | Predict a BrainData spatial distribution from a linear combination.
[`r_to_z`](#data-braindata-analysis-r-to-z) | Apply Fisher's r-to-z transformation to each data element.
[`reduce_per_roi`](#data-braindata-analysis-reduce-per-roi) | Apply a reducer within each parcel and paint results back to voxel space.
[`regions`](#data-braindata-analysis-regions) | Extract brain connected regions into separate regions.
[`scale_data`](#data-braindata-analysis-scale-data) | Scale data via mean scaling.
[`similarity`](#data-braindata-analysis-similarity) | Calculate similarity to a single BrainData or nibabel image.
[`smooth`](#data-braindata-analysis-smooth) | Apply spatial smoothing using nilearn's ``smooth_img``.
[`standardize`](#data-braindata-analysis-standardize) | Standardize BrainData() instance.
[`temporal_resample`](#data-braindata-analysis-temporal-resample) | Resample a BrainData time series to a target frequency or sample count.
[`threshold_data`](#data-braindata-analysis-threshold-data) | Threshold BrainData instance with optional cluster filtering.
[`transform_pairwise_data`](#data-braindata-analysis-transform-pairwise-data) | Transform BrainData into pairwise comparisons.
[`z_to_r`](#data-braindata-analysis-z-to-r) | Convert z score back into r value for each element of data object.



### Methods

(data-braindata-analysis-align)=
#### `align`

```python
align(bd, target, method = 'procrustes', axis = 0)
```

Align a BrainData instance to a target using functional alignment.

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

(data-braindata-analysis-align-per-roi)=
#### `align_per_roi`

```python
align_per_roi(bd, target, *, method, axis, roi_mask)
```

Per-parcel functional alignment + voxel-space reassembly.

For each atlas parcel, runs ``align()`` on the slice of ``bd`` and
``target`` restricted to that parcel's voxels and collects results.
The ``transformed`` field is reassembled into a single
`BrainData` of the same shape as the input (each voxel filled
with its parcel's transformed value per image; voxels outside any
parcel = NaN). Per-parcel transform matrices and common-model
objects are kept as dicts keyed by atlas label, since matrices over
different voxel subsets can't be painted into one image.

(data-braindata-analysis-apply-mask)=
#### `apply_mask`

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

(data-braindata-analysis-check-masks)=
#### `check_masks`

```python
check_masks(bd, image)
```

Ensure two datasets use compatible masks, creating a union mask if needed.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance | *required*
`image` |  | BrainData instance to compare masks with | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` |  | (data2, image2) arrays with compatible masks

(data-braindata-analysis-decompose)=
#### `decompose`

```python
decompose(bd, *, method = 'pca', axis = 'voxels', n_components = None, **kwargs)
```

Decompose a BrainData object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`method` |  | (str) Algorithm to perform decomposition         types=['pca','ica','nnmf','fa','dictionary','kernelpca'] | <code>'pca'</code>
`axis` |  | dimension to decompose ['voxels','images'] | <code>'voxels'</code>
`n_components` |  | (int) number of components. If None then retain         as many as possible (default: None). | <code>None</code>
`**kwargs` |  | Additional keyword arguments passed to the decomposition algorithm. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`output` |  | a dictionary of decomposition parameters

(data-braindata-analysis-detrend-data)=
#### `detrend_data`

```python
detrend_data(bd, method = 'linear')
```

Remove the linear trend from each voxel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`method` |  | ('linear','constant', optional) type of detrending | <code>'linear'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (BrainData) detrended BrainData instance

(data-braindata-analysis-distance)=
#### `distance`

```python
distance(bd, metric = 'euclidean', *, spatial_scale: str = 'whole_brain', roi_mask: str = None, radius_mm: float = 10.0, **kwargs: float)
```

Calculate distance between images within a BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`metric` |  | (str) type of distance metric (can use any scipy.spatial.distance     metric supported by cdist, e.g., 'euclidean', 'cityblock', 'cosine',     'correlation', 'hamming', 'jaccard', etc.) | <code>'euclidean'</code>
`spatial_scale` | <code>[str](#str)</code> | ``'whole_brain'`` (default), ``'roi'``, or ``'searchlight'``. See `BrainData.distance`. | <code>'whole_brain'</code>
`roi_mask` |  | Atlas for ``spatial_scale='roi'``. | <code>None</code>
`radius_mm` | <code>[float](#float)</code> | Searchlight radius for ``spatial_scale='searchlight'``. | <code>10.0</code>
`**kwargs` |  | Additional arguments passed to scipy.spatial.distance.cdist. | <code>{}</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dist` |  | (Adjacency) Whole-brain pairwise distance matrix, or a stacked Adjacency (one per parcel/searchlight) with ``spatial_scale`` provenance set.

(data-braindata-analysis-extract-roi)=
#### `extract_roi`

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

(data-braindata-analysis-filter-data)=
#### `filter_data`

```python
filter_data(bd, *, sampling_freq = None, high_pass = None, low_pass = None, **kwargs)
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

(data-braindata-analysis-find-spikes-data)=
#### `find_spikes_data`

```python
find_spikes_data(bd, global_spike_cutoff = 3, diff_spike_cutoff = 3, *, TR = None, sampling_freq = None)
```

Identify spikes from time-series data; see `find_spikes`.

(data-braindata-analysis-multivariate-similarity)=
#### `multivariate_similarity`

```python
multivariate_similarity(bd, images, method = 'ols')
```

Predict a BrainData spatial distribution from a linear combination.

The predictors may be other BrainData instances or nibabel images.

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

(data-braindata-analysis-r-to-z)=
#### `r_to_z`

```python
r_to_z(bd)
```

Apply Fisher's r-to-z transformation to each data element.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Transformed BrainData instance.

(data-braindata-analysis-reduce-per-roi)=
#### `reduce_per_roi`

```python
reduce_per_roi(bd, reducer, *, roi_mask)
```

Apply a reducer within each parcel and paint results back to voxel space.

This performs spatial smoothing via parcellation using a reducer such as
``np.mean``.

For each image ``i`` and each parcel ``p``, computes
``reducer(bd.data[i, voxels-in-p])`` and assigns that scalar to every
voxel in parcel ``p`` for image ``i``. Voxels outside any parcel get
NaN. Output is a `BrainData` of the same shape as the input.

Used by ``BrainData.{mean,std,median}(spatial_scale='roi')``.

(data-braindata-analysis-regions)=
#### `regions`

```python
regions(bd, *, min_region_size = 1350, method = 'local_regions', smoothing_fwhm = 6, is_mask = False)
```

Extract brain connected regions into separate regions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`min_region_size` | <code>[int](#int)</code> | Minimum volume in mm3 for a region to be                 kept. | <code>1350</code>
`method` | <code>[str](#str)</code> | Type of extraction method                 ['connected_components', 'local_regions'].                 If 'connected_components', each component/region                 in the image is extracted automatically by                 labelling each region based upon the presence of                 unique features in their respective regions.                 If 'local_regions', each component/region is                 extracted based on their maximum peak value to                 define a seed marker and then using random                 walker segementation algorithm on these                 markers for region separation. | <code>'local_regions'</code>
`smoothing_fwhm` | <code>[scalar](#scalar)</code> | Smooth an image to extract more sparser                 regions. Only works for method='local_regions'. | <code>6</code>
`is_mask` | <code>[bool](#bool)</code> | Whether the BrainData instance should be treated             as a boolean mask and if so, calls             connected_label_regions instead. Default: False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | BrainData instance with extracted ROIs as data.

(data-braindata-analysis-scale-data)=
#### `scale_data`

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

(data-braindata-analysis-similarity)=
#### `similarity`

```python
similarity(bd, image, method = 'correlation')
```

Calculate similarity to a single BrainData or nibabel image.

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

(data-braindata-analysis-smooth)=
#### `smooth`

```python
smooth(bd, fwhm)
```

Apply spatial smoothing using nilearn's ``smooth_img``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`fwhm` |  | (float) full width half maximum of gaussian spatial filter | *required*

**Returns:**

Type | Description
---- | -----------
 | BrainData instance (copy with smoothed data)

(data-braindata-analysis-standardize)=
#### `standardize`

```python
standardize(bd, *, axis = 0, method = 'center', verbose = True)
```

Standardize BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` |  | BrainData instance. | *required*
`axis` |  | 0 for observations 1 for voxels (default: 0) | <code>0</code>
`method` |  | ['center','zscore'] (default: 'center') | <code>'center'</code>
`verbose` |  | If False, suppress sklearn numerical warnings that occur when voxels have near-zero variance. (default: True) | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`BrainData` |  | Standardized BrainData instance.

(data-braindata-analysis-temporal-resample)=
#### `temporal_resample`

```python
temporal_resample(bd, *, sampling_freq = None, target = None, target_type = 'hz')
```

Resample a BrainData time series to a target frequency or sample count.

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

(data-braindata-analysis-threshold-data)=
#### `threshold_data`

```python
threshold_data(bd, *, upper = None, lower = None, binarize = False, coerce_nan = True, cluster_threshold = 0)
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

(data-braindata-analysis-transform-pairwise-data)=
#### `transform_pairwise_data`

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

(data-braindata-analysis-z-to-r)=
#### `z_to_r`

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

