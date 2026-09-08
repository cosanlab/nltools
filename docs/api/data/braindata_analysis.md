---
title: data.braindata.analysis
label: page-data-braindata-analysis
---

Analysis operations on `BrainData`.

Functions for similarity, distance, masking, ROI extraction, filtering,
thresholding, decomposition, alignment, smoothing, and related operations.
Each takes a `BrainData` as its first argument; the corresponding
`BrainData` methods delegate here.

**Functions:**

Name | Description
---- | -----------
[`align`](#data-braindata-analysis-align) | Align a BrainData instance to a target using functional alignment.
[`align_per_roi`](#data-braindata-analysis-align-per-roi) | Per-parcel functional alignment + voxel-space reassembly.
[`apply_mask`](#data-braindata-analysis-apply-mask) | Mask BrainData instance using nilearn functionality.
[`check_masks`](#data-braindata-analysis-check-masks) | Ensure two datasets use compatible masks, creating a union mask if needed.
[`decompose`](#data-braindata-analysis-decompose) | Decompose a BrainData object.
[`detrend_data`](#data-braindata-analysis-detrend-data) | Remove the linear trend from each voxel.
[`distance`](#data-braindata-analysis-distance) | Calculate distance between images within a BrainData() instance.
[`extract_roi`](#data-braindata-analysis-extract-roi) | Extract activity from a binary mask or a labeled ROI atlas.
[`filter_data`](#data-braindata-analysis-filter-data) | Apply a Butterworth filter to data (wraps `nilearn.signal.clean`).
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
[`z_to_r`](#data-braindata-analysis-z-to-r) | Convert Fisher z scores back into r values for each data element.



## Functions

(data-braindata-analysis-align)=
### `align`

```python
align(bd, target, method = 'procrustes', axis = 0)
```

Align a BrainData instance to a target using functional alignment.

Alignment type can be hyperalignment or Shared Response Model. When
using hyperalignment, `target` image can be another subject or an
already estimated common model. When using SRM, `target` must be a previously
estimated common model stored as a numpy array. Transformed data can be back
projected to original data using Transformation matrix.

See `nltools.algorithms.align` for aligning multiple BrainData instances.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data to align. | *required*
`target` | <code>[BrainData](#page-data-brain-data) \| ndarray</code> | Alignment target — another subject or a fitted common model (array) for the SRM methods. | *required*
`method` | <code>str</code> | ``'procrustes'`` (default), ``'probabilistic_srm'``, or ``'deterministic_srm'``. | <code>'procrustes'</code>
`axis` | <code>int</code> | Axis to align on. Default ``0``. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | ``'transformed'``, ``'transformation_matrix'``, and     ``'common_model'`` (plus ``'disparity'`` and ``'scale'`` for     ``'procrustes'``).

**Examples:**

```python
# Hyperalign using procrustes transform
out = data.align(target, method='procrustes')

# Align using shared response model
out = data.align(target, method='probabilistic_srm')

# Project aligned data back into original data space
original_data = np.dot(out['transformed'].data, out['transformation_matrix'].T)
```

(data-braindata-analysis-align-per-roi)=
### `align_per_roi`

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

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Source data to align. | *required*
`target` | <code>[BrainData](#page-data-brain-data) \| ndarray</code> | Alignment target (a `BrainData` for ``'procrustes'``; a common-model array for the SRM methods). | *required*
`method` | <code>str</code> | ``'procrustes'``, ``'probabilistic_srm'``, or ``'deterministic_srm'``. | *required*
`axis` | <code>int</code> | Axis to align over; see `align`. | *required*
`roi_mask` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image \| str</code> | Integer-labeled atlas defining the parcels. | *required*

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | ``'transformed'`` (`BrainData`), ``'transformation_matrix'`` and     ``'common_model'`` (dicts keyed by atlas label), ``'disparity'`` and     ``'scale'`` (arrays, one entry per parcel), and ``'roi_labels'``.

(data-braindata-analysis-apply-mask)=
### `apply_mask`

```python
apply_mask(bd, mask, resample_mask_to_brain = False)
```

Mask BrainData instance using nilearn functionality.

Note target data will be resampled into the same space as the mask. If you would like the mask
resampled into the BrainData space, then set resample_mask_to_brain=True.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data to mask. | *required*
`mask` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image</code> | Mask to apply. | *required*
`resample_mask_to_brain` | <code>bool</code> | Resample the mask into the brain's space before applying it. Default: ``False``. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Masked copy of ``bd``.

<details class="note" open markdown="1">
<summary>Note</summary>

Masking is delegated to ``nilearn.masking.apply_mask``.

</details>

(data-braindata-analysis-check-masks)=
### `check_masks`

```python
check_masks(bd, image)
```

Ensure two datasets use compatible masks, creating a union mask if needed.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Reference dataset. | *required*
`image` | <code>[BrainData](#page-data-brain-data)</code> | Dataset whose mask is compared with ``bd``'s. | *required*

**Returns:**

Type | Description
---- | -----------
<code>tuple[ndarray, ndarray]</code> | ``(data, image_data)`` arrays sampled on     a shared mask.

(data-braindata-analysis-decompose)=
### `decompose`

```python
decompose(bd, *, method = 'pca', axis = 'voxels', n_components = None, **kwargs)
```

Decompose a BrainData object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data to decompose. | *required*
`method` | <code>str</code> | Decomposition algorithm: ``'pca'`` (default), ``'ica'``, ``'nnmf'``, ``'fa'``, ``'dictionary'``, or ``'kernelpca'``. | <code>'pca'</code>
`axis` | <code>str</code> | Dimension to decompose: ``'voxels'`` (default) or ``'images'``. | <code>'voxels'</code>
`n_components` | <code>int \| None</code> | Number of components. ``None`` retains as many as possible. | <code>None</code>
`**kwargs` | <code>dict</code> | Forwarded to the ``sklearn.decomposition`` estimator. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | ``'decomposition_object'`` (the fitted sklearn estimator),     ``'components'`` (`BrainData`), and ``'weights'`` (array).

(data-braindata-analysis-detrend-data)=
### `detrend_data`

```python
detrend_data(bd, method = 'linear')
```

Remove the linear trend from each voxel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data to detrend (must hold more than one image). | *required*
`method` | <code>str</code> | ``'linear'`` (default) or ``'constant'``. | <code>'linear'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Detrended copy of ``bd``.

(data-braindata-analysis-distance)=
### `distance`

```python
distance(bd, metric = 'euclidean', *, spatial_scale: str = 'whole_brain', roi_mask: str = None, radius_mm: float = 10.0, **kwargs: float)
```

Calculate distance between images within a BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Dataset whose images are compared. | *required*
`metric` | <code>str</code> | Any distance metric supported by ``scipy.spatial.distance.cdist`` (e.g. ``'euclidean'``, ``'cityblock'``, ``'cosine'``, ``'correlation'``, ``'hamming'``, ``'jaccard'``). | <code>'euclidean'</code>
`spatial_scale` | <code>str</code> | ``'whole_brain'`` (default), ``'roi'``, or ``'searchlight'``. See `BrainData.distance`. | <code>'whole_brain'</code>
`roi_mask` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image \| str \| None</code> | Atlas for ``spatial_scale='roi'``. | <code>None</code>
`radius_mm` | <code>float</code> | Searchlight radius for ``spatial_scale='searchlight'``. | <code>10.0</code>
`**kwargs` | <code>dict</code> | Forwarded to ``scipy.spatial.distance.cdist``. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[Adjacency](#page-data-adjacency)</code> | Whole-brain pairwise distance matrix, or an ordinary stack.     ROI matrices follow sorted nonzero atlas labels present in the source     mask after resampling; searchlights follow source-mask voxel order.

(data-braindata-analysis-extract-roi)=
### `extract_roi`

```python
extract_roi(bd, mask, method = 'mean', n_components = None)
```

Extract activity from a binary mask or a labeled ROI atlas.

Labeled atlases (multiple ROIs) are handled with nilearn's
``NiftiLabelsMasker``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data to extract from. | *required*
`mask` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image \| str</code> | A binary mask (extracts from a single ROI) or a labeled atlas (extracts from every ROI). | *required*
`method` | <code>str</code> | Extraction method: ``'mean'`` (default), ``'median'``, or ``'pca'``. | <code>'mean'</code>
`n_components` | <code>int \| None</code> | Number of components to return when ``method='pca'``. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>float \| ndarray</code> | For a binary mask, a scalar (single image) or 1D array     of values (multiple images). For a labeled atlas, a 1D array with one     value per ROI (single image), a 2D array of images x ROIs (multiple     images), or the components array when `method='pca'`.

**Examples:**

```python
# Extract mean from binary mask
roi_values = brain.extract_roi(binary_mask)

# Extract from atlas
atlas_values = brain.extract_roi(atlas_mask)

# PCA extraction
components = brain.extract_roi(mask, method='pca', n_components=5)
```

(data-braindata-analysis-filter-data)=
### `filter_data`

```python
filter_data(bd, *, sampling_freq = None, high_pass = None, low_pass = None, **kwargs)
```

Apply a Butterworth filter to data (wraps `nilearn.signal.clean`).

Does not default to detrending and standardizing like nilearn
implementation, but this can be overridden using kwargs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Time series to filter. | *required*
`sampling_freq` | <code>float \| None</code> | Sampling frequency in Hz (i.e. 1 / TR). | <code>None</code>
`high_pass` | <code>float \| None</code> | High-pass cutoff frequency in Hz. | <code>None</code>
`low_pass` | <code>float \| None</code> | Low-pass cutoff frequency in Hz. | <code>None</code>
`**kwargs` | <code>dict</code> | Forwarded to ``nilearn.signal.clean``. Common options: ``confounds`` (confound time series to remove), ``sample_mask`` (volumes to exclude), ``detrend`` (default ``False``), ``standardize`` (``'zscore_sample'``, ``'psc'``, or ``None`` — the default; ``True``/``False`` are accepted as aliases for ``'zscore_sample'``/``None``), and ``ensure_finite`` (replace NaN/inf; default ``False``). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Filtered copy of ``bd``.

<details class="see-also" open markdown="1">
<summary>See Also</summary>

``nilearn.signal.clean`` for all available options.

</details>

(data-braindata-analysis-find-spikes-data)=
### `find_spikes_data`

```python
find_spikes_data(bd, global_spike_cutoff = 3, diff_spike_cutoff = 3, *, TR = None, sampling_freq = None)
```

Identify spikes from time-series data; see `find_spikes`.

(data-braindata-analysis-multivariate-similarity)=
### `multivariate_similarity`

```python
multivariate_similarity(bd, images, method = 'ols', tail = 2)
```

Predict a BrainData spatial distribution from a linear combination.

The predictors may be other BrainData instances or nibabel images.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Single image to be explained. | *required*
`images` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image</code> | Predictor images (weight maps). | *required*
`method` | <code>str</code> | Regression method. Default: ``'ols'``. | <code>'ols'</code>
`tail` | <code>int</code> | ``1`` or ``2`` for one- or two-tailed p-values. | <code>2</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Raw regression statistics (numpy arrays/scalars, not BrainData)     with keys ``'beta'``, ``'t'``, ``'p'``, ``'df'``, ``'sigma'``,     ``'residual'``.

(data-braindata-analysis-r-to-z)=
### `r_to_z`

```python
r_to_z(bd)
```

Apply Fisher's r-to-z transformation to each data element.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Correlation values to transform. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Transformed copy of ``bd``.

(data-braindata-analysis-reduce-per-roi)=
### `reduce_per_roi`

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

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data to reduce. | *required*
`reducer` | <code>Callable</code> | NumPy-style reducer accepting ``axis=``, e.g. ``np.mean``. | *required*
`roi_mask` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image \| str</code> | Integer-labeled atlas defining the parcels. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Parcel-wise reduced values painted back to voxel space.

(data-braindata-analysis-regions)=
### `regions`

```python
regions(bd, *, min_region_size = 1350, method = 'local_regions', smoothing_fwhm = 6, is_mask = False)
```

Extract brain connected regions into separate regions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Image to segment. | *required*
`min_region_size` | <code>int</code> | Minimum volume in mm³ for a region to be kept. | <code>1350</code>
`method` | <code>str</code> | ``'connected_components'`` labels each connected component directly; ``'local_regions'`` (default) seeds a marker at each component's peak and separates regions with a random-walker segmentation. | <code>'local_regions'</code>
`smoothing_fwhm` | <code>float</code> | Smooth the image first to extract sparser regions. Only used for ``method='local_regions'``. | <code>6</code>
`is_mask` | <code>bool</code> | Treat ``bd`` as a boolean mask and use ``connected_label_regions`` instead. Default ``False``. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | One image per extracted region.

(data-braindata-analysis-scale-data)=
### `scale_data`

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
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data to scale. | *required*
`scale_val` | <code>float</code> | Target value for the mean after scaling. Default ``100``. | <code>100.0</code>
`axis` | <code>int \| None</code> | ``None`` for grand-mean scaling (default, FSL/SPM style); ``0`` for voxel-wise scaling (AFNI style, each voxel scaled by its own temporal mean). | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Scaled copy of ``bd``.

**Examples:**

```python
# Grand-mean scaling (default)
scaled = brain.scale(100.0)

# Voxel-wise scaling (AFNI style)
scaled = brain.scale(100.0, axis=0)
```

(data-braindata-analysis-similarity)=
### `similarity`

```python
similarity(bd, image, metric = 'correlation')
```

Calculate similarity to a single BrainData or nibabel image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Dataset to compare. | *required*
`image` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image</code> | Image to evaluate similarity against. | *required*
`metric` | <code>str</code> | Similarity metric, one of ``'correlation'``, ``'pearson'``, ``'rank_correlation'``, ``'spearman'``, ``'dot_product'``, or ``'cosine'``. | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Similarity values.

(data-braindata-analysis-smooth)=
### `smooth`

```python
smooth(bd, fwhm)
```

Apply spatial smoothing using nilearn's ``smooth_img``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data to smooth. | *required*
`fwhm` | <code>float</code> | Full width at half maximum of the Gaussian kernel, in mm. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Smoothed copy of ``bd``.

(data-braindata-analysis-standardize)=
### `standardize`

```python
standardize(bd, *, axis = 0, method = 'center')
```

Standardize BrainData() instance.

Computed in float64 and cast back to the input dtype, so raw float32 BOLD
(large offsets) stays exact. Constant voxels/observations z-score to 0.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data to standardize. | *required*
`axis` | <code>int</code> | ``0`` to standardize each voxel across observations (default), ``1`` to standardize each observation across voxels. | <code>0</code>
`method` | <code>str</code> | ``'center'`` (default) or ``'zscore'``. | <code>'center'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Standardized copy of ``bd``.

(data-braindata-analysis-temporal-resample)=
### `temporal_resample`

```python
temporal_resample(bd, *, sampling_freq = None, target = None, target_type = 'hz')
```

Resample a BrainData time series to a target frequency or sample count.

Resample BrainData timeseries to a new target frequency or number of samples
using Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) interpolation.
This function can up- or down-sample data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Time series to resample. | *required*
`sampling_freq` | <code>float \| None</code> | Sampling frequency of the data in Hz. | <code>None</code>
`target` | <code>float \| None</code> | Resampling target, interpreted per ``target_type``. | <code>None</code>
`target_type` | <code>str</code> | Units of ``target``: ``'hz'`` (default), ``'samples'``, or ``'seconds'``. | <code>'hz'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Resampled copy of ``bd``.

<details class="note" open markdown="1">
<summary>Note</summary>

This function can use quite a bit of RAM.

</details>

(data-braindata-analysis-threshold-data)=
### `threshold_data`

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
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data to threshold. | *required*
`upper` | <code>float \| str \| None</code> | Upper cutoff. A string like ``'98%'`` resolves as a percentile over the finite **nonzero** voxels (via `nltools.utils.resolve_threshold` — zeros on a masked map are absence of data and would skew the percentile). ``None`` for one-sided thresholding. | <code>None</code>
`lower` | <code>float \| str \| None</code> | Lower cutoff, with the same percentile semantics as ``upper``. ``None`` for one-sided thresholding. | <code>None</code>
`binarize` | <code>bool</code> | Return a binary image respecting the thresholds if provided, otherwise binarize every non-zero value. Default ``False``. | <code>False</code>
`coerce_nan` | <code>bool</code> | Replace NaN values with 0 first. Default ``True``. | <code>True</code>
`cluster_threshold` | <code>int</code> | Minimum cluster size in voxels. If ``> 0``, thresholds with ``nilearn.image.threshold_img`` and drops smaller clusters; band-pass thresholding (both ``upper`` and ``lower``) is not supported in that mode. Default ``0`` (disabled). | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Thresholded copy of ``bd``.

<details class="note" open markdown="1">
<summary>Note</summary>

With ``cluster_threshold=0`` (default) thresholding runs on the data
array directly and supports band-pass thresholds; with
``cluster_threshold>0`` nilearn performs the cluster filtering.

</details>

(data-braindata-analysis-transform-pairwise-data)=
### `transform_pairwise_data`

```python
transform_pairwise_data(bd)
```

Transform BrainData into pairwise comparisons.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | Data with a ``Y`` column to compare pairwise. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Pairwise-difference images with a recoded ``Y``.

(data-braindata-analysis-z-to-r)=
### `z_to_r`

```python
z_to_r(bd)
```

Convert Fisher z scores back into r values for each data element.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`bd` | <code>[BrainData](#page-data-brain-data)</code> | z-scored values to transform. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Transformed copy of ``bd``.
