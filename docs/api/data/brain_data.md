---
title: BrainData
label: page-data-brain-data
---

```python
BrainData(data = None, *, Y = None, X = None, mask = None, masker = None, h5_compression = 'gzip', verbose = False, resample = True, interpolation = 'auto')
```

Represent neuroimaging data as vectors instead of three-dimensional matrices.

Each image is flattened to its in-mask voxels, so a stack of images is a 2D
``(n_images, n_voxels)`` array. This representation makes it easier to perform
data manipulation and analyses.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>None \| [BrainData](#page-data-brain-data) \| list \| str \| Path \| Nifti1Image \| ndarray</code> | Neuroimaging data. Accepts ``None`` (an empty BrainData), another BrainData, a list of BrainData objects or file paths, a file path to ``.nii``/``.nii.gz``/``.h5``/``.hdf5``, a nibabel ``Nifti1Image``, a URL to download from, or a numpy array (1D ``(n_voxels,)`` for a single image or 2D ``(n_images, n_voxels)`` for a stack). Array input requires ``mask``, whose in-mask voxel count must match the array's last axis. | <code>None</code>
`mask` | <code>None \| Nifti1Image \| str \| Path</code> | Brain mask. ``None`` uses the MNI template; otherwise a nibabel ``Nifti1Image``, a file path to a mask file, or a template name string like ``'2mm-MNI152-2009c'`` (version: ``'fsl'`` for default/, ``'a'`` for nilearn/, ``'c'`` for fmriprep/). | <code>None</code>
`masker` | <code>nilearn masker \| None</code> | nilearn masker object (e.g. ROI or searchlight extractor). Default ``None`` loads data as voxels. | <code>None</code>
`Y` | <code>DataFrame \| ndarray \| str \| None</code> | Optional per-image target/label values, stored as a polars DataFrame (``.Y``). Default ``None``. If ``data`` is a BrainData with a ``.Y``, that value is inherited when this is ``None``. | <code>None</code>
`X` | <code>DataFrame \| ndarray \| str \| None</code> | Optional per-image design/feature values, stored as a polars DataFrame (``.X``). Default ``None``. If ``data`` is a BrainData with an ``.X``, that value is inherited when this is ``None``. | <code>None</code>
`h5_compression` | <code>str</code> | Compression filter used when writing HDF5 (``.h5``/``.hdf5``) output. Default ``'gzip'``. | <code>'gzip'</code>
`verbose` | <code>bool</code> | Emit informational messages during loading and other operations. Default ``False``. | <code>False</code>
`resample` | <code>bool</code> | Whether to automatically resample data to mask space. If ``True`` (default), data is resampled to match the mask's spatial characteristics. If ``False``, data must already be in mask space. | <code>True</code>
`interpolation` | <code>str</code> | Interpolation method for resampling. ``'auto'`` (default) detects based on data type — ``'nearest'`` for discrete data like atlases/masks and ``'continuous'`` for stat maps; ``'nearest'`` (nearest-neighbor, preserves discrete values), ``'linear'`` (linear interpolation), or ``'continuous'`` (higher-order spline, use for stat maps). | <code>'auto'</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`data` | <code>ndarray</code> | In-mask voxel values, shape ``(n_voxels,)`` for a single image or ``(n_images, n_voxels)`` for a stack.
`mask` | <code>Nifti1Image</code> | The brain mask every image is flattened against.
`masker` | <code>nilearn masker \| None</code> | Masker used to extract data, or ``None`` when data are plain voxels.
`design_matrix` | <code>[DesignMatrix](#page-data-design-matrix) \| None</code> | Design matrix attached by ``fit(model='glm', ...)``; ``None`` until a GLM is fit.
`verbose` | <code>bool</code> | Whether informational messages are emitted.
`X` | <code>DataFrame</code> | Design matrix / per-image covariates (possibly empty).
`Y` | <code>DataFrame</code> | Per-image targets (possibly empty).
`dtype` | <code>dtype</code> | Data type of ``data``.
`is_empty` | <code>bool</code> | Whether ``data`` holds no elements.
`shape` | <code>tuple[int, ...]</code> | Images-by-voxels shape of ``data``.
`size` | <code>int</code> | Total number of elements in ``data`` (numpy convention).

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
[`copy`](#data-brain-data-copy) | Create an independent snapshot of a BrainData instance.
[`create_empty`](#data-brain-data-create-empty) | Create a copy of BrainData with empty data array.
[`decompose`](#data-brain-data-decompose) | Decompose BrainData object.
[`detrend`](#data-brain-data-detrend) | Remove linear trend from each voxel.
[`distance`](#data-brain-data-distance) | Calculate distance between images within a BrainData() instance.
[`extract_roi`](#data-brain-data-extract-roi) | Extract activity from mask or ROI atlas using NiftiLabelsMasker.
[`filter`](#data-brain-data-filter) | Apply a Butterworth filter to data (wraps `nilearn.signal.clean`).
[`find_spikes`](#data-brain-data-find-spikes) | Identify spikes from Time Series Data.
[`fit`](#data-brain-data-fit) | Fit a model to brain imaging data.
[`iplot`](#data-brain-data-iplot) | Interactive WebGL brain viewer powered by niivue.
[`mean`](#data-brain-data-mean) | Get mean of each voxel or image.
[`median`](#data-brain-data-median) | Get median of each voxel or image.
[`multivariate_similarity`](#data-brain-data-multivariate-similarity) | Predict a BrainData spatial distribution from a linear combination.
[`plot`](#data-brain-data-plot) | Plot BrainData instance using nilearn visualization or matplotlib.
[`plot_flatmap`](#data-brain-data-plot-flatmap) | Plot brain data on cortical flatmap.
[`plot_surf`](#data-brain-data-plot-surf) | Render this BrainData on fsaverage surfaces as a tight 2×2 montage.
[`predict`](#data-brain-data-predict) | Predict voxel timeseries (encoding) or decode labels (MVPA).
[`r_to_z`](#data-brain-data-r-to-z) | Apply Fisher's r-to-z transformation to each data element.
[`regions`](#data-brain-data-regions) | Extract brain connected regions into separate regions.
[`report`](#data-brain-data-report) | Generate a nilearn HTML report for a fitted GLM.
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

## Methods

(data-brain-data-align)=
### `align`

```python
align(target, method = 'procrustes', axis = 0, *, spatial_scale: str = 'whole_brain', roi_mask: str = None, radius_mm: float = 10.0)
```

Align BrainData instance to target object using functional alignment.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`target` | <code>[BrainData](#page-data-brain-data)</code> | Object to align to. | *required*
`method` | <code>str</code> | Alignment method: ``'probabilistic_srm'``, ``'deterministic_srm'``, or ``'procrustes'``. Default ``'procrustes'``. | <code>'procrustes'</code>
`axis` | <code>int</code> | Axis to align on. Default 0. | <code>0</code>
`spatial_scale` | <code>str</code> | ``'whole_brain'`` (default), ``'roi'``, or ``'searchlight'``. ``'roi'`` is supported (per-parcel transforms + reassembly, requires `roi_mask`). ``'searchlight'`` is not yet implemented (overlapping spheres have no canonical per-voxel transform). | <code>'whole_brain'</code>
`roi_mask` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image \| str \| Path \| None</code> | Atlas image used when ``spatial_scale='roi'``. | <code>None</code>
`radius_mm` | <code>float</code> | Reserved for ``spatial_scale='searchlight'``. | <code>10.0</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | A dictionary containing the transformed object, transformation     matrix, and the shared response matrix.

**Examples:**

```python
# Hyperalign using procrustes transform
out = data.align(target, method='procrustes')

# Align using shared response model
out = data.align(target, method='probabilistic_srm')
```

(data-brain-data-append)=
### `append`

```python
append(data, ignore_attrs = False, **kwargs)
```

Append data to BrainData instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[BrainData](#page-data-brain-data)</code> | BrainData instance to append. | *required*
`ignore_attrs` | <code>bool</code> | Clear both X and Y on the result when True. Otherwise, each metadata frame must be empty on both inputs or have compatible columns on both inputs. Default False. | <code>False</code>
`**kwargs` | <code>dict</code> | Currently ignored. X/Y are concatenated with polars' ``pl.concat(..., how="vertical_relaxed")``, which takes no caller-supplied options. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Independently owned data with concatenated row metadata.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | Metadata is present on only one input or has incompatible columns.

(data-brain-data-apply-mask)=
### `apply_mask`

```python
apply_mask(mask, resample_mask_to_brain = False)
```

Mask BrainData instance using nilearn functionality.

Note target data will be resampled into the same space as the mask. If you would like the mask
resampled into the BrainData space, then set resample_mask_to_brain=True.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image</code> | Mask to apply to BrainData object. | *required*
`resample_mask_to_brain` | <code>bool</code> | Resample the mask to brain space before applying it. Default False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Masked BrainData object.

(data-brain-data-astype)=
### `astype`

```python
astype(dtype)
```

Cast BrainData.data as type.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dtype` | <code>dtype \| type \| str</code> | Datatype to convert to. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | BrainData instance with new datatype.

(data-brain-data-bootstrap)=
### `bootstrap`

```python
bootstrap(stat, *, n_samples = 5000, save_boots = False, percentiles = (2.5, 97.5), X_test = None, device = 'cpu', max_gpu_memory_gb = None, tail = 2, n_jobs = -1, random_state = None, progress_bar: bool = False)
```

Bootstrap statistics using efficient online algorithms.

Uses memory-efficient bootstrap infrastructure with CPU parallelization or GPU acceleration.
Supports simple aggregation statistics and fitted model statistics (Ridge).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stat` | <code>str</code> | Statistic to bootstrap. Simple stats: ``'mean'``, ``'median'``, ``'std'``, ``'sum'``, ``'min'``, ``'max'``. Model stats: ``'weights'`` (requires a fitted Ridge model) or ``'predict'`` (requires a fitted Ridge model plus ``X_test``). | *required*
`n_samples` | <code>int</code> | Number of bootstrap iterations. Default 5000. | <code>5000</code>
`save_boots` | <code>bool</code> | If True, store all bootstrap samples. Default False. | <code>False</code>
`percentiles` | <code>tuple[float, float]</code> | Percentiles for confidence intervals. Default ``(2.5, 97.5)``. | <code>(2.5, 97.5)</code>
`X_test` | <code>ndarray \| None</code> | Test features for the ``'predict'`` bootstrap. | <code>None</code>
`device` | <code>str</code> | Compute device for the Ridge bootstrap: ``'cpu'`` (default), ``'gpu'`` (PyTorch on CUDA/MPS if available), or ``'auto'`` (GPU if present, else CPU). Ignored for simple stats. | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>float \| None</code> | Explicit GPU memory budget in GB when device is ``'gpu'`` or ``'auto'``. ``None`` (default) measures the device. | <code>None</code>
`tail` | <code>int \| str</code> | ``2``/``'two'`` for two-tailed p-values (default), ``1``/``'one'`` for one-tailed. | <code>2</code>
`n_jobs` | <code>int</code> | Number of CPU cores for parallelization. -1 (default) means all CPUs. | <code>-1</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. Default False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data) \| dict</code> | For simple stats, a BrainData holding the bootstrap     mean. For model stats, a dict of BrainData objects keyed ``'mean'``,     ``'std'``, ``'Z'``, ``'p'``, ``'ci_lower'``, ``'ci_upper'``. With     ``save_boots=True`` the dict also carries a ``'samples'`` key     holding every bootstrap sample.

**Examples:**

```python
boot = brain.bootstrap(stat='mean', n_samples=1000)
brain.fit(X=dm, model='ridge', alpha=1.0)
boot = brain.bootstrap(stat='weights', n_samples=1000)
```

(data-brain-data-cluster-report)=
### `cluster_report`

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
`stat_threshold` | <code>float \| None</code> | Voxel-level threshold (e.g. z- or t-cutoff). ``None`` treats ``self`` as already thresholded. | <code>3.0</code>
`cluster_threshold` | <code>int</code> | Minimum cluster size in voxels. | <code>10</code>
`two_sided` | <code>bool</code> | Report negative clusters separately. | <code>True</code>
`min_distance` | <code>float</code> | Minimum mm between sub-peaks within a cluster. | <code>8.0</code>
`atlas` | <code>str \| Sequence[str] \| None</code> | Atlas name or list of names (see `list_atlases`). Defaults to ``("harvard_oxford", "aal", "schaefer_200")``. | <code>None</code>
`prob_threshold` | <code>float</code> | Drop probabilistic-atlas regions below this %. | <code>5.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[ClusterReport](#tasks-atlases-clusterreport)</code> | Report with `peaks` and `clusters` (polars DataFrames)     and `stat_img` (BrainData).

(data-brain-data-compute-contrasts)=
### `compute_contrasts`

```python
compute_contrasts(contrasts, statistic = 't')
```

Compute contrasts from fitted GLM results.

This method computes contrasts as linear combinations of the GLM beta coefficients.
Must be called after ``fit(model='glm', X=design_matrix)`` has been run.

A contrast can be given three ways. A **string** names design-matrix columns
with optional coefficients, e.g. ``"conditionA - conditionB"`` or
``"2*conditionA - conditionB - conditionC"``. A **numeric vector** lists one
weight per regressor, e.g. ``[1, -1, 0, 0]`` for a 4-regressor model. A
**dict** maps contrast names to either form, e.g.
``{"main_effect": "conditionA - conditionB", "interaction": [1, -1, -1, 1]}``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`contrasts` | <code>str \| array - like \| dict</code> | The contrast(s) to compute — a string, a numeric vector, or a dict of named contrasts (see above). | *required*
`statistic` | <code>str</code> | Which statistic to return per contrast. One of ``"t"`` (default, t-statistic map), ``"z"`` (z-score), ``"p"`` (p-value), ``"beta"`` / ``"effect_size"`` (effect-size β map — use this when feeding a second-level group analysis), or ``"all"`` (a bundle dict ``{"beta", "t", "z", "p", "se"}`` of maps for one contrast). | <code>'t'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data) \| dict</code> | A single contrast with a scalar ``statistic`` returns a     ``BrainData`` map; with ``statistic="all"`` it returns a flat dict keyed     by ``"beta"``/``"t"``/``"z"``/``"p"``/``"se"``. A dict of contrasts     returns a dict keyed by contrast name (nested under the five keys when     ``statistic="all"``).

**Raises:**

Type | Description
---- | -----------
<code>RuntimeError</code> | If ``fit(model='glm')`` hasn't been called yet.
<code>ValueError</code> | If a contrast vector's length doesn't match the number of regressors, or a column named in a string contrast is not in the design matrix.

**Examples:**

```python
brain.fit(model='glm', X=design_matrix)
contrast1 = brain.compute_contrasts([0, 1, -1])
contrast2 = brain.compute_contrasts("conditionA - conditionB")
results = brain.compute_contrasts({
    "A_vs_B": "conditionA - conditionB",
    "avg_effect": [0, 0.5, 0.5],
})
```

<details class="note" open markdown="1">
<summary>Note</summary>

String contrasts support coefficients (``"2*A - B"``, ``"0.5*A + 0.5*B"``).
Column names must match design-matrix columns exactly (case-sensitive).
Contrast weights should sum to zero for proper inference in most cases.

</details>

(data-brain-data-copy)=
### `copy`

```python
copy()
```

Create an independent snapshot of a BrainData instance.

Data, metadata, mask state, and any fitted model/results are copied.
Mutating either object after copying does not affect the other.
Python's `copy.copy()` and `copy.deepcopy()` have the same semantics.

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | An independent copy, including fitted state.

(data-brain-data-create-empty)=
### `create_empty`

```python
create_empty()
```

Create a copy of BrainData with empty data array.

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | A copy of this object with an empty data array.

(data-brain-data-decompose)=
### `decompose`

```python
decompose(*, method = 'pca', axis = 'voxels', n_components = None, **kwargs)
```

Decompose BrainData object.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>str</code> | Decomposition algorithm: ``'pca'``, ``'ica'``, ``'nnmf'``, ``'fa'``, ``'dictionary'``, or ``'kernelpca'``. Default ``'pca'``. | <code>'pca'</code>
`axis` | <code>str</code> | Dimension to decompose: ``'voxels'`` (default) or ``'images'``. | <code>'voxels'</code>
`n_components` | <code>int \| None</code> | Number of components. If ``None`` then retain as many as possible. | <code>None</code>
`**kwargs` | <code>dict</code> | Forwarded to the underlying sklearn decomposition estimator. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | A dictionary of decomposition parameters.

(data-brain-data-detrend)=
### `detrend`

```python
detrend(method = 'linear')
```

Remove linear trend from each voxel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>str</code> | Type of detrending: ``'linear'`` (default) or ``'constant'``. | <code>'linear'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Detrended BrainData instance.

(data-brain-data-distance)=
### `distance`

```python
distance(metric = 'euclidean', *, spatial_scale: str = 'whole_brain', roi_mask: str = None, radius_mm: float = 10.0, **kwargs: float)
```

Calculate distance between images within a BrainData() instance.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` | <code>str</code> | Distance metric — any ``scipy.spatial.distance`` metric supported by ``cdist``. Default ``'euclidean'``. | <code>'euclidean'</code>
`spatial_scale` | <code>str</code> | One of ``'whole_brain'`` (default), ``'roi'``, or ``'searchlight'``. ``'whole_brain'`` returns a single pairwise distance ``Adjacency`` between images. ``'roi'`` requires ``roi_mask`` and returns a stacked ``Adjacency`` with one RDM per sorted nonzero atlas label present inside the source mask after nearest-neighbor resampling. `'searchlight'` returns one RDM per source-mask voxel in mask order. | <code>'whole_brain'</code>
`roi_mask` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image \| str \| Path \| None</code> | Atlas image for ``spatial_scale='roi'``. | <code>None</code>
`radius_mm` | <code>float</code> | Searchlight radius in mm. Default 10.0. | <code>10.0</code>
`**kwargs` | <code>dict</code> | Additional metric options forwarded to ``scipy.spatial.distance.cdist`` (e.g. ``p`` for minkowski). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[Adjacency](#page-data-adjacency)</code> | Single pairwise distance matrix for ``'whole_brain'``;     ordinary stack for `'roi'` / `'searchlight'`. Map per-matrix values     externally using `roi_to_brain_from_atlas` with the aligned atlas     and sorted surviving ROI labels, or `nilearn.masking.unmask`     with the source mask for searchlights. Subset the mapping whenever     selecting matrices from the returned stack.

(data-brain-data-extract-roi)=
### `extract_roi`

```python
extract_roi(mask, method = 'mean', n_components = None)
```

Extract activity from mask or ROI atlas using NiftiLabelsMasker.

The mask may be binary (a single ROI) or a labeled atlas (one value per
region, extracting from every ROI at once).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mask` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image \| str \| Path</code> | Binary mask or labeled atlas to extract from. | *required*
`method` | <code>str</code> | Extraction method: ``'mean'`` (default), ``'median'``, or ``'pca'``. | <code>'mean'</code>
`n_components` | <code>int \| None</code> | Number of components to return when ``method='pca'``. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>float \| ndarray</code> | For a binary mask, a scalar (single image) or 1D     array (multiple images). For a labeled atlas, a 1D array (single     image), a 2D array of images x ROIs (multiple images), or the PCA     components array when ``method='pca'``.

**Examples:**

```python
roi_values = brain.extract_roi(binary_mask)
atlas_values = brain.extract_roi(atlas_mask)
components = brain.extract_roi(mask, method='pca', n_components=5)
```

(data-brain-data-filter)=
### `filter`

```python
filter(*, sampling_freq = None, high_pass = None, low_pass = None, **kwargs)
```

Apply a Butterworth filter to data (wraps `nilearn.signal.clean`).

<details class="note" open markdown="1">
<summary>Note</summary>

Unlike nilearn's default, does not detrend or standardize. Pass
detrend=True or standardize=True via kwargs to enable.

</details>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sampling_freq` | <code>float \| None</code> | Sampling frequency in hertz (i.e. 1 / TR). | <code>None</code>
`high_pass` | <code>float \| None</code> | High-pass cutoff frequency in hertz. | <code>None</code>
`low_pass` | <code>float \| None</code> | Low-pass cutoff frequency in hertz. | <code>None</code>
`**kwargs` | <code>dict</code> | Additional arguments passed to ``nilearn.signal.clean``. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Filtered BrainData instance.

(data-brain-data-find-spikes)=
### `find_spikes`

```python
find_spikes(global_spike_cutoff = 3, diff_spike_cutoff = 3, *, TR: float | None = None, sampling_freq: float | None = None)
```

Identify spikes from Time Series Data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`global_spike_cutoff` | <code>int or None</code> | cutoff to identify spikes in global signal in standard deviations, or None to skip. | <code>3</code>
`diff_spike_cutoff` | <code>int or None</code> | cutoff to identify spikes in average frame difference in standard deviations, or None to skip. | <code>3</code>
`TR` | <code>float \| None</code> | Repetition time in seconds. Sets the returned DesignMatrix's sampling_freq for downstream `.append(...)` / `.convolve()`. Pass exactly one of `TR` or `sampling_freq`. | <code>None</code>
`sampling_freq` | <code>float \| None</code> | Sampling frequency in Hz (= 1/TR). See `TR`. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#page-data-design-matrix)</code> | One indicator column per detected spike TR, with all     spike columns pre-marked as confounds. A TR flagged by both     detectors yields a single column (named `global_spike*`); the     colliding detections are bitwise identical, so only the retained     name differs.

(data-brain-data-fit)=
### `fit`

```python
fit(model = 'glm', *, X = None, cv = None, device = 'cpu', local_alpha = True, fit_intercept = False, inplace = True, scale = 'auto', standardize = 'auto', progress_bar = False, **kwargs)
```

Fit a model to brain imaging data.

Creates and fits a model from string specification. The brain data
(self.data) is always used as the target variable. Model and results
are stored for later use with predict().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model` | <code>str</code> | Model type: 'ridge', 'glm', or future model names | <code>'glm'</code>
`X` | <code>array - like or DataFrame</code> | Design matrix or feature matrix | <code>None</code>
`cv` | <code>int or sklearn CV splitter</code> | Cross-validation specification (Ridge only). int → ``KFold(cv)``; pass a splitter object (e.g. ``KFold(5, shuffle=True)``, ``GroupKFold(8)``) for non-contiguous folds. Generators (``splitter.split(X)``) are rejected. | <code>None</code>
`device` | <code>str, default='cpu'</code> | Ridge only. Compute device for the ridge solve/CV: ``'cpu'`` (NumPy), ``'gpu'`` (PyTorch on CUDA/MPS when available), or ``'auto'`` (GPU if present, else CPU). Ignored when ``model='glm'``. | <code>'cpu'</code>
`local_alpha` | <code>bool, default=True</code> | Ridge only. If True, select α independently per voxel via ``solve_ridge_cv``. If False, pick a single α shared across all voxels. | <code>True</code>
`fit_intercept` | <code>bool, default=False</code> | Ridge only. Forwarded to the Ridge model — center X and y on the training fold mean per fold and recover the intercept after. | <code>False</code>
`inplace` | <code>bool, default=True</code> | If True, mutate self and return self. If False, fit and return an independent `BrainData` copy while leaving every part of self untouched. | <code>True</code>
`scale` | <code>bool or 'auto', default='auto'</code> | Apply percent-signal-change scaling before fitting via nilearn's per-voxel ``mean_scaling``. ``'auto'`` → False for both models (PSC is opt-in). Redundant with ``standardize='zscore'`` (warns). Applied before ``standardize``. | <code>'auto'</code>
`standardize` | <code>str or None or 'auto', default='auto'</code> | Standardize each voxel across observations after scaling. ``'center'``, ``'zscore'``, or ``None``. ``'auto'`` → ``'zscore'`` for ridge, ``None`` for glm. | <code>'auto'</code>
`progress_bar` | <code>bool</code> | Display a progress bar during fitting. Default: False. | <code>False</code>
`**kwargs` | <code>dict</code> | Additional arguments passed to the model constructor (e.g. ``alpha`` for ridge). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Self when ``inplace=True``; otherwise an independently     owned fitted copy.

<details class="note" open markdown="1">
<summary>Note</summary>

After ``model="glm"``, the following per-regressor BrainData
attributes are populated — one map per design-matrix column:
``glm_betas`` (effect-size β maps), ``glm_t`` (marginal t-statistic for
each regressor), ``glm_p`` (marginal p-value), ``glm_se`` (standard
error of β), and ``glm_r2`` (voxel-wise R²).

``glm_t[i]`` is a valid t-map for the trivial one-hot contrast on
regressor ``i`` only. For contrasts across regressors
(``"A - B"``, ``[1, -1, 0, ...]``) use `compute_contrasts` —
you cannot correctly combine these per-regressor maps by hand
because t-statistic arithmetic requires the off-diagonal elements
of the parameter covariance matrix, which are not stored. Pass
``statistic="all"`` to get ``β``/``t``/``z``/``p``/``se`` for
one contrast in a single call.

</details>

**Examples:**

```python
brain_data.fit(model='ridge', alpha=1.0, cv=5, X=features)
fit = brain_data.fit(model='ridge', alpha=1.0, X=features, inplace=False)
```

(data-brain-data-iplot)=
### `iplot`

```python
iplot(*, view: str = 'ortho', threshold: float | str | None = None, lower: float | str | None = None, upper: float | str | None = None, autoscale: bool = True, symmetric: bool | Literal['auto'] = 'auto', cmap: str | None = None, bg_img: str | bool | None = None, atlas: str | Atlas | None = None, opacity: float = 1.0, outline: float = 0.0, colorbar: bool = True, controls: bool = True, **kwargs: bool)
```

Interactive WebGL brain viewer powered by niivue.

Renders inline in a live kernel (Jupyter, marimo) with
live windowing (right-drag to set the threshold/contrast), slice
scrolling, native 4D frame scrubbing, true 3D rendering, a stat-map
colorbar, and optional nltools-atlas overlays. Static-built docs (plain
Markdown) are not interactive; use `plot` there.

Returns a `NiivueViewer` widget. By default (``controls=True``) it
renders an in-widget threshold slider above the viewer; the window is
reactive through the ``cal_min`` / ``cal_max`` traits. Pass
``controls=False`` to hide the slider (right-drag windowing still
works).

Thresholding uses positive and negative display limbs. ``cal_min`` is
the magnitude floor and ``cal_max`` the positive saturation point;
niivue receives the negative endpoints explicitly. By default, mixed
maps use symmetric limbs while each sign in a one-sided map determines
its own ceiling. The window is computed in Python, and the two controls
show the shared floor and positive-limb ceiling.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`view` | <code>str</code> | ``"ortho"`` (default), ``"axial"``, ``"coronal"``, ``"sagittal"``, or ``"render"`` (3D volume render). ``"surface"`` is no longer supported — use ``"render"`` or `plot_flatmap` / `plot_surf`. | <code>'ortho'</code>
`threshold` | <code>float \| str \| None</code> | Convenience symmetric magnitude floor (→ ``cal_min``). Accepts a percentile string (``"95%"``) resolved over the finite nonzero magnitudes, consistent with `threshold`. | <code>None</code>
`lower` | <code>float \| str \| None</code> | Window floor (→ ``cal_min``). Overrides ``threshold``. Accepts a percentile string. | <code>None</code>
`upper` | <code>float \| str \| None</code> | Window ceiling (→ ``cal_max``). Overrides ``threshold``. Accepts a percentile string. | <code>None</code>
`autoscale` | <code>bool</code> | Robust default window for the edges not set above. ``True`` (default): ceiling at the 98th percentile of the finite nonzero magnitudes — a couple of outlier voxels no longer wash out the whole map — and an epsilon floor, never above the smallest nonzero magnitude, so zeros render transparent and every real voxel stays visible (threshold up from there). ``False``: the raw magnitude range from zero to the largest absolute value. For a custom percentile window pass ``lower``/``upper`` (e.g. ``lower="60%", upper="98%"``). | <code>True</code>
`symmetric` | <code>bool \| Literal['auto']</code> | ``"auto"`` (default) mirrors mixed-signed maps but lets each sign in a one-sided map determine its own ceiling. ``True`` always mirrors; ``False`` scales positive and negative limbs independently. | <code>'auto'</code>
`cmap` | <code>str \| None</code> | niivue colormap for the positive limb. The default uses niivue's red positive and blue negative palettes. Common matplotlib names are auto-mapped with a warning. | <code>None</code>
`bg_img` | <code>str \| bool \| None</code> | ``None``/``True`` auto-loads the matching MNI template when the data is in standard space (else none); ``False`` disables the background; a path string uses that image. | <code>None</code>
`atlas` | <code>str \| [Atlas](#tasks-atlases-atlas) \| None</code> | Atlas overlay — a registry name (e.g. ``"aal"``), a loaded `Atlas`, or ``None``. Deterministic atlases only; probabilistic atlases raise. | <code>None</code>
`opacity` | <code>float</code> | Stat-map (and filled-atlas) opacity in ``0..1``. | <code>1.0</code>
`outline` | <code>float</code> | ``> 0`` draws atlas region boundaries of that width (stat map stays visible); ``0`` draws filled regions. | <code>0.0</code>
`colorbar` | <code>bool</code> | Show the stat-map colorbar (default ``True``). An explicit ``is_colorbar`` kwarg overrides this. | <code>True</code>
`controls` | <code>bool</code> | Render an in-widget threshold slider above the viewer (default ``True``). ``False`` hides it; the viewer still supports niivue's right-drag windowing. No extra dependency either way — the slider is native to the widget frontend. | <code>True</code>
`**kwargs` | <code>dict</code> | Passed as niivue options. ``height`` configures the canvas and ``is_colorbar`` overrides ``colorbar``. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>NiivueViewer</code> | An `anywidget.AnyWidget` whose threshold window is     reactive via the `cal_min` and `cal_max` traits.

**Raises:**

Type | Description
---- | -----------
<code>TypeError</code> | If ``autoscale`` is not a bool or ``symmetric`` is not ``True``, ``False``, or ``"auto"``.

(data-brain-data-mean)=
### `mean`

```python
mean(axis = 0, *, spatial_scale: str = 'whole_brain', roi_mask: str = None)
```

Get mean of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>int</code> | 0 = across images (default, returns BrainData), 1 = within images (returns array). Ignored when ``spatial_scale='roi'``. | <code>0</code>
`spatial_scale` | <code>str</code> | ``'whole_brain'`` (default) reduces along ``axis``. ``'roi'`` requires ``roi_mask`` and returns a BrainData of the same shape with each voxel painted with its parcel's mean per image (parcellation smoothing). | <code>'whole_brain'</code>
`roi_mask` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image \| str \| Path \| None</code> | Atlas image for ``spatial_scale='roi'``. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>float \| ndarray \| [BrainData](#page-data-brain-data)</code> | Mean values.

(data-brain-data-median)=
### `median`

```python
median(axis = 0, *, spatial_scale: str = 'whole_brain', roi_mask: str = None)
```

Get median of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>int</code> | 0 = across images (default, returns BrainData), 1 = within images (returns array). Ignored when ``spatial_scale='roi'``. | <code>0</code>
`spatial_scale` | <code>str</code> | ``'whole_brain'`` (default) or ``'roi'`` (paints each voxel with its parcel's median per image). | <code>'whole_brain'</code>
`roi_mask` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image \| str \| Path \| None</code> | Atlas image for ``spatial_scale='roi'``. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>float \| ndarray \| [BrainData](#page-data-brain-data)</code> | Median values.

(data-brain-data-multivariate-similarity)=
### `multivariate_similarity`

```python
multivariate_similarity(images, method = 'ols', tail = 2)
```

Predict a BrainData spatial distribution from a linear combination.

The predictors may be other BrainData instances or nibabel images.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`images` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image \| list</code> | Predictor image(s) — a BrainData stack of weight maps or nibabel images. | *required*
`method` | <code>str</code> | Regression method. Default: 'ols'. | <code>'ols'</code>
`tail` | <code>int \| str</code> | ``2`` or ``'two'`` for two-tailed (default); ``1`` or ``'one'`` for one-tailed (positive direction) regression p-values. | <code>2</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Regression statistics as BrainData instances, keyed     `'beta'`, `'t'`, `'p'`, `'df'`, `'residual'`.

(data-brain-data-plot)=
### `plot`

```python
plot(*, method = 'glass', upper = None, lower = None, threshold = None, view = 'z', cut_coords = None, cmap = None, bg_img = None, ax = None, figsize = (8, 6), title = None, colorbar = True, save = None, stat = 'mean', limit = 3, **kwargs)
```

Plot BrainData instance using nilearn visualization or matplotlib.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>str</code> | Visualization type: 'glass', 'slices', 'timeseries', 'histogram' | <code>'glass'</code>
`upper` | <code>str / float</code> | Upper threshold. | <code>None</code>
`lower` | <code>str / float</code> | Lower threshold. | <code>None</code>
`threshold` | <code>float \| str</code> | Absolute transparency cutoff. Percentile strings resolve over finite, nonzero magnitudes. | <code>None</code>
`view` | <code>str</code> | For ``method="slices"``, any non-empty combination of ``"x"``, ``"y"``, ``"z"`` (e.g. ``"xyz"``, ``"xz"``, ``"y"``). Default: ``"z"``. | <code>'z'</code>
`cut_coords` | <code>list or dict</code> | Cut coordinates for multi-slice views. Takes precedence over ``view``-based defaults. Either a list matching ``len(view)`` or a dict keyed by axis letter. | <code>None</code>
`cmap` | <code>str</code> | Colormap name. Defaults are sign-aware. | <code>None</code>
`bg_img` | <code>str/nibabel image</code> | Background image. | <code>None</code>
`ax` | <code>Axes</code> | Matplotlib axis. | <code>None</code>
`figsize` | <code>tuple</code> | default figure size if no axis (8, 6) | <code>(8, 6)</code>
`title` | <code>str</code> | Plot title. | <code>None</code>
`colorbar` | <code>bool</code> | Whether to show colorbar. Default: True. | <code>True</code>
`save` | <code>str</code> | Path to save figure(s). | <code>None</code>
`stat` | <code>str</code> | Statistic for timeseries plots. Default: 'mean'. | <code>'mean'</code>
`limit` | <code>int</code> | Maximum number of images to render when this BrainData contains multiple maps and ``method`` is ``"glass"`` or ``"slices"``. Default: 3. Warns when more images exist than ``limit``. | <code>3</code>
`**kwargs` | <code>dict</code> | Additional arguments passed to nilearn plot functions. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>Figure \| list[Figure]</code> | A single     figure for single-image data; a list of figures for multi-image     data with `method` in `{"glass", "slices"}` (one per image for     glass; one per image-and-view pair for slices).

(data-brain-data-plot-flatmap)=
### `plot_flatmap`

```python
plot_flatmap(*, threshold = None, cmap = None, vmax = None, vmin = None, template = 'fsaverage5', with_curvature = True, curvature_contrast = 0.5, curvature_brightness = 0.5, transparency = 'auto', colorbar = True, colorbar_orientation = 'horizontal', figsize = (12, 6), title = None, radius_mm = 3.0, interpolation = 'linear', axes = None, save = None)
```

Plot brain data on cortical flatmap.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`threshold` | <code>float \| str</code> | Absolute cutoff or percentile string. | <code>None</code>
`cmap` | <code>str</code> | Matplotlib colormap. Defaults are sign-aware. | <code>None</code>
`vmax` | <code>float</code> | Maximum value; inferred from displayed data. | <code>None</code>
`vmin` | <code>float</code> | Minimum value; inferred from displayed data. | <code>None</code>
`template` | <code>str</code> | Freesurfer surface resolution. Default: 'fsaverage5'. | <code>'fsaverage5'</code>
`with_curvature` | <code>bool</code> | Show sulcal/gyral pattern. Default: True. | <code>True</code>
`curvature_contrast` | <code>float</code> | Contrast of curvature overlay. Default: 0.5. | <code>0.5</code>
`curvature_brightness` | <code>float</code> | Mean brightness of curvature overlay. Default: 0.5. | <code>0.5</code>
`transparency` | <code>BrainData, Nifti1Image, str, or "auto"</code> | Binary mask used to render vertices outside the mask as transparent. ``"auto"`` (default) uses the instance's ``.mask``; pass ``None`` to disable masking. | <code>'auto'</code>
`colorbar` | <code>bool</code> | Show colorbar. Default: True. | <code>True</code>
`colorbar_orientation` | <code>str</code> | 'horizontal' or 'vertical'. Default: 'horizontal'. | <code>'horizontal'</code>
`figsize` | <code>tuple</code> | Figure size as (width, height). Default: (12, 6). | <code>(12, 6)</code>
`title` | <code>str</code> | Figure title. | <code>None</code>
`radius_mm` | <code>float</code> | Sampling radius in mm. Default: 3.0. | <code>3.0</code>
`interpolation` | <code>str</code> | Interpolation method. Default: 'linear'. | <code>'linear'</code>
`axes` | <code>Axes</code> | Existing axes to plot on. | <code>None</code>
`save` | <code>str</code> | File path to save figure. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>Figure</code> | The rendered figure.

(data-brain-data-plot-surf)=
### `plot_surf`

```python
plot_surf(*, hemi = 'both', view = 'montage', surface = 'pial', template = 'fsaverage5', threshold = None, cmap = None, vmin = None, vmax = None, transparency = 'auto', bg_on_data = False, colorbar = True, colorbar_orientation = 'horizontal', figsize = (10, 8), title = None, radius_mm = 3.0, interpolation = 'linear', zoom = 1.2, axes = None, save = None)
```

Render this BrainData on fsaverage surfaces as a tight 2×2 montage.

Facade over `plot_surf`. See that function's
docstring for the full argument reference. Notable defaults:
``surface="pial"``, ``zoom=1.2``, ``transparency="auto"`` (uses
this instance's ``.mask``).

**Returns:**

Type | Description
---- | -----------
<code>Figure</code> | The rendered figure.

(data-brain-data-predict)=
### `predict`

```python
predict(*, y: np.ndarray | str | None = None, X: np.ndarray | None = None, spatial_scale: str = 'whole_brain', model: str = 'svm', cv: int | str = 5, standardize: bool = True, reduce: str | None = None, n_components: int | None = None, scoring: str = 'auto', groups: np.ndarray | str | None = None, roi_mask: np.ndarray | str | None = None, radius_mm: float = 10.0, inplace: bool = False, n_jobs: int = 1, random_state: int | None = None, progress_bar: bool = False)
```

Predict voxel timeseries (encoding) or decode labels (MVPA).

Dispatched by which of ``X`` or ``y`` is provided:

1. **Timeseries prediction** (``X`` provided): use a fitted ridge /
   GLM encoding model on ``self`` to predict voxel responses.
   Returns a fresh ``BrainData`` whose ``.data`` holds the predicted
   timeseries (composes directly with ``.plot()``, ``.standardize()``
   etc.). ``inplace`` has no effect in this mode.
2. **MVPA decoding** (``y`` provided, or resolvable from ``.Y``):
   train a classifier or regressor with cross-validation. Returns a
   `Predict` dataclass. Spatial fields (``weight_map``,
   ``fold_weight_maps``, ``final_weight_map``, ``accuracy_map``) are
   `BrainData` objects so ``result.weight_map.plot()`` works
   directly. Drop down to numpy via ``result.weight_map.data``.

Labels travel with the data: when ``y`` is omitted and this object
carries a single-column ``.Y`` frame, that column is decoded
(``y='name'`` picks a column of a multi-column ``.Y``; ``groups``
accepts a ``.Y`` column name the same way). An object with both a
fitted encoding model and a stored ``.Y`` refuses the no-argument
call as ambiguous — pass ``y=`` or ``X=`` explicitly.

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

**Choosing a model.** String shortcuts for classification are ``'svm'``
(LinearSVC), ``'logistic'``, ``'lda'``, and ``'ridge_classifier'``; for
regression, ``'ridge'``, ``'lasso'``, and ``'svr'``. Any sklearn estimator
or ``Pipeline`` is also accepted (e.g.
``make_pipeline(StandardScaler(), SelectKBest(k=500), LinearSVC())``).
When ``model`` is a sklearn ``Pipeline``, ``standardize`` is auto-defaulted
to ``False`` (with a warning) so we don't wrap another StandardScaler
around your pipeline; pass ``standardize=True`` explicitly to override.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` | <code>(array - like, str)</code> | Labels (classification) or continuous targets (regression), shape ``(n_samples,)``, or the name of a ``.Y`` column. Triggers MVPA mode; omitted, it falls back to a single-column ``.Y``. | <code>None</code>
`X` | <code>array - like</code> | Features for timeseries prediction, shape ``(n_samples, n_features)``. Triggers encoding mode. | <code>None</code>
`spatial_scale` | <code>str</code> | MVPA dispatch — ``'whole_brain'``, ``'searchlight'``, or ``'roi'``. | <code>'whole_brain'</code>
`model` | <code>str \| sklearn estimator</code> | Algorithm — a string shortcut (``'svm'``, ``'logistic'``, ``'lda'``, ``'ridge_classifier'``, ``'ridge'``, ``'lasso'``, ``'svr'``) or any sklearn estimator / Pipeline. Default ``'svm'``; see "Choosing a model" above. | <code>'svm'</code>
`cv` | <code>int, str, or sklearn CV splitter</code> | ``int`` → shuffled KFold (regression) or StratifiedKFold (classification), honoring ``groups`` via the Group variants; ``'loo'`` (leave-one-out); ``'logo'`` (leave-one-group-out — pass the grouping variable via ``groups``, e.g. runs for leave-one-run-out); or any sklearn splitter. | <code>5</code>
`standardize` | <code>bool</code> | Z-score features per fold before fitting. Default ``True``. Auto-flipped to ``False`` when ``model`` is a sklearn ``Pipeline`` (see ``model`` above). | <code>True</code>
`reduce` | <code>str</code> | Per-fold dimensionality reduction. Currently only ``'pca'`` supported. Default ``None``. Weight maps are back-projected through PCA to voxel space. | <code>None</code>
`n_components` | <code>int</code> | PCA components when ``reduce='pca'``. | <code>None</code>
`scoring` | <code>str</code> | Sklearn scoring string. Default ``'auto'`` → ``'accuracy'`` if classifier, ``'r2'`` if regressor. | <code>'auto'</code>
`groups` | <code>(array - like, str)</code> | Group labels for CV splitters that need them (e.g., leave-one-run-out), or the name of a ``.Y`` column holding them. | <code>None</code>
`roi_mask` | <code>Nifti1Image or path - like</code> | Atlas image for ``spatial_scale='roi'``. | <code>None</code>
`radius_mm` | <code>float</code> | Searchlight radius in mm. Default ``10.0``. | <code>10.0</code>
`inplace` | <code>bool</code> | If ``True``, populate result fields as ``predict_*`` attributes on ``self`` and return ``self``. Default ``False`` returns a fresh `Predict`. | <code>False</code>
`n_jobs` | <code>int</code> | Parallel jobs for searchlight / ROI. Default ``1``; searchlight on a real brain at higher ``n_jobs`` can be memory-heavy. | <code>1</code>
`random_state` | <code>int</code> | Seed for the shuffled fold splitter when ``cv`` is an int (MVPA mode). Default ``None`` (unseeded shuffle each call). Ignored when ``cv`` is a splitter object — set its own ``random_state`` instead. | <code>None</code>
`progress_bar` | <code>bool</code> | Show progress bar for searchlight / ROI. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[Predict](#data-results-predict) \| [BrainData](#page-data-brain-data)</code> | ``Predict`` dataclass when ``inplace=False``;     ``self`` (mutated, with ``predict_*`` attrs) when ``inplace=True``.

**Examples:**

Whole-brain decoding:

```python
result = brain.predict(y=labels, spatial_scale='whole_brain', cv=5)
result.weight_map.plot()       # publishable map (all-data fit)
result.mean_score              # honest CV-derived accuracy
new_pred = result.estimator.predict(new_X)  # apply to new data
```

Searchlight and ROI decoding:

```python
result = brain.predict(y=labels, spatial_scale='searchlight',
                       radius_mm=8.0, n_jobs=4)
result.accuracy_map.plot()

result = brain.predict(y=labels, spatial_scale='roi', roi_mask=atlas)
top = result.roi_labels[result.mean_score.argsort()[::-1][:10]]
result.accuracy_map.plot()  # brain-space view of the same map
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
### `r_to_z`

```python
r_to_z()
```

Apply Fisher's r-to-z transformation to each data element.

(data-brain-data-regions)=
### `regions`

```python
regions(*, min_region_size = 1350, method = 'local_regions', smoothing_fwhm = 6, is_mask = False)
```

Extract brain connected regions into separate regions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`min_region_size` | <code>int</code> | Minimum volume in mm3 for a region to be kept. | <code>1350</code>
`method` | <code>str</code> | Type of extraction method                 ['connected_components', 'local_regions']. | <code>'local_regions'</code>
`smoothing_fwhm` | <code>scalar</code> | Smooth an image to extract more sparser regions. | <code>6</code>
`is_mask` | <code>bool</code> | Whether to treat as boolean mask. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | BrainData instance with extracted ROIs as data.

(data-brain-data-report)=
### `report`

```python
report(contrasts = None, **kwargs)
```

Generate a nilearn HTML report for a fitted GLM.

Must be called after ``fit(model='glm', ...)``. Renders the design
matrix, requested contrast maps, and model parameters as a
self-contained HTML report.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`contrasts` | <code>str \| list \| dict \| None</code> | Contrast(s) to render, same forms as `compute_contrasts`. | <code>None</code>
`**kwargs` | <code>dict</code> | Forwarded to nilearn's ``generate_report`` (e.g. ``title``, ``threshold``, ``alpha``). | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>HTMLReport</code> | nilearn report; call ``.save_as_html(path)`` or display     it in a notebook.

**Raises:**

Type | Description
---- | -----------
<code>RuntimeError</code> | If a GLM has not been fit yet.

**Examples:**

```python
brain.fit(model='glm', X=design_matrix)
brain.report(contrasts='conditionA - conditionB').save_as_html('report.html')
```

(data-brain-data-resample-to)=
### `resample_to`

```python
resample_to(*, img = None, resolution = None, interpolation = None)
```

Resample BrainData to match target image or resolution.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`img` | <code>Nifti1Image \| str \| Path \| None</code> | Target image for resampling. | <code>None</code>
`resolution` | <code>float \| int \| None</code> | Target isotropic voxel size in mm. | <code>None</code>
`interpolation` | <code>str \| None</code> | Interpolation method: ``'nearest'``, ``'linear'``, ``'continuous'``, or ``None`` to use the instance's setting. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | New BrainData instance with resampled data.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If both ``img`` and ``resolution`` are None, or both are provided.

(data-brain-data-scale)=
### `scale`

```python
scale(scale_val = 100.0, axis = None)
```

Scale data via mean scaling.

Two scaling modes are available. **Grand-mean scaling** (``axis=None``,
default) divides all values by the global mean across all voxels and
timepoints. **Voxel-wise scaling** (``axis=0``) divides each voxel's
time-series by its own temporal mean.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`scale_val` | <code>int \| float</code> | Target value for the mean after scaling. Default 100. | <code>100.0</code>
`axis` | <code>int \| None</code> | ``None`` for grand-mean scaling (default), ``0`` for voxel-wise scaling. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | New BrainData instance with scaled data.

(data-brain-data-similarity)=
### `similarity`

```python
similarity(image, metric = 'correlation')
```

Calculate similarity to a single BrainData or nibabel image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`image` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image</code> | Image to evaluate similarity against. | *required*
`metric` | <code>str</code> | Type of similarity: ``'correlation'`` (default), ``'pearson'``, ``'rank_correlation'``, ``'spearman'``, ``'dot_product'``, or ``'cosine'``. | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
<code>float or ndarray</code> | Similarity value(s).

(data-brain-data-smooth)=
### `smooth`

```python
smooth(fwhm)
```

Apply spatial smoothing using nilearn smooth_img().

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fwhm` | <code>float</code> | Full width at half maximum of the Gaussian spatial filter, in mm. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Copy with smoothed data.

(data-brain-data-standardize)=
### `standardize`

```python
standardize(*, axis = 0, method = 'center')
```

Standardize BrainData() instance.

Constant voxels (or observations) z-score to 0 rather than NaN.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>int</code> | 0 standardizes each voxel across observations (default). 1 standardizes each observation across voxels. | <code>0</code>
`method` | <code>str</code> | 'center' subtracts the mean (default). 'zscore' subtracts the mean and divides by standard deviation. | <code>'center'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Standardized BrainData instance.

(data-brain-data-std)=
### `std`

```python
std(axis = 0, *, spatial_scale: str = 'whole_brain', roi_mask: str = None)
```

Get standard deviation of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>int</code> | 0 = across images (default, returns BrainData), 1 = within images (returns array). Ignored when ``spatial_scale='roi'``. | <code>0</code>
`spatial_scale` | <code>str</code> | ``'whole_brain'`` (default) or ``'roi'`` (paints each voxel with its parcel's std per image). | <code>'whole_brain'</code>
`roi_mask` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image \| str \| Path \| None</code> | Atlas image for ``spatial_scale='roi'``. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>float \| ndarray \| [BrainData](#page-data-brain-data)</code> | Standard deviation values.

(data-brain-data-sum)=
### `sum`

```python
sum(axis = 0)
```

Get sum of each voxel or image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`axis` | <code>int</code> | 0 = across images (default, returns BrainData), 1 = within images (returns array). | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>float \| ndarray \| [BrainData](#page-data-brain-data)</code> | Sum values.

(data-brain-data-temporal-resample)=
### `temporal_resample`

```python
temporal_resample(*, sampling_freq = None, target = None, target_type = 'hz')
```

Resample BrainData timeseries to a new target frequency or number of samples.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sampling_freq` | <code>float \| None</code> | Sampling frequency of the data in hertz. | <code>None</code>
`target` | <code>float \| None</code> | Resampling target, interpreted per ``target_type``. | <code>None</code>
`target_type` | <code>str</code> | How to read ``target``: ``'hz'`` (default), ``'samples'``, or ``'seconds'``. | <code>'hz'</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Resampled BrainData instance.

(data-brain-data-threshold)=
### `threshold`

```python
threshold(*, upper = None, lower = None, binarize = False, coerce_nan = True, cluster_threshold = 0)
```

Threshold BrainData instance with optional cluster filtering.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`upper` | <code>float \| str \| None</code> | Upper cutoff for thresholding; a percentile string like ``'95%'`` is accepted. | <code>None</code>
`lower` | <code>float \| str \| None</code> | Lower cutoff for thresholding; a percentile string is accepted. | <code>None</code>
`binarize` | <code>bool</code> | Return a binarized image. Default False. | <code>False</code>
`coerce_nan` | <code>bool</code> | Coerce NaN values to 0s. Default True. | <code>True</code>
`cluster_threshold` | <code>int</code> | Minimum cluster size in voxels. Default 0. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Thresholded BrainData object.

(data-brain-data-to-nifti)=
### `to_nifti`

```python
to_nifti()
```

Convert BrainData Instance into Nifti Object.

**Returns:**

Type | Description
---- | -----------
<code>Nifti1Image</code> | Brain data as a NIfTI image.

(data-brain-data-transform-pairwise)=
### `transform_pairwise`

```python
transform_pairwise()
```

Transform data into pairwise comparisons.

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | BrainData instance transformed into pairwise comparisons

(data-brain-data-ttest)=
### `ttest`

```python
ttest(*, popmean = 0.0, permutation = False, n_permute = 5000, tail = 2, return_null = False, n_jobs = -1, random_state = None)
```

One-sample voxelwise t-test across images (axis 0).

Tests whether the per-voxel mean across images differs from
``popmean``. Operates on a stack of images (e.g. subject-level
contrast maps) with shape ``(n_samples, n_voxels)``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`popmean` | <code>float</code> | Population mean to test against. Default 0.0. | <code>0.0</code>
`permutation` | <code>bool</code> | If True, use a sign-flip permutation test via `one_sample_permutation_test`. Default False. | <code>False</code>
`n_permute` | <code>int</code> | Number of permutations (used only when ``permutation=True``). Default 5000. | <code>5000</code>
`tail` | <code>int \| str</code> | ``2`` or ``'two'`` for two-tailed (default); ``1`` or ``'one'`` for one-tailed (positive direction). | <code>2</code>
`return_null` | <code>bool</code> | If True, also return the null distribution. Default False. | <code>False</code>
`n_jobs` | <code>int</code> | Number of parallel jobs. Default -1 (all cores). | <code>-1</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>dict[str, [BrainData](#page-data-brain-data)]</code> | Four keys. ``"mean"`` is the voxelwise mean across     images (effect size); ``"t"`` the parametric one-sample t-statistic;     ``"z"`` the signed z-score, ``sign(t) * norm.isf(p/2)``, matching     nilearn's ``output_type='z_score'``; ``"p"`` the parametric p-value,     or empirical p when ``permutation=True``. The effect size is always     returned alongside the inferential maps so group-level code never     has to recompute the mean.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If this BrainData contains fewer than 2 images.

**Examples:**

```python
# Stack of subject-level contrast maps
result = contrast_maps.ttest()
sig = result["p"].data < 0.05
effect = result["mean"]       # for reporting magnitude
z_map = result["z"]           # for nilearn-style thresholding

# Permutation-based p-values; still reports t/z/mean
result = contrast_maps.ttest(permutation=True, n_permute=5000)
```

(data-brain-data-ttest2)=
### `ttest2`

```python
ttest2(other, equal_var = True, tail = 2)
```

Two-sample voxelwise t-test between two BrainData stacks.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`other` | <code>[BrainData](#page-data-brain-data)</code> | BrainData to compare against. Must have the same number of voxels. | *required*
`equal_var` | <code>bool</code> | If True (default), standard two-sample t-test. If False, Welch's t-test. | <code>True</code>
`tail` | <code>int \| str</code> | ``2`` or ``'two'`` for two-tailed (default); ``1`` or ``'one'`` for one-tailed (self > other; swap the operands for the other direction). | <code>2</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | ``{"t": BrainData, "p": BrainData}``.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If the two BrainData objects have different ``n_voxels``.

(data-brain-data-upload-neurovault)=
### `upload_neurovault`

```python
upload_neurovault(*, access_token = None, collection_name = None, collection_id = None, img_type = None, img_modality = None, **kwargs)
```

Upload BrainData images and metadata to NeuroVault.

Adds any columns in ``self.X`` to image metadata. The index is used as
the image name.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`access_token` | <code>str</code> | NeuroVault API access token. Required. | <code>None</code>
`collection_name` | <code>str \| None</code> | Name of a new collection to create. | <code>None</code>
`collection_id` | <code>int \| None</code> | NeuroVault ``collection_id`` when adding images to an existing collection. | <code>None</code>
`img_type` | <code>str</code> | NeuroVault ``map_type``. Required. | <code>None</code>
`img_modality` | <code>str</code> | NeuroVault image modality. Required. | <code>None</code>
`**kwargs` | <code>dict</code> | Additional image metadata forwarded to the NeuroVault API. | <code>{}</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | NeuroVault collection information.

(data-brain-data-write)=
### `write`

```python
write(file_name)
```

Write out BrainData object to Nifti or HDF5 File.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`file_name` | <code>str or Path</code> | Output file path (.nii/.nii.gz for NIfTI, .h5/.hdf5 for HDF5). | *required*

(data-brain-data-z-to-r)=
### `z_to_r`

```python
z_to_r()
```

Convert z score back into r value for each element of data object.
