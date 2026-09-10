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
[`bootstrap`](#data-brain-data-bootstrap) | Bootstrap a statistic and its uncertainty, on CPU workers or a GPU.
[`cluster_report`](#data-brain-data-cluster-report) | Generate a cluster report with anatomical labels.
[`compute_contrasts`](#data-brain-data-compute-contrasts) | Compute contrasts on a fitted GLM.
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
[`predict`](#data-brain-data-predict) | Predict voxel responses from a fitted model, or decode labels with MVPA.
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
[`ttest`](#data-brain-data-ttest) | Run a one-sample voxelwise t-test across images (axis 0).
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
bootstrap(statistic, *, X = None, X_test = None, n_samples = 5000, confidence_level = 0.95, device = 'cpu', memory_budget_gb = None, return_samples = False, n_jobs = -1, random_state = None, progress_bar: bool = False)
```

Bootstrap a statistic and its uncertainty, on CPU workers or a GPU.

Resamples rows with replacement and aggregates the replicates as they
complete, into a running Welford variance plus just enough retained
order statistics per output element to reproduce the exact percentile
interval. What the run holds is that retained tail — about
``(1 - confidence_level)`` of the replicates per element — plus one
dispatch window, rather than all ``n_samples`` maps. This is
memory-efficient, not constant-memory: the tail still grows with
``n_samples``, and ``return_samples=True`` keeps the whole
distribution.

A Ridge bootstrap resamples the training features you pass as ``X``
together with ``self.data``, using the same row indices for every
feature space, and refits with the fitted model's selected ``alpha_``
— and, for a banded model, its ``feature_space_weights_`` — held fixed.
It never reruns cross-validation or the banded random search. Fitting
keeps no hidden copy of the training features, so ``X`` is required
even when the same features were passed to `fit`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`statistic` | <code>str</code> | Statistic to bootstrap. Basic aggregates: ``'mean'``, ``'median'``, ``'std'``, ``'sum'``, ``'min'``, ``'max'`` — each the corresponding NumPy reduction over rows, with ``'std'`` at ``ddof=0``. Model statistics (require a fitted `Ridge`): ``'weights'`` or ``'predict'``. | *required*
`X` | <code>ndarray \| Mapping[str, ndarray] \| None</code> | Training features in their original row order — a matrix for ordinary Ridge, a mapping with exactly the fitted feature-space names for banded Ridge. Required by both model statistics; rejected by the basic ones. | <code>None</code>
`X_test` | <code>ndarray \| Mapping[str, ndarray] \| None</code> | Evaluation features for ``statistic='predict'``, in the same structure as ``X``. Any row count is allowed. | <code>None</code>
`n_samples` | <code>int</code> | Number of bootstrap replicates, at least two. Default 5000. | <code>5000</code>
`confidence_level` | <code>float</code> | Confidence level of the reported interval, strictly between zero and one. Default 0.95. The bounds are the central percentile interval by linear interpolation, and they are elementwise marginal: the nominal level applies separately to each voxel, feature, or test row, with no simultaneous-coverage claim. A different level needs a new run unless ``return_samples=True`` kept the distribution. | <code>0.95</code>
`device` | <code>str</code> | Compute device for the Ridge refits: ``'cpu'`` (default) or ``'gpu'`` (PyTorch on CUDA/MPS, or an error when neither is available). Basic statistics reject ``'gpu'``. | <code>'cpu'</code>
`memory_budget_gb` | <code>float \| None</code> | Working-memory budget in GB. It governs the output preflight and CPU-worker planning for every statistic, and GPU batch sizing for the Ridge ones. ``None`` (default) measures the device. | <code>None</code>
`return_samples` | <code>bool</code> | Retain and return every replicate. Default False. It changes retention only, never interval semantics. | <code>False</code>
`n_jobs` | <code>int</code> | CPU worker ceiling. -1 (default) means all cores; the planner may use fewer. | <code>-1</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | If True, show a progress bar. Default False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[BootstrapResult](#data-results-bootstrapresult)</code> | ``estimate`` (the statistic on the unresampled     full sample — for ``'weights'`` the fitted coefficients, for     ``'predict'`` the full-data model at ``X_test``),     ``standard_error`` (the ``ddof=1`` deviation across     replicates), ``ci_lower`` and ``ci_upper``, all `BrainData` of     identical shape, plus ``samples`` as a NumPy array with the     bootstrap axis first when ``return_samples=True``.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If `statistic` is unknown, a basic statistic is given ``X``, ``X_test`` or ``device='gpu'``, a Ridge statistic is missing its features, the fitted model is not a `Ridge`, an argument is out of range, or the retained output cannot fit the memory budget.

**Examples:**

```python
boot = brain.bootstrap('mean', n_samples=1000)
boot.estimate.plot()

brain.fit(model='ridge', X=features, ridge_alpha=1.0)
boot = brain.bootstrap('weights', X=features, n_samples=1000)
```

<details class="note" open markdown="1">
<summary>Note</summary>

This is an IID row bootstrap. Rows must be exchangeable for the
interval to be meaningful; it implements no grouped, clustered,
stratified, or block resampling, so an autocorrelated fMRI time
series must not be treated as IID rows.

</details>

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
compute_contrasts(contrasts, *, inference = False)
```

Compute contrasts on a fitted GLM.

Call after ``fit(model='glm', X=design)``. The fitted `Glm` owns
contrast parsing and inference; this method forwards each definition
unchanged and wraps the results as `BrainData` maps.

A contrast is a **string** naming design columns with optional
coefficients (``"conditionA - conditionB"``, ``"2*A - B - C"``) or a
**numeric vector** with one weight per column (``[1, -1, 0, 0]``). A
**mapping** of names to those forms computes several at once and is the
only batch form.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`contrasts` | <code>str \| array - like \| Mapping</code> | One contrast definition, or a mapping of names to definitions. | *required*
`inference` | <code>bool</code> | If True, return `ContrastResult` records carrying effect, variance, standard error, t-statistic, z-score, one-sided p-value, and degrees of freedom. Default False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data) \| [ContrastResult](#models-contrastresult) \| dict</code> | An effect map for one contrast,     or a `ContrastResult` of maps when ``inference=True``; a     dictionary with the same keys for a mapping.

**Raises:**

Type | Description
---- | -----------
<code>RuntimeError</code> | If no model has been fitted.
<code>ValueError</code> | If the fitted model is not a `Glm`, or a contrast is invalid (see `Glm.compute_contrasts`).

**Examples:**

```python
brain.fit(model='glm', X=design)

# Effect maps — what a second-level model consumes
effect = brain.compute_contrasts("conditionA - conditionB")
effects = brain.compute_contrasts({
    "A_vs_B": "conditionA - conditionB",
    "avg": [0, 0.5, 0.5],
})

# First-level inference
result = brain.compute_contrasts("conditionA - conditionB", inference=True)
result.statistic.plot(threshold=3.09)
```

<details class="note" open markdown="1">
<summary>Note</summary>

Contrast p-values are one-sided, following the nilearn/SPM
directional-contrast convention; negate the contrast to test the
other direction.

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
fit(model = 'glm', *, X = None, ridge_alpha = 1.0, ridge_cv = None, ridge_search_iterations = 100, ridge_dirichlet_concentration = (0.1, 1.0), ridge_device = 'cpu', ridge_memory_budget_gb = None, ridge_per_target_alpha = True, ridge_prefer_conservative_alpha = False, ridge_progress_bar = False, glm_noise_model = 'ols', glm_bins = 100, glm_n_jobs = 1, inplace = True, random_state = None)
```

Fit a model to brain imaging data.

``self.data`` is always the response. The fitted estimator and its
results are stored for later use with `predict` and, for a GLM,
`compute_contrasts`.

Every model-specific option carries a ``glm_`` or ``ridge_`` prefix
naming the estimator it configures; ``random_state`` keeps its bare
name because both estimators accept it. Supplying a non-default option
belonging to the estimator ``model`` did not select raises
`ValueError`.

`fit` does not preprocess the response. Compose `scale` and
`standardize` before calling it when you want them, so the fitted
object stays in the response space you supplied.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`model` | <code>str</code> | ``'glm'`` (default) or ``'ridge'``. | <code>'glm'</code>
`X` | <code>[DesignMatrix](#page-data-design-matrix) \| array - like \| Mapping</code> | A precomputed `DesignMatrix` for a GLM; a feature matrix for ridge, or a mapping of feature-space names to matrices for banded ridge. Required. | <code>None</code>
`ridge_alpha` | <code>float \| Sequence[float]</code> | Ridge only. A positive scalar fits a fixed α and requires ``ridge_cv=None``; a sequence selects α by cross-validation and requires ``ridge_cv``. Default 1.0. | <code>1.0</code>
`ridge_cv` | <code>int \| sklearn splitter \| None</code> | Ridge only. Cross-validation specification; ``int`` → unshuffled ``KFold(cv)``. Generators are rejected. Default None. | <code>None</code>
`ridge_search_iterations` | <code>int</code> | Ridge only, banded. Sampled feature-space weight vectors. Default 100. | <code>100</code>
`ridge_dirichlet_concentration` | <code>float \| Sequence[float]</code> | Ridge only, banded. Dirichlet concentration for those candidate weights. Default ``(0.1, 1.0)``. | <code>(0.1, 1.0)</code>
`ridge_device` | <code>str</code> | Ridge only. ``'cpu'`` (default) or ``'gpu'``. | <code>'cpu'</code>
`ridge_memory_budget_gb` | <code>float \| None</code> | Ridge only. Working-memory budget in GB for the solver's internal batching. Default None (measure the device). | <code>None</code>
`ridge_per_target_alpha` | <code>bool</code> | Ridge only. Select α per voxel (default True) or one shared α. | <code>True</code>
`ridge_prefer_conservative_alpha` | <code>bool</code> | Ridge only. Select the largest α within one standard deviation of the best score. Default False. | <code>False</code>
`ridge_progress_bar` | <code>bool</code> | Ridge only. Show a progress bar over the banded search. Default False. | <code>False</code>
`glm_noise_model` | <code>str</code> | GLM only. ``'ols'`` (default) or ``'arN'`` for Nilearn's autoregressive model of order N. | <code>'ols'</code>
`glm_bins` | <code>int</code> | GLM only. Nilearn's discretization of the estimated AR coefficients. Default 100. | <code>100</code>
`glm_n_jobs` | <code>int</code> | GLM only. CPUs Nilearn uses for autoregressive groups; the default OLS fit does not use this path. Default 1. | <code>1</code>
`inplace` | <code>bool</code> | If True (default), mutate self and return self. If False, fit and return an independent `BrainData` copy while leaving every part of self untouched. | <code>True</code>
`random_state` | <code>int \| None</code> | Seed shared by both estimators. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Self when ``inplace=True``; otherwise an independently     owned fitted copy.

<details class="note" open markdown="1">
<summary>Note</summary>

A GLM fit attaches ``model_``, ``glm_betas`` (one map per design
column), ``glm_residual``, ``glm_predicted``, and ``glm_r2``.
``glm_r2`` is Nilearn's whitened variance ratio: conventional
R-squared for an OLS fit whose design has an intercept, and a
pseudo-R-squared in the whitened space for an autoregressive one.
A GLM fit does not compute eager per-regressor t, p, or
standard-error maps: ask for them one contrast at a time with
``compute_contrasts(..., inference=True)``, which uses the full
per-voxel parameter covariance and is therefore correct for
contrasts spanning several regressors.

</details>

**Examples:**

```python
brain_data.fit(model='glm', X=design)
effect = brain_data.compute_contrasts('conditionA - conditionB')

fitted = brain_data.fit(
    model='ridge', ridge_alpha=1.0, X=features, inplace=False
)
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
predict(*, X: DesignMatrix | np.ndarray | Mapping[str, np.ndarray] | None = None, y: np.ndarray | str | None = None, estimator: str | BaseEstimator = 'linear_svc', cv: int | BaseCrossValidator | None = None, groups: np.ndarray | str | None = None, scoring: str | Callable | None = None, spatial_scale: Literal['whole_brain', 'roi', 'searchlight'] = 'whole_brain', roi_mask: Nifti1Image | str | Path | None = None, radius: float = 10.0, n_jobs: int = 1, progress_bar: bool = False)
```

Predict voxel responses from a fitted model, or decode labels with MVPA.

Exactly one mode is resolved before any work happens:

- an explicit ``y=`` runs MVPA decoding and returns a `Predict`;
- an explicit ``X=`` predicts from the fitted `Glm` or `Ridge` and
  returns a new, independently owned `BrainData`;
- with neither argument and a fitted model, an independent copy of the
  stored training predictions;
- with neither argument, no fitted model, and exactly one ``.Y`` column,
  MVPA on that column.

Supplying both ``X`` and ``y``, or a decoding argument on a
fitted-model call, raises before prediction begins. A fitted model wins
over an attached ``.Y`` on the no-argument call — pass ``y=``
explicitly to decode instead. `predict` never mutates the source and
attaches nothing to it.

Labels travel with the data: ``y='name'`` picks a column of ``.Y``, and
``groups`` accepts a ``.Y`` column name the same way. With an explicit
``X=``, the estimator validates and aligns it: a `DesignMatrix` whose
column names `Glm.predict` matches to the fitted order, or, for a
banded `Ridge`, a mapping with exactly the fitted feature-space names
in any order.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[DesignMatrix](#page-data-design-matrix) \| array - like \| Mapping</code> | Features for fitted-model prediction, shape ``(n_samples, n_features)``, or a mapping of feature-space names to matrices for a banded `Ridge`. | <code>None</code>
`y` | <code>array - like \| str</code> | Labels (classification) or continuous targets (regression), shape ``(n_samples,)``, or the name of a ``.Y`` column. Must be one-dimensional with one value per row; multioutput and multilabel targets are not accepted. | <code>None</code>
`estimator` | <code>str \| sklearn estimator</code> | A built-in shortcut — ``'linear_svc'``, ``'logistic_regression'``, ``'linear_discriminant_analysis'``, ``'ridge_classifier'``, ``'ridge'``, ``'lasso'``, ``'linear_svr'`` — or any sklearn estimator or `Pipeline`, which is used exactly as supplied. Default ``'linear_svc'``. Every shortcut standardizes voxels inside each fold and then fits a linear estimator; a classification shortcut on a multiclass target is wrapped in `OneVsRestClassifier`, so every class gets its own signed map. A caller-supplied estimator is never wrapped and never has its multiclass strategy overridden — pass a `OneVsRestClassifier` to get one. Every preprocessing step, in every spatial scale, must be one of `StandardScaler`, `PCA`, `VarianceThreshold`, `GenericUnivariateSelect`, `SelectPercentile`, `SelectKBest`, `SelectFpr`, `SelectFdr`, `SelectFwe`, `SelectFromModel`, `RFE`, `RFECV`, `SequentialFeatureSelector`, ``None``, or ``'passthrough'``. Whole-brain and ROI pipelines must also end in an estimator exposing ``coef_``, since those two scales extract a weight map; searchlight builds none and does not require it. | <code>'linear_svc'</code>
`cv` | <code>int \| sklearn splitter</code> | ``None`` (the default) is a deterministic five-fold ``KFold`` (regression) or ``StratifiedKFold`` (classification); an int selects that many folds; an sklearn splitter is used as supplied. Test folds must partition the rows, so shuffle-split and repeated splitters raise. Rows ordered by condition make unshuffled contiguous folds degenerate — pass a shuffled splitter to control that, e.g. ``cv=KFold(n_splits=5, shuffle=True, random_state=0)``. | <code>None</code>
`groups` | <code>array - like \| str</code> | Group labels passed to the splitter (e.g. ``LeaveOneGroupOut`` for leave-one-run-out), one value per row, or the name of a ``.Y`` column holding them. | <code>None</code>
`scoring` | <code>str \| callable</code> | Follows scikit-learn's single-metric scoring contract. ``None`` (the default) uses the estimator's own ``score`` method; a scoring name or callable overrides it. Multimetric mappings are not accepted. | <code>None</code>
`spatial_scale` | <code>str</code> | MVPA dispatch — ``'whole_brain'``, ``'roi'``, or ``'searchlight'``. | <code>'whole_brain'</code>
`roi_mask` | <code>Nifti1Image \| path - like</code> | Atlas image; required by, and only valid for, ``spatial_scale='roi'``. | <code>None</code>
`radius` | <code>float</code> | Searchlight sphere radius in millimeters; only valid for ``spatial_scale='searchlight'``. Default ``10.0``. | <code>10.0</code>
`n_jobs` | <code>int</code> | Parallel workers for the outer independent work of the selected spatial scale — cross-validation folds for whole-brain, parcels for ROI, spheres for searchlight. Default ``1``; every worker holds a copy of the data, so a real brain at higher ``n_jobs`` can be memory-heavy. | <code>1</code>
`progress_bar` | <code>bool</code> | Show a progress bar for searchlight and ROI. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[Predict](#data-results-predict) \| [BrainData](#page-data-brain-data)</code> | A `Predict` record for MVPA; a new `BrainData`     holding the predicted timeseries for fitted-model prediction.     The record's ``spatial_scale`` says which of its fields carry     values: whole-brain fills ``predictions``, ``cv_folds``,     ``scores``, ``estimator`` and ``weight_map``; ROI fills     ``scores``, ``roi_labels``, ``score_map`` and ``weight_map``;     searchlight fills ``score_map`` alone. ``classes`` accompanies     any classifier and ``scoring`` records the scoring     specification in every mode. ``mean_score`` and ``std_score``     are computed from ``scores`` on demand and do not exist for a     searchlight result. ``weight_map`` holds one coefficient map     for regression and binary classification (the signed map for     ``classes[1]`` versus ``classes[0]``) and one map per class, in     ``classes`` order, for multiclass — never an average across     classes. It is projected back to voxel units through the     pipeline's fitted preprocessing, but centering is not undone,     so ``raw_data @ weight_map`` does not reproduce the decision     function; use ``result.estimator`` to predict.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | On both ``X`` and ``y``, a decoding argument on a fitted-model call, an unknown estimator shortcut or spatial scale, a target or group vector that is not one value per row, cross-validation folds that do not partition the rows, a preprocessing step outside the supported set, or — for whole-brain and ROI decoding — a pipeline whose coefficients cannot be projected back onto the voxel axis.
<code>TypeError</code> | On a removed keyword, an `estimator` that is neither a shortcut name nor an object with `fit`/`predict`, or a `cv` that is neither `None`, an int, nor a splitter.

**Examples:**

Whole-brain decoding:

```python
result = brain.predict(y=labels, cv=5)
result.weight_map.plot()   # the all-data refit — the publishable map
result.mean_score          # the cross-validated score
new_pred = result.estimator.predict(new_X)
```

Searchlight and ROI decoding:

```python
result = brain.predict(
    y=labels, spatial_scale='searchlight', radius=8.0, n_jobs=4
)
result.score_map.plot()    # one score per sphere center

result = brain.predict(y=labels, spatial_scale='roi', roi_mask=atlas)
result.mean_score          # one score per parcel
result.score_map.plot()    # those scores painted into voxel space
```

Prediction from a fitted encoding model:

```python
brain.fit(model='ridge', X=features)
predicted = brain.predict(X=new_features)
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

Run a one-sample voxelwise t-test across images (axis 0).

Tests whether the per-voxel mean across a stack of images (e.g.
subject-level contrast maps, shape `(n_images, n_voxels)`) differs from
`popmean`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`popmean` | <code>float</code> | Population mean to test against. Default 0.0. | <code>0.0</code>
`permutation` | <code>bool</code> | If True, take p from a sign-flip permutation test on `images - popmean`. The reported `t` stays the observed parametric statistic. Default False. | <code>False</code>
`n_permute` | <code>int</code> | Number of permutations, used only when `permutation=True`. Default 5000. | <code>5000</code>
`tail` | <code>int \| str</code> | `2` or `'two'` for two-tailed (default); `1` or `'one'` for one-tailed (mean > `popmean`). | <code>2</code>
`return_null` | <code>bool</code> | If True, also return the permutation null. Has no effect on the parametric path, which computes no null. Default False. | <code>False</code>
`n_jobs` | <code>int</code> | Number of parallel jobs. Default -1 (all cores). | <code>-1</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | `"mean"`, `"t"`, `"z"` and `"p"` as independent `BrainData`     images with observation metadata cleared. `"mean"` is the     voxelwise mean minus `popmean` — the effect relative to the     tested null, equal to the raw mean only when `popmean=0`.     `"t"` is the observed one-sample t-statistic on both paths.     `"p"` is parametric, or the empirical sign-flip p-value when     `permutation=True`. `"z"` is the tail-aware normal score of `p`     (`sign(t) * norm.isf(p/2)` two-tailed), matching nilearn's     `output_type='z_score'`. With `permutation=True` and     `return_null=True` the dict also holds `"null_dist"`, an owned     `(n_permute, n_voxels)` array of centered means in the units of     `"mean"`. Maps are unthresholded. Apply a cutoff or a     multiple-comparison correction afterwards.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If this BrainData contains fewer than 2 images.

**Examples:**

```python
# Stack of subject-level contrast maps
result = contrast_maps.ttest()
effect = result["mean"]  # magnitude, for reporting
z_map = result["z"]  # for nilearn-style thresholding

# Threshold after testing, never inside it
from nltools.algorithms import threshold

z_thresh = threshold(result["z"], result["p"], thr=0.001)

# Permutation p-values, keeping the null for a custom correction
perm = contrast_maps.ttest(
    permutation=True, n_permute=5000, return_null=True, random_state=0
)
perm["null_dist"].shape  # → (5000, n_voxels)
```

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
