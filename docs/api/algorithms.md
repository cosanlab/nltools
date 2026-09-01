(algorithms-algorithms)=
## `algorithms`

nltools.algorithms — the functional core of nltools.

Every user-facing statistical function and algorithm is importable flat from
here (`from nltools.algorithms import fdr, zscore, isc`), organized into
focused submodules underneath:

- **corrections**: multiple-comparison corrections (FDR, Holm-Bonferroni, thresholding)
- **outliers**: outlier detection, winsorizing, z-scoring
- **signal**: temporal signal processing (resampling, filtering, basis functions)
- **similarity**: similarity metrics and Fisher transforms
- **regression**: standalone OLS regression on numpy arrays
- **alignment**: SRM, HyperAlignment, LocalAlignment, and the functional
  `align`/`procrustes` entry points
- **inference**: permutation tests, bootstrap resampling, and intersubject
  statistics (ISC/ISFC/ISPS) with CPU-parallel and GPU backends
- **ridge**: regularized regression (also exported as a module for advanced usage)
- **hrf**: hemodynamic response functions

**Classes:**

Name | Description
---- | -----------
[`DetSRM`](#algorithms-detsrm) | Deterministic Shared Response Model (DetSRM).
[`HyperAlignment`](#algorithms-hyperalignment) | Hyperalignment using iterative Procrustes alignment.
[`LocalAlignment`](#algorithms-localalignment) | Local (neighborhood-based) functional alignment across subjects.
[`SRM`](#algorithms-srm) | Probabilistic Shared Response Model (SRM).

**Methods:**

Name | Description
---- | -----------
[`align`](#algorithms-align) | Align subject data into a common response model.
[`align_states`](#algorithms-align-states) | Align state weight maps by minimizing pairwise distance between group states.
[`calc_bpm`](#algorithms-calc-bpm) | Calculate instantaneous BPM from beat to beat interval.
[`circle_shift`](#algorithms-circle-shift) | Circular shift for time-series data.
[`compute_multivariate_similarity`](#algorithms-compute-multivariate-similarity) | Compute multivariate similarity via OLS regression.
[`compute_similarity`](#algorithms-compute-similarity) | Compute similarity between two data arrays.
[`correlation_permutation_test`](#algorithms-correlation-permutation-test) | Correlation permutation test.
[`distance_correlation`](#algorithms-distance-correlation) | Compute the distance correlation between 2 arrays to test for multivariate dependence (linear or non-linear).
[`double_center`](#algorithms-double-center) | Double center a 2d array.
[`downsample`](#algorithms-downsample) | Downsample a Polars DataFrame/Series to a new target frequency or number of samples using averaging.
[`fdr`](#algorithms-fdr) | Determine an FDR threshold for an array of p-values.
[`find_spikes`](#algorithms-find-spikes) | Identify spikes (motion artifacts, intensity outliers) in 4D fMRI data.
[`fisher_r_to_z`](#algorithms-fisher-r-to-z) | Use Fisher transformation to convert correlation to z score.
[`fisher_z_to_r`](#algorithms-fisher-z-to-r) | Convert Fisher z back to a correlation coefficient.
[`glover_dispersion_derivative`](#algorithms-glover-dispersion-derivative) | Implement the Glover dispersion derivative :term:`HRF` model.
[`glover_hrf`](#algorithms-glover-hrf) | Implement the Glover :term:`HRF` model.
[`glover_time_derivative`](#algorithms-glover-time-derivative) | Implement the Glover time derivative :term:`HRF` (dhrf) model.
[`holm_bonf`](#algorithms-holm-bonf) | Compute Holm-Bonferroni-corrected p-values.
[`isc`](#algorithms-isc) | Compute pairwise intersubject correlation from observations by subjects array.
[`isc_group`](#algorithms-isc-group) | Compute difference in intersubject correlation between groups.
[`isc_group_permutation_test`](#algorithms-isc-group-permutation-test) | Compute ISC difference between groups with permutation testing.
[`isc_permutation_test`](#algorithms-isc-permutation-test) | Compute intersubject correlation with permutation testing.
[`isfc`](#algorithms-isfc) | Compute intersubject functional connectivity (ISFC) from a list of observation x feature matrices.
[`isps`](#algorithms-isps) | Compute dynamic intersubject phase synchrony (ISPS) from an observations-by-subjects array.
[`make_cosine_basis`](#algorithms-make-cosine-basis) | Create basis functions for a discrete cosine transform.
[`matrix_permutation_test`](#algorithms-matrix-permutation-test) | Matrix permutation test (Mantel test) for correlating two square matrices.
[`multi_threshold`](#algorithms-multi-threshold) | Threshold test image by multiple p-values from p image.
[`one_sample_permutation_test`](#algorithms-one-sample-permutation-test) | One-sample permutation test using sign-flipping.
[`phase_randomize`](#algorithms-phase-randomize) | FFT-based phase randomization for time-series data.
[`procrustes_distance`](#algorithms-procrustes-distance) | Test matrix similarity using Procrustes superposition.
[`regress`](#algorithms-regress) | Fit an OLS regression of ``Y`` on ``X``.
[`ridge_cv`](#algorithms-ridge-cv) | Ridge regression with cross-validation for hyperparameter selection.
[`ridge_svd`](#algorithms-ridge-svd) | Solve ridge regression using Singular Value Decomposition.
[`spm_dispersion_derivative`](#algorithms-spm-dispersion-derivative) | Implement the :term:`SPM` dispersion derivative :term:`HRF` model.
[`spm_hrf`](#algorithms-spm-hrf) | Implement the :term:`SPM` :term:`HRF` model.
[`spm_time_derivative`](#algorithms-spm-time-derivative) | Implement the :term:`SPM` time derivative :term:`HRF` (dhrf) model.
[`threshold`](#algorithms-threshold) | Threshold test image by p-value from p image.
[`timeseries_correlation_permutation_test`](#algorithms-timeseries-correlation-permutation-test) | Time-series correlation permutation test.
[`transform_pairwise`](#algorithms-transform-pairwise) | Transform data into pairs with balanced labels for ranking.
[`trim`](#algorithms-trim) | Trim a Polars DataFrame/Series by replacing outlier values with NaNs.
[`two_sample_permutation_test`](#algorithms-two-sample-permutation-test) | Two-sample permutation test using group label shuffling.
[`u_center`](#algorithms-u-center) | U-center a 2d array. U-centering is a bias-corrected form of double-centering.
[`upsample`](#algorithms-upsample) | Upsample a Polars DataFrame/Series to a new target frequency or number of samples using interpolation.
[`winsorize`](#algorithms-winsorize) | Winsorize a Polars DataFrame/Series with the largest/lowest value not considered outlier.
[`zscore`](#algorithms-zscore) | Z-score every column of a Polars or pandas DataFrame/Series.



**Modules:**

Name | Description
---- | -----------
[`alignment`](#algorithms-alignment) | Multi-subject functional alignment algorithms.
[`backends`](#algorithms-backends) | Backend abstraction for CPU/GPU operations.
[`corrections`](#algorithms-corrections) | Multiple comparison corrections and thresholding.
[`hrf`](#algorithms-hrf) | Hemodynamic response functions — re-exported from nilearn.
[`inference`](#algorithms-inference) | GPU-accelerated statistical inference for neuroimaging.
[`outliers`](#algorithms-outliers) | Outlier detection, robust statistics, and data normalization.
[`procrustes`](#algorithms-procrustes) | Data alignment — SRM, Procrustes, and state alignment.
[`random`](#algorithms-random) | Shared random-state utilities for deterministic parallel execution.
[`regression`](#algorithms-regression) | Standalone OLS regression on numpy arrays.
[`ridge`](#algorithms-ridge) | Ridge regression algorithms and utilities.
[`shape_utils`](#algorithms-shape-utils) | Shared shape-manipulation helpers for triangle extraction and symmetric permutation.
[`signal`](#algorithms-signal) | Temporal signal processing — resampling, filtering, and basis functions.
[`similarity`](#algorithms-similarity) | Similarity metrics and correlation.

### Classes

(algorithms-detsrm)=
#### `DetSRM`

```python
DetSRM(*, n_iter: int = 10, features: int = 50, rand_seed: int = 0) -> None
```

Bases: <code>[BaseEstimator](#sklearn.base.BaseEstimator)</code>, <code>[TransformerMixin](#sklearn.base.TransformerMixin)</code>

Deterministic Shared Response Model (DetSRM).

Given multi-subject data, factorize it as a shared response S among all
subjects and an orthogonal transform W per subject:

$$
X_i \approx W_i S, \forall i=1 \dots N
$$

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_iter` | <code>int, default=10</code> | Number of iterations to run the algorithm. | <code>10</code>
`features` | <code>int, default=50</code> | Number of features to compute. | <code>50</code>
`rand_seed` | <code>int, default=0</code> | Seed for initializing the random number generator. | <code>0</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`w_` | <code>list of array, element i has shape=[voxels_i, features]</code> | The orthogonal transforms (mappings) for each subject.
`s_` | <code>array, shape=[features, samples]</code> | The shared response.
`random_state_` | <code>`RandomState`</code> | Random number generator initialized using rand_seed

<details class="note" open markdown="1">
<summary>Note</summary>

The number of voxels may be different between subjects. However, the
number of samples must be the same across subjects.

The Deterministic Shared Response Model is approximated using the
Block Coordinate Descent (BCD) algorithm proposed in **Chen2015**.

This is a single node version.

The run-time complexity is $O(I (V T K + V K^2))$ and the memory
complexity is $O(V T)$ with I - the number of iterations, V - the
sum of voxels from all subjects, T - the number of samples, K - the
number of features (typically, $V \gg T \gg K$), and N - the
number of subjects.

</details>

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Compute the Deterministic Shared Response Model.
[`transform`](#algorithms-transform) | Use the model to transform data to the Shared Response subspace.
[`transform_subject`](#algorithms-transform-subject) | Transform a new subject using the existing model.



**Examples:**

Basic multi-subject DetSRM fitting:

```pycon
>>> from nltools.algorithms import DetSRM
>>> import numpy as np
>>>
>>> # Create sample data (3 subjects)
>>> data = [np.random.randn(100, 50) for _ in range(3)]
>>>
>>> # Fit DetSRM with CPU parallelization (default)
>>> detsrm = DetSRM(n_iter=10, features=50)
>>> detsrm.fit(data, parallel="cpu", n_jobs=-1)
>>>
>>> # Transform to shared response space
>>> shared_responses = detsrm.transform(data)
>>>
>>> # Access fitted model components
>>> w = detsrm.w_  # Subject-specific transforms
>>> s = detsrm.s_  # Shared response
```

##### Methods

(algorithms-fit)=
###### `fit`

```python
fit(X: list[np.ndarray], y: Any | None = None, *, parallel: str | None = 'cpu', n_jobs: int = -1) -> DetSRM
```

Compute the Deterministic Shared Response Model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples]</code> | Each element in the list contains the fMRI data of one subject. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": not implemented -- raises `NotImplementedError` (never a silent CPU fallback) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`self` | <code>[DetSRM](#nltools.algorithms.alignment.srm.DetSRM)</code> | Fitted model

(algorithms-transform)=
###### `transform`

```python
transform(X: list[np.ndarray], y: Any | None = None, *, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray]
```

Use the model to transform data to the Shared Response subspace.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples_i]</code> | Each element in the list contains the fMRI data of one subject. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": not implemented -- raises `NotImplementedError` (never a silent CPU fallback) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`s` | <code>list of 2D arrays, element i has shape=[features_i, samples_i]</code> | Shared responses from input data (X)

(algorithms-transform-subject)=
###### `transform_subject`

```python
transform_subject(X: np.ndarray) -> np.ndarray
```

Transform a new subject using the existing model.

The subject is assumed to have received equivalent stimulation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>2D array, shape=[voxels, timepoints]</code> | The fMRI data of the new subject. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`w` | <code>2D array, shape=[voxels, features]</code> | Orthogonal mapping `W_{new}` for new subject

(algorithms-hyperalignment)=
#### `HyperAlignment`

```python
HyperAlignment(n_iter: int = 2, auto_pad: bool = True) -> None
```

Bases: <code>[BaseEstimator](#sklearn.base.BaseEstimator)</code>, <code>[TransformerMixin](#sklearn.base.TransformerMixin)</code>

Hyperalignment using iterative Procrustes alignment.

Three-stage iterative process for aligning multi-subject data:
1. Create initial average template
2. Refine template through n_iter iterations
3. Final alignment of all subjects to refined template

This implements the Procrustes-based hyperalignment method commonly
used in multi-subject neuroimaging analysis.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_iter` | <code>int, default=2</code> | Number of template refinement iterations (stages 1-2). | <code>2</code>
`auto_pad` | <code>bool, default=True</code> | If True, automatically zero-pad matrices to standardize sizes. If False, caller must ensure all matrices have same dimensions. | <code>True</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_iter` | <code>int, default=2</code> | Number of template refinement iterations | <code>2</code>
`auto_pad` | <code>bool, default=True</code> | Whether to automatically pad matrices to same size | <code>True</code>



**Attributes:**

Name | Type | Description
---- | ---- | -----------
`w_` | <code>list of ndarray, element i has shape=[features_i, features]</code> | The transformation matrices (rotation + reflection) for each subject.
`s_` | <code>ndarray, shape=[features, samples]</code> | The aligned common template (shared response).
`disparity_` | <code>list of float</code> | Disparity (sum of squared differences) for each subject.
`scale_` | <code>list of float</code> | Scale factors for each subject.

<details class="note" open markdown="1">
<summary>Note</summary>

``common_model_`` property provides alias for ``s_`` (backward compatibility).

</details>

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Fit hyperalignment model to data.
[`transform`](#algorithms-transform) | Transform data to common space using fitted transformations.
[`transform_subject`](#algorithms-transform-subject) | Align a new subject to the common space.

**Examples:**

Basic multi-subject alignment:

```pycon
>>> from nltools.algorithms import HyperAlignment
>>> import numpy as np
>>>
>>> # Create sample data (3 subjects)
>>> data = [np.random.randn(100, 50) for _ in range(3)]
>>>
>>> # Fit hyperalignment with CPU parallelization (default)
>>> hyper = HyperAlignment(n_iter=2)
>>> hyper.fit(data, parallel="cpu", n_jobs=-1)
>>>
>>> # Transform to common space
>>> aligned = hyper.transform(data)
>>>
>>> # Access common template
>>> template = hyper.s_  # or hyper.common_model_
>>>
>>> # Align a new subject
>>> new_subject = np.random.randn(100, 50)
>>> new_transform = hyper.transform_subject(new_subject)
```

<details class="note" open markdown="1">
<summary>Note</summary>

When to use parallel processing:

- Use ``parallel="cpu"`` (default) for datasets with 3+ subjects to speed up
  pairwise Procrustes operations during template refinement.
- Use ``parallel=None`` for debugging or small datasets (<3 subjects) where
  parallelization overhead isn't beneficial.
- Parallel processing is most beneficial when subjects have many voxels
  (>10K) and template refinement requires multiple iterations.

</details>

<details class="references" open markdown="1">
<summary>References</summary>

Haxby, J. V., Guntupalli, J. S., Connolly, A. C., Halchenko, Y. O.,
Conroy, B. R., Gobbini, M. I., ... & Ramadge, P. J. (2011).
A common, high-dimensional model of the representational space in
human ventral temporal cortex. Neuron, 72(2), 404-416.

</details>

##### Methods

###### `fit`

```python
fit(data: list[np.ndarray], *, parallel: str | None = 'cpu', n_jobs: int = -1) -> HyperAlignment
```

Fit hyperalignment model to data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>list of ndarray</code> | List of data matrices, each with shape (n_features, n_samples). Different subjects can have different numbers of features if auto_pad=True. | *required*
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`self` | <code>[HyperAlignment](#nltools.algorithms.alignment.hyperalignment.HyperAlignment)</code> | Fitted model

###### `transform`

```python
transform(data: list[np.ndarray], *, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray]
```

Transform data to common space using fitted transformations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>list of ndarray</code> | List of data matrices to transform. Should be the same data used for fitting (or have compatible dimensions). | *required*
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`transformed` | <code>list of ndarray</code> | List of transformed data matrices in common space

###### `transform_subject`

```python
transform_subject(subject_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]
```

Align a new subject to the common space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`subject_data` | <code>([ndarray](#ndarray), [shape](#shape)([n_features](#n_features), [n_samples](#n_samples)))</code> | Data from a new subject to align to the common template | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`transformed` | <code>[ndarray](#ndarray)</code> | Aligned data in common space
`R` | <code>[ndarray](#ndarray)</code> | Transformation matrix used
`disparity` | <code>[float](#float)</code> | Alignment quality (sum of squared differences)
`scale` | <code>[float](#float)</code> | Scale factor used

(algorithms-localalignment)=
#### `LocalAlignment`

```python
LocalAlignment(spatial_scale: str = 'searchlight', method: str = 'procrustes', radius_mm: float = 10.0, roi_mask: nib.Nifti1Image | None = None, n_features: int | None = None, n_iter: int = 3, aggregation: str = 'center', parallel: str | None = 'cpu', n_jobs: int = -1, progress_bar: bool = False, n_neighborhoods_batch: int | None = None, max_memory_gb: float | None = None, transforms_: dict[int, list[np.ndarray]] | None = None, template_: dict[int, np.ndarray] | None = None, neighborhoods_: SphereNeighborhoods | dict[int, np.ndarray] | None = None, n_voxels_: int | None = None, mask_: nib.Nifti1Image | None = None, backend_: Backend | None = None) -> None
```

Local (neighborhood-based) functional alignment across subjects.

Learns alignment transforms within local neighborhoods (searchlight spheres
or parcels) and applies center-only aggregation to preserve orthogonality.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`spatial_scale` | <code>[str](#str)</code> | Spatial scale, either 'searchlight' (overlapping spheres) or 'roi' (non-overlapping parcels). Defaults to 'searchlight'. | <code>'searchlight'</code>
`method` | <code>[str](#str)</code> | Alignment method, one of 'procrustes', 'srm', or 'hyperalignment'. Defaults to 'procrustes'. | <code>'procrustes'</code>
`radius_mm` | <code>[float](#float)</code> | Sphere radius in millimeters for the searchlight scale. Defaults to 10.0. | <code>10.0</code>
`roi_mask` | <code>[Nifti1Image](#Nifti1Image) \| None</code> | Parcellation image for the ROI scale. Required if `spatial_scale='roi'`. Defaults to None. | <code>None</code>
`n_features` | <code>[int](#int) \| None</code> | Number of features for SRM. None uses full Procrustes (preserves dims). Defaults to None. | <code>None</code>
`n_iter` | <code>[int](#int)</code> | Number of iterations for alignment refinement. Defaults to 3. | <code>3</code>
`aggregation` | <code>[str](#str)</code> | Aggregation method: 'center' (center-only, preserves orthogonality) or 'all'. Defaults to 'center'. | <code>'center'</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization mode. None runs single-threaded numpy, 'cpu' uses joblib CPU parallelization, and 'gpu' uses PyTorch. GPU acceleration applies only to `method='procrustes'`; requesting 'gpu' with the 'srm' or 'hyperalignment' methods raises `NotImplementedError` (an explicit GPU request never silently runs on CPU). Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of jobs for CPU parallelization. Defaults to -1. | <code>-1</code>
`progress_bar` | <code>[bool](#bool)</code> | Whether to display tqdm progress bars during fit and transform. Defaults to False. | <code>False</code>
`n_neighborhoods_batch` | <code>[int](#int) \| None</code> | Number of neighborhoods to process per batch on the GPU. None auto-calculates a batch size from `max_memory_gb`. Defaults to None. | <code>None</code>
`max_memory_gb` | <code>[float](#float) \| None</code> | Explicit memory budget (in GB) used to auto-size GPU batches when `n_neighborhoods_batch` is None. None (default) measures the device's available memory. | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`transforms_` | <code>[dict](#dict)[[int](#int), [list](#list)[[ndarray](#numpy.ndarray)]]</code> | Per-neighborhood transforms. Keys are center voxel indices, values are lists of transform matrices (one per subject).
`template_` | <code>[dict](#dict)[[int](#int), [ndarray](#numpy.ndarray)]</code> | Per-neighborhood templates used for alignment.
`neighborhoods_` | <code>[SphereNeighborhoods](#nltools.data.braindata.neighborhoods.SphereNeighborhoods) \| [dict](#dict)</code> | Computed neighborhoods (searchlight or roi).
`n_voxels_` | <code>[int](#int)</code> | Total number of voxels in the mask.
`mask_` | <code>[Nifti1Image](#Nifti1Image)</code> | Brain mask used for fitting.

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Fit local alignment on multi-subject data.
[`fit_transform`](#algorithms-fit-transform) | Fit alignment and transform data in one step.
[`transform`](#algorithms-transform) | Apply local transforms to data.



**Examples:**

```pycon
>>> import numpy as np
>>> import nibabel as nib
>>> from nltools.algorithms.alignment import LocalAlignment
>>> # Create synthetic multi-subject data (voxels, samples)
>>> data = [np.random.randn(1000, 100) for _ in range(5)]
>>> # Build a mask whose nonzero voxels match the 1000-voxel data
>>> mask = nib.Nifti1Image(np.ones((10, 10, 10), dtype=np.int8), np.eye(4))
>>> la = LocalAlignment(spatial_scale='searchlight', method='procrustes', radius_mm=10.0)
>>> la.fit(data, mask)
>>> aligned = la.transform(data)
```

<details class="note" open markdown="1">
<summary>Note</summary>

Based on Bazeille et al. 2021 "An empirical evaluation of functional
alignment using inter-subject decoding". Center-only aggregation is
used to preserve local orthogonality of transforms.

</details>

##### Methods

###### `fit`

```python
fit(data: list[np.ndarray], mask: nib.Nifti1Image) -> LocalAlignment
```

Fit local alignment on multi-subject data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | List of subject data arrays, each shape (n_voxels, n_samples). Subjects can have different numbers of samples - the underlying alignment methods (SRM, HyperAlignment) handle this via zero-padding. | *required*
`mask` | <code>[Nifti1Image](#Nifti1Image)</code> | Brain mask defining the voxel space. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`LocalAlignment` | <code>[LocalAlignment](#nltools.algorithms.alignment.local.LocalAlignment)</code> | The fitted alignment model (`self`).

(algorithms-fit-transform)=
###### `fit_transform`

```python
fit_transform(data: list[np.ndarray], mask: nib.Nifti1Image) -> list[np.ndarray]
```

Fit alignment and transform data in one step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | List of subject data arrays, each shape (n_voxels, n_samples). | *required*
`mask` | <code>[Nifti1Image](#Nifti1Image)</code> | Brain mask defining the voxel space. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | list[np.ndarray]: Aligned data for each subject.

###### `transform`

```python
transform(data: list[np.ndarray]) -> list[np.ndarray]
```

Apply local transforms to data.

For the searchlight scale with center-only aggregation: each voxel uses
the transform from the neighborhood where it was the center.

For the roi scale: all voxels in each parcel use the same transform.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | List of subject data arrays, each shape (n_voxels, n_samples). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | list[np.ndarray]: Aligned data for each subject, each shape (n_voxels, n_samples).

(algorithms-srm)=
#### `SRM`

```python
SRM(*, n_iter: int = 10, features: int = 50, rand_seed: int = 0) -> None
```

Bases: <code>[BaseEstimator](#sklearn.base.BaseEstimator)</code>, <code>[TransformerMixin](#sklearn.base.TransformerMixin)</code>

Probabilistic Shared Response Model (SRM).

Given multi-subject data, factorize it as a shared response S among all
subjects and an orthogonal transform W per subject:

$$
X_i \approx W_i S, \forall i=1 \dots N
$$

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_iter` | <code>int, default=10</code> | Number of iterations to run the algorithm. | <code>10</code>
`features` | <code>int, default=50</code> | Number of features to compute. | <code>50</code>
`rand_seed` | <code>int, default=0</code> | Seed for initializing the random number generator. | <code>0</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`w_` | <code>list of array, element i has shape=[voxels_i, features]</code> | The orthogonal transforms (mappings) for each subject.
`s_` | <code>array, shape=[features, samples]</code> | The shared response.
`sigma_s_` | <code>array, shape=[features, features]</code> | The covariance of the shared response Normal distribution.
`mu_` | <code>list of array, element i has shape=[voxels_i]</code> | The voxel means over the samples for each subject.
`rho2_` | <code>array, shape=[subjects]</code> | The estimated noise variance $\rho_i^2$ for each subject
`random_state_` | <code>`RandomState`</code> | Random number generator initialized using rand_seed

<details class="note" open markdown="1">
<summary>Note</summary>

The number of voxels may be different between subjects. However, the
number of samples must be the same across subjects.

The probabilistic Shared Response Model is approximated using the
Expectation Maximization (EM) algorithm proposed in **Chen2015**. The
implementation follows the optimizations published in **Anderson2016**.

This is a single node version.

The run-time complexity is $O(I (V T K + V K^2 + K^3))$ and the
memory complexity is $O(V T)$ with I - the number of iterations,
V - the sum of voxels from all subjects, T - the number of samples, and
K - the number of features (typically, $V \gg T \gg K$).

</details>

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Compute the probabilistic Shared Response Model.
[`transform`](#algorithms-transform) | Use the model to transform matrix to Shared Response space.
[`transform_subject`](#algorithms-transform-subject) | Transform a new subject using the existing model.



**Examples:**

Basic multi-subject SRM fitting:

```pycon
>>> from nltools.algorithms import SRM
>>> import numpy as np
>>>
>>> # Create sample data (3 subjects)
>>> data = [np.random.randn(100, 50) for _ in range(3)]
>>>
>>> # Fit SRM with CPU parallelization (default)
>>> srm = SRM(n_iter=10, features=50)
>>> srm.fit(data, parallel="cpu", n_jobs=-1)
>>>
>>> # Transform to shared response space
>>> shared_responses = srm.transform(data)
>>>
>>> # Access fitted model components
>>> w = srm.w_  # Subject-specific transforms
>>> s = srm.s_  # Shared response
```

##### Methods

###### `fit`

```python
fit(X: list[np.ndarray], y: Any | None = None, *, parallel: str | None = 'cpu', n_jobs: int = -1, pad_samples: bool = True) -> SRM
```

Compute the probabilistic Shared Response Model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples]</code> | Each element in the list contains the fMRI data of one subject. Subjects can have different numbers of samples if pad_samples=True. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": not implemented -- raises `NotImplementedError` (never a silent CPU fallback) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>
`pad_samples` | <code>[bool](#bool)</code> | If True (default), automatically zero-pad subjects with fewer samples to match the longest subject. This allows fitting SRM on data with unequal numbers of time points across subjects. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`self` | <code>[SRM](#nltools.algorithms.alignment.srm.SRM)</code> | Fitted model

###### `transform`

```python
transform(X: list[np.ndarray], y: Any | None = None, *, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray | None]
```

Use the model to transform matrix to Shared Response space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples_i]</code> | Each element in the list contains the fMRI data of one subject. Note that number of voxels and samples can vary across subjects. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used (as it is unsupervised learning) | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": not implemented -- raises `NotImplementedError` (never a silent CPU fallback) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`s` | <code>list of 2D arrays, element i has shape=[features_i, samples_i]</code> | Shared responses from input data (X)

###### `transform_subject`

```python
transform_subject(X: np.ndarray) -> np.ndarray
```

Transform a new subject using the existing model.

The subject is assumed to have received equivalent stimulation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>2D array, shape=[voxels, timepoints]</code> | The fMRI data of the new subject. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`w` | <code>2D array, shape=[voxels, features]</code> | Orthogonal mapping `W_{new}` for new subject



### Methods

(algorithms-align)=
#### `align`

```python
align(data, method = 'deterministic_srm', n_features = None, axis = 0, *args, **kwargs)
```

Align subject data into a common response model.

This function is a convenience wrapper around `HyperAlignment` and `SRM` classes.

Can be used to hyperalign source data to target data using
Hyperalignment from Dartmouth (i.e., procrustes transformation; see
nltools.algorithms.procrustes) or Shared Response Model from Princeton (see
nltools.algorithms.srm). (see nltools.data.BrainData.align for aligning
a single Brain object to another). Common Model is shared response
model or centered target data. Transformed data can be back projected to
original data using Tranformation matrix. Inputs must be a list of BrainData
instances or numpy arrays (observations by features).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (list) A list of BrainData objects | *required*
`method` |  | (str) alignment method to use ['probabilistic_srm','deterministic_srm','procrustes'] | <code>'deterministic_srm'</code>
`n_features` |  | (int) number of features to align to common space. If None then will select number of voxels | <code>None</code>
`axis` |  | (int) axis to align on | <code>0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (dict) a dictionary containing a list of transformed subject matrices, a list of transformation matrices, the shared response matrix, and the intersubject correlation of the shared responses

**Examples:**

- Hyperalign using procrustes transform:
    >>> out = align(data, method='procrustes')
- Align using shared response model:
    >>> out = align(data, method='probabilistic_srm', n_features=None)
- Project aligned data into original data:
    >>> original_data = [np.dot(t.data,tm.T) for t,tm in zip(out['transformed'], out['transformation_matrix'])]

(algorithms-align-states)=
#### `align_states`

```python
align_states(reference, target, *, metric = 'correlation', return_index = False, replace_zero_variance = False)
```

Align state weight maps by minimizing pairwise distance between group states.

This function uses the Hungarian algorithm for state alignment, which is
different from aligning multiple subjects' data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`reference` |  | (np.array) reference pattern x state matrix | *required*
`target` |  | (np.array) target pattern x state matrix to align to reference | *required*
`metric` |  | (str) distance metric to use | <code>'correlation'</code>
`return_index` |  | (bool) return index if True, return remapped data if False | <code>False</code>
`replace_zero_variance` |  | (bool) transform a vector with zero variance to random numbers from a uniform distribution.                     Useful for when using correlation as a distance metric to avoid NaNs. | <code>False</code>

Returns:
    If ``return_index=False`` (default): ``target[:, remapping]``, a single
    ndarray of the target's columns reordered to match the reference,
    oriented pattern x state (same shape as ``target``).
    If ``return_index=True``: the remapping index array (ndarray) that
    reorders the target's state columns.

(algorithms-calc-bpm)=
#### `calc_bpm`

```python
calc_bpm(beat_interval, sampling_freq)
```

Calculate instantaneous BPM from beat to beat interval.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`beat_interval` |  | (int) number of samples in between each beat             (typically R-R Interval) | *required*
`sampling_freq` |  | (float) sampling frequency in Hz | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`bpm` |  | (float) beats per minute for time interval

(algorithms-circle-shift)=
#### `circle_shift`

```python
circle_shift(data: np.ndarray, shift_amount: int | np.ndarray | None = None, random_state: int | np.random.RandomState | None = None) -> np.ndarray
```

Circular shift for time-series data.

Performs a circular shift that preserves autocorrelation structure.
Useful for permutation tests on autocorrelated time series (e.g., fMRI).
For 1D data, shifts by a single amount. For 2D data, shifts each
feature (column) independently.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Time series data, shape (n_samples,) or (n_samples, n_features) | *required*
`shift_amount` | <code>[int](#int) \| [ndarray](#numpy.ndarray) \| None</code> | Shift amount(s). If None, random shift is used. For 1D: int specifying shift amount For 2D: array of length n_features with shift per feature | <code>None</code>
`random_state` | <code>[int](#int) \| [RandomState](#numpy.random.RandomState) \| None</code> | Random seed for reproducibility (if shift_amount is None) | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Circularly shifted data with same shape as input

**Examples:**

```pycon
>>> x = np.array([1, 2, 3, 4, 5])
>>> circle_shift(x, shift_amount=2)
array([4, 5, 1, 2, 3])
```

```pycon
>>> X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
>>> circle_shift(X, shift_amount=np.array([1, 2]))
array([[ 4, 30],
       [ 1, 40],
       [ 2, 10],
       [ 3, 20]])
```

(algorithms-compute-multivariate-similarity)=
#### `compute_multivariate_similarity`

```python
compute_multivariate_similarity(y, X, method = 'ols', tail = 2)
```

Compute multivariate similarity via OLS regression.

This is the functional core implementation for multivariate similarity computation.
Used by BrainData.multivariate_similarity() to delegate computation to the functional core.

Predicts spatial distribution of y from linear combination of X columns.
Computes OLS regression statistics including beta coefficients, t-statistics,
p-values, and residuals.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` | <code>[ndarray](#numpy.ndarray)</code> | Target data, shape (n_features,) - single image | *required*
`X` | <code>[ndarray](#numpy.ndarray)</code> | Predictor data, shape (n_features, n_predictors) where first column should be intercept (ones) if intercept is desired. If X does not include intercept, an intercept will be added automatically. | *required*
`method` | <code>[str](#str)</code> | Regression method (currently only 'ols' supported) | <code>'ols'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | Dictionary with keys: - 'beta': Regression coefficients including intercept, shape (n_predictors+1,) - 't': t-statistics, shape (n_predictors+1,) - 'p': p-values, shape (n_predictors+1,) - 'df': Degrees of freedom (int) - 'sigma': Residual standard deviation (float) - 'residual': Residuals, shape (n_features,)

**Examples:**

```pycon
>>> y = np.random.randn(100)
>>> X = np.random.randn(100, 5)
>>> result = compute_multivariate_similarity(y, X, method='ols')
>>> 'beta' in result
True
>>> result['beta'].shape
(6,)  # 5 predictors + intercept
```

(algorithms-compute-similarity)=
#### `compute_similarity`

```python
compute_similarity(data1, data2, metric = 'correlation')
```

Compute similarity between two data arrays.

This is the functional core implementation for similarity computation.
Used by BrainData.similarity() to delegate computation to the functional core.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | First data array, shape (n_samples1, n_features) | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Second data array, shape (n_samples2, n_features) | *required*
`metric` | <code>[str](#str)</code> | Type of similarity metric - 'correlation' or 'pearson': Pearson correlation - 'spearman' or 'rank_correlation': Spearman rank correlation - 'dot_product': Dot product - 'cosine': Cosine similarity | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
 | np.ndarray: Similarity matrix or vector - If data1.shape[0] == 1 and data2.shape[0] == 1: scalar - If data1.shape[0] == 1 or data2.shape[0] == 1: 1D array - Otherwise: 2D array shape (n_samples1, n_samples2)

**Examples:**

```pycon
>>> data1 = np.random.randn(10, 100)
>>> data2 = np.random.randn(5, 100)
>>> sim = compute_similarity(data1, data2, metric='correlation')
>>> sim.shape
(10, 5)
```

(algorithms-correlation-permutation-test)=
#### `correlation_permutation_test`

```python
correlation_permutation_test(data1: np.ndarray, data2: np.ndarray, *, n_permute: int = 5000, metric: str = 'pearson', tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Correlation permutation test.

Tests whether the correlation between data1 and data2 is significantly
different from zero by randomly permuting data1 and computing correlations.

Assumption: Observations are independent (i.i.d.). For autocorrelated time
series, use timeseries_correlation_permutation_test with circle_shift or
phase_randomize methods instead.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | Data to permute - shape (n_samples,) for single feature - shape (n_samples, n_features) for multi-feature | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Data to correlate with - shape (n_samples,) for single feature - shape (n_samples, n_features) for multi-feature | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000) | <code>5000</code>
`metric` | <code>[str](#str)</code> | Correlation metric (default: 'pearson') - 'pearson': Pearson correlation (linear relationships) - 'spearman': Spearman rank correlation (monotonic relationships) - 'kendall': Kendall tau rank correlation (ordinal association, robust to ties) | <code>'pearson'</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type — 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction) - 2 | 'two': Two-tailed test (r != 0) - 1 | 'one': One-tailed (r > 0; negate one variable for the other   direction). The fixed direction keeps MCP correction valid. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show a progress bar over permutations (default: False) | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'correlation' (float or np.ndarray): Observed correlation(s) - 'p' (float or np.ndarray): P-value(s) - 'null_dist' (np.ndarray): Null distribution (if return_null=True) - 'device' (str): Parallelization method used

**Examples:**

```pycon
>>> # Single feature (default CPU parallelization)
>>> x = np.random.randn(100)
>>> y = x + np.random.randn(100) * 0.5  # Correlated
>>> result = correlation_permutation_test(x, y, n_permute=5000)
>>> result['correlation']
0.85
>>> result['p']
0.001
```

```pycon
>>> # Multi-feature (2D arrays)
>>> data1 = np.random.randn(100, 10)  # 100 samples, 10 features
>>> data2 = data1 + np.random.randn(100, 10) * 0.3  # Correlated
>>> result = correlation_permutation_test(data1, data2, n_permute=5000)
>>> result['correlation'].shape
(10,)
>>> result['p'].shape
(10,)
```

```pycon
>>> # GPU acceleration
>>> result = correlation_permutation_test(data1, data2, n_permute=5000, device='gpu')
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (device='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
    - Pearson: Fully vectorized across all features (5-20× speedup for multi-feature)
    - Spearman: GPU rank transform (average ties) + vectorized Pearson on ranks
    - Kendall: tie-corrected tau-b via pre-computed pairwise sign tensors;
      O(n²) memory per permutation, so batches are sized accordingly
- Single-threaded (device=None): Use for small problems or debugging
- For multi-feature data, each feature pair tested independently
- Kendall is O(n^2) complexity, slower than Pearson/Spearman for large samples

</details>

(algorithms-distance-correlation)=
#### `distance_correlation`

```python
distance_correlation(x: np.ndarray, y: np.ndarray, bias_corrected: bool = True, ttest: bool = False) -> dict
```

Compute the distance correlation between 2 arrays to test for multivariate dependence (linear or non-linear).

Arrays must match on their first dimension. It's almost always preferable to compute the bias_corrected
version which can also optionally perform a ttest. This ttest operates on a statistic thats ~dcorr^2
and will be also returned.

Explanation:
Distance correlation involves computing the normalized covariance of two centered euclidean distance
matrices. Each distance matrix is the euclidean distance between rows (if x or y are 2d) or scalars
(if x or y are 1d). Each matrix is centered prior to computing the covariance either using double-centering
or u-centering, which corrects for bias as the number of dimensions increases. U-centering is almost always
preferred in all cases. It also permits inference of the normalized covariance between each distance matrix
using a one-tailed directional t-test. (Szekely & Rizzo, 2013). While distance correlation is normally
bounded between 0 and 1, u-centering can produce negative estimates, which are never significant.

Validated against the dcor and dcor.ttest functions in the 'energy' R package and the
dcor.distance_correlation, dcor.udistance_correlation_sqr, and dcor.independence.distance_correlation_t_test
functions in the dcor Python package.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` | <code>[ndarray](#ndarray)</code> | 1d or 2d numpy array of observations by features | *required*
`y` | <code>[ndarray](#ndarray)</code> | 1d or 2d numpy array of observations by features | *required*
`bias_corrected` | <code>[bool](#bool)</code> | if false use double-centering which produces a biased-estimate that converges to 1 as the number of dimensions increase. Otherwise used u-centering to correct this bias. **Note** this must be True if ttest=True; default True | <code>True</code>
`ttest` | <code>[bool](#bool)</code> | perform a ttest using the bias_corrected distance correlation; default False | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`results` | <code>[dict](#dict)</code> | dictionary of results (correlation, t, p, and df.) Optionally, covariance, x variance, and y variance

**Examples:**

```pycon
>>> import numpy as np
>>> x = np.random.randn(20, 3)
>>> y = x + np.random.randn(20, 3) * 0.1  # Strongly correlated
>>> result = distance_correlation(x, y, bias_corrected=True)
>>> 'dcorr' in result
True
>>> 0 <= result['dcorr'] <= 1
True
```

(algorithms-double-center)=
#### `double_center`

```python
double_center(mat: np.ndarray) -> np.ndarray
```

Double center a 2d array.

Double-centering subtracts row means, column means, and adds the grand mean.
This centers both rows and columns around zero.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mat` | <code>[ndarray](#ndarray)</code> | 2d numpy array | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`mat` | <code>[ndarray](#ndarray)</code> | double-centered version of input

**Examples:**

```pycon
>>> mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
>>> result = double_center(mat)
>>> np.allclose(result.mean(axis=0), 0)
True
>>> np.allclose(result.mean(axis=1), 0)
True
```

(algorithms-downsample)=
#### `downsample`

```python
downsample(data, *, sampling_freq = None, target = None, target_type = 'samples', method = 'mean')
```

Downsample a Polars DataFrame/Series to a new target frequency or number of samples using averaging.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pl.DataFrame, pl.Series) data to downsample | *required*
`sampling_freq` |  | (float) Sampling frequency of data in hertz | <code>None</code>
`target` |  | (float) downsampling target | <code>None</code>
`target_type` |  | type of target can be [samples,seconds,hz] | <code>'samples'</code>
`method` |  | (str) type of downsample method ['mean','median'],     default: mean | <code>'mean'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (pl.DataFrame, pl.Series) downsampled data (same type as input)

(algorithms-fdr)=
#### `fdr`

```python
fdr(p, q = 0.05)
```

Determine an FDR threshold for an array of p-values.

Uses the desired false discovery rate ``q``. Written by Tal Yarkoni.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`p` |  | (np.array) vector of p-values | *required*
`q` |  | (float) false discovery rate level | <code>0.05</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`fdr_p` |  | (float) p-value threshold based on independence or positive     dependence

(algorithms-find-spikes)=
#### `find_spikes`

```python
find_spikes(data, global_spike_cutoff = 3, diff_spike_cutoff = 3, *, TR: float | None = None, sampling_freq: float | None = None)
```

Identify spikes (motion artifacts, intensity outliers) in 4D fMRI data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | BrainData or nibabel instance | *required*
`global_spike_cutoff` |  | (int, None) cutoff in std-deviations for spikes in the per-TR global signal. None to skip. | <code>3</code>
`diff_spike_cutoff` |  | (int, None) cutoff in std-deviations for spikes in the per-TR mean absolute frame-to-frame difference. None to skip. | <code>3</code>
`TR` | <code>[float](#float) \| None</code> | Repetition time in seconds. Sets the returned DesignMatrix's sampling_freq for downstream `.append(...)` / `.convolve()`. Pass exactly one of `TR` or `sampling_freq`. | <code>None</code>
`sampling_freq` | <code>[float](#float) \| None</code> | Sampling frequency in Hz (= 1/TR). See `TR`. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` |  | one indicator column per detected spike TR, named
 |  | ``.nl_global_spike{n}`` / ``.nl_diff_spike{n}`` in the reserved
 |  | namespace for generated columns (see `RESERVED_PREFIX`), with all
 |  | spike columns pre-marked as confounds. The two detectors run
 |  | independently, so a single bad volume is routinely caught by both;
 |  | those detections are bitwise-identical one-hot columns, and only one
 |  | is kept (the ``.nl_global_spike*`` name, a deterministic tie-break —
 |  | the column values are the same either way). Row position is the time
 |  | axis (no separate `TR` index column — that was a pandas-era
 |  | artifact). When `TR` / `sampling_freq` aren't provided the DM has
 |  | `sampling_freq=None`; you can still `.append()` it onto a DM that
 |  | does have one.

(algorithms-fisher-r-to-z)=
#### `fisher_r_to_z`

```python
fisher_r_to_z(r)
```

Use Fisher transformation to convert correlation to z score.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`r` |  | correlation coefficient(s) | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`z` |  | Fisher z-transformed correlation(s)

(algorithms-fisher-z-to-r)=
#### `fisher_z_to_r`

```python
fisher_z_to_r(z)
```

Convert Fisher z back to a correlation coefficient.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`z` |  | Fisher z-transformed value(s) | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`r` |  | correlation coefficient(s)

(algorithms-glover-dispersion-derivative)=
#### `glover_dispersion_derivative`

```python
glover_dispersion_derivative(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Implement the Glover dispersion derivative :term:`HRF` model.

Parameters
----------
t_r : `float`
    :term:`Repetition time<TR>`, in seconds (sampling period).

<details class="oversampling-" open markdown="1">
<summary>`int`, default=50</summary>

Temporal oversampling factor in seconds.

</details>

<details class="time_length-" open markdown="1">
<summary>`float`, default=32.0</summary>

:term:`HRF` kernel length, in seconds.

</details>

<details class="onset-" open markdown="1">
<summary>`float`, default=0.0</summary>

Onset of the response in seconds.

</details>

Returns
-------
dhrf : array of shape (length / t_r * oversampling), dtype=float
      dhrf sampling on the oversampled time grid

Examples
--------
>>> import numpy as np
>>> from nilearn.glm.first_level import glover_dispersion_derivative
>>> ddhrf = glover_dispersion_derivative(
...     t_r=2.0, oversampling=1, time_length=20.0
... )
>>> np.round(ddhrf, 3).tolist()
[0.0, -0.0, -0.373, 0.282, 0.295, -0.04, -0.094, -0.048, -0.017, -0.005]

(algorithms-glover-hrf)=
#### `glover_hrf`

```python
glover_hrf(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Implement the Glover :term:`HRF` model.

Parameters
----------
t_r : `float`
    :term:`Repetition time<TR>`, in seconds (sampling period).

<details class="oversampling-" open markdown="1">
<summary>`int`, default=50</summary>

Temporal oversampling factor.

</details>

<details class="time_length-" open markdown="1">
<summary>`float`, default=32.0</summary>

:term:`HRF` kernel length, in seconds.

</details>

<details class="onset-" open markdown="1">
<summary>`float`, default=0.0</summary>

Onset of the response.

</details>

Returns
-------
hrf : array of shape (length / t_r * oversampling, dtype=float)
     :term:`HRF` sampling on the oversampled time grid.

Examples
--------
>>> import numpy as np
>>> from nilearn.glm.first_level import glover_hrf
>>> hrf = glover_hrf(t_r=2.0, oversampling=1, time_length=20.0)
>>> np.round(hrf, 3).tolist()
[0.0, 0.0, 0.226, 0.741, 0.5, 0.037, -0.181, -0.176, -0.103, -0.045]

(algorithms-glover-time-derivative)=
#### `glover_time_derivative`

```python
glover_time_derivative(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Implement the Glover time derivative :term:`HRF` (dhrf) model.

Parameters
----------
t_r : `float`
    :term:`Repetition time<TR>`, in seconds (sampling period).

<details class="oversampling-" open markdown="1">
<summary>`int`, default=50</summary>

Temporal oversampling factor.

</details>

<details class="time_length-" open markdown="1">
<summary>`float`, default=32.0</summary>

:term:`HRF` kernel length, in seconds.

</details>

<details class="onset-" open markdown="1">
<summary>`float`, default=0.0</summary>

Onset of the response.

</details>

Returns
-------
dhrf : array of shape (length / t_r), dtype=float
      dhrf sampling on the provided grid

Examples
--------
>>> import numpy as np
>>> from nilearn.glm.first_level import glover_time_derivative
>>> dhrf = glover_time_derivative(
...     t_r=2.0, oversampling=1, time_length=20.0
... )
>>> np.round(dhrf, 3).tolist()
[0.0, 0.0, 0.267, 0.076, -0.215, -0.168, -0.039, 0.027, 0.033, 0.019]

(algorithms-holm-bonf)=
#### `holm_bonf`

```python
holm_bonf(p, alpha = 0.05)
```

Compute Holm-Bonferroni-corrected p-values.

This step-down procedure applies iteratively less correction to the highest
p-values. It is a bit more conservative than FDR, but much more powerful than
vanilla Bonferroni correction.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`p` |  | (np.array) vector of p-values | *required*
`alpha` |  | (float) alpha level | <code>0.05</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`bonf_p` |  | (float) p-value threshold based on bonferroni     step-down procedure

(algorithms-isc)=
#### `isc`

```python
isc(data, *, n_samples = 5000, summary = 'median', method = 'bootstrap', ci_percentile = 95, exclude_self_corr = True, tail = 2, metric = 'correlation', return_null = False, n_jobs = -1, random_state = None, progress_bar = False)
```

Compute pairwise intersubject correlation from observations by subjects array.

This function computes pairwise intersubject correlations (ISC) using the median as recommended by Chen
et al., 2016). However, if the mean is preferred, we compute the mean correlation after performing
the fisher r-to-z transformation and then convert back to correlations to minimize artificially
inflating the correlation values.

There are currently three different methods to compute p-values. These include the classic methods for
computing permuted time-series by either circle-shifting the data or phase-randomizing the data
(see Lancaster et al., 2018). These methods create random surrogate data while preserving the temporal
autocorrelation inherent to the signal. By default, we use the subject-wise bootstrap method from
Chen et al., 2016. Instead of recomputing the pairwise ISC using circle_shift or phase_randomization methods,
this approach uses the computationally more efficient method of bootstrapping the subjects
and computing a new pairwise similarity matrix with randomly selected subjects with replacement.
If the same subject is selected multiple times, we set the perfect correlation to a nan with
(exclude_self_corr=True). We compute the p-values using the percentile method using the same
method in Brainiak.

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C., Israel, R. B.,
& Cox, R. W. (2016). Untangling the relatedness among correlations, part I:
nonparametric approaches to inter-subject correlation analysis at the group level.
NeuroImage, 142, 248-259.

Hall, P., & Wilson, S. R. (1991). Two guidelines for bootstrap hypothesis testing.
Biometrics, 757-762.

Lancaster, G., Iatsenko, D., Pidde, A., Ticcinelli, V., & Stefanovska, A. (2018).
Surrogate data for hypothesis testing of physical systems. Physics Reports, 748, 1-60.

This function is a wrapper around `isc_permutation_test` from the inference module,
which provides optimized implementations with CPU-parallel and GPU acceleration support.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pd.DataFrame, np.array) observations by subjects where isc is computed across subjects | *required*
`n_samples` |  | (int) number of random samples/bootstraps | <code>5000</code>
`summary` |  | (str) type of isc summary statistic ['mean','median'] (default: median) | <code>'median'</code>
`method` |  | (str) method to compute p-values ['bootstrap', 'circle_shift','phase_randomize'] (default: bootstrap) | <code>'bootstrap'</code>
`ci_percentile` |  | (int) confidence-interval width in percent for the bootstrap CI (default: 95) | <code>95</code>
`exclude_self_corr` |  | (bool) set self-correlations (same subject bootstrapped twice) to nan (default: True) | <code>True</code>
`tail` |  | (int | str) 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction) | <code>2</code>
`metric` |  | (str) pairwise distance metric. See sklearn's pairwise_distances for valid inputs (default: correlation) | <code>'correlation'</code>
`return_null` |  | (bool) Return the permutation distribution along with the p-value; default False | <code>False</code>
`n_jobs` |  | (int) The number of CPUs to use to do the computation. -1 means all CPUs. | <code>-1</code>
`random_state` |  | (int, np.random.RandomState, or None) seed or generator for the resampling; default None | <code>None</code>
`progress_bar` |  | (bool) If True, display a progress bar. Default False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`stats` |  | (dict) dictionary of permutation results ['isc', 'p', 'ci', 'null_dist']

(algorithms-isc-group)=
#### `isc_group`

```python
isc_group(group1, group2, *, n_samples = 5000, summary = 'median', method = 'permute', ci_percentile = 95, exclude_self_corr = True, return_null = False, tail = 2, metric = 'correlation', n_jobs = -1, random_state = None, progress_bar = False)
```

Compute difference in intersubject correlation between groups.

This function computes pairwise intersubject correlations (ISC) using the median as recommended by Chen
et al., 2016). However, if the mean is preferred, we compute the mean correlation after performing
the fisher r-to-z transformation and then convert back to correlations to minimize artificially
inflating the correlation values.

There are currently two different methods to compute p-values. By default, we use the subject-wise permutation
method recommended Chen et al., 2016. This method combines the two groups and computes pairwise similarity both
within and between the groups. Then the group labels are permuted and the mean difference between the two groups
are recomputed to generate a null distribution. The second method uses subject-wise bootstrapping, where a new
pairwise similarity matrix with randomly selected subjects with replacement is created separately for each group
and the ISC difference between these groups is used to generate a null distribution. If the same subject is
selected multiple times, we set the perfect correlation to a nan with (exclude_self_corr=True). We compute the
p-values using the percentile method (Hall & Wilson, 1991).

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C., Israel, R. B.,
& Cox, R. W. (2016). Untangling the relatedness among correlations, part I:
nonparametric approaches to inter-subject correlation analysis at the group level.
NeuroImage, 142, 248-259.

Hall, P., & Wilson, S. R. (1991). Two guidelines for bootstrap hypothesis testing.
Biometrics, 757-762.

This function is a thin wrapper around `isc_group_permutation_test` from the inference
module (which provides optimized CPU parallelization and optional GPU acceleration),
pinning the classic pairwise behavior and the `n_samples` vocabulary.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`group1` |  | (pd.DataFrame, np.array) observations by subjects where isc is computed across subjects | *required*
`group2` |  | (pd.DataFrame, np.array) observations by subjects where isc is computed across subjects | *required*
`n_samples` |  | (int) number of samples for permutation or bootstrapping | <code>5000</code>
`summary` |  | (str) type of isc summary statistic ['mean','median'] (default: median) | <code>'median'</code>
`method` |  | (str) method to compute p-values ['permute', 'bootstrap'] (default: permute) | <code>'permute'</code>
`ci_percentile` |  | (float) confidence interval percentile (default: 95) | <code>95</code>
`exclude_self_corr` |  | (bool) exclude self-correlations in bootstrap (default: True) | <code>True</code>
`return_null` |  | (bool) Return the permutation distribution along with the p-value; default False | <code>False</code>
`tail` |  | (int | str) 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction) | <code>2</code>
`metric` |  | (str) pairwise distance metric. See sklearn's pairwise_distances for valid inputs (default: correlation) | <code>'correlation'</code>
`n_jobs` |  | (int) The number of CPUs to use to do the computation. -1 means all CPUs. | <code>-1</code>
`random_state` |  | (int or RandomState) Random seed for reproducibility | <code>None</code>
`progress_bar` |  | (bool) If True, display a progress bar. Default False. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`stats` |  | (dict) dictionary of permutation results with keys: - 'isc_group_difference': Observed ISC difference (float or array) - 'p': P-value (float or array) - 'ci': Confidence interval tuple (lower, upper) - 'null_dist': Null distribution (if return_null=True)

(algorithms-isc-group-permutation-test)=
#### `isc_group_permutation_test`

```python
isc_group_permutation_test(group1: np.ndarray, group2: np.ndarray, *, n_permute: int = 5000, summary: Literal['median', 'mean'] = 'median', method: Literal['permute', 'bootstrap'] = 'permute', summary_statistic: Literal['leave-one-out', 'pairwise'] = 'pairwise', ci_percentile: float = 95, tail: int | str = 2, device: Literal['cpu', 'gpu'] | None = 'cpu', n_jobs: int = -1, random_state: int | None = None, return_null: bool = False, progress_bar: bool = False, exclude_self_corr: bool = True, metric: str = 'correlation') -> dict[str, Any]
```

Compute ISC difference between groups with permutation testing.

Supports both subject-wise permutation and bootstrap methods with efficient
CPU-parallel and optional GPU acceleration. Follows the statistical methods
from Chen et al. (2016) for correct group comparison inference.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`group1` | <code>[ndarray](#numpy.ndarray)</code> | First group data with one of the following shapes: - (n_observations, n_subjects1): Single feature - (n_observations, n_subjects1, n_voxels): Voxel-wise | *required*
`group2` | <code>[ndarray](#numpy.ndarray)</code> | Second group data with one of the following shapes: - (n_observations, n_subjects2): Single feature - (n_observations, n_subjects2, n_voxels): Voxel-wise | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations/bootstrap iterations. Defaults to 5000. | <code>5000</code>
`summary` | <code>[Literal](#typing.Literal)['median', 'mean']</code> | Summary statistic for aggregating ISC values: - 'median': Direct median (robust to outliers) - 'mean': Fisher z-transformed mean (unbiased averaging) Defaults to 'median'. | <code>'median'</code>
`method` | <code>[Literal](#typing.Literal)['permute', 'bootstrap']</code> | Resampling method for p-value computation: - 'permute': Subject-wise permutation (combines groups, permutes labels) - 'bootstrap': Subject-wise bootstrap (resamples within each group) Defaults to 'permute'. | <code>'permute'</code>
`summary_statistic` | <code>[Literal](#typing.Literal)['leave-one-out', 'pairwise']</code> | ISC computation method: - 'pairwise': Average all pairwise correlations - 'leave-one-out': Correlate each subject with mean of others Defaults to 'pairwise'. | <code>'pairwise'</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95 for 95% CI). Defaults to 95. | <code>95</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Two-tailed (2 or 'two', default) or one-tailed (1 or 'one', positive direction) p-value. | <code>2</code>
`device` | <code>[Literal](#typing.Literal)['cpu', 'gpu'] \| None</code> | Parallelization method: - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (10-30× speedup for voxel-wise LOO) - None: Single-threaded NumPy (for debugging/small problems) Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when device='cpu'. Defaults to -1. | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, return null distribution in result dict. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during bootstrap/permutation. Defaults to False. | <code>False</code>
`exclude_self_corr` | <code>[bool](#bool)</code> | Mask self-correlations in bootstrap (pairwise only). Defaults to True. | <code>True</code>
`metric` | <code>[str](#str)</code> | Similarity metric for pairwise ISC computation. See sklearn.metrics.pairwise_distances for valid options. Only applies when summary_statistic='pairwise'. Defaults to 'correlation'. | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with the following keys:
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'isc_group_difference': Observed ISC difference (float or array per voxel)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'p': P-value (Phipson-Smyth corrected)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'ci': Confidence interval tuple (lower, upper)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'device': Parallelization method used
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'null_dist': (optional) Bootstrap/permutation distribution

**Examples:**

```pycon
>>> # Single-feature ISC group comparison
>>> group1 = np.random.randn(100, 10)  # 10 subjects
>>> group2 = np.random.randn(100, 10)
>>> result = isc_group_permutation_test(group1, group2, n_permute=1000)
>>> print(f"ISC difference: {result['isc_group_difference']:.3f}, p: {result['p']:.3f}")
```

```pycon
>>> # Voxel-wise ISC group comparison with GPU acceleration
>>> group1_voxels = np.random.randn(100, 10, 5000)  # 5K voxels
>>> group2_voxels = np.random.randn(100, 10, 5000)
>>> result = isc_group_permutation_test(
...     group1_voxels,
...     group2_voxels,
...     summary_statistic='leave-one-out',
...     device='gpu',  # GPU for LOO computation
...     n_permute=5000
... )
>>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")
```

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Permutation method combines groups and permutes labels (Chen et al. 2016)
- Bootstrap method resamples subjects within each group independently
- Bootstrap distribution is centered by subtracting observed difference
- GPU acceleration available for voxel-wise LOO computation

</details>

(algorithms-isc-permutation-test)=
#### `isc_permutation_test`

```python
isc_permutation_test(data: np.ndarray, *, n_permute: int = 5000, summary: Literal['median', 'mean'] = 'median', summary_statistic: Literal['leave-one-out', 'pairwise'] = 'pairwise', method: Literal['bootstrap', 'circle_shift', 'phase_randomize'] = 'bootstrap', ci_percentile: float = 95, tail: int | str = 2, return_null: bool = False, progress_bar: bool = False, exclude_self_corr: bool = True, metric: str = 'correlation', device: Literal['cpu', 'gpu'] | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> dict[str, Any]
```

Compute intersubject correlation with permutation testing.

Supports both leave-one-out and pairwise ISC computation modes with
GPU acceleration for large voxel-wise problems and CPU-parallel
bootstrap resampling.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Data array with one of the following shapes: - (n_observations, n_subjects): Single feature ISC - (n_observations, n_subjects, n_voxels): Voxel-wise ISC | *required*
`n_permute` | <code>[int](#int)</code> | Number of bootstrap iterations or permutations. Defaults to 5000. | <code>5000</code>
`summary` | <code>[Literal](#typing.Literal)['median', 'mean']</code> | Summary statistic to aggregate ISC values. - 'median': Direct median (robust to outliers) - 'mean': Fisher z-transformed mean (unbiased averaging) Defaults to 'median'. | <code>'median'</code>
`summary_statistic` | <code>[Literal](#typing.Literal)['leave-one-out', 'pairwise']</code> | ISC computation method. Options: - 'leave-one-out': Correlate each subject with mean of others. O(n_subjects), unbiased, recommended by Chen et al. 2016. - 'pairwise': Average all pairwise correlations. O(n_subjects²), captures full correlation structure. Note: These methods are statistically different and monotonically but non-linearly related (see Chen et al. 2016, Figure 3). Defaults to 'pairwise'. | <code>'pairwise'</code>
`method` | <code>[Literal](#typing.Literal)['bootstrap', 'circle_shift', 'phase_randomize']</code> | Resampling method for p-value computation: - 'bootstrap': Subject-wise bootstrap (default, Chen et al. 2016) - 'circle_shift': Circular time-series shift (preserves autocorrelation) - 'phase_randomize': FFT phase randomization (preserves power spectrum) Defaults to 'bootstrap'. | <code>'bootstrap'</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95 for 95% CI). Defaults to 95. | <code>95</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Two-tailed (2 or 'two', default) or one-tailed (1 or 'one', positive direction) p-value. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return bootstrap/permutation distribution in result dict. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during bootstrap/permutation. Defaults to False. | <code>False</code>
`exclude_self_corr` | <code>[bool](#bool)</code> | If True, mask self-correlations (perfect correlations from duplicate subjects in bootstrap samples) as NaN. If False, include them in the summary statistic. Only applies when method='bootstrap' and summary_statistic='pairwise'. Defaults to True. | <code>True</code>
`metric` | <code>[str](#str)</code> | Similarity metric for pairwise ISC computation. See sklearn.metrics.pairwise_distances for valid options. Only applies when summary_statistic='pairwise'. For 'correlation', uses optimized np.corrcoef. Other metrics use pairwise_distances. Defaults to 'correlation'. | <code>'correlation'</code>
`device` | <code>[Literal](#typing.Literal)['cpu', 'gpu'] \| None</code> | Parallelization method: - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (10-30× speedup for voxel-wise LOO) - None: Single-threaded NumPy (for debugging/small problems) Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when device='cpu'. Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float) \| None</code> | GPU working-set budget in GB. For the pairwise GPU bootstrap (``device='gpu'``, ``summary_statistic='pairwise'``, ``method='bootstrap'``) this bounds the ``(perm_batch, voxel_chunk, n_subjects, n_subjects)`` resample tensor, chunking voxels and permutations to fit — so whole-brain runs stay within budget. Not used by the LOO or surrogate (circle_shift/phase_randomize) paths. Defaults to 4. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with the following keys:
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'isc': Observed ISC value (float or array per voxel)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'p': P-value (Phipson-Smyth corrected)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'ci': Confidence interval tuple (lower, upper)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'device': Parallelization method used
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'null_dist': (optional) Bootstrap/permutation distribution

**Examples:**

```pycon
>>> # Single-feature ISC
>>> data = np.random.randn(100, 10)  # 100 timepoints, 10 subjects
>>> result = isc_permutation_test(data, n_permute=1000)
>>> print(f"ISC: {result['isc']:.3f}, p: {result['p']:.3f}")
```

```pycon
>>> # Voxel-wise ISC with GPU acceleration
>>> data_voxels = np.random.randn(100, 50, 5000)  # 5K voxels
>>> result = isc_permutation_test(
...     data_voxels,
...     summary_statistic='leave-one-out',
...     device='gpu',  # GPU for LOO computation
...     n_permute=5000
... )
>>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")
```

```pycon
>>> # Compare LOO vs pairwise
>>> result_loo = isc_permutation_test(data, summary_statistic='leave-one-out')
>>> result_pair = isc_permutation_test(data, summary_statistic='pairwise')
>>> print(f"LOO: {result_loo['isc']:.3f}, Pairwise: {result_pair['isc']:.3f}")
```

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Leave-one-out is 20-30× faster than pairwise for large n_subjects
- GPU acceleration helps most for voxel-wise LOO (10-30× speedup)
- Pairwise bootstrap uses correct subject-wise resampling (Chen 2016)
- Bootstrap distribution is centered by subtracting observed ISC

</details>

(algorithms-isfc)=
#### `isfc`

```python
isfc(data, method = 'average', n_jobs = -1)
```

Compute intersubject functional connectivity (ISFC) from a list of observation x feature matrices.

This function uses the leave one out approach to compute ISFC (Simony et al., 2016).
For each subject, compute the cross-correlation between each voxel/roi
with the average of the rest of the subjects data. In other words,
compute the mean voxel/ROI response for all participants except the
target subject. Then compute the correlation between each ROI within
the target subject with the mean ROI response in the group average.

Simony, E., Honey, C. J., Chen, J., Lositsky, O., Yeshurun, Y., Wiesel, A., & Hasson, U. (2016).
Dynamic reconfiguration of the default mode network during narrative comprehension.
Nature communications, 7, 12141.

This function now uses the optimized implementation from the inference module,
which provides efficient cross-correlation computation between matrix columns.
CPU parallelization is available via joblib when n_jobs > 1 or n_jobs=-1.
Each subject's ISFC computation is independent and can be parallelized efficiently.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | list of subject matrices (observations x voxels/rois) | *required*
`method` |  | approach to computing ISFC. 'average' uses leave one out | <code>'average'</code>
`n_jobs` |  | (int) Number of parallel jobs to use. -1 means all available cores.     Default is -1 (parallel execution by default, consistent with other stats functions). | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
 | list of subject ISFC matrices

(algorithms-isps)=
#### `isps`

```python
isps(data, *, sampling_freq = 0.5, low_cut = 0.04, high_cut = 0.07, order = 5, pairwise = False)
```

Compute dynamic intersubject phase synchrony (ISPS) from an observations-by-subjects array.

This function computes the instantaneous intersubject phase synchrony for a single voxel/roi
timeseries. Requires multiple subjects. This method is largely based on that described by Glerean
et al., 2012 and performs a hilbert transform on narrow bandpass filtered timeseries (butterworth)
data to get the instantaneous phase angle. The function returns a dictionary containing the
average phase angle, the average vector length, and parametric p-values computed using the rayleigh test using circular
statistics (Fisher, 1993). If pairwise=True, then it will compute these on the pairwise phase angle differences,
if pairwise=False, it will compute these on the actual phase angles. This is called inter-site phase coupling
or inter-trial phase coupling respectively in the EEG literatures.

This function requires narrow band filtering your data. As a default we use the recommendations
by (Glerean et al., 2012) of .04-.07Hz. This is similar to the "slow-4" band (0.025–0.067 Hz)
described by (Zuo et al., 2010; Penttonen & Buzsáki, 2003), but excludes the .03 band, which has been
demonstrated to contain aliased respiration signals (Birn, 2006).

Birn RM, Smith MA, Bandettini PA, Diamond JB. 2006. Separating respiratory-variation-related
fluctuations from neuronal-activity- related fluctuations in fMRI. Neuroimage 31:1536–1548.

Buzsáki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical networks. Science,
304(5679), 1926-1929.

Fisher, N. I. (1995). Statistical analysis of circular data. cambridge university press.

Glerean, E., Salmi, J., Lahnakoski, J. M., Jääskeläinen, I. P., & Sams, M. (2012).
Functional magnetic resonance imaging phase synchronization as a measure of dynamic
functional connectivity. Brain connectivity, 2(2), 91-101.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pd.DataFrame, np.ndarray) observations x subjects data | *required*
`sampling_freq` |  | (float) sampling freqency of data in Hz | <code>0.5</code>
`low_cut` |  | (float) lower bound cutoff for high pass filter | <code>0.04</code>
`high_cut` |  | (float) upper bound cutoff for low pass filter | <code>0.07</code>
`order` |  | (int) filter order for butterworth bandpass | <code>5</code>
`pairwise` |  | (bool) compute phase angle coherence on pairwise phase angle differences     or on raw phase angle. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
 | dictionary with mean phase angle, vector length, and rayleigh statistic

(algorithms-make-cosine-basis)=
#### `make_cosine_basis`

```python
make_cosine_basis(nsamples, sampling_freq, filter_length, unit_scale = True, drop = 0)
```

Create basis functions for a discrete cosine transform.

Based on the implementation in ``spm_filter`` and ``spm_dctmtx`` because
scipy DCT can only apply transforms but not return the basis functions. Like
SPM, this does not add a constant (i.e. intercept), but does retain the first
basis (i.e. sigmoidal/linear drift).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`nsamples` | <code>[int](#int)</code> | number of observations (e.g. TRs) | *required*
`sampling_freq` | <code>[float](#float)</code> | sampling frequency in hertz (i.e. 1 / TR) | *required*
`filter_length` | <code>[int](#int)</code> | length of filter in seconds | *required*
`unit_scale` | <code>[bool](#bool)</code> | assure that the basis functions are on the normalized range [-1, 1]; default True | <code>True</code>
`drop` | <code>[int](#int)</code> | index of which early/slow bases to drop if any; default is to drop constant (i.e. intercept) like SPM. Unlike SPM, retains first basis (i.e. linear/sigmoidal). Will cumulatively drop bases up to and inclusive of index provided (e.g. 2, drops bases 1 and 2) | <code>0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` | <code>[ndarray](#ndarray)</code> | nsamples x number of basis sets numpy array

(algorithms-matrix-permutation-test)=
#### `matrix_permutation_test`

```python
matrix_permutation_test(data1: np.ndarray, data2: np.ndarray, *, n_permute: int = 5000, metric: str = 'pearson', how: str = 'upper', include_diag: bool = False, tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Matrix permutation test (Mantel test) for correlating two square matrices.

Tests whether the correlation between elements of two matrices is significant
by permuting rows and columns of one matrix symmetrically while keeping the
other fixed.

**Statistical Method**:
For each permutation, create random permutation `perm`, then apply:
`matrix1[perm][:, perm]`. This preserves matrix structure while destroying
correlation. Count how often permuted correlation is as extreme as observed.

**Assumptions**:
- Matrices are square and same size
- Under H₀, row/column ordering is exchangeable
- Symmetric permutation preserves matrix properties (e.g., symmetry)

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | First square matrix (n×n) | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Second square matrix (n×n) | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000) | <code>5000</code>
`metric` | <code>[str](#str)</code> | Correlation metric ['pearson'|'spearman'|'kendall'] (default: 'pearson') | <code>'pearson'</code>
`how` | <code>[str](#str)</code> | Which elements to compare ['upper'|'lower'|'full'] (default: 'upper') - 'upper': Upper triangle only (assumes symmetric matrices) - 'lower': Lower triangle only - 'full': All elements (see include_diag) | <code>'upper'</code>
`include_diag` | <code>[bool](#bool)</code> | Include diagonal elements (only applies if how='full') (default: False) | <code>False</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type — 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction) - 2 | 'two': Two-tailed test (r != 0) - 1 | 'one': One-tailed (r > 0; negate the data for the other direction) | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | Return null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel workers, -1 = all cores (default: -1) Only used when device='cpu' | <code>-1</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show a progress bar over permutations (default: False) | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'correlation' (float): Observed correlation coefficient - 'p' (float): P-value using Phipson-Smyth correction - 'device' (str): Parallelization method used ('cpu' or None) - 'null_dist' (np.ndarray): Null distribution (if return_null=True)

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G. et al. (2016). Untangling the relatedness among correlations,
part I: nonparametric approaches to inter-subject correlation analysis
at the group level. NeuroImage, 142, 248-259.

Mantel, N. (1967). The detection of disease clustering and a generalized
regression approach. Cancer Research, 27(2), 209-220.

</details>

**Examples:**

```pycon
>>> import numpy as np
>>> from nltools.algorithms.inference import matrix_permutation_test
>>>
>>> # Create two correlated similarity matrices
>>> np.random.seed(42)
>>> n = 50
>>> true_pattern = np.random.randn(n)
>>> data1 = np.corrcoef(true_pattern + np.random.randn(n) * 0.1)
>>> data2 = np.corrcoef(true_pattern + np.random.randn(n) * 0.1)
>>>
>>> # Test if matrices are correlated
>>> result = matrix_permutation_test(data1, data2, n_permute=1000)
>>> print(f"Correlation: {result['correlation']:.3f}, p = {result['p']:.4f}")
```

(algorithms-multi-threshold)=
#### `multi_threshold`

```python
multi_threshold(t_map, p_map, thresh)
```

Threshold test image by multiple p-values from p image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_map` |  | (BrainData) BrainData instance of statistic metric (e.g., t-statistic, beta, etc) | *required*
`p_map` |  | (BrainData) BrainData instance of p-values | *required*
`thresh` |  | (list) list of p-values to threshold stat image | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | Thresholded BrainData instance with cumulative map - Positive values indicate how many thresholds were passed for positive stats - Negative values indicate how many thresholds were passed for negative stats

<details class="note" open markdown="1">
<summary>Note</summary>

This function provides unique cumulative threshold map functionality:
- Creates a single map showing which thresholds were passed
- Different from calling threshold() multiple times (which would give separate images)
- Useful for visualizing threshold hierarchies
- nilearn.threshold_img() does not support cumulative multi-threshold maps

</details>

(algorithms-one-sample-permutation-test)=
#### `one_sample_permutation_test`

```python
one_sample_permutation_test(data: np.ndarray, *, n_permute: int = 5000, tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None, progress_bar: bool = False) -> dict
```

One-sample permutation test using sign-flipping.

Tests whether the mean of data is significantly different from zero
by randomly flipping the sign of each observation. This is the
permutation test equivalent of a one-sample t-test.

Assumption: Symmetric error distribution around zero. For highly skewed
distributions, consider alternative methods (e.g., bootstrap resampling).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Data to test - shape (n_samples,) for single feature - shape (n_samples, n_features) for multi-feature (voxel-wise) | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000) | <code>5000</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type — 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction) - 2 | 'two': Two-tailed test (mean != 0) - 1 | 'one': One-tailed (mean > 0; negate the data for the other   direction). The fixed direction keeps MCP correction valid. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Whether to display a progress bar (default: False) | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'mean' (float or np.ndarray): Observed mean(s) - 'p' (float or np.ndarray): P-value(s) - 'null_dist' (np.ndarray): Null distribution (if return_null=True) - 'device' (str): Parallelization method used

**Examples:**

```pycon
>>> # Single feature (default CPU parallelization)
>>> data = np.random.randn(30)
>>> result = one_sample_permutation_test(data, n_permute=5000)
>>> result['p']
0.23
```

```pycon
>>> # Voxel-wise test with GPU
>>> data = np.random.randn(30, 10000)  # 30 subjects, 10K voxels
>>> result = one_sample_permutation_test(data, n_permute=5000, device='gpu')
>>> result['mean'].shape
(10000,)
>>> result['p'].shape
(10000,)
```

```pycon
>>> # Single-threaded (for debugging)
>>> result = one_sample_permutation_test(data, n_permute=5000, device=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (device='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
- Single-threaded (device=None): Use for small problems or debugging
- For voxel-wise tests, each voxel tested independently
- Progress bars show completion for both CPU parallel and GPU batched modes

</details>

(algorithms-phase-randomize)=
#### `phase_randomize`

```python
phase_randomize(data: np.ndarray, *, device: str | None = 'cpu', random_state: int | np.random.RandomState | None = None) -> np.ndarray
```

FFT-based phase randomization for time-series data.

Preserves the power spectrum (autocorrelation) but destroys nonlinear
temporal structure by randomizing Fourier phases. Used to test whether
data was generated by a linear Gaussian process or contains nonlinear
dynamics.

<details class="algorithm" open markdown="1">
<summary>Algorithm</summary>

1. Compute FFT of input signal
2. Generate random phases [0, 2π] for positive frequencies
3. Apply phase shifts to positive frequencies: multiply by exp(i*φ)
4. Apply conjugate phase shifts to negative frequencies (for real output)
5. Compute inverse FFT to get phase-randomized signal

</details>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Time series data, shape (n_samples,) or (n_samples, n_features) | *required*
`device` | <code>[str](#str) \| None</code> | Compute device. - 'cpu' / None: NumPy FFT (default, float64 precision) - 'gpu': PyTorch FFT on CUDA/MPS (float32 precision, 5-20× faster for large data) - 'auto': use a GPU if present, else CPU | <code>'cpu'</code>
`random_state` | <code>[int](#int) \| [RandomState](#numpy.random.RandomState) \| None</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Phase-randomized data with same shape as input

<details class="notes" open markdown="1">
<summary>Notes</summary>

- **CRITICAL**: Preserves power spectrum exactly (within numerical precision)
- Precision: the CPU path uses float64, the GPU path float32
- Conjugate symmetry is maintained for real-valued output

</details>

**Examples:**

```pycon
>>> x = np.sin(np.linspace(0, 10*np.pi, 100))  # Sine wave
>>> x_rand = phase_randomize(x, random_state=42)
>>> # Power spectrum preserved:
>>> np.allclose(np.abs(np.fft.rfft(x))**2, np.abs(np.fft.rfft(x_rand))**2)
True
```

```pycon
>>> # GPU acceleration for large datasets:
>>> x_large = np.random.randn(10000)
>>> x_rand_gpu = phase_randomize(x_large, device='gpu', random_state=42)
```

(algorithms-procrustes-distance)=
#### `procrustes_distance`

```python
procrustes_distance(mat1, mat2, *, n_permute = 5000, tail = 2, n_jobs = -1, random_state = None)
```

Test matrix similarity using Procrustes superposition.

Matrices need to match in size on their first dimension only, as the smaller
matrix on the second dimension will be padded with zeros. After aligning two
matrices using the Procrustes transformation, use the computed disparity
between them (sum of squared error of elements) as a similarity metric.
Shuffle the rows of one of the matrices and recompute the disparity to perform
inference (Peres-Neto & Jackson, 2001).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mat1` | <code>[ndarray](#ndarray)</code> | 2d numpy array; must have same number of rows as mat2 | *required*
`mat2` | <code>[ndarray](#ndarray)</code> | 1d or 2d numpy array; must have same number of rows as mat1 | *required*
`n_permute` | <code>[int](#int)</code> | number of permutation iterations to perform | <code>5000</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | 2|'two' (two-tailed, default) or 1|'one' (one-tailed: similarity > chance) | <code>2</code>
`n_jobs` | <code>[int](#int)</code> | The number of CPUs to use to do permutation; default -1 (all) | <code>-1</code>
`random_state` | <code>int, np.random.RandomState, or None</code> | seed or generator for the permutation shuffling; default None | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | results with keys `similarity` (float in [0, 1]) and `p` (permuted p-value)

(algorithms-regress)=
#### `regress`

```python
regress(X, Y, *, method: str = 'ols', stats: str = 'full', tail: int | str = 2)
```

Fit an OLS regression of ``Y`` on ``X``.

Does not add an intercept — include one in ``X`` explicitly. If ``Y``
is 2D, a separate regression is fit to each column.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` |  | Design matrix, shape ``(n_samples, n_regressors)``. | *required*
`Y` |  | Response, shape ``(n_samples,)`` or ``(n_samples, n_targets)``. | *required*
`method` | <code>[str](#str)</code> | Only ``'ols'`` is supported in v0.6.0. The legacy ``'robust'`` and ``'arma'`` methods were dropped; use statsmodels or a dedicated package if you need them. | <code>'ols'</code>
`stats` | <code>[str](#str)</code> | ``'full'`` returns the 6-tuple below; ``'betas'`` returns just ``b``; ``'tstats'`` returns ``(b, t)``. | <code>'full'</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | 2|'two' (two-tailed, default) or 1|'one' (one-tailed: beta > 0; negate a regressor for the other direction). | <code>2</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` |  | ``(b, se, t, p, df, res)`` when ``stats='full'``:
 |  | - ``b``: coefficients
 |  | - ``se``: standard errors
 |  | - ``t``: t-statistics
 |  | - ``p``: p-values (per ``tail``)
 |  | - ``df``: residual degrees of freedom
 |  | - ``res``: residuals

(algorithms-ridge-cv)=
#### `ridge_cv`

```python
ridge_cv(X: np.ndarray, y: np.ndarray, *, alphas: np.ndarray | None = None, cv: int | BaseCrossValidator = 5, fit_intercept: bool = False, parallel: str | None = 'cpu', max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> dict
```

Ridge regression with cross-validation for hyperparameter selection.

Performs k-fold cross-validation to select the best alpha parameter,
then fits a final model on all data using the selected alpha.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Training data features with shape (n_samples, n_features) | *required*
`y` | <code>[ndarray](#numpy.ndarray)</code> | Target values with shape (n_samples,) or (n_samples, n_targets) | *required*
`alphas` | <code>[ndarray](#numpy.ndarray)</code> | Array of alpha values to try. If None, uses default range: np.logspace(-2, 4, 20) = [0.01, 0.015, ..., 10000] | <code>None</code>
`cv` | <code>int or sklearn CV splitter</code> | Number of folds (int) or an sklearn cross-validator (anything with ``.split(X)`` and ``.get_n_splits()``, e.g. ``KFold(5, shuffle=True)`` or ``GroupKFold(8)``). Splitters are honored for the actual fold iteration, so leave-one-run-out and shuffled-K-fold give different results from contiguous K-fold. Defaults to 5. | <code>5</code>
`fit_intercept` | <code>[bool](#bool)</code> | If True, center X and y on the training mean before fitting and recover the intercept after. The returned ``coef`` is on the centered scale; the recovered intercept is returned under the ``intercept`` key. Defaults to False. | <code>False</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU-only using NumPy (default) - "gpu": GPU acceleration via PyTorch. Requires torch installed   (raises ImportError otherwise); degrades to torch-CPU only when no   GPU device is present. Use "auto" for torch-optional CPU fallback. Defaults to "cpu". | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel='gpu'). Defaults to 4.0. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed (not currently used, kept for consistency). Defaults to None. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary containing:<br>- 'alpha' (float): Best alpha value selected by CV - 'coef' (np.ndarray): Coefficients using best alpha on full dataset - 'cv_scores' (np.ndarray): Cross-validation R**2 scores for each fold, alpha, and target     with shape (n_folds, n_alphas, n_targets) - 'backend' (str): Backend used for computation

**Examples:**

```pycon
>>> X = np.random.randn(100, 50)
>>> y = np.random.randn(100)
>>> result = ridge_cv(X, y, cv=3)
>>> result['alpha']  # Best alpha selected
1.0
>>> result['coef'].shape
(50,)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Uses R**2 (coefficient of determination) as the scoring metric
- For multi-target regression, selects alpha that maximizes mean R**2 across targets
- parallel='gpu' requires torch installed; with torch present but no GPU device it
  runs on torch-CPU. It does not fall back to NumPy when torch is absent — use
  parallel='auto' for that.

</details>

(algorithms-ridge-svd)=
#### `ridge_svd`

```python
ridge_svd(X: np.ndarray, y: np.ndarray, *, alpha: float = 1.0, parallel: str | None = None, max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> np.ndarray
```

Solve ridge regression using Singular Value Decomposition.

This function implements ridge regression using SVD, which provides
numerical stability and efficiency for high-dimensional problems.
The implementation is inspired by the himalaya library.

<details class="algorithm" open markdown="1">
<summary>Algorithm</summary>

The ridge regression solution is:
    beta = (X.T @ X + alpha*I)^(-1) @ X.T @ y

Using SVD of X = U @ diag(s) @ V.T, this becomes:
    beta = V @ diag(s / (s**2 + alpha)) @ U.T @ y

This formulation avoids explicit matrix inversion and is numerically stable.
The shrinkage factor s / (s**2 + alpha) regularizes small singular values.

</details>

<details class="performance" open markdown="1">
<summary>Performance</summary>

- Time complexity: O(n_samples × n_features × min(n_samples, n_features))
- Space complexity: O(n_samples × n_features)
- GPU acceleration: ~10-100× speedup for large problems (n_features > 10K)
- See `solve_ridge_cv()` for cross-validation with GPU support

</details>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Training data features with shape (n_samples, n_features) | *required*
`y` | <code>[ndarray](#numpy.ndarray)</code> | Target values with shape (n_samples,) or (n_samples, n_targets). Can be 1D for single-target or 2D for multi-target | *required*
`alpha` | <code>[float](#float)</code> | Regularization strength. Must be positive. Higher values increase regularization (shrink coefficients toward zero). Defaults to 1.0. | <code>1.0</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU-only using NumPy (default) - "gpu": GPU acceleration via PyTorch. Requires torch installed   (raises ImportError otherwise); degrades to torch-CPU only when no   GPU device is present. Use "auto" for torch-optional CPU fallback. Defaults to None. | <code>None</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel='gpu'). Defaults to 4.0. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed (not currently used, kept for consistency). Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | np.ndarray: Ridge regression coefficients - shape (n_features,) for single-target regression - shape (n_features, n_targets) for multi-target regression

**Examples:**

```pycon
>>> X = np.random.randn(100, 50)
>>> y = np.random.randn(100)
>>> beta = ridge_svd(X, y, alpha=1.0)
>>> beta.shape
(50,)
```

```pycon
>>> # Multi-target regression
>>> Y = np.random.randn(100, 5)
>>> beta = ridge_svd(X, Y, alpha=1.0)
>>> beta.shape
(50, 5)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Time complexity: O(n_samples * n_features * min(n_samples, n_features))
- Space complexity: O(n_samples * n_features)
- For alpha→0, this reduces to ordinary least squares (OLS). Use alpha=1e-6
  for OLS in practice (more numerically stable than alpha=0)
- Supports both CPU (NumPy) and GPU (PyTorch) backends
- See `nltools.algorithms.ridge.solvers.solve_ridge_cv()` for cross-validation
- See `nltools.algorithms.ridge.utils._decompose_ridge()` for generator pattern

</details>

(algorithms-spm-dispersion-derivative)=
#### `spm_dispersion_derivative`

```python
spm_dispersion_derivative(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Implement the :term:`SPM` dispersion derivative :term:`HRF` model.

Parameters
----------
t_r : `float`
    :term:`Repetition time<TR>`, in seconds (sampling period).

<details class="oversampling-" open markdown="1">
<summary>`int`, default=50</summary>

Temporal oversampling factor in seconds.

</details>

<details class="time_length-" open markdown="1">
<summary>`float`, default=32.0</summary>

:term:`HRF` kernel length, in seconds.

</details>

<details class="onset-" open markdown="1">
<summary>`float`, default=0.0</summary>

Onset of the response in seconds.

</details>

Returns
-------
dhrf : array of shape (length / tr * oversampling), dtype=float
      dhrf sampling on the oversampled time grid

Examples
--------
>>> import numpy as np
>>> from nilearn.glm.first_level import glover_dispersion_derivative
>>> ddhrf = glover_dispersion_derivative(
...     t_r=2.0, oversampling=1, time_length=20.0
... )
>>> np.round(ddhrf, 3).tolist()
[0.0, -0.0, -0.373, 0.282, 0.295, -0.04, -0.094, -0.048, -0.017, -0.005]

(algorithms-spm-hrf)=
#### `spm_hrf`

```python
spm_hrf(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Implement the :term:`SPM` :term:`HRF` model.

Parameters
----------
t_r : `float`
    :term:`Repetition time<TR>`, in seconds (sampling period).

<details class="oversampling-" open markdown="1">
<summary>`int`, default=50</summary>

Temporal oversampling factor.

</details>

<details class="time_length-" open markdown="1">
<summary>`float`, default=32.0</summary>

:term:`HRF` kernel length, in seconds.

</details>

<details class="onset-" open markdown="1">
<summary>`float`, default=0.0</summary>

:term:`HRF` onset time, in seconds.

</details>

Returns
-------
hrf : array of shape (length / t_r * oversampling, dtype=float)
     :term:`HRF` sampling on the oversampled time grid

Examples
--------
>>> import numpy as np
>>> from nilearn.glm.first_level import spm_hrf
>>> hrf = spm_hrf(t_r=2.0, oversampling=1, time_length=20.0)
>>> np.round(hrf, 3).tolist()
[0.0, 0.0, 0.161, 0.443, 0.335, 0.139, 0.022, -0.028, -0.04, -0.033]

(algorithms-spm-time-derivative)=
#### `spm_time_derivative`

```python
spm_time_derivative(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Implement the :term:`SPM` time derivative :term:`HRF` (dhrf) model.

Parameters
----------
t_r : `float`
    :term:`Repetition time<TR>`, in seconds (sampling period).

<details class="oversampling-" open markdown="1">
<summary>`int`, default=50</summary>

Temporal oversampling factor.

</details>

<details class="time_length-" open markdown="1">
<summary>`float`, default=32.0</summary>

:term:`HRF` kernel length, in seconds.

</details>

<details class="onset-" open markdown="1">
<summary>`float`, default=0.0</summary>

Onset of the response in seconds.

</details>

Returns
-------
dhrf : array of shape (length / t_r, dtype=float)
      dhrf sampling on the provided grid

Examples
--------
>>> import numpy as np
>>> from nilearn.glm.first_level import spm_time_derivative
>>> dhrf = spm_time_derivative(t_r=2.0, oversampling=1, time_length=20.0)
>>> np.round(dhrf, 3).tolist()
[0.0, 0.0, 0.167, 0.04, -0.091, -0.072, -0.035, -0.013, -0.0, 0.005]

(algorithms-threshold)=
#### `threshold`

```python
threshold(stat, p, thr = 0.05, return_mask = False)
```

Threshold test image by p-value from p image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stat` |  | (BrainData) BrainData instance of arbitrary statistic metric   (e.g., beta, t, etc) | *required*
`p` |  | (BrainData) BrainData instance of p-values | *required*
`thr` |  | (float) p-value threshold to apply | <code>0.05</code>
`return_mask` |  | (bool) optionally return the thresholding mask; default False | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | Thresholded BrainData instance
`mask` |  | (optional) BrainData instance of thresholding mask if return_mask=True

<details class="note" open markdown="1">
<summary>Note</summary>

This function provides unique functionality not available in nilearn:
- Thresholds stat image based on p-values from separate p-value image
- Neither nilearn.threshold_img nor BrainData.threshold() support this
- BrainData.threshold() thresholds based on stat values themselves
- nilearn.threshold_img() thresholds based on image intensity values

</details>

(algorithms-timeseries-correlation-permutation-test)=
#### `timeseries_correlation_permutation_test`

```python
timeseries_correlation_permutation_test(data1: np.ndarray, data2: np.ndarray, *, method: Literal['circle_shift', 'phase_randomize'] = 'circle_shift', n_permute: int = 5000, metric: Literal['pearson', 'spearman', 'kendall'] = 'pearson', tail: int | str = 2, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, return_null: bool = False, random_state: int | np.random.RandomState | None = None, progress_bar: bool = False) -> dict
```

Time-series correlation permutation test.

Unlike standard permutation tests that shuffle data independently,
this test uses time-series-aware permutation methods that preserve
temporal structure (circle_shift) or power spectrum (phase_randomize).

Use this test when data contains temporal autocorrelation. Standard
permutation tests inflate Type I error for autocorrelated data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | First time series, shape (n_samples,) or (n_samples, 1) | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Second time series, shape (n_samples,) or (n_samples, 1) | *required*
`method` | <code>[Literal](#typing.Literal)['circle_shift', 'phase_randomize']</code> | Permutation method: - 'circle_shift': Circular shift (preserves autocorrelation) - 'phase_randomize': FFT-based (preserves power spectrum) | <code>'circle_shift'</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations | <code>5000</code>
`metric` | <code>[Literal](#typing.Literal)['pearson', 'spearman', 'kendall']</code> | Correlation type ('pearson', 'spearman', 'kendall') | <code>'pearson'</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type (default: 2) - 2 or 'two': Two-tailed test (default) - 1 or 'one': One-tailed test in the test's positive direction   (to test the negative direction, negate the data / swap groups) | <code>2</code>
`device` | <code>[str](#str) \| None</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs (-1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float) \| None</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | Whether to return null distribution | <code>False</code>
`random_state` | <code>[int](#int) \| [RandomState](#numpy.random.RandomState) \| None</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show a progress bar over permutations (default: False) | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with keys: - 'correlation': Observed correlation coefficient - 'p': P-value - 'null_dist': (if return_null=True) Null distribution - 'device': Parallelization method used

**Examples:**

```pycon
>>> x = np.sin(np.linspace(0, 10*np.pi, 100))
>>> y = np.cos(np.linspace(0, 10*np.pi, 100))
>>> result = timeseries_correlation_permutation_test(
...     x, y, method='circle_shift', n_permute=1000, random_state=42
... )
>>> result['correlation']  # Strong negative correlation
-0.999...
>>> result['p'] < 0.05  # Significant
True
```

```pycon
>>> # GPU acceleration
>>> result = timeseries_correlation_permutation_test(
...     x, y, method='phase_randomize', device='gpu', n_permute=5000
... )
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (device='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): 5-20× faster for large problems (n_samples > 1000)
- Single-threaded (device=None): Use for small problems or debugging
- For independent data, use regular correlation_permutation_test
- circle_shift is faster and suitable for most fMRI time series
- phase_randomize preserves power spectrum exactly (tests nonlinearity)
- Only data1 is randomized; data2 remains fixed to test correlation
- phase_randomize benefits most from GPU (FFT acceleration)

</details>

(algorithms-transform-pairwise)=
#### `transform_pairwise`

```python
transform_pairwise(X, y)
```

Transform data into pairs with balanced labels for ranking.

Transforms a n-class ranking problem into a two-class classification
problem. Subclasses implementing particular strategies for choosing
pairs should override this method.
In this method, all pairs are choosen, except for those that have the
same target value. The output is an array of balanced classes, i.e.
there are the same number of -1 as +1

Reference: "Large Margin Rank Boundaries for Ordinal Regression",
R. Herbrich, T. Graepel, K. Obermayer. Authors: Fabian Pedregosa
<fabian@fseoane.net> Alexandre Gramfort <alexandre.gramfort@inria.fr>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` |  | (np.array), shape (n_samples, n_features) The data | *required*
`y` |  | (np.array), shape (n_samples,) or (n_samples, 2) Target labels. If it's a 2D array, the second column represents the grouping of samples, i.e., samples with different groups will not be considered. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`X_trans` |  | (np.array), shape (k, n_features) Data as pairs, where k = n_samples * (n_samples-1)) / 2 if grouping values were not passed. If grouping variables exist, then returns values computed for each group.
`y_trans` |  | (np.array), shape (k,) Output class labels, where classes have values {-1, +1} If y was shape (n_samples, 2), then returns (k, 2) with groups on the second dimension.

(algorithms-trim)=
#### `trim`

```python
trim(data, cutoff = None)
```

Trim a Polars DataFrame/Series by replacing outlier values with NaNs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pl.DataFrame, pl.Series) data to trim | *required*
`cutoff` |  | (dict) a dictionary with keys {'std':[low,high]} or     {'quantile':[low,high]} | <code>None</code>

Returns:
    out: (pl.DataFrame, pl.Series) trimmed data (same type as input)

(algorithms-two-sample-permutation-test)=
#### `two_sample_permutation_test`

```python
two_sample_permutation_test(data1: np.ndarray, data2: np.ndarray, *, n_permute: int = 5000, tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Two-sample permutation test using group label shuffling.

Tests whether two independent groups have different means by randomly
permuting group labels. This is the permutation test equivalent of an
independent samples t-test.

Assumption: Exchangeability under the null hypothesis (group assignments
are arbitrary). Valid for independent samples from similar distributions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | Group 1 data - shape (n_samples1,) for single feature - shape (n_samples1, n_features) for multi-feature (voxel-wise) | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Group 2 data - shape (n_samples2,) for single feature - shape (n_samples2, n_features) for multi-feature (voxel-wise) | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000) | <code>5000</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type — 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction) - 2 | 'two': Two-tailed test (mean1 != mean2) - 1 | 'one': One-tailed (mean1 > mean2; swap the groups for the   other direction). The fixed direction keeps MCP correction valid. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'mean_diff' (float or np.ndarray): Observed mean difference (data1 - data2) - 'p' (float or np.ndarray): P-value(s) - 'null_dist' (np.ndarray): Null distribution (if return_null=True) - 'device' (str): Parallelization method used

**Examples:**

```pycon
>>> # Single feature (default CPU parallelization)
>>> data1 = np.random.randn(20)  # Group 1: 20 subjects
>>> data2 = np.random.randn(25)  # Group 2: 25 subjects
>>> result = two_sample_permutation_test(data1, data2, n_permute=5000)
>>> result['p']
0.45
```

```pycon
>>> # Voxel-wise test with GPU
>>> data1 = np.random.randn(20, 10000)  # 20 subjects, 10K voxels
>>> data2 = np.random.randn(25, 10000)  # 25 subjects, 10K voxels
>>> result = two_sample_permutation_test(data1, data2, n_permute=5000, device='gpu')
>>> result['mean_diff'].shape
(10000,)
>>> result['p'].shape
(10000,)
```

```pycon
>>> # Single-threaded (for debugging)
>>> result = two_sample_permutation_test(data1, data2, n_permute=5000, device=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (device='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
- Single-threaded (device=None): Use for small problems or debugging
- For voxel-wise tests, each voxel tested independently
- Group sizes can be unequal

</details>

(algorithms-u-center)=
#### `u_center`

```python
u_center(mat: np.ndarray) -> np.ndarray
```

U-center a 2d array. U-centering is a bias-corrected form of double-centering.

U-centering corrects for bias that occurs with double-centering as the number
of dimensions increases. The diagonal is explicitly set to zero.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mat` | <code>[ndarray](#ndarray)</code> | 2d numpy array | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`mat` | <code>[ndarray](#ndarray)</code> | u-centered version of input

**Examples:**

```pycon
>>> mat = np.random.randn(5, 5)
>>> result = u_center(mat)
>>> np.allclose(np.diag(result), 0)
True
```

(algorithms-upsample)=
#### `upsample`

```python
upsample(data, *, sampling_freq = None, target = None, target_type = 'samples', method = 'linear')
```

Upsample a Polars DataFrame/Series to a new target frequency or number of samples using interpolation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pl.DataFrame, pl.Series) data to upsample   (Note: will drop non-numeric columns from DataFrame) | *required*
`sampling_freq` |  | Sampling frequency of data in hertz | <code>None</code>
`target` |  | (float) upsampling target | <code>None</code>
`target_type` |  | (str) type of target can be [samples,seconds,hz] | <code>'samples'</code>
`method` |  | (str) ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']           where 'zero', 'slinear', 'quadratic' and 'cubic'           refer to a spline interpolation of zeroth, first,           second or third order  (default: linear) | <code>'linear'</code>

Returns:
    upsampled Polars DataFrame or Series (same type as input)

(algorithms-winsorize)=
#### `winsorize`

```python
winsorize(data, cutoff = None, replace_with_cutoff = True)
```

Winsorize a Polars DataFrame/Series with the largest/lowest value not considered outlier.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pl.DataFrame, pl.Series) data to winsorize | *required*
`cutoff` |  | (dict) a dictionary with keys {'std':[low,high]} or     {'quantile':[low,high]} | <code>None</code>
`replace_with_cutoff` |  | (bool) If True, replace outliers with cutoff.                  If False, replaces outliers with closest                  existing values; (default: True) | <code>True</code>

Returns:
    out: (pl.DataFrame, pl.Series) winsorized data (same type as input)

(algorithms-zscore)=
#### `zscore`

```python
zscore(data)
```

Z-score every column of a Polars or pandas DataFrame/Series.

Accepts pandas inputs at the boundary for convenience and converts to
Polars internally. Always returns Polars output (DataFrame or Series,
matching the input shape).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | pl.DataFrame, pl.Series, pd.DataFrame, or pd.Series. | *required*

**Returns:**

Type | Description
---- | -----------
 | pl.DataFrame or pl.Series with each column z-scored using sample
 | standard deviation (ddof=1), matching the input shape.



### Modules

(algorithms-alignment)=
#### `alignment`

Multi-subject functional alignment algorithms.

This package provides algorithms for aligning functional data across subjects:

- **LocalAlignment**: Searchlight/ROI-scale alignment (Bazeille et al. 2021)
- **HyperAlignment**: Iterative Procrustes alignment (Haxby et al. 2011)
- **SRM** / **DetSRM**: Shared Response Model (Chen et al. 2015)
- **align** / **procrustes** / **procrustes_distance** / **align_states**:
  functional entry points for whole-brain alignment (SRM or Procrustes)

**Classes:**

Name | Description
---- | -----------
[`DetSRM`](#algorithms-detsrm) | Deterministic Shared Response Model (DetSRM).
[`HyperAlignment`](#algorithms-hyperalignment) | Hyperalignment using iterative Procrustes alignment.
[`LocalAlignment`](#algorithms-localalignment) | Local (neighborhood-based) functional alignment across subjects.
[`SRM`](#algorithms-srm) | Probabilistic Shared Response Model (SRM).

**Methods:**

Name | Description
---- | -----------
[`align`](#algorithms-align) | Align subject data into a common response model.
[`align_states`](#algorithms-align-states) | Align state weight maps by minimizing pairwise distance between group states.
[`procrustes_distance`](#algorithms-procrustes-distance) | Test matrix similarity using Procrustes superposition.



**Modules:**

Name | Description
---- | -----------
[`hyperalignment`](#algorithms-hyperalignment) | HyperAlignment: Multi-subject cortical surface alignment using iterative Procrustes refinement.
[`local`](#algorithms-local) | LocalAlignment: Neighborhood-based functional alignment.
[`procrustes`](#algorithms-procrustes) | Data alignment — SRM, Procrustes, and state alignment.
[`srm`](#algorithms-srm) | Shared Response Model (SRM) for multi-subject fMRI alignment.

##### Classes

###### `DetSRM`

```python
DetSRM(*, n_iter: int = 10, features: int = 50, rand_seed: int = 0) -> None
```

Bases: <code>[BaseEstimator](#sklearn.base.BaseEstimator)</code>, <code>[TransformerMixin](#sklearn.base.TransformerMixin)</code>

Deterministic Shared Response Model (DetSRM).

Given multi-subject data, factorize it as a shared response S among all
subjects and an orthogonal transform W per subject:

$$
X_i \approx W_i S, \forall i=1 \dots N
$$

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_iter` | <code>int, default=10</code> | Number of iterations to run the algorithm. | <code>10</code>
`features` | <code>int, default=50</code> | Number of features to compute. | <code>50</code>
`rand_seed` | <code>int, default=0</code> | Seed for initializing the random number generator. | <code>0</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`w_` | <code>list of array, element i has shape=[voxels_i, features]</code> | The orthogonal transforms (mappings) for each subject.
`s_` | <code>array, shape=[features, samples]</code> | The shared response.
`random_state_` | <code>`RandomState`</code> | Random number generator initialized using rand_seed

<details class="note" open markdown="1">
<summary>Note</summary>

The number of voxels may be different between subjects. However, the
number of samples must be the same across subjects.

The Deterministic Shared Response Model is approximated using the
Block Coordinate Descent (BCD) algorithm proposed in **Chen2015**.

This is a single node version.

The run-time complexity is $O(I (V T K + V K^2))$ and the memory
complexity is $O(V T)$ with I - the number of iterations, V - the
sum of voxels from all subjects, T - the number of samples, K - the
number of features (typically, $V \gg T \gg K$), and N - the
number of subjects.

</details>

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Compute the Deterministic Shared Response Model.
[`transform`](#algorithms-transform) | Use the model to transform data to the Shared Response subspace.
[`transform_subject`](#algorithms-transform-subject) | Transform a new subject using the existing model.



####### Attributes##

**Examples:**

Basic multi-subject DetSRM fitting:

```pycon
>>> from nltools.algorithms import DetSRM
>>> import numpy as np
>>>
>>> # Create sample data (3 subjects)
>>> data = [np.random.randn(100, 50) for _ in range(3)]
>>>
>>> # Fit DetSRM with CPU parallelization (default)
>>> detsrm = DetSRM(n_iter=10, features=50)
>>> detsrm.fit(data, parallel="cpu", n_jobs=-1)
>>>
>>> # Transform to shared response space
>>> shared_responses = detsrm.transform(data)
>>>
>>> # Access fitted model components
>>> w = detsrm.w_  # Subject-specific transforms
>>> s = detsrm.s_  # Shared response
```

(algorithms-features)=
###### `features`

```python
features = features
```

######## `n_iter`

```python
n_iter = n_iter
```

######## `rand_seed`

```python
rand_seed = rand_seed
```



####### Functions##

###### `fit`

```python
fit(X: list[np.ndarray], y: Any | None = None, *, parallel: str | None = 'cpu', n_jobs: int = -1) -> DetSRM
```

Compute the Deterministic Shared Response Model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples]</code> | Each element in the list contains the fMRI data of one subject. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": not implemented -- raises `NotImplementedError` (never a silent CPU fallback) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples_i]</code> | Each element in the list contains the fMRI data of one subject. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": not implemented -- raises `NotImplementedError` (never a silent CPU fallback) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>2D array, shape=[voxels, timepoints]</code> | The fMRI data of the new subject. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`self` | <code>[DetSRM](#nltools.algorithms.alignment.srm.DetSRM)</code> | Fitted model

######## `transform`

```python
transform(X: list[np.ndarray], y: Any | None = None, *, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray]
```

Use the model to transform data to the Shared Response subspace.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`s` | <code>list of 2D arrays, element i has shape=[features_i, samples_i]</code> | Shared responses from input data (X)

######## `transform_subject`

```python
transform_subject(X: np.ndarray) -> np.ndarray
```

Transform a new subject using the existing model.

The subject is assumed to have received equivalent stimulation.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`w` | <code>2D array, shape=[voxels, features]</code> | Orthogonal mapping `W_{new}` for new subject

###### `HyperAlignment`

```python
HyperAlignment(n_iter: int = 2, auto_pad: bool = True) -> None
```

Bases: <code>[BaseEstimator](#sklearn.base.BaseEstimator)</code>, <code>[TransformerMixin](#sklearn.base.TransformerMixin)</code>

Hyperalignment using iterative Procrustes alignment.

Three-stage iterative process for aligning multi-subject data:
1. Create initial average template
2. Refine template through n_iter iterations
3. Final alignment of all subjects to refined template

This implements the Procrustes-based hyperalignment method commonly
used in multi-subject neuroimaging analysis.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_iter` | <code>int, default=2</code> | Number of template refinement iterations (stages 1-2). | <code>2</code>
`auto_pad` | <code>bool, default=True</code> | If True, automatically zero-pad matrices to standardize sizes. If False, caller must ensure all matrices have same dimensions. | <code>True</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_iter` | <code>int, default=2</code> | Number of template refinement iterations | <code>2</code>
`auto_pad` | <code>bool, default=True</code> | Whether to automatically pad matrices to same size | <code>True</code>



####### Attributes##

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`w_` | <code>list of ndarray, element i has shape=[features_i, features]</code> | The transformation matrices (rotation + reflection) for each subject.
`s_` | <code>ndarray, shape=[features, samples]</code> | The aligned common template (shared response).
`disparity_` | <code>list of float</code> | Disparity (sum of squared differences) for each subject.
`scale_` | <code>list of float</code> | Scale factors for each subject.

<details class="note" open markdown="1">
<summary>Note</summary>

``common_model_`` property provides alias for ``s_`` (backward compatibility).

</details>

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Fit hyperalignment model to data.
[`transform`](#algorithms-transform) | Transform data to common space using fitted transformations.
[`transform_subject`](#algorithms-transform-subject) | Align a new subject to the common space.

**Examples:**

Basic multi-subject alignment:

```pycon
>>> from nltools.algorithms import HyperAlignment
>>> import numpy as np
>>>
>>> # Create sample data (3 subjects)
>>> data = [np.random.randn(100, 50) for _ in range(3)]
>>>
>>> # Fit hyperalignment with CPU parallelization (default)
>>> hyper = HyperAlignment(n_iter=2)
>>> hyper.fit(data, parallel="cpu", n_jobs=-1)
>>>
>>> # Transform to common space
>>> aligned = hyper.transform(data)
>>>
>>> # Access common template
>>> template = hyper.s_  # or hyper.common_model_
>>>
>>> # Align a new subject
>>> new_subject = np.random.randn(100, 50)
>>> new_transform = hyper.transform_subject(new_subject)
```

<details class="note" open markdown="1">
<summary>Note</summary>

When to use parallel processing:

- Use ``parallel="cpu"`` (default) for datasets with 3+ subjects to speed up
  pairwise Procrustes operations during template refinement.
- Use ``parallel=None`` for debugging or small datasets (<3 subjects) where
  parallelization overhead isn't beneficial.
- Parallel processing is most beneficial when subjects have many voxels
  (>10K) and template refinement requires multiple iterations.

</details>

<details class="references" open markdown="1">
<summary>References</summary>

Haxby, J. V., Guntupalli, J. S., Connolly, A. C., Halchenko, Y. O.,
Conroy, B. R., Gobbini, M. I., ... & Ramadge, P. J. (2011).
A common, high-dimensional model of the representational space in
human ventral temporal cortex. Neuron, 72(2), 404-416.

</details>

(algorithms-auto-pad)=
###### `auto_pad`

```python
auto_pad = auto_pad
```

######## `common_model_`

```python
common_model_
```

Alias for ``s_`` (common template).

######## `n_iter`

```python
n_iter = n_iter
```



####### Functions##

###### `fit`

```python
fit(data: list[np.ndarray], *, parallel: str | None = 'cpu', n_jobs: int = -1) -> HyperAlignment
```

Fit hyperalignment model to data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>list of ndarray</code> | List of data matrices, each with shape (n_features, n_samples). Different subjects can have different numbers of features if auto_pad=True. | *required*
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>list of ndarray</code> | List of data matrices to transform. Should be the same data used for fitting (or have compatible dimensions). | *required*
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`subject_data` | <code>([ndarray](#ndarray), [shape](#shape)([n_features](#n_features), [n_samples](#n_samples)))</code> | Data from a new subject to align to the common template | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`self` | <code>[HyperAlignment](#nltools.algorithms.alignment.hyperalignment.HyperAlignment)</code> | Fitted model

######## `transform`

```python
transform(data: list[np.ndarray], *, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray]
```

Transform data to common space using fitted transformations.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`transformed` | <code>list of ndarray</code> | List of transformed data matrices in common space

######## `transform_subject`

```python
transform_subject(subject_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]
```

Align a new subject to the common space.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`transformed` | <code>[ndarray](#ndarray)</code> | Aligned data in common space
`R` | <code>[ndarray](#ndarray)</code> | Transformation matrix used
`disparity` | <code>[float](#float)</code> | Alignment quality (sum of squared differences)
`scale` | <code>[float](#float)</code> | Scale factor used

###### `LocalAlignment`

```python
LocalAlignment(spatial_scale: str = 'searchlight', method: str = 'procrustes', radius_mm: float = 10.0, roi_mask: nib.Nifti1Image | None = None, n_features: int | None = None, n_iter: int = 3, aggregation: str = 'center', parallel: str | None = 'cpu', n_jobs: int = -1, progress_bar: bool = False, n_neighborhoods_batch: int | None = None, max_memory_gb: float | None = None, transforms_: dict[int, list[np.ndarray]] | None = None, template_: dict[int, np.ndarray] | None = None, neighborhoods_: SphereNeighborhoods | dict[int, np.ndarray] | None = None, n_voxels_: int | None = None, mask_: nib.Nifti1Image | None = None, backend_: Backend | None = None) -> None
```

Local (neighborhood-based) functional alignment across subjects.

Learns alignment transforms within local neighborhoods (searchlight spheres
or parcels) and applies center-only aggregation to preserve orthogonality.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`spatial_scale` | <code>[str](#str)</code> | Spatial scale, either 'searchlight' (overlapping spheres) or 'roi' (non-overlapping parcels). Defaults to 'searchlight'. | <code>'searchlight'</code>
`method` | <code>[str](#str)</code> | Alignment method, one of 'procrustes', 'srm', or 'hyperalignment'. Defaults to 'procrustes'. | <code>'procrustes'</code>
`radius_mm` | <code>[float](#float)</code> | Sphere radius in millimeters for the searchlight scale. Defaults to 10.0. | <code>10.0</code>
`roi_mask` | <code>[Nifti1Image](#Nifti1Image) \| None</code> | Parcellation image for the ROI scale. Required if `spatial_scale='roi'`. Defaults to None. | <code>None</code>
`n_features` | <code>[int](#int) \| None</code> | Number of features for SRM. None uses full Procrustes (preserves dims). Defaults to None. | <code>None</code>
`n_iter` | <code>[int](#int)</code> | Number of iterations for alignment refinement. Defaults to 3. | <code>3</code>
`aggregation` | <code>[str](#str)</code> | Aggregation method: 'center' (center-only, preserves orthogonality) or 'all'. Defaults to 'center'. | <code>'center'</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization mode. None runs single-threaded numpy, 'cpu' uses joblib CPU parallelization, and 'gpu' uses PyTorch. GPU acceleration applies only to `method='procrustes'`; requesting 'gpu' with the 'srm' or 'hyperalignment' methods raises `NotImplementedError` (an explicit GPU request never silently runs on CPU). Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of jobs for CPU parallelization. Defaults to -1. | <code>-1</code>
`progress_bar` | <code>[bool](#bool)</code> | Whether to display tqdm progress bars during fit and transform. Defaults to False. | <code>False</code>
`n_neighborhoods_batch` | <code>[int](#int) \| None</code> | Number of neighborhoods to process per batch on the GPU. None auto-calculates a batch size from `max_memory_gb`. Defaults to None. | <code>None</code>
`max_memory_gb` | <code>[float](#float) \| None</code> | Explicit memory budget (in GB) used to auto-size GPU batches when `n_neighborhoods_batch` is None. None (default) measures the device's available memory. | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`transforms_` | <code>[dict](#dict)[[int](#int), [list](#list)[[ndarray](#numpy.ndarray)]]</code> | Per-neighborhood transforms. Keys are center voxel indices, values are lists of transform matrices (one per subject).
`template_` | <code>[dict](#dict)[[int](#int), [ndarray](#numpy.ndarray)]</code> | Per-neighborhood templates used for alignment.
`neighborhoods_` | <code>[SphereNeighborhoods](#nltools.data.braindata.neighborhoods.SphereNeighborhoods) \| [dict](#dict)</code> | Computed neighborhoods (searchlight or roi).
`n_voxels_` | <code>[int](#int)</code> | Total number of voxels in the mask.
`mask_` | <code>[Nifti1Image](#Nifti1Image)</code> | Brain mask used for fitting.

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Fit local alignment on multi-subject data.
[`fit_transform`](#algorithms-fit-transform) | Fit alignment and transform data in one step.
[`transform`](#algorithms-transform) | Apply local transforms to data.



####### Attributes##

**Examples:**

```pycon
>>> import numpy as np
>>> import nibabel as nib
>>> from nltools.algorithms.alignment import LocalAlignment
>>> # Create synthetic multi-subject data (voxels, samples)
>>> data = [np.random.randn(1000, 100) for _ in range(5)]
>>> # Build a mask whose nonzero voxels match the 1000-voxel data
>>> mask = nib.Nifti1Image(np.ones((10, 10, 10), dtype=np.int8), np.eye(4))
>>> la = LocalAlignment(spatial_scale='searchlight', method='procrustes', radius_mm=10.0)
>>> la.fit(data, mask)
>>> aligned = la.transform(data)
```

<details class="note" open markdown="1">
<summary>Note</summary>

Based on Bazeille et al. 2021 "An empirical evaluation of functional
alignment using inter-subject decoding". Center-only aggregation is
used to preserve local orthogonality of transforms.

</details>

(algorithms-aggregation)=
###### `aggregation`

```python
aggregation: str = 'center'
```

######## `backend_`

```python
backend_: Backend | None = field(default=None, repr=False)
```

######## `mask_`

```python
mask_: nib.Nifti1Image | None = field(default=None, repr=False)
```

######## `max_memory_gb`

```python
max_memory_gb: float | None = None
```

######## `method`

```python
method: str = 'procrustes'
```

######## `n_features`

```python
n_features: int | None = None
```

######## `n_iter`

```python
n_iter: int = 3
```

######## `n_jobs`

```python
n_jobs: int = -1
```

######## `n_neighborhoods_batch`

```python
n_neighborhoods_batch: int | None = None
```

######## `n_voxels_`

```python
n_voxels_: int | None = field(default=None, repr=False)
```

######## `neighborhoods_`

```python
neighborhoods_: SphereNeighborhoods | dict[int, np.ndarray] | None = field(default=None, repr=False)
```

######## `parallel`

```python
parallel: str | None = 'cpu'
```

######## `progress_bar`

```python
progress_bar: bool = False
```

######## `radius_mm`

```python
radius_mm: float = 10.0
```

######## `roi_mask`

```python
roi_mask: nib.Nifti1Image | None = None
```

######## `spatial_scale`

```python
spatial_scale: str = 'searchlight'
```

######## `template_`

```python
template_: dict[int, np.ndarray] | None = field(default=None, repr=False)
```

######## `transforms_`

```python
transforms_: dict[int, list[np.ndarray]] | None = field(default=None, repr=False)
```



####### Functions##

###### `fit`

```python
fit(data: list[np.ndarray], mask: nib.Nifti1Image) -> LocalAlignment
```

Fit local alignment on multi-subject data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | List of subject data arrays, each shape (n_voxels, n_samples). Subjects can have different numbers of samples - the underlying alignment methods (SRM, HyperAlignment) handle this via zero-padding. | *required*
`mask` | <code>[Nifti1Image](#Nifti1Image)</code> | Brain mask defining the voxel space. | *required*

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | List of subject data arrays, each shape (n_voxels, n_samples). | *required*
`mask` | <code>[Nifti1Image](#Nifti1Image)</code> | Brain mask defining the voxel space. | *required*

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | List of subject data arrays, each shape (n_voxels, n_samples). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`LocalAlignment` | <code>[LocalAlignment](#nltools.algorithms.alignment.local.LocalAlignment)</code> | The fitted alignment model (`self`).

######## `fit_transform`

```python
fit_transform(data: list[np.ndarray], mask: nib.Nifti1Image) -> list[np.ndarray]
```

Fit alignment and transform data in one step.

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | list[np.ndarray]: Aligned data for each subject.

######## `transform`

```python
transform(data: list[np.ndarray]) -> list[np.ndarray]
```

Apply local transforms to data.

For the searchlight scale with center-only aggregation: each voxel uses
the transform from the neighborhood where it was the center.

For the roi scale: all voxels in each parcel use the same transform.

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | list[np.ndarray]: Aligned data for each subject, each shape (n_voxels, n_samples).

###### `SRM`

```python
SRM(*, n_iter: int = 10, features: int = 50, rand_seed: int = 0) -> None
```

Bases: <code>[BaseEstimator](#sklearn.base.BaseEstimator)</code>, <code>[TransformerMixin](#sklearn.base.TransformerMixin)</code>

Probabilistic Shared Response Model (SRM).

Given multi-subject data, factorize it as a shared response S among all
subjects and an orthogonal transform W per subject:

$$
X_i \approx W_i S, \forall i=1 \dots N
$$

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_iter` | <code>int, default=10</code> | Number of iterations to run the algorithm. | <code>10</code>
`features` | <code>int, default=50</code> | Number of features to compute. | <code>50</code>
`rand_seed` | <code>int, default=0</code> | Seed for initializing the random number generator. | <code>0</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`w_` | <code>list of array, element i has shape=[voxels_i, features]</code> | The orthogonal transforms (mappings) for each subject.
`s_` | <code>array, shape=[features, samples]</code> | The shared response.
`sigma_s_` | <code>array, shape=[features, features]</code> | The covariance of the shared response Normal distribution.
`mu_` | <code>list of array, element i has shape=[voxels_i]</code> | The voxel means over the samples for each subject.
`rho2_` | <code>array, shape=[subjects]</code> | The estimated noise variance $\rho_i^2$ for each subject
`random_state_` | <code>`RandomState`</code> | Random number generator initialized using rand_seed

<details class="note" open markdown="1">
<summary>Note</summary>

The number of voxels may be different between subjects. However, the
number of samples must be the same across subjects.

The probabilistic Shared Response Model is approximated using the
Expectation Maximization (EM) algorithm proposed in **Chen2015**. The
implementation follows the optimizations published in **Anderson2016**.

This is a single node version.

The run-time complexity is $O(I (V T K + V K^2 + K^3))$ and the
memory complexity is $O(V T)$ with I - the number of iterations,
V - the sum of voxels from all subjects, T - the number of samples, and
K - the number of features (typically, $V \gg T \gg K$).

</details>

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Compute the probabilistic Shared Response Model.
[`transform`](#algorithms-transform) | Use the model to transform matrix to Shared Response space.
[`transform_subject`](#algorithms-transform-subject) | Transform a new subject using the existing model.



####### Attributes##

**Examples:**

Basic multi-subject SRM fitting:

```pycon
>>> from nltools.algorithms import SRM
>>> import numpy as np
>>>
>>> # Create sample data (3 subjects)
>>> data = [np.random.randn(100, 50) for _ in range(3)]
>>>
>>> # Fit SRM with CPU parallelization (default)
>>> srm = SRM(n_iter=10, features=50)
>>> srm.fit(data, parallel="cpu", n_jobs=-1)
>>>
>>> # Transform to shared response space
>>> shared_responses = srm.transform(data)
>>>
>>> # Access fitted model components
>>> w = srm.w_  # Subject-specific transforms
>>> s = srm.s_  # Shared response
```

###### `features`

```python
features = features
```

######## `n_iter`

```python
n_iter = n_iter
```

######## `rand_seed`

```python
rand_seed = rand_seed
```



####### Functions##

###### `fit`

```python
fit(X: list[np.ndarray], y: Any | None = None, *, parallel: str | None = 'cpu', n_jobs: int = -1, pad_samples: bool = True) -> SRM
```

Compute the probabilistic Shared Response Model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples]</code> | Each element in the list contains the fMRI data of one subject. Subjects can have different numbers of samples if pad_samples=True. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": not implemented -- raises `NotImplementedError` (never a silent CPU fallback) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>
`pad_samples` | <code>[bool](#bool)</code> | If True (default), automatically zero-pad subjects with fewer samples to match the longest subject. This allows fitting SRM on data with unequal numbers of time points across subjects. | <code>True</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples_i]</code> | Each element in the list contains the fMRI data of one subject. Note that number of voxels and samples can vary across subjects. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used (as it is unsupervised learning) | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": not implemented -- raises `NotImplementedError` (never a silent CPU fallback) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>2D array, shape=[voxels, timepoints]</code> | The fMRI data of the new subject. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`self` | <code>[SRM](#nltools.algorithms.alignment.srm.SRM)</code> | Fitted model

######## `transform`

```python
transform(X: list[np.ndarray], y: Any | None = None, *, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray | None]
```

Use the model to transform matrix to Shared Response space.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`s` | <code>list of 2D arrays, element i has shape=[features_i, samples_i]</code> | Shared responses from input data (X)

######## `transform_subject`

```python
transform_subject(X: np.ndarray) -> np.ndarray
```

Transform a new subject using the existing model.

The subject is assumed to have received equivalent stimulation.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`w` | <code>2D array, shape=[voxels, features]</code> | Orthogonal mapping `W_{new}` for new subject



##### Methods

###### `align`

```python
align(data, method = 'deterministic_srm', n_features = None, axis = 0, *args, **kwargs)
```

Align subject data into a common response model.

This function is a convenience wrapper around `HyperAlignment` and `SRM` classes.

Can be used to hyperalign source data to target data using
Hyperalignment from Dartmouth (i.e., procrustes transformation; see
nltools.algorithms.procrustes) or Shared Response Model from Princeton (see
nltools.algorithms.srm). (see nltools.data.BrainData.align for aligning
a single Brain object to another). Common Model is shared response
model or centered target data. Transformed data can be back projected to
original data using Tranformation matrix. Inputs must be a list of BrainData
instances or numpy arrays (observations by features).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (list) A list of BrainData objects | *required*
`method` |  | (str) alignment method to use ['probabilistic_srm','deterministic_srm','procrustes'] | <code>'deterministic_srm'</code>
`n_features` |  | (int) number of features to align to common space. If None then will select number of voxels | <code>None</code>
`axis` |  | (int) axis to align on | <code>0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (dict) a dictionary containing a list of transformed subject matrices, a list of transformation matrices, the shared response matrix, and the intersubject correlation of the shared responses

**Examples:**

- Hyperalign using procrustes transform:
    >>> out = align(data, method='procrustes')
- Align using shared response model:
    >>> out = align(data, method='probabilistic_srm', n_features=None)
- Project aligned data into original data:
    >>> original_data = [np.dot(t.data,tm.T) for t,tm in zip(out['transformed'], out['transformation_matrix'])]

###### `align_states`

```python
align_states(reference, target, *, metric = 'correlation', return_index = False, replace_zero_variance = False)
```

Align state weight maps by minimizing pairwise distance between group states.

This function uses the Hungarian algorithm for state alignment, which is
different from aligning multiple subjects' data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`reference` |  | (np.array) reference pattern x state matrix | *required*
`target` |  | (np.array) target pattern x state matrix to align to reference | *required*
`metric` |  | (str) distance metric to use | <code>'correlation'</code>
`return_index` |  | (bool) return index if True, return remapped data if False | <code>False</code>
`replace_zero_variance` |  | (bool) transform a vector with zero variance to random numbers from a uniform distribution.                     Useful for when using correlation as a distance metric to avoid NaNs. | <code>False</code>

Returns:
    If ``return_index=False`` (default): ``target[:, remapping]``, a single
    ndarray of the target's columns reordered to match the reference,
    oriented pattern x state (same shape as ``target``).
    If ``return_index=True``: the remapping index array (ndarray) that
    reorders the target's state columns.

###### `procrustes_distance`

```python
procrustes_distance(mat1, mat2, *, n_permute = 5000, tail = 2, n_jobs = -1, random_state = None)
```

Test matrix similarity using Procrustes superposition.

Matrices need to match in size on their first dimension only, as the smaller
matrix on the second dimension will be padded with zeros. After aligning two
matrices using the Procrustes transformation, use the computed disparity
between them (sum of squared error of elements) as a similarity metric.
Shuffle the rows of one of the matrices and recompute the disparity to perform
inference (Peres-Neto & Jackson, 2001).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mat1` | <code>[ndarray](#ndarray)</code> | 2d numpy array; must have same number of rows as mat2 | *required*
`mat2` | <code>[ndarray](#ndarray)</code> | 1d or 2d numpy array; must have same number of rows as mat1 | *required*
`n_permute` | <code>[int](#int)</code> | number of permutation iterations to perform | <code>5000</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | 2|'two' (two-tailed, default) or 1|'one' (one-tailed: similarity > chance) | <code>2</code>
`n_jobs` | <code>[int](#int)</code> | The number of CPUs to use to do permutation; default -1 (all) | <code>-1</code>
`random_state` | <code>int, np.random.RandomState, or None</code> | seed or generator for the permutation shuffling; default None | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | results with keys `similarity` (float in [0, 1]) and `p` (permuted p-value)



##### Modules

###### `hyperalignment`

HyperAlignment: Multi-subject cortical surface alignment using iterative Procrustes refinement.

Hyperalignment finds a common representational space across subjects by iteratively
refining pairwise Procrustes transformations. Unlike simple alignment, hyperalignment
preserves both spatial structure and representational similarity.

<details class="algorithm-overview" open markdown="1">
<summary>Algorithm overview</summary>

1. Initialize template (first subject or group average)
2. For each iteration:
   - Align each subject to template (Procrustes transformation)
   - Update template (average in aligned space)
3. Converge when transformations stabilize or max iterations reached
4. Final alignment: Apply learned transformations to all subjects

</details>

<details class="performance" open markdown="1">
<summary>Performance</summary>

- Time complexity: O(n_iter × n_subjects² × n_voxels × n_samples)
- Memory complexity: O(n_subjects × n_voxels × n_features)
- Parallelization: ~4-8× speedup with CPU-parallel (parallel="cpu")
- Most beneficial when subjects have many voxels (>10K) and multiple iterations

</details>

<details class="when-to-use-hyperalignment" open markdown="1">
<summary>When to use hyperalignment</summary>

- Multi-subject alignment preserving spatial structure
- Alternative to SRM when spatial structure is important
- See `nltools.algorithms.srm.SRM` for dimension-reduction approach
- See `nltools.algorithms.procrustes()` for single-subject alignment

</details>

This module implements the hyperalignment technique described in:

Haxby, J. V., Guntupalli, J. S., Connolly, A. C., Halchenko, Y. O.,
Conroy, B. R., Gobbini, M. I., ... & Ramadge, P. J. (2011).
A common, high-dimensional model of the representational space in
human ventral temporal cortex. Neuron, 72(2), 404-416.

**Classes:**

Name | Description
---- | -----------
[`HyperAlignment`](#algorithms-hyperalignment) | Hyperalignment using iterative Procrustes alignment.



####### Classes##

###### `HyperAlignment`

```python
HyperAlignment(n_iter: int = 2, auto_pad: bool = True) -> None
```

Bases: <code>[BaseEstimator](#sklearn.base.BaseEstimator)</code>, <code>[TransformerMixin](#sklearn.base.TransformerMixin)</code>

Hyperalignment using iterative Procrustes alignment.

Three-stage iterative process for aligning multi-subject data:
1. Create initial average template
2. Refine template through n_iter iterations
3. Final alignment of all subjects to refined template

This implements the Procrustes-based hyperalignment method commonly
used in multi-subject neuroimaging analysis.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_iter` | <code>int, default=2</code> | Number of template refinement iterations (stages 1-2). | <code>2</code>
`auto_pad` | <code>bool, default=True</code> | If True, automatically zero-pad matrices to standardize sizes. If False, caller must ensure all matrices have same dimensions. | <code>True</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_iter` | <code>int, default=2</code> | Number of template refinement iterations | <code>2</code>
`auto_pad` | <code>bool, default=True</code> | Whether to automatically pad matrices to same size | <code>True</code>



######### Attributes####

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`w_` | <code>list of ndarray, element i has shape=[features_i, features]</code> | The transformation matrices (rotation + reflection) for each subject.
`s_` | <code>ndarray, shape=[features, samples]</code> | The aligned common template (shared response).
`disparity_` | <code>list of float</code> | Disparity (sum of squared differences) for each subject.
`scale_` | <code>list of float</code> | Scale factors for each subject.

<details class="note" open markdown="1">
<summary>Note</summary>

``common_model_`` property provides alias for ``s_`` (backward compatibility).

</details>

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Fit hyperalignment model to data.
[`transform`](#algorithms-transform) | Transform data to common space using fitted transformations.
[`transform_subject`](#algorithms-transform-subject) | Align a new subject to the common space.

**Examples:**

Basic multi-subject alignment:

```pycon
>>> from nltools.algorithms import HyperAlignment
>>> import numpy as np
>>>
>>> # Create sample data (3 subjects)
>>> data = [np.random.randn(100, 50) for _ in range(3)]
>>>
>>> # Fit hyperalignment with CPU parallelization (default)
>>> hyper = HyperAlignment(n_iter=2)
>>> hyper.fit(data, parallel="cpu", n_jobs=-1)
>>>
>>> # Transform to common space
>>> aligned = hyper.transform(data)
>>>
>>> # Access common template
>>> template = hyper.s_  # or hyper.common_model_
>>>
>>> # Align a new subject
>>> new_subject = np.random.randn(100, 50)
>>> new_transform = hyper.transform_subject(new_subject)
```

<details class="note" open markdown="1">
<summary>Note</summary>

When to use parallel processing:

- Use ``parallel="cpu"`` (default) for datasets with 3+ subjects to speed up
  pairwise Procrustes operations during template refinement.
- Use ``parallel=None`` for debugging or small datasets (<3 subjects) where
  parallelization overhead isn't beneficial.
- Parallel processing is most beneficial when subjects have many voxels
  (>10K) and template refinement requires multiple iterations.

</details>

<details class="references" open markdown="1">
<summary>References</summary>

Haxby, J. V., Guntupalli, J. S., Connolly, A. C., Halchenko, Y. O.,
Conroy, B. R., Gobbini, M. I., ... & Ramadge, P. J. (2011).
A common, high-dimensional model of the representational space in
human ventral temporal cortex. Neuron, 72(2), 404-416.

</details>

###### `auto_pad`

```python
auto_pad = auto_pad
```

########## `common_model_`

```python
common_model_
```

Alias for ``s_`` (common template).

########## `n_iter`

```python
n_iter = n_iter
```



######### Functions####

###### `fit`

```python
fit(data: list[np.ndarray], *, parallel: str | None = 'cpu', n_jobs: int = -1) -> HyperAlignment
```

Fit hyperalignment model to data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>list of ndarray</code> | List of data matrices, each with shape (n_features, n_samples). Different subjects can have different numbers of features if auto_pad=True. | *required*
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>list of ndarray</code> | List of data matrices to transform. Should be the same data used for fitting (or have compatible dimensions). | *required*
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`subject_data` | <code>([ndarray](#ndarray), [shape](#shape)([n_features](#n_features), [n_samples](#n_samples)))</code> | Data from a new subject to align to the common template | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`self` | <code>[HyperAlignment](#nltools.algorithms.alignment.hyperalignment.HyperAlignment)</code> | Fitted model

########## `transform`

```python
transform(data: list[np.ndarray], *, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray]
```

Transform data to common space using fitted transformations.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`transformed` | <code>list of ndarray</code> | List of transformed data matrices in common space

########## `transform_subject`

```python
transform_subject(subject_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]
```

Align a new subject to the common space.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`transformed` | <code>[ndarray](#ndarray)</code> | Aligned data in common space
`R` | <code>[ndarray](#ndarray)</code> | Transformation matrix used
`disparity` | <code>[float](#float)</code> | Alignment quality (sum of squared differences)
`scale` | <code>[float](#float)</code> | Scale factor used

(algorithms-local)=
###### `local`

LocalAlignment: Neighborhood-based functional alignment.

Implements the ``'searchlight'`` and ``'roi'`` spatial scales (the searchlight
and piecewise schemes of Bazeille et al. 2021). Uses center-only aggregation to
preserve orthogonality of local transforms.

**Classes:**

Name | Description
---- | -----------
[`LocalAlignment`](#algorithms-localalignment) | Local (neighborhood-based) functional alignment across subjects.



####### Attributes

####### Classes##

###### `LocalAlignment`

```python
LocalAlignment(spatial_scale: str = 'searchlight', method: str = 'procrustes', radius_mm: float = 10.0, roi_mask: nib.Nifti1Image | None = None, n_features: int | None = None, n_iter: int = 3, aggregation: str = 'center', parallel: str | None = 'cpu', n_jobs: int = -1, progress_bar: bool = False, n_neighborhoods_batch: int | None = None, max_memory_gb: float | None = None, transforms_: dict[int, list[np.ndarray]] | None = None, template_: dict[int, np.ndarray] | None = None, neighborhoods_: SphereNeighborhoods | dict[int, np.ndarray] | None = None, n_voxels_: int | None = None, mask_: nib.Nifti1Image | None = None, backend_: Backend | None = None) -> None
```

Local (neighborhood-based) functional alignment across subjects.

Learns alignment transforms within local neighborhoods (searchlight spheres
or parcels) and applies center-only aggregation to preserve orthogonality.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`spatial_scale` | <code>[str](#str)</code> | Spatial scale, either 'searchlight' (overlapping spheres) or 'roi' (non-overlapping parcels). Defaults to 'searchlight'. | <code>'searchlight'</code>
`method` | <code>[str](#str)</code> | Alignment method, one of 'procrustes', 'srm', or 'hyperalignment'. Defaults to 'procrustes'. | <code>'procrustes'</code>
`radius_mm` | <code>[float](#float)</code> | Sphere radius in millimeters for the searchlight scale. Defaults to 10.0. | <code>10.0</code>
`roi_mask` | <code>[Nifti1Image](#Nifti1Image) \| None</code> | Parcellation image for the ROI scale. Required if `spatial_scale='roi'`. Defaults to None. | <code>None</code>
`n_features` | <code>[int](#int) \| None</code> | Number of features for SRM. None uses full Procrustes (preserves dims). Defaults to None. | <code>None</code>
`n_iter` | <code>[int](#int)</code> | Number of iterations for alignment refinement. Defaults to 3. | <code>3</code>
`aggregation` | <code>[str](#str)</code> | Aggregation method: 'center' (center-only, preserves orthogonality) or 'all'. Defaults to 'center'. | <code>'center'</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization mode. None runs single-threaded numpy, 'cpu' uses joblib CPU parallelization, and 'gpu' uses PyTorch. GPU acceleration applies only to `method='procrustes'`; requesting 'gpu' with the 'srm' or 'hyperalignment' methods raises `NotImplementedError` (an explicit GPU request never silently runs on CPU). Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of jobs for CPU parallelization. Defaults to -1. | <code>-1</code>
`progress_bar` | <code>[bool](#bool)</code> | Whether to display tqdm progress bars during fit and transform. Defaults to False. | <code>False</code>
`n_neighborhoods_batch` | <code>[int](#int) \| None</code> | Number of neighborhoods to process per batch on the GPU. None auto-calculates a batch size from `max_memory_gb`. Defaults to None. | <code>None</code>
`max_memory_gb` | <code>[float](#float) \| None</code> | Explicit memory budget (in GB) used to auto-size GPU batches when `n_neighborhoods_batch` is None. None (default) measures the device's available memory. | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`transforms_` | <code>[dict](#dict)[[int](#int), [list](#list)[[ndarray](#numpy.ndarray)]]</code> | Per-neighborhood transforms. Keys are center voxel indices, values are lists of transform matrices (one per subject).
`template_` | <code>[dict](#dict)[[int](#int), [ndarray](#numpy.ndarray)]</code> | Per-neighborhood templates used for alignment.
`neighborhoods_` | <code>[SphereNeighborhoods](#nltools.data.braindata.neighborhoods.SphereNeighborhoods) \| [dict](#dict)</code> | Computed neighborhoods (searchlight or roi).
`n_voxels_` | <code>[int](#int)</code> | Total number of voxels in the mask.
`mask_` | <code>[Nifti1Image](#Nifti1Image)</code> | Brain mask used for fitting.

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Fit local alignment on multi-subject data.
[`fit_transform`](#algorithms-fit-transform) | Fit alignment and transform data in one step.
[`transform`](#algorithms-transform) | Apply local transforms to data.



######### Attributes####

**Examples:**

```pycon
>>> import numpy as np
>>> import nibabel as nib
>>> from nltools.algorithms.alignment import LocalAlignment
>>> # Create synthetic multi-subject data (voxels, samples)
>>> data = [np.random.randn(1000, 100) for _ in range(5)]
>>> # Build a mask whose nonzero voxels match the 1000-voxel data
>>> mask = nib.Nifti1Image(np.ones((10, 10, 10), dtype=np.int8), np.eye(4))
>>> la = LocalAlignment(spatial_scale='searchlight', method='procrustes', radius_mm=10.0)
>>> la.fit(data, mask)
>>> aligned = la.transform(data)
```

<details class="note" open markdown="1">
<summary>Note</summary>

Based on Bazeille et al. 2021 "An empirical evaluation of functional
alignment using inter-subject decoding". Center-only aggregation is
used to preserve local orthogonality of transforms.

</details>

###### `aggregation`

```python
aggregation: str = 'center'
```

########## `backend_`

```python
backend_: Backend | None = field(default=None, repr=False)
```

########## `mask_`

```python
mask_: nib.Nifti1Image | None = field(default=None, repr=False)
```

########## `max_memory_gb`

```python
max_memory_gb: float | None = None
```

########## `method`

```python
method: str = 'procrustes'
```

########## `n_features`

```python
n_features: int | None = None
```

########## `n_iter`

```python
n_iter: int = 3
```

########## `n_jobs`

```python
n_jobs: int = -1
```

########## `n_neighborhoods_batch`

```python
n_neighborhoods_batch: int | None = None
```

########## `n_voxels_`

```python
n_voxels_: int | None = field(default=None, repr=False)
```

########## `neighborhoods_`

```python
neighborhoods_: SphereNeighborhoods | dict[int, np.ndarray] | None = field(default=None, repr=False)
```

########## `parallel`

```python
parallel: str | None = 'cpu'
```

########## `progress_bar`

```python
progress_bar: bool = False
```

########## `radius_mm`

```python
radius_mm: float = 10.0
```

########## `roi_mask`

```python
roi_mask: nib.Nifti1Image | None = None
```

########## `spatial_scale`

```python
spatial_scale: str = 'searchlight'
```

########## `template_`

```python
template_: dict[int, np.ndarray] | None = field(default=None, repr=False)
```

########## `transforms_`

```python
transforms_: dict[int, list[np.ndarray]] | None = field(default=None, repr=False)
```



######### Functions####

###### `fit`

```python
fit(data: list[np.ndarray], mask: nib.Nifti1Image) -> LocalAlignment
```

Fit local alignment on multi-subject data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | List of subject data arrays, each shape (n_voxels, n_samples). Subjects can have different numbers of samples - the underlying alignment methods (SRM, HyperAlignment) handle this via zero-padding. | *required*
`mask` | <code>[Nifti1Image](#Nifti1Image)</code> | Brain mask defining the voxel space. | *required*

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | List of subject data arrays, each shape (n_voxels, n_samples). | *required*
`mask` | <code>[Nifti1Image](#Nifti1Image)</code> | Brain mask defining the voxel space. | *required*

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | List of subject data arrays, each shape (n_voxels, n_samples). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`LocalAlignment` | <code>[LocalAlignment](#nltools.algorithms.alignment.local.LocalAlignment)</code> | The fitted alignment model (`self`).

########## `fit_transform`

```python
fit_transform(data: list[np.ndarray], mask: nib.Nifti1Image) -> list[np.ndarray]
```

Fit alignment and transform data in one step.

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | list[np.ndarray]: Aligned data for each subject.

########## `transform`

```python
transform(data: list[np.ndarray]) -> list[np.ndarray]
```

Apply local transforms to data.

For the searchlight scale with center-only aggregation: each voxel uses
the transform from the neighborhood where it was the center.

For the roi scale: all voxels in each parcel use the same transform.

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | list[np.ndarray]: Aligned data for each subject, each shape (n_voxels, n_samples).



####### Functions

(algorithms-procrustes)=
###### `procrustes`

Data alignment — SRM, Procrustes, and state alignment.

**Methods:**

Name | Description
---- | -----------
[`align`](#algorithms-align) | Align subject data into a common response model.
[`align_states`](#algorithms-align-states) | Align state weight maps by minimizing pairwise distance between group states.
[`procrustes`](#algorithms-procrustes) | Perform a Procrustes similarity analysis on two data sets.
[`procrustes_distance`](#algorithms-procrustes-distance) | Test matrix similarity using Procrustes superposition.



####### Classes

####### Functions##

###### `align`

```python
align(data, method = 'deterministic_srm', n_features = None, axis = 0, *args, **kwargs)
```

Align subject data into a common response model.

This function is a convenience wrapper around `HyperAlignment` and `SRM` classes.

Can be used to hyperalign source data to target data using
Hyperalignment from Dartmouth (i.e., procrustes transformation; see
nltools.algorithms.procrustes) or Shared Response Model from Princeton (see
nltools.algorithms.srm). (see nltools.data.BrainData.align for aligning
a single Brain object to another). Common Model is shared response
model or centered target data. Transformed data can be back projected to
original data using Tranformation matrix. Inputs must be a list of BrainData
instances or numpy arrays (observations by features).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (list) A list of BrainData objects | *required*
`method` |  | (str) alignment method to use ['probabilistic_srm','deterministic_srm','procrustes'] | <code>'deterministic_srm'</code>
`n_features` |  | (int) number of features to align to common space. If None then will select number of voxels | <code>None</code>
`axis` |  | (int) axis to align on | <code>0</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`reference` |  | (np.array) reference pattern x state matrix | *required*
`target` |  | (np.array) target pattern x state matrix to align to reference | *required*
`metric` |  | (str) distance metric to use | <code>'correlation'</code>
`return_index` |  | (bool) return index if True, return remapped data if False | <code>False</code>
`replace_zero_variance` |  | (bool) transform a vector with zero variance to random numbers from a uniform distribution.                     Useful for when using correlation as a distance metric to avoid NaNs. | <code>False</code>

Returns:
    If ``return_index=False`` (default): ``target[:, remapping]``, a single
    ndarray of the target's columns reordered to match the reference,
    oriented pattern x state (same shape as ``target``).
    If ``return_index=True``: the remapping index array (ndarray) that
    reorders the target's state columns.

######## `procrustes`

```python
procrustes(data1, data2)
```

Perform a Procrustes similarity analysis on two data sets.

For more comprehensive Procrustes-based alignment tasks, use
`HyperAlignment` and `align()` instead.

Each input matrix is a set of points or vectors (the rows of the matrix).
The dimension of the space is the number of columns of each matrix. Given
two identically sized matrices, procrustes standardizes both such that:
- $tr(AA^{T}) = 1$.
- Both sets of points are centered around the origin.
Procrustes then applies the optimal transform to the second
matrix (including scaling/dilation, rotations, and reflections) to minimize
$M^{2}=\sum(data1-data2)^{2}$, or the sum of the squares of the
pointwise differences between the two input datasets.
This function was not designed to handle datasets with different numbers of
datapoints (rows).  If two data sets have different dimensionality
(different number of columns), this function will add columns of zeros to
the smaller of the two.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` |  | Matrix whose n rows represent points in k (columns) space. `data1` is the reference data; after it is standardized, the data from `data2` will be transformed to fit the pattern in `data1` (must have >1 unique points). | *required*
`data2` |  | n rows of data in k space to be fit to `data1`. Must be the same shape `(numrows, numcols)` as `data1` (must have >1 unique points). | *required*

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mat1` | <code>[ndarray](#ndarray)</code> | 2d numpy array; must have same number of rows as mat2 | *required*
`mat2` | <code>[ndarray](#ndarray)</code> | 1d or 2d numpy array; must have same number of rows as mat1 | *required*
`n_permute` | <code>[int](#int)</code> | number of permutation iterations to perform | <code>5000</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | 2|'two' (two-tailed, default) or 1|'one' (one-tailed: similarity > chance) | <code>2</code>
`n_jobs` | <code>[int](#int)</code> | The number of CPUs to use to do permutation; default -1 (all) | <code>-1</code>
`random_state` | <code>int, np.random.RandomState, or None</code> | seed or generator for the permutation shuffling; default None | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (dict) a dictionary containing a list of transformed subject matrices, a list of transformation matrices, the shared response matrix, and the intersubject correlation of the shared responses

**Examples:**

- Hyperalign using procrustes transform:
    >>> out = align(data, method='procrustes')
- Align using shared response model:
    >>> out = align(data, method='probabilistic_srm', n_features=None)
- Project aligned data into original data:
    >>> original_data = [np.dot(t.data,tm.T) for t,tm in zip(out['transformed'], out['transformation_matrix'])]

######## `align_states`

```python
align_states(reference, target, *, metric = 'correlation', return_index = False, replace_zero_variance = False)
```

Align state weight maps by minimizing pairwise distance between group states.

This function uses the Hungarian algorithm for state alignment, which is
different from aligning multiple subjects' data.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`mtx1` |  | A standardized version of `data1`.
`mtx2` |  | The orientation of `data2` that best fits `data1`. Centered, but not necessarily $tr(AA^{T}) = 1$.
`disparity` |  | $M^{2}$ as defined above.
`R` |  | The `(N, N)` matrix solution of the orthogonal Procrustes problem. Minimizes the Frobenius norm of `dot(data1, R) - data2`, subject to `dot(R.T, R) == I`.
`scale` |  | Sum of the singular values of `dot(data1.T, data2)`.

######## `procrustes_distance`

```python
procrustes_distance(mat1, mat2, *, n_permute = 5000, tail = 2, n_jobs = -1, random_state = None)
```

Test matrix similarity using Procrustes superposition.

Matrices need to match in size on their first dimension only, as the smaller
matrix on the second dimension will be padded with zeros. After aligning two
matrices using the Procrustes transformation, use the computed disparity
between them (sum of squared error of elements) as a similarity metric.
Shuffle the rows of one of the matrices and recompute the disparity to perform
inference (Peres-Neto & Jackson, 2001).

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | results with keys `similarity` (float in [0, 1]) and `p` (permuted p-value)

###### `srm`

Shared Response Model (SRM) for multi-subject fMRI alignment.

SRM finds a shared low-dimensional representation across subjects while
allowing subject-specific transformations. This enables cross-subject
analyses while preserving individual variability.

<details class="algorithm-overview" open markdown="1">
<summary>Algorithm overview</summary>

1. Initialize subject-specific transforms W_i (random orthogonal matrices)
2. Iteratively optimize using Expectation-Maximization (EM):
   - E-step: Update shared response S (group average in shared space)
   - M-step: Update subject transforms W_i (solve Procrustes problem)
   - Update noise variance rho_i^2 per subject
   - Compute likelihood (measure of fit)
3. Converge when likelihood stabilizes or max iterations reached

</details>

<details class="performance" open markdown="1">
<summary>Performance</summary>

- Time complexity: O(n_iter × (n_subjects × n_voxels × n_features × n_samples + n_features^3))
- Memory complexity: O(n_subjects × n_voxels × n_features)
- Parallelization: ~4-8× speedup with CPU-parallel (parallel="cpu")
- GPU acceleration: Falls back to CPU (not yet implemented)

</details>

<details class="when-to-use-srm" open markdown="1">
<summary>When to use SRM</summary>

- Multi-subject alignment preserving representational structure
- Cross-subject analysis requiring shared response space
- Alternative to hyperalignment when spatial structure is less important
- See `nltools.algorithms.hyperalignment.HyperAlignment` for spatial-preserving alignment

</details>

The implementations are based on the following publications:

Chen, P. H. C., Chen, J., Yeshurun, Y., Hasson, U., Haxby, J., & Ramadge,
P. J. (2015). A reduced-dimension fMRI shared response model. In Advances
in Neural Information Processing Systems (pp. 460-468).

Anderson, M. J., Capota, M., Turek, J. S., Zhu, X., Willke, T. L., Wang,
Y., & Norman, K. A. (2016, December). Enabling factor analysis on
thousand-subject neuroimaging datasets. In Big Data (Big Data),
2016 IEEE International Conference on (pp. 1151-1160). IEEE.

References:
- **Chen2015:** Chen, P. H. C., Chen, J., Yeshurun, Y., Hasson, U., Haxby, J.,
   & Ramadge, P. J. (2015). A reduced-dimension fMRI shared response model.
   In Advances in Neural Information Processing Systems (pp. 460-468).

- **Anderson2016:** Anderson, M. J., Capota, M., Turek, J. S., Zhu, X.,
   Willke, T. L., Wang, Y., & Norman, K. A. (2016, December). Enabling
   factor analysis on thousand-subject neuroimaging datasets. In Big Data
   (Big Data), 2016 IEEE International Conference on (pp. 1151-1160). IEEE.

Copyright 2016 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

**Classes:**

Name | Description
---- | -----------
[`DetSRM`](#algorithms-detsrm) | Deterministic Shared Response Model (DetSRM).
[`SRM`](#algorithms-srm) | Probabilistic Shared Response Model (SRM).



####### Attributes

####### Classes##

###### `DetSRM`

```python
DetSRM(*, n_iter: int = 10, features: int = 50, rand_seed: int = 0) -> None
```

Bases: <code>[BaseEstimator](#sklearn.base.BaseEstimator)</code>, <code>[TransformerMixin](#sklearn.base.TransformerMixin)</code>

Deterministic Shared Response Model (DetSRM).

Given multi-subject data, factorize it as a shared response S among all
subjects and an orthogonal transform W per subject:

$$
X_i \approx W_i S, \forall i=1 \dots N
$$

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_iter` | <code>int, default=10</code> | Number of iterations to run the algorithm. | <code>10</code>
`features` | <code>int, default=50</code> | Number of features to compute. | <code>50</code>
`rand_seed` | <code>int, default=0</code> | Seed for initializing the random number generator. | <code>0</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`w_` | <code>list of array, element i has shape=[voxels_i, features]</code> | The orthogonal transforms (mappings) for each subject.
`s_` | <code>array, shape=[features, samples]</code> | The shared response.
`random_state_` | <code>`RandomState`</code> | Random number generator initialized using rand_seed

<details class="note" open markdown="1">
<summary>Note</summary>

The number of voxels may be different between subjects. However, the
number of samples must be the same across subjects.

The Deterministic Shared Response Model is approximated using the
Block Coordinate Descent (BCD) algorithm proposed in **Chen2015**.

This is a single node version.

The run-time complexity is $O(I (V T K + V K^2))$ and the memory
complexity is $O(V T)$ with I - the number of iterations, V - the
sum of voxels from all subjects, T - the number of samples, K - the
number of features (typically, $V \gg T \gg K$), and N - the
number of subjects.

</details>

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Compute the Deterministic Shared Response Model.
[`transform`](#algorithms-transform) | Use the model to transform data to the Shared Response subspace.
[`transform_subject`](#algorithms-transform-subject) | Transform a new subject using the existing model.



######### Attributes####

**Examples:**

Basic multi-subject DetSRM fitting:

```pycon
>>> from nltools.algorithms import DetSRM
>>> import numpy as np
>>>
>>> # Create sample data (3 subjects)
>>> data = [np.random.randn(100, 50) for _ in range(3)]
>>>
>>> # Fit DetSRM with CPU parallelization (default)
>>> detsrm = DetSRM(n_iter=10, features=50)
>>> detsrm.fit(data, parallel="cpu", n_jobs=-1)
>>>
>>> # Transform to shared response space
>>> shared_responses = detsrm.transform(data)
>>>
>>> # Access fitted model components
>>> w = detsrm.w_  # Subject-specific transforms
>>> s = detsrm.s_  # Shared response
```

###### `features`

```python
features = features
```

########## `n_iter`

```python
n_iter = n_iter
```

########## `rand_seed`

```python
rand_seed = rand_seed
```



######### Functions####

###### `fit`

```python
fit(X: list[np.ndarray], y: Any | None = None, *, parallel: str | None = 'cpu', n_jobs: int = -1) -> DetSRM
```

Compute the Deterministic Shared Response Model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples]</code> | Each element in the list contains the fMRI data of one subject. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": not implemented -- raises `NotImplementedError` (never a silent CPU fallback) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples_i]</code> | Each element in the list contains the fMRI data of one subject. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": not implemented -- raises `NotImplementedError` (never a silent CPU fallback) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>2D array, shape=[voxels, timepoints]</code> | The fMRI data of the new subject. | *required*

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_iter` | <code>int, default=10</code> | Number of iterations to run the algorithm. | <code>10</code>
`features` | <code>int, default=50</code> | Number of features to compute. | <code>50</code>
`rand_seed` | <code>int, default=0</code> | Seed for initializing the random number generator. | <code>0</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`w_` | <code>list of array, element i has shape=[voxels_i, features]</code> | The orthogonal transforms (mappings) for each subject.
`s_` | <code>array, shape=[features, samples]</code> | The shared response.
`sigma_s_` | <code>array, shape=[features, features]</code> | The covariance of the shared response Normal distribution.
`mu_` | <code>list of array, element i has shape=[voxels_i]</code> | The voxel means over the samples for each subject.
`rho2_` | <code>array, shape=[subjects]</code> | The estimated noise variance $\rho_i^2$ for each subject
`random_state_` | <code>`RandomState`</code> | Random number generator initialized using rand_seed

<details class="note" open markdown="1">
<summary>Note</summary>

The number of voxels may be different between subjects. However, the
number of samples must be the same across subjects.

The probabilistic Shared Response Model is approximated using the
Expectation Maximization (EM) algorithm proposed in **Chen2015**. The
implementation follows the optimizations published in **Anderson2016**.

This is a single node version.

The run-time complexity is $O(I (V T K + V K^2 + K^3))$ and the
memory complexity is $O(V T)$ with I - the number of iterations,
V - the sum of voxels from all subjects, T - the number of samples, and
K - the number of features (typically, $V \gg T \gg K$).

</details>

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Compute the probabilistic Shared Response Model.
[`transform`](#algorithms-transform) | Use the model to transform matrix to Shared Response space.
[`transform_subject`](#algorithms-transform-subject) | Transform a new subject using the existing model.



######### Attributes####

**Returns:**

Name | Type | Description
---- | ---- | -----------
`self` | <code>[DetSRM](#nltools.algorithms.alignment.srm.DetSRM)</code> | Fitted model

########## `transform`

```python
transform(X: list[np.ndarray], y: Any | None = None, *, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray]
```

Use the model to transform data to the Shared Response subspace.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`s` | <code>list of 2D arrays, element i has shape=[features_i, samples_i]</code> | Shared responses from input data (X)

########## `transform_subject`

```python
transform_subject(X: np.ndarray) -> np.ndarray
```

Transform a new subject using the existing model.

The subject is assumed to have received equivalent stimulation.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`w` | <code>2D array, shape=[voxels, features]</code> | Orthogonal mapping `W_{new}` for new subject

######## `SRM`

```python
SRM(*, n_iter: int = 10, features: int = 50, rand_seed: int = 0) -> None
```

Bases: <code>[BaseEstimator](#sklearn.base.BaseEstimator)</code>, <code>[TransformerMixin](#sklearn.base.TransformerMixin)</code>

Probabilistic Shared Response Model (SRM).

Given multi-subject data, factorize it as a shared response S among all
subjects and an orthogonal transform W per subject:

$$
X_i \approx W_i S, \forall i=1 \dots N
$$

**Examples:**

Basic multi-subject SRM fitting:

```pycon
>>> from nltools.algorithms import SRM
>>> import numpy as np
>>>
>>> # Create sample data (3 subjects)
>>> data = [np.random.randn(100, 50) for _ in range(3)]
>>>
>>> # Fit SRM with CPU parallelization (default)
>>> srm = SRM(n_iter=10, features=50)
>>> srm.fit(data, parallel="cpu", n_jobs=-1)
>>>
>>> # Transform to shared response space
>>> shared_responses = srm.transform(data)
>>>
>>> # Access fitted model components
>>> w = srm.w_  # Subject-specific transforms
>>> s = srm.s_  # Shared response
```

###### `features`

```python
features = features
```

########## `n_iter`

```python
n_iter = n_iter
```

########## `rand_seed`

```python
rand_seed = rand_seed
```



######### Functions####

###### `fit`

```python
fit(X: list[np.ndarray], y: Any | None = None, *, parallel: str | None = 'cpu', n_jobs: int = -1, pad_samples: bool = True) -> SRM
```

Compute the probabilistic Shared Response Model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples]</code> | Each element in the list contains the fMRI data of one subject. Subjects can have different numbers of samples if pad_samples=True. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": not implemented -- raises `NotImplementedError` (never a silent CPU fallback) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>
`pad_samples` | <code>[bool](#bool)</code> | If True (default), automatically zero-pad subjects with fewer samples to match the longest subject. This allows fitting SRM on data with unequal numbers of time points across subjects. | <code>True</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples_i]</code> | Each element in the list contains the fMRI data of one subject. Note that number of voxels and samples can vary across subjects. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used (as it is unsupervised learning) | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": not implemented -- raises `NotImplementedError` (never a silent CPU fallback) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>2D array, shape=[voxels, timepoints]</code> | The fMRI data of the new subject. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`self` | <code>[SRM](#nltools.algorithms.alignment.srm.SRM)</code> | Fitted model

########## `transform`

```python
transform(X: list[np.ndarray], y: Any | None = None, *, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray | None]
```

Use the model to transform matrix to Shared Response space.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`s` | <code>list of 2D arrays, element i has shape=[features_i, samples_i]</code> | Shared responses from input data (X)

########## `transform_subject`

```python
transform_subject(X: np.ndarray) -> np.ndarray
```

Transform a new subject using the existing model.

The subject is assumed to have received equivalent stimulation.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`w` | <code>2D array, shape=[voxels, features]</code> | Orthogonal mapping `W_{new}` for new subject

(algorithms-backends)=
#### `backends`

Backend abstraction for CPU/GPU operations.

Supports NumPy (CPU-only) and PyTorch (CPU/CUDA/MPS) backends for
linear algebra operations. Enables transparent acceleration while
maintaining NumPy-first development.

**Classes:**

Name | Description
---- | -----------
[`Backend`](#algorithms-backend) | Backend abstraction for numerical operations.

**Methods:**

Name | Description
---- | -----------
[`assert_array_almost_equal`](#algorithms-assert-array-almost-equal) | Test array equality with automatic precision adjustment for MPS backend.
[`auto_batch_size`](#algorithms-auto-batch-size) | Split `n_items` into batches that fit a memory budget.
[`auto_n_jobs_for_arrays`](#algorithms-auto-n-jobs-for-arrays) | Memory-aware joblib worker count for a per-item map over arrays.
[`auto_select_backend`](#algorithms-auto-select-backend) | Automatically select backend based on problem size.
[`check_gpu_available`](#algorithms-check-gpu-available) | Check if GPU acceleration is available.
[`compute_oom_safe`](#algorithms-compute-oom-safe) | Run `fn(*arrays)` with reactive out-of-memory recovery.
[`device_memory_budget`](#algorithms-device-memory-budget) | Usable memory budget in GB for a backend's device.
[`empty_device_cache`](#algorithms-empty-device-cache) | Release cached device memory. No-op without torch or a GPU.
[`gb_to_bytes`](#algorithms-gb-to-bytes) | Convert a GB budget to bytes — the package's one GB↔bytes conversion.
[`is_oom_error`](#algorithms-is-oom-error) | True if `exc` is a device out-of-memory error (CUDA or MPS).
[`resolve_backend`](#algorithms-resolve-backend) | Coerce a backend specifier into a `Backend` instance.



##### Classes

(algorithms-backend)=
###### `Backend`

```python
Backend(backend: str = 'numpy')
```

Backend abstraction for numerical operations.

Provides a unified interface for NumPy and PyTorch operations,
enabling transparent GPU acceleration when available.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`backend` | <code>[str](#str)</code> | Backend type: 'numpy', 'torch', or 'auto' - 'numpy': CPU-only using NumPy - 'torch': PyTorch with automatic device detection (cuda/mps/cpu) - 'auto': Automatically select best available backend | <code>'numpy'</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`name` | <code>[str](#str)</code> | Backend identifier (e.g., 'numpy', 'torch-cuda', 'torch-mps')
`device` | <code>[str](#str)</code> | Device type ('cpu', 'cuda', or 'mps')
`xp` | <code>[module](#module)</code> | Array library module (numpy or torch)

**Methods:**

Name | Description
---- | -----------
[`asarray`](#algorithms-asarray) | Convert input to a backend array.
`asarray_like` | Convert *x* to an array matching *ref*'s dtype (and device for torch).
`check_arrays` | Coerce all inputs to the same dtype (and device) as the first.
`concatenate` | Concatenate arrays along an axis.
`copy` | Return an independent copy of the array.
`dtype_to_str` | Normalize a dtype (numpy, torch, or string) to its string name.
`expand_dims` | Insert a new axis.
`flatnonzero` | Return indices of non-zero elements in the flattened array.
`full` | Create array filled with *fill_value*.
`full_like` | Create array filled with *fill_value*, optionally with a different shape.
`matmul` | Matrix multiplication.
`ones_like` | Create ones array, optionally with a different shape.
`sort` | Sort along an axis, returning values only.
`svd` | Compute Singular Value Decomposition.
`to_cpu` | Transfer array to CPU. No-op for numpy.
`to_device` | Transfer array to backend device.
`to_gpu` | Transfer array to GPU. No-op for numpy.
`to_numpy` | Convert array back to NumPy.
`zeros_like` | Create zeros array, optionally with a different shape.



####### Attributes##

(algorithms-is-gpu)=
###### `is_gpu`

```python
is_gpu
```

True if backend is using a GPU device (CUDA or MPS).



####### Functions##

(algorithms-asarray)=
###### `asarray`

```python
asarray(x, dtype = None, device = None)
```

Convert input to a backend array.

Handles numpy arrays, lists, and torch tensors. Places result on
the backend's device (or an explicit *device*).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` |  | Input data (array-like, tensor, list). | *required*
`dtype` |  | Desired dtype as string, numpy, or torch dtype. If None, inferred from input. | <code>None</code>
`device` |  | Target device string (e.g. "cpu", "cuda"). Ignored for numpy backend. If None, uses the backend's default device. | <code>None</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` |  | Input data. | *required*
`ref` |  | Reference array whose dtype/device to match. | *required*

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`*inputs` |  | Arrays, lists of arrays, or None. | <code>()</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`arrays` |  | Sequence of arrays. | *required*
`axis` |  | Axis to concatenate along (default 0). | <code>0</code>

######## `copy`

```python
copy(array)
```

Return an independent copy of the array.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Input array. | *required*

######## `dtype_to_str`

```python
dtype_to_str(dtype)
```

Normalize a dtype (numpy, torch, or string) to its string name.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`dtype` |  | Data type to convert (str, numpy dtype, torch dtype, or None). | *required*

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Input array. | *required*
`axis` |  | Position of the new axis. | *required*

######## `flatnonzero`

```python
flatnonzero(array)
```

Return indices of non-zero elements in the flattened array.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Input array. | *required*

######## `full`

```python
full(shape, fill_value, dtype = None)
```

Create array filled with *fill_value*.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`shape` |  | Output shape (int or tuple). | *required*
`fill_value` |  | Scalar fill value. | *required*
`dtype` |  | Output dtype. If None, inferred by the backend. | <code>None</code>

######## `full_like`

```python
full_like(array, fill_value, shape = None, dtype = None, device = None)
```

Create array filled with *fill_value*, optionally with a different shape.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Reference array for dtype inference. | *required*
`fill_value` |  | Scalar fill value. | *required*
`shape` |  | Output shape. If None, uses array.shape. | <code>None</code>
`dtype` |  | Output dtype. If None, uses array.dtype. | <code>None</code>
`device` |  | Target device (torch only). If None, uses array's device. | <code>None</code>

######## `matmul`

```python
matmul(A, B)
```

Matrix multiplication.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`A` | <code>[array](#array)</code> | First matrix | *required*
`B` | <code>[array](#array)</code> | Second matrix | *required*

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Reference array for dtype inference. | *required*
`shape` |  | Output shape. If None, uses array.shape. | <code>None</code>
`dtype` |  | Output dtype. If None, uses array.dtype. | <code>None</code>
`device` |  | Target device (torch only). If None, uses array's device. | <code>None</code>

######## `sort`

```python
sort(array, axis = -1)
```

Sort along an axis, returning values only.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Input array. | *required*
`axis` |  | Axis to sort along (default -1). | <code>-1</code>

######## `svd`

```python
svd(X, full_matrices = False)
```

Compute Singular Value Decomposition.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[array](#array)</code> | Input matrix (n_samples, n_features) | *required*
`full_matrices` | <code>bool, default=False</code> | If False, returns reduced SVD | <code>False</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Input array or tensor. | *required*

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`arr` | <code>[ndarray](#numpy.ndarray)</code> | Input numpy array | *required*

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Input array or tensor. | *required*
`device` |  | Target device (defaults to backend's device). | <code>None</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`arr` | <code>[ndarray](#numpy.ndarray) or [Tensor](#torch.Tensor)</code> | Array to convert | *required*

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Reference array for dtype inference. | *required*
`shape` |  | Output shape. If None, uses array.shape. | <code>None</code>
`dtype` |  | Output dtype. If None, uses array.dtype. | <code>None</code>
`device` |  | Target device (torch only). If None, uses array's device. | <code>None</code>



**Returns:**

Type | Description
---- | -----------
 | Backend array (numpy ndarray or torch Tensor).

######## `asarray_like`

```python
asarray_like(x, ref)
```

Convert *x* to an array matching *ref*'s dtype (and device for torch).

**Returns:**

Type | Description
---- | -----------
 | Backend array with same dtype/device as ref.

######## `check_arrays`

```python
check_arrays(*inputs)
```

Coerce all inputs to the same dtype (and device) as the first.

None values are passed through. Lists of arrays are converted
element-wise.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`list` |  | Converted arrays in the same order as inputs.

######## `concatenate`

```python
concatenate(arrays, axis = 0)
```

Concatenate arrays along an axis.

**Returns:**

Type | Description
---- | -----------
 | str or None: e.g. "float32", "float64", or None if input was None.

######## `expand_dims`

```python
expand_dims(array, axis)
```

Insert a new axis.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`array` |  | Result of A @ B

######## `ones_like`

```python
ones_like(array, shape = None, dtype = None, device = None)
```

Create ones array, optionally with a different shape.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` |  | (U, s, Vt) where: - U (array): Left singular vectors - s (array): Singular values - Vt (array): Right singular vectors (transposed)

######## `to_cpu`

```python
to_cpu(array)
```

Transfer array to CPU. No-op for numpy.

**Returns:**

Type | Description
---- | -----------
 | Array on CPU.

######## `to_device`

```python
to_device(arr: np.ndarray)
```

Transfer array to backend device.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`array` |  | Array on device (numpy array or torch tensor)

######## `to_gpu`

```python
to_gpu(array, device = None)
```

Transfer array to GPU. No-op for numpy.

**Returns:**

Type | Description
---- | -----------
 | Array on GPU device.

######## `to_numpy`

```python
to_numpy(arr)
```

Convert array back to NumPy.

**Returns:**

Type | Description
---- | -----------
 | np.ndarray: NumPy array

######## `zeros_like`

```python
zeros_like(array, shape = None, dtype = None, device = None)
```

Create zeros array, optionally with a different shape.

##### Methods

(algorithms-assert-array-almost-equal)=
###### `assert_array_almost_equal`

```python
assert_array_almost_equal(x, y, decimal = 6, err_msg = '', verbose = True, backend = None)
```

Test array equality with automatic precision adjustment for MPS backend.

This utility automatically reduces precision expectations for torch-mps backend
due to float32 precision limitations, preventing test failures while maintaining
realistic precision checks for other backends.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` |  | First array to compare | *required*
`y` |  | Second array to compare | *required*
`decimal` |  | Desired decimal precision (default: 6) | <code>6</code>
`err_msg` |  | Error message prefix | <code>''</code>
`verbose` |  | Whether to print detailed error messages | <code>True</code>
`backend` |  | Backend instance (optional). If None, attempts to detect from x/y. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | None (raises AssertionError if arrays don't match)

(algorithms-auto-batch-size)=
###### `auto_batch_size`

```python
auto_batch_size(n_items: int, bytes_per_item: float, *, budget_gb: float, overhead: float = 1.0, min_batch: int = 1) -> tuple[int, int]
```

Split `n_items` into batches that fit a memory budget.

The one batch calculator for the package. Callers supply only the
per-item working-set estimate (`bytes_per_item`) and an algorithm's
allocation `overhead` factor; the clamp/ceil policy lives here.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_items` | <code>[int](#int)</code> | Total number of items (permutations, targets, ...). | *required*
`bytes_per_item` | <code>[float](#float)</code> | Dominant working-set size of one item in bytes. | *required*
`budget_gb` | <code>[float](#float)</code> | Memory budget from `device_memory_budget`. | *required*
`overhead` | <code>[float](#float)</code> | Multiplier for intermediate allocations (e.g. 3.0 when the computation holds ~3x the input working set). | <code>1.0</code>
`min_batch` | <code>[int](#int)</code> | Smallest batch worth dispatching (amortizes launch and transfer overhead). Never exceeds `n_items`. | <code>1</code>

**Returns:**

Type | Description
---- | -----------
<code>[int](#int)</code> | tuple[int, int]: `(batch_size, n_batches)` with
<code>[int](#int)</code> | `batch_size * n_batches >= n_items`.

(algorithms-auto-n-jobs-for-arrays)=
###### `auto_n_jobs_for_arrays`

```python
auto_n_jobs_for_arrays(arrays, *, max_memory_gb: float | None = None, min_jobs: int = 1) -> int
```

Memory-aware joblib worker count for a per-item map over arrays.

Sizes workers by the largest item (each worker pickles its item), using
the same measured budget as the device batching layer. None entries are
ignored; an empty list returns ``min_jobs``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`arrays` |  | Iterable of numpy arrays (None entries allowed). | *required*
`max_memory_gb` | <code>[float](#float) \| None</code> | Explicit memory budget in GB. None (default) measures available system RAM with headroom via `device_memory_budget`. | <code>None</code>
`min_jobs` | <code>[int](#int)</code> | Minimum number of workers (default: 1). | <code>1</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`int` | <code>[int](#int)</code> | Worker count for ``joblib.Parallel(n_jobs=...)``.

(algorithms-auto-select-backend)=
###### `auto_select_backend`

```python
auto_select_backend(n_samples: int, n_features: int, cv: int = 1) -> Backend
```

Automatically select backend based on problem size.

Uses heuristics to decide between NumPy (CPU) and PyTorch (GPU)
based on the computational workload. Small problems use NumPy
to avoid GPU transfer overhead. Large problems prefer GPU when
available.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_samples` | <code>[int](#int)</code> | Number of samples in dataset | *required*
`n_features` | <code>[int](#int)</code> | Number of features in dataset | *required*
`cv` | <code>int, default=1</code> | Number of cross-validation folds (multiplies effective size) | <code>1</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Backend` | <code>[Backend](#nltools.algorithms.backends.Backend)</code> | Selected backend instance

<details class="notes" open markdown="1">
<summary>Notes</summary>

Selection criteria:
- Small problems (< 10M elements): Use NumPy
- Large problems (> 30M elements): Use GPU if available
- Cross-validation: Prefer GPU even for medium problems

</details>

(algorithms-check-gpu-available)=
###### `check_gpu_available`

```python
check_gpu_available() -> tuple[bool, dict[str, Any]]
```

Check if GPU acceleration is available.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` | <code>[tuple](#tuple)[[bool](#bool), [dict](#dict)[[str](#str), [Any](#typing.Any)]]</code> | (available, info) where: - available (bool): True if GPU (CUDA or MPS) is available - info (dict): Dictionary with keys:     - 'backend': 'torch' or 'numpy'     - 'device': 'cpu', 'cuda', or 'mps'     - 'device_name': Human-readable device name

(algorithms-compute-oom-safe)=
###### `compute_oom_safe`

```python
compute_oom_safe(fn, *arrays, min_chunk: int = 1)
```

Run `fn(*arrays)` with reactive out-of-memory recovery.

All `arrays` must share their axis-0 length, and `fn` must map them to
a numpy array whose axis 0 corresponds row-for-row to its inputs. On a
device OOM the cache is emptied, the arrays are split in half along
axis 0, and the halves are retried recursively; partial results are
concatenated along axis 0.

Because splitting reuses the *already generated* inputs rather than
re-drawing them, recovery never changes which permutations a seeded
result is computed from — RNG-consuming input generation stays outside
this function. For a row-independent `fn` the recovered output matches
the unsplit computation to within floating-point reduction order
(backends may block reductions differently per batch shape; observed
differences are ~1 float32 ulp).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`fn` |  | Callable mapping the arrays to a numpy result (axis-0 aligned). | *required*
`*arrays` |  | Input arrays sharing axis-0 length. | <code>()</code>
`min_chunk` | <code>[int](#int)</code> | Chunk size below which an OOM is considered fatal. | <code>1</code>

**Returns:**

Type | Description
---- | -----------
 | np.ndarray: `fn`'s result, possibly assembled from retried chunks.

(algorithms-device-memory-budget)=
###### `device_memory_budget`

```python
device_memory_budget(backend: Backend | None = None, max_gpu_memory_gb: float | None = None) -> float
```

Usable memory budget in GB for a backend's device.

An explicit `max_gpu_memory_gb` always wins. Otherwise the budget is
measured at call time: free CUDA memory (with headroom) on CUDA
devices; available system RAM (with headroom) for CPU and MPS, which
share unified/system memory. When nothing can be measured the
conservative 4 GB fallback applies.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`backend` | <code>[Backend](#nltools.algorithms.backends.Backend) \| None</code> | Resolved `Backend` whose device the work runs on. None is treated as CPU. | <code>None</code>
`max_gpu_memory_gb` | <code>[float](#float) \| None</code> | Explicit budget override in GB. Must be positive. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`float` | <code>[float](#float)</code> | Budget in GB.

(algorithms-empty-device-cache)=
###### `empty_device_cache`

```python
empty_device_cache() -> None
```

Release cached device memory. No-op without torch or a GPU.

(algorithms-gb-to-bytes)=
###### `gb_to_bytes`

```python
gb_to_bytes(gb: float) -> int
```

Convert a GB budget to bytes — the package's one GB↔bytes conversion.

(algorithms-is-oom-error)=
###### `is_oom_error`

```python
is_oom_error(exc: BaseException) -> bool
```

True if `exc` is a device out-of-memory error (CUDA or MPS).

(algorithms-resolve-backend)=
###### `resolve_backend`

```python
resolve_backend(parallel)
```

Coerce a backend specifier into a `Backend` instance.

Accepts the values callers typically thread through the algorithms
package (``None``/``"cpu"`` → numpy, ``"gpu"``/``"torch"`` → torch,
``"numpy"``/``"auto"`` → their direct `Backend` constructors).
Existing `Backend` instances are returned unchanged — this is
the main reason to prefer ``resolve_backend`` over constructing a new
``Backend(...)`` at each call site: it avoids repeated device
detection/torch imports when a backend has already been chosen upstream.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`parallel` |  | Backend specifier. One of:<br>- ``None`` or ``"cpu"``: numpy backend. - ``"numpy"``, ``"torch"``, ``"auto"``: forwarded to ``Backend(...)``. - ``"gpu"``: alias for ``"torch"`` (auto-detects cuda/mps/cpu). - An existing `Backend` instance (returned as-is). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`Backend` |  | Resolved backend instance.

(algorithms-corrections)=
#### `corrections`

Multiple comparison corrections and thresholding.

**Methods:**

Name | Description
---- | -----------
[`fdr`](#algorithms-fdr) | Determine an FDR threshold for an array of p-values.
[`holm_bonf`](#algorithms-holm-bonf) | Compute Holm-Bonferroni-corrected p-values.
[`multi_threshold`](#algorithms-multi-threshold) | Threshold test image by multiple p-values from p image.
[`threshold`](#algorithms-threshold) | Threshold test image by p-value from p image.



##### Methods

###### `fdr`

```python
fdr(p, q = 0.05)
```

Determine an FDR threshold for an array of p-values.

Uses the desired false discovery rate ``q``. Written by Tal Yarkoni.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`p` |  | (np.array) vector of p-values | *required*
`q` |  | (float) false discovery rate level | <code>0.05</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`fdr_p` |  | (float) p-value threshold based on independence or positive     dependence

###### `holm_bonf`

```python
holm_bonf(p, alpha = 0.05)
```

Compute Holm-Bonferroni-corrected p-values.

This step-down procedure applies iteratively less correction to the highest
p-values. It is a bit more conservative than FDR, but much more powerful than
vanilla Bonferroni correction.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`p` |  | (np.array) vector of p-values | *required*
`alpha` |  | (float) alpha level | <code>0.05</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`bonf_p` |  | (float) p-value threshold based on bonferroni     step-down procedure

###### `multi_threshold`

```python
multi_threshold(t_map, p_map, thresh)
```

Threshold test image by multiple p-values from p image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_map` |  | (BrainData) BrainData instance of statistic metric (e.g., t-statistic, beta, etc) | *required*
`p_map` |  | (BrainData) BrainData instance of p-values | *required*
`thresh` |  | (list) list of p-values to threshold stat image | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | Thresholded BrainData instance with cumulative map - Positive values indicate how many thresholds were passed for positive stats - Negative values indicate how many thresholds were passed for negative stats

<details class="note" open markdown="1">
<summary>Note</summary>

This function provides unique cumulative threshold map functionality:
- Creates a single map showing which thresholds were passed
- Different from calling threshold() multiple times (which would give separate images)
- Useful for visualizing threshold hierarchies
- nilearn.threshold_img() does not support cumulative multi-threshold maps

</details>

###### `threshold`

```python
threshold(stat, p, thr = 0.05, return_mask = False)
```

Threshold test image by p-value from p image.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stat` |  | (BrainData) BrainData instance of arbitrary statistic metric   (e.g., beta, t, etc) | *required*
`p` |  | (BrainData) BrainData instance of p-values | *required*
`thr` |  | (float) p-value threshold to apply | <code>0.05</code>
`return_mask` |  | (bool) optionally return the thresholding mask; default False | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | Thresholded BrainData instance
`mask` |  | (optional) BrainData instance of thresholding mask if return_mask=True

<details class="note" open markdown="1">
<summary>Note</summary>

This function provides unique functionality not available in nilearn:
- Thresholds stat image based on p-values from separate p-value image
- Neither nilearn.threshold_img nor BrainData.threshold() support this
- BrainData.threshold() thresholds based on stat values themselves
- nilearn.threshold_img() thresholds based on image intensity values

</details>

(algorithms-hrf)=
#### `hrf`

Hemodynamic response functions — re-exported from nilearn.

nilearn ships canonical SPM and Glover HRFs (and their derivatives) under
``nilearn.glm.first_level``. This module just re-exports them so existing
``nltools.algorithms.hrf`` imports keep working.

**Methods:**

Name | Description
---- | -----------
[`glover_dispersion_derivative`](#algorithms-glover-dispersion-derivative) | Implement the Glover dispersion derivative :term:`HRF` model.
[`glover_hrf`](#algorithms-glover-hrf) | Implement the Glover :term:`HRF` model.
[`glover_time_derivative`](#algorithms-glover-time-derivative) | Implement the Glover time derivative :term:`HRF` (dhrf) model.
[`spm_dispersion_derivative`](#algorithms-spm-dispersion-derivative) | Implement the :term:`SPM` dispersion derivative :term:`HRF` model.
[`spm_hrf`](#algorithms-spm-hrf) | Implement the :term:`SPM` :term:`HRF` model.
[`spm_time_derivative`](#algorithms-spm-time-derivative) | Implement the :term:`SPM` time derivative :term:`HRF` (dhrf) model.



##### Methods

###### `glover_dispersion_derivative`

```python
glover_dispersion_derivative(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Implement the Glover dispersion derivative :term:`HRF` model.

Parameters
----------
t_r : `float`
    :term:`Repetition time<TR>`, in seconds (sampling period).

<details class="oversampling-" open markdown="1">
<summary>`int`, default=50</summary>

Temporal oversampling factor in seconds.

</details>

<details class="time_length-" open markdown="1">
<summary>`float`, default=32.0</summary>

:term:`HRF` kernel length, in seconds.

</details>

<details class="onset-" open markdown="1">
<summary>`float`, default=0.0</summary>

Onset of the response in seconds.

</details>

Returns
-------
dhrf : array of shape (length / t_r * oversampling), dtype=float
      dhrf sampling on the oversampled time grid

Examples
--------
>>> import numpy as np
>>> from nilearn.glm.first_level import glover_dispersion_derivative
>>> ddhrf = glover_dispersion_derivative(
...     t_r=2.0, oversampling=1, time_length=20.0
... )
>>> np.round(ddhrf, 3).tolist()
[0.0, -0.0, -0.373, 0.282, 0.295, -0.04, -0.094, -0.048, -0.017, -0.005]

###### `glover_hrf`

```python
glover_hrf(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Implement the Glover :term:`HRF` model.

Parameters
----------
t_r : `float`
    :term:`Repetition time<TR>`, in seconds (sampling period).

<details class="oversampling-" open markdown="1">
<summary>`int`, default=50</summary>

Temporal oversampling factor.

</details>

<details class="time_length-" open markdown="1">
<summary>`float`, default=32.0</summary>

:term:`HRF` kernel length, in seconds.

</details>

<details class="onset-" open markdown="1">
<summary>`float`, default=0.0</summary>

Onset of the response.

</details>

Returns
-------
hrf : array of shape (length / t_r * oversampling, dtype=float)
     :term:`HRF` sampling on the oversampled time grid.

Examples
--------
>>> import numpy as np
>>> from nilearn.glm.first_level import glover_hrf
>>> hrf = glover_hrf(t_r=2.0, oversampling=1, time_length=20.0)
>>> np.round(hrf, 3).tolist()
[0.0, 0.0, 0.226, 0.741, 0.5, 0.037, -0.181, -0.176, -0.103, -0.045]

###### `glover_time_derivative`

```python
glover_time_derivative(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Implement the Glover time derivative :term:`HRF` (dhrf) model.

Parameters
----------
t_r : `float`
    :term:`Repetition time<TR>`, in seconds (sampling period).

<details class="oversampling-" open markdown="1">
<summary>`int`, default=50</summary>

Temporal oversampling factor.

</details>

<details class="time_length-" open markdown="1">
<summary>`float`, default=32.0</summary>

:term:`HRF` kernel length, in seconds.

</details>

<details class="onset-" open markdown="1">
<summary>`float`, default=0.0</summary>

Onset of the response.

</details>

Returns
-------
dhrf : array of shape (length / t_r), dtype=float
      dhrf sampling on the provided grid

Examples
--------
>>> import numpy as np
>>> from nilearn.glm.first_level import glover_time_derivative
>>> dhrf = glover_time_derivative(
...     t_r=2.0, oversampling=1, time_length=20.0
... )
>>> np.round(dhrf, 3).tolist()
[0.0, 0.0, 0.267, 0.076, -0.215, -0.168, -0.039, 0.027, 0.033, 0.019]

###### `spm_dispersion_derivative`

```python
spm_dispersion_derivative(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Implement the :term:`SPM` dispersion derivative :term:`HRF` model.

Parameters
----------
t_r : `float`
    :term:`Repetition time<TR>`, in seconds (sampling period).

<details class="oversampling-" open markdown="1">
<summary>`int`, default=50</summary>

Temporal oversampling factor in seconds.

</details>

<details class="time_length-" open markdown="1">
<summary>`float`, default=32.0</summary>

:term:`HRF` kernel length, in seconds.

</details>

<details class="onset-" open markdown="1">
<summary>`float`, default=0.0</summary>

Onset of the response in seconds.

</details>

Returns
-------
dhrf : array of shape (length / tr * oversampling), dtype=float
      dhrf sampling on the oversampled time grid

Examples
--------
>>> import numpy as np
>>> from nilearn.glm.first_level import glover_dispersion_derivative
>>> ddhrf = glover_dispersion_derivative(
...     t_r=2.0, oversampling=1, time_length=20.0
... )
>>> np.round(ddhrf, 3).tolist()
[0.0, -0.0, -0.373, 0.282, 0.295, -0.04, -0.094, -0.048, -0.017, -0.005]

###### `spm_hrf`

```python
spm_hrf(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Implement the :term:`SPM` :term:`HRF` model.

Parameters
----------
t_r : `float`
    :term:`Repetition time<TR>`, in seconds (sampling period).

<details class="oversampling-" open markdown="1">
<summary>`int`, default=50</summary>

Temporal oversampling factor.

</details>

<details class="time_length-" open markdown="1">
<summary>`float`, default=32.0</summary>

:term:`HRF` kernel length, in seconds.

</details>

<details class="onset-" open markdown="1">
<summary>`float`, default=0.0</summary>

:term:`HRF` onset time, in seconds.

</details>

Returns
-------
hrf : array of shape (length / t_r * oversampling, dtype=float)
     :term:`HRF` sampling on the oversampled time grid

Examples
--------
>>> import numpy as np
>>> from nilearn.glm.first_level import spm_hrf
>>> hrf = spm_hrf(t_r=2.0, oversampling=1, time_length=20.0)
>>> np.round(hrf, 3).tolist()
[0.0, 0.0, 0.161, 0.443, 0.335, 0.139, 0.022, -0.028, -0.04, -0.033]

###### `spm_time_derivative`

```python
spm_time_derivative(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Implement the :term:`SPM` time derivative :term:`HRF` (dhrf) model.

Parameters
----------
t_r : `float`
    :term:`Repetition time<TR>`, in seconds (sampling period).

<details class="oversampling-" open markdown="1">
<summary>`int`, default=50</summary>

Temporal oversampling factor.

</details>

<details class="time_length-" open markdown="1">
<summary>`float`, default=32.0</summary>

:term:`HRF` kernel length, in seconds.

</details>

<details class="onset-" open markdown="1">
<summary>`float`, default=0.0</summary>

Onset of the response in seconds.

</details>

Returns
-------
dhrf : array of shape (length / t_r, dtype=float)
      dhrf sampling on the provided grid

Examples
--------
>>> import numpy as np
>>> from nilearn.glm.first_level import spm_time_derivative
>>> dhrf = spm_time_derivative(t_r=2.0, oversampling=1, time_length=20.0)
>>> np.round(dhrf, 3).tolist()
[0.0, 0.0, 0.167, 0.04, -0.091, -0.072, -0.035, -0.013, -0.0, 0.005]

(algorithms-inference)=
#### `inference`

GPU-accelerated statistical inference for neuroimaging.

This module provides fast permutation testing and bootstrap resampling using
optional GPU acceleration via PyTorch. When GPU is unavailable, efficiently
uses CPU parallelization.

Inspired by BROCCOLI's GPU permutation testing (Eklund et al. 2014).

<details class="key-features" open markdown="1">
<summary>Key Features</summary>

- 10-100× speedup for permutation tests with GPU
- Efficient CPU parallelization when GPU unavailable
- Transparent CPU/GPU support via Backend abstraction
- Intersubject statistics (`isc`, `isc_group`, `isfc`, `isps`) built on the
  same permutation/bootstrap engine

</details>

**Classes:**

Name | Description
---- | -----------
[`OnlineBootstrapStats`](#algorithms-onlinebootstrapstats) | Memory-efficient online statistics aggregator for bootstrap samples.

**Methods:**

Name | Description
---- | -----------
[`circle_shift`](#algorithms-circle-shift) | Circular shift for time-series data.
[`correlation_permutation_test`](#algorithms-correlation-permutation-test) | Correlation permutation test.
[`distance_correlation`](#algorithms-distance-correlation) | Compute the distance correlation between 2 arrays to test for multivariate dependence (linear or non-linear).
[`double_center`](#algorithms-double-center) | Double center a 2d array.
[`isc_group_permutation_test`](#algorithms-isc-group-permutation-test) | Compute ISC difference between groups with permutation testing.
[`isc_permutation_test`](#algorithms-isc-permutation-test) | Compute intersubject correlation with permutation testing.
[`matrix_permutation_test`](#algorithms-matrix-permutation-test) | Matrix permutation test (Mantel test) for correlating two square matrices.
[`one_sample_permutation_test`](#algorithms-one-sample-permutation-test) | One-sample permutation test using sign-flipping.
[`phase_randomize`](#algorithms-phase-randomize) | FFT-based phase randomization for time-series data.
[`timeseries_correlation_permutation_test`](#algorithms-timeseries-correlation-permutation-test) | Time-series correlation permutation test.
[`two_sample_permutation_test`](#algorithms-two-sample-permutation-test) | Two-sample permutation test using group label shuffling.
[`u_center`](#algorithms-u-center) | U-center a 2d array. U-centering is a bias-corrected form of double-centering.



**Modules:**

Name | Description
---- | -----------
[`bootstrap`](#algorithms-bootstrap) | Bootstrap inference utilities with CPU/GPU support.
[`correlation`](#algorithms-correlation) | Correlation permutation test implementations.
[`intersubject`](#algorithms-intersubject) | Intersubject correlation, functional connectivity, and phase synchrony.
[`isc`](#algorithms-isc) | Intersubject Correlation (ISC) with GPU-Accelerated Permutation Testing.
[`matrix`](#algorithms-matrix) | Matrix permutation test implementations (Mantel test).
[`one_sample`](#algorithms-one-sample) | One-sample permutation test implementations.
[`timeseries`](#algorithms-timeseries) | Time-series permutation test implementations.
[`two_sample`](#algorithms-two-sample) | Two-sample permutation test implementations.
[`utils`](#algorithms-utils) | Utility functions for permutation testing.
[`validation`](#algorithms-validation) | Shared validation utilities for algorithms module.

**Examples:**

```pycon
>>> import numpy as np
>>> from nltools.algorithms.inference import one_sample_permutation_test
```

```pycon
>>> # Simple one-sample test
>>> data = np.random.randn(30)  # 30 subjects
>>> result = one_sample_permutation_test(data, n_permute=5000)
>>> print(f"p-value: {result['p']:.3f}")
```

```pycon
>>> # Voxel-wise test with GPU acceleration
>>> data = np.random.randn(30, 50000)  # 30 subjects, 50K voxels
>>> result = one_sample_permutation_test(data, n_permute=10000, device='gpu')
>>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")
```

<details class="performance" open markdown="1">
<summary>Performance</summary>

- CPU (NumPy): Good for small problems (< 5K permutations)
- GPU (PyTorch): Excellent for large problems (> 5K permutations)
- CPU Parallel (joblib): Efficient fallback when GPU unavailable
- Select with device='cpu' | 'gpu' | None (no 'auto' selector)

</details>

<details class="references" open markdown="1">
<summary>References</summary>

Eklund, A., Dufort, P., Villani, M., & LaConte, S. M. (2014).
BROCCOLI: Software for fast fMRI analysis on many-core CPUs and GPUs.
Frontiers in Neuroinformatics, 8, 24.

</details>

<details class="notes" open markdown="1">
<summary>Notes</summary>

This module is part of the "functional core" of nltools. For integration
with BrainData objects, see nltools.data.brain_data.

</details>

##### Classes

(algorithms-onlinebootstrapstats)=
###### `OnlineBootstrapStats`

```python
OnlineBootstrapStats(shape: tuple[int, ...], save_samples: bool = False, percentiles: tuple[float, float] = (2.5, 97.5))
```

Memory-efficient online statistics aggregator for bootstrap samples.

Uses Welford's algorithm for numerically stable online computation of
mean and variance. Optionally stores all samples for exact percentile CIs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`shape` | <code>[tuple](#tuple)[[int](#int), ...]</code> | Shape of each bootstrap sample. | *required*
`save_samples` | <code>[bool](#bool)</code> | If True, store all samples for exact percentile confidence intervals. If False, use normal approximation (much more memory efficient). Defaults to False. | <code>False</code>
`percentiles` | <code>[tuple](#tuple)[[float](#float), [float](#float)]</code> | Percentiles for confidence intervals (e.g., (2.5, 97.5) for 95% CI). Defaults to (2.5, 97.5). | <code>(2.5, 97.5)</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`M2`](#algorithms-m2) |  | 
`mean` |  | 
`n` |  | 
`percentiles` |  | 
`samples` |  | 
`save_samples` |  | 
`shape` |  | 



####### Attributes##

**Methods:**

Name | Description
---- | -----------
[`get_results`](#algorithms-get-results) | Compute final bootstrap statistics.
`update` | Update statistics with a new bootstrap sample.

**Examples:**

```pycon
>>> stats = OnlineBootstrapStats(shape=(100,), save_samples=False)
>>> for i in range(1000):
...     sample = np.random.randn(100)
...     stats.update(sample)
>>> results = stats.get_results()
>>> print(results.keys())
dict_keys(['mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper'])
```

(algorithms-m2)=
###### `M2`

```python
M2 = np.zeros(shape, dtype=(np.float64))
```

######## `mean`

```python
mean = np.zeros(shape, dtype=(np.float64))
```

######## `n`

```python
n = 0
```

######## `percentiles`

```python
percentiles = percentiles
```

######## `samples`

```python
samples = [] if save_samples else None
```

######## `save_samples`

```python
save_samples = save_samples
```

######## `shape`

```python
shape = shape
```



####### Functions##

(algorithms-get-results)=
###### `get_results`

```python
get_results(tail: int | str = 2) -> dict[str, np.ndarray]
```

Compute final bootstrap statistics.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`tail` | <code>[int](#int) \| [str](#str)</code> | 2|'two' (two-tailed, default) or 1|'one' (one-tailed: statistic > 0; negate the data for the other direction). | <code>2</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sample` | <code>[ndarray](#numpy.ndarray)</code> | New bootstrap sample with shape matching self.shape. | *required*



**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | Dictionary containing:
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'mean': Bootstrap mean
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'std': Bootstrap standard deviation
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'Z': Z-scores (mean/std)
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'p': P-values (per ``tail``)
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'ci_lower': Lower confidence bound
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'ci_upper': Upper confidence bound
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'samples': All samples (only if save_samples=True)

**Examples:**

```python
stats = OnlineBootstrapStats(shape=(100,), save_samples=False)
for _ in range(1000):
    stats.update(np.random.randn(100))
results = stats.get_results()
# results.keys() -> mean, std, Z, p, ci_lower, ci_upper
```

######## `update`

```python
update(sample: np.ndarray) -> None
```

Update statistics with a new bootstrap sample.

Uses Welford's algorithm for numerical stability.

##### Methods

###### `circle_shift`

```python
circle_shift(data: np.ndarray, shift_amount: int | np.ndarray | None = None, random_state: int | np.random.RandomState | None = None) -> np.ndarray
```

Circular shift for time-series data.

Performs a circular shift that preserves autocorrelation structure.
Useful for permutation tests on autocorrelated time series (e.g., fMRI).
For 1D data, shifts by a single amount. For 2D data, shifts each
feature (column) independently.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Time series data, shape (n_samples,) or (n_samples, n_features) | *required*
`shift_amount` | <code>[int](#int) \| [ndarray](#numpy.ndarray) \| None</code> | Shift amount(s). If None, random shift is used. For 1D: int specifying shift amount For 2D: array of length n_features with shift per feature | <code>None</code>
`random_state` | <code>[int](#int) \| [RandomState](#numpy.random.RandomState) \| None</code> | Random seed for reproducibility (if shift_amount is None) | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Circularly shifted data with same shape as input

**Examples:**

```pycon
>>> x = np.array([1, 2, 3, 4, 5])
>>> circle_shift(x, shift_amount=2)
array([4, 5, 1, 2, 3])
```

```pycon
>>> X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
>>> circle_shift(X, shift_amount=np.array([1, 2]))
array([[ 4, 30],
       [ 1, 40],
       [ 2, 10],
       [ 3, 20]])
```

###### `correlation_permutation_test`

```python
correlation_permutation_test(data1: np.ndarray, data2: np.ndarray, *, n_permute: int = 5000, metric: str = 'pearson', tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Correlation permutation test.

Tests whether the correlation between data1 and data2 is significantly
different from zero by randomly permuting data1 and computing correlations.

Assumption: Observations are independent (i.i.d.). For autocorrelated time
series, use timeseries_correlation_permutation_test with circle_shift or
phase_randomize methods instead.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | Data to permute - shape (n_samples,) for single feature - shape (n_samples, n_features) for multi-feature | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Data to correlate with - shape (n_samples,) for single feature - shape (n_samples, n_features) for multi-feature | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000) | <code>5000</code>
`metric` | <code>[str](#str)</code> | Correlation metric (default: 'pearson') - 'pearson': Pearson correlation (linear relationships) - 'spearman': Spearman rank correlation (monotonic relationships) - 'kendall': Kendall tau rank correlation (ordinal association, robust to ties) | <code>'pearson'</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type — 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction) - 2 | 'two': Two-tailed test (r != 0) - 1 | 'one': One-tailed (r > 0; negate one variable for the other   direction). The fixed direction keeps MCP correction valid. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show a progress bar over permutations (default: False) | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'correlation' (float or np.ndarray): Observed correlation(s) - 'p' (float or np.ndarray): P-value(s) - 'null_dist' (np.ndarray): Null distribution (if return_null=True) - 'device' (str): Parallelization method used

**Examples:**

```pycon
>>> # Single feature (default CPU parallelization)
>>> x = np.random.randn(100)
>>> y = x + np.random.randn(100) * 0.5  # Correlated
>>> result = correlation_permutation_test(x, y, n_permute=5000)
>>> result['correlation']
0.85
>>> result['p']
0.001
```

```pycon
>>> # Multi-feature (2D arrays)
>>> data1 = np.random.randn(100, 10)  # 100 samples, 10 features
>>> data2 = data1 + np.random.randn(100, 10) * 0.3  # Correlated
>>> result = correlation_permutation_test(data1, data2, n_permute=5000)
>>> result['correlation'].shape
(10,)
>>> result['p'].shape
(10,)
```

```pycon
>>> # GPU acceleration
>>> result = correlation_permutation_test(data1, data2, n_permute=5000, device='gpu')
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (device='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
    - Pearson: Fully vectorized across all features (5-20× speedup for multi-feature)
    - Spearman: GPU rank transform (average ties) + vectorized Pearson on ranks
    - Kendall: tie-corrected tau-b via pre-computed pairwise sign tensors;
      O(n²) memory per permutation, so batches are sized accordingly
- Single-threaded (device=None): Use for small problems or debugging
- For multi-feature data, each feature pair tested independently
- Kendall is O(n^2) complexity, slower than Pearson/Spearman for large samples

</details>

###### `distance_correlation`

```python
distance_correlation(x: np.ndarray, y: np.ndarray, bias_corrected: bool = True, ttest: bool = False) -> dict
```

Compute the distance correlation between 2 arrays to test for multivariate dependence (linear or non-linear).

Arrays must match on their first dimension. It's almost always preferable to compute the bias_corrected
version which can also optionally perform a ttest. This ttest operates on a statistic thats ~dcorr^2
and will be also returned.

Explanation:
Distance correlation involves computing the normalized covariance of two centered euclidean distance
matrices. Each distance matrix is the euclidean distance between rows (if x or y are 2d) or scalars
(if x or y are 1d). Each matrix is centered prior to computing the covariance either using double-centering
or u-centering, which corrects for bias as the number of dimensions increases. U-centering is almost always
preferred in all cases. It also permits inference of the normalized covariance between each distance matrix
using a one-tailed directional t-test. (Szekely & Rizzo, 2013). While distance correlation is normally
bounded between 0 and 1, u-centering can produce negative estimates, which are never significant.

Validated against the dcor and dcor.ttest functions in the 'energy' R package and the
dcor.distance_correlation, dcor.udistance_correlation_sqr, and dcor.independence.distance_correlation_t_test
functions in the dcor Python package.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` | <code>[ndarray](#ndarray)</code> | 1d or 2d numpy array of observations by features | *required*
`y` | <code>[ndarray](#ndarray)</code> | 1d or 2d numpy array of observations by features | *required*
`bias_corrected` | <code>[bool](#bool)</code> | if false use double-centering which produces a biased-estimate that converges to 1 as the number of dimensions increase. Otherwise used u-centering to correct this bias. **Note** this must be True if ttest=True; default True | <code>True</code>
`ttest` | <code>[bool](#bool)</code> | perform a ttest using the bias_corrected distance correlation; default False | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`results` | <code>[dict](#dict)</code> | dictionary of results (correlation, t, p, and df.) Optionally, covariance, x variance, and y variance

**Examples:**

```pycon
>>> import numpy as np
>>> x = np.random.randn(20, 3)
>>> y = x + np.random.randn(20, 3) * 0.1  # Strongly correlated
>>> result = distance_correlation(x, y, bias_corrected=True)
>>> 'dcorr' in result
True
>>> 0 <= result['dcorr'] <= 1
True
```

###### `double_center`

```python
double_center(mat: np.ndarray) -> np.ndarray
```

Double center a 2d array.

Double-centering subtracts row means, column means, and adds the grand mean.
This centers both rows and columns around zero.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mat` | <code>[ndarray](#ndarray)</code> | 2d numpy array | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`mat` | <code>[ndarray](#ndarray)</code> | double-centered version of input

**Examples:**

```pycon
>>> mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
>>> result = double_center(mat)
>>> np.allclose(result.mean(axis=0), 0)
True
>>> np.allclose(result.mean(axis=1), 0)
True
```

###### `isc_group_permutation_test`

```python
isc_group_permutation_test(group1: np.ndarray, group2: np.ndarray, *, n_permute: int = 5000, summary: Literal['median', 'mean'] = 'median', method: Literal['permute', 'bootstrap'] = 'permute', summary_statistic: Literal['leave-one-out', 'pairwise'] = 'pairwise', ci_percentile: float = 95, tail: int | str = 2, device: Literal['cpu', 'gpu'] | None = 'cpu', n_jobs: int = -1, random_state: int | None = None, return_null: bool = False, progress_bar: bool = False, exclude_self_corr: bool = True, metric: str = 'correlation') -> dict[str, Any]
```

Compute ISC difference between groups with permutation testing.

Supports both subject-wise permutation and bootstrap methods with efficient
CPU-parallel and optional GPU acceleration. Follows the statistical methods
from Chen et al. (2016) for correct group comparison inference.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`group1` | <code>[ndarray](#numpy.ndarray)</code> | First group data with one of the following shapes: - (n_observations, n_subjects1): Single feature - (n_observations, n_subjects1, n_voxels): Voxel-wise | *required*
`group2` | <code>[ndarray](#numpy.ndarray)</code> | Second group data with one of the following shapes: - (n_observations, n_subjects2): Single feature - (n_observations, n_subjects2, n_voxels): Voxel-wise | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations/bootstrap iterations. Defaults to 5000. | <code>5000</code>
`summary` | <code>[Literal](#typing.Literal)['median', 'mean']</code> | Summary statistic for aggregating ISC values: - 'median': Direct median (robust to outliers) - 'mean': Fisher z-transformed mean (unbiased averaging) Defaults to 'median'. | <code>'median'</code>
`method` | <code>[Literal](#typing.Literal)['permute', 'bootstrap']</code> | Resampling method for p-value computation: - 'permute': Subject-wise permutation (combines groups, permutes labels) - 'bootstrap': Subject-wise bootstrap (resamples within each group) Defaults to 'permute'. | <code>'permute'</code>
`summary_statistic` | <code>[Literal](#typing.Literal)['leave-one-out', 'pairwise']</code> | ISC computation method: - 'pairwise': Average all pairwise correlations - 'leave-one-out': Correlate each subject with mean of others Defaults to 'pairwise'. | <code>'pairwise'</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95 for 95% CI). Defaults to 95. | <code>95</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Two-tailed (2 or 'two', default) or one-tailed (1 or 'one', positive direction) p-value. | <code>2</code>
`device` | <code>[Literal](#typing.Literal)['cpu', 'gpu'] \| None</code> | Parallelization method: - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (10-30× speedup for voxel-wise LOO) - None: Single-threaded NumPy (for debugging/small problems) Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when device='cpu'. Defaults to -1. | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, return null distribution in result dict. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during bootstrap/permutation. Defaults to False. | <code>False</code>
`exclude_self_corr` | <code>[bool](#bool)</code> | Mask self-correlations in bootstrap (pairwise only). Defaults to True. | <code>True</code>
`metric` | <code>[str](#str)</code> | Similarity metric for pairwise ISC computation. See sklearn.metrics.pairwise_distances for valid options. Only applies when summary_statistic='pairwise'. Defaults to 'correlation'. | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with the following keys:
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'isc_group_difference': Observed ISC difference (float or array per voxel)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'p': P-value (Phipson-Smyth corrected)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'ci': Confidence interval tuple (lower, upper)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'device': Parallelization method used
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'null_dist': (optional) Bootstrap/permutation distribution

**Examples:**

```pycon
>>> # Single-feature ISC group comparison
>>> group1 = np.random.randn(100, 10)  # 10 subjects
>>> group2 = np.random.randn(100, 10)
>>> result = isc_group_permutation_test(group1, group2, n_permute=1000)
>>> print(f"ISC difference: {result['isc_group_difference']:.3f}, p: {result['p']:.3f}")
```

```pycon
>>> # Voxel-wise ISC group comparison with GPU acceleration
>>> group1_voxels = np.random.randn(100, 10, 5000)  # 5K voxels
>>> group2_voxels = np.random.randn(100, 10, 5000)
>>> result = isc_group_permutation_test(
...     group1_voxels,
...     group2_voxels,
...     summary_statistic='leave-one-out',
...     device='gpu',  # GPU for LOO computation
...     n_permute=5000
... )
>>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")
```

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Permutation method combines groups and permutes labels (Chen et al. 2016)
- Bootstrap method resamples subjects within each group independently
- Bootstrap distribution is centered by subtracting observed difference
- GPU acceleration available for voxel-wise LOO computation

</details>

###### `isc_permutation_test`

```python
isc_permutation_test(data: np.ndarray, *, n_permute: int = 5000, summary: Literal['median', 'mean'] = 'median', summary_statistic: Literal['leave-one-out', 'pairwise'] = 'pairwise', method: Literal['bootstrap', 'circle_shift', 'phase_randomize'] = 'bootstrap', ci_percentile: float = 95, tail: int | str = 2, return_null: bool = False, progress_bar: bool = False, exclude_self_corr: bool = True, metric: str = 'correlation', device: Literal['cpu', 'gpu'] | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> dict[str, Any]
```

Compute intersubject correlation with permutation testing.

Supports both leave-one-out and pairwise ISC computation modes with
GPU acceleration for large voxel-wise problems and CPU-parallel
bootstrap resampling.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Data array with one of the following shapes: - (n_observations, n_subjects): Single feature ISC - (n_observations, n_subjects, n_voxels): Voxel-wise ISC | *required*
`n_permute` | <code>[int](#int)</code> | Number of bootstrap iterations or permutations. Defaults to 5000. | <code>5000</code>
`summary` | <code>[Literal](#typing.Literal)['median', 'mean']</code> | Summary statistic to aggregate ISC values. - 'median': Direct median (robust to outliers) - 'mean': Fisher z-transformed mean (unbiased averaging) Defaults to 'median'. | <code>'median'</code>
`summary_statistic` | <code>[Literal](#typing.Literal)['leave-one-out', 'pairwise']</code> | ISC computation method. Options: - 'leave-one-out': Correlate each subject with mean of others. O(n_subjects), unbiased, recommended by Chen et al. 2016. - 'pairwise': Average all pairwise correlations. O(n_subjects²), captures full correlation structure. Note: These methods are statistically different and monotonically but non-linearly related (see Chen et al. 2016, Figure 3). Defaults to 'pairwise'. | <code>'pairwise'</code>
`method` | <code>[Literal](#typing.Literal)['bootstrap', 'circle_shift', 'phase_randomize']</code> | Resampling method for p-value computation: - 'bootstrap': Subject-wise bootstrap (default, Chen et al. 2016) - 'circle_shift': Circular time-series shift (preserves autocorrelation) - 'phase_randomize': FFT phase randomization (preserves power spectrum) Defaults to 'bootstrap'. | <code>'bootstrap'</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95 for 95% CI). Defaults to 95. | <code>95</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Two-tailed (2 or 'two', default) or one-tailed (1 or 'one', positive direction) p-value. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return bootstrap/permutation distribution in result dict. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during bootstrap/permutation. Defaults to False. | <code>False</code>
`exclude_self_corr` | <code>[bool](#bool)</code> | If True, mask self-correlations (perfect correlations from duplicate subjects in bootstrap samples) as NaN. If False, include them in the summary statistic. Only applies when method='bootstrap' and summary_statistic='pairwise'. Defaults to True. | <code>True</code>
`metric` | <code>[str](#str)</code> | Similarity metric for pairwise ISC computation. See sklearn.metrics.pairwise_distances for valid options. Only applies when summary_statistic='pairwise'. For 'correlation', uses optimized np.corrcoef. Other metrics use pairwise_distances. Defaults to 'correlation'. | <code>'correlation'</code>
`device` | <code>[Literal](#typing.Literal)['cpu', 'gpu'] \| None</code> | Parallelization method: - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (10-30× speedup for voxel-wise LOO) - None: Single-threaded NumPy (for debugging/small problems) Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when device='cpu'. Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float) \| None</code> | GPU working-set budget in GB. For the pairwise GPU bootstrap (``device='gpu'``, ``summary_statistic='pairwise'``, ``method='bootstrap'``) this bounds the ``(perm_batch, voxel_chunk, n_subjects, n_subjects)`` resample tensor, chunking voxels and permutations to fit — so whole-brain runs stay within budget. Not used by the LOO or surrogate (circle_shift/phase_randomize) paths. Defaults to 4. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with the following keys:
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'isc': Observed ISC value (float or array per voxel)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'p': P-value (Phipson-Smyth corrected)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'ci': Confidence interval tuple (lower, upper)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'device': Parallelization method used
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'null_dist': (optional) Bootstrap/permutation distribution

**Examples:**

```pycon
>>> # Single-feature ISC
>>> data = np.random.randn(100, 10)  # 100 timepoints, 10 subjects
>>> result = isc_permutation_test(data, n_permute=1000)
>>> print(f"ISC: {result['isc']:.3f}, p: {result['p']:.3f}")
```

```pycon
>>> # Voxel-wise ISC with GPU acceleration
>>> data_voxels = np.random.randn(100, 50, 5000)  # 5K voxels
>>> result = isc_permutation_test(
...     data_voxels,
...     summary_statistic='leave-one-out',
...     device='gpu',  # GPU for LOO computation
...     n_permute=5000
... )
>>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")
```

```pycon
>>> # Compare LOO vs pairwise
>>> result_loo = isc_permutation_test(data, summary_statistic='leave-one-out')
>>> result_pair = isc_permutation_test(data, summary_statistic='pairwise')
>>> print(f"LOO: {result_loo['isc']:.3f}, Pairwise: {result_pair['isc']:.3f}")
```

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Leave-one-out is 20-30× faster than pairwise for large n_subjects
- GPU acceleration helps most for voxel-wise LOO (10-30× speedup)
- Pairwise bootstrap uses correct subject-wise resampling (Chen 2016)
- Bootstrap distribution is centered by subtracting observed ISC

</details>

###### `matrix_permutation_test`

```python
matrix_permutation_test(data1: np.ndarray, data2: np.ndarray, *, n_permute: int = 5000, metric: str = 'pearson', how: str = 'upper', include_diag: bool = False, tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Matrix permutation test (Mantel test) for correlating two square matrices.

Tests whether the correlation between elements of two matrices is significant
by permuting rows and columns of one matrix symmetrically while keeping the
other fixed.

**Statistical Method**:
For each permutation, create random permutation `perm`, then apply:
`matrix1[perm][:, perm]`. This preserves matrix structure while destroying
correlation. Count how often permuted correlation is as extreme as observed.

**Assumptions**:
- Matrices are square and same size
- Under H₀, row/column ordering is exchangeable
- Symmetric permutation preserves matrix properties (e.g., symmetry)

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | First square matrix (n×n) | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Second square matrix (n×n) | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000) | <code>5000</code>
`metric` | <code>[str](#str)</code> | Correlation metric ['pearson'|'spearman'|'kendall'] (default: 'pearson') | <code>'pearson'</code>
`how` | <code>[str](#str)</code> | Which elements to compare ['upper'|'lower'|'full'] (default: 'upper') - 'upper': Upper triangle only (assumes symmetric matrices) - 'lower': Lower triangle only - 'full': All elements (see include_diag) | <code>'upper'</code>
`include_diag` | <code>[bool](#bool)</code> | Include diagonal elements (only applies if how='full') (default: False) | <code>False</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type — 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction) - 2 | 'two': Two-tailed test (r != 0) - 1 | 'one': One-tailed (r > 0; negate the data for the other direction) | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | Return null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel workers, -1 = all cores (default: -1) Only used when device='cpu' | <code>-1</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show a progress bar over permutations (default: False) | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'correlation' (float): Observed correlation coefficient - 'p' (float): P-value using Phipson-Smyth correction - 'device' (str): Parallelization method used ('cpu' or None) - 'null_dist' (np.ndarray): Null distribution (if return_null=True)

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G. et al. (2016). Untangling the relatedness among correlations,
part I: nonparametric approaches to inter-subject correlation analysis
at the group level. NeuroImage, 142, 248-259.

Mantel, N. (1967). The detection of disease clustering and a generalized
regression approach. Cancer Research, 27(2), 209-220.

</details>

**Examples:**

```pycon
>>> import numpy as np
>>> from nltools.algorithms.inference import matrix_permutation_test
>>>
>>> # Create two correlated similarity matrices
>>> np.random.seed(42)
>>> n = 50
>>> true_pattern = np.random.randn(n)
>>> data1 = np.corrcoef(true_pattern + np.random.randn(n) * 0.1)
>>> data2 = np.corrcoef(true_pattern + np.random.randn(n) * 0.1)
>>>
>>> # Test if matrices are correlated
>>> result = matrix_permutation_test(data1, data2, n_permute=1000)
>>> print(f"Correlation: {result['correlation']:.3f}, p = {result['p']:.4f}")
```

###### `one_sample_permutation_test`

```python
one_sample_permutation_test(data: np.ndarray, *, n_permute: int = 5000, tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None, progress_bar: bool = False) -> dict
```

One-sample permutation test using sign-flipping.

Tests whether the mean of data is significantly different from zero
by randomly flipping the sign of each observation. This is the
permutation test equivalent of a one-sample t-test.

Assumption: Symmetric error distribution around zero. For highly skewed
distributions, consider alternative methods (e.g., bootstrap resampling).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Data to test - shape (n_samples,) for single feature - shape (n_samples, n_features) for multi-feature (voxel-wise) | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000) | <code>5000</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type — 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction) - 2 | 'two': Two-tailed test (mean != 0) - 1 | 'one': One-tailed (mean > 0; negate the data for the other   direction). The fixed direction keeps MCP correction valid. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Whether to display a progress bar (default: False) | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'mean' (float or np.ndarray): Observed mean(s) - 'p' (float or np.ndarray): P-value(s) - 'null_dist' (np.ndarray): Null distribution (if return_null=True) - 'device' (str): Parallelization method used

**Examples:**

```pycon
>>> # Single feature (default CPU parallelization)
>>> data = np.random.randn(30)
>>> result = one_sample_permutation_test(data, n_permute=5000)
>>> result['p']
0.23
```

```pycon
>>> # Voxel-wise test with GPU
>>> data = np.random.randn(30, 10000)  # 30 subjects, 10K voxels
>>> result = one_sample_permutation_test(data, n_permute=5000, device='gpu')
>>> result['mean'].shape
(10000,)
>>> result['p'].shape
(10000,)
```

```pycon
>>> # Single-threaded (for debugging)
>>> result = one_sample_permutation_test(data, n_permute=5000, device=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (device='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
- Single-threaded (device=None): Use for small problems or debugging
- For voxel-wise tests, each voxel tested independently
- Progress bars show completion for both CPU parallel and GPU batched modes

</details>

###### `phase_randomize`

```python
phase_randomize(data: np.ndarray, *, device: str | None = 'cpu', random_state: int | np.random.RandomState | None = None) -> np.ndarray
```

FFT-based phase randomization for time-series data.

Preserves the power spectrum (autocorrelation) but destroys nonlinear
temporal structure by randomizing Fourier phases. Used to test whether
data was generated by a linear Gaussian process or contains nonlinear
dynamics.

<details class="algorithm" open markdown="1">
<summary>Algorithm</summary>

1. Compute FFT of input signal
2. Generate random phases [0, 2π] for positive frequencies
3. Apply phase shifts to positive frequencies: multiply by exp(i*φ)
4. Apply conjugate phase shifts to negative frequencies (for real output)
5. Compute inverse FFT to get phase-randomized signal

</details>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Time series data, shape (n_samples,) or (n_samples, n_features) | *required*
`device` | <code>[str](#str) \| None</code> | Compute device. - 'cpu' / None: NumPy FFT (default, float64 precision) - 'gpu': PyTorch FFT on CUDA/MPS (float32 precision, 5-20× faster for large data) - 'auto': use a GPU if present, else CPU | <code>'cpu'</code>
`random_state` | <code>[int](#int) \| [RandomState](#numpy.random.RandomState) \| None</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Phase-randomized data with same shape as input

<details class="notes" open markdown="1">
<summary>Notes</summary>

- **CRITICAL**: Preserves power spectrum exactly (within numerical precision)
- Precision: the CPU path uses float64, the GPU path float32
- Conjugate symmetry is maintained for real-valued output

</details>

**Examples:**

```pycon
>>> x = np.sin(np.linspace(0, 10*np.pi, 100))  # Sine wave
>>> x_rand = phase_randomize(x, random_state=42)
>>> # Power spectrum preserved:
>>> np.allclose(np.abs(np.fft.rfft(x))**2, np.abs(np.fft.rfft(x_rand))**2)
True
```

```pycon
>>> # GPU acceleration for large datasets:
>>> x_large = np.random.randn(10000)
>>> x_rand_gpu = phase_randomize(x_large, device='gpu', random_state=42)
```

###### `timeseries_correlation_permutation_test`

```python
timeseries_correlation_permutation_test(data1: np.ndarray, data2: np.ndarray, *, method: Literal['circle_shift', 'phase_randomize'] = 'circle_shift', n_permute: int = 5000, metric: Literal['pearson', 'spearman', 'kendall'] = 'pearson', tail: int | str = 2, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, return_null: bool = False, random_state: int | np.random.RandomState | None = None, progress_bar: bool = False) -> dict
```

Time-series correlation permutation test.

Unlike standard permutation tests that shuffle data independently,
this test uses time-series-aware permutation methods that preserve
temporal structure (circle_shift) or power spectrum (phase_randomize).

Use this test when data contains temporal autocorrelation. Standard
permutation tests inflate Type I error for autocorrelated data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | First time series, shape (n_samples,) or (n_samples, 1) | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Second time series, shape (n_samples,) or (n_samples, 1) | *required*
`method` | <code>[Literal](#typing.Literal)['circle_shift', 'phase_randomize']</code> | Permutation method: - 'circle_shift': Circular shift (preserves autocorrelation) - 'phase_randomize': FFT-based (preserves power spectrum) | <code>'circle_shift'</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations | <code>5000</code>
`metric` | <code>[Literal](#typing.Literal)['pearson', 'spearman', 'kendall']</code> | Correlation type ('pearson', 'spearman', 'kendall') | <code>'pearson'</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type (default: 2) - 2 or 'two': Two-tailed test (default) - 1 or 'one': One-tailed test in the test's positive direction   (to test the negative direction, negate the data / swap groups) | <code>2</code>
`device` | <code>[str](#str) \| None</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs (-1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float) \| None</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | Whether to return null distribution | <code>False</code>
`random_state` | <code>[int](#int) \| [RandomState](#numpy.random.RandomState) \| None</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show a progress bar over permutations (default: False) | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with keys: - 'correlation': Observed correlation coefficient - 'p': P-value - 'null_dist': (if return_null=True) Null distribution - 'device': Parallelization method used

**Examples:**

```pycon
>>> x = np.sin(np.linspace(0, 10*np.pi, 100))
>>> y = np.cos(np.linspace(0, 10*np.pi, 100))
>>> result = timeseries_correlation_permutation_test(
...     x, y, method='circle_shift', n_permute=1000, random_state=42
... )
>>> result['correlation']  # Strong negative correlation
-0.999...
>>> result['p'] < 0.05  # Significant
True
```

```pycon
>>> # GPU acceleration
>>> result = timeseries_correlation_permutation_test(
...     x, y, method='phase_randomize', device='gpu', n_permute=5000
... )
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (device='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): 5-20× faster for large problems (n_samples > 1000)
- Single-threaded (device=None): Use for small problems or debugging
- For independent data, use regular correlation_permutation_test
- circle_shift is faster and suitable for most fMRI time series
- phase_randomize preserves power spectrum exactly (tests nonlinearity)
- Only data1 is randomized; data2 remains fixed to test correlation
- phase_randomize benefits most from GPU (FFT acceleration)

</details>

###### `two_sample_permutation_test`

```python
two_sample_permutation_test(data1: np.ndarray, data2: np.ndarray, *, n_permute: int = 5000, tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Two-sample permutation test using group label shuffling.

Tests whether two independent groups have different means by randomly
permuting group labels. This is the permutation test equivalent of an
independent samples t-test.

Assumption: Exchangeability under the null hypothesis (group assignments
are arbitrary). Valid for independent samples from similar distributions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | Group 1 data - shape (n_samples1,) for single feature - shape (n_samples1, n_features) for multi-feature (voxel-wise) | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Group 2 data - shape (n_samples2,) for single feature - shape (n_samples2, n_features) for multi-feature (voxel-wise) | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000) | <code>5000</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type — 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction) - 2 | 'two': Two-tailed test (mean1 != mean2) - 1 | 'one': One-tailed (mean1 > mean2; swap the groups for the   other direction). The fixed direction keeps MCP correction valid. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'mean_diff' (float or np.ndarray): Observed mean difference (data1 - data2) - 'p' (float or np.ndarray): P-value(s) - 'null_dist' (np.ndarray): Null distribution (if return_null=True) - 'device' (str): Parallelization method used

**Examples:**

```pycon
>>> # Single feature (default CPU parallelization)
>>> data1 = np.random.randn(20)  # Group 1: 20 subjects
>>> data2 = np.random.randn(25)  # Group 2: 25 subjects
>>> result = two_sample_permutation_test(data1, data2, n_permute=5000)
>>> result['p']
0.45
```

```pycon
>>> # Voxel-wise test with GPU
>>> data1 = np.random.randn(20, 10000)  # 20 subjects, 10K voxels
>>> data2 = np.random.randn(25, 10000)  # 25 subjects, 10K voxels
>>> result = two_sample_permutation_test(data1, data2, n_permute=5000, device='gpu')
>>> result['mean_diff'].shape
(10000,)
>>> result['p'].shape
(10000,)
```

```pycon
>>> # Single-threaded (for debugging)
>>> result = two_sample_permutation_test(data1, data2, n_permute=5000, device=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (device='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
- Single-threaded (device=None): Use for small problems or debugging
- For voxel-wise tests, each voxel tested independently
- Group sizes can be unequal

</details>

###### `u_center`

```python
u_center(mat: np.ndarray) -> np.ndarray
```

U-center a 2d array. U-centering is a bias-corrected form of double-centering.

U-centering corrects for bias that occurs with double-centering as the number
of dimensions increases. The diagonal is explicitly set to zero.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mat` | <code>[ndarray](#ndarray)</code> | 2d numpy array | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`mat` | <code>[ndarray](#ndarray)</code> | u-centered version of input

**Examples:**

```pycon
>>> mat = np.random.randn(5, 5)
>>> result = u_center(mat)
>>> np.allclose(np.diag(result), 0)
True
```



##### Modules

(algorithms-bootstrap)=
###### `bootstrap`

Bootstrap inference utilities with CPU/GPU support.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`FITTED_METHODS`](#algorithms-fitted-methods) |  | 
`SIMPLE_METHODS` |  | 



####### Attributes##

**Classes:**

Name | Description
---- | -----------
[`OnlineBootstrapStats`](#algorithms-onlinebootstrapstats) | Memory-efficient online statistics aggregator for bootstrap samples.

(algorithms-fitted-methods)=
###### `FITTED_METHODS`

```python
FITTED_METHODS = ['weights', 'predict']
```

######## `SIMPLE_METHODS`

```python
SIMPLE_METHODS = ['mean', 'median', 'std', 'sum', 'min', 'max']
```



####### Classes##

###### `OnlineBootstrapStats`

```python
OnlineBootstrapStats(shape: tuple[int, ...], save_samples: bool = False, percentiles: tuple[float, float] = (2.5, 97.5))
```

Memory-efficient online statistics aggregator for bootstrap samples.

Uses Welford's algorithm for numerically stable online computation of
mean and variance. Optionally stores all samples for exact percentile CIs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`shape` | <code>[tuple](#tuple)[[int](#int), ...]</code> | Shape of each bootstrap sample. | *required*
`save_samples` | <code>[bool](#bool)</code> | If True, store all samples for exact percentile confidence intervals. If False, use normal approximation (much more memory efficient). Defaults to False. | <code>False</code>
`percentiles` | <code>[tuple](#tuple)[[float](#float), [float](#float)]</code> | Percentiles for confidence intervals (e.g., (2.5, 97.5) for 95% CI). Defaults to (2.5, 97.5). | <code>(2.5, 97.5)</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`M2`](#algorithms-m2) |  | 
`mean` |  | 
`n` |  | 
`percentiles` |  | 
`samples` |  | 
`save_samples` |  | 
`shape` |  | 



######### Attributes####

**Methods:**

Name | Description
---- | -----------
[`get_results`](#algorithms-get-results) | Compute final bootstrap statistics.
`update` | Update statistics with a new bootstrap sample.

**Examples:**

```pycon
>>> stats = OnlineBootstrapStats(shape=(100,), save_samples=False)
>>> for i in range(1000):
...     sample = np.random.randn(100)
...     stats.update(sample)
>>> results = stats.get_results()
>>> print(results.keys())
dict_keys(['mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper'])
```

###### `M2`

```python
M2 = np.zeros(shape, dtype=(np.float64))
```

########## `mean`

```python
mean = np.zeros(shape, dtype=(np.float64))
```

########## `n`

```python
n = 0
```

########## `percentiles`

```python
percentiles = percentiles
```

########## `samples`

```python
samples = [] if save_samples else None
```

########## `save_samples`

```python
save_samples = save_samples
```

########## `shape`

```python
shape = shape
```



######### Functions####

###### `get_results`

```python
get_results(tail: int | str = 2) -> dict[str, np.ndarray]
```

Compute final bootstrap statistics.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`tail` | <code>[int](#int) \| [str](#str)</code> | 2|'two' (two-tailed, default) or 1|'one' (one-tailed: statistic > 0; negate the data for the other direction). | <code>2</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sample` | <code>[ndarray](#numpy.ndarray)</code> | New bootstrap sample with shape matching self.shape. | *required*



####### Functions

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | Dictionary containing:
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'mean': Bootstrap mean
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'std': Bootstrap standard deviation
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'Z': Z-scores (mean/std)
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'p': P-values (per ``tail``)
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'ci_lower': Lower confidence bound
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'ci_upper': Upper confidence bound
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'samples': All samples (only if save_samples=True)

**Examples:**

```python
stats = OnlineBootstrapStats(shape=(100,), save_samples=False)
for _ in range(1000):
    stats.update(np.random.randn(100))
results = stats.get_results()
# results.keys() -> mean, std, Z, p, ci_lower, ci_upper
```

########## `update`

```python
update(sample: np.ndarray) -> None
```

Update statistics with a new bootstrap sample.

Uses Welford's algorithm for numerical stability.

(algorithms-correlation)=
###### `correlation`

Correlation permutation test implementations.

This module provides CPU-parallel and GPU-batched implementations
of correlation permutation tests for assessing statistical significance
of correlations.

**Methods:**

Name | Description
---- | -----------
[`correlation_permutation_test`](#algorithms-correlation-permutation-test) | Correlation permutation test.



####### Attributes

####### Classes

####### Functions##

###### `correlation_permutation_test`

```python
correlation_permutation_test(data1: np.ndarray, data2: np.ndarray, *, n_permute: int = 5000, metric: str = 'pearson', tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Correlation permutation test.

Tests whether the correlation between data1 and data2 is significantly
different from zero by randomly permuting data1 and computing correlations.

Assumption: Observations are independent (i.i.d.). For autocorrelated time
series, use timeseries_correlation_permutation_test with circle_shift or
phase_randomize methods instead.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | Data to permute - shape (n_samples,) for single feature - shape (n_samples, n_features) for multi-feature | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Data to correlate with - shape (n_samples,) for single feature - shape (n_samples, n_features) for multi-feature | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000) | <code>5000</code>
`metric` | <code>[str](#str)</code> | Correlation metric (default: 'pearson') - 'pearson': Pearson correlation (linear relationships) - 'spearman': Spearman rank correlation (monotonic relationships) - 'kendall': Kendall tau rank correlation (ordinal association, robust to ties) | <code>'pearson'</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type — 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction) - 2 | 'two': Two-tailed test (r != 0) - 1 | 'one': One-tailed (r > 0; negate one variable for the other   direction). The fixed direction keeps MCP correction valid. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show a progress bar over permutations (default: False) | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'correlation' (float or np.ndarray): Observed correlation(s) - 'p' (float or np.ndarray): P-value(s) - 'null_dist' (np.ndarray): Null distribution (if return_null=True) - 'device' (str): Parallelization method used

**Examples:**

```pycon
>>> # Single feature (default CPU parallelization)
>>> x = np.random.randn(100)
>>> y = x + np.random.randn(100) * 0.5  # Correlated
>>> result = correlation_permutation_test(x, y, n_permute=5000)
>>> result['correlation']
0.85
>>> result['p']
0.001
```

```pycon
>>> # Multi-feature (2D arrays)
>>> data1 = np.random.randn(100, 10)  # 100 samples, 10 features
>>> data2 = data1 + np.random.randn(100, 10) * 0.3  # Correlated
>>> result = correlation_permutation_test(data1, data2, n_permute=5000)
>>> result['correlation'].shape
(10,)
>>> result['p'].shape
(10,)
```

```pycon
>>> # GPU acceleration
>>> result = correlation_permutation_test(data1, data2, n_permute=5000, device='gpu')
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (device='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
    - Pearson: Fully vectorized across all features (5-20× speedup for multi-feature)
    - Spearman: GPU rank transform (average ties) + vectorized Pearson on ranks
    - Kendall: tie-corrected tau-b via pre-computed pairwise sign tensors;
      O(n²) memory per permutation, so batches are sized accordingly
- Single-threaded (device=None): Use for small problems or debugging
- For multi-feature data, each feature pair tested independently
- Kendall is O(n^2) complexity, slower than Pearson/Spearman for large samples

</details>

(algorithms-intersubject)=
###### `intersubject`

Intersubject correlation, functional connectivity, and phase synchrony.

**Methods:**

Name | Description
---- | -----------
[`isc`](#algorithms-isc) | Compute pairwise intersubject correlation from observations by subjects array.
[`isc_group`](#algorithms-isc-group) | Compute difference in intersubject correlation between groups.
[`isfc`](#algorithms-isfc) | Compute intersubject functional connectivity (ISFC) from a list of observation x feature matrices.
[`isps`](#algorithms-isps) | Compute dynamic intersubject phase synchrony (ISPS) from an observations-by-subjects array.



####### Functions##

###### `isc`

```python
isc(data, *, n_samples = 5000, summary = 'median', method = 'bootstrap', ci_percentile = 95, exclude_self_corr = True, tail = 2, metric = 'correlation', return_null = False, n_jobs = -1, random_state = None, progress_bar = False)
```

Compute pairwise intersubject correlation from observations by subjects array.

This function computes pairwise intersubject correlations (ISC) using the median as recommended by Chen
et al., 2016). However, if the mean is preferred, we compute the mean correlation after performing
the fisher r-to-z transformation and then convert back to correlations to minimize artificially
inflating the correlation values.

There are currently three different methods to compute p-values. These include the classic methods for
computing permuted time-series by either circle-shifting the data or phase-randomizing the data
(see Lancaster et al., 2018). These methods create random surrogate data while preserving the temporal
autocorrelation inherent to the signal. By default, we use the subject-wise bootstrap method from
Chen et al., 2016. Instead of recomputing the pairwise ISC using circle_shift or phase_randomization methods,
this approach uses the computationally more efficient method of bootstrapping the subjects
and computing a new pairwise similarity matrix with randomly selected subjects with replacement.
If the same subject is selected multiple times, we set the perfect correlation to a nan with
(exclude_self_corr=True). We compute the p-values using the percentile method using the same
method in Brainiak.

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C., Israel, R. B.,
& Cox, R. W. (2016). Untangling the relatedness among correlations, part I:
nonparametric approaches to inter-subject correlation analysis at the group level.
NeuroImage, 142, 248-259.

Hall, P., & Wilson, S. R. (1991). Two guidelines for bootstrap hypothesis testing.
Biometrics, 757-762.

Lancaster, G., Iatsenko, D., Pidde, A., Ticcinelli, V., & Stefanovska, A. (2018).
Surrogate data for hypothesis testing of physical systems. Physics Reports, 748, 1-60.

This function is a wrapper around `isc_permutation_test` from the inference module,
which provides optimized implementations with CPU-parallel and GPU acceleration support.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pd.DataFrame, np.array) observations by subjects where isc is computed across subjects | *required*
`n_samples` |  | (int) number of random samples/bootstraps | <code>5000</code>
`summary` |  | (str) type of isc summary statistic ['mean','median'] (default: median) | <code>'median'</code>
`method` |  | (str) method to compute p-values ['bootstrap', 'circle_shift','phase_randomize'] (default: bootstrap) | <code>'bootstrap'</code>
`ci_percentile` |  | (int) confidence-interval width in percent for the bootstrap CI (default: 95) | <code>95</code>
`exclude_self_corr` |  | (bool) set self-correlations (same subject bootstrapped twice) to nan (default: True) | <code>True</code>
`tail` |  | (int | str) 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction) | <code>2</code>
`metric` |  | (str) pairwise distance metric. See sklearn's pairwise_distances for valid inputs (default: correlation) | <code>'correlation'</code>
`return_null` |  | (bool) Return the permutation distribution along with the p-value; default False | <code>False</code>
`n_jobs` |  | (int) The number of CPUs to use to do the computation. -1 means all CPUs. | <code>-1</code>
`random_state` |  | (int, np.random.RandomState, or None) seed or generator for the resampling; default None | <code>None</code>
`progress_bar` |  | (bool) If True, display a progress bar. Default False. | <code>False</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`group1` |  | (pd.DataFrame, np.array) observations by subjects where isc is computed across subjects | *required*
`group2` |  | (pd.DataFrame, np.array) observations by subjects where isc is computed across subjects | *required*
`n_samples` |  | (int) number of samples for permutation or bootstrapping | <code>5000</code>
`summary` |  | (str) type of isc summary statistic ['mean','median'] (default: median) | <code>'median'</code>
`method` |  | (str) method to compute p-values ['permute', 'bootstrap'] (default: permute) | <code>'permute'</code>
`ci_percentile` |  | (float) confidence interval percentile (default: 95) | <code>95</code>
`exclude_self_corr` |  | (bool) exclude self-correlations in bootstrap (default: True) | <code>True</code>
`return_null` |  | (bool) Return the permutation distribution along with the p-value; default False | <code>False</code>
`tail` |  | (int | str) 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction) | <code>2</code>
`metric` |  | (str) pairwise distance metric. See sklearn's pairwise_distances for valid inputs (default: correlation) | <code>'correlation'</code>
`n_jobs` |  | (int) The number of CPUs to use to do the computation. -1 means all CPUs. | <code>-1</code>
`random_state` |  | (int or RandomState) Random seed for reproducibility | <code>None</code>
`progress_bar` |  | (bool) If True, display a progress bar. Default False. | <code>False</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | list of subject matrices (observations x voxels/rois) | *required*
`method` |  | approach to computing ISFC. 'average' uses leave one out | <code>'average'</code>
`n_jobs` |  | (int) Number of parallel jobs to use. -1 means all available cores.     Default is -1 (parallel execution by default, consistent with other stats functions). | <code>-1</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pd.DataFrame, np.ndarray) observations x subjects data | *required*
`sampling_freq` |  | (float) sampling freqency of data in Hz | <code>0.5</code>
`low_cut` |  | (float) lower bound cutoff for high pass filter | <code>0.04</code>
`high_cut` |  | (float) upper bound cutoff for low pass filter | <code>0.07</code>
`order` |  | (int) filter order for butterworth bandpass | <code>5</code>
`pairwise` |  | (bool) compute phase angle coherence on pairwise phase angle differences     or on raw phase angle. | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`stats` |  | (dict) dictionary of permutation results ['isc', 'p', 'ci', 'null_dist']

######## `isc_group`

```python
isc_group(group1, group2, *, n_samples = 5000, summary = 'median', method = 'permute', ci_percentile = 95, exclude_self_corr = True, return_null = False, tail = 2, metric = 'correlation', n_jobs = -1, random_state = None, progress_bar = False)
```

Compute difference in intersubject correlation between groups.

This function computes pairwise intersubject correlations (ISC) using the median as recommended by Chen
et al., 2016). However, if the mean is preferred, we compute the mean correlation after performing
the fisher r-to-z transformation and then convert back to correlations to minimize artificially
inflating the correlation values.

There are currently two different methods to compute p-values. By default, we use the subject-wise permutation
method recommended Chen et al., 2016. This method combines the two groups and computes pairwise similarity both
within and between the groups. Then the group labels are permuted and the mean difference between the two groups
are recomputed to generate a null distribution. The second method uses subject-wise bootstrapping, where a new
pairwise similarity matrix with randomly selected subjects with replacement is created separately for each group
and the ISC difference between these groups is used to generate a null distribution. If the same subject is
selected multiple times, we set the perfect correlation to a nan with (exclude_self_corr=True). We compute the
p-values using the percentile method (Hall & Wilson, 1991).

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C., Israel, R. B.,
& Cox, R. W. (2016). Untangling the relatedness among correlations, part I:
nonparametric approaches to inter-subject correlation analysis at the group level.
NeuroImage, 142, 248-259.

Hall, P., & Wilson, S. R. (1991). Two guidelines for bootstrap hypothesis testing.
Biometrics, 757-762.

This function is a thin wrapper around `isc_group_permutation_test` from the inference
module (which provides optimized CPU parallelization and optional GPU acceleration),
pinning the classic pairwise behavior and the `n_samples` vocabulary.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`stats` |  | (dict) dictionary of permutation results with keys: - 'isc_group_difference': Observed ISC difference (float or array) - 'p': P-value (float or array) - 'ci': Confidence interval tuple (lower, upper) - 'null_dist': Null distribution (if return_null=True)

######## `isfc`

```python
isfc(data, method = 'average', n_jobs = -1)
```

Compute intersubject functional connectivity (ISFC) from a list of observation x feature matrices.

This function uses the leave one out approach to compute ISFC (Simony et al., 2016).
For each subject, compute the cross-correlation between each voxel/roi
with the average of the rest of the subjects data. In other words,
compute the mean voxel/ROI response for all participants except the
target subject. Then compute the correlation between each ROI within
the target subject with the mean ROI response in the group average.

Simony, E., Honey, C. J., Chen, J., Lositsky, O., Yeshurun, Y., Wiesel, A., & Hasson, U. (2016).
Dynamic reconfiguration of the default mode network during narrative comprehension.
Nature communications, 7, 12141.

This function now uses the optimized implementation from the inference module,
which provides efficient cross-correlation computation between matrix columns.
CPU parallelization is available via joblib when n_jobs > 1 or n_jobs=-1.
Each subject's ISFC computation is independent and can be parallelized efficiently.

**Returns:**

Type | Description
---- | -----------
 | list of subject ISFC matrices

######## `isps`

```python
isps(data, *, sampling_freq = 0.5, low_cut = 0.04, high_cut = 0.07, order = 5, pairwise = False)
```

Compute dynamic intersubject phase synchrony (ISPS) from an observations-by-subjects array.

This function computes the instantaneous intersubject phase synchrony for a single voxel/roi
timeseries. Requires multiple subjects. This method is largely based on that described by Glerean
et al., 2012 and performs a hilbert transform on narrow bandpass filtered timeseries (butterworth)
data to get the instantaneous phase angle. The function returns a dictionary containing the
average phase angle, the average vector length, and parametric p-values computed using the rayleigh test using circular
statistics (Fisher, 1993). If pairwise=True, then it will compute these on the pairwise phase angle differences,
if pairwise=False, it will compute these on the actual phase angles. This is called inter-site phase coupling
or inter-trial phase coupling respectively in the EEG literatures.

This function requires narrow band filtering your data. As a default we use the recommendations
by (Glerean et al., 2012) of .04-.07Hz. This is similar to the "slow-4" band (0.025–0.067 Hz)
described by (Zuo et al., 2010; Penttonen & Buzsáki, 2003), but excludes the .03 band, which has been
demonstrated to contain aliased respiration signals (Birn, 2006).

Birn RM, Smith MA, Bandettini PA, Diamond JB. 2006. Separating respiratory-variation-related
fluctuations from neuronal-activity- related fluctuations in fMRI. Neuroimage 31:1536–1548.

Buzsáki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical networks. Science,
304(5679), 1926-1929.

Fisher, N. I. (1995). Statistical analysis of circular data. cambridge university press.

Glerean, E., Salmi, J., Lahnakoski, J. M., Jääskeläinen, I. P., & Sams, M. (2012).
Functional magnetic resonance imaging phase synchronization as a measure of dynamic
functional connectivity. Brain connectivity, 2(2), 91-101.

**Returns:**

Type | Description
---- | -----------
 | dictionary with mean phase angle, vector length, and rayleigh statistic

###### `isc`

Intersubject Correlation (ISC) with GPU-Accelerated Permutation Testing.

This module provides both leave-one-out (LOO) and pairwise ISC computation
with efficient CPU-parallel and GPU-batched implementations. Follows the
statistical methods from Chen et al. (2016) for correct bootstrap resampling
of correlation matrices.

<details class="key-features" open markdown="1">
<summary>Key Features</summary>

- Two ISC modes: leave-one-out and pairwise (statistically different)
- GPU acceleration for voxel-wise computation (10-30× speedup)
- CPU-parallel bootstrap with joblib
- Correct subject-wise bootstrap (Chen et al. 2016)
- Memory-efficient condensed matrix storage

</details>

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

<details class="notes" open markdown="1">
<summary>Notes</summary>

Leave-one-out and pairwise ISC are monotonically correlated but
statistically different. LOO is computationally more efficient
and provides unbiased estimates. Pairwise captures full correlation
structure but is O(n²) in subjects.

</details>

**Methods:**

Name | Description
---- | -----------
[`isc_group_permutation_test`](#algorithms-isc-group-permutation-test) | Compute ISC difference between groups with permutation testing.
[`isc_permutation_test`](#algorithms-isc-permutation-test) | Compute intersubject correlation with permutation testing.



####### Attributes

####### Functions##

###### `isc_group_permutation_test`

```python
isc_group_permutation_test(group1: np.ndarray, group2: np.ndarray, *, n_permute: int = 5000, summary: Literal['median', 'mean'] = 'median', method: Literal['permute', 'bootstrap'] = 'permute', summary_statistic: Literal['leave-one-out', 'pairwise'] = 'pairwise', ci_percentile: float = 95, tail: int | str = 2, device: Literal['cpu', 'gpu'] | None = 'cpu', n_jobs: int = -1, random_state: int | None = None, return_null: bool = False, progress_bar: bool = False, exclude_self_corr: bool = True, metric: str = 'correlation') -> dict[str, Any]
```

Compute ISC difference between groups with permutation testing.

Supports both subject-wise permutation and bootstrap methods with efficient
CPU-parallel and optional GPU acceleration. Follows the statistical methods
from Chen et al. (2016) for correct group comparison inference.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`group1` | <code>[ndarray](#numpy.ndarray)</code> | First group data with one of the following shapes: - (n_observations, n_subjects1): Single feature - (n_observations, n_subjects1, n_voxels): Voxel-wise | *required*
`group2` | <code>[ndarray](#numpy.ndarray)</code> | Second group data with one of the following shapes: - (n_observations, n_subjects2): Single feature - (n_observations, n_subjects2, n_voxels): Voxel-wise | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations/bootstrap iterations. Defaults to 5000. | <code>5000</code>
`summary` | <code>[Literal](#typing.Literal)['median', 'mean']</code> | Summary statistic for aggregating ISC values: - 'median': Direct median (robust to outliers) - 'mean': Fisher z-transformed mean (unbiased averaging) Defaults to 'median'. | <code>'median'</code>
`method` | <code>[Literal](#typing.Literal)['permute', 'bootstrap']</code> | Resampling method for p-value computation: - 'permute': Subject-wise permutation (combines groups, permutes labels) - 'bootstrap': Subject-wise bootstrap (resamples within each group) Defaults to 'permute'. | <code>'permute'</code>
`summary_statistic` | <code>[Literal](#typing.Literal)['leave-one-out', 'pairwise']</code> | ISC computation method: - 'pairwise': Average all pairwise correlations - 'leave-one-out': Correlate each subject with mean of others Defaults to 'pairwise'. | <code>'pairwise'</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95 for 95% CI). Defaults to 95. | <code>95</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Two-tailed (2 or 'two', default) or one-tailed (1 or 'one', positive direction) p-value. | <code>2</code>
`device` | <code>[Literal](#typing.Literal)['cpu', 'gpu'] \| None</code> | Parallelization method: - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (10-30× speedup for voxel-wise LOO) - None: Single-threaded NumPy (for debugging/small problems) Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when device='cpu'. Defaults to -1. | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, return null distribution in result dict. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during bootstrap/permutation. Defaults to False. | <code>False</code>
`exclude_self_corr` | <code>[bool](#bool)</code> | Mask self-correlations in bootstrap (pairwise only). Defaults to True. | <code>True</code>
`metric` | <code>[str](#str)</code> | Similarity metric for pairwise ISC computation. See sklearn.metrics.pairwise_distances for valid options. Only applies when summary_statistic='pairwise'. Defaults to 'correlation'. | <code>'correlation'</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Data array with one of the following shapes: - (n_observations, n_subjects): Single feature ISC - (n_observations, n_subjects, n_voxels): Voxel-wise ISC | *required*
`n_permute` | <code>[int](#int)</code> | Number of bootstrap iterations or permutations. Defaults to 5000. | <code>5000</code>
`summary` | <code>[Literal](#typing.Literal)['median', 'mean']</code> | Summary statistic to aggregate ISC values. - 'median': Direct median (robust to outliers) - 'mean': Fisher z-transformed mean (unbiased averaging) Defaults to 'median'. | <code>'median'</code>
`summary_statistic` | <code>[Literal](#typing.Literal)['leave-one-out', 'pairwise']</code> | ISC computation method. Options: - 'leave-one-out': Correlate each subject with mean of others. O(n_subjects), unbiased, recommended by Chen et al. 2016. - 'pairwise': Average all pairwise correlations. O(n_subjects²), captures full correlation structure. Note: These methods are statistically different and monotonically but non-linearly related (see Chen et al. 2016, Figure 3). Defaults to 'pairwise'. | <code>'pairwise'</code>
`method` | <code>[Literal](#typing.Literal)['bootstrap', 'circle_shift', 'phase_randomize']</code> | Resampling method for p-value computation: - 'bootstrap': Subject-wise bootstrap (default, Chen et al. 2016) - 'circle_shift': Circular time-series shift (preserves autocorrelation) - 'phase_randomize': FFT phase randomization (preserves power spectrum) Defaults to 'bootstrap'. | <code>'bootstrap'</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95 for 95% CI). Defaults to 95. | <code>95</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Two-tailed (2 or 'two', default) or one-tailed (1 or 'one', positive direction) p-value. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return bootstrap/permutation distribution in result dict. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during bootstrap/permutation. Defaults to False. | <code>False</code>
`exclude_self_corr` | <code>[bool](#bool)</code> | If True, mask self-correlations (perfect correlations from duplicate subjects in bootstrap samples) as NaN. If False, include them in the summary statistic. Only applies when method='bootstrap' and summary_statistic='pairwise'. Defaults to True. | <code>True</code>
`metric` | <code>[str](#str)</code> | Similarity metric for pairwise ISC computation. See sklearn.metrics.pairwise_distances for valid options. Only applies when summary_statistic='pairwise'. For 'correlation', uses optimized np.corrcoef. Other metrics use pairwise_distances. Defaults to 'correlation'. | <code>'correlation'</code>
`device` | <code>[Literal](#typing.Literal)['cpu', 'gpu'] \| None</code> | Parallelization method: - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (10-30× speedup for voxel-wise LOO) - None: Single-threaded NumPy (for debugging/small problems) Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when device='cpu'. Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float) \| None</code> | GPU working-set budget in GB. For the pairwise GPU bootstrap (``device='gpu'``, ``summary_statistic='pairwise'``, ``method='bootstrap'``) this bounds the ``(perm_batch, voxel_chunk, n_subjects, n_subjects)`` resample tensor, chunking voxels and permutations to fit — so whole-brain runs stay within budget. Not used by the LOO or surrogate (circle_shift/phase_randomize) paths. Defaults to 4. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with the following keys:
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'isc_group_difference': Observed ISC difference (float or array per voxel)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'p': P-value (Phipson-Smyth corrected)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'ci': Confidence interval tuple (lower, upper)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'device': Parallelization method used
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'null_dist': (optional) Bootstrap/permutation distribution

**Examples:**

```pycon
>>> # Single-feature ISC group comparison
>>> group1 = np.random.randn(100, 10)  # 10 subjects
>>> group2 = np.random.randn(100, 10)
>>> result = isc_group_permutation_test(group1, group2, n_permute=1000)
>>> print(f"ISC difference: {result['isc_group_difference']:.3f}, p: {result['p']:.3f}")
```

```pycon
>>> # Voxel-wise ISC group comparison with GPU acceleration
>>> group1_voxels = np.random.randn(100, 10, 5000)  # 5K voxels
>>> group2_voxels = np.random.randn(100, 10, 5000)
>>> result = isc_group_permutation_test(
...     group1_voxels,
...     group2_voxels,
...     summary_statistic='leave-one-out',
...     device='gpu',  # GPU for LOO computation
...     n_permute=5000
... )
>>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")
```

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Permutation method combines groups and permutes labels (Chen et al. 2016)
- Bootstrap method resamples subjects within each group independently
- Bootstrap distribution is centered by subtracting observed difference
- GPU acceleration available for voxel-wise LOO computation

</details>

######## `isc_permutation_test`

```python
isc_permutation_test(data: np.ndarray, *, n_permute: int = 5000, summary: Literal['median', 'mean'] = 'median', summary_statistic: Literal['leave-one-out', 'pairwise'] = 'pairwise', method: Literal['bootstrap', 'circle_shift', 'phase_randomize'] = 'bootstrap', ci_percentile: float = 95, tail: int | str = 2, return_null: bool = False, progress_bar: bool = False, exclude_self_corr: bool = True, metric: str = 'correlation', device: Literal['cpu', 'gpu'] | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> dict[str, Any]
```

Compute intersubject correlation with permutation testing.

Supports both leave-one-out and pairwise ISC computation modes with
GPU acceleration for large voxel-wise problems and CPU-parallel
bootstrap resampling.

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with the following keys:
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'isc': Observed ISC value (float or array per voxel)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'p': P-value (Phipson-Smyth corrected)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'ci': Confidence interval tuple (lower, upper)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'device': Parallelization method used
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'null_dist': (optional) Bootstrap/permutation distribution

**Examples:**

```pycon
>>> # Single-feature ISC
>>> data = np.random.randn(100, 10)  # 100 timepoints, 10 subjects
>>> result = isc_permutation_test(data, n_permute=1000)
>>> print(f"ISC: {result['isc']:.3f}, p: {result['p']:.3f}")
```

```pycon
>>> # Voxel-wise ISC with GPU acceleration
>>> data_voxels = np.random.randn(100, 50, 5000)  # 5K voxels
>>> result = isc_permutation_test(
...     data_voxels,
...     summary_statistic='leave-one-out',
...     device='gpu',  # GPU for LOO computation
...     n_permute=5000
... )
>>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")
```

```pycon
>>> # Compare LOO vs pairwise
>>> result_loo = isc_permutation_test(data, summary_statistic='leave-one-out')
>>> result_pair = isc_permutation_test(data, summary_statistic='pairwise')
>>> print(f"LOO: {result_loo['isc']:.3f}, Pairwise: {result_pair['isc']:.3f}")
```

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Leave-one-out is 20-30× faster than pairwise for large n_subjects
- GPU acceleration helps most for voxel-wise LOO (10-30× speedup)
- Pairwise bootstrap uses correct subject-wise resampling (Chen 2016)
- Bootstrap distribution is centered by subtracting observed ISC

</details>

(algorithms-matrix)=
###### `matrix`

Matrix permutation test implementations (Mantel test).

This module provides CPU-parallel implementations of matrix permutation tests
for testing correlation between two square matrices, as well as matrix utility
functions for distance correlation and matrix centering operations.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`MAX_INT`](#algorithms-max-int) |  | 



####### Attributes##

**Methods:**

Name | Description
---- | -----------
[`distance_correlation`](#algorithms-distance-correlation) | Compute the distance correlation between 2 arrays to test for multivariate dependence (linear or non-linear).
[`double_center`](#algorithms-double-center) | Double center a 2d array.
[`matrix_permutation_test`](#algorithms-matrix-permutation-test) | Matrix permutation test (Mantel test) for correlating two square matrices.
[`u_center`](#algorithms-u-center) | U-center a 2d array. U-centering is a bias-corrected form of double-centering.

(algorithms-max-int)=
###### `MAX_INT`

```python
MAX_INT = np.iinfo(np.int32).max
```



####### Functions##

###### `distance_correlation`

```python
distance_correlation(x: np.ndarray, y: np.ndarray, bias_corrected: bool = True, ttest: bool = False) -> dict
```

Compute the distance correlation between 2 arrays to test for multivariate dependence (linear or non-linear).

Arrays must match on their first dimension. It's almost always preferable to compute the bias_corrected
version which can also optionally perform a ttest. This ttest operates on a statistic thats ~dcorr^2
and will be also returned.

Explanation:
Distance correlation involves computing the normalized covariance of two centered euclidean distance
matrices. Each distance matrix is the euclidean distance between rows (if x or y are 2d) or scalars
(if x or y are 1d). Each matrix is centered prior to computing the covariance either using double-centering
or u-centering, which corrects for bias as the number of dimensions increases. U-centering is almost always
preferred in all cases. It also permits inference of the normalized covariance between each distance matrix
using a one-tailed directional t-test. (Szekely & Rizzo, 2013). While distance correlation is normally
bounded between 0 and 1, u-centering can produce negative estimates, which are never significant.

Validated against the dcor and dcor.ttest functions in the 'energy' R package and the
dcor.distance_correlation, dcor.udistance_correlation_sqr, and dcor.independence.distance_correlation_t_test
functions in the dcor Python package.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` | <code>[ndarray](#ndarray)</code> | 1d or 2d numpy array of observations by features | *required*
`y` | <code>[ndarray](#ndarray)</code> | 1d or 2d numpy array of observations by features | *required*
`bias_corrected` | <code>[bool](#bool)</code> | if false use double-centering which produces a biased-estimate that converges to 1 as the number of dimensions increase. Otherwise used u-centering to correct this bias. **Note** this must be True if ttest=True; default True | <code>True</code>
`ttest` | <code>[bool](#bool)</code> | perform a ttest using the bias_corrected distance correlation; default False | <code>False</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mat` | <code>[ndarray](#ndarray)</code> | 2d numpy array | *required*

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | First square matrix (n×n) | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Second square matrix (n×n) | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000) | <code>5000</code>
`metric` | <code>[str](#str)</code> | Correlation metric ['pearson'|'spearman'|'kendall'] (default: 'pearson') | <code>'pearson'</code>
`how` | <code>[str](#str)</code> | Which elements to compare ['upper'|'lower'|'full'] (default: 'upper') - 'upper': Upper triangle only (assumes symmetric matrices) - 'lower': Lower triangle only - 'full': All elements (see include_diag) | <code>'upper'</code>
`include_diag` | <code>[bool](#bool)</code> | Include diagonal elements (only applies if how='full') (default: False) | <code>False</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type — 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction) - 2 | 'two': Two-tailed test (r != 0) - 1 | 'one': One-tailed (r > 0; negate the data for the other direction) | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | Return null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel workers, -1 = all cores (default: -1) Only used when device='cpu' | <code>-1</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show a progress bar over permutations (default: False) | <code>False</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mat` | <code>[ndarray](#ndarray)</code> | 2d numpy array | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`results` | <code>[dict](#dict)</code> | dictionary of results (correlation, t, p, and df.) Optionally, covariance, x variance, and y variance

**Examples:**

```pycon
>>> import numpy as np
>>> x = np.random.randn(20, 3)
>>> y = x + np.random.randn(20, 3) * 0.1  # Strongly correlated
>>> result = distance_correlation(x, y, bias_corrected=True)
>>> 'dcorr' in result
True
>>> 0 <= result['dcorr'] <= 1
True
```

######## `double_center`

```python
double_center(mat: np.ndarray) -> np.ndarray
```

Double center a 2d array.

Double-centering subtracts row means, column means, and adds the grand mean.
This centers both rows and columns around zero.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`mat` | <code>[ndarray](#ndarray)</code> | double-centered version of input

**Examples:**

```pycon
>>> mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
>>> result = double_center(mat)
>>> np.allclose(result.mean(axis=0), 0)
True
>>> np.allclose(result.mean(axis=1), 0)
True
```

######## `matrix_permutation_test`

```python
matrix_permutation_test(data1: np.ndarray, data2: np.ndarray, *, n_permute: int = 5000, metric: str = 'pearson', how: str = 'upper', include_diag: bool = False, tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Matrix permutation test (Mantel test) for correlating two square matrices.

Tests whether the correlation between elements of two matrices is significant
by permuting rows and columns of one matrix symmetrically while keeping the
other fixed.

**Statistical Method**:
For each permutation, create random permutation `perm`, then apply:
`matrix1[perm][:, perm]`. This preserves matrix structure while destroying
correlation. Count how often permuted correlation is as extreme as observed.

**Assumptions**:
- Matrices are square and same size
- Under H₀, row/column ordering is exchangeable
- Symmetric permutation preserves matrix properties (e.g., symmetry)

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'correlation' (float): Observed correlation coefficient - 'p' (float): P-value using Phipson-Smyth correction - 'device' (str): Parallelization method used ('cpu' or None) - 'null_dist' (np.ndarray): Null distribution (if return_null=True)

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G. et al. (2016). Untangling the relatedness among correlations,
part I: nonparametric approaches to inter-subject correlation analysis
at the group level. NeuroImage, 142, 248-259.

Mantel, N. (1967). The detection of disease clustering and a generalized
regression approach. Cancer Research, 27(2), 209-220.

</details>

**Examples:**

```pycon
>>> import numpy as np
>>> from nltools.algorithms.inference import matrix_permutation_test
>>>
>>> # Create two correlated similarity matrices
>>> np.random.seed(42)
>>> n = 50
>>> true_pattern = np.random.randn(n)
>>> data1 = np.corrcoef(true_pattern + np.random.randn(n) * 0.1)
>>> data2 = np.corrcoef(true_pattern + np.random.randn(n) * 0.1)
>>>
>>> # Test if matrices are correlated
>>> result = matrix_permutation_test(data1, data2, n_permute=1000)
>>> print(f"Correlation: {result['correlation']:.3f}, p = {result['p']:.4f}")
```

######## `u_center`

```python
u_center(mat: np.ndarray) -> np.ndarray
```

U-center a 2d array. U-centering is a bias-corrected form of double-centering.

U-centering corrects for bias that occurs with double-centering as the number
of dimensions increases. The diagonal is explicitly set to zero.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`mat` | <code>[ndarray](#ndarray)</code> | u-centered version of input

**Examples:**

```pycon
>>> mat = np.random.randn(5, 5)
>>> result = u_center(mat)
>>> np.allclose(np.diag(result), 0)
True
```

(algorithms-one-sample)=
###### `one_sample`

One-sample permutation test implementations.

This module provides CPU-parallel and GPU-batched implementations
of the one-sample permutation test (sign-flipping test).

**Methods:**

Name | Description
---- | -----------
[`one_sample_permutation_test`](#algorithms-one-sample-permutation-test) | One-sample permutation test using sign-flipping.



####### Classes

####### Functions##

###### `one_sample_permutation_test`

```python
one_sample_permutation_test(data: np.ndarray, *, n_permute: int = 5000, tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None, progress_bar: bool = False) -> dict
```

One-sample permutation test using sign-flipping.

Tests whether the mean of data is significantly different from zero
by randomly flipping the sign of each observation. This is the
permutation test equivalent of a one-sample t-test.

Assumption: Symmetric error distribution around zero. For highly skewed
distributions, consider alternative methods (e.g., bootstrap resampling).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Data to test - shape (n_samples,) for single feature - shape (n_samples, n_features) for multi-feature (voxel-wise) | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000) | <code>5000</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type — 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction) - 2 | 'two': Two-tailed test (mean != 0) - 1 | 'one': One-tailed (mean > 0; negate the data for the other   direction). The fixed direction keeps MCP correction valid. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Whether to display a progress bar (default: False) | <code>False</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'mean' (float or np.ndarray): Observed mean(s) - 'p' (float or np.ndarray): P-value(s) - 'null_dist' (np.ndarray): Null distribution (if return_null=True) - 'device' (str): Parallelization method used

**Examples:**

```pycon
>>> # Single feature (default CPU parallelization)
>>> data = np.random.randn(30)
>>> result = one_sample_permutation_test(data, n_permute=5000)
>>> result['p']
0.23
```

```pycon
>>> # Voxel-wise test with GPU
>>> data = np.random.randn(30, 10000)  # 30 subjects, 10K voxels
>>> result = one_sample_permutation_test(data, n_permute=5000, device='gpu')
>>> result['mean'].shape
(10000,)
>>> result['p'].shape
(10000,)
```

```pycon
>>> # Single-threaded (for debugging)
>>> result = one_sample_permutation_test(data, n_permute=5000, device=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (device='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
- Single-threaded (device=None): Use for small problems or debugging
- For voxel-wise tests, each voxel tested independently
- Progress bars show completion for both CPU parallel and GPU batched modes

</details>

(algorithms-timeseries)=
###### `timeseries`

Time-series permutation test implementations.

This module provides GPU-accelerated implementations of time-series
permutation tests that preserve temporal structure:

- circle_shift: Circular shift permutation (preserves autocorrelation)
- phase_randomize: FFT-based phase randomization (preserves power spectrum)
- timeseries_correlation_permutation_test: Correlation test with timeseries methods

<details class="references" open markdown="1">
<summary>References</summary>

Theiler, J., Galdrikian, B., Longtin, A., Eubank, S., & Farmer, J. D. (1991).
Testing for nonlinearity in time series: the method of surrogate data
(No. LA-UR-91-3343; CONF-9108181-1). Los Alamos National Lab., NM (United States).

Lancaster, G., Iatsenko, D., Pidde, A., Ticcinelli, V., & Stefanovska, A. (2018).
Surrogate data for hypothesis testing of physical systems. Physics Reports, 748, 1-60.

</details>

**Methods:**

Name | Description
---- | -----------
[`circle_shift`](#algorithms-circle-shift) | Circular shift for time-series data.
[`phase_randomize`](#algorithms-phase-randomize) | FFT-based phase randomization for time-series data.
[`timeseries_correlation_permutation_test`](#algorithms-timeseries-correlation-permutation-test) | Time-series correlation permutation test.



####### Classes

####### Functions##

###### `circle_shift`

```python
circle_shift(data: np.ndarray, shift_amount: int | np.ndarray | None = None, random_state: int | np.random.RandomState | None = None) -> np.ndarray
```

Circular shift for time-series data.

Performs a circular shift that preserves autocorrelation structure.
Useful for permutation tests on autocorrelated time series (e.g., fMRI).
For 1D data, shifts by a single amount. For 2D data, shifts each
feature (column) independently.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Time series data, shape (n_samples,) or (n_samples, n_features) | *required*
`shift_amount` | <code>[int](#int) \| [ndarray](#numpy.ndarray) \| None</code> | Shift amount(s). If None, random shift is used. For 1D: int specifying shift amount For 2D: array of length n_features with shift per feature | <code>None</code>
`random_state` | <code>[int](#int) \| [RandomState](#numpy.random.RandomState) \| None</code> | Random seed for reproducibility (if shift_amount is None) | <code>None</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Time series data, shape (n_samples,) or (n_samples, n_features) | *required*
`device` | <code>[str](#str) \| None</code> | Compute device. - 'cpu' / None: NumPy FFT (default, float64 precision) - 'gpu': PyTorch FFT on CUDA/MPS (float32 precision, 5-20× faster for large data) - 'auto': use a GPU if present, else CPU | <code>'cpu'</code>
`random_state` | <code>[int](#int) \| [RandomState](#numpy.random.RandomState) \| None</code> | Random seed for reproducibility | <code>None</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | First time series, shape (n_samples,) or (n_samples, 1) | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Second time series, shape (n_samples,) or (n_samples, 1) | *required*
`method` | <code>[Literal](#typing.Literal)['circle_shift', 'phase_randomize']</code> | Permutation method: - 'circle_shift': Circular shift (preserves autocorrelation) - 'phase_randomize': FFT-based (preserves power spectrum) | <code>'circle_shift'</code>
`n_permute` | <code>[int](#int)</code> | Number of permutations | <code>5000</code>
`metric` | <code>[Literal](#typing.Literal)['pearson', 'spearman', 'kendall']</code> | Correlation type ('pearson', 'spearman', 'kendall') | <code>'pearson'</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type (default: 2) - 2 or 'two': Two-tailed test (default) - 1 or 'one': One-tailed test in the test's positive direction   (to test the negative direction, negate the data / swap groups) | <code>2</code>
`device` | <code>[str](#str) \| None</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs (-1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float) \| None</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | Whether to return null distribution | <code>False</code>
`random_state` | <code>[int](#int) \| [RandomState](#numpy.random.RandomState) \| None</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show a progress bar over permutations (default: False) | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Circularly shifted data with same shape as input

**Examples:**

```pycon
>>> x = np.array([1, 2, 3, 4, 5])
>>> circle_shift(x, shift_amount=2)
array([4, 5, 1, 2, 3])
```

```pycon
>>> X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
>>> circle_shift(X, shift_amount=np.array([1, 2]))
array([[ 4, 30],
       [ 1, 40],
       [ 2, 10],
       [ 3, 20]])
```

######## `phase_randomize`

```python
phase_randomize(data: np.ndarray, *, device: str | None = 'cpu', random_state: int | np.random.RandomState | None = None) -> np.ndarray
```

FFT-based phase randomization for time-series data.

Preserves the power spectrum (autocorrelation) but destroys nonlinear
temporal structure by randomizing Fourier phases. Used to test whether
data was generated by a linear Gaussian process or contains nonlinear
dynamics.

<details class="algorithm" open markdown="1">
<summary>Algorithm</summary>

1. Compute FFT of input signal
2. Generate random phases [0, 2π] for positive frequencies
3. Apply phase shifts to positive frequencies: multiply by exp(i*φ)
4. Apply conjugate phase shifts to negative frequencies (for real output)
5. Compute inverse FFT to get phase-randomized signal

</details>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Phase-randomized data with same shape as input

<details class="notes" open markdown="1">
<summary>Notes</summary>

- **CRITICAL**: Preserves power spectrum exactly (within numerical precision)
- Precision: the CPU path uses float64, the GPU path float32
- Conjugate symmetry is maintained for real-valued output

</details>

**Examples:**

```pycon
>>> x = np.sin(np.linspace(0, 10*np.pi, 100))  # Sine wave
>>> x_rand = phase_randomize(x, random_state=42)
>>> # Power spectrum preserved:
>>> np.allclose(np.abs(np.fft.rfft(x))**2, np.abs(np.fft.rfft(x_rand))**2)
True
```

```pycon
>>> # GPU acceleration for large datasets:
>>> x_large = np.random.randn(10000)
>>> x_rand_gpu = phase_randomize(x_large, device='gpu', random_state=42)
```

######## `timeseries_correlation_permutation_test`

```python
timeseries_correlation_permutation_test(data1: np.ndarray, data2: np.ndarray, *, method: Literal['circle_shift', 'phase_randomize'] = 'circle_shift', n_permute: int = 5000, metric: Literal['pearson', 'spearman', 'kendall'] = 'pearson', tail: int | str = 2, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, return_null: bool = False, random_state: int | np.random.RandomState | None = None, progress_bar: bool = False) -> dict
```

Time-series correlation permutation test.

Unlike standard permutation tests that shuffle data independently,
this test uses time-series-aware permutation methods that preserve
temporal structure (circle_shift) or power spectrum (phase_randomize).

Use this test when data contains temporal autocorrelation. Standard
permutation tests inflate Type I error for autocorrelated data.

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with keys: - 'correlation': Observed correlation coefficient - 'p': P-value - 'null_dist': (if return_null=True) Null distribution - 'device': Parallelization method used

**Examples:**

```pycon
>>> x = np.sin(np.linspace(0, 10*np.pi, 100))
>>> y = np.cos(np.linspace(0, 10*np.pi, 100))
>>> result = timeseries_correlation_permutation_test(
...     x, y, method='circle_shift', n_permute=1000, random_state=42
... )
>>> result['correlation']  # Strong negative correlation
-0.999...
>>> result['p'] < 0.05  # Significant
True
```

```pycon
>>> # GPU acceleration
>>> result = timeseries_correlation_permutation_test(
...     x, y, method='phase_randomize', device='gpu', n_permute=5000
... )
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (device='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): 5-20× faster for large problems (n_samples > 1000)
- Single-threaded (device=None): Use for small problems or debugging
- For independent data, use regular correlation_permutation_test
- circle_shift is faster and suitable for most fMRI time series
- phase_randomize preserves power spectrum exactly (tests nonlinearity)
- Only data1 is randomized; data2 remains fixed to test correlation
- phase_randomize benefits most from GPU (FFT acceleration)

</details>

(algorithms-two-sample)=
###### `two_sample`

Two-sample permutation test implementations.

This module provides CPU-parallel and GPU-batched implementations
of the two-sample permutation test (group permutation test).

**Methods:**

Name | Description
---- | -----------
[`two_sample_permutation_test`](#algorithms-two-sample-permutation-test) | Two-sample permutation test using group label shuffling.



####### Classes

####### Functions##

###### `two_sample_permutation_test`

```python
two_sample_permutation_test(data1: np.ndarray, data2: np.ndarray, *, n_permute: int = 5000, tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Two-sample permutation test using group label shuffling.

Tests whether two independent groups have different means by randomly
permuting group labels. This is the permutation test equivalent of an
independent samples t-test.

Assumption: Exchangeability under the null hypothesis (group assignments
are arbitrary). Valid for independent samples from similar distributions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | Group 1 data - shape (n_samples1,) for single feature - shape (n_samples1, n_features) for multi-feature (voxel-wise) | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Group 2 data - shape (n_samples2,) for single feature - shape (n_samples2, n_features) for multi-feature (voxel-wise) | *required*
`n_permute` | <code>[int](#int)</code> | Number of permutations (default: 5000) | <code>5000</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type — 2|'two' (two-tailed, default) or 1|'one' (one-tailed, positive direction) - 2 | 'two': Two-tailed test (mean1 != mean2) - 1 | 'one': One-tailed (mean1 > mean2; swap the groups for the   other direction). The fixed direction keeps MCP correction valid. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'mean_diff' (float or np.ndarray): Observed mean difference (data1 - data2) - 'p' (float or np.ndarray): P-value(s) - 'null_dist' (np.ndarray): Null distribution (if return_null=True) - 'device' (str): Parallelization method used

**Examples:**

```pycon
>>> # Single feature (default CPU parallelization)
>>> data1 = np.random.randn(20)  # Group 1: 20 subjects
>>> data2 = np.random.randn(25)  # Group 2: 25 subjects
>>> result = two_sample_permutation_test(data1, data2, n_permute=5000)
>>> result['p']
0.45
```

```pycon
>>> # Voxel-wise test with GPU
>>> data1 = np.random.randn(20, 10000)  # 20 subjects, 10K voxels
>>> data2 = np.random.randn(25, 10000)  # 25 subjects, 10K voxels
>>> result = two_sample_permutation_test(data1, data2, n_permute=5000, device='gpu')
>>> result['mean_diff'].shape
(10000,)
>>> result['p'].shape
(10000,)
```

```pycon
>>> # Single-threaded (for debugging)
>>> result = two_sample_permutation_test(data1, data2, n_permute=5000, device=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (device='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
- Single-threaded (device=None): Use for small problems or debugging
- For voxel-wise tests, each voxel tested independently
- Group sizes can be unequal

</details>

(algorithms-utils)=
###### `utils`

Utility functions for permutation testing.

This module contains shared helper functions used across different
permutation test implementations.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`EPSILON`](#algorithms-epsilon) |  | 



####### Attributes##

(algorithms-epsilon)=
###### `EPSILON`

```python
EPSILON = 1e-10
```



####### Functions

(algorithms-validation)=
###### `validation`

Shared validation utilities for algorithms module.

This module provides common validation functions to reduce code duplication
and ensure consistent error handling across the algorithms module.

<details class="usage" open markdown="1">
<summary>Usage</summary>

These functions are used throughout the algorithms module to validate
input parameters. They provide consistent error messages and behavior.

Example:
    >>> from nltools.algorithms.validation import validate_device_parameter
    >>> validate_device_parameter("cpu")  # OK
    >>> validate_device_parameter("invalid")  # Raises ValueError

</details>

**Methods:**

Name | Description
---- | -----------
[`validate_array_shape`](#algorithms-validate-array-shape) | Validate array dimensionality.
`validate_array_shape_range` | Validate array dimensionality is within a range.
`validate_bootstrap_data` | Validate input data for bootstrapping.
`validate_bootstrap_method` | Validate bootstrap method name.
`validate_device_parameter` | Validate device parameter.
`validate_device_parameter_matrix` | Validate device parameter for matrix operations.
`validate_how_parameter` | Validate 'how' parameter for matrix operations.
`validate_metric_parameter` | Validate metric parameter.
`validate_percentiles` | Validate percentile values for confidence intervals.
`validate_same_shape` | Validate two arrays have same shape.
`validate_shape_compatibility` | Validate that X and y have compatible shapes for regression.
`validate_square_matrix` | Validate matrix is square.
`validate_tail_parameter` | Validate the public tail vocabulary and normalize to the internal form.



####### Functions##

(algorithms-validate-array-shape)=
###### `validate_array_shape`

```python
validate_array_shape(array: np.ndarray, expected_ndim: int, name: str = 'array') -> None
```

Validate array dimensionality.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` | <code>[ndarray](#numpy.ndarray)</code> | Array to validate | *required*
`expected_ndim` | <code>[int](#int)</code> | Expected number of dimensions | *required*
`name` | <code>[str](#str)</code> | Name of array for error message | <code>'array'</code>

######## `validate_array_shape_range`

```python
validate_array_shape_range(array: np.ndarray, min_ndim: int, max_ndim: int, name: str = 'array') -> None
```

Validate array dimensionality is within a range.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` | <code>[ndarray](#numpy.ndarray)</code> | Array to validate | *required*
`min_ndim` | <code>[int](#int)</code> | Minimum number of dimensions (inclusive) | *required*
`max_ndim` | <code>[int](#int)</code> | Maximum number of dimensions (inclusive) | *required*
`name` | <code>[str](#str)</code> | Name of array for error message | <code>'array'</code>

######## `validate_bootstrap_data`

```python
validate_bootstrap_data(data: np.ndarray, method: str) -> None
```

Validate input data for bootstrapping.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[ndarray](#numpy.ndarray)</code> | Data to validate | *required*
`method` | <code>[str](#str)</code> | Bootstrap method | *required*

######## `validate_bootstrap_method`

```python
validate_bootstrap_method(method: str, simple_methods: list[str], fitted_methods: list[str]) -> None
```

Validate bootstrap method name.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`method` | <code>[str](#str)</code> | Method name to validate | *required*
`simple_methods` | <code>[list](#list)[[str](#str)]</code> | List of simple method names | *required*
`fitted_methods` | <code>[list](#list)[[str](#str)]</code> | List of fitted method names | *required*

######## `validate_device_parameter`

```python
validate_device_parameter(device: str | None, *, allow_auto: bool = False) -> None
```

Validate device parameter.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`device` | <code>[str](#str) \| None</code> | Device parameter value (None, 'cpu', or 'gpu') | *required*
`allow_auto` | <code>[bool](#bool)</code> | Also accept 'auto' (entry points that resolve the device themselves, e.g. `phase_randomize`) | <code>False</code>

######## `validate_device_parameter_matrix`

```python
validate_device_parameter_matrix(device: str | None) -> None
```

Validate device parameter for matrix operations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`device` | <code>[str](#str) \| None</code> | Parallel parameter value | *required*

######## `validate_how_parameter`

```python
validate_how_parameter(how: str) -> None
```

Validate 'how' parameter for matrix operations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`how` | <code>[str](#str)</code> | How parameter value | *required*

######## `validate_metric_parameter`

```python
validate_metric_parameter(metric: str, allowed: list[str], name: str = 'metric') -> None
```

Validate metric parameter.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` | <code>[str](#str)</code> | Metric parameter value | *required*
`allowed` | <code>[list](#list)[[str](#str)]</code> | List of allowed metric values | *required*
`name` | <code>[str](#str)</code> | Name of parameter for error message | <code>'metric'</code>

######## `validate_percentiles`

```python
validate_percentiles(percentiles: tuple[float, float]) -> None
```

Validate percentile values for confidence intervals.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`percentiles` | <code>[tuple](#tuple)[[float](#float), [float](#float)]</code> | Percentile values (lower, upper) | *required*

######## `validate_same_shape`

```python
validate_same_shape(array1: np.ndarray, array2: np.ndarray, name1: str = 'array1', name2: str = 'array2') -> None
```

Validate two arrays have same shape.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array1` | <code>[ndarray](#numpy.ndarray)</code> | First array | *required*
`array2` | <code>[ndarray](#numpy.ndarray)</code> | Second array | *required*
`name1` | <code>[str](#str)</code> | Name of first array for error message | <code>'array1'</code>
`name2` | <code>[str](#str)</code> | Name of second array for error message | <code>'array2'</code>

######## `validate_shape_compatibility`

```python
validate_shape_compatibility(X: np.ndarray, y: np.ndarray, X_name: str = 'X', y_name: str = 'y') -> None
```

Validate that X and y have compatible shapes for regression.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Feature matrix | *required*
`y` | <code>[ndarray](#numpy.ndarray)</code> | Target vector or matrix | *required*
`X_name` | <code>[str](#str)</code> | Name of X for error message | <code>'X'</code>
`y_name` | <code>[str](#str)</code> | Name of y for error message | <code>'y'</code>

######## `validate_square_matrix`

```python
validate_square_matrix(matrix: np.ndarray, name: str = 'matrix') -> None
```

Validate matrix is square.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`matrix` | <code>[ndarray](#numpy.ndarray)</code> | Matrix to validate | *required*
`name` | <code>[str](#str)</code> | Name of matrix for error message | <code>'matrix'</code>

######## `validate_tail_parameter`

```python
validate_tail_parameter(tail: int | str) -> str
```

Validate the public tail vocabulary and normalize to the internal form.

The public vocabulary (v0.6.0) is deliberately two-valued — the *direction*
of a one-tailed test is fixed by the test's convention, never chosen from
the data (a data-driven direction would silently halve every p-value):

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`tail` | <code>[int](#int) \| [str](#str)</code> | Tail parameter value. Can be: - 2 or 'two' (default everywhere): two-tailed test (|obs| vs |null|) - 1 or 'one': one-tailed test in the test's canonical positive   direction (correlation/ISC/similarity > 0, mean > popmean,   group1 > group2). To test the negative direction, negate your   data, swap the groups, or flip the contrast. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | Normalized internal tail string: 'two' or 'upper'

<details class="notes" open markdown="1">
<summary>Notes</summary>

For multiple comparisons correction (FDR, Bonferroni) a fixed direction
across all tests is essential — which is exactly why the direction is
part of the vocabulary, not the data. See GH #315.

</details>

(algorithms-outliers)=
#### `outliers`

Outlier detection, robust statistics, and data normalization.

**Methods:**

Name | Description
---- | -----------
[`find_spikes`](#algorithms-find-spikes) | Identify spikes (motion artifacts, intensity outliers) in 4D fMRI data.
[`trim`](#algorithms-trim) | Trim a Polars DataFrame/Series by replacing outlier values with NaNs.
[`winsorize`](#algorithms-winsorize) | Winsorize a Polars DataFrame/Series with the largest/lowest value not considered outlier.
[`zscore`](#algorithms-zscore) | Z-score every column of a Polars or pandas DataFrame/Series.



##### Methods

###### `find_spikes`

```python
find_spikes(data, global_spike_cutoff = 3, diff_spike_cutoff = 3, *, TR: float | None = None, sampling_freq: float | None = None)
```

Identify spikes (motion artifacts, intensity outliers) in 4D fMRI data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | BrainData or nibabel instance | *required*
`global_spike_cutoff` |  | (int, None) cutoff in std-deviations for spikes in the per-TR global signal. None to skip. | <code>3</code>
`diff_spike_cutoff` |  | (int, None) cutoff in std-deviations for spikes in the per-TR mean absolute frame-to-frame difference. None to skip. | <code>3</code>
`TR` | <code>[float](#float) \| None</code> | Repetition time in seconds. Sets the returned DesignMatrix's sampling_freq for downstream `.append(...)` / `.convolve()`. Pass exactly one of `TR` or `sampling_freq`. | <code>None</code>
`sampling_freq` | <code>[float](#float) \| None</code> | Sampling frequency in Hz (= 1/TR). See `TR`. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`DesignMatrix` |  | one indicator column per detected spike TR, named
 |  | ``.nl_global_spike{n}`` / ``.nl_diff_spike{n}`` in the reserved
 |  | namespace for generated columns (see `RESERVED_PREFIX`), with all
 |  | spike columns pre-marked as confounds. The two detectors run
 |  | independently, so a single bad volume is routinely caught by both;
 |  | those detections are bitwise-identical one-hot columns, and only one
 |  | is kept (the ``.nl_global_spike*`` name, a deterministic tie-break —
 |  | the column values are the same either way). Row position is the time
 |  | axis (no separate `TR` index column — that was a pandas-era
 |  | artifact). When `TR` / `sampling_freq` aren't provided the DM has
 |  | `sampling_freq=None`; you can still `.append()` it onto a DM that
 |  | does have one.

###### `trim`

```python
trim(data, cutoff = None)
```

Trim a Polars DataFrame/Series by replacing outlier values with NaNs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pl.DataFrame, pl.Series) data to trim | *required*
`cutoff` |  | (dict) a dictionary with keys {'std':[low,high]} or     {'quantile':[low,high]} | <code>None</code>

Returns:
    out: (pl.DataFrame, pl.Series) trimmed data (same type as input)

###### `winsorize`

```python
winsorize(data, cutoff = None, replace_with_cutoff = True)
```

Winsorize a Polars DataFrame/Series with the largest/lowest value not considered outlier.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pl.DataFrame, pl.Series) data to winsorize | *required*
`cutoff` |  | (dict) a dictionary with keys {'std':[low,high]} or     {'quantile':[low,high]} | <code>None</code>
`replace_with_cutoff` |  | (bool) If True, replace outliers with cutoff.                  If False, replaces outliers with closest                  existing values; (default: True) | <code>True</code>

Returns:
    out: (pl.DataFrame, pl.Series) winsorized data (same type as input)

###### `zscore`

```python
zscore(data)
```

Z-score every column of a Polars or pandas DataFrame/Series.

Accepts pandas inputs at the boundary for convenience and converts to
Polars internally. Always returns Polars output (DataFrame or Series,
matching the input shape).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | pl.DataFrame, pl.Series, pd.DataFrame, or pd.Series. | *required*

**Returns:**

Type | Description
---- | -----------
 | pl.DataFrame or pl.Series with each column z-scored using sample
 | standard deviation (ddof=1), matching the input shape.

#### `procrustes`

Data alignment — SRM, Procrustes, and state alignment.

**Methods:**

Name | Description
---- | -----------
[`align`](#algorithms-align) | Align subject data into a common response model.
[`align_states`](#algorithms-align-states) | Align state weight maps by minimizing pairwise distance between group states.
[`procrustes`](#algorithms-procrustes) | Perform a Procrustes similarity analysis on two data sets.
[`procrustes_distance`](#algorithms-procrustes-distance) | Test matrix similarity using Procrustes superposition.



##### Classes

##### Methods

###### `align`

```python
align(data, method = 'deterministic_srm', n_features = None, axis = 0, *args, **kwargs)
```

Align subject data into a common response model.

This function is a convenience wrapper around `HyperAlignment` and `SRM` classes.

Can be used to hyperalign source data to target data using
Hyperalignment from Dartmouth (i.e., procrustes transformation; see
nltools.algorithms.procrustes) or Shared Response Model from Princeton (see
nltools.algorithms.srm). (see nltools.data.BrainData.align for aligning
a single Brain object to another). Common Model is shared response
model or centered target data. Transformed data can be back projected to
original data using Tranformation matrix. Inputs must be a list of BrainData
instances or numpy arrays (observations by features).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (list) A list of BrainData objects | *required*
`method` |  | (str) alignment method to use ['probabilistic_srm','deterministic_srm','procrustes'] | <code>'deterministic_srm'</code>
`n_features` |  | (int) number of features to align to common space. If None then will select number of voxels | <code>None</code>
`axis` |  | (int) axis to align on | <code>0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (dict) a dictionary containing a list of transformed subject matrices, a list of transformation matrices, the shared response matrix, and the intersubject correlation of the shared responses

**Examples:**

- Hyperalign using procrustes transform:
    >>> out = align(data, method='procrustes')
- Align using shared response model:
    >>> out = align(data, method='probabilistic_srm', n_features=None)
- Project aligned data into original data:
    >>> original_data = [np.dot(t.data,tm.T) for t,tm in zip(out['transformed'], out['transformation_matrix'])]

###### `align_states`

```python
align_states(reference, target, *, metric = 'correlation', return_index = False, replace_zero_variance = False)
```

Align state weight maps by minimizing pairwise distance between group states.

This function uses the Hungarian algorithm for state alignment, which is
different from aligning multiple subjects' data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`reference` |  | (np.array) reference pattern x state matrix | *required*
`target` |  | (np.array) target pattern x state matrix to align to reference | *required*
`metric` |  | (str) distance metric to use | <code>'correlation'</code>
`return_index` |  | (bool) return index if True, return remapped data if False | <code>False</code>
`replace_zero_variance` |  | (bool) transform a vector with zero variance to random numbers from a uniform distribution.                     Useful for when using correlation as a distance metric to avoid NaNs. | <code>False</code>

Returns:
    If ``return_index=False`` (default): ``target[:, remapping]``, a single
    ndarray of the target's columns reordered to match the reference,
    oriented pattern x state (same shape as ``target``).
    If ``return_index=True``: the remapping index array (ndarray) that
    reorders the target's state columns.

###### `procrustes`

```python
procrustes(data1, data2)
```

Perform a Procrustes similarity analysis on two data sets.

For more comprehensive Procrustes-based alignment tasks, use
`HyperAlignment` and `align()` instead.

Each input matrix is a set of points or vectors (the rows of the matrix).
The dimension of the space is the number of columns of each matrix. Given
two identically sized matrices, procrustes standardizes both such that:
- $tr(AA^{T}) = 1$.
- Both sets of points are centered around the origin.
Procrustes then applies the optimal transform to the second
matrix (including scaling/dilation, rotations, and reflections) to minimize
$M^{2}=\sum(data1-data2)^{2}$, or the sum of the squares of the
pointwise differences between the two input datasets.
This function was not designed to handle datasets with different numbers of
datapoints (rows).  If two data sets have different dimensionality
(different number of columns), this function will add columns of zeros to
the smaller of the two.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` |  | Matrix whose n rows represent points in k (columns) space. `data1` is the reference data; after it is standardized, the data from `data2` will be transformed to fit the pattern in `data1` (must have >1 unique points). | *required*
`data2` |  | n rows of data in k space to be fit to `data1`. Must be the same shape `(numrows, numcols)` as `data1` (must have >1 unique points). | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`mtx1` |  | A standardized version of `data1`.
`mtx2` |  | The orientation of `data2` that best fits `data1`. Centered, but not necessarily $tr(AA^{T}) = 1$.
`disparity` |  | $M^{2}$ as defined above.
`R` |  | The `(N, N)` matrix solution of the orthogonal Procrustes problem. Minimizes the Frobenius norm of `dot(data1, R) - data2`, subject to `dot(R.T, R) == I`.
`scale` |  | Sum of the singular values of `dot(data1.T, data2)`.

###### `procrustes_distance`

```python
procrustes_distance(mat1, mat2, *, n_permute = 5000, tail = 2, n_jobs = -1, random_state = None)
```

Test matrix similarity using Procrustes superposition.

Matrices need to match in size on their first dimension only, as the smaller
matrix on the second dimension will be padded with zeros. After aligning two
matrices using the Procrustes transformation, use the computed disparity
between them (sum of squared error of elements) as a similarity metric.
Shuffle the rows of one of the matrices and recompute the disparity to perform
inference (Peres-Neto & Jackson, 2001).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mat1` | <code>[ndarray](#ndarray)</code> | 2d numpy array; must have same number of rows as mat2 | *required*
`mat2` | <code>[ndarray](#ndarray)</code> | 1d or 2d numpy array; must have same number of rows as mat1 | *required*
`n_permute` | <code>[int](#int)</code> | number of permutation iterations to perform | <code>5000</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | 2|'two' (two-tailed, default) or 1|'one' (one-tailed: similarity > chance) | <code>2</code>
`n_jobs` | <code>[int](#int)</code> | The number of CPUs to use to do permutation; default -1 (all) | <code>-1</code>
`random_state` | <code>int, np.random.RandomState, or None</code> | seed or generator for the permutation shuffling; default None | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | results with keys `similarity` (float in [0, 1]) and `p` (permuted p-value)

(algorithms-random)=
#### `random`

Shared random-state utilities for deterministic parallel execution.

<details class="key-features" open markdown="1">
<summary>Key features</summary>

- Deterministic parallelization: Pre-generates seeds for reproducible parallel execution
- Consistent RNG patterns: Matches stats.py patterns for backward compatibility
- Thread-safe design: Each parallel worker gets independent RandomState

</details>

<details class="usage" open markdown="1">
<summary>Usage</summary>

These utilities are used in bootstrap and permutation tests to ensure
deterministic behavior when using parallel processing.

Example:
    >>> from nltools.algorithms.random import generate_seeds
    >>> seeds = generate_seeds(100, random_state=42)
    >>> # Use seeds in parallel workers for deterministic results

</details>

**Methods:**

Name | Description
---- | -----------
[`generate_bootstrap_indices`](#algorithms-generate-bootstrap-indices) | Generate bootstrap indices deterministically for resampling.
[`generate_seeds`](#algorithms-generate-seeds) | Generate random seeds for deterministic parallelization.
[`generate_sign_flips`](#algorithms-generate-sign-flips) | Generate random sign-flip matrix for one-sample permutation tests.



##### Methods

(algorithms-generate-bootstrap-indices)=
###### `generate_bootstrap_indices`

```python
generate_bootstrap_indices(n_samples: int, n_bootstrap: int, random_state: int | None = None) -> np.ndarray
```

Generate bootstrap indices deterministically for resampling.

Uses the same pattern as permutation tests: pre-generate seeds for
reproducible parallelization.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_samples` | <code>[int](#int)</code> | Number of samples in original dataset. | *required*
`n_bootstrap` | <code>[int](#int)</code> | Number of bootstrap iterations. | *required*
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Bootstrap indices with shape (n_bootstrap, n_samples). Each row contains indices sampled with replacement from [0, n_samples).

**Examples:**

```pycon
>>> indices = generate_bootstrap_indices(100, 1000, random_state=42)
>>> indices.shape
(1000, 100)
>>> indices[0]  # First bootstrap sample indices
array([23, 45, 23, 67, ...])  # Some repeated (sampling with replacement)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Uses same seed generation pattern as permutation tests for consistency
- Each bootstrap iteration gets independent RandomState for reproducibility
- Sampling is with replacement (some indices may repeat)

</details>

(algorithms-generate-seeds)=
###### `generate_seeds`

```python
generate_seeds(n_permute: int, random_state: int | None = None) -> np.ndarray
```

Generate random seeds for deterministic parallelization.

Pre-generates unique seeds for each permutation/bootstrap iteration
to ensure deterministic behavior across parallel workers.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_permute` | <code>[int](#int)</code> | Number of permutations/bootstrap iterations | *required*
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Array of seeds with shape (n_permute,)

**Examples:**

```pycon
>>> seeds = generate_seeds(100, random_state=42)
>>> seeds.shape
(100,)
>>> isinstance(seeds[0], (int, np.integer))
True
```

(algorithms-generate-sign-flips)=
###### `generate_sign_flips`

```python
generate_sign_flips(n_permute: int, n_samples: int, random_state: int | None = None) -> np.ndarray
```

Generate random sign-flip matrix for one-sample permutation tests.

Creates a matrix of random +1/-1 values for sign-flipping permutation tests.
Each row represents one permutation, where each sample is randomly multiplied
by +1 or -1 to create the null distribution.

This implementation matches the RNG pattern from the original nltools.algorithms one_sample_permutation
for exact backward compatibility: each permutation gets an independent RandomState
derived from a unique seed.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_permute` | <code>[int](#int)</code> | Number of permutations to generate | *required*
`n_samples` | <code>[int](#int)</code> | Number of samples in the dataset | *required*
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Sign-flip matrix of shape (n_permute, n_samples) containing only +1 and -1 values

**Examples:**

```pycon
>>> sign_flips = generate_sign_flips(n_permute=100, n_samples=30, random_state=42)
>>> sign_flips.shape
(100, 30)
>>> np.all(np.isin(sign_flips, [-1, 1]))
True
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Each permutation uses independent RandomState for stats.py compatibility
- Values are uniformly sampled from {+1, -1} (matching stats.py order)
- Returns NumPy array (device transfer handled by caller)
- Memory cost: n_permute × n_samples × 1 byte (negligible for typical use)

</details>

(algorithms-regression)=
#### `regression`

Standalone OLS regression on numpy arrays.

Pedagogical helper used in tutorials and notebooks where callers want a
``(b, se, t, p, df, res)`` tuple from a design matrix ``X`` and response
``Y`` without constructing a `BrainData` or `Glm`. For
4D neuroimaging data use `BrainData.fit` with ``model='glm'``.

**Methods:**

Name | Description
---- | -----------
[`regress`](#algorithms-regress) | Fit an OLS regression of ``Y`` on ``X``.



##### Methods

###### `regress`

```python
regress(X, Y, *, method: str = 'ols', stats: str = 'full', tail: int | str = 2)
```

Fit an OLS regression of ``Y`` on ``X``.

Does not add an intercept — include one in ``X`` explicitly. If ``Y``
is 2D, a separate regression is fit to each column.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` |  | Design matrix, shape ``(n_samples, n_regressors)``. | *required*
`Y` |  | Response, shape ``(n_samples,)`` or ``(n_samples, n_targets)``. | *required*
`method` | <code>[str](#str)</code> | Only ``'ols'`` is supported in v0.6.0. The legacy ``'robust'`` and ``'arma'`` methods were dropped; use statsmodels or a dedicated package if you need them. | <code>'ols'</code>
`stats` | <code>[str](#str)</code> | ``'full'`` returns the 6-tuple below; ``'betas'`` returns just ``b``; ``'tstats'`` returns ``(b, t)``. | <code>'full'</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | 2|'two' (two-tailed, default) or 1|'one' (one-tailed: beta > 0; negate a regressor for the other direction). | <code>2</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` |  | ``(b, se, t, p, df, res)`` when ``stats='full'``:
 |  | - ``b``: coefficients
 |  | - ``se``: standard errors
 |  | - ``t``: t-statistics
 |  | - ``p``: p-values (per ``tail``)
 |  | - ``df``: residual degrees of freedom
 |  | - ``res``: residuals

(algorithms-ridge)=
#### `ridge`

Ridge regression algorithms and utilities.

This package contains ridge regression implementations with GPU acceleration.

Features:
- Cross-validation with per-target or global alpha selection
- Memory-efficient batching for large-scale problems
- GPU acceleration (10-100x speedup on large datasets)
- Banded ridge for multiple feature spaces

<details class="quick-start" open markdown="1">
<summary>Quick Start</summary>

>>> X = np.random.randn(100, 50)
>>> Y = np.random.randn(100, 10)
>>> result = solve_ridge_cv(X, Y, alphas=[0.1, 1.0, 10.0])

</details>

**Methods:**

Name | Description
---- | -----------
[`cross_val_predict_ridge`](#algorithms-cross-val-predict-ridge) | Held-out ridge predictions per CV fold under a (per-target) alpha.
[`generate_dirichlet_samples`](#algorithms-generate-dirichlet-samples) | Generate samples from a Dirichlet distribution.
[`ridge_cv`](#algorithms-ridge-cv) | Ridge regression with cross-validation for hyperparameter selection.
[`ridge_svd`](#algorithms-ridge-svd) | Solve ridge regression using Singular Value Decomposition.
[`solve_banded_ridge_cv`](#algorithms-solve-banded-ridge-cv) | Solve banded ridge regression with cross-validation using random search.
[`solve_ridge_cv`](#algorithms-solve-ridge-cv) | Solve ridge regression with cross-validation.



**Modules:**

Name | Description
---- | -----------
[`core`](#algorithms-core) | Ridge regression algorithms using SVD decomposition.
[`solvers`](#algorithms-solvers) | Ridge regression solvers with cross-validation.
[`utils`](#algorithms-utils) | Utility functions for ridge regression.

##### Methods

(algorithms-cross-val-predict-ridge)=
###### `cross_val_predict_ridge`

```python
cross_val_predict_ridge(X: np.ndarray, Y: np.ndarray, *, alphas: float | np.ndarray, cv: int | BaseCrossValidator = 5, fit_intercept: bool = False, n_targets_batch: int | None = None, n_alphas_batch: int | None = None, Y_in_cpu: bool = True, score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, parallel: str | None = 'cpu', max_gpu_memory_gb: float | None = None) -> dict[str, Any]
```

Held-out ridge predictions per CV fold under a (per-target) alpha.

For each fold, refits ridge with the supplied alpha (per-target or
scalar) on the training fold and predicts the held-out fold. Targets
sharing the same alpha share an SVD of the training fold via
`_refit_banded_ridge`, so the cost scales with the number of
*unique* alphas, not the number of targets.

Designed to be the BrainData CV layer's source of held-out predictions
when alpha selection has already been done by ``solve_ridge_cv``: pass
the selected per-voxel alphas back through here to get the fold-by-fold
predictions and per-fold R² needed for ``cv_results_``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Feature matrix of shape (n_samples, n_features). | *required*
`Y` | <code>[ndarray](#numpy.ndarray)</code> | Target data of shape (n_samples, n_targets). 1D ``Y`` is promoted to (n_samples, 1). | *required*
`alphas` | <code>[float](#float) \| [ndarray](#numpy.ndarray)</code> | Per-target alpha array of shape (n_targets,) or a scalar (broadcast to every target). | *required*
`cv` | <code>[int](#int) \| [BaseCrossValidator](#sklearn.model_selection.BaseCrossValidator)</code> | Cross-validation strategy. If int, uses KFold with that many splits (no shuffling). Generators (e.g. ``KFold(5).split(X)``) are rejected — pass the splitter object instead. | <code>5</code>
`fit_intercept` | <code>[bool](#bool)</code> | If True, center X and Y on the *training fold's* mean per fold (sklearn convention) and add the intercept back so predictions live on the original Y scale. | <code>False</code>
`n_targets_batch` | <code>[int](#int) \| None</code> | Batch size for targets during refit (for memory efficiency). If None, processes all targets at once. | <code>None</code>
`n_alphas_batch` | <code>[int](#int) \| None</code> | Batch size for alphas. If None, processes all unique alphas at once. | <code>None</code>
`Y_in_cpu` | <code>[bool](#bool)</code> | If True, keep Y on CPU and transfer batches to backend device as needed (recommended for large neuroimaging Y). | <code>True</code>
`score_func` | <code>[Callable](#collections.abc.Callable)[[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)] \| None</code> | Per-fold scoring function ``(y_true, y_pred) -> per-target scores``. If None, uses R² in NumPy on CPU (cheap at one fold's size and decoupled from backend ops to avoid stray transfers). | <code>None</code>
`parallel` | <code>[str](#str) \| None</code> | Backend to use: "cpu", "gpu", or None. | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>[float](#float) \| None</code> | GPU memory budget in GB (only used if parallel="gpu"). | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with keys: - 'predictions': (n_samples, n_targets) held-out per-target   predictions on the original Y scale (CPU numpy). - 'folds': (n_samples,) int fold index per row (CPU numpy). - 'scores': (n_splits, n_targets) per-fold R² (or   ``score_func``) at the supplied alpha (CPU numpy). - 'backend': Backend used (for transparency).

(algorithms-generate-dirichlet-samples)=
###### `generate_dirichlet_samples`

```python
generate_dirichlet_samples(n_samples: int, n_kernels: int, concentration: float | list[float] = [0.1, 1.0], random_state: int | None = None) -> np.ndarray
```

Generate samples from a Dirichlet distribution.

This function generates random samples from a Dirichlet distribution,
which is used for sampling feature space weights (gamma) in banded ridge
regression random search.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_samples` | <code>[int](#int)</code> | Number of samples to generate. | *required*
`n_kernels` | <code>[int](#int)</code> | Number of dimensions (feature spaces) of the distribution. | *required*
`concentration` | <code>[float](#float) \| [list](#list)[[float](#float)]</code> | Concentration parameters of the Dirichlet distribution. - A value of 1 corresponds to uniform sampling over the simplex. - A value of infinity corresponds to equal weights. - If a list, samples cycle through the list. Defaults to [0.1, 1.0]. | <code>[0.1, 1.0]</code>
`random_state` | <code>[int](#int) \| None</code> | Random generator seed. Use an int for deterministic samples. Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | np.ndarray: Dirichlet samples of shape (n_samples, n_kernels). Each row sums to 1 (lies on simplex).

**Examples:**

```pycon
>>> # Generate 10 samples for 3 feature spaces
>>> gammas = generate_dirichlet_samples(10, 3, concentration=[0.1, 1.0])
>>> gammas.shape
(10, 3)
>>> # Each row sums to 1
>>> np.allclose(gammas.sum(axis=1), 1.0)
True
```

###### `ridge_cv`

```python
ridge_cv(X: np.ndarray, y: np.ndarray, *, alphas: np.ndarray | None = None, cv: int | BaseCrossValidator = 5, fit_intercept: bool = False, parallel: str | None = 'cpu', max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> dict
```

Ridge regression with cross-validation for hyperparameter selection.

Performs k-fold cross-validation to select the best alpha parameter,
then fits a final model on all data using the selected alpha.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Training data features with shape (n_samples, n_features) | *required*
`y` | <code>[ndarray](#numpy.ndarray)</code> | Target values with shape (n_samples,) or (n_samples, n_targets) | *required*
`alphas` | <code>[ndarray](#numpy.ndarray)</code> | Array of alpha values to try. If None, uses default range: np.logspace(-2, 4, 20) = [0.01, 0.015, ..., 10000] | <code>None</code>
`cv` | <code>int or sklearn CV splitter</code> | Number of folds (int) or an sklearn cross-validator (anything with ``.split(X)`` and ``.get_n_splits()``, e.g. ``KFold(5, shuffle=True)`` or ``GroupKFold(8)``). Splitters are honored for the actual fold iteration, so leave-one-run-out and shuffled-K-fold give different results from contiguous K-fold. Defaults to 5. | <code>5</code>
`fit_intercept` | <code>[bool](#bool)</code> | If True, center X and y on the training mean before fitting and recover the intercept after. The returned ``coef`` is on the centered scale; the recovered intercept is returned under the ``intercept`` key. Defaults to False. | <code>False</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU-only using NumPy (default) - "gpu": GPU acceleration via PyTorch. Requires torch installed   (raises ImportError otherwise); degrades to torch-CPU only when no   GPU device is present. Use "auto" for torch-optional CPU fallback. Defaults to "cpu". | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel='gpu'). Defaults to 4.0. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed (not currently used, kept for consistency). Defaults to None. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary containing:<br>- 'alpha' (float): Best alpha value selected by CV - 'coef' (np.ndarray): Coefficients using best alpha on full dataset - 'cv_scores' (np.ndarray): Cross-validation R**2 scores for each fold, alpha, and target     with shape (n_folds, n_alphas, n_targets) - 'backend' (str): Backend used for computation

**Examples:**

```pycon
>>> X = np.random.randn(100, 50)
>>> y = np.random.randn(100)
>>> result = ridge_cv(X, y, cv=3)
>>> result['alpha']  # Best alpha selected
1.0
>>> result['coef'].shape
(50,)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Uses R**2 (coefficient of determination) as the scoring metric
- For multi-target regression, selects alpha that maximizes mean R**2 across targets
- parallel='gpu' requires torch installed; with torch present but no GPU device it
  runs on torch-CPU. It does not fall back to NumPy when torch is absent — use
  parallel='auto' for that.

</details>

###### `ridge_svd`

```python
ridge_svd(X: np.ndarray, y: np.ndarray, *, alpha: float = 1.0, parallel: str | None = None, max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> np.ndarray
```

Solve ridge regression using Singular Value Decomposition.

This function implements ridge regression using SVD, which provides
numerical stability and efficiency for high-dimensional problems.
The implementation is inspired by the himalaya library.

<details class="algorithm" open markdown="1">
<summary>Algorithm</summary>

The ridge regression solution is:
    beta = (X.T @ X + alpha*I)^(-1) @ X.T @ y

Using SVD of X = U @ diag(s) @ V.T, this becomes:
    beta = V @ diag(s / (s**2 + alpha)) @ U.T @ y

This formulation avoids explicit matrix inversion and is numerically stable.
The shrinkage factor s / (s**2 + alpha) regularizes small singular values.

</details>

<details class="performance" open markdown="1">
<summary>Performance</summary>

- Time complexity: O(n_samples × n_features × min(n_samples, n_features))
- Space complexity: O(n_samples × n_features)
- GPU acceleration: ~10-100× speedup for large problems (n_features > 10K)
- See `solve_ridge_cv()` for cross-validation with GPU support

</details>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Training data features with shape (n_samples, n_features) | *required*
`y` | <code>[ndarray](#numpy.ndarray)</code> | Target values with shape (n_samples,) or (n_samples, n_targets). Can be 1D for single-target or 2D for multi-target | *required*
`alpha` | <code>[float](#float)</code> | Regularization strength. Must be positive. Higher values increase regularization (shrink coefficients toward zero). Defaults to 1.0. | <code>1.0</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU-only using NumPy (default) - "gpu": GPU acceleration via PyTorch. Requires torch installed   (raises ImportError otherwise); degrades to torch-CPU only when no   GPU device is present. Use "auto" for torch-optional CPU fallback. Defaults to None. | <code>None</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel='gpu'). Defaults to 4.0. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed (not currently used, kept for consistency). Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | np.ndarray: Ridge regression coefficients - shape (n_features,) for single-target regression - shape (n_features, n_targets) for multi-target regression

**Examples:**

```pycon
>>> X = np.random.randn(100, 50)
>>> y = np.random.randn(100)
>>> beta = ridge_svd(X, y, alpha=1.0)
>>> beta.shape
(50,)
```

```pycon
>>> # Multi-target regression
>>> Y = np.random.randn(100, 5)
>>> beta = ridge_svd(X, Y, alpha=1.0)
>>> beta.shape
(50, 5)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Time complexity: O(n_samples * n_features * min(n_samples, n_features))
- Space complexity: O(n_samples * n_features)
- For alpha→0, this reduces to ordinary least squares (OLS). Use alpha=1e-6
  for OLS in practice (more numerically stable than alpha=0)
- Supports both CPU (NumPy) and GPU (PyTorch) backends
- See `nltools.algorithms.ridge.solvers.solve_ridge_cv()` for cross-validation
- See `nltools.algorithms.ridge.utils._decompose_ridge()` for generator pattern

</details>

(algorithms-solve-banded-ridge-cv)=
###### `solve_banded_ridge_cv`

```python
solve_banded_ridge_cv(Xs: list[np.ndarray], Y: np.ndarray, *, n_iter: int | np.integer | np.ndarray = 100, concentration: float | list[float] = [0.1, 1.0], alphas: float | np.ndarray | list[float] = [0.1, 1.0, 10.0], cv: int | BaseCrossValidator = 5, local_alpha: bool = True, n_targets_batch: int | None = None, n_targets_batch_refit: int | None = None, n_alphas_batch: int | None = None, Y_in_cpu: bool = True, score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, fit_intercept: bool = False, progress_bar: bool = False, conservative: bool = False, jitter_alphas: bool = False, return_weights: bool = True, diagonalize_method: str = 'svd', warn: bool = True, parallel: str | None = 'cpu', max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> dict[str, Any]
```

Solve banded ridge regression with cross-validation using random search.

This function implements true banded/group ridge regression (as in Himalaya).
It searches over feature space weights (gamma) sampled from a Dirichlet
distribution, combined with alpha grid search.

Banded ridge (also called group ridge) applies different scaling weights
per feature space: Z_i = sqrt(gamma_i) * X_i, then solves standard ridge
regression on the scaled concatenated features. This allows optimizing
the relative importance of different feature spaces.

The feature spaces are scaled by sqrt(gamma) for each gamma sample, then
standard ridge regression is applied with alpha grid search.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`Xs` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | Feature matrices for different feature spaces. Each array has shape (n_samples, n_features_i). All must have the same n_samples. | *required*
`Y` | <code>[ndarray](#numpy.ndarray)</code> | Target data of shape (n_samples, n_targets). | *required*
`n_iter` | <code>[int](#int) \| [integer](#numpy.integer) \| [ndarray](#numpy.ndarray)</code> | Number of feature-space weights combination to search, or array of shape (n_iter, n_spaces). If an array is given, the solver uses it as the list of weights to try, instead of sampling from a Dirichlet distribution. Defaults to 100. | <code>100</code>
`concentration` | <code>[float](#float) \| [list](#list)[[float](#float)]</code> | Concentration parameters of the Dirichlet distribution. - A value of 1 corresponds to uniform sampling over the simplex. - A value of infinity corresponds to equal weights. - If a list, iteratively cycle through the list. Not used if n_iter is an array. Defaults to [0.1, 1.0]. | <code>[0.1, 1.0]</code>
`alphas` | <code>[float](#float) \| [ndarray](#numpy.ndarray) \| [list](#list)[[float](#float)]</code> | Range of ridge regularization parameters to try. Can be float or array of shape (n_alphas,). Defaults to [0.1, 1.0, 10.0]. | <code>[0.1, 1.0, 10.0]</code>
`cv` | <code>[int](#int) \| [BaseCrossValidator](#sklearn.model_selection.BaseCrossValidator)</code> | Cross-validation strategy. If int, uses KFold with that many splits. Defaults to 5. | <code>5</code>
`local_alpha` | <code>[bool](#bool)</code> | If True, select best alpha independently for each target. If False, select single best alpha for all targets. Defaults to True. | <code>True</code>
`n_targets_batch` | <code>[int](#int) \| None</code> | Batch size for targets during CV (for memory efficiency). If None, processes all targets at once. Defaults to None. | <code>None</code>
`n_targets_batch_refit` | <code>[int](#int) \| None</code> | Batch size for targets during refit. If None, uses n_targets_batch value. Defaults to None. | <code>None</code>
`n_alphas_batch` | <code>[int](#int) \| None</code> | Batch size for alphas (for memory efficiency). If None, processes all alphas at once. Defaults to None. | <code>None</code>
`Y_in_cpu` | <code>[bool](#bool)</code> | If True, keep Y on CPU and transfer batches to GPU as needed. This prevents OOM when Y is large (e.g., 300k voxels). Defaults to True (recommended for neuroimaging). | <code>True</code>
`score_func` | <code>[Callable](#collections.abc.Callable)[[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)] \| None</code> | Scoring function (y_true, y_pred) -> scores. If None, uses R² score. Defaults to None. | <code>None</code>
`fit_intercept` | <code>[bool](#bool)</code> | Whether to fit an intercept. If False, X and Y should be centered. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Whether to display progress bar (requires tqdm). Defaults to False. | <code>False</code>
`conservative` | <code>[bool](#bool)</code> | If True, select largest alpha within 1 std of best score. Defaults to False. | <code>False</code>
`jitter_alphas` | <code>[bool](#bool)</code> | If True, alphas range is slightly jittered for each gamma. Defaults to False. | <code>False</code>
`return_weights` | <code>[bool](#bool)</code> | Whether to refit on the entire dataset and return the weights. Defaults to True. | <code>True</code>
`diagonalize_method` | <code>[str](#str)</code> | Method used to diagonalize the features. Currently only "svd" is supported. Defaults to "svd". | <code>'svd'</code>
`warn` | <code>[bool](#bool)</code> | If True, warn if the number of samples is smaller than the number of features. Defaults to True. | <code>True</code>
`parallel` | <code>[str](#str) \| None</code> | Backend to use: "cpu", "gpu", or None. Defaults to "cpu". | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>[float](#float) \| None</code> | GPU memory budget in GB (only used if parallel="gpu"). Defaults to 4.0. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random generator seed. Use an int for deterministic search. Defaults to None. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with keys: - 'deltas': Best log feature-space weights for each target,     shape (n_spaces, n_targets). deltas = log(gamma / alpha), where     gamma are the feature space weights. - 'cv_scores': Cross-validation scores per iteration, averaged over splits,     for the best alpha, shape (n_iter, n_targets). Always returned on CPU     (numpy array). - 'coefs': Ridge coefficients refit on entire dataset using best hyperparameters,     shape (n_features_total, n_targets), or None if return_weights=False.     Always returned on CPU (numpy array). - 'intercept': Intercept of shape (n_targets,), or None if     fit_intercept=False or return_weights=False. - 'backend': Backend used (for transparency).

**Examples:**

```pycon
>>> # Multiple feature spaces (banded ridge with random search)
>>> X1 = np.random.randn(100, 30)  # First feature space
>>> X2 = np.random.randn(100, 20)  # Second feature space
>>> Y = np.random.randn(100, 10)
>>> result = solve_banded_ridge_cv(
...     [X1, X2], Y, n_iter=50, alphas=[0.1, 1.0, 10.0]
... )
>>> deltas = result['deltas']
>>> coefs = result['coefs']
>>> scores = result['cv_scores']
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

This implements true banded/group ridge regression (as in Himalaya's
solve_group_ridge_random_search) with:
- Dirichlet sampling for feature space weights (gamma)
- Scaling each feature space by sqrt(gamma) for each gamma sample
- Cross-validation with alpha grid search
- Per-target selection of best gamma and alpha combination

This is the correct implementation of banded/group ridge regression, which
allows different scaling weights per feature space. For single feature space
ridge regression, use solve_ridge_cv instead.

Algorithm details:

- Random search: Samples gamma weights from Dirichlet distribution
- Banded ridge: Scales each feature space by sqrt(gamma_i), then solves standard ridge
- Cross-validation: Evaluates each (gamma, alpha) combination via k-fold CV
- Best selection: Chooses (gamma, alpha) that maximizes CV score per target

Memory efficiency strategies (Principle 2: automatic memory efficiency):

- Generator pattern for alpha batching (via _decompose_ridge): Processes alphas
  in batches to avoid storing all resolution matrices simultaneously
- Target batching (n_targets_batch): Processes targets in chunks to fit GPU memory
- Y_in_cpu strategy: Keeps large Y on CPU, transfers only batches needed
  for computation
- Immediate cleanup with del statements: Explicitly frees memory after each batch

Performance:

- Time complexity: O(n_iter × n_splits × (n_alphas_batch × n_features^2 + n_targets_batch × n_samples))
- Memory complexity: O(n_features × n_targets_batch) per batch
- GPU acceleration: ~10-100× speedup for large problems (n_features > 10K)

See ``nltools.algorithms.ridge.utils._decompose_ridge()`` for generator pattern details.
See ``docs/development/ridge-internals.md`` for detailed algorithm explanation.

</details>

(algorithms-solve-ridge-cv)=
###### `solve_ridge_cv`

```python
solve_ridge_cv(X: np.ndarray, Y: np.ndarray, *, alphas: float | np.ndarray | list[float] = [0.1, 1.0, 10.0], cv: int | BaseCrossValidator = 5, local_alpha: bool = True, n_targets_batch: int | None = None, n_targets_batch_refit: int | None = None, n_alphas_batch: int | None = None, Y_in_cpu: bool = True, score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, fit_intercept: bool = False, progress_bar: bool = False, conservative: bool = False, parallel: str | None = 'cpu', max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> dict[str, Any]
```

Solve ridge regression with cross-validation.

This function solves ridge regression for a single feature space with
cross-validation for hyperparameter selection.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Feature matrix of shape (n_samples, n_features). | *required*
`Y` | <code>[ndarray](#numpy.ndarray)</code> | Target data of shape (n_samples, n_targets). | *required*
`alphas` | <code>[float](#float) \| [ndarray](#numpy.ndarray) \| [list](#list)[[float](#float)]</code> | Ridge regularization parameters to try. Defaults to [0.1, 1.0, 10.0]. | <code>[0.1, 1.0, 10.0]</code>
`cv` | <code>[int](#int) \| [BaseCrossValidator](#sklearn.model_selection.BaseCrossValidator)</code> | Cross-validation strategy. If int, uses KFold with that many splits. Defaults to 5. | <code>5</code>
`local_alpha` | <code>[bool](#bool)</code> | If True, select best alpha independently for each target. If False, select single best alpha for all targets. Defaults to True. | <code>True</code>
`n_targets_batch` | <code>[int](#int) \| None</code> | Batch size for targets during CV (for memory efficiency). If None, processes all targets at once. Defaults to None. | <code>None</code>
`n_targets_batch_refit` | <code>[int](#int) \| None</code> | Batch size for targets during refit. If None, uses n_targets_batch value. Defaults to None. | <code>None</code>
`n_alphas_batch` | <code>[int](#int) \| None</code> | Batch size for alphas (for memory efficiency). If None, processes all alphas at once. Defaults to None. | <code>None</code>
`Y_in_cpu` | <code>[bool](#bool)</code> | If True, keep Y on CPU and transfer batches to GPU as needed. This prevents OOM when Y is large (e.g., 300k voxels). Defaults to True (recommended for neuroimaging). | <code>True</code>
`score_func` | <code>[Callable](#collections.abc.Callable)[[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)] \| None</code> | Scoring function (y_true, y_pred) -> scores. If None, uses R² score. Defaults to None. | <code>None</code>
`fit_intercept` | <code>[bool](#bool)</code> | Whether to fit an intercept. If False, X and Y should be centered. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Whether to display progress bar (requires tqdm). Defaults to False. | <code>False</code>
`conservative` | <code>[bool](#bool)</code> | If True, select largest alpha within 1 std of best score. Defaults to False. | <code>False</code>
`parallel` | <code>[str](#str) \| None</code> | Backend to use: "cpu", "gpu", or None. Defaults to "cpu". | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>[float](#float) \| None</code> | GPU memory budget in GB (only used if parallel="gpu"). Defaults to 4.0. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random generator seed. Use an int for deterministic search. Defaults to None. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with keys: - 'best_alphas': Selected best alpha for each target (or same alpha repeated     if local_alpha=False), shape (n_targets,). - 'coefs': Ridge coefficients refit on entire dataset using best alphas,     shape (n_features, n_targets). Always returned on CPU (numpy array). - 'cv_scores': Cross-validation scores for best alphas, shape (n_splits, n_alphas, n_targets).     Always returned on CPU (numpy array). - 'intercept': Per-target intercept of shape (n_targets,). Only present     when ``fit_intercept=True``. - 'backend': Backend used (for transparency).

**Examples:**

```pycon
>>> X = np.random.randn(100, 50)
>>> Y = np.random.randn(100, 10)
>>> result = solve_ridge_cv(X, Y, alphas=[0.1, 1.0, 10.0])
>>> alphas = result['best_alphas']
>>> coefs = result['coefs']
>>> scores = result['cv_scores']
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

This is the efficient implementation for single feature space ridge regression
with cross-validation. For multiple feature spaces (banded/group ridge),
use solve_banded_ridge_cv instead.

Algorithm details:

- Cross-validation: k-fold CV evaluates each alpha value
- Alpha selection: Chooses best alpha per target (or globally if local_alpha=False)
- Refit: Fits final model on full dataset using best alpha(s)

Memory efficiency strategies (Principle 2: automatic memory efficiency):

- Generator pattern for alpha batching (via _decompose_ridge): Processes alphas
  in batches to avoid storing all resolution matrices simultaneously
- Target batching (n_targets_batch): Processes targets in chunks to fit GPU memory
- Y_in_cpu strategy: Keeps large Y on CPU, transfers only batches needed
  for computation
- Immediate cleanup with del statements: Explicitly frees memory after each batch

Performance:

- Time complexity: O(n_splits × (n_alphas_batch × n_features^2 + n_targets_batch × n_samples))
- Memory complexity: O(n_features × n_targets_batch) per batch
- GPU acceleration: ~10-100× speedup for large problems (n_features > 10K)

See ``nltools.algorithms.ridge.utils._decompose_ridge()`` for generator pattern details.
See ``docs/development/ridge-internals.md`` for detailed algorithm explanation.

</details>



##### Modules

(algorithms-core)=
###### `core`

Ridge regression algorithms using SVD decomposition.

This module implements ridge regression using Singular Value Decomposition (SVD),
which provides numerical stability and efficiency for high-dimensional problems.

<details class="algorithm-approach" open markdown="1">
<summary>Algorithm approach</summary>

Why SVD vs direct inversion:
    - Direct inversion: beta = (X.T @ X + alpha*I)^(-1) @ X.T @ y
    - SVD approach: X = U @ diag(s) @ V.T, then beta = V @ diag(s / (s**2 + alpha)) @ U.T @ y
    - Benefits: Avoids explicit matrix inversion (numerically stable), efficient for rank-deficient X
    - Performance: O(n_samples × n_features × min(n_samples, n_features)) for SVD

</details>

<details class="backend-choice-trade-offs" open markdown="1">
<summary>Backend choice trade-offs</summary>

- NumPy (CPU): Default, reliable, works everywhere
- PyTorch CPU: Similar performance to NumPy, useful for consistent API
- PyTorch GPU: ~10-100× speedup for large problems (n_features > 10K), requires GPU

</details>

<details class="cross-references" open markdown="1">
<summary>Cross-references</summary>

- See `nltools.algorithms.ridge.solvers.solve_ridge_cv()` for GPU-accelerated cross-validation
- See `nltools.algorithms.ridge.utils._decompose_ridge()` for generator-based batching pattern
- See `docs/development/ridge-internals.md` for detailed algorithm explanation

</details>

Inspired by the himalaya library's efficient SVD-based ridge regression approach.
himalaya is licensed under BSD-3-Clause: https://github.com/gallantlab/himalaya

<details class="references" open markdown="1">
<summary>References</summary>

- Huth, A. G., et al. (2016). "Natural speech reveals the semantic maps that tile
  human cerebral cortex." Nature, 532(7600), 453-458.
- himalaya documentation: https://gallantlab.github.io/himalaya/

</details>

**Methods:**

Name | Description
---- | -----------
[`ridge_cv`](#algorithms-ridge-cv) | Ridge regression with cross-validation for hyperparameter selection.
[`ridge_svd`](#algorithms-ridge-svd) | Solve ridge regression using Singular Value Decomposition.



####### Functions##

###### `ridge_cv`

```python
ridge_cv(X: np.ndarray, y: np.ndarray, *, alphas: np.ndarray | None = None, cv: int | BaseCrossValidator = 5, fit_intercept: bool = False, parallel: str | None = 'cpu', max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> dict
```

Ridge regression with cross-validation for hyperparameter selection.

Performs k-fold cross-validation to select the best alpha parameter,
then fits a final model on all data using the selected alpha.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Training data features with shape (n_samples, n_features) | *required*
`y` | <code>[ndarray](#numpy.ndarray)</code> | Target values with shape (n_samples,) or (n_samples, n_targets) | *required*
`alphas` | <code>[ndarray](#numpy.ndarray)</code> | Array of alpha values to try. If None, uses default range: np.logspace(-2, 4, 20) = [0.01, 0.015, ..., 10000] | <code>None</code>
`cv` | <code>int or sklearn CV splitter</code> | Number of folds (int) or an sklearn cross-validator (anything with ``.split(X)`` and ``.get_n_splits()``, e.g. ``KFold(5, shuffle=True)`` or ``GroupKFold(8)``). Splitters are honored for the actual fold iteration, so leave-one-run-out and shuffled-K-fold give different results from contiguous K-fold. Defaults to 5. | <code>5</code>
`fit_intercept` | <code>[bool](#bool)</code> | If True, center X and y on the training mean before fitting and recover the intercept after. The returned ``coef`` is on the centered scale; the recovered intercept is returned under the ``intercept`` key. Defaults to False. | <code>False</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU-only using NumPy (default) - "gpu": GPU acceleration via PyTorch. Requires torch installed   (raises ImportError otherwise); degrades to torch-CPU only when no   GPU device is present. Use "auto" for torch-optional CPU fallback. Defaults to "cpu". | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel='gpu'). Defaults to 4.0. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed (not currently used, kept for consistency). Defaults to None. | <code>None</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Training data features with shape (n_samples, n_features) | *required*
`y` | <code>[ndarray](#numpy.ndarray)</code> | Target values with shape (n_samples,) or (n_samples, n_targets). Can be 1D for single-target or 2D for multi-target | *required*
`alpha` | <code>[float](#float)</code> | Regularization strength. Must be positive. Higher values increase regularization (shrink coefficients toward zero). Defaults to 1.0. | <code>1.0</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU-only using NumPy (default) - "gpu": GPU acceleration via PyTorch. Requires torch installed   (raises ImportError otherwise); degrades to torch-CPU only when no   GPU device is present. Use "auto" for torch-optional CPU fallback. Defaults to None. | <code>None</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel='gpu'). Defaults to 4.0. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed (not currently used, kept for consistency). Defaults to None. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary containing:<br>- 'alpha' (float): Best alpha value selected by CV - 'coef' (np.ndarray): Coefficients using best alpha on full dataset - 'cv_scores' (np.ndarray): Cross-validation R**2 scores for each fold, alpha, and target     with shape (n_folds, n_alphas, n_targets) - 'backend' (str): Backend used for computation

**Examples:**

```pycon
>>> X = np.random.randn(100, 50)
>>> y = np.random.randn(100)
>>> result = ridge_cv(X, y, cv=3)
>>> result['alpha']  # Best alpha selected
1.0
>>> result['coef'].shape
(50,)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Uses R**2 (coefficient of determination) as the scoring metric
- For multi-target regression, selects alpha that maximizes mean R**2 across targets
- parallel='gpu' requires torch installed; with torch present but no GPU device it
  runs on torch-CPU. It does not fall back to NumPy when torch is absent — use
  parallel='auto' for that.

</details>

######## `ridge_svd`

```python
ridge_svd(X: np.ndarray, y: np.ndarray, *, alpha: float = 1.0, parallel: str | None = None, max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> np.ndarray
```

Solve ridge regression using Singular Value Decomposition.

This function implements ridge regression using SVD, which provides
numerical stability and efficiency for high-dimensional problems.
The implementation is inspired by the himalaya library.

<details class="algorithm" open markdown="1">
<summary>Algorithm</summary>

The ridge regression solution is:
    beta = (X.T @ X + alpha*I)^(-1) @ X.T @ y

Using SVD of X = U @ diag(s) @ V.T, this becomes:
    beta = V @ diag(s / (s**2 + alpha)) @ U.T @ y

This formulation avoids explicit matrix inversion and is numerically stable.
The shrinkage factor s / (s**2 + alpha) regularizes small singular values.

</details>

<details class="performance" open markdown="1">
<summary>Performance</summary>

- Time complexity: O(n_samples × n_features × min(n_samples, n_features))
- Space complexity: O(n_samples × n_features)
- GPU acceleration: ~10-100× speedup for large problems (n_features > 10K)
- See `solve_ridge_cv()` for cross-validation with GPU support

</details>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | np.ndarray: Ridge regression coefficients - shape (n_features,) for single-target regression - shape (n_features, n_targets) for multi-target regression

**Examples:**

```pycon
>>> X = np.random.randn(100, 50)
>>> y = np.random.randn(100)
>>> beta = ridge_svd(X, y, alpha=1.0)
>>> beta.shape
(50,)
```

```pycon
>>> # Multi-target regression
>>> Y = np.random.randn(100, 5)
>>> beta = ridge_svd(X, Y, alpha=1.0)
>>> beta.shape
(50, 5)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Time complexity: O(n_samples * n_features * min(n_samples, n_features))
- Space complexity: O(n_samples * n_features)
- For alpha→0, this reduces to ordinary least squares (OLS). Use alpha=1e-6
  for OLS in practice (more numerically stable than alpha=0)
- Supports both CPU (NumPy) and GPU (PyTorch) backends
- See `nltools.algorithms.ridge.solvers.solve_ridge_cv()` for cross-validation
- See `nltools.algorithms.ridge.utils._decompose_ridge()` for generator pattern

</details>

(algorithms-solvers)=
###### `solvers`

Ridge regression solvers with cross-validation.

Implements banded ridge regression (multiple feature spaces) and regular ridge
regression (single feature space) with cross-validation for hyperparameter selection.

Follows himalaya's implementation patterns:
- Generator-based batching for memory efficiency
- Y_in_cpu strategy for large target datasets
- Backend abstraction (NumPy, PyTorch, PyTorch+CUDA)
- Per-target or global alpha selection
- Random search over feature space weights (Dirichlet sampling)

**Methods:**

Name | Description
---- | -----------
[`cross_val_predict_ridge`](#algorithms-cross-val-predict-ridge) | Held-out ridge predictions per CV fold under a (per-target) alpha.
[`solve_banded_ridge_cv`](#algorithms-solve-banded-ridge-cv) | Solve banded ridge regression with cross-validation using random search.
[`solve_ridge_cv`](#algorithms-solve-ridge-cv) | Solve ridge regression with cross-validation.



####### Functions##

###### `cross_val_predict_ridge`

```python
cross_val_predict_ridge(X: np.ndarray, Y: np.ndarray, *, alphas: float | np.ndarray, cv: int | BaseCrossValidator = 5, fit_intercept: bool = False, n_targets_batch: int | None = None, n_alphas_batch: int | None = None, Y_in_cpu: bool = True, score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, parallel: str | None = 'cpu', max_gpu_memory_gb: float | None = None) -> dict[str, Any]
```

Held-out ridge predictions per CV fold under a (per-target) alpha.

For each fold, refits ridge with the supplied alpha (per-target or
scalar) on the training fold and predicts the held-out fold. Targets
sharing the same alpha share an SVD of the training fold via
`_refit_banded_ridge`, so the cost scales with the number of
*unique* alphas, not the number of targets.

Designed to be the BrainData CV layer's source of held-out predictions
when alpha selection has already been done by ``solve_ridge_cv``: pass
the selected per-voxel alphas back through here to get the fold-by-fold
predictions and per-fold R² needed for ``cv_results_``.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Feature matrix of shape (n_samples, n_features). | *required*
`Y` | <code>[ndarray](#numpy.ndarray)</code> | Target data of shape (n_samples, n_targets). 1D ``Y`` is promoted to (n_samples, 1). | *required*
`alphas` | <code>[float](#float) \| [ndarray](#numpy.ndarray)</code> | Per-target alpha array of shape (n_targets,) or a scalar (broadcast to every target). | *required*
`cv` | <code>[int](#int) \| [BaseCrossValidator](#sklearn.model_selection.BaseCrossValidator)</code> | Cross-validation strategy. If int, uses KFold with that many splits (no shuffling). Generators (e.g. ``KFold(5).split(X)``) are rejected — pass the splitter object instead. | <code>5</code>
`fit_intercept` | <code>[bool](#bool)</code> | If True, center X and Y on the *training fold's* mean per fold (sklearn convention) and add the intercept back so predictions live on the original Y scale. | <code>False</code>
`n_targets_batch` | <code>[int](#int) \| None</code> | Batch size for targets during refit (for memory efficiency). If None, processes all targets at once. | <code>None</code>
`n_alphas_batch` | <code>[int](#int) \| None</code> | Batch size for alphas. If None, processes all unique alphas at once. | <code>None</code>
`Y_in_cpu` | <code>[bool](#bool)</code> | If True, keep Y on CPU and transfer batches to backend device as needed (recommended for large neuroimaging Y). | <code>True</code>
`score_func` | <code>[Callable](#collections.abc.Callable)[[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)] \| None</code> | Per-fold scoring function ``(y_true, y_pred) -> per-target scores``. If None, uses R² in NumPy on CPU (cheap at one fold's size and decoupled from backend ops to avoid stray transfers). | <code>None</code>
`parallel` | <code>[str](#str) \| None</code> | Backend to use: "cpu", "gpu", or None. | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>[float](#float) \| None</code> | GPU memory budget in GB (only used if parallel="gpu"). | <code>None</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`Xs` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | Feature matrices for different feature spaces. Each array has shape (n_samples, n_features_i). All must have the same n_samples. | *required*
`Y` | <code>[ndarray](#numpy.ndarray)</code> | Target data of shape (n_samples, n_targets). | *required*
`n_iter` | <code>[int](#int) \| [integer](#numpy.integer) \| [ndarray](#numpy.ndarray)</code> | Number of feature-space weights combination to search, or array of shape (n_iter, n_spaces). If an array is given, the solver uses it as the list of weights to try, instead of sampling from a Dirichlet distribution. Defaults to 100. | <code>100</code>
`concentration` | <code>[float](#float) \| [list](#list)[[float](#float)]</code> | Concentration parameters of the Dirichlet distribution. - A value of 1 corresponds to uniform sampling over the simplex. - A value of infinity corresponds to equal weights. - If a list, iteratively cycle through the list. Not used if n_iter is an array. Defaults to [0.1, 1.0]. | <code>[0.1, 1.0]</code>
`alphas` | <code>[float](#float) \| [ndarray](#numpy.ndarray) \| [list](#list)[[float](#float)]</code> | Range of ridge regularization parameters to try. Can be float or array of shape (n_alphas,). Defaults to [0.1, 1.0, 10.0]. | <code>[0.1, 1.0, 10.0]</code>
`cv` | <code>[int](#int) \| [BaseCrossValidator](#sklearn.model_selection.BaseCrossValidator)</code> | Cross-validation strategy. If int, uses KFold with that many splits. Defaults to 5. | <code>5</code>
`local_alpha` | <code>[bool](#bool)</code> | If True, select best alpha independently for each target. If False, select single best alpha for all targets. Defaults to True. | <code>True</code>
`n_targets_batch` | <code>[int](#int) \| None</code> | Batch size for targets during CV (for memory efficiency). If None, processes all targets at once. Defaults to None. | <code>None</code>
`n_targets_batch_refit` | <code>[int](#int) \| None</code> | Batch size for targets during refit. If None, uses n_targets_batch value. Defaults to None. | <code>None</code>
`n_alphas_batch` | <code>[int](#int) \| None</code> | Batch size for alphas (for memory efficiency). If None, processes all alphas at once. Defaults to None. | <code>None</code>
`Y_in_cpu` | <code>[bool](#bool)</code> | If True, keep Y on CPU and transfer batches to GPU as needed. This prevents OOM when Y is large (e.g., 300k voxels). Defaults to True (recommended for neuroimaging). | <code>True</code>
`score_func` | <code>[Callable](#collections.abc.Callable)[[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)] \| None</code> | Scoring function (y_true, y_pred) -> scores. If None, uses R² score. Defaults to None. | <code>None</code>
`fit_intercept` | <code>[bool](#bool)</code> | Whether to fit an intercept. If False, X and Y should be centered. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Whether to display progress bar (requires tqdm). Defaults to False. | <code>False</code>
`conservative` | <code>[bool](#bool)</code> | If True, select largest alpha within 1 std of best score. Defaults to False. | <code>False</code>
`jitter_alphas` | <code>[bool](#bool)</code> | If True, alphas range is slightly jittered for each gamma. Defaults to False. | <code>False</code>
`return_weights` | <code>[bool](#bool)</code> | Whether to refit on the entire dataset and return the weights. Defaults to True. | <code>True</code>
`diagonalize_method` | <code>[str](#str)</code> | Method used to diagonalize the features. Currently only "svd" is supported. Defaults to "svd". | <code>'svd'</code>
`warn` | <code>[bool](#bool)</code> | If True, warn if the number of samples is smaller than the number of features. Defaults to True. | <code>True</code>
`parallel` | <code>[str](#str) \| None</code> | Backend to use: "cpu", "gpu", or None. Defaults to "cpu". | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>[float](#float) \| None</code> | GPU memory budget in GB (only used if parallel="gpu"). Defaults to 4.0. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random generator seed. Use an int for deterministic search. Defaults to None. | <code>None</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>[ndarray](#numpy.ndarray)</code> | Feature matrix of shape (n_samples, n_features). | *required*
`Y` | <code>[ndarray](#numpy.ndarray)</code> | Target data of shape (n_samples, n_targets). | *required*
`alphas` | <code>[float](#float) \| [ndarray](#numpy.ndarray) \| [list](#list)[[float](#float)]</code> | Ridge regularization parameters to try. Defaults to [0.1, 1.0, 10.0]. | <code>[0.1, 1.0, 10.0]</code>
`cv` | <code>[int](#int) \| [BaseCrossValidator](#sklearn.model_selection.BaseCrossValidator)</code> | Cross-validation strategy. If int, uses KFold with that many splits. Defaults to 5. | <code>5</code>
`local_alpha` | <code>[bool](#bool)</code> | If True, select best alpha independently for each target. If False, select single best alpha for all targets. Defaults to True. | <code>True</code>
`n_targets_batch` | <code>[int](#int) \| None</code> | Batch size for targets during CV (for memory efficiency). If None, processes all targets at once. Defaults to None. | <code>None</code>
`n_targets_batch_refit` | <code>[int](#int) \| None</code> | Batch size for targets during refit. If None, uses n_targets_batch value. Defaults to None. | <code>None</code>
`n_alphas_batch` | <code>[int](#int) \| None</code> | Batch size for alphas (for memory efficiency). If None, processes all alphas at once. Defaults to None. | <code>None</code>
`Y_in_cpu` | <code>[bool](#bool)</code> | If True, keep Y on CPU and transfer batches to GPU as needed. This prevents OOM when Y is large (e.g., 300k voxels). Defaults to True (recommended for neuroimaging). | <code>True</code>
`score_func` | <code>[Callable](#collections.abc.Callable)[[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray)], [ndarray](#numpy.ndarray)] \| None</code> | Scoring function (y_true, y_pred) -> scores. If None, uses R² score. Defaults to None. | <code>None</code>
`fit_intercept` | <code>[bool](#bool)</code> | Whether to fit an intercept. If False, X and Y should be centered. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Whether to display progress bar (requires tqdm). Defaults to False. | <code>False</code>
`conservative` | <code>[bool](#bool)</code> | If True, select largest alpha within 1 std of best score. Defaults to False. | <code>False</code>
`parallel` | <code>[str](#str) \| None</code> | Backend to use: "cpu", "gpu", or None. Defaults to "cpu". | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>[float](#float) \| None</code> | GPU memory budget in GB (only used if parallel="gpu"). Defaults to 4.0. | <code>None</code>
`random_state` | <code>[int](#int) \| None</code> | Random generator seed. Use an int for deterministic search. Defaults to None. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with keys: - 'predictions': (n_samples, n_targets) held-out per-target   predictions on the original Y scale (CPU numpy). - 'folds': (n_samples,) int fold index per row (CPU numpy). - 'scores': (n_splits, n_targets) per-fold R² (or   ``score_func``) at the supplied alpha (CPU numpy). - 'backend': Backend used (for transparency).

######## `solve_banded_ridge_cv`

```python
solve_banded_ridge_cv(Xs: list[np.ndarray], Y: np.ndarray, *, n_iter: int | np.integer | np.ndarray = 100, concentration: float | list[float] = [0.1, 1.0], alphas: float | np.ndarray | list[float] = [0.1, 1.0, 10.0], cv: int | BaseCrossValidator = 5, local_alpha: bool = True, n_targets_batch: int | None = None, n_targets_batch_refit: int | None = None, n_alphas_batch: int | None = None, Y_in_cpu: bool = True, score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, fit_intercept: bool = False, progress_bar: bool = False, conservative: bool = False, jitter_alphas: bool = False, return_weights: bool = True, diagonalize_method: str = 'svd', warn: bool = True, parallel: str | None = 'cpu', max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> dict[str, Any]
```

Solve banded ridge regression with cross-validation using random search.

This function implements true banded/group ridge regression (as in Himalaya).
It searches over feature space weights (gamma) sampled from a Dirichlet
distribution, combined with alpha grid search.

Banded ridge (also called group ridge) applies different scaling weights
per feature space: Z_i = sqrt(gamma_i) * X_i, then solves standard ridge
regression on the scaled concatenated features. This allows optimizing
the relative importance of different feature spaces.

The feature spaces are scaled by sqrt(gamma) for each gamma sample, then
standard ridge regression is applied with alpha grid search.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with keys: - 'deltas': Best log feature-space weights for each target,     shape (n_spaces, n_targets). deltas = log(gamma / alpha), where     gamma are the feature space weights. - 'cv_scores': Cross-validation scores per iteration, averaged over splits,     for the best alpha, shape (n_iter, n_targets). Always returned on CPU     (numpy array). - 'coefs': Ridge coefficients refit on entire dataset using best hyperparameters,     shape (n_features_total, n_targets), or None if return_weights=False.     Always returned on CPU (numpy array). - 'intercept': Intercept of shape (n_targets,), or None if     fit_intercept=False or return_weights=False. - 'backend': Backend used (for transparency).

**Examples:**

```pycon
>>> # Multiple feature spaces (banded ridge with random search)
>>> X1 = np.random.randn(100, 30)  # First feature space
>>> X2 = np.random.randn(100, 20)  # Second feature space
>>> Y = np.random.randn(100, 10)
>>> result = solve_banded_ridge_cv(
...     [X1, X2], Y, n_iter=50, alphas=[0.1, 1.0, 10.0]
... )
>>> deltas = result['deltas']
>>> coefs = result['coefs']
>>> scores = result['cv_scores']
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

This implements true banded/group ridge regression (as in Himalaya's
solve_group_ridge_random_search) with:
- Dirichlet sampling for feature space weights (gamma)
- Scaling each feature space by sqrt(gamma) for each gamma sample
- Cross-validation with alpha grid search
- Per-target selection of best gamma and alpha combination

This is the correct implementation of banded/group ridge regression, which
allows different scaling weights per feature space. For single feature space
ridge regression, use solve_ridge_cv instead.

Algorithm details:

- Random search: Samples gamma weights from Dirichlet distribution
- Banded ridge: Scales each feature space by sqrt(gamma_i), then solves standard ridge
- Cross-validation: Evaluates each (gamma, alpha) combination via k-fold CV
- Best selection: Chooses (gamma, alpha) that maximizes CV score per target

Memory efficiency strategies (Principle 2: automatic memory efficiency):

- Generator pattern for alpha batching (via _decompose_ridge): Processes alphas
  in batches to avoid storing all resolution matrices simultaneously
- Target batching (n_targets_batch): Processes targets in chunks to fit GPU memory
- Y_in_cpu strategy: Keeps large Y on CPU, transfers only batches needed
  for computation
- Immediate cleanup with del statements: Explicitly frees memory after each batch

Performance:

- Time complexity: O(n_iter × n_splits × (n_alphas_batch × n_features^2 + n_targets_batch × n_samples))
- Memory complexity: O(n_features × n_targets_batch) per batch
- GPU acceleration: ~10-100× speedup for large problems (n_features > 10K)

See ``nltools.algorithms.ridge.utils._decompose_ridge()`` for generator pattern details.
See ``docs/development/ridge-internals.md`` for detailed algorithm explanation.

</details>

######## `solve_ridge_cv`

```python
solve_ridge_cv(X: np.ndarray, Y: np.ndarray, *, alphas: float | np.ndarray | list[float] = [0.1, 1.0, 10.0], cv: int | BaseCrossValidator = 5, local_alpha: bool = True, n_targets_batch: int | None = None, n_targets_batch_refit: int | None = None, n_alphas_batch: int | None = None, Y_in_cpu: bool = True, score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, fit_intercept: bool = False, progress_bar: bool = False, conservative: bool = False, parallel: str | None = 'cpu', max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> dict[str, Any]
```

Solve ridge regression with cross-validation.

This function solves ridge regression for a single feature space with
cross-validation for hyperparameter selection.

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with keys: - 'best_alphas': Selected best alpha for each target (or same alpha repeated     if local_alpha=False), shape (n_targets,). - 'coefs': Ridge coefficients refit on entire dataset using best alphas,     shape (n_features, n_targets). Always returned on CPU (numpy array). - 'cv_scores': Cross-validation scores for best alphas, shape (n_splits, n_alphas, n_targets).     Always returned on CPU (numpy array). - 'intercept': Per-target intercept of shape (n_targets,). Only present     when ``fit_intercept=True``. - 'backend': Backend used (for transparency).

**Examples:**

```pycon
>>> X = np.random.randn(100, 50)
>>> Y = np.random.randn(100, 10)
>>> result = solve_ridge_cv(X, Y, alphas=[0.1, 1.0, 10.0])
>>> alphas = result['best_alphas']
>>> coefs = result['coefs']
>>> scores = result['cv_scores']
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

This is the efficient implementation for single feature space ridge regression
with cross-validation. For multiple feature spaces (banded/group ridge),
use solve_banded_ridge_cv instead.

Algorithm details:

- Cross-validation: k-fold CV evaluates each alpha value
- Alpha selection: Chooses best alpha per target (or globally if local_alpha=False)
- Refit: Fits final model on full dataset using best alpha(s)

Memory efficiency strategies (Principle 2: automatic memory efficiency):

- Generator pattern for alpha batching (via _decompose_ridge): Processes alphas
  in batches to avoid storing all resolution matrices simultaneously
- Target batching (n_targets_batch): Processes targets in chunks to fit GPU memory
- Y_in_cpu strategy: Keeps large Y on CPU, transfers only batches needed
  for computation
- Immediate cleanup with del statements: Explicitly frees memory after each batch

Performance:

- Time complexity: O(n_splits × (n_alphas_batch × n_features^2 + n_targets_batch × n_samples))
- Memory complexity: O(n_features × n_targets_batch) per batch
- GPU acceleration: ~10-100× speedup for large problems (n_features > 10K)

See ``nltools.algorithms.ridge.utils._decompose_ridge()`` for generator pattern details.
See ``docs/development/ridge-internals.md`` for detailed algorithm explanation.

</details>

###### `utils`

Utility functions for ridge regression.

Contains helper functions for batching, decomposition, and other utilities
following himalaya's implementation patterns.

**Methods:**

Name | Description
---- | -----------
[`generate_dirichlet_samples`](#algorithms-generate-dirichlet-samples) | Generate samples from a Dirichlet distribution.



####### Classes

####### Functions##

###### `generate_dirichlet_samples`

```python
generate_dirichlet_samples(n_samples: int, n_kernels: int, concentration: float | list[float] = [0.1, 1.0], random_state: int | None = None) -> np.ndarray
```

Generate samples from a Dirichlet distribution.

This function generates random samples from a Dirichlet distribution,
which is used for sampling feature space weights (gamma) in banded ridge
regression random search.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_samples` | <code>[int](#int)</code> | Number of samples to generate. | *required*
`n_kernels` | <code>[int](#int)</code> | Number of dimensions (feature spaces) of the distribution. | *required*
`concentration` | <code>[float](#float) \| [list](#list)[[float](#float)]</code> | Concentration parameters of the Dirichlet distribution. - A value of 1 corresponds to uniform sampling over the simplex. - A value of infinity corresponds to equal weights. - If a list, samples cycle through the list. Defaults to [0.1, 1.0]. | <code>[0.1, 1.0]</code>
`random_state` | <code>[int](#int) \| None</code> | Random generator seed. Use an int for deterministic samples. Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | np.ndarray: Dirichlet samples of shape (n_samples, n_kernels). Each row sums to 1 (lies on simplex).

**Examples:**

```pycon
>>> # Generate 10 samples for 3 feature spaces
>>> gammas = generate_dirichlet_samples(10, 3, concentration=[0.1, 1.0])
>>> gammas.shape
(10, 3)
>>> # Each row sums to 1
>>> np.allclose(gammas.sum(axis=1), 1.0)
True
```

(algorithms-shape-utils)=
#### `shape_utils`

Shared shape-manipulation helpers for triangle extraction and symmetric permutation.

<details class="key-functions" open markdown="1">
<summary>Key functions</summary>

- extract_triangle_elements: Extract upper/lower triangle from matrices
- permute_matrix_symmetric: Apply symmetric permutation (key for matrix tests)

</details>

<details class="usage" open markdown="1">
<summary>Usage</summary>

These utilities are used throughout the algorithms module for consistent
shape handling and matrix operations.

Example:
    >>> from nltools.algorithms.shape_utils import extract_triangle_elements
    >>> matrix = np.arange(16).reshape(4, 4)
    >>> upper = extract_triangle_elements(matrix, triangle='upper')

</details>

**Methods:**

Name | Description
---- | -----------
[`extract_triangle_elements`](#algorithms-extract-triangle-elements) | Extract triangle elements from square matrix.
[`permute_matrix_symmetric`](#algorithms-permute-matrix-symmetric) | Apply symmetric row+column permutation to square matrix.



##### Methods

(algorithms-extract-triangle-elements)=
###### `extract_triangle_elements`

```python
extract_triangle_elements(matrix: np.ndarray, triangle: str = 'upper', include_diag: bool = False) -> np.ndarray
```

Extract triangle elements from square matrix.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`matrix` | <code>[ndarray](#numpy.ndarray)</code> | Square matrix (n×n) | *required*
`triangle` | <code>[str](#str)</code> | Which triangle ['upper'|'lower'|'full'] | <code>'upper'</code>
`include_diag` | <code>[bool](#bool)</code> | Include diagonal (only for 'full') | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Extracted elements as 1D array

**Examples:**

```pycon
>>> matrix = np.arange(16).reshape(4, 4)
>>> extract_triangle_elements(matrix, triangle='upper')
array([ 1,  2,  3,  6,  7, 11])
```

(algorithms-permute-matrix-symmetric)=
###### `permute_matrix_symmetric`

```python
permute_matrix_symmetric(matrix: np.ndarray, permutation: np.ndarray) -> np.ndarray
```

Apply symmetric row+column permutation to square matrix.

This is the KEY operation for matrix permutation tests. It reorders
both rows AND columns together, preserving matrix structure while
destroying correlation between matrices.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`matrix` | <code>[ndarray](#numpy.ndarray)</code> | Square matrix (n×n) | *required*
`permutation` | <code>[ndarray](#numpy.ndarray)</code> | Permutation indices (length n) | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Symmetrically permuted matrix (n×n)

**Examples:**

```pycon
>>> matrix = np.arange(9).reshape(3, 3)
>>> perm = np.array([2, 0, 1])  # Rotate indices
>>> permute_matrix_symmetric(matrix, perm)
array([[8, 6, 7],
       [2, 0, 1],
       [5, 3, 4]])
```

(algorithms-signal)=
#### `signal`

Temporal signal processing — resampling, filtering, and basis functions.

**Methods:**

Name | Description
---- | -----------
[`calc_bpm`](#algorithms-calc-bpm) | Calculate instantaneous BPM from beat to beat interval.
[`downsample`](#algorithms-downsample) | Downsample a Polars DataFrame/Series to a new target frequency or number of samples using averaging.
[`make_cosine_basis`](#algorithms-make-cosine-basis) | Create basis functions for a discrete cosine transform.
[`upsample`](#algorithms-upsample) | Upsample a Polars DataFrame/Series to a new target frequency or number of samples using interpolation.



##### Methods

###### `calc_bpm`

```python
calc_bpm(beat_interval, sampling_freq)
```

Calculate instantaneous BPM from beat to beat interval.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`beat_interval` |  | (int) number of samples in between each beat             (typically R-R Interval) | *required*
`sampling_freq` |  | (float) sampling frequency in Hz | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`bpm` |  | (float) beats per minute for time interval

###### `downsample`

```python
downsample(data, *, sampling_freq = None, target = None, target_type = 'samples', method = 'mean')
```

Downsample a Polars DataFrame/Series to a new target frequency or number of samples using averaging.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pl.DataFrame, pl.Series) data to downsample | *required*
`sampling_freq` |  | (float) Sampling frequency of data in hertz | <code>None</code>
`target` |  | (float) downsampling target | <code>None</code>
`target_type` |  | type of target can be [samples,seconds,hz] | <code>'samples'</code>
`method` |  | (str) type of downsample method ['mean','median'],     default: mean | <code>'mean'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` |  | (pl.DataFrame, pl.Series) downsampled data (same type as input)

###### `make_cosine_basis`

```python
make_cosine_basis(nsamples, sampling_freq, filter_length, unit_scale = True, drop = 0)
```

Create basis functions for a discrete cosine transform.

Based on the implementation in ``spm_filter`` and ``spm_dctmtx`` because
scipy DCT can only apply transforms but not return the basis functions. Like
SPM, this does not add a constant (i.e. intercept), but does retain the first
basis (i.e. sigmoidal/linear drift).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`nsamples` | <code>[int](#int)</code> | number of observations (e.g. TRs) | *required*
`sampling_freq` | <code>[float](#float)</code> | sampling frequency in hertz (i.e. 1 / TR) | *required*
`filter_length` | <code>[int](#int)</code> | length of filter in seconds | *required*
`unit_scale` | <code>[bool](#bool)</code> | assure that the basis functions are on the normalized range [-1, 1]; default True | <code>True</code>
`drop` | <code>[int](#int)</code> | index of which early/slow bases to drop if any; default is to drop constant (i.e. intercept) like SPM. Unlike SPM, retains first basis (i.e. linear/sigmoidal). Will cumulatively drop bases up to and inclusive of index provided (e.g. 2, drops bases 1 and 2) | <code>0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`out` | <code>[ndarray](#ndarray)</code> | nsamples x number of basis sets numpy array

###### `upsample`

```python
upsample(data, *, sampling_freq = None, target = None, target_type = 'samples', method = 'linear')
```

Upsample a Polars DataFrame/Series to a new target frequency or number of samples using interpolation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` |  | (pl.DataFrame, pl.Series) data to upsample   (Note: will drop non-numeric columns from DataFrame) | *required*
`sampling_freq` |  | Sampling frequency of data in hertz | <code>None</code>
`target` |  | (float) upsampling target | <code>None</code>
`target_type` |  | (str) type of target can be [samples,seconds,hz] | <code>'samples'</code>
`method` |  | (str) ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']           where 'zero', 'slinear', 'quadratic' and 'cubic'           refer to a spline interpolation of zeroth, first,           second or third order  (default: linear) | <code>'linear'</code>

Returns:
    upsampled Polars DataFrame or Series (same type as input)

(algorithms-similarity)=
#### `similarity`

Similarity metrics and correlation.

**Methods:**

Name | Description
---- | -----------
[`compute_multivariate_similarity`](#algorithms-compute-multivariate-similarity) | Compute multivariate similarity via OLS regression.
[`compute_similarity`](#algorithms-compute-similarity) | Compute similarity between two data arrays.
[`fisher_r_to_z`](#algorithms-fisher-r-to-z) | Use Fisher transformation to convert correlation to z score.
[`fisher_z_to_r`](#algorithms-fisher-z-to-r) | Convert Fisher z back to a correlation coefficient.
[`transform_pairwise`](#algorithms-transform-pairwise) | Transform data into pairs with balanced labels for ranking.



##### Methods

###### `compute_multivariate_similarity`

```python
compute_multivariate_similarity(y, X, method = 'ols', tail = 2)
```

Compute multivariate similarity via OLS regression.

This is the functional core implementation for multivariate similarity computation.
Used by BrainData.multivariate_similarity() to delegate computation to the functional core.

Predicts spatial distribution of y from linear combination of X columns.
Computes OLS regression statistics including beta coefficients, t-statistics,
p-values, and residuals.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` | <code>[ndarray](#numpy.ndarray)</code> | Target data, shape (n_features,) - single image | *required*
`X` | <code>[ndarray](#numpy.ndarray)</code> | Predictor data, shape (n_features, n_predictors) where first column should be intercept (ones) if intercept is desired. If X does not include intercept, an intercept will be added automatically. | *required*
`method` | <code>[str](#str)</code> | Regression method (currently only 'ols' supported) | <code>'ols'</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` |  | Dictionary with keys: - 'beta': Regression coefficients including intercept, shape (n_predictors+1,) - 't': t-statistics, shape (n_predictors+1,) - 'p': p-values, shape (n_predictors+1,) - 'df': Degrees of freedom (int) - 'sigma': Residual standard deviation (float) - 'residual': Residuals, shape (n_features,)

**Examples:**

```pycon
>>> y = np.random.randn(100)
>>> X = np.random.randn(100, 5)
>>> result = compute_multivariate_similarity(y, X, method='ols')
>>> 'beta' in result
True
>>> result['beta'].shape
(6,)  # 5 predictors + intercept
```

###### `compute_similarity`

```python
compute_similarity(data1, data2, metric = 'correlation')
```

Compute similarity between two data arrays.

This is the functional core implementation for similarity computation.
Used by BrainData.similarity() to delegate computation to the functional core.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>[ndarray](#numpy.ndarray)</code> | First data array, shape (n_samples1, n_features) | *required*
`data2` | <code>[ndarray](#numpy.ndarray)</code> | Second data array, shape (n_samples2, n_features) | *required*
`metric` | <code>[str](#str)</code> | Type of similarity metric - 'correlation' or 'pearson': Pearson correlation - 'spearman' or 'rank_correlation': Spearman rank correlation - 'dot_product': Dot product - 'cosine': Cosine similarity | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
 | np.ndarray: Similarity matrix or vector - If data1.shape[0] == 1 and data2.shape[0] == 1: scalar - If data1.shape[0] == 1 or data2.shape[0] == 1: 1D array - Otherwise: 2D array shape (n_samples1, n_samples2)

**Examples:**

```pycon
>>> data1 = np.random.randn(10, 100)
>>> data2 = np.random.randn(5, 100)
>>> sim = compute_similarity(data1, data2, metric='correlation')
>>> sim.shape
(10, 5)
```

###### `fisher_r_to_z`

```python
fisher_r_to_z(r)
```

Use Fisher transformation to convert correlation to z score.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`r` |  | correlation coefficient(s) | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`z` |  | Fisher z-transformed correlation(s)

###### `fisher_z_to_r`

```python
fisher_z_to_r(z)
```

Convert Fisher z back to a correlation coefficient.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`z` |  | Fisher z-transformed value(s) | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`r` |  | correlation coefficient(s)

###### `transform_pairwise`

```python
transform_pairwise(X, y)
```

Transform data into pairs with balanced labels for ranking.

Transforms a n-class ranking problem into a two-class classification
problem. Subclasses implementing particular strategies for choosing
pairs should override this method.
In this method, all pairs are choosen, except for those that have the
same target value. The output is an array of balanced classes, i.e.
there are the same number of -1 as +1

Reference: "Large Margin Rank Boundaries for Ordinal Regression",
R. Herbrich, T. Graepel, K. Obermayer. Authors: Fabian Pedregosa
<fabian@fseoane.net> Alexandre Gramfort <alexandre.gramfort@inria.fr>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` |  | (np.array), shape (n_samples, n_features) The data | *required*
`y` |  | (np.array), shape (n_samples,) or (n_samples, 2) Target labels. If it's a 2D array, the second column represents the grouping of samples, i.e., samples with different groups will not be considered. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`X_trans` |  | (np.array), shape (k, n_features) Data as pairs, where k = n_samples * (n_samples-1)) / 2 if grouping values were not passed. If grouping variables exist, then returns values computed for each group.
`y_trans` |  | (np.array), shape (k,) Output class labels, where classes have values {-1, +1} If y was shape (n_samples, 2), then returns (k, 2) with groups on the second dimension.

