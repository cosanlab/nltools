---
title: algorithms
---

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
[`glover_dispersion_derivative`](#algorithms-glover-dispersion-derivative) | Sample the dispersion derivative of the Glover hemodynamic response function.
[`glover_hrf`](#algorithms-glover-hrf) | Sample the Glover hemodynamic response function.
[`glover_time_derivative`](#algorithms-glover-time-derivative) | Sample the time derivative of the Glover hemodynamic response function.
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
[`spm_dispersion_derivative`](#algorithms-spm-dispersion-derivative) | Sample the dispersion derivative of the SPM canonical hemodynamic response function.
[`spm_hrf`](#algorithms-spm-hrf) | Sample the SPM canonical hemodynamic response function.
[`spm_time_derivative`](#algorithms-spm-time-derivative) | Sample the time derivative of the SPM canonical hemodynamic response function.
[`threshold`](#algorithms-threshold) | Threshold test image by p-value from p image.
[`timeseries_correlation_permutation_test`](#algorithms-timeseries-correlation-permutation-test) | Time-series correlation permutation test.
[`transform_pairwise`](#algorithms-transform-pairwise) | Transform data into pairs with balanced labels for ranking.
[`trim`](#algorithms-trim) | Trim a Polars DataFrame/Series by replacing outlier values with NaNs.
[`two_sample_permutation_test`](#algorithms-two-sample-permutation-test) | Two-sample permutation test using group label shuffling.
[`u_center`](#algorithms-u-center) | U-center a 2d array.
[`upsample`](#algorithms-upsample) | Upsample a Polars DataFrame/Series to a new target frequency or number of samples using interpolation.
[`winsorize`](#algorithms-winsorize) | Winsorize a Polars DataFrame/Series with the largest/lowest value not considered outlier.
[`zscore`](#algorithms-zscore) | Z-score every column of a Polars or pandas DataFrame/Series.



## Classes

(algorithms-detsrm)=
### `DetSRM`

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

#### Methods

(algorithms-fit)=
##### `fit`

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

Type | Description
---- | -----------
<code>[DetSRM](#nltools.algorithms.alignment.srm.DetSRM)</code> | Fitted model (`self`).

(algorithms-transform)=
##### `transform`

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

Type | Description
---- | -----------
<code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | Shared responses from input data (X); element i has     shape=[features_i, samples_i].

(algorithms-transform-subject)=
##### `transform_subject`

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

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Orthogonal mapping `W_{new}` for the new subject,     shape=[voxels, features].

(algorithms-hyperalignment)=
### `HyperAlignment`

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

<details class="note" open markdown="1">
<summary>Note</summary>

Reference: Haxby, J. V., Guntupalli, J. S., Connolly, A. C., Halchenko, Y. O.,
Conroy, B. R., Gobbini, M. I., ... & Ramadge, P. J. (2011).
A common, high-dimensional model of the representational space in
human ventral temporal cortex. Neuron, 72(2), 404-416.

</details>

#### Methods

##### `fit`

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

Type | Description
---- | -----------
<code>[HyperAlignment](#nltools.algorithms.alignment.hyperalignment.HyperAlignment)</code> | Fitted model (`self`).

##### `transform`

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

Type | Description
---- | -----------
<code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | List of transformed data matrices in common space.

##### `transform_subject`

```python
transform_subject(subject_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]
```

Align a new subject to the common space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`subject_data` | <code>([ndarray](#ndarray), [shape](#shape)([n_features](#n_features), [n_samples](#n_samples)))</code> | Data from a new subject to align to the common template | *required*

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray), [float](#float), [float](#float)]</code> | `(transformed, R, disparity,     scale)` — aligned data in common space, the transformation matrix     used, the alignment quality (sum of squared differences), and the     scale factor used.

(algorithms-localalignment)=
### `LocalAlignment`

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

#### Methods

##### `fit`

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

Type | Description
---- | -----------
<code>[LocalAlignment](#nltools.algorithms.alignment.local.LocalAlignment)</code> | The fitted alignment model (`self`).

(algorithms-fit-transform)=
##### `fit_transform`

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
<code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | Aligned data for each subject.

##### `transform`

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
<code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | Aligned data for each subject, each shape     (n_voxels, n_samples).

(algorithms-srm)=
### `SRM`

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

#### Methods

##### `fit`

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

Type | Description
---- | -----------
<code>[SRM](#nltools.algorithms.alignment.srm.SRM)</code> | Fitted model (`self`).

##### `transform`

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

Type | Description
---- | -----------
<code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | Shared responses from input data (X); element i has     shape=[features_i, samples_i].

##### `transform_subject`

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

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Orthogonal mapping `W_{new}` for the new subject,     shape=[voxels, features].



## Methods

(algorithms-align)=
### `align`

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

Type | Description
---- | -----------
<code>[dict](#dict)</code> | A dictionary containing a list of transformed subject matrices, a     list of transformation matrices, the shared response matrix, and the     intersubject correlation of the shared responses.

**Examples:**

```python
# Hyperalign using procrustes transform
out = align(data, method='procrustes')

# Align using shared response model
out = align(data, method='probabilistic_srm', n_features=None)

# Project aligned data back into original data space
original_data = [
    np.dot(t.data, tm.T)
    for t, tm in zip(out['transformed'], out['transformation_matrix'])
]
```

(algorithms-align-states)=
### `align_states`

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
`replace_zero_variance` |  | (bool) transform a vector with zero variance to random numbers from a uniform distribution. Useful when using correlation as a distance metric to avoid NaNs. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | If `return_index=False` (default), `target[:, remapping]` — the     target's columns reordered to match the reference, oriented pattern x     state (same shape as `target`). If `return_index=True`, the remapping     index array that reorders the target's state columns.

(algorithms-calc-bpm)=
### `calc_bpm`

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

Type | Description
---- | -----------
<code>[float](#float)</code> | Beats per minute for the time interval.

(algorithms-circle-shift)=
### `circle_shift`

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
### `compute_multivariate_similarity`

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

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with keys:     - 'beta': Regression coefficients including intercept, shape (n_predictors+1,)     - 't': t-statistics, shape (n_predictors+1,)     - 'p': p-values, shape (n_predictors+1,)     - 'df': Degrees of freedom (int)     - 'sigma': Residual standard deviation (float)     - 'residual': Residuals, shape (n_features,)

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
### `compute_similarity`

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
<code>[ndarray](#numpy.ndarray)</code> | Similarity matrix or vector     - If data1.shape[0] == 1 and data2.shape[0] == 1: scalar     - If data1.shape[0] == 1 or data2.shape[0] == 1: 1D array     - Otherwise: 2D array shape (n_samples1, n_samples2)

**Examples:**

```pycon
>>> data1 = np.random.randn(10, 100)
>>> data2 = np.random.randn(5, 100)
>>> sim = compute_similarity(data1, data2, metric='correlation')
>>> sim.shape
(10, 5)
```

(algorithms-correlation-permutation-test)=
### `correlation_permutation_test`

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
`tail` | <code>[int](#int) \| [str](#str)</code> | `2` or `'two'` for two-tailed (default); `1` or `'one'` for one-tailed (positive direction). - `2`/`'two'`: Two-tailed test (r != 0) - `1`/`'one'`: One-tailed (r > 0; negate one variable for the other   direction). The fixed direction keeps MCP correction valid. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show a progress bar over permutations (default: False) | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with keys:     - 'correlation' (float or np.ndarray): Observed correlation(s)     - 'p' (float or np.ndarray): P-value(s)     - 'null_dist' (np.ndarray): Null distribution (if return_null=True)     - 'device' (str): Parallelization method used

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
### `distance_correlation`

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

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary of results (correlation, t, p, and df); optionally also     covariance, x variance, and y variance.

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
### `double_center`

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

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Double-centered version of the input.

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
### `downsample`

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

Type | Description
---- | -----------
<code>[DataFrame](#polars.DataFrame) \| [Series](#polars.Series)</code> | Downsampled data (same type as input).

(algorithms-fdr)=
### `fdr`

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

Type | Description
---- | -----------
<code>[float](#float)</code> | p-value threshold based on independence or positive dependence.

(algorithms-find-spikes)=
### `find_spikes`

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

Type | Description
---- | -----------
<code>[DesignMatrix](#nltools.data.DesignMatrix)</code> | One indicator column per detected spike TR, named     ``.nl_global_spike{n}`` / ``.nl_diff_spike{n}`` in the reserved     namespace for generated columns (see `RESERVED_PREFIX`), with all     spike columns pre-marked as confounds. The two detectors run     independently, so a single bad volume is routinely caught by both;     those detections are bitwise-identical one-hot columns, and only one     is kept (the ``.nl_global_spike*`` name, a deterministic tie-break —     the column values are the same either way). Row position is the time     axis (no separate `TR` index column — that was a pandas-era     artifact). When `TR` / `sampling_freq` aren't provided the DM has     `sampling_freq=None`; you can still `.append()` it onto a DM that     does have one.

(algorithms-fisher-r-to-z)=
### `fisher_r_to_z`

```python
fisher_r_to_z(r)
```

Use Fisher transformation to convert correlation to z score.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`r` |  | correlation coefficient(s) | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Fisher z-transformed correlation(s).

(algorithms-fisher-z-to-r)=
### `fisher_z_to_r`

```python
fisher_z_to_r(z)
```

Convert Fisher z back to a correlation coefficient.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`z` |  | Fisher z-transformed value(s) | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Correlation coefficient(s).

(algorithms-glover-dispersion-derivative)=
### `glover_dispersion_derivative`

```python
glover_dispersion_derivative(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Sample the dispersion derivative of the Glover hemodynamic response function.

Thin wrapper over nilearn's `glover_dispersion_derivative`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>[float](#float)</code> | Repetition time in seconds. | *required*
`oversampling` | <code>[int](#int)</code> | Temporal oversampling factor (default: 50). | <code>50</code>
`time_length` | <code>[float](#float)</code> | HRF kernel length in seconds (default: 32.0). | <code>32.0</code>
`onset` | <code>[float](#float)</code> | Onset of the response in seconds (default: 0.0). | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#ndarray)</code> | The dispersion derivative sampled every `t_r / oversampling` seconds.

(algorithms-glover-hrf)=
### `glover_hrf`

```python
glover_hrf(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Sample the Glover hemodynamic response function.

Thin wrapper over nilearn's `glover_hrf`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>[float](#float)</code> | Repetition time in seconds. | *required*
`oversampling` | <code>[int](#int)</code> | Temporal oversampling factor (default: 50). | <code>50</code>
`time_length` | <code>[float](#float)</code> | HRF kernel length in seconds (default: 32.0). | <code>32.0</code>
`onset` | <code>[float](#float)</code> | Onset of the response in seconds (default: 0.0). | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#ndarray)</code> | The HRF sampled every `t_r / oversampling` seconds, peak scaled to 1.

(algorithms-glover-time-derivative)=
### `glover_time_derivative`

```python
glover_time_derivative(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Sample the time derivative of the Glover hemodynamic response function.

Thin wrapper over nilearn's `glover_time_derivative`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>[float](#float)</code> | Repetition time in seconds. | *required*
`oversampling` | <code>[int](#int)</code> | Temporal oversampling factor (default: 50). | <code>50</code>
`time_length` | <code>[float](#float)</code> | HRF kernel length in seconds (default: 32.0). | <code>32.0</code>
`onset` | <code>[float](#float)</code> | Onset of the response in seconds (default: 0.0). | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#ndarray)</code> | The time derivative sampled every `t_r / oversampling` seconds.

(algorithms-holm-bonf)=
### `holm_bonf`

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

Type | Description
---- | -----------
<code>[float](#float)</code> | p-value threshold based on the Bonferroni step-down procedure.

(algorithms-isc)=
### `isc`

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
`data` | <code>[DataFrame](#pd.DataFrame) \| [ndarray](#numpy.ndarray)</code> | Observations by subjects; ISC is computed across subjects. | *required*
`n_samples` | <code>[int](#int)</code> | Number of random samples/bootstraps. | <code>5000</code>
`summary` | <code>[str](#str)</code> | ISC summary statistic, one of 'mean' or 'median' (default: 'median'). | <code>'median'</code>
`method` | <code>[str](#str)</code> | Method to compute p-values, one of 'bootstrap', 'circle_shift', or 'phase_randomize' (default: 'bootstrap'). | <code>'bootstrap'</code>
`ci_percentile` | <code>[int](#int)</code> | Confidence-interval width in percent for the bootstrap CI (default: 95). | <code>95</code>
`exclude_self_corr` | <code>[bool](#bool)</code> | Set self-correlations (same subject bootstrapped twice) to nan (default: True). | <code>True</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | `2`/`'two'` (two-tailed, default) or `1`/`'one'` (one-tailed, positive direction). | <code>2</code>
`metric` | <code>[str](#str)</code> | Pairwise distance metric; see sklearn's `pairwise_distances` for valid inputs (default: 'correlation'). | <code>'correlation'</code>
`return_null` | <code>[bool](#bool)</code> | Return the permutation distribution along with the p-value (default: False). | <code>False</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPUs to use; -1 means all CPUs. | <code>-1</code>
`random_state` | <code>[int](#int) \| [RandomState](#numpy.random.RandomState) \| None</code> | Seed or generator for the resampling (default: None). | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | If True, display a progress bar (default: False). | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Permutation results with keys 'isc', 'p', 'ci', and 'null_dist'.

(algorithms-isc-group)=
### `isc_group`

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
`group1` | <code>[DataFrame](#pd.DataFrame) \| [ndarray](#numpy.ndarray)</code> | Observations by subjects for the first group. | *required*
`group2` | <code>[DataFrame](#pd.DataFrame) \| [ndarray](#numpy.ndarray)</code> | Observations by subjects for the second group. | *required*
`n_samples` | <code>[int](#int)</code> | Number of samples for permutation or bootstrapping. | <code>5000</code>
`summary` | <code>[str](#str)</code> | ISC summary statistic, one of 'mean' or 'median' (default: 'median'). | <code>'median'</code>
`method` | <code>[str](#str)</code> | Method to compute p-values, one of 'permute' or 'bootstrap' (default: 'permute'). | <code>'permute'</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (default: 95). | <code>95</code>
`exclude_self_corr` | <code>[bool](#bool)</code> | Exclude self-correlations in bootstrap (default: True). | <code>True</code>
`return_null` | <code>[bool](#bool)</code> | Return the permutation distribution along with the p-value (default: False). | <code>False</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | `2`/`'two'` (two-tailed, default) or `1`/`'one'` (one-tailed, positive direction). | <code>2</code>
`metric` | <code>[str](#str)</code> | Pairwise distance metric; see sklearn's `pairwise_distances` for valid inputs (default: 'correlation'). | <code>'correlation'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPUs to use; -1 means all CPUs. | <code>-1</code>
`random_state` | <code>[int](#int) \| [RandomState](#numpy.random.RandomState) \| None</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | If True, display a progress bar (default: False). | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Permutation results with keys 'isc_group_difference' (observed ISC difference,     float or array), 'p' (p-value, float or array), 'ci' (confidence interval tuple     `(lower, upper)`), and 'null_dist' (null distribution, only if `return_null=True`).

(algorithms-isc-group-permutation-test)=
### `isc_group_permutation_test`

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
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with keys 'isc_group_difference' (observed ISC difference, float or     array per voxel), 'p' (Phipson-Smyth corrected p-value), 'ci' (confidence     interval tuple `(lower, upper)`), 'device' (parallelization method used),     and optionally 'null_dist' (bootstrap/permutation distribution).

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
### `isc_permutation_test`

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
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with keys 'isc' (observed ISC value, float or array per voxel),     'p' (Phipson-Smyth corrected p-value), 'ci' (confidence interval tuple     `(lower, upper)`), 'device' (parallelization method used), and     optionally 'null_dist' (bootstrap/permutation distribution).

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
### `isfc`

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
`data` | <code>[list](#list)[[ndarray](#numpy.ndarray)]</code> | Subject matrices (observations x voxels/rois). | *required*
`method` | <code>[str](#str)</code> | Approach to computing ISFC; 'average' uses leave-one-out. | <code>'average'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs; -1 means all available cores (default: -1). | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
<code>[list](#list)</code> | One ISFC matrix (`np.ndarray`) per subject.

(algorithms-isps)=
### `isps`

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
`data` | <code>[DataFrame](#pd.DataFrame) \| [ndarray](#numpy.ndarray)</code> | Observations x subjects data. | *required*
`sampling_freq` | <code>[float](#float)</code> | Sampling frequency of the data in Hz. | <code>0.5</code>
`low_cut` | <code>[float](#float)</code> | Lower cutoff for the bandpass filter. | <code>0.04</code>
`high_cut` | <code>[float](#float)</code> | Upper cutoff for the bandpass filter. | <code>0.07</code>
`order` | <code>[int](#int)</code> | Butterworth bandpass filter order. | <code>5</code>
`pairwise` | <code>[bool](#bool)</code> | If True, compute phase-angle coherence on pairwise phase-angle differences instead of on the raw phase angles. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Mean phase angle, vector length, and Rayleigh statistic.

(algorithms-make-cosine-basis)=
### `make_cosine_basis`

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

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | nsamples x number of basis sets numpy array.

(algorithms-matrix-permutation-test)=
### `matrix_permutation_test`

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
`metric` | <code>[str](#str)</code> | Correlation metric, one of 'pearson', 'spearman', or 'kendall' (default: 'pearson') | <code>'pearson'</code>
`how` | <code>[str](#str)</code> | Which elements to compare, one of 'upper', 'lower', or 'full' (default: 'upper') - 'upper': Upper triangle only (assumes symmetric matrices) - 'lower': Lower triangle only - 'full': All elements (see include_diag) | <code>'upper'</code>
`include_diag` | <code>[bool](#bool)</code> | Include diagonal elements (only applies if how='full') (default: False) | <code>False</code>
`tail` | <code>[int](#int) \| [str](#str)</code> | `2` or `'two'` for two-tailed (default); `1` or `'one'` for one-tailed (positive direction). - `2`/`'two'`: Two-tailed test (r != 0) - `1`/`'one'`: One-tailed (r > 0; negate the data for the other direction) | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | Return null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel workers, -1 = all cores (default: -1) Only used when device='cpu' | <code>-1</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Show a progress bar over permutations (default: False) | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with keys:     - 'correlation' (float): Observed correlation coefficient     - 'p' (float): P-value using Phipson-Smyth correction     - 'device' (str): Parallelization method used ('cpu' or None)     - 'null_dist' (np.ndarray): Null distribution (if return_null=True)

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
### `multi_threshold`

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

Type | Description
---- | -----------
<code>[BrainData](#nltools.data.BrainData)</code> | Thresholded BrainData instance with cumulative map. Positive     values indicate how many thresholds were passed for positive stats;     negative values indicate how many thresholds were passed for negative     stats.

<details class="note" open markdown="1">
<summary>Note</summary>

This function provides unique cumulative threshold map functionality:
- Creates a single map showing which thresholds were passed
- Different from calling threshold() multiple times (which would give separate images)
- Useful for visualizing threshold hierarchies
- nilearn.threshold_img() does not support cumulative multi-threshold maps

</details>

(algorithms-one-sample-permutation-test)=
### `one_sample_permutation_test`

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
`tail` | <code>[int](#int) \| [str](#str)</code> | `2` or `'two'` for two-tailed (default); `1` or `'one'` for one-tailed (positive direction). - `2`/`'two'`: Two-tailed test (mean != 0) - `1`/`'one'`: One-tailed (mean > 0; negate the data for the other   direction). The fixed direction keeps MCP correction valid. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>
`progress_bar` | <code>[bool](#bool)</code> | Whether to display a progress bar (default: False) | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with keys:     - 'mean' (float or np.ndarray): Observed mean(s)     - 'p' (float or np.ndarray): P-value(s)     - 'null_dist' (np.ndarray): Null distribution (if return_null=True)     - 'device' (str): Parallelization method used

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
### `phase_randomize`

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
### `procrustes_distance`

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
`tail` | <code>[int](#int) or [str](#str)</code> | `2` or `'two'` for two-tailed (default); `1` or `'one'` for one-tailed (similarity > chance) | <code>2</code>
`n_jobs` | <code>[int](#int)</code> | The number of CPUs to use to do permutation; default -1 (all) | <code>-1</code>
`random_state` | <code>int, np.random.RandomState, or None</code> | seed or generator for the permutation shuffling; default None | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | results with keys `similarity` (float in [0, 1]) and `p` (permuted p-value)

(algorithms-regress)=
### `regress`

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
`tail` | <code>[int](#int) \| [str](#str)</code> | `2`/`'two'` (two-tailed, default) or `1`/`'one'` (one-tailed: beta > 0; negate a regressor for the other direction). | <code>2</code>

**Returns:**

Type | Description
---- | -----------
<code>[tuple](#tuple)</code> | ``(b, se, t, p, df, res)`` when ``stats='full'`` — coefficients,     standard errors, t-statistics, p-values (per ``tail``), residual     degrees of freedom, and residuals. ``stats='betas'`` returns just     ``b``; ``stats='tstats'`` returns ``(b, t)``.

(algorithms-ridge-cv)=
### `ridge_cv`

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

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary containing:<br>    - 'alpha' (float): Best alpha value selected by CV     - 'coef' (np.ndarray): Coefficients using best alpha on full dataset     - 'cv_scores' (np.ndarray): Cross-validation R**2 scores for each fold, alpha, and target         with shape (n_folds, n_alphas, n_targets)     - 'backend' (str): Backend used for computation

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
### `ridge_svd`

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
<code>[ndarray](#numpy.ndarray)</code> | Ridge regression coefficients     - shape (n_features,) for single-target regression     - shape (n_features, n_targets) for multi-target regression

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
### `spm_dispersion_derivative`

```python
spm_dispersion_derivative(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Sample the dispersion derivative of the SPM canonical hemodynamic response function.

Thin wrapper over nilearn's `spm_dispersion_derivative`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>[float](#float)</code> | Repetition time in seconds. | *required*
`oversampling` | <code>[int](#int)</code> | Temporal oversampling factor (default: 50). | <code>50</code>
`time_length` | <code>[float](#float)</code> | HRF kernel length in seconds (default: 32.0). | <code>32.0</code>
`onset` | <code>[float](#float)</code> | Onset of the response in seconds (default: 0.0). | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#ndarray)</code> | The dispersion derivative sampled every `t_r / oversampling` seconds.

(algorithms-spm-hrf)=
### `spm_hrf`

```python
spm_hrf(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Sample the SPM canonical hemodynamic response function.

Thin wrapper over nilearn's `spm_hrf`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>[float](#float)</code> | Repetition time in seconds. | *required*
`oversampling` | <code>[int](#int)</code> | Temporal oversampling factor (default: 50). | <code>50</code>
`time_length` | <code>[float](#float)</code> | HRF kernel length in seconds (default: 32.0). | <code>32.0</code>
`onset` | <code>[float](#float)</code> | Onset of the response in seconds (default: 0.0). | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#ndarray)</code> | The HRF sampled every `t_r / oversampling` seconds, peak scaled to 1.

(algorithms-spm-time-derivative)=
### `spm_time_derivative`

```python
spm_time_derivative(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Sample the time derivative of the SPM canonical hemodynamic response function.

Thin wrapper over nilearn's `spm_time_derivative`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>[float](#float)</code> | Repetition time in seconds. | *required*
`oversampling` | <code>[int](#int)</code> | Temporal oversampling factor (default: 50). | <code>50</code>
`time_length` | <code>[float](#float)</code> | HRF kernel length in seconds (default: 32.0). | <code>32.0</code>
`onset` | <code>[float](#float)</code> | Onset of the response in seconds (default: 0.0). | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#ndarray)</code> | The time derivative sampled every `t_r / oversampling` seconds.

(algorithms-threshold)=
### `threshold`

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

Type | Description
---- | -----------
<code>[BrainData](#nltools.data.BrainData) \| [tuple](#tuple)[[BrainData](#nltools.data.BrainData), [BrainData](#nltools.data.BrainData)]</code> | The thresholded BrainData instance;     if `return_mask=True`, a tuple `(out, mask)` where `mask` is the     BrainData instance of the thresholding mask.

<details class="note" open markdown="1">
<summary>Note</summary>

This function provides unique functionality not available in nilearn:
- Thresholds stat image based on p-values from separate p-value image
- Neither nilearn.threshold_img nor BrainData.threshold() support this
- BrainData.threshold() thresholds based on stat values themselves
- nilearn.threshold_img() thresholds based on image intensity values

</details>

(algorithms-timeseries-correlation-permutation-test)=
### `timeseries_correlation_permutation_test`

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
<code>Dictionary with keys</code> | - 'correlation': Observed correlation coefficient     - 'p': P-value     - 'null_dist': (if return_null=True) Null distribution     - 'device': Parallelization method used

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
### `transform_pairwise`

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

Type | Description
---- | -----------
<code>[tuple](#tuple)[[ndarray](#numpy.ndarray), [ndarray](#numpy.ndarray)]</code> | `(X_trans, y_trans)` — `X_trans` has shape     (k, n_features) and holds the data as pairs, where     k = n_samples * (n_samples - 1) / 2 if grouping values were not passed;     if grouping variables exist, values are computed within each group.     `y_trans` has shape (k,) and holds the output class labels with values     {-1, +1}; if y was shape (n_samples, 2), it is (k, 2) with groups on the     second dimension.

(algorithms-trim)=
### `trim`

```python
trim(data, cutoff = None)
```

Trim a Polars DataFrame/Series by replacing outlier values with NaNs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[DataFrame](#polars.DataFrame) \| [Series](#polars.Series)</code> | Data to trim. | *required*
`cutoff` | <code>[dict](#dict)</code> | A dictionary with keys `{'std': [low, high]}` or `{'quantile': [low, high]}`. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#polars.DataFrame) \| [Series](#polars.Series)</code> | Trimmed data (outliers replaced with NaN), the     same type as the input.

(algorithms-two-sample-permutation-test)=
### `two_sample_permutation_test`

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
`tail` | <code>[int](#int) \| [str](#str)</code> | `2` or `'two'` for two-tailed (default); `1` or `'one'` for one-tailed (positive direction). - `2`/`'two'`: Two-tailed test (mean1 != mean2) - `1`/`'one'`: One-tailed (mean1 > mean2; swap the groups for the   other direction). The fixed direction keeps MCP correction valid. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`device` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when device='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Explicit GPU memory budget in GB. None (default) measures the device's available memory. Controls automatic batching to prevent OOM errors. Only used with device='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with keys:     - 'mean_diff' (float or np.ndarray): Observed mean difference (data1 - data2)     - 'p' (float or np.ndarray): P-value(s)     - 'null_dist' (np.ndarray): Null distribution (if return_null=True)     - 'device' (str): Parallelization method used

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
### `u_center`

```python
u_center(mat: np.ndarray) -> np.ndarray
```

U-center a 2d array.

U-centering is a bias-corrected form of double-centering: it corrects for the
bias that grows with the number of dimensions under plain double-centering.
The diagonal is explicitly set to zero.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mat` | <code>[ndarray](#ndarray)</code> | 2d numpy array | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#ndarray)</code> | U-centered version of the input.

**Examples:**

```pycon
>>> mat = np.random.randn(5, 5)
>>> result = u_center(mat)
>>> np.allclose(np.diag(result), 0)
True
```

(algorithms-upsample)=
### `upsample`

```python
upsample(data, *, sampling_freq = None, target = None, target_type = 'samples', method = 'linear')
```

Upsample a Polars DataFrame/Series to a new target frequency or number of samples using interpolation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[DataFrame](#polars.DataFrame) \| [Series](#polars.Series)</code> | Data to upsample. Non-numeric columns are dropped from a DataFrame. | *required*
`sampling_freq` | <code>[float](#float)</code> | Sampling frequency of the data in Hz. | <code>None</code>
`target` | <code>[float](#float)</code> | Upsampling target. | <code>None</code>
`target_type` | <code>[str](#str)</code> | Unit of `target`, one of 'samples', 'seconds', or 'hz'. | <code>'samples'</code>
`method` | <code>[str](#str)</code> | Interpolation method, one of 'linear', 'nearest', 'zero', 'slinear', 'quadratic', or 'cubic'; 'zero', 'slinear', 'quadratic' and 'cubic' refer to spline interpolation of zeroth, first, second or third order (default: 'linear'). | <code>'linear'</code>

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#polars.DataFrame) \| [Series](#polars.Series)</code> | Upsampled data, the same type as the input.

(algorithms-winsorize)=
### `winsorize`

```python
winsorize(data, cutoff = None, replace_with_cutoff = True)
```

Winsorize a Polars DataFrame/Series with the largest/lowest value not considered outlier.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[DataFrame](#polars.DataFrame) \| [Series](#polars.Series)</code> | Data to winsorize. | *required*
`cutoff` | <code>[dict](#dict)</code> | A dictionary with keys `{'std': [low, high]}` or `{'quantile': [low, high]}`. | <code>None</code>
`replace_with_cutoff` | <code>[bool](#bool)</code> | If True, replace outliers with the cutoff value; if False, replace them with the closest existing values (default: True). | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[DataFrame](#polars.DataFrame) \| [Series](#polars.Series)</code> | Winsorized data, the same type as the input.

(algorithms-zscore)=
### `zscore`

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
<code>[DataFrame](#polars.DataFrame) \| [Series](#polars.Series)</code> | Same type and shape as the input, each column     z-scored using the sample standard deviation (ddof=1).
