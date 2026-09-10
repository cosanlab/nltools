---
title: nltools.algorithms (A–Z index)
label: page-algorithms
---

Every public function and class of `nltools.algorithms`, alphabetically. Use this page when you know the name; the *Functions by task* pages group the same objects by what they are for.

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
- **hrf**: hemodynamic response functions

Ridge regression lives in `nltools.models.Ridge`, which delegates its numerics
to the Himalaya library.

**Classes:**

Name | Description
---- | -----------
[`DetSRM`](#algorithms-detsrm) | Deterministic Shared Response Model (DetSRM).
[`HyperAlignment`](#algorithms-hyperalignment) | Hyperalignment using iterative Procrustes alignment (Haxby et al., 2011).
[`LocalAlignment`](#algorithms-localalignment) | Local (neighborhood-based) functional alignment across subjects.
[`SRM`](#algorithms-srm) | Probabilistic Shared Response Model (SRM).

**Functions:**

Name | Description
---- | -----------
[`align`](#algorithms-align) | Align subject data into a common response model.
[`align_states`](#algorithms-align-states) | Align state weight maps by minimizing pairwise distance between group states.
[`calc_bpm`](#algorithms-calc-bpm) | Calculate instantaneous BPM from beat to beat interval.
[`circle_shift`](#algorithms-circle-shift) | Circular shift for time-series data.
[`compute_multivariate_similarity`](#algorithms-compute-multivariate-similarity) | Compute multivariate similarity by regressing one pattern on several.
[`compute_similarity`](#algorithms-compute-similarity) | Compute row-wise similarity between two data arrays.
[`correlation_permutation_test`](#algorithms-correlation-permutation-test) | Permutation test for whether the correlation between two arrays differs from zero.
[`distance_correlation`](#algorithms-distance-correlation) | Compute the distance correlation between two arrays to test for multivariate dependence.
[`double_center`](#algorithms-double-center) | Double center a 2d array.
[`downsample`](#algorithms-downsample) | Downsample a Polars DataFrame/Series to a new target frequency or number of samples using averaging.
[`fdr`](#algorithms-fdr) | Determine an FDR threshold for an array of p-values.
[`find_spikes`](#algorithms-find-spikes) | Identify spikes (motion artifacts, intensity outliers) in 4D fMRI data.
[`fisher_r_to_z`](#algorithms-fisher-r-to-z) | Convert correlation coefficients to Fisher z values.
[`fisher_z_to_r`](#algorithms-fisher-z-to-r) | Convert Fisher z back to a correlation coefficient.
[`glover_dispersion_derivative`](#algorithms-glover-dispersion-derivative) | Sample the dispersion derivative of the Glover hemodynamic response function.
[`glover_hrf`](#algorithms-glover-hrf) | Sample the Glover hemodynamic response function.
[`glover_time_derivative`](#algorithms-glover-time-derivative) | Sample the time derivative of the Glover hemodynamic response function.
[`holm_bonf`](#algorithms-holm-bonf) | Determine a Holm-Bonferroni (step-down) threshold for an array of p-values.
[`isc`](#algorithms-isc) | Compute pairwise intersubject correlation from an observations-by-subjects array.
[`isc_group`](#algorithms-isc-group) | Test the difference in pairwise intersubject correlation between two groups.
[`isc_group_permutation_test`](#algorithms-isc-group-permutation-test) | Test the difference in intersubject correlation between two groups.
[`isc_permutation_test`](#algorithms-isc-permutation-test) | Compute intersubject correlation with bootstrap or permutation inference.
[`isfc`](#algorithms-isfc) | Compute intersubject functional connectivity (ISFC) from per-subject matrices.
[`isps`](#algorithms-isps) | Compute dynamic intersubject phase synchrony (ISPS) from an observations-by-subjects array.
[`make_cosine_basis`](#algorithms-make-cosine-basis) | Create basis functions for a discrete cosine transform.
[`matrix_permutation_test`](#algorithms-matrix-permutation-test) | Matrix permutation test (Mantel test) for correlating two square matrices.
[`multi_threshold`](#algorithms-multi-threshold) | Threshold a statistic image at several p-values and count the passes per voxel.
[`one_sample_permutation_test`](#algorithms-one-sample-permutation-test) | One-sample permutation test using sign flipping.
[`phase_randomize`](#algorithms-phase-randomize) | FFT-based phase randomization for time-series data.
[`procrustes`](#algorithms-procrustes) | Perform a Procrustes similarity analysis on two data sets.
[`procrustes_distance`](#algorithms-procrustes-distance) | Test matrix similarity using Procrustes superposition.
[`regress`](#algorithms-regress) | Fit an OLS regression of `Y` on `X`.
[`spm_dispersion_derivative`](#algorithms-spm-dispersion-derivative) | Sample the dispersion derivative of the SPM canonical hemodynamic response function.
[`spm_hrf`](#algorithms-spm-hrf) | Sample the SPM canonical hemodynamic response function.
[`spm_time_derivative`](#algorithms-spm-time-derivative) | Sample the time derivative of the SPM canonical hemodynamic response function.
[`threshold`](#algorithms-threshold) | Threshold a statistic image by the p-values in a separate image.
[`timeseries_correlation_permutation_test`](#algorithms-timeseries-correlation-permutation-test) | Permutation test for the correlation between two autocorrelated time series.
[`transform_pairwise`](#algorithms-transform-pairwise) | Transform data into pairwise differences with balanced labels for ranking.
[`trim`](#algorithms-trim) | Trim a Polars DataFrame/Series by replacing outlier values with NaNs.
[`two_sample_permutation_test`](#algorithms-two-sample-permutation-test) | Two-sample permutation test using group-label shuffling.
[`u_center`](#algorithms-u-center) | U-center a 2d array.
[`upsample`](#algorithms-upsample) | Upsample a Polars DataFrame/Series to a new target frequency or number of samples using interpolation.
[`winsorize`](#algorithms-winsorize) | Winsorize a Polars DataFrame/Series with the largest/lowest value not considered outlier.
[`zscore`](#algorithms-zscore) | Z-score every column of a Polars or pandas DataFrame/Series.

## Classes

(algorithms-detsrm)=
### `DetSRM`

```python
DetSRM(*, n_iter: int = 10, n_features: int = 50, random_state: int = 0)
```

Bases: `sklearn.base.BaseEstimator`, `sklearn.base.TransformerMixin`

Deterministic Shared Response Model (DetSRM).

Factorizes multi-subject data as a shared response S plus one orthogonal
transform W per subject, so that for every subject i

$$
X_i \approx W_i S, \forall i=1 \dots N
$$

The model is fit by the block coordinate descent algorithm of Chen et al.
(2015). Subjects may have different numbers of voxels but must have the
same number of samples. Run time is $O(I (V T K + V K^2))$ and memory
$O(V T)$, with I iterations, V the sum of voxels across subjects, T
samples, and K features (typically $V \gg T \gg K$).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_iter` | <code>int</code> | Number of coordinate-descent iterations. Defaults to 10. | <code>10</code>
`n_features` | <code>int</code> | Number of shared features to compute. Defaults to 50. | <code>50</code>
`random_state` | <code>int</code> | Seed for the random initialization. Defaults to 0. | <code>0</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`w_` | <code>list[ndarray]</code> | Per-subject orthogonal transforms, element i of shape (voxels_i, n_features).
`s_` | <code>ndarray</code> | The shared response, shape (n_features, samples).
`random_state_` | <code>RandomState</code> | Generator seeded from `random_state`.

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Compute the Deterministic Shared Response Model.
[`transform`](#algorithms-transform) | Project each subject's data into the shared response subspace.
[`transform_subject`](#algorithms-transform-subject) | Transform a new subject using the existing model.



**Examples:**

```python
import numpy as np
from nltools.algorithms import DetSRM

data = [np.random.randn(100, 50) for _ in range(3)]  # 3 subjects

detsrm = DetSRM(n_iter=10, n_features=50)
detsrm.fit(data, parallel="cpu", n_jobs=-1)
shared_responses = detsrm.transform(data)  # list of (50, 50) arrays

w = detsrm.w_  # subject-specific transforms
s = detsrm.s_  # shared response
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
`X` | <code>list[ndarray]</code> | One (voxels_i, samples) array per subject; all subjects must have the same number of samples. | *required*
`y` | <code>Any \| None</code> | Ignored; present for scikit-learn compatibility. | <code>None</code>
`parallel` | <code>str \| None</code> | `'cpu'` (default) updates subjects in parallel with joblib; None runs single-threaded NumPy; `'gpu'` raises `NotImplementedError` (never a silent CPU fallback). | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of CPU workers when `parallel='cpu'`; -1 (default) picks a count from available memory. | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
<code>[DetSRM](#tasks-alignment-detsrm)</code> | Fitted model (`self`).

(algorithms-transform)=
##### `transform`

```python
transform(X: list[np.ndarray], y: Any | None = None, *, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray]
```

Project each subject's data into the shared response subspace.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list[ndarray]</code> | One (voxels_i, samples_i) array per fitted subject, in the same order as `fit`; voxel and sample counts may vary across subjects. | *required*
`y` | <code>Any \| None</code> | Ignored; present for scikit-learn compatibility. | <code>None</code>
`parallel` | <code>str \| None</code> | `'cpu'` (default) transforms subjects in parallel with joblib; None runs single-threaded NumPy; `'gpu'` raises `NotImplementedError`. | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of CPU workers when `parallel='cpu'`; -1 (default) reuses the value from `fit`, itself resolved from available memory. | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
<code>list[ndarray]</code> | Shared responses, element i of shape     (n_features, samples_i).

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
`X` | <code>ndarray</code> | The new subject's data, shape (voxels, timepoints); the timepoints must match the fitted shared response. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Orthogonal mapping $W_{new}$ for the new subject, shape     (voxels, n_features).

(algorithms-hyperalignment)=
### `HyperAlignment`

```python
HyperAlignment(n_iter: int = 2, auto_pad: bool = True)
```

Bases: `sklearn.base.BaseEstimator`, `sklearn.base.TransformerMixin`

Hyperalignment using iterative Procrustes alignment (Haxby et al., 2011).

Aligns multi-subject data in three stages: build an initial average
template, refine it over `n_iter` rounds of align-and-average, then align
every subject to the refined template. Each subject's data is a
(n_features, n_samples) matrix; subjects may differ in `n_features` when
`auto_pad=True`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_iter` | <code>int</code> | Number of template refinement iterations. Defaults to 2. | <code>2</code>
`auto_pad` | <code>bool</code> | If True, zero-pad each subject's feature axis up to the largest feature count. If False, all matrices must already have the same shape. Defaults to True. | <code>True</code>

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_iter` | <code>int</code> | Number of template refinement iterations. Defaults to 2. | <code>2</code>
`auto_pad` | <code>bool</code> | Whether to zero-pad matrices to the same size. Defaults to True. | <code>True</code>



**Attributes:**

Name | Type | Description
---- | ---- | -----------
`w_` | <code>list[ndarray]</code> | Per-subject transformation matrices (rotation + reflection), each of shape (n_features, n_features).
`s_` | <code>ndarray</code> | The common template, shape (n_features, n_samples).
`common_model_` | <code>ndarray</code> | Alias for `s_`.
`disparity_` | <code>list[float]</code> | Per-subject sum of squared differences from the template after alignment.
`scale_` | <code>list[float]</code> | Per-subject scale factors.

<details class="note" open markdown="1">
<summary>Note</summary>

`parallel='cpu'` (the default of `fit` and `transform`) runs the
per-subject Procrustes fits with joblib; it pays off with 3+ subjects,
many voxels (>10K), or several iterations. Use `parallel=None` for
debugging or small problems.

</details>

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Fit hyperalignment model to data.
[`transform`](#algorithms-transform) | Transform data to the common space using the fitted transformations.
[`transform_subject`](#algorithms-transform-subject) | Align a new subject to the common space.

**Examples:**

```python
import numpy as np
from nltools.algorithms import HyperAlignment

data = [np.random.randn(100, 50) for _ in range(3)]  # 3 subjects

hyper = HyperAlignment(n_iter=2)
hyper.fit(data, parallel="cpu", n_jobs=-1)
aligned = hyper.transform(data)  # list of arrays in the common space
template = hyper.s_  # or hyper.common_model_

# Align a new subject to the fitted template
new_subject = np.random.randn(100, 50)
transformed, R, disparity, scale = hyper.transform_subject(new_subject)
```

#### Methods

##### `fit`

```python
fit(data: list[np.ndarray], *, parallel: str | None = 'cpu', n_jobs: int = -1) -> HyperAlignment
```

Fit hyperalignment model to data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>list[ndarray]</code> | Data matrices, each of shape (n_features, n_samples). Subjects may differ in `n_features` when `auto_pad=True`. | *required*
`parallel` | <code>str \| None</code> | `'cpu'` (default) aligns subjects in parallel with joblib; None runs single-threaded NumPy. | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of CPU workers when `parallel='cpu'`; -1 (default) picks a count from available memory. | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
<code>[HyperAlignment](#tasks-alignment-hyperalignment)</code> | Fitted model (`self`).

##### `transform`

```python
transform(data: list[np.ndarray], *, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray]
```

Transform data to the common space using the fitted transformations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>list[ndarray]</code> | Data matrices to transform, one per fitted subject in the same order as `fit` (the same data or data of compatible shape). | *required*
`parallel` | <code>str \| None</code> | `'cpu'` (default) transforms subjects in parallel with joblib; None falls back to the setting used in `fit`. | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of CPU workers when parallel; -1 (default) reuses the value from `fit`, itself resolved from available memory. | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
<code>list[ndarray]</code> | Transformed data matrices in the common space.

##### `transform_subject`

```python
transform_subject(subject_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]
```

Align a new subject to the common space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`subject_data` | <code>ndarray</code> | Data from a new subject, shape (n_features, n_samples), to align to the common template. | *required*

**Returns:**

Type | Description
---- | -----------
<code>tuple[ndarray, ndarray, float, float]</code> | `(transformed, R, disparity,     scale)` — aligned data in common space, the transformation matrix     used, the alignment quality (sum of squared differences), and the     scale factor used.

(algorithms-localalignment)=
### `LocalAlignment`

```python
LocalAlignment(spatial_scale: str = 'searchlight', method: str = 'procrustes', radius: float = 10.0, roi_mask: nib.Nifti1Image | None = None, n_features: int | None = None, n_iter: int = 3, aggregation: str = 'center', parallel: str | None = 'cpu', n_jobs: int = -1, progress_bar: bool = False, n_neighborhoods_batch: int | None = None, memory_budget_gb: float | None = None, transforms_: dict[int, list[np.ndarray]] | None = None, template_: dict[int, np.ndarray] | None = None, neighborhoods_: SphereNeighborhoods | dict[int, np.ndarray] | None = None, n_voxels_: int | None = None, mask_: nib.Nifti1Image | None = None, backend_: Backend | None = None)
```

Local (neighborhood-based) functional alignment across subjects.

Learns alignment transforms within local neighborhoods (searchlight spheres
or parcels) and applies center-only aggregation to preserve orthogonality.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`spatial_scale` | <code>str</code> | Spatial scale, either 'searchlight' (overlapping spheres) or 'roi' (non-overlapping parcels). Defaults to 'searchlight'. | <code>'searchlight'</code>
`method` | <code>str</code> | Alignment method, one of 'procrustes', 'srm', or 'hyperalignment'. Defaults to 'procrustes'. | <code>'procrustes'</code>
`radius` | <code>float</code> | Sphere radius in millimeters for the searchlight scale. Defaults to 10.0. | <code>10.0</code>
`roi_mask` | <code>Nifti1Image \| None</code> | Parcellation image for the ROI scale. Required if `spatial_scale='roi'`. Defaults to None. | <code>None</code>
`n_features` | <code>int \| None</code> | Number of SRM features per neighborhood. None uses `min(n_local_voxels, n_samples)`; ignored by the other methods. Defaults to None. | <code>None</code>
`n_iter` | <code>int</code> | Number of iterations for alignment refinement. Defaults to 3. | <code>3</code>
`aggregation` | <code>str</code> | `'center'` writes only each sphere's center voxel (preserves orthogonality); `'all'` writes every voxel in the region and is selected automatically for `spatial_scale='roi'`. Defaults to `'center'`. | <code>'center'</code>
`parallel` | <code>str \| None</code> | Parallelization mode. None runs single-threaded numpy, 'cpu' uses joblib CPU parallelization, and 'gpu' uses PyTorch. GPU acceleration applies only to `method='procrustes'`; requesting 'gpu' with the 'srm' or 'hyperalignment' methods raises `NotImplementedError` (an explicit GPU request never silently runs on CPU). Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of jobs for CPU parallelization. Defaults to -1. | <code>-1</code>
`progress_bar` | <code>bool</code> | Whether to display tqdm progress bars during fit and transform. Defaults to False. | <code>False</code>
`n_neighborhoods_batch` | <code>int \| None</code> | Number of neighborhoods to process per batch on the GPU. None auto-calculates a batch size from `memory_budget_gb`. Defaults to None. | <code>None</code>
`memory_budget_gb` | <code>float \| None</code> | Explicit memory budget (in GB) used to auto-size GPU batches when `n_neighborhoods_batch` is None. None (default) measures the device's available memory. | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`transforms_` | <code>dict[int, list[ndarray]]</code> | Per-neighborhood transforms. Keys are center voxel indices (searchlight) or parcel ids (roi); values are lists of transform matrices, one per subject.
`template_` | <code>dict[int, ndarray]</code> | Per-neighborhood templates used for alignment.
`neighborhoods_` | <code>[SphereNeighborhoods](#neighborhoods-sphereneighborhoods) \| RoiNeighborhoods</code> | Computed neighborhoods (searchlight spheres or parcels).
`n_voxels_` | <code>int</code> | Total number of voxels in the mask.
`mask_` | <code>Nifti1Image</code> | Brain mask used for fitting.
`backend_` | <code>[Backend](#backends-backend)</code> | Execution backend selected from `parallel`.

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Fit local alignment on multi-subject data.
[`fit_transform`](#algorithms-fit-transform) | Fit alignment and transform data in one step.
[`transform`](#algorithms-transform) | Apply local transforms to data.



**Examples:**

```python
import numpy as np
import nibabel as nib
from nltools.algorithms.alignment import LocalAlignment

# Synthetic multi-subject data (voxels, samples) and a matching 1000-voxel mask
data = [np.random.randn(1000, 100) for _ in range(5)]
mask = nib.Nifti1Image(np.ones((10, 10, 10), dtype=np.int8), np.eye(4))

la = LocalAlignment(spatial_scale="searchlight", method="procrustes", radius=10.0)
la.fit(data, mask)
aligned = la.transform(data)  # list of (1000, 100) arrays
```

<details class="note" open markdown="1">
<summary>Note</summary>

Based on Bazeille et al. 2021, "An empirical evaluation of functional
alignment using inter-subject decoding". Center-only aggregation
preserves the local orthogonality of the transforms.

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
`data` | <code>list[ndarray]</code> | Subject data arrays, each of shape (n_voxels, n_samples). Subjects may differ in the number of samples; shorter subjects are zero-padded within each neighborhood. | *required*
`mask` | <code>Nifti1Image</code> | Brain mask defining the voxel space. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[LocalAlignment](#tasks-alignment-localalignment)</code> | The fitted alignment model (`self`).

(algorithms-fit-transform)=
##### `fit_transform`

```python
fit_transform(data: list[np.ndarray], mask: nib.Nifti1Image) -> list[np.ndarray]
```

Fit alignment and transform data in one step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>list[ndarray]</code> | Subject data arrays, each of shape (n_voxels, n_samples). | *required*
`mask` | <code>Nifti1Image</code> | Brain mask defining the voxel space. | *required*

**Returns:**

Type | Description
---- | -----------
<code>list[ndarray]</code> | Aligned data for each subject.

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
`data` | <code>list[ndarray]</code> | List of subject data arrays, each shape (n_voxels, n_samples). | *required*

**Returns:**

Type | Description
---- | -----------
<code>list[ndarray]</code> | Aligned data for each subject, each shape     (n_voxels, n_samples).

(algorithms-srm)=
### `SRM`

```python
SRM(*, n_iter: int = 10, n_features: int = 50, random_state: int = 0)
```

Bases: `sklearn.base.BaseEstimator`, `sklearn.base.TransformerMixin`

Probabilistic Shared Response Model (SRM).

Factorizes multi-subject data as a shared response S plus one orthogonal
transform W per subject, so that for every subject i

$$
X_i \approx W_i S, \forall i=1 \dots N
$$

The model is fit by the expectation-maximization algorithm of Chen et al.
(2015) with the optimizations of Anderson et al. (2016). Subjects may have
different numbers of voxels; they must have the same number of samples
unless `fit(pad_samples=True)` zero-pads the shorter ones. Run time is
$O(I (V T K + V K^2 + K^3))$ and memory $O(V T)$, with I iterations, V the
sum of voxels across subjects, T samples, and K features (typically
$V \gg T \gg K$).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_iter` | <code>int</code> | Number of EM iterations. Defaults to 10. | <code>10</code>
`n_features` | <code>int</code> | Number of shared features to compute. Defaults to 50. | <code>50</code>
`random_state` | <code>int</code> | Seed for the random initialization. Defaults to 0. | <code>0</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`w_` | <code>list[ndarray]</code> | Per-subject orthogonal transforms, element i of shape (voxels_i, n_features).
`s_` | <code>ndarray</code> | The shared response, shape (n_features, samples).
`sigma_s_` | <code>ndarray</code> | Covariance of the shared response's Normal distribution, shape (n_features, n_features).
`mu_` | <code>list[ndarray]</code> | Per-subject voxel means over samples, element i of shape (voxels_i,).
`rho2_` | <code>ndarray</code> | Estimated noise variance $\rho_i^2$ per subject, shape (subjects,).
`random_state_` | <code>RandomState</code> | Generator seeded from `random_state`.

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Compute the probabilistic Shared Response Model.
[`transform`](#algorithms-transform) | Project each subject's data into the shared response space.
[`transform_subject`](#algorithms-transform-subject) | Transform a new subject using the existing model.



**Examples:**

```python
import numpy as np
from nltools.algorithms import SRM

data = [np.random.randn(100, 50) for _ in range(3)]  # 3 subjects

srm = SRM(n_iter=10, n_features=50)
srm.fit(data, parallel="cpu", n_jobs=-1)
shared_responses = srm.transform(data)  # list of (50, 50) arrays

w = srm.w_  # subject-specific transforms
s = srm.s_  # shared response
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
`X` | <code>list[ndarray]</code> | One (voxels_i, samples) array per subject. Subjects may differ in the number of samples when `pad_samples=True`. | *required*
`y` | <code>Any \| None</code> | Ignored; present for scikit-learn compatibility. | <code>None</code>
`parallel` | <code>str \| None</code> | `'cpu'` (default) updates subjects in parallel with joblib; None runs single-threaded NumPy; `'gpu'` raises `NotImplementedError` (never a silent CPU fallback). | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of CPU workers when `parallel='cpu'`; -1 (default) picks a count from available memory. | <code>-1</code>
`pad_samples` | <code>bool</code> | If True (default), zero-pad subjects with fewer samples up to the longest subject; if False, unequal sample counts raise `ValueError`. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[SRM](#tasks-alignment-srm)</code> | Fitted model (`self`).

##### `transform`

```python
transform(X: list[np.ndarray], y: Any | None = None, *, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray | None]
```

Project each subject's data into the shared response space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list[ndarray \| None]</code> | One (voxels_i, samples_i) array per fitted subject, in the same order as `fit`; voxel and sample counts may vary across subjects. A None entry yields None. | *required*
`y` | <code>Any \| None</code> | Ignored; present for scikit-learn compatibility. | <code>None</code>
`parallel` | <code>str \| None</code> | `'cpu'` (default) transforms subjects in parallel with joblib; None runs single-threaded NumPy; `'gpu'` raises `NotImplementedError`. | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of CPU workers when `parallel='cpu'`; -1 (default) reuses the value from `fit`, itself resolved from available memory. | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
<code>list[ndarray \| None]</code> | Shared responses, element i of shape     (n_features, samples_i).

##### `transform_subject`

```python
transform_subject(X: np.ndarray) -> np.ndarray
```

Transform a new subject using the existing model.

The subject is assumed to have received equivalent stimulation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray</code> | The new subject's data, shape (voxels, timepoints); the timepoints must match the fitted shared response. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Orthogonal mapping $W_{new}$ for the new subject, shape     (voxels, n_features).

## Functions

(algorithms-align)=
### `align`

```python
align(data, method = 'deterministic_srm', n_features = None, axis = 0, *, n_iter = 10, random_state = 0)
```

Align subject data into a common response model.

A convenience wrapper around the `HyperAlignment` and `SRM`/`DetSRM` classes.
Aligns a group of subjects either by Procrustes-based hyperalignment
(Haxby et al., 2011) or by the Shared Response Model (Chen et al., 2015).
The common model is the shared response (SRM) or the centered group template
(Procrustes). Transformed data can be projected back into each subject's
original space with its transformation matrix. To align a single `BrainData`
to another, use `BrainData.align` instead.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>list[[BrainData](#page-data-brain-data)] \| list[ndarray]</code> | Subjects to align; all elements must be the same type. Arrays are observations x features. | *required*
`method` | <code>str</code> | One of `'probabilistic_srm'`, `'deterministic_srm'`, or `'procrustes'`. Defaults to `'deterministic_srm'`. | <code>'deterministic_srm'</code>
`n_features` | <code>int \| None</code> | Number of features in the common space (SRM only). None uses the number of voxels. Must be None for `'procrustes'`. | <code>None</code>
`axis` | <code>int</code> | Axis to align on: 0 aligns timepoints (ISC computed per voxel), 1 aligns voxels (ISC computed per timepoint). Defaults to 0. | <code>0</code>
`n_iter` | <code>int</code> | Number of `SRM`/`DetSRM` iterations; ignored by `method='procrustes'`. Defaults to 10. | <code>10</code>
`random_state` | <code>int</code> | Seed forwarded to the constructed `SRM`/`DetSRM`; ignored by `method='procrustes'`. Defaults to 0. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'transformed'` (list of aligned subject data, same type as the     input), `'transformation_matrix'` (per-subject transforms),     `'common_model'` (shared response or group template), and `'isc'`     (dict mapping each aligned unit to its mean intersubject correlation).     With `method='procrustes'` also `'disparity'` and `'scale'`.

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
`reference` | <code>ndarray</code> | Reference pattern x state matrix. | *required*
`target` | <code>ndarray</code> | Target pattern x state matrix to align to `reference`; must have the same shape. | *required*
`metric` | <code>str</code> | Distance metric passed to `sklearn.metrics.pairwise_distances`. Defaults to `'correlation'`. | <code>'correlation'</code>
`return_index` | <code>bool</code> | If True return the remapping index instead of the reordered data. Defaults to False. | <code>False</code>
`replace_zero_variance` | <code>bool</code> | Replace zero-variance columns with uniform random numbers before computing distances; avoids NaNs with the correlation metric. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | If `return_index=False` (default), `target[:, remapping]` — the     target's columns reordered to match the reference, oriented pattern x     state (same shape as `target`). If `return_index=True`, the remapping     index array that reorders the target's state columns.

(algorithms-calc-bpm)=
### `calc_bpm`

```python
calc_bpm(beat_interval, sampling_freq)
```

Calculate instantaneous BPM from beat to beat interval.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`beat_interval` | <code>int</code> | Number of samples between beats (typically the R-R interval). | *required*
`sampling_freq` | <code>float</code> | Sampling frequency in Hz. | *required*

**Returns:**

Type | Description
---- | -----------
<code>float</code> | Beats per minute for the time interval.

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
`data` | <code>ndarray</code> | Time series, shape (n_samples,) or (n_samples, n_features). | *required*
`shift_amount` | <code>int \| ndarray \| None</code> | Shift amount: an int for 1D data, or an array of length n_features (one shift per column) for 2D data. None draws random shift(s). Defaults to None. | <code>None</code>
`random_state` | <code>int \| RandomState \| None</code> | Random seed used when `shift_amount` is None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Circularly shifted data with the same shape as the input.

**Examples:**

```python
x = np.array([1, 2, 3, 4, 5])
circle_shift(x, shift_amount=2)  # → array([4, 5, 1, 2, 3])

X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])
circle_shift(X, shift_amount=np.array([1, 2]))
# → array([[ 4, 30],
#          [ 1, 40],
#          [ 2, 10],
#          [ 3, 20]])
```

(algorithms-compute-multivariate-similarity)=
### `compute_multivariate_similarity`

```python
compute_multivariate_similarity(y, X, method = 'ols', tail = 2)
```

Compute multivariate similarity by regressing one pattern on several.

The array engine behind `BrainData.multivariate_similarity`: predicts the
spatial pattern `y` from a linear combination of the columns of `X` and
returns the OLS coefficients, t-statistics, p-values, and residuals.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`y` | <code>ndarray</code> | Target pattern, shape (n_features,). | *required*
`X` | <code>ndarray</code> | Predictor patterns, shape (n_features, n_predictors) (the transpose is accepted). An intercept column is always prepended, so do not include one. | *required*
`method` | <code>str</code> | Regression method; only 'ols' is implemented. Defaults to 'ols'. | <code>'ols'</code>
`tail` | <code>int</code> | 2 for two-sided p-values, 1 for an upper-tail test. Defaults to 2. | <code>2</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys 'beta' (coefficients, intercept first, shape (n_predictors + 1,)),     't' (t-statistics, same shape), 'p' (p-values, same shape), 'df'     (residual degrees of freedom), 'sigma' (residual standard deviation),     and 'residual' (residuals, shape (n_features,)).

**Examples:**

```python
y = np.random.randn(100)
X = np.random.randn(100, 5)
result = compute_multivariate_similarity(y, X, method="ols")
result["beta"].shape  # → (6,)  5 predictors + intercept
```

(algorithms-compute-similarity)=
### `compute_similarity`

```python
compute_similarity(data1, data2, metric = 'correlation')
```

Compute row-wise similarity between two data arrays.

The array engine behind `BrainData.similarity`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>ndarray</code> | First data array, shape (n_samples1, n_features). | *required*
`data2` | <code>ndarray</code> | Second data array, shape (n_samples2, n_features). | *required*
`metric` | <code>str</code> | 'correlation' (or 'pearson'), 'spearman' (or 'rank_correlation'), 'dot_product', or 'cosine'. Defaults to 'correlation'. | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Similarities of shape (n_samples1, n_samples2), squeezed: a 1D     array when either input has a single row, a scalar when both do.

**Examples:**

```python
data1 = np.random.randn(10, 100)
data2 = np.random.randn(5, 100)
sim = compute_similarity(data1, data2, metric="correlation")
sim.shape  # → (10, 5)
```

(algorithms-correlation-permutation-test)=
### `correlation_permutation_test`

```python
correlation_permutation_test(data1: np.ndarray, data2: np.ndarray, *, n_permute: int = 5000, metric: str = 'pearson', tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Permutation test for whether the correlation between two arrays differs from zero.

Builds the null distribution by randomly permuting the observations of `data1`
and re-correlating with `data2`. Assumes observations are independent (i.i.d.);
for autocorrelated time series use `timeseries_correlation_permutation_test`,
whose `'circle_shift'` and `'phase_randomize'` methods preserve temporal
structure. With 2D inputs each column of `data1` is tested against the
matching column of `data2`, independently.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>ndarray</code> | Data to permute, shape (n_samples,) for a single feature or (n_samples, n_features) for several. | *required*
`data2` | <code>ndarray</code> | Data to correlate with, same shape as `data1`. | *required*
`n_permute` | <code>int</code> | Number of permutations. Defaults to 5000. | <code>5000</code>
`metric` | <code>str</code> | 'pearson' (linear), 'spearman' (rank-based, monotonic), or 'kendall' (tau-b, ordinal association, tie-corrected). Defaults to 'pearson'. | <code>'pearson'</code>
`tail` | <code>int \| str</code> | `2` or `'two'` for a two-tailed test (r != 0); `1` or `'one'` for a one-tailed test of r > 0 (negate one variable for the other direction; the fixed direction keeps multiple-comparison correction valid). Defaults to 2. | <code>2</code>
`return_null` | <code>bool</code> | Also return the full null distribution. Defaults to False. | <code>False</code>
`device` | <code>str \| None</code> | Execution path. `'cpu'` parallelizes permutations across `n_jobs` joblib workers (4-8× speedup); `'gpu'` vectorizes them with PyTorch in memory-bounded batches (fastest for large problems; Pearson and Spearman run 5-20× faster on multi-feature data, Kendall needs O(n²) memory per permutation so its batches are smaller); `None` runs single-threaded NumPy (for debugging or small problems). Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of CPU workers, -1 = all cores; only used when `device='cpu'`. Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>float \| None</code> | GPU memory budget in GB that sizes the permutation batches; only used when `device='gpu'`. None (default) measures the device's available memory. Larger values fit more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | Show a progress bar over permutations. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys 'correlation' (float, or np.ndarray of shape (n_features,) for     2D inputs: the observed correlation), 'p' (float or np.ndarray, the     matching p-values), 'device' (the execution path used: `'cpu'`,     `'gpu'`, or `None`), and 'null_dist' (np.ndarray of shape     (n_permute,) or (n_permute, n_features)) when `return_null=True`.

**Examples:**

```python
import numpy as np
from nltools.algorithms import correlation_permutation_test

# Single feature (default: CPU parallel)
x = np.random.randn(100)
y = x + np.random.randn(100) * 0.5
result = correlation_permutation_test(x, y, n_permute=5000)
result["correlation"]  # → 0.85 (approximately)
result["p"]  # → 0.0002

# Multi-feature: each column pair tested independently
data1 = np.random.randn(100, 10)
data2 = data1 + np.random.randn(100, 10) * 0.3
result = correlation_permutation_test(data1, data2, n_permute=5000)
result["correlation"].shape  # → (10,)
result["p"].shape  # → (10,)

# GPU acceleration
result = correlation_permutation_test(data1, data2, n_permute=5000, device="gpu")
```

<details class="note" open markdown="1">
<summary>Note</summary>

Kendall's tau is O(n²) in the number of samples on every path, so it is
markedly slower than Pearson or Spearman for large samples.

</details>

(algorithms-distance-correlation)=
### `distance_correlation`

```python
distance_correlation(x: np.ndarray, y: np.ndarray, bias_corrected: bool = True, ttest: bool = False) -> dict
```

Compute the distance correlation between two arrays to test for multivariate dependence.

Distance correlation detects linear and non-linear dependence. The arrays must
match on their first dimension. Prefer the bias-corrected version (the default),
which can also perform a t-test; that test operates on a statistic that is
approximately the squared distance correlation, which is also returned.

Distance correlation is the normalized covariance of two centered Euclidean
distance matrices. Each distance matrix holds the distances between rows (if x
or y is 2d) or scalars (if 1d). Each matrix is centered before the covariance
is computed, either by double-centering or by U-centering, which corrects the
bias that grows with the number of dimensions. U-centering is almost always
preferable and also permits a one-tailed directional t-test on the normalized
covariance (Szekely & Rizzo, 2013). Distance correlation is normally bounded
between 0 and 1, but U-centering can produce negative estimates, which are
never significant.

Validated against `dcor` and `dcor.ttest` in the R package *energy* and
`dcor.distance_correlation`, `dcor.u_distance_correlation_sqr`, and
`dcor.independence.distance_correlation_t_test` in the Python package *dcor*.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` | <code>ndarray</code> | 1d or 2d array of observations by features. | *required*
`y` | <code>ndarray</code> | 1d or 2d array of observations by features. | *required*
`bias_corrected` | <code>bool</code> | If True, U-center the distance matrices; if False, double-center them, which gives a biased estimate that converges to 1 as the number of dimensions grows. Must be True when `ttest=True`. Defaults to True. | <code>True</code>
`ttest` | <code>bool</code> | Perform a t-test on the bias-corrected distance correlation. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Key 'dcorr' (float, distance correlation); with `bias_corrected=True`     also 'dcorr_squared' (float, the U-centered statistic, which can be     negative); with `ttest=True` also 't', 'p', and 'df'.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If arrays are not 1d or 2d, or if `ttest=True` and `bias_corrected=False`.

**Examples:**

```python
import numpy as np

x = np.random.randn(20, 3)
y = x + np.random.randn(20, 3) * 0.1  # strongly dependent
result = distance_correlation(x, y, bias_corrected=True)
"dcorr" in result  # → True
0 <= result["dcorr"] <= 1  # → True
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
`mat` | <code>ndarray</code> | 2d numpy array. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Double-centered version of the input.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If input is not 2D.

**Examples:**

```python
mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
result = double_center(mat)
np.allclose(result.mean(axis=0), 0)  # → True
np.allclose(result.mean(axis=1), 0)  # → True
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
`data` | <code>DataFrame \| Series</code> | Data to downsample. | *required*
`sampling_freq` | <code>float</code> | Sampling frequency of the data in Hz. | <code>None</code>
`target` | <code>float</code> | Downsampling target. | <code>None</code>
`target_type` | <code>str</code> | Unit of `target`, one of 'samples', 'seconds', or 'hz'. Defaults to 'samples'. | <code>'samples'</code>
`method` | <code>str</code> | Aggregation within each bin, 'mean' or 'median'. Defaults to 'mean'. | <code>'mean'</code>

**Returns:**

Type | Description
---- | -----------
<code>DataFrame \| Series</code> | Downsampled data (same type as input).

(algorithms-fdr)=
### `fdr`

```python
fdr(p, q = 0.05)
```

Determine an FDR threshold for an array of p-values.

Benjamini-Hochberg procedure at false discovery rate `q` (valid under
independence or positive dependence). Written by Tal Yarkoni.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`p` | <code>ndarray</code> | Vector of p-values. | *required*
`q` | <code>float</code> | False discovery rate level. Defaults to 0.05. | <code>0.05</code>

**Returns:**

Type | Description
---- | -----------
<code>float</code> | The p-value threshold; `-1` when no p-value survives correction.

(algorithms-find-spikes)=
### `find_spikes`

```python
find_spikes(data, global_spike_cutoff = 3, diff_spike_cutoff = 3, *, TR: float | None = None, sampling_freq: float | None = None)
```

Identify spikes (motion artifacts, intensity outliers) in 4D fMRI data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image</code> | 4D functional data. | *required*
`global_spike_cutoff` | <code>float \| None</code> | Cutoff in standard deviations for spikes in the per-TR global mean signal; None skips this detector. Defaults to 3. | <code>3</code>
`diff_spike_cutoff` | <code>float \| None</code> | Cutoff in standard deviations for spikes in the per-TR mean absolute frame-to-frame difference; None skips this detector. Defaults to 3. | <code>3</code>
`TR` | <code>float \| None</code> | Repetition time in seconds; sets the returned DesignMatrix's `sampling_freq` for downstream `.append()` / `.convolve()`. Pass at most one of `TR` and `sampling_freq`. | <code>None</code>
`sampling_freq` | <code>float \| None</code> | Sampling frequency in Hz (1 / TR). See `TR`. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[DesignMatrix](#page-data-design-matrix)</code> | One indicator column per detected spike TR, named     `.nl_global_spike{n}` / `.nl_diff_spike{n}` in the reserved namespace     for generated columns (see `RESERVED_PREFIX`) and pre-marked as     confounds. Row position is the time axis. A volume flagged by both     detectors yields identical one-hot columns, so only the     `.nl_global_spike*` one is kept. Without `TR` / `sampling_freq` the     result has `sampling_freq=None` and can still be appended to a     DesignMatrix that has one.

(algorithms-fisher-r-to-z)=
### `fisher_r_to_z`

```python
fisher_r_to_z(r)
```

Convert correlation coefficients to Fisher z values.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`r` | <code>float \| ndarray</code> | Correlation coefficient(s). | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Fisher z-transformed correlation(s).

(algorithms-fisher-z-to-r)=
### `fisher_z_to_r`

```python
fisher_z_to_r(z)
```

Convert Fisher z back to a correlation coefficient.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`z` | <code>float \| ndarray</code> | Fisher z value(s). | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Correlation coefficient(s).

(algorithms-glover-dispersion-derivative)=
### `glover_dispersion_derivative`

```python
glover_dispersion_derivative(t_r, *, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Sample the dispersion derivative of the Glover hemodynamic response function.

Thin wrapper over nilearn's `glover_dispersion_derivative`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>float</code> | Repetition time in seconds. | *required*
`oversampling` | <code>int</code> | Temporal oversampling factor (default: 50). | <code>50</code>
`time_length` | <code>float</code> | HRF kernel length in seconds (default: 32.0). | <code>32.0</code>
`onset` | <code>float</code> | Onset of the response in seconds (default: 0.0). | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | The dispersion derivative sampled every `t_r / oversampling` seconds.

(algorithms-glover-hrf)=
### `glover_hrf`

```python
glover_hrf(t_r, *, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Sample the Glover hemodynamic response function.

Thin wrapper over nilearn's `glover_hrf`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>float</code> | Repetition time in seconds. | *required*
`oversampling` | <code>int</code> | Temporal oversampling factor (default: 50). | <code>50</code>
`time_length` | <code>float</code> | HRF kernel length in seconds (default: 32.0). | <code>32.0</code>
`onset` | <code>float</code> | Onset of the response in seconds (default: 0.0). | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | The HRF sampled every `t_r / oversampling` seconds, peak scaled to 1.

(algorithms-glover-time-derivative)=
### `glover_time_derivative`

```python
glover_time_derivative(t_r, *, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Sample the time derivative of the Glover hemodynamic response function.

Thin wrapper over nilearn's `glover_time_derivative`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>float</code> | Repetition time in seconds. | *required*
`oversampling` | <code>int</code> | Temporal oversampling factor (default: 50). | <code>50</code>
`time_length` | <code>float</code> | HRF kernel length in seconds (default: 32.0). | <code>32.0</code>
`onset` | <code>float</code> | Onset of the response in seconds (default: 0.0). | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | The time derivative sampled every `t_r / oversampling` seconds.

(algorithms-holm-bonf)=
### `holm_bonf`

```python
holm_bonf(p, alpha = 0.05)
```

Determine a Holm-Bonferroni (step-down) threshold for an array of p-values.

The step-down procedure applies progressively less correction to larger
p-values. It is more conservative than FDR but much more powerful than plain
Bonferroni correction.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`p` | <code>ndarray</code> | Vector of p-values. | *required*
`alpha` | <code>float</code> | Family-wise alpha level. Defaults to 0.05. | <code>0.05</code>

**Returns:**

Type | Description
---- | -----------
<code>float</code> | The p-value threshold; `-1` when no p-value survives correction.

(algorithms-isc)=
### `isc`

```python
isc(data, *, n_samples = 5000, summary = 'median', method = 'bootstrap', ci_percentile = 95, exclude_self_corr = True, tail = 2, metric = 'correlation', return_null = False, n_jobs = -1, random_state = None, progress_bar = False)
```

Compute pairwise intersubject correlation from an observations-by-subjects array.

Pairwise ISC is summarized with the median, as Chen et al. (2016) recommend;
`summary='mean'` instead averages after the Fisher r-to-z transform and
converts back, which avoids inflating the estimate.

Three null distributions are available. The default subject-wise bootstrap
(Chen et al., 2016) resamples subjects with replacement and recomputes the
pairwise similarity matrix; a subject drawn twice correlates perfectly with
itself, so those entries are set to NaN when `exclude_self_corr=True`.
P-values use the percentile method, as in Brainiak. The classic surrogate
methods instead circle-shift or phase-randomize each time series (Lancaster
et al., 2018), preserving its temporal autocorrelation, and recompute ISC.

Runs on plain arrays with observations aligned across subjects.
`isc_permutation_test` exposes the same engine with `device='gpu'` and
leave-one-out ISC.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>ndarray \| DataFrame \| DataFrame</code> | Observations by subjects; ISC is computed across the columns. | *required*
`n_samples` | <code>int</code> | Number of bootstrap draws or surrogate permutations. Defaults to 5000. | <code>5000</code>
`summary` | <code>str</code> | `'median'` (default) or `'mean'`. | <code>'median'</code>
`method` | <code>str</code> | `'bootstrap'` (default), `'circle_shift'`, or `'phase_randomize'`. | <code>'bootstrap'</code>
`ci_percentile` | <code>int</code> | Confidence-interval width in percent. Defaults to 95. | <code>95</code>
`exclude_self_corr` | <code>bool</code> | Set self-correlations (the same subject bootstrapped twice) to NaN. Defaults to True. | <code>True</code>
`tail` | <code>int \| str</code> | `2` or `'two'` (two-tailed, default) or `1` or `'one'` (one-tailed, ISC > 0). | <code>2</code>
`metric` | <code>str</code> | Pairwise similarity metric; any metric accepted by sklearn's `pairwise_distances`. Defaults to `'correlation'`. | <code>'correlation'</code>
`return_null` | <code>bool</code> | Include the null distribution in the result. Defaults to False. | <code>False</code>
`n_jobs` | <code>int</code> | CPU workers for the resamples; -1 (default) picks the count from available memory. | <code>-1</code>
`random_state` | <code>int \| RandomState \| None</code> | Seed or generator for the resampling. | <code>None</code>
`progress_bar` | <code>bool</code> | Display a progress bar. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'isc'` (float, observed ISC), `'p'` (float), `'ci'` (tuple     `(lower, upper)`), `'device'`, and — when `return_null=True` —     `'null_dist'` (np.ndarray).

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

Hall, P., & Wilson, S. R. (1991). Two guidelines for bootstrap
hypothesis testing. Biometrics, 757-762.

Lancaster, G., Iatsenko, D., Pidde, A., Ticcinelli, V., & Stefanovska,
A. (2018). Surrogate data for hypothesis testing of physical systems.
Physics Reports, 748, 1-60.

</details>

(algorithms-isc-group)=
### `isc_group`

```python
isc_group(group1, group2, *, n_samples = 5000, summary = 'median', method = 'permute', ci_percentile = 95, exclude_self_corr = True, return_null = False, tail = 2, metric = 'correlation', n_jobs = -1, random_state = None, progress_bar = False)
```

Test the difference in pairwise intersubject correlation between two groups.

ISC within each group is summarized with the median, as Chen et al. (2016)
recommend (`summary='mean'` averages after the Fisher r-to-z transform), and
the observed statistic is `group1 - group2`.

Two null distributions are available. The default subject-wise permutation
(Chen et al., 2016) pools the subjects, computes pairwise similarity within
and between groups, then reshuffles the group labels and recomputes the
difference. The subject-wise bootstrap instead resamples subjects with
replacement within each group; a subject drawn twice correlates perfectly
with itself, so those entries are set to NaN when `exclude_self_corr=True`.
P-values use the percentile method (Hall & Wilson, 1991).

Runs on plain arrays; `isc_group_permutation_test` exposes the same engine
with `device='gpu'` and leave-one-out ISC.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`group1` | <code>ndarray \| DataFrame \| DataFrame</code> | Observations by subjects for the first group. | *required*
`group2` | <code>ndarray \| DataFrame \| DataFrame</code> | Observations by subjects for the second group (same number of observations). | *required*
`n_samples` | <code>int</code> | Number of permutations or bootstrap draws. Defaults to 5000. | <code>5000</code>
`summary` | <code>str</code> | `'median'` (default) or `'mean'`. | <code>'median'</code>
`method` | <code>str</code> | `'permute'` (default) or `'bootstrap'`. | <code>'permute'</code>
`ci_percentile` | <code>float</code> | Confidence-interval width in percent. Defaults to 95. | <code>95</code>
`exclude_self_corr` | <code>bool</code> | In the bootstrap, set self-correlations to NaN. Defaults to True. | <code>True</code>
`return_null` | <code>bool</code> | Include the null distribution in the result. Defaults to False. | <code>False</code>
`tail` | <code>int \| str</code> | `2` or `'two'` (two-tailed, default) or `1` or `'one'` (one-tailed, group1 > group2). | <code>2</code>
`metric` | <code>str</code> | Pairwise similarity metric; any metric accepted by sklearn's `pairwise_distances`. Defaults to `'correlation'`. | <code>'correlation'</code>
`n_jobs` | <code>int</code> | CPU workers for the resamples; -1 (default) picks the count from available memory. | <code>-1</code>
`random_state` | <code>int \| RandomState \| None</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | Display a progress bar. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'isc_group_difference'` (float, observed difference), `'p'`     (float), `'ci'` (tuple `(lower, upper)`), `'device'`, and — when     `return_null=True` — `'null_dist'` (np.ndarray).

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

Hall, P., & Wilson, S. R. (1991). Two guidelines for bootstrap
hypothesis testing. Biometrics, 757-762.

</details>

(algorithms-isc-group-permutation-test)=
### `isc_group_permutation_test`

```python
isc_group_permutation_test(group1: np.ndarray, group2: np.ndarray, *, n_permute: int = 5000, summary: Literal['median', 'mean'] = 'median', method: Literal['permute', 'bootstrap'] = 'permute', summary_statistic: Literal['leave-one-out', 'pairwise'] = 'pairwise', ci_percentile: float = 95, tail: int | str = 2, device: Literal['cpu', 'gpu'] | None = 'cpu', n_jobs: int = -1, random_state: int | None = None, return_null: bool = False, progress_bar: bool = False, exclude_self_corr: bool = True, metric: str = 'correlation') -> dict[str, Any]
```

Test the difference in intersubject correlation between two groups.

Computes ISC within each group, takes `group1 - group2`, and builds a null
distribution by either subject-wise permutation (pool the subjects and
reshuffle the group labels — the Chen et al. 2016 recommendation) or
subject-wise bootstrap (resample subjects within each group; the bootstrap
draws are centered on the observed difference before the p-value is
computed). The confidence interval brackets the observed difference for
`method='bootstrap'` and describes the null spread for `method='permute'`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`group1` | <code>ndarray</code> | First group, shape `(n_observations, n_subjects1)` for a single feature or `(n_observations, n_subjects1, n_voxels)` for voxel-wise data. | *required*
`group2` | <code>ndarray</code> | Second group, shape `(n_observations, n_subjects2)` or `(n_observations, n_subjects2, n_voxels)`; `n_observations` must match `group1`. | *required*
`n_permute` | <code>int</code> | Number of permutations or bootstrap draws. Defaults to 5000. | <code>5000</code>
`summary` | <code>str</code> | How ISC values are aggregated: `'median'` (default, robust to outliers) or `'mean'` (Fisher z-transformed mean). | <code>'median'</code>
`method` | <code>str</code> | `'permute'` (default; pool subjects and permute labels) or `'bootstrap'` (resample subjects within each group). | <code>'permute'</code>
`summary_statistic` | <code>str</code> | `'pairwise'` (default; summarize all pairwise correlations) or `'leave-one-out'` (correlate each subject with the mean of the others). | <code>'pairwise'</code>
`ci_percentile` | <code>float</code> | Confidence-interval width in percent (95 gives a 95% CI). Defaults to 95. | <code>95</code>
`tail` | <code>int \| str</code> | `2` or `'two'` (default) for a two-tailed p-value; `1` or `'one'` for one-tailed (group1 > group2). | <code>2</code>
`device` | <code>str \| None</code> | Execution path. `'cpu'` (default) parallelizes the resamples with joblib; `'gpu'` computes the observed voxel-wise ISC through PyTorch (10-30× speedup; the resamples still run on the CPU); None runs single-threaded numpy. | <code>'cpu'</code>
`n_jobs` | <code>int</code> | CPU cores for the resamples when `device` is not None. -1 (default) picks the worker count from available memory. | <code>-1</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>bool</code> | If True, include the null distribution in the result. Defaults to False. | <code>False</code>
`progress_bar` | <code>bool</code> | Show a progress bar over the resamples. Defaults to False. | <code>False</code>
`exclude_self_corr` | <code>bool</code> | In the bootstrap, mask the perfect correlations a duplicated subject produces (pairwise only). Defaults to True. | <code>True</code>
`metric` | <code>str</code> | Similarity metric for pairwise ISC; any metric accepted by `sklearn.metrics.pairwise_distances`. Ignored for `summary_statistic='leave-one-out'`. Defaults to `'correlation'`. | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'isc_group_difference'` (float or np.ndarray, observed     difference), `'p'` (float or np.ndarray, p-value with the     `(count + 1) / (n + 1)` correction), `'ci'` (tuple     `(lower, upper)`), `'device'` (the execution path used), and — when     `return_null=True` — `'null_dist'` (np.ndarray).

**Examples:**

```python
# Single-feature comparison
group1 = np.random.randn(100, 10)  # 10 subjects
group2 = np.random.randn(100, 10)
result = isc_group_permutation_test(group1, group2, n_permute=1000)
result["isc_group_difference"], result["p"]

# Voxel-wise comparison, observed ISC on the GPU
group1_voxels = np.random.randn(100, 10, 5000)  # 5K voxels
group2_voxels = np.random.randn(100, 10, 5000)
result = isc_group_permutation_test(
    group1_voxels,
    group2_voxels,
    summary_statistic="leave-one-out",
    device="gpu",
    n_permute=5000,
)
(result["p"] < 0.05).sum()  # → number of significant voxels
```

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

(algorithms-isc-permutation-test)=
### `isc_permutation_test`

```python
isc_permutation_test(data: np.ndarray, *, n_permute: int = 5000, summary: Literal['median', 'mean'] = 'median', summary_statistic: Literal['leave-one-out', 'pairwise'] = 'pairwise', method: Literal['bootstrap', 'circle_shift', 'phase_randomize'] = 'bootstrap', ci_percentile: float = 95, tail: int | str = 2, return_null: bool = False, progress_bar: bool = False, exclude_self_corr: bool = True, metric: str = 'correlation', device: Literal['cpu', 'gpu'] | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None) -> dict[str, Any]
```

Compute intersubject correlation with bootstrap or permutation inference.

Summarizes how similarly subjects respond over time — either leave-one-out
(each subject against the mean of the others; O(n_subjects), unbiased) or
pairwise (all subject pairs; O(n_subjects²), full correlation structure).
The two are monotonically but non-linearly related and statistically
different (Chen et al. 2016, Figure 3). The null distribution comes from a
subject-wise bootstrap (centered on the observed ISC, so the p-value tests
H0: ISC = 0) or from surrogate time series that preserve each subject's
autocorrelation (circular shift) or power spectrum (phase randomization).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>ndarray</code> | Shape `(n_observations, n_subjects)` for a single feature or `(n_observations, n_subjects, n_voxels)` for voxel-wise ISC. | *required*
`n_permute` | <code>int</code> | Number of bootstrap draws or permutations. Defaults to 5000. | <code>5000</code>
`summary` | <code>str</code> | How ISC values are aggregated: `'median'` (default, robust to outliers) or `'mean'` (Fisher z-transformed mean). | <code>'median'</code>
`summary_statistic` | <code>str</code> | `'pairwise'` (default) or `'leave-one-out'`. | <code>'pairwise'</code>
`method` | <code>str</code> | `'bootstrap'` (default; subject-wise bootstrap, Chen et al. 2016), `'circle_shift'` (circular time-series shift), or `'phase_randomize'` (FFT phase randomization). | <code>'bootstrap'</code>
`ci_percentile` | <code>float</code> | Confidence-interval width in percent (95 gives a 95% CI). Defaults to 95. | <code>95</code>
`tail` | <code>int \| str</code> | `2` or `'two'` (default) for a two-tailed p-value; `1` or `'one'` for one-tailed (ISC > 0). | <code>2</code>
`return_null` | <code>bool</code> | If True, include the null distribution in the result. Defaults to False. | <code>False</code>
`progress_bar` | <code>bool</code> | Show a progress bar over the resamples. Defaults to False. | <code>False</code>
`exclude_self_corr` | <code>bool</code> | In the pairwise bootstrap, mask the perfect correlations a duplicated subject produces as NaN. Defaults to True. | <code>True</code>
`metric` | <code>str</code> | Similarity metric for pairwise ISC; any metric accepted by `sklearn.metrics.pairwise_distances` (`'correlation'`, `'spearman'`, `'cosine'`, and `'euclidean'` take fast paths). Ignored for `summary_statistic='leave-one-out'`; the GPU pairwise path supports only `'correlation'`. Defaults to `'correlation'`. | <code>'correlation'</code>
`device` | <code>str \| None</code> | Execution path. `'cpu'` (default) parallelizes the resamples with joblib; `'gpu'` computes voxel-wise ISC through PyTorch (10-30× speedup) and, for the pairwise bootstrap, runs the resamples on the device too; None runs single-threaded numpy. | <code>'cpu'</code>
`n_jobs` | <code>int</code> | CPU cores for the resamples when `device` is not None. -1 (default) picks the worker count from available memory. | <code>-1</code>
`max_gpu_memory_gb` | <code>float \| None</code> | GPU working-set budget in GB for the pairwise GPU bootstrap (`device='gpu'`, `summary_statistic='pairwise'`, `method='bootstrap'`): bounds the `(perm_batch, voxel_chunk, n_subjects, n_subjects)` resample tensor, chunking voxels and permutations to fit. Not used by the leave-one-out or surrogate paths. None (default) measures the device. | <code>None</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'isc'` (float or np.ndarray, observed ISC), `'p'` (float or     np.ndarray, p-value with the `(count + 1) / (n + 1)` correction),     `'ci'` (tuple `(lower, upper)` percentiles of the resamples),     `'device'` (the execution path used), and — when     `return_null=True` — `'null_dist'` (np.ndarray).

**Examples:**

```python
# Single-feature ISC
data = np.random.randn(100, 10)  # 100 timepoints, 10 subjects
result = isc_permutation_test(data, n_permute=1000)
result["isc"], result["p"]

# Voxel-wise leave-one-out ISC on the GPU
data_voxels = np.random.randn(100, 50, 5000)  # 5K voxels
result = isc_permutation_test(
    data_voxels,
    summary_statistic="leave-one-out",
    device="gpu",
    n_permute=5000,
)
(result["p"] < 0.05).sum()  # → number of significant voxels

# Leave-one-out vs pairwise
result_loo = isc_permutation_test(data, summary_statistic="leave-one-out")
result_pair = isc_permutation_test(data, summary_statistic="pairwise")
```

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G., Shin, Y. W., Taylor, P. A., Glen, D. R., Reynolds, R. C.,
Israel, R. B., & Cox, R. W. (2016). Untangling the relatedness among
correlations, part I: nonparametric approaches to inter-subject
correlation analysis at the group level. NeuroImage, 142, 248-259.

</details>

(algorithms-isfc)=
### `isfc`

```python
isfc(data, method = 'average', n_jobs = -1)
```

Compute intersubject functional connectivity (ISFC) from per-subject matrices.

Uses the leave-one-out approach of Simony et al. (2016): for each subject,
average the other subjects' data and correlate every voxel/ROI time series
of the target subject with every voxel/ROI time series of that average.
Subjects are independent, so they are processed in parallel with joblib
unless `n_jobs=1`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>list[ndarray]</code> | One matrix per subject, each `(n_observations, n_features)` with identical shapes. | *required*
`method` | <code>str</code> | Only `'average'` (leave-one-out) is implemented. | <code>'average'</code>
`n_jobs` | <code>int</code> | Parallel workers; -1 (default) uses all cores, 1 runs serially. | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
<code>list[ndarray]</code> | One `(n_features, n_features)` ISFC matrix per     subject.

<details class="references" open markdown="1">
<summary>References</summary>

Simony, E., Honey, C. J., Chen, J., Lositsky, O., Yeshurun, Y., Wiesel,
A., & Hasson, U. (2016). Dynamic reconfiguration of the default mode
network during narrative comprehension. Nature Communications, 7, 12141.

</details>

(algorithms-isps)=
### `isps`

```python
isps(data, *, sampling_freq = 0.5, low_cut = 0.04, high_cut = 0.07, order = 5, pairwise = False)
```

Compute dynamic intersubject phase synchrony (ISPS) from an observations-by-subjects array.

Instantaneous phase synchrony across subjects for a single voxel/ROI time
series, after Glerean et al. (2012): the data are narrow-band filtered
(Butterworth) and Hilbert-transformed to get each subject's instantaneous
phase angle at every time point. Across subjects, the result gives the
mean phase angle, the mean resultant vector length, and a parametric
p-value from the Rayleigh test for circular uniformity (Fisher, 1995).
With `pairwise=True` these are computed on pairwise phase-angle differences
(inter-site phase coupling in the EEG literature) rather than on the raw
angles (inter-trial phase coupling).

The default band, 0.04-0.07 Hz, follows Glerean et al. (2012). It is close
to the "slow-4" band (0.025-0.067 Hz; Zuo et al., 2010; Penttonen &
Buzsáki, 2003) but excludes ~0.03 Hz, which carries aliased respiration
(Birn et al., 2006).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>ndarray \| DataFrame \| DataFrame</code> | Observations by subjects. | *required*
`sampling_freq` | <code>float</code> | Sampling frequency in Hz. Defaults to 0.5. | <code>0.5</code>
`low_cut` | <code>float</code> | Lower band-pass cutoff in Hz. Defaults to 0.04. | <code>0.04</code>
`high_cut` | <code>float</code> | Upper band-pass cutoff in Hz. Defaults to 0.07. | <code>0.07</code>
`order` | <code>int</code> | Butterworth filter order. Defaults to 5. | <code>5</code>
`pairwise` | <code>bool</code> | Compute on pairwise phase-angle differences instead of the raw phase angles. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'average_angle'` (np.ndarray, mean phase angle per time     point), `'vector_length'` (np.ndarray, mean resultant length per     time point), and `'p'` (np.ndarray, Rayleigh-test p-value per time     point).

<details class="references" open markdown="1">
<summary>References</summary>

Birn, R. M., Smith, M. A., Bandettini, P. A., & Diamond, J. B. (2006).
Separating respiratory-variation-related fluctuations from
neuronal-activity-related fluctuations in fMRI. NeuroImage, 31,
1536-1548.

Buzsáki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical
networks. Science, 304(5679), 1926-1929.

Fisher, N. I. (1995). Statistical analysis of circular data. Cambridge
University Press.

Glerean, E., Salmi, J., Lahnakoski, J. M., Jääskeläinen, I. P., & Sams,
M. (2012). Functional magnetic resonance imaging phase synchronization
as a measure of dynamic functional connectivity. Brain Connectivity,
2(2), 91-101.

</details>

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
`nsamples` | <code>int</code> | Number of observations (e.g. TRs). | *required*
`sampling_freq` | <code>float</code> | Sampling frequency in Hz (i.e. 1 / TR). | *required*
`filter_length` | <code>int</code> | Filter length in seconds. | *required*
`unit_scale` | <code>bool</code> | Scale the basis functions to the range [-1, 1]. Defaults to True. | <code>True</code>
`drop` | <code>int</code> | Number of leading (slowest) bases to drop after the constant is removed; `drop=2` removes the first two. Defaults to 0, which keeps the linear/sigmoidal drift basis that SPM discards. | <code>0</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Basis matrix of shape (nsamples, n_bases).

(algorithms-matrix-permutation-test)=
### `matrix_permutation_test`

```python
matrix_permutation_test(data1: np.ndarray, data2: np.ndarray, *, n_permute: int = 5000, metric: str = 'pearson', how: str = 'upper', include_diag: bool = False, tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Matrix permutation test (Mantel test) for correlating two square matrices.

Tests whether the correlation between the elements of two matrices is
significant by permuting the rows and columns of one matrix together
(`data1[perm][:, perm]`) while keeping the other fixed. Each permutation
preserves the matrix's structure (including symmetry) but destroys its
relationship to `data2`; the p-value is the fraction of permuted correlations
at least as extreme as the observed one. Assumes both matrices are square and
the same size, and that row/column ordering is exchangeable under the null.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>ndarray</code> | First square matrix (n×n). | *required*
`data2` | <code>ndarray</code> | Second square matrix (n×n). | *required*
`n_permute` | <code>int</code> | Number of permutations. Defaults to 5000. | <code>5000</code>
`metric` | <code>str</code> | Correlation metric, one of 'pearson', 'spearman', or 'kendall'. Defaults to 'pearson'. | <code>'pearson'</code>
`how` | <code>str</code> | Which elements to compare: 'upper' (upper triangle; assumes symmetric matrices), 'lower' (lower triangle), or 'full' (all elements, see `include_diag`). Defaults to 'upper'. | <code>'upper'</code>
`include_diag` | <code>bool</code> | Include diagonal elements (only when `how='full'`). Defaults to False. | <code>False</code>
`tail` | <code>int \| str</code> | `2` or `'two'` for a two-tailed test (r != 0); `1` or `'one'` for a one-tailed test of r > 0 (negate one matrix for the other direction). Defaults to 2. | <code>2</code>
`return_null` | <code>bool</code> | Also return the null distribution. Defaults to False. | <code>False</code>
`device` | <code>str \| None</code> | `'cpu'` parallelizes permutations across `n_jobs` joblib workers (4-8× speedup); `None` runs single-threaded NumPy (for debugging or small problems). Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of parallel workers, -1 = all cores; only used when `device='cpu'`. Defaults to -1. | <code>-1</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | Show a progress bar over permutations. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys 'correlation' (float, observed correlation), 'p' (float,     Phipson-Smyth corrected p-value), 'device' (`'cpu'` or `None`, the     execution path used), and 'null_dist' (np.ndarray) when     `return_null=True`.

<details class="references" open markdown="1">
<summary>References</summary>

Chen, G. et al. (2016). Untangling the relatedness among correlations,
part I: nonparametric approaches to inter-subject correlation analysis
at the group level. NeuroImage, 142, 248-259.

Mantel, N. (1967). The detection of disease clustering and a generalized
regression approach. Cancer Research, 27(2), 209-220.

</details>

**Examples:**

```python
import numpy as np
from nltools.algorithms.inference import matrix_permutation_test

# Two 20×20 similarity matrices sharing a common pattern
rng = np.random.default_rng(42)
pattern = rng.standard_normal((20, 10))
data1 = np.corrcoef(pattern + rng.standard_normal((20, 10)) * 0.5)
data2 = np.corrcoef(pattern + rng.standard_normal((20, 10)) * 0.5)

result = matrix_permutation_test(data1, data2, n_permute=1000)
print(f"Correlation: {result['correlation']:.3f}, p = {result['p']:.4f}")
```

(algorithms-multi-threshold)=
### `multi_threshold`

```python
multi_threshold(t_map, p_map, thresh)
```

Threshold a statistic image at several p-values and count the passes per voxel.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_map` | <code>[BrainData](#page-data-brain-data)</code> | Statistic image (e.g. t-values or betas). | *required*
`p_map` | <code>[BrainData](#page-data-brain-data)</code> | P-value image with the same voxels as `t_map`. | *required*
`thresh` | <code>list[float]</code> | P-value thresholds to apply. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | Cumulative map. Positive values count how many thresholds a     positive statistic passed; negative values count the same for negative     statistics.

<details class="note" open markdown="1">
<summary>Note</summary>

Calling `threshold` once per level gives separate images; this returns a
single map of the threshold hierarchy, which `nilearn.image.threshold_img`
cannot produce.

</details>

(algorithms-one-sample-permutation-test)=
### `one_sample_permutation_test`

```python
one_sample_permutation_test(data: np.ndarray, *, n_permute: int = 5000, tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None, progress_bar: bool = False) -> dict
```

One-sample permutation test using sign flipping.

Tests whether the mean of `data` differs from zero by randomly flipping the
sign of each observation — the permutation analogue of a one-sample t-test.
Multi-feature (voxel-wise) data tests each column independently against
the same permutations.

Assumes errors are distributed symmetrically around zero. For strongly
skewed data, prefer bootstrap resampling.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>ndarray</code> | Data to test, shape `(n_samples,)` for a single feature or `(n_samples, n_features)` for voxel-wise data. | *required*
`n_permute` | <code>int</code> | Number of permutations. Defaults to 5000. | <code>5000</code>
`tail` | <code>int \| str</code> | `2` or `'two'` (default) for a two-tailed test (mean != 0); `1` or `'one'` for a one-tailed test of mean > 0 (negate the data for the other direction — the fixed direction keeps multiple-comparison correction valid). | <code>2</code>
`return_null` | <code>bool</code> | If True, include the full null distribution in the result. Defaults to False. | <code>False</code>
`device` | <code>str \| None</code> | Execution path. `'cpu'` (default) parallelizes with joblib across `n_jobs` cores (4-8× speedup); `'gpu'` batches permutations through PyTorch (fastest for large problems); None runs single-threaded numpy (small problems, debugging). | <code>'cpu'</code>
`n_jobs` | <code>int</code> | CPU cores for `device='cpu'`. Defaults to -1 (all cores). | <code>-1</code>
`max_gpu_memory_gb` | <code>float \| None</code> | GPU memory budget in GB for `device='gpu'`; controls automatic batching. None (default) measures the device's available memory. | <code>None</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | Whether to display a progress bar. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'mean'` (float or np.ndarray, observed mean(s)), `'p'`     (float or np.ndarray, p-value(s)), `'device'` (the execution path     used), and — when `return_null=True` — `'null_dist'` (np.ndarray,     shape `(n_permute,)` or `(n_permute, n_features)`).

**Examples:**

```python
# Single feature (default CPU parallelization)
data = np.random.randn(30)
result = one_sample_permutation_test(data, n_permute=5000)
result["p"]  # → 0.23

# Voxel-wise test on the GPU
data = np.random.randn(30, 10000)  # 30 subjects, 10K voxels
result = one_sample_permutation_test(data, n_permute=5000, device="gpu")
result["mean"].shape  # → (10000,)
result["p"].shape  # → (10000,)

# Single-threaded (for debugging)
result = one_sample_permutation_test(data, n_permute=5000, device=None)
```

(algorithms-phase-randomize)=
### `phase_randomize`

```python
phase_randomize(data: np.ndarray, *, device: str | None = 'cpu', random_state: int | np.random.RandomState | None = None) -> np.ndarray
```

FFT-based phase randomization for time-series data.

Preserves the power spectrum (and therefore the autocorrelation) exactly, up
to numerical precision, while destroying nonlinear temporal structure. Used to
test whether data was generated by a linear Gaussian process or contains
nonlinear dynamics.

The signal is transformed with an FFT, each positive frequency is multiplied
by `exp(iφ)` with φ drawn uniformly from [0, 2π], the matching negative
frequency by the conjugate `exp(-iφ)` so the inverse FFT is real, and the
result is transformed back.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>ndarray</code> | Time series, shape (n_samples,) or (n_samples, n_features). | *required*
`device` | <code>str \| None</code> | `'cpu'` or None runs NumPy's FFT in float64; `'gpu'` runs PyTorch's FFT on CUDA/MPS in float32 (5-20× faster for large data); `'auto'` uses a GPU if present, else CPU. Defaults to 'cpu'. | <code>'cpu'</code>
`random_state` | <code>int \| RandomState \| None</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | Phase-randomized data with the same shape as the input.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If `device` is not None, 'cpu', 'gpu', or 'auto'. An explicit `'gpu'` runs on the GPU or raises; it never silently falls back to CPU.

**Examples:**

```python
x = np.sin(np.linspace(0, 10 * np.pi, 100))
x_rand = phase_randomize(x, random_state=42)
# Power spectrum is preserved
np.allclose(np.abs(np.fft.rfft(x)) ** 2, np.abs(np.fft.rfft(x_rand)) ** 2)  # → True

# GPU acceleration for large data
x_large = np.random.randn(10000)
x_rand_gpu = phase_randomize(x_large, device="gpu", random_state=42)
```

(algorithms-procrustes)=
### `procrustes`

```python
procrustes(data1, data2)
```

Perform a Procrustes similarity analysis on two data sets.

For more comprehensive Procrustes-based alignment tasks, use
`HyperAlignment` and `align()` instead.

Each input matrix is a set of points or vectors (the rows of the matrix).
The dimension of the space is the number of columns of each matrix. Given
two identically sized matrices, procrustes standardizes both so that
$tr(AA^{T}) = 1$ and both sets of points are centered around the origin.
It then applies the optimal transform to the second matrix (including
scaling/dilation, rotations, and reflections) to minimize
$M^{2}=\sum(data1-data2)^{2}$, the sum of squared pointwise differences
between the two datasets. Both inputs must have the same number of rows;
if they differ in the number of columns, the narrower one is padded with
columns of zeros.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>ndarray</code> | Matrix whose n rows represent points in k (columns) space. `data1` is the reference data; after it is standardized, the data from `data2` will be transformed to fit the pattern in `data1` (must have >1 unique points). | *required*
`data2` | <code>ndarray</code> | n rows of data in k space to be fit to `data1`. Must have the same number of rows as `data1` (must have >1 unique points). | *required*

**Returns:**

Type | Description
---- | -----------
<code>tuple[ndarray, ndarray, float, ndarray, float]</code> | `(mtx1, mtx2,     disparity, R, scale)` — `mtx1` is a standardized version of `data1`;     `mtx2` is the orientation of `data2` that best fits `data1` (centered,     but not necessarily $tr(AA^{T}) = 1$); `disparity` is $M^{2}$ as defined     above; `R` is the `(N, N)` matrix solution of the orthogonal Procrustes     problem, minimizing the Frobenius norm of `dot(data1, R) - data2` subject     to `dot(R.T, R) == I`; `scale` is the sum of the singular values of     `dot(data1.T, data2)`.

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
`mat1` | <code>ndarray</code> | 1d or 2d array; must have the same number of rows as `mat2`. | *required*
`mat2` | <code>ndarray</code> | 1d or 2d array; must have the same number of rows as `mat1`. | *required*
`n_permute` | <code>int</code> | Number of permutation iterations. Defaults to 5000. | <code>5000</code>
`tail` | <code>int \| str</code> | `2` or `'two'` for a two-tailed test (default); `1` or `'one'` for one-tailed (similarity greater than chance). | <code>2</code>
`n_jobs` | <code>int</code> | Number of CPUs for the permutations; -1 (default) uses all. | <code>-1</code>
`random_state` | <code>int \| RandomState \| None</code> | Seed or generator for the row shuffling. Defaults to None. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'similarity'` (float in [0, 1], one minus the Procrustes     disparity) and `'p'` (permutation p-value).

(algorithms-regress)=
### `regress`

```python
regress(X, Y, *, method: str = 'ols', stats: str = 'full', tail: int | str = 2)
```

Fit an OLS regression of `Y` on `X`.

Does not add an intercept; include one in `X` explicitly. If `Y` is 2D, a
separate regression is fit to each column.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray</code> | Design matrix, shape (n_samples, n_regressors). | *required*
`Y` | <code>ndarray</code> | Response, shape (n_samples,) or (n_samples, n_targets). | *required*
`method` | <code>str</code> | Only 'ols' is implemented; for robust or ARMA fits use statsmodels. Defaults to 'ols'. | <code>'ols'</code>
`stats` | <code>str</code> | 'full' returns the 6-tuple below, 'betas' returns just `b`, 'tstats' returns `(b, t)`. Defaults to 'full'. | <code>'full'</code>
`tail` | <code>int \| str</code> | 2 or 'two' for two-tailed p-values (default); 1 or 'one' for a one-tailed test of beta > 0 (negate a regressor for the other direction). | <code>2</code>

**Returns:**

Type | Description
---- | -----------
<code>tuple</code> | `(b, se, t, p, df, res)` when `stats='full'`: coefficients,     standard errors, t-statistics, p-values (per `tail`), residual     degrees of freedom, and residuals. `stats='betas'` returns just `b`;     `stats='tstats'` returns `(b, t)`.

(algorithms-spm-dispersion-derivative)=
### `spm_dispersion_derivative`

```python
spm_dispersion_derivative(t_r, *, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Sample the dispersion derivative of the SPM canonical hemodynamic response function.

Thin wrapper over nilearn's `spm_dispersion_derivative`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>float</code> | Repetition time in seconds. | *required*
`oversampling` | <code>int</code> | Temporal oversampling factor (default: 50). | <code>50</code>
`time_length` | <code>float</code> | HRF kernel length in seconds (default: 32.0). | <code>32.0</code>
`onset` | <code>float</code> | Onset of the response in seconds (default: 0.0). | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | The dispersion derivative sampled every `t_r / oversampling` seconds.

(algorithms-spm-hrf)=
### `spm_hrf`

```python
spm_hrf(t_r, *, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Sample the SPM canonical hemodynamic response function.

Thin wrapper over nilearn's `spm_hrf`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>float</code> | Repetition time in seconds. | *required*
`oversampling` | <code>int</code> | Temporal oversampling factor (default: 50). | <code>50</code>
`time_length` | <code>float</code> | HRF kernel length in seconds (default: 32.0). | <code>32.0</code>
`onset` | <code>float</code> | Onset of the response in seconds (default: 0.0). | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | The HRF sampled every `t_r / oversampling` seconds, peak scaled to 1.

(algorithms-spm-time-derivative)=
### `spm_time_derivative`

```python
spm_time_derivative(t_r, *, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Sample the time derivative of the SPM canonical hemodynamic response function.

Thin wrapper over nilearn's `spm_time_derivative`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`t_r` | <code>float</code> | Repetition time in seconds. | *required*
`oversampling` | <code>int</code> | Temporal oversampling factor (default: 50). | <code>50</code>
`time_length` | <code>float</code> | HRF kernel length in seconds (default: 32.0). | <code>32.0</code>
`onset` | <code>float</code> | Onset of the response in seconds (default: 0.0). | <code>0.0</code>

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | The time derivative sampled every `t_r / oversampling` seconds.

(algorithms-threshold)=
### `threshold`

```python
threshold(stat, p, thr = 0.05, return_mask = False)
```

Threshold a statistic image by the p-values in a separate image.

Voxels whose p-value is at or above `thr` are set to zero in a copy of `stat`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`stat` | <code>[BrainData](#page-data-brain-data)</code> | Statistic image (e.g. betas or t-values). | *required*
`p` | <code>[BrainData](#page-data-brain-data)</code> | P-value image with the same voxels as `stat`. | *required*
`thr` | <code>float</code> | P-value threshold; voxels with `p < thr` are kept. Defaults to 0.05. | <code>0.05</code>
`return_mask` | <code>bool</code> | Also return the binary thresholding mask. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data) \| tuple[[BrainData](#page-data-brain-data), [BrainData](#page-data-brain-data)]</code> | The thresholded image, or the     tuple `(thresholded, mask)` when `return_mask=True`.

<details class="note" open markdown="1">
<summary>Note</summary>

`BrainData.threshold` and `nilearn.image.threshold_img` threshold an image
by its own values; this function is the only one that thresholds one
image by the p-values of another.

</details>

(algorithms-timeseries-correlation-permutation-test)=
### `timeseries_correlation_permutation_test`

```python
timeseries_correlation_permutation_test(data1: np.ndarray, data2: np.ndarray, *, method: Literal['circle_shift', 'phase_randomize'] = 'circle_shift', n_permute: int = 5000, metric: Literal['pearson', 'spearman', 'kendall'] = 'pearson', tail: int | str = 2, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, return_null: bool = False, random_state: int | np.random.RandomState | None = None, progress_bar: bool = False) -> dict
```

Permutation test for the correlation between two autocorrelated time series.

Standard permutation tests shuffle samples independently, which destroys
autocorrelation and inflates Type I error on time series. This test instead
builds the null from surrogates of `data1` that preserve temporal structure
— circular shifts (`'circle_shift'`) or Fourier phase randomization
(`'phase_randomize'`) — while `data2` stays fixed. For independent
observations use `correlation_permutation_test`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>ndarray</code> | First time series, shape (n_samples,) or (n_samples, 1). | *required*
`data2` | <code>ndarray</code> | Second time series, shape (n_samples,) or (n_samples, 1). | *required*
`method` | <code>str</code> | `'circle_shift'` rotates the series (preserves autocorrelation; fast, and suitable for most fMRI time series); `'phase_randomize'` randomizes Fourier phases (preserves the power spectrum exactly; tests for nonlinear structure). Defaults to 'circle_shift'. | <code>'circle_shift'</code>
`n_permute` | <code>int</code> | Number of permutations. Defaults to 5000. | <code>5000</code>
`metric` | <code>str</code> | Correlation type, one of 'pearson', 'spearman', or 'kendall'. Defaults to 'pearson'. | <code>'pearson'</code>
`tail` | <code>int \| str</code> | `2` or `'two'` for a two-tailed test; `1` or `'one'` for a one-tailed test in the positive direction (negate one series for the other direction). Defaults to 2. | <code>2</code>
`device` | <code>str \| None</code> | Execution path. `'cpu'` parallelizes permutations across `n_jobs` joblib workers (4-8× speedup); `'gpu'` generates surrogates with PyTorch in memory-bounded batches (5-20× faster for n_samples > 1000; `'phase_randomize'` benefits most from the GPU FFT); `None` runs single-threaded NumPy (for debugging or small problems). Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of CPU workers, -1 = all cores; only used when `device='cpu'`. Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>float \| None</code> | GPU memory budget in GB that sizes the permutation batches; only used when `device='gpu'`. None (default) measures the device's available memory. Larger values fit more permutations per batch but risk OOM on smaller GPUs. | <code>None</code>
`return_null` | <code>bool</code> | Also return the null distribution. Defaults to False. | <code>False</code>
`random_state` | <code>int \| RandomState \| None</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | Show a progress bar over permutations. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys 'correlation' (float, observed correlation), 'p' (float),     'device' (the execution path used: `'cpu'`, `'gpu'`, or `None`), and     'null_dist' (np.ndarray of shape (n_permute,)) when     `return_null=True`.

**Examples:**

```python
import numpy as np
from nltools.algorithms import timeseries_correlation_permutation_test

rng = np.random.default_rng(0)
x = np.sin(np.linspace(0, 10 * np.pi, 100))  # strongly autocorrelated
y = x + rng.standard_normal(100) * 0.5
result = timeseries_correlation_permutation_test(
    x, y, method="circle_shift", n_permute=1000, random_state=42
)
result["correlation"]  # → 0.853
result["p"]  # → 0.078 — the autocorrelation-aware null is far wider
#   than a sample-shuffling null would be

# GPU acceleration
result = timeseries_correlation_permutation_test(
    x, y, method="phase_randomize", device="gpu", n_permute=5000
)
```

(algorithms-transform-pairwise)=
### `transform_pairwise`

```python
transform_pairwise(X, y)
```

Transform data into pairwise differences with balanced labels for ranking.

Turns an n-class ranking problem into a two-class classification problem:
every pair of samples with different target values becomes one difference
row, and signs are flipped so that the -1 and +1 classes are balanced.

Reference: Herbrich, R., Graepel, T., & Obermayer, K. "Large Margin Rank
Boundaries for Ordinal Regression".

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>ndarray</code> | Data, shape (n_samples, n_features). | *required*
`y` | <code>ndarray</code> | Target labels, shape (n_samples,) or (n_samples, 2). A second column groups the samples; pairs from different groups are skipped. | *required*

**Returns:**

Type | Description
---- | -----------
<code>tuple[ndarray, ndarray]</code> | `(X_trans, y_trans)`. `X_trans` has shape     (k, n_features) with one row per retained pair (k is at most     n_samples * (n_samples - 1) / 2; pairs are formed within groups when     given). `y_trans` holds the labels in {-1, +1}, shape (k,), or (k, 2)     with the group in the second column when `y` had two columns.

(algorithms-trim)=
### `trim`

```python
trim(data, cutoff = None)
```

Trim a Polars DataFrame/Series by replacing outlier values with NaNs.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>DataFrame \| Series</code> | Data to trim. | *required*
`cutoff` | <code>dict</code> | A dictionary with keys `{'std': [low, high]}` or `{'quantile': [low, high]}`. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>DataFrame \| Series</code> | Trimmed data (outliers replaced with NaN), the     same type as the input.

(algorithms-two-sample-permutation-test)=
### `two_sample_permutation_test`

```python
two_sample_permutation_test(data1: np.ndarray, data2: np.ndarray, *, n_permute: int = 5000, tail: int | str = 2, return_null: bool = False, device: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float | None = None, random_state: int | None = None, progress_bar: bool = False) -> dict
```

Two-sample permutation test using group-label shuffling.

Tests whether two independent groups have different means by randomly
reassigning observations to groups — the permutation analogue of an
independent-samples t-test. Group sizes may differ. Multi-feature
(voxel-wise) data tests each column independently against the same
permutations.

Assumes exchangeability under the null (group assignment is arbitrary):
independent samples from similarly shaped distributions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data1` | <code>ndarray</code> | Group 1 data, shape `(n_samples1,)` for a single feature or `(n_samples1, n_features)` for voxel-wise data. | *required*
`data2` | <code>ndarray</code> | Group 2 data, shape `(n_samples2,)` or `(n_samples2, n_features)`; must have the same number of features as `data1`. | *required*
`n_permute` | <code>int</code> | Number of permutations. Defaults to 5000. | <code>5000</code>
`tail` | <code>int \| str</code> | `2` or `'two'` (default) for a two-tailed test (mean1 != mean2); `1` or `'one'` for a one-tailed test of mean1 > mean2 (swap the groups for the other direction — the fixed direction keeps multiple-comparison correction valid). | <code>2</code>
`return_null` | <code>bool</code> | If True, include the full null distribution in the result. Defaults to False. | <code>False</code>
`device` | <code>str \| None</code> | Execution path. `'cpu'` (default) parallelizes with joblib across `n_jobs` cores (4-8× speedup); `'gpu'` batches permutations through PyTorch (fastest for large problems); None runs single-threaded numpy (small problems, debugging). | <code>'cpu'</code>
`n_jobs` | <code>int</code> | CPU cores for `device='cpu'`. Defaults to -1 (all cores). | <code>-1</code>
`max_gpu_memory_gb` | <code>float \| None</code> | GPU memory budget in GB for `device='gpu'`; controls automatic batching. None (default) measures the device's available memory. | <code>None</code>
`random_state` | <code>int \| None</code> | Random seed for reproducibility. | <code>None</code>
`progress_bar` | <code>bool</code> | Whether to display a progress bar. Defaults to False. | <code>False</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | Keys `'mean_diff'` (float or np.ndarray, observed     `mean(data1) - mean(data2)`), `'p'` (float or np.ndarray,     p-value(s)), `'device'` (the execution path used), and — when     `return_null=True` — `'null_dist'` (np.ndarray, shape     `(n_permute,)` or `(n_permute, n_features)`).

**Examples:**

```python
# Single feature (default CPU parallelization)
data1 = np.random.randn(20)  # Group 1: 20 subjects
data2 = np.random.randn(25)  # Group 2: 25 subjects
result = two_sample_permutation_test(data1, data2, n_permute=5000)
result["p"]  # → 0.45

# Voxel-wise test on the GPU
data1 = np.random.randn(20, 10000)  # 20 subjects, 10K voxels
data2 = np.random.randn(25, 10000)  # 25 subjects, 10K voxels
result = two_sample_permutation_test(data1, data2, n_permute=5000, device="gpu")
result["mean_diff"].shape  # → (10000,)
result["p"].shape  # → (10000,)

# Single-threaded (for debugging)
result = two_sample_permutation_test(data1, data2, n_permute=5000, device=None)
```

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
`mat` | <code>ndarray</code> | 2d numpy array. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | U-centered version of the input.

**Raises:**

Type | Description
---- | -----------
<code>ValueError</code> | If input is not 2D.

**Examples:**

```python
mat = np.random.randn(5, 5)
result = u_center(mat)
np.allclose(np.diag(result), 0)  # → True
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
`data` | <code>DataFrame \| Series</code> | Data to upsample. Non-numeric columns are dropped from a DataFrame. | *required*
`sampling_freq` | <code>float</code> | Sampling frequency of the data in Hz. | <code>None</code>
`target` | <code>float</code> | Upsampling target. | <code>None</code>
`target_type` | <code>str</code> | Unit of `target`, one of 'samples', 'seconds', or 'hz'. | <code>'samples'</code>
`method` | <code>str</code> | Interpolation method, one of 'linear', 'nearest', 'zero', 'slinear', 'quadratic', or 'cubic'; 'zero', 'slinear', 'quadratic' and 'cubic' refer to spline interpolation of zeroth, first, second or third order (default: 'linear'). | <code>'linear'</code>

**Returns:**

Type | Description
---- | -----------
<code>DataFrame \| Series</code> | Upsampled data, the same type as the input.

(algorithms-winsorize)=
### `winsorize`

```python
winsorize(data, cutoff = None, replace_with_cutoff = True)
```

Winsorize a Polars DataFrame/Series with the largest/lowest value not considered outlier.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>DataFrame \| Series</code> | Data to winsorize. | *required*
`cutoff` | <code>dict</code> | A dictionary with keys `{'std': [low, high]}` or `{'quantile': [low, high]}`. | <code>None</code>
`replace_with_cutoff` | <code>bool</code> | If True, replace outliers with the cutoff value; if False, replace them with the closest existing values (default: True). | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>DataFrame \| Series</code> | Winsorized data, the same type as the input.

(algorithms-zscore)=
### `zscore`

```python
zscore(data)
```

Z-score every column of a Polars or pandas DataFrame/Series.

Pandas inputs are converted to Polars; the result is always Polars (a
DataFrame for DataFrame input, a Series for Series input).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>DataFrame \| Series \| DataFrame \| Series</code> | Data to z-score. | *required*

**Returns:**

Type | Description
---- | -----------
<code>DataFrame \| Series</code> | Same shape as the input, each column z-scored     with the sample standard deviation (ddof=1).
