---
title: Functional alignment
label: page-tasks-alignment
---

Put subjects into a shared functional space. `align` is the whole-brain entry point, with `method='procrustes'` or an SRM variant. `SRM`, `DetSRM`, `HyperAlignment`, and `LocalAlignment` are the sklearn-style estimators; `LocalAlignment` fits one transform per ROI or searchlight. `align_states` matches state maps across groups. `BrainData.align` calls `align`.

**Classes:**

Name | Description
---- | -----------
[`SRM`](#tasks-alignment-srm) | Probabilistic Shared Response Model (SRM).
[`DetSRM`](#tasks-alignment-detsrm) | Deterministic Shared Response Model (DetSRM).
[`HyperAlignment`](#tasks-alignment-hyperalignment) | Hyperalignment using iterative Procrustes alignment (Haxby et al., 2011).
[`LocalAlignment`](#tasks-alignment-localalignment) | Local (neighborhood-based) functional alignment across subjects.

**Functions:**

Name | Description
---- | -----------
[`align`](#tasks-alignment-align) | Align subject data into a common response model.
[`align_states`](#tasks-alignment-align-states) | Align state weight maps by minimizing pairwise distance between group states.
[`procrustes`](#tasks-alignment-procrustes) | Perform a Procrustes similarity analysis on two data sets.
[`procrustes_distance`](#tasks-alignment-procrustes-distance) | Test matrix similarity using Procrustes superposition.

## Classes

(tasks-alignment-srm)=
### `SRM`

```python
SRM(*, n_iter: int = 10, features: int = 50, rand_seed: int = 0)
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
`features` | <code>int</code> | Number of shared features to compute. Defaults to 50. | <code>50</code>
`rand_seed` | <code>int</code> | Seed for the random initialization. Defaults to 0. | <code>0</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`w_` | <code>list[ndarray]</code> | Per-subject orthogonal transforms, element i of shape (voxels_i, features).
`s_` | <code>ndarray</code> | The shared response, shape (features, samples).
`sigma_s_` | <code>ndarray</code> | Covariance of the shared response's Normal distribution, shape (features, features).
`mu_` | <code>list[ndarray]</code> | Per-subject voxel means over samples, element i of shape (voxels_i,).
`rho2_` | <code>ndarray</code> | Estimated noise variance $\rho_i^2$ per subject, shape (subjects,).
`random_state_` | <code>RandomState</code> | Generator seeded from `rand_seed`.

**Methods:**

Name | Description
---- | -----------
[`fit`](#tasks-alignment-fit) | Compute the probabilistic Shared Response Model.
[`transform`](#tasks-alignment-transform) | Project each subject's data into the shared response space.
[`transform_subject`](#tasks-alignment-transform-subject) | Transform a new subject using the existing model.



**Examples:**

```python
import numpy as np
from nltools.algorithms import SRM

data = [np.random.randn(100, 50) for _ in range(3)]  # 3 subjects

srm = SRM(n_iter=10, features=50)
srm.fit(data, parallel="cpu", n_jobs=-1)
shared_responses = srm.transform(data)  # list of (50, 50) arrays

w = srm.w_  # subject-specific transforms
s = srm.s_  # shared response
```

#### Methods

(tasks-alignment-fit)=
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

(tasks-alignment-transform)=
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
<code>list[ndarray \| None]</code> | Shared responses, element i of shape     (features, samples_i).

(tasks-alignment-transform-subject)=
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
<code>ndarray</code> | Orthogonal mapping $W_{new}$ for the new subject, shape     (voxels, features).

(tasks-alignment-detsrm)=
### `DetSRM`

```python
DetSRM(*, n_iter: int = 10, features: int = 50, rand_seed: int = 0)
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
`features` | <code>int</code> | Number of shared features to compute. Defaults to 50. | <code>50</code>
`rand_seed` | <code>int</code> | Seed for the random initialization. Defaults to 0. | <code>0</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`w_` | <code>list[ndarray]</code> | Per-subject orthogonal transforms, element i of shape (voxels_i, features).
`s_` | <code>ndarray</code> | The shared response, shape (features, samples).
`random_state_` | <code>RandomState</code> | Generator seeded from `rand_seed`.

**Methods:**

Name | Description
---- | -----------
[`fit`](#tasks-alignment-fit) | Compute the Deterministic Shared Response Model.
[`transform`](#tasks-alignment-transform) | Project each subject's data into the shared response subspace.
[`transform_subject`](#tasks-alignment-transform-subject) | Transform a new subject using the existing model.



**Examples:**

```python
import numpy as np
from nltools.algorithms import DetSRM

data = [np.random.randn(100, 50) for _ in range(3)]  # 3 subjects

detsrm = DetSRM(n_iter=10, features=50)
detsrm.fit(data, parallel="cpu", n_jobs=-1)
shared_responses = detsrm.transform(data)  # list of (50, 50) arrays

w = detsrm.w_  # subject-specific transforms
s = detsrm.s_  # shared response
```

#### Methods

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
<code>list[ndarray]</code> | Shared responses, element i of shape     (features, samples_i).

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
<code>ndarray</code> | Orthogonal mapping $W_{new}$ for the new subject, shape     (voxels, features).

(tasks-alignment-hyperalignment)=
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
[`fit`](#tasks-alignment-fit) | Fit hyperalignment model to data.
[`transform`](#tasks-alignment-transform) | Transform data to the common space using the fitted transformations.
[`transform_subject`](#tasks-alignment-transform-subject) | Align a new subject to the common space.

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

(tasks-alignment-localalignment)=
### `LocalAlignment`

```python
LocalAlignment(spatial_scale: str = 'searchlight', method: str = 'procrustes', radius_mm: float = 10.0, roi_mask: nib.Nifti1Image | None = None, n_features: int | None = None, n_iter: int = 3, aggregation: str = 'center', parallel: str | None = 'cpu', n_jobs: int = -1, progress_bar: bool = False, n_neighborhoods_batch: int | None = None, max_memory_gb: float | None = None, transforms_: dict[int, list[np.ndarray]] | None = None, template_: dict[int, np.ndarray] | None = None, neighborhoods_: SphereNeighborhoods | dict[int, np.ndarray] | None = None, n_voxels_: int | None = None, mask_: nib.Nifti1Image | None = None, backend_: Backend | None = None)
```

Local (neighborhood-based) functional alignment across subjects.

Learns alignment transforms within local neighborhoods (searchlight spheres
or parcels) and applies center-only aggregation to preserve orthogonality.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`spatial_scale` | <code>str</code> | Spatial scale, either 'searchlight' (overlapping spheres) or 'roi' (non-overlapping parcels). Defaults to 'searchlight'. | <code>'searchlight'</code>
`method` | <code>str</code> | Alignment method, one of 'procrustes', 'srm', or 'hyperalignment'. Defaults to 'procrustes'. | <code>'procrustes'</code>
`radius_mm` | <code>float</code> | Sphere radius in millimeters for the searchlight scale. Defaults to 10.0. | <code>10.0</code>
`roi_mask` | <code>Nifti1Image \| None</code> | Parcellation image for the ROI scale. Required if `spatial_scale='roi'`. Defaults to None. | <code>None</code>
`n_features` | <code>int \| None</code> | Number of SRM features per neighborhood. None uses `min(n_local_voxels, n_samples)`; ignored by the other methods. Defaults to None. | <code>None</code>
`n_iter` | <code>int</code> | Number of iterations for alignment refinement. Defaults to 3. | <code>3</code>
`aggregation` | <code>str</code> | `'center'` writes only each sphere's center voxel (preserves orthogonality); `'all'` writes every voxel in the region and is selected automatically for `spatial_scale='roi'`. Defaults to `'center'`. | <code>'center'</code>
`parallel` | <code>str \| None</code> | Parallelization mode. None runs single-threaded numpy, 'cpu' uses joblib CPU parallelization, and 'gpu' uses PyTorch. GPU acceleration applies only to `method='procrustes'`; requesting 'gpu' with the 'srm' or 'hyperalignment' methods raises `NotImplementedError` (an explicit GPU request never silently runs on CPU). Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of jobs for CPU parallelization. Defaults to -1. | <code>-1</code>
`progress_bar` | <code>bool</code> | Whether to display tqdm progress bars during fit and transform. Defaults to False. | <code>False</code>
`n_neighborhoods_batch` | <code>int \| None</code> | Number of neighborhoods to process per batch on the GPU. None auto-calculates a batch size from `max_memory_gb`. Defaults to None. | <code>None</code>
`max_memory_gb` | <code>float \| None</code> | Explicit memory budget (in GB) used to auto-size GPU batches when `n_neighborhoods_batch` is None. None (default) measures the device's available memory. | <code>None</code>

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
[`fit`](#tasks-alignment-fit) | Fit local alignment on multi-subject data.
[`fit_transform`](#tasks-alignment-fit-transform) | Fit alignment and transform data in one step.
[`transform`](#tasks-alignment-transform) | Apply local transforms to data.



**Examples:**

```python
import numpy as np
import nibabel as nib
from nltools.algorithms.alignment import LocalAlignment

# Synthetic multi-subject data (voxels, samples) and a matching 1000-voxel mask
data = [np.random.randn(1000, 100) for _ in range(5)]
mask = nib.Nifti1Image(np.ones((10, 10, 10), dtype=np.int8), np.eye(4))

la = LocalAlignment(spatial_scale="searchlight", method="procrustes", radius_mm=10.0)
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

(tasks-alignment-fit-transform)=
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

## Functions

(tasks-alignment-align)=
### `align`

```python
align(data, method = 'deterministic_srm', n_features = None, axis = 0, *args, **kwargs)
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
`*args` | <code>Any</code> | Positional arguments forwarded to the `SRM`/`DetSRM` constructor. | <code>()</code>
`**kwargs` | <code>Any</code> | Keyword arguments forwarded to the `SRM`/`DetSRM` constructor. | <code>{}</code>

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

(tasks-alignment-align-states)=
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

(tasks-alignment-procrustes)=
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

(tasks-alignment-procrustes-distance)=
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
