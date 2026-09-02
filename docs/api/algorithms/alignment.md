---
title: algorithms.alignment
label: algorithms-alignment
---

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
[`DetSRM`](#algorithms-alignment-detsrm) | Deterministic Shared Response Model (DetSRM).
[`HyperAlignment`](#algorithms-alignment-hyperalignment) | Hyperalignment using iterative Procrustes alignment.
[`LocalAlignment`](#algorithms-alignment-localalignment) | Local (neighborhood-based) functional alignment across subjects.
[`SRM`](#algorithms-alignment-srm) | Probabilistic Shared Response Model (SRM).

**Functions:**

Name | Description
---- | -----------
[`align`](#algorithms-alignment-align) | Align subject data into a common response model.
[`align_states`](#algorithms-alignment-align-states) | Align state weight maps by minimizing pairwise distance between group states.
[`procrustes_distance`](#algorithms-alignment-procrustes-distance) | Test matrix similarity using Procrustes superposition.



## Classes

(algorithms-alignment-detsrm)=
### `DetSRM`

```python
DetSRM(*, n_iter: int = 10, features: int = 50, rand_seed: int = 0)
```

Bases: `sklearn.base.BaseEstimator`, `sklearn.base.TransformerMixin`

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
[`fit`](#algorithms-alignment-fit) | Compute the Deterministic Shared Response Model.
[`transform`](#algorithms-alignment-transform) | Use the model to transform data to the Shared Response subspace.
[`transform_subject`](#algorithms-alignment-transform-subject) | Transform a new subject using the existing model.



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

(algorithms-alignment-fit)=
##### `fit`

```python
fit(X: list[np.ndarray], y: Any | None = None, *, parallel: str | None = 'cpu', n_jobs: int = -1) -> DetSRM
```

Compute the Deterministic Shared Response Model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples]</code> | Each element in the list contains the fMRI data of one subject. | *required*
`y` | <code>Any \| None</code> | not used | <code>None</code>
`parallel` | <code>str</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": not implemented -- raises `NotImplementedError` (never a silent CPU fallback) | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
<code>[DetSRM](#algorithms-alignment-detsrm)</code> | Fitted model (`self`).

(algorithms-alignment-transform)=
##### `transform`

```python
transform(X: list[np.ndarray], y: Any | None = None, *, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray]
```

Use the model to transform data to the Shared Response subspace.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples_i]</code> | Each element in the list contains the fMRI data of one subject. | *required*
`y` | <code>Any \| None</code> | not used | <code>None</code>
`parallel` | <code>str</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": not implemented -- raises `NotImplementedError` (never a silent CPU fallback) | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
<code>list[ndarray]</code> | Shared responses from input data (X); element i has     shape=[features_i, samples_i].

(algorithms-alignment-transform-subject)=
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
<code>ndarray</code> | Orthogonal mapping `W_{new}` for the new subject,     shape=[voxels, features].

(algorithms-alignment-hyperalignment)=
### `HyperAlignment`

```python
HyperAlignment(n_iter: int = 2, auto_pad: bool = True)
```

Bases: `sklearn.base.BaseEstimator`, `sklearn.base.TransformerMixin`

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
[`fit`](#algorithms-alignment-fit) | Fit hyperalignment model to data.
[`transform`](#algorithms-alignment-transform) | Transform data to common space using fitted transformations.
[`transform_subject`](#algorithms-alignment-transform-subject) | Align a new subject to the common space.

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
`parallel` | <code>str</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
<code>[HyperAlignment](#algorithms-alignment-hyperalignment)</code> | Fitted model (`self`).

##### `transform`

```python
transform(data: list[np.ndarray], *, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray]
```

Transform data to common space using fitted transformations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>list of ndarray</code> | List of data matrices to transform. Should be the same data used for fitting (or have compatible dimensions). | *required*
`parallel` | <code>str</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
<code>list[ndarray]</code> | List of transformed data matrices in common space.

##### `transform_subject`

```python
transform_subject(subject_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]
```

Align a new subject to the common space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`subject_data` | <code>(ndarray, shape(n_features, n_samples))</code> | Data from a new subject to align to the common template | *required*

**Returns:**

Type | Description
---- | -----------
<code>tuple[ndarray, ndarray, float, float]</code> | `(transformed, R, disparity,     scale)` — aligned data in common space, the transformation matrix     used, the alignment quality (sum of squared differences), and the     scale factor used.

(algorithms-alignment-localalignment)=
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
`n_features` | <code>int \| None</code> | Number of features for SRM. None uses full Procrustes (preserves dims). Defaults to None. | <code>None</code>
`n_iter` | <code>int</code> | Number of iterations for alignment refinement. Defaults to 3. | <code>3</code>
`aggregation` | <code>str</code> | Aggregation method: 'center' (center-only, preserves orthogonality) or 'all'. Defaults to 'center'. | <code>'center'</code>
`parallel` | <code>str \| None</code> | Parallelization mode. None runs single-threaded numpy, 'cpu' uses joblib CPU parallelization, and 'gpu' uses PyTorch. GPU acceleration applies only to `method='procrustes'`; requesting 'gpu' with the 'srm' or 'hyperalignment' methods raises `NotImplementedError` (an explicit GPU request never silently runs on CPU). Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of jobs for CPU parallelization. Defaults to -1. | <code>-1</code>
`progress_bar` | <code>bool</code> | Whether to display tqdm progress bars during fit and transform. Defaults to False. | <code>False</code>
`n_neighborhoods_batch` | <code>int \| None</code> | Number of neighborhoods to process per batch on the GPU. None auto-calculates a batch size from `max_memory_gb`. Defaults to None. | <code>None</code>
`max_memory_gb` | <code>float \| None</code> | Explicit memory budget (in GB) used to auto-size GPU batches when `n_neighborhoods_batch` is None. None (default) measures the device's available memory. | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`transforms_` | <code>dict[int, list[ndarray]]</code> | Per-neighborhood transforms. Keys are center voxel indices, values are lists of transform matrices (one per subject).
`template_` | <code>dict[int, ndarray]</code> | Per-neighborhood templates used for alignment.
`neighborhoods_` | <code>[SphereNeighborhoods](#neighborhoods-sphereneighborhoods) \| dict</code> | Computed neighborhoods (searchlight or roi).
`n_voxels_` | <code>int</code> | Total number of voxels in the mask.
`mask_` | <code>Nifti1Image</code> | Brain mask used for fitting.

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-alignment-fit) | Fit local alignment on multi-subject data.
[`fit_transform`](#algorithms-alignment-fit-transform) | Fit alignment and transform data in one step.
[`transform`](#algorithms-alignment-transform) | Apply local transforms to data.



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
`data` | <code>list[ndarray]</code> | List of subject data arrays, each shape (n_voxels, n_samples). Subjects can have different numbers of samples - the underlying alignment methods (SRM, HyperAlignment) handle this via zero-padding. | *required*
`mask` | <code>Nifti1Image</code> | Brain mask defining the voxel space. | *required*

**Returns:**

Type | Description
---- | -----------
<code>[LocalAlignment](#algorithms-alignment-localalignment)</code> | The fitted alignment model (`self`).

(algorithms-alignment-fit-transform)=
##### `fit_transform`

```python
fit_transform(data: list[np.ndarray], mask: nib.Nifti1Image) -> list[np.ndarray]
```

Fit alignment and transform data in one step.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`data` | <code>list[ndarray]</code> | List of subject data arrays, each shape (n_voxels, n_samples). | *required*
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

(algorithms-alignment-srm)=
### `SRM`

```python
SRM(*, n_iter: int = 10, features: int = 50, rand_seed: int = 0)
```

Bases: `sklearn.base.BaseEstimator`, `sklearn.base.TransformerMixin`

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
[`fit`](#algorithms-alignment-fit) | Compute the probabilistic Shared Response Model.
[`transform`](#algorithms-alignment-transform) | Use the model to transform matrix to Shared Response space.
[`transform_subject`](#algorithms-alignment-transform-subject) | Transform a new subject using the existing model.



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
`y` | <code>Any \| None</code> | not used | <code>None</code>
`parallel` | <code>str</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": not implemented -- raises `NotImplementedError` (never a silent CPU fallback) | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>
`pad_samples` | <code>bool</code> | If True (default), automatically zero-pad subjects with fewer samples to match the longest subject. This allows fitting SRM on data with unequal numbers of time points across subjects. | <code>True</code>

**Returns:**

Type | Description
---- | -----------
<code>[SRM](#algorithms-alignment-srm)</code> | Fitted model (`self`).

##### `transform`

```python
transform(X: list[np.ndarray], y: Any | None = None, *, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray | None]
```

Use the model to transform matrix to Shared Response space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples_i]</code> | Each element in the list contains the fMRI data of one subject. Note that number of voxels and samples can vary across subjects. | *required*
`y` | <code>Any \| None</code> | not used (as it is unsupervised learning) | <code>None</code>
`parallel` | <code>str</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": not implemented -- raises `NotImplementedError` (never a silent CPU fallback) | <code>'cpu'</code>
`n_jobs` | <code>int</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Returns:**

Type | Description
---- | -----------
<code>list[ndarray]</code> | Shared responses from input data (X); element i has     shape=[features_i, samples_i].

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
<code>ndarray</code> | Orthogonal mapping `W_{new}` for the new subject,     shape=[voxels, features].



## Functions

(algorithms-alignment-align)=
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
<code>dict</code> | A dictionary containing a list of transformed subject matrices, a     list of transformation matrices, the shared response matrix, and the     intersubject correlation of the shared responses.

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

(algorithms-alignment-align-states)=
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
<code>ndarray</code> | If `return_index=False` (default), `target[:, remapping]` — the     target's columns reordered to match the reference, oriented pattern x     state (same shape as `target`). If `return_index=True`, the remapping     index array that reorders the target's state columns.

(algorithms-alignment-procrustes-distance)=
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
`mat1` | <code>ndarray</code> | 2d numpy array; must have same number of rows as mat2 | *required*
`mat2` | <code>ndarray</code> | 1d or 2d numpy array; must have same number of rows as mat1 | *required*
`n_permute` | <code>int</code> | number of permutation iterations to perform | <code>5000</code>
`tail` | <code>int or str</code> | `2` or `'two'` for two-tailed (default); `1` or `'one'` for one-tailed (similarity > chance) | <code>2</code>
`n_jobs` | <code>int</code> | The number of CPUs to use to do permutation; default -1 (all) | <code>-1</code>
`random_state` | <code>int, np.random.RandomState, or None</code> | seed or generator for the permutation shuffling; default None | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>dict</code> | results with keys `similarity` (float in [0, 1]) and `p` (permuted p-value)
