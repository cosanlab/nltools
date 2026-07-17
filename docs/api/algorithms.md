(algorithms-algorithms)=
## `algorithms`

External functions.

**Modules:**

Name | Description
---- | -----------
[`alignment`](#algorithms-alignment) | Multi-subject functional alignment algorithms.
[`backends`](#algorithms-backends) | Backend abstraction for CPU/GPU operations.
[`hrf`](#algorithms-hrf) | Hemodynamic response functions — re-exported from nilearn.
[`inference`](#algorithms-inference) | GPU-accelerated statistical inference for neuroimaging.
[`random`](#algorithms-random) | Shared random state utilities for algorithms module.
[`ridge`](#algorithms-ridge) | Ridge regression algorithms and utilities.
[`shape_utils`](#algorithms-shape-utils) | Shared shape manipulation utilities for algorithms module.

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
[`glover_dispersion_derivative`](#algorithms-glover-dispersion-derivative) | Implement the Glover dispersion derivative :term:`HRF` model.
[`glover_hrf`](#algorithms-glover-hrf) | Implement the Glover :term:`HRF` model.
[`glover_time_derivative`](#algorithms-glover-time-derivative) | Implement the Glover time derivative :term:`HRF` (dhrf) model.
[`one_sample_permutation_test`](#algorithms-one-sample-permutation-test) | One-sample permutation test using sign-flipping.
[`ridge_cv`](#algorithms-ridge-cv) | Ridge regression with cross-validation for hyperparameter selection.
[`ridge_svd`](#algorithms-ridge-svd) | Solve ridge regression using Singular Value Decomposition.
[`spm_dispersion_derivative`](#algorithms-spm-dispersion-derivative) | Implement the :term:`SPM` dispersion derivative :term:`HRF` model.
[`spm_hrf`](#algorithms-spm-hrf) | Implement the :term:`SPM` :term:`HRF` model.
[`spm_time_derivative`](#algorithms-spm-time-derivative) | Implement the :term:`SPM` time derivative :term:`HRF` (dhrf) model.



### Classes

(algorithms-detsrm)=
#### `DetSRM`

```python
DetSRM(n_iter: int = 10, features: int = 50, rand_seed: int = 0) -> None
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

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Compute the Deterministic Shared Response Model.
[`transform`](#algorithms-transform) | Use the model to transform data to the Shared Response subspace.
[`transform_subject`](#algorithms-transform-subject) | Transform a new subject using the existing model.

##### Methods

(algorithms-fit)=
###### `fit`

```python
fit(X: list[np.ndarray], y: Any | None = None, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0) -> DetSRM
```

Compute the Deterministic Shared Response Model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples]</code> | Each element in the list contains the fMRI data of one subject. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": GPU acceleration (not yet implemented, falls back to CPU) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Maximum GPU memory budget in GB (default: 4.0). Only used when parallel="gpu". Defaults to 4.0. | <code>4.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`self` | <code>[DetSRM](#nltools.algorithms.alignment.srm.DetSRM)</code> | Fitted model

(algorithms-transform)=
###### `transform`

```python
transform(X: list[np.ndarray], y: Any | None = None, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray]
```

Use the model to transform data to the Shared Response subspace.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples_i]</code> | Each element in the list contains the fMRI data of one subject. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": GPU acceleration (not yet implemented, falls back to CPU) | <code>'cpu'</code>
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
The subject is assumed to have recieved equivalent stimulation

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

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Fit hyperalignment model to data.
[`transform`](#algorithms-transform) | Transform data to common space using fitted transformations.
[`transform_subject`](#algorithms-transform-subject) | Align a new subject to the common space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_iter` | <code>int, default=2</code> | Number of template refinement iterations | <code>2</code>
`auto_pad` | <code>bool, default=True</code> | Whether to automatically pad matrices to same size | <code>True</code>

##### Methods

###### `fit`

```python
fit(data: list[np.ndarray], parallel: str | None = 'cpu', n_jobs: int = -1) -> HyperAlignment
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
transform(data: list[np.ndarray], parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray]
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
LocalAlignment(scheme: str = 'searchlight', method: str = 'procrustes', radius_mm: float = 10.0, parcellation: Any | None = None, n_features: int | None = None, n_iter: int = 3, aggregation: str = 'center', parallel: str | None = 'cpu', n_jobs: int = -1, n_neighborhoods_batch: int | None = None, max_memory_gb: float = 4.0, transforms_: dict[int, list[np.ndarray]] | None = None, template_: dict[int, np.ndarray] | None = None, neighborhoods_: SphereNeighborhoods | dict[int, np.ndarray] | None = None, n_voxels_: int | None = None, mask_: Any | None = None, backend_: Backend | None = None) -> None
```

Local (neighborhood-based) functional alignment across subjects.

Learns alignment transforms within local neighborhoods (searchlight spheres
or parcels) and applies center-only aggregation to preserve orthogonality.

scheme : str, default='searchlight'
    Spatial scheme: 'searchlight' (overlapping spheres) or 'piecewise'
    (non-overlapping parcels).
method : str, default='procrustes'
    Alignment method: 'procrustes', 'srm', or 'hyperalignment'.
radius_mm : float, default=10.0
    Sphere radius in millimeters for searchlight scheme.
parcellation : Nifti1Image, optional
    Parcellation image for piecewise scheme. Required if scheme='piecewise'.
n_features : int, optional
    Number of features for SRM. None uses full Procrustes (preserves dims).
n_iter : int, default=3
    Number of iterations for alignment refinement.
aggregation : str, default='center'
    Aggregation method: 'center' (center-only, preserves orthogonality).
parallel : str, optional
    Parallelization: 'cpu', 'gpu', or None.
    - None: Single-threaded numpy
    - 'cpu': CPU parallelization with joblib
    - 'gpu': GPU acceleration via PyTorch (falls back to CPU if unavailable)
n_jobs : int, default=-1
    Number of jobs for CPU parallelization.

transforms_ : Dict[int, List[np.ndarray]]
    Per-neighborhood transforms. Keys are center voxel indices,
    values are lists of transform matrices (one per subject).
template_ : Dict[int, np.ndarray]
    Per-neighborhood templates used for alignment.
neighborhoods_ : SphereNeighborhoods or Dict
    Computed neighborhoods (searchlight or piecewise).
n_voxels_ : int
    Total number of voxels in the mask.
mask_ : Nifti1Image
    Brain mask used for fitting.

Examples:
>>> import numpy as np
>>> from nltools.algorithms.alignment import LocalAlignment
>>> # Create synthetic multi-subject data (voxels, samples)
>>> data = [np.random.randn(1000, 100) for _ in range(5)]
>>> la = LocalAlignment(scheme='searchlight', method='procrustes', radius_mm=10.0)
>>> la.fit(data, mask)
>>> aligned = la.transform(data)

Notes:
Based on Bazeille et al. 2021 "An empirical evaluation of functional
alignment using inter-subject decoding". Center-only aggregation is
used to preserve local orthogonality of transforms.

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Fit local alignment on multi-subject data.
[`fit_transform`](#algorithms-fit-transform) | Fit alignment and transform data in one step.
[`transform`](#algorithms-transform) | Apply local transforms to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`aggregation`](#algorithms-aggregation) | <code>[str](#str)</code> | 
`backend_` | <code>[Backend](#nltools.algorithms.backends.Backend) \| None</code> | 
`mask_` | <code>[Any](#typing.Any) \| None</code> | 
`max_memory_gb` | <code>[float](#float)</code> | 
`method` | <code>[str](#str)</code> | 
`n_features` | <code>[int](#int) \| None</code> | 
`n_iter` | <code>[int](#int)</code> | 
`n_jobs` | <code>[int](#int)</code> | 
`n_neighborhoods_batch` | <code>[int](#int) \| None</code> | 
`n_voxels_` | <code>[int](#int) \| None</code> | 
`neighborhoods_` | <code>[SphereNeighborhoods](#nltools.data.braindata.neighborhoods.SphereNeighborhoods) \| [dict](#dict)[[int](#int), [ndarray](#numpy.ndarray)] \| None</code> | 
`parallel` | <code>[str](#str) \| None</code> | 
`parcellation` | <code>[Any](#typing.Any) \| None</code> | 
`radius_mm` | <code>[float](#float)</code> | 
`scheme` | <code>[str](#str)</code> | 
`template_` | <code>[dict](#dict)[[int](#int), [ndarray](#numpy.ndarray)] \| None</code> | 
`transforms_` | <code>[dict](#dict)[[int](#int), [list](#list)[[ndarray](#numpy.ndarray)]] \| None</code> | 

##### Methods

###### `fit`

```python
fit(data: list[np.ndarray], mask: nib.Nifti1Image) -> LocalAlignment
```

Fit local alignment on multi-subject data.

data : List[np.ndarray]
    List of subject data arrays, each shape (n_voxels, n_samples).
    Subjects can have different numbers of samples - the underlying
    alignment methods (SRM, HyperAlignment) handle this via zero-padding.
mask : Nifti1Image
    Brain mask defining the voxel space.

self : LocalAlignment
    Fitted alignment model.

(algorithms-fit-transform)=
###### `fit_transform`

```python
fit_transform(data: list[np.ndarray], mask: nib.Nifti1Image) -> list[np.ndarray]
```

Fit alignment and transform data in one step.

data : List[np.ndarray]
    List of subject data arrays, each shape (n_voxels, n_samples).
mask : Nifti1Image
    Brain mask defining the voxel space.

List[np.ndarray]
    Aligned data for each subject.

###### `transform`

```python
transform(data: list[np.ndarray]) -> list[np.ndarray]
```

Apply local transforms to data.

For searchlight scheme with center-only aggregation: each voxel uses
the transform from the neighborhood where it was the center.

For piecewise scheme: all voxels in each parcel use the same transform.

data : List[np.ndarray]
    List of subject data arrays, each shape (n_voxels, n_samples).

List[np.ndarray]
    Aligned data for each subject, shape (n_voxels, n_samples).

(algorithms-srm)=
#### `SRM`

```python
SRM(n_iter: int = 10, features: int = 50, rand_seed: int = 0) -> None
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

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Compute the probabilistic Shared Response Model.
[`transform`](#algorithms-transform) | Use the model to transform matrix to Shared Response space.
[`transform_subject`](#algorithms-transform-subject) | Transform a new subject using the existing model.

##### Methods

###### `fit`

```python
fit(X: list[np.ndarray], y: Any | None = None, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, pad_samples: bool = True) -> SRM
```

Compute the probabilistic Shared Response Model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples]</code> | Each element in the list contains the fMRI data of one subject. Subjects can have different numbers of samples if pad_samples=True. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": GPU acceleration (not yet implemented, falls back to CPU) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Maximum GPU memory budget in GB (default: 4.0). Only used when parallel="gpu". Defaults to 4.0. | <code>4.0</code>
`pad_samples` | <code>[bool](#bool)</code> | If True (default), automatically zero-pad subjects with fewer samples to match the longest subject. This allows fitting SRM on data with unequal numbers of time points across subjects. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`self` | <code>[SRM](#nltools.algorithms.alignment.srm.SRM)</code> | Fitted model

###### `transform`

```python
transform(X: list[np.ndarray], y: Any | None = None, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray | None]
```

Use the model to transform matrix to Shared Response space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples_i]</code> | Each element in the list contains the fMRI data of one subject. Note that number of voxels and samples can vary across subjects. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used (as it is unsupervised learning) | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": GPU acceleration (not yet implemented, falls back to CPU) | <code>'cpu'</code>
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
The subject is assumed to have recieved equivalent stimulation

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>2D array, shape=[voxels, timepoints]</code> | The fMRI data of the new subject. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`w` | <code>2D array, shape=[voxels, features]</code> | Orthogonal mapping `W_{new}` for new subject



### Methods

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

tr:

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
dhrf : array of shape(length / t_r * oversampling), dtype=float
      dhrf sampling on the oversampled time grid

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

tr:

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
hrf : array of shape(length / t_r * oversampling, dtype=float)
     :term:`HRF` sampling on the oversampled time grid.

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

tr:

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
dhrf : array of shape(length / t_r), dtype=float
      dhrf sampling on the provided grid

(algorithms-one-sample-permutation-test)=
#### `one_sample_permutation_test`

```python
one_sample_permutation_test(data: np.ndarray, n_permute: int = 5000, tail: int | str = 2, return_null: bool = False, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None) -> dict
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
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type (default: 2) - 'two' or 2: Two-tailed test (mean != 0) - 'upper' or 1: One-tailed upper (mean > 0) - 'lower' or -1: One-tailed lower (mean < 0) For MCP correction (FDR), use 'upper' or 'lower' for consistent direction. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`parallel` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when parallel='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Maximum GPU memory to use in GB (default: 4.0) Controls automatic batching to prevent OOM errors. Only used with parallel='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>4.0</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'mean' (float or np.ndarray): Observed mean(s) - 'p' (float or np.ndarray): P-value(s) - 'null_dist' (np.ndarray): Null distribution (if return_null=True) - 'parallel' (str): Parallelization method used

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
>>> result = one_sample_permutation_test(data, n_permute=5000, parallel='gpu')
>>> result['mean'].shape
(10000,)
>>> result['p'].shape
(10000,)
```

```pycon
>>> # Single-threaded (for debugging)
>>> result = one_sample_permutation_test(data, n_permute=5000, parallel=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (parallel='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
- Single-threaded (parallel=None): Use for small problems or debugging
- For voxel-wise tests, each voxel tested independently
- Progress bars show completion for both CPU parallel and GPU batched modes

</details>

(algorithms-ridge-cv)=
#### `ridge_cv`

```python
ridge_cv(X: np.ndarray, y: np.ndarray, alphas: np.ndarray | None = None, cv: int | BaseCrossValidator = 5, fit_intercept: bool = False, parallel: str | None = 'cpu', max_gpu_memory_gb: float = 4.0, random_state: int | None = None) -> dict
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
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU-only using NumPy (default) - "gpu": GPU acceleration via PyTorch (falls back to CPU if GPU unavailable) Defaults to "cpu". | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel='gpu'). Defaults to 4.0. | <code>4.0</code>
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
- When parallel='gpu' is requested but GPU is unavailable, gracefully falls back to CPU

</details>

(algorithms-ridge-svd)=
#### `ridge_svd`

```python
ridge_svd(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, parallel: str | None = None, max_gpu_memory_gb: float = 4.0, random_state: int | None = None) -> np.ndarray
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
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU-only using NumPy (default) - "gpu": GPU acceleration via PyTorch (falls back to CPU if GPU unavailable) Defaults to None. | <code>None</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel='gpu'). Defaults to 4.0. | <code>4.0</code>
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

tr:

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
dhrf : array of shape(length / tr * oversampling), dtype=float
      dhrf sampling on the oversampled time grid

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

tr:

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
hrf : array of shape(length / t_r * oversampling, dtype=float)
     :term:`HRF` sampling on the oversampled time grid

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

tr:

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
dhrf : array of shape(length / t_r, dtype=float)
      dhrf sampling on the provided grid



### Modules

(algorithms-alignment)=
#### `alignment`

Multi-subject functional alignment algorithms.

This package provides algorithms for aligning functional data across subjects:

- **LocalAlignment**: Searchlight/piecewise alignment (Bazeille et al. 2021)
- **HyperAlignment**: Iterative Procrustes alignment (Haxby et al. 2011)
- **SRM** / **DetSRM**: Shared Response Model (Chen et al. 2015)

**Modules:**

Name | Description
---- | -----------
[`hyperalignment`](#algorithms-hyperalignment) | HyperAlignment: Multi-subject cortical surface alignment using iterative Procrustes refinement.
[`local`](#algorithms-local) | LocalAlignment: Neighborhood-based functional alignment.
[`srm`](#algorithms-srm) | Shared Response Model (SRM) for multi-subject fMRI alignment.

**Classes:**

Name | Description
---- | -----------
[`DetSRM`](#algorithms-detsrm) | Deterministic Shared Response Model (DetSRM).
[`HyperAlignment`](#algorithms-hyperalignment) | Hyperalignment using iterative Procrustes alignment.
[`LocalAlignment`](#algorithms-localalignment) | Local (neighborhood-based) functional alignment across subjects.
[`SRM`](#algorithms-srm) | Probabilistic Shared Response Model (SRM).



##### Classes

###### `DetSRM`

```python
DetSRM(n_iter: int = 10, features: int = 50, rand_seed: int = 0) -> None
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

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Compute the Deterministic Shared Response Model.
[`transform`](#algorithms-transform) | Use the model to transform data to the Shared Response subspace.
[`transform_subject`](#algorithms-transform-subject) | Transform a new subject using the existing model.



####### Attributes##

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
fit(X: list[np.ndarray], y: Any | None = None, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0) -> DetSRM
```

Compute the Deterministic Shared Response Model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples]</code> | Each element in the list contains the fMRI data of one subject. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": GPU acceleration (not yet implemented, falls back to CPU) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Maximum GPU memory budget in GB (default: 4.0). Only used when parallel="gpu". Defaults to 4.0. | <code>4.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`self` | <code>[DetSRM](#nltools.algorithms.alignment.srm.DetSRM)</code> | Fitted model

######## `transform`

```python
transform(X: list[np.ndarray], y: Any | None = None, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray]
```

Use the model to transform data to the Shared Response subspace.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples_i]</code> | Each element in the list contains the fMRI data of one subject. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": GPU acceleration (not yet implemented, falls back to CPU) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`s` | <code>list of 2D arrays, element i has shape=[features_i, samples_i]</code> | Shared responses from input data (X)

######## `transform_subject`

```python
transform_subject(X: np.ndarray) -> np.ndarray
```

Transform a new subject using the existing model.
The subject is assumed to have recieved equivalent stimulation

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>2D array, shape=[voxels, timepoints]</code> | The fMRI data of the new subject. | *required*

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

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Fit hyperalignment model to data.
[`transform`](#algorithms-transform) | Transform data to common space using fitted transformations.
[`transform_subject`](#algorithms-transform-subject) | Align a new subject to the common space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_iter` | <code>int, default=2</code> | Number of template refinement iterations | <code>2</code>
`auto_pad` | <code>bool, default=True</code> | Whether to automatically pad matrices to same size | <code>True</code>



####### Attributes##

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
fit(data: list[np.ndarray], parallel: str | None = 'cpu', n_jobs: int = -1) -> HyperAlignment
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

######## `transform`

```python
transform(data: list[np.ndarray], parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray]
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

######## `transform_subject`

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

###### `LocalAlignment`

```python
LocalAlignment(scheme: str = 'searchlight', method: str = 'procrustes', radius_mm: float = 10.0, parcellation: Any | None = None, n_features: int | None = None, n_iter: int = 3, aggregation: str = 'center', parallel: str | None = 'cpu', n_jobs: int = -1, n_neighborhoods_batch: int | None = None, max_memory_gb: float = 4.0, transforms_: dict[int, list[np.ndarray]] | None = None, template_: dict[int, np.ndarray] | None = None, neighborhoods_: SphereNeighborhoods | dict[int, np.ndarray] | None = None, n_voxels_: int | None = None, mask_: Any | None = None, backend_: Backend | None = None) -> None
```

Local (neighborhood-based) functional alignment across subjects.

Learns alignment transforms within local neighborhoods (searchlight spheres
or parcels) and applies center-only aggregation to preserve orthogonality.

scheme : str, default='searchlight'
    Spatial scheme: 'searchlight' (overlapping spheres) or 'piecewise'
    (non-overlapping parcels).
method : str, default='procrustes'
    Alignment method: 'procrustes', 'srm', or 'hyperalignment'.
radius_mm : float, default=10.0
    Sphere radius in millimeters for searchlight scheme.
parcellation : Nifti1Image, optional
    Parcellation image for piecewise scheme. Required if scheme='piecewise'.
n_features : int, optional
    Number of features for SRM. None uses full Procrustes (preserves dims).
n_iter : int, default=3
    Number of iterations for alignment refinement.
aggregation : str, default='center'
    Aggregation method: 'center' (center-only, preserves orthogonality).
parallel : str, optional
    Parallelization: 'cpu', 'gpu', or None.
    - None: Single-threaded numpy
    - 'cpu': CPU parallelization with joblib
    - 'gpu': GPU acceleration via PyTorch (falls back to CPU if unavailable)
n_jobs : int, default=-1
    Number of jobs for CPU parallelization.

transforms_ : Dict[int, List[np.ndarray]]
    Per-neighborhood transforms. Keys are center voxel indices,
    values are lists of transform matrices (one per subject).
template_ : Dict[int, np.ndarray]
    Per-neighborhood templates used for alignment.
neighborhoods_ : SphereNeighborhoods or Dict
    Computed neighborhoods (searchlight or piecewise).
n_voxels_ : int
    Total number of voxels in the mask.
mask_ : Nifti1Image
    Brain mask used for fitting.

Examples:
>>> import numpy as np
>>> from nltools.algorithms.alignment import LocalAlignment
>>> # Create synthetic multi-subject data (voxels, samples)
>>> data = [np.random.randn(1000, 100) for _ in range(5)]
>>> la = LocalAlignment(scheme='searchlight', method='procrustes', radius_mm=10.0)
>>> la.fit(data, mask)
>>> aligned = la.transform(data)

Notes:
Based on Bazeille et al. 2021 "An empirical evaluation of functional
alignment using inter-subject decoding". Center-only aggregation is
used to preserve local orthogonality of transforms.

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Fit local alignment on multi-subject data.
[`fit_transform`](#algorithms-fit-transform) | Fit alignment and transform data in one step.
[`transform`](#algorithms-transform) | Apply local transforms to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`aggregation`](#algorithms-aggregation) | <code>[str](#str)</code> | 
`backend_` | <code>[Backend](#nltools.algorithms.backends.Backend) \| None</code> | 
`mask_` | <code>[Any](#typing.Any) \| None</code> | 
`max_memory_gb` | <code>[float](#float)</code> | 
`method` | <code>[str](#str)</code> | 
`n_features` | <code>[int](#int) \| None</code> | 
`n_iter` | <code>[int](#int)</code> | 
`n_jobs` | <code>[int](#int)</code> | 
`n_neighborhoods_batch` | <code>[int](#int) \| None</code> | 
`n_voxels_` | <code>[int](#int) \| None</code> | 
`neighborhoods_` | <code>[SphereNeighborhoods](#nltools.data.braindata.neighborhoods.SphereNeighborhoods) \| [dict](#dict)[[int](#int), [ndarray](#numpy.ndarray)] \| None</code> | 
`parallel` | <code>[str](#str) \| None</code> | 
`parcellation` | <code>[Any](#typing.Any) \| None</code> | 
`radius_mm` | <code>[float](#float)</code> | 
`scheme` | <code>[str](#str)</code> | 
`template_` | <code>[dict](#dict)[[int](#int), [ndarray](#numpy.ndarray)] \| None</code> | 
`transforms_` | <code>[dict](#dict)[[int](#int), [list](#list)[[ndarray](#numpy.ndarray)]] \| None</code> | 



####### Attributes##

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
mask_: Any | None = field(default=None, repr=False)
```

######## `max_memory_gb`

```python
max_memory_gb: float = 4.0
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

######## `parcellation`

```python
parcellation: Any | None = None
```

######## `radius_mm`

```python
radius_mm: float = 10.0
```

######## `scheme`

```python
scheme: str = 'searchlight'
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

data : List[np.ndarray]
    List of subject data arrays, each shape (n_voxels, n_samples).
    Subjects can have different numbers of samples - the underlying
    alignment methods (SRM, HyperAlignment) handle this via zero-padding.
mask : Nifti1Image
    Brain mask defining the voxel space.

self : LocalAlignment
    Fitted alignment model.

######## `fit_transform`

```python
fit_transform(data: list[np.ndarray], mask: nib.Nifti1Image) -> list[np.ndarray]
```

Fit alignment and transform data in one step.

data : List[np.ndarray]
    List of subject data arrays, each shape (n_voxels, n_samples).
mask : Nifti1Image
    Brain mask defining the voxel space.

List[np.ndarray]
    Aligned data for each subject.

######## `transform`

```python
transform(data: list[np.ndarray]) -> list[np.ndarray]
```

Apply local transforms to data.

For searchlight scheme with center-only aggregation: each voxel uses
the transform from the neighborhood where it was the center.

For piecewise scheme: all voxels in each parcel use the same transform.

data : List[np.ndarray]
    List of subject data arrays, each shape (n_voxels, n_samples).

List[np.ndarray]
    Aligned data for each subject, shape (n_voxels, n_samples).

###### `SRM`

```python
SRM(n_iter: int = 10, features: int = 50, rand_seed: int = 0) -> None
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

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Compute the probabilistic Shared Response Model.
[`transform`](#algorithms-transform) | Use the model to transform matrix to Shared Response space.
[`transform_subject`](#algorithms-transform-subject) | Transform a new subject using the existing model.



####### Attributes##

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
fit(X: list[np.ndarray], y: Any | None = None, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, pad_samples: bool = True) -> SRM
```

Compute the probabilistic Shared Response Model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples]</code> | Each element in the list contains the fMRI data of one subject. Subjects can have different numbers of samples if pad_samples=True. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": GPU acceleration (not yet implemented, falls back to CPU) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Maximum GPU memory budget in GB (default: 4.0). Only used when parallel="gpu". Defaults to 4.0. | <code>4.0</code>
`pad_samples` | <code>[bool](#bool)</code> | If True (default), automatically zero-pad subjects with fewer samples to match the longest subject. This allows fitting SRM on data with unequal numbers of time points across subjects. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`self` | <code>[SRM](#nltools.algorithms.alignment.srm.SRM)</code> | Fitted model

######## `transform`

```python
transform(X: list[np.ndarray], y: Any | None = None, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray | None]
```

Use the model to transform matrix to Shared Response space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples_i]</code> | Each element in the list contains the fMRI data of one subject. Note that number of voxels and samples can vary across subjects. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used (as it is unsupervised learning) | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": GPU acceleration (not yet implemented, falls back to CPU) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`s` | <code>list of 2D arrays, element i has shape=[features_i, samples_i]</code> | Shared responses from input data (X)

######## `transform_subject`

```python
transform_subject(X: np.ndarray) -> np.ndarray
```

Transform a new subject using the existing model.
The subject is assumed to have recieved equivalent stimulation

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>2D array, shape=[voxels, timepoints]</code> | The fMRI data of the new subject. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`w` | <code>2D array, shape=[voxels, features]</code> | Orthogonal mapping `W_{new}` for new subject



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
- See `nltools.stats.procrustes()` for single-subject alignment

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

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Fit hyperalignment model to data.
[`transform`](#algorithms-transform) | Transform data to common space using fitted transformations.
[`transform_subject`](#algorithms-transform-subject) | Align a new subject to the common space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_iter` | <code>int, default=2</code> | Number of template refinement iterations | <code>2</code>
`auto_pad` | <code>bool, default=True</code> | Whether to automatically pad matrices to same size | <code>True</code>



######### Attributes####

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
fit(data: list[np.ndarray], parallel: str | None = 'cpu', n_jobs: int = -1) -> HyperAlignment
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

########## `transform`

```python
transform(data: list[np.ndarray], parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray]
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

########## `transform_subject`

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

(algorithms-local)=
###### `local`

LocalAlignment: Neighborhood-based functional alignment.

Implements searchlight and piecewise schemes from Bazeille et al. 2021.
Uses center-only aggregation to preserve orthogonality of local transforms.

**Classes:**

Name | Description
---- | -----------
[`LocalAlignment`](#algorithms-localalignment) | Local (neighborhood-based) functional alignment across subjects.



####### Attributes

####### Classes##

###### `LocalAlignment`

```python
LocalAlignment(scheme: str = 'searchlight', method: str = 'procrustes', radius_mm: float = 10.0, parcellation: Any | None = None, n_features: int | None = None, n_iter: int = 3, aggregation: str = 'center', parallel: str | None = 'cpu', n_jobs: int = -1, n_neighborhoods_batch: int | None = None, max_memory_gb: float = 4.0, transforms_: dict[int, list[np.ndarray]] | None = None, template_: dict[int, np.ndarray] | None = None, neighborhoods_: SphereNeighborhoods | dict[int, np.ndarray] | None = None, n_voxels_: int | None = None, mask_: Any | None = None, backend_: Backend | None = None) -> None
```

Local (neighborhood-based) functional alignment across subjects.

Learns alignment transforms within local neighborhoods (searchlight spheres
or parcels) and applies center-only aggregation to preserve orthogonality.

scheme : str, default='searchlight'
    Spatial scheme: 'searchlight' (overlapping spheres) or 'piecewise'
    (non-overlapping parcels).
method : str, default='procrustes'
    Alignment method: 'procrustes', 'srm', or 'hyperalignment'.
radius_mm : float, default=10.0
    Sphere radius in millimeters for searchlight scheme.
parcellation : Nifti1Image, optional
    Parcellation image for piecewise scheme. Required if scheme='piecewise'.
n_features : int, optional
    Number of features for SRM. None uses full Procrustes (preserves dims).
n_iter : int, default=3
    Number of iterations for alignment refinement.
aggregation : str, default='center'
    Aggregation method: 'center' (center-only, preserves orthogonality).
parallel : str, optional
    Parallelization: 'cpu', 'gpu', or None.
    - None: Single-threaded numpy
    - 'cpu': CPU parallelization with joblib
    - 'gpu': GPU acceleration via PyTorch (falls back to CPU if unavailable)
n_jobs : int, default=-1
    Number of jobs for CPU parallelization.

transforms_ : Dict[int, List[np.ndarray]]
    Per-neighborhood transforms. Keys are center voxel indices,
    values are lists of transform matrices (one per subject).
template_ : Dict[int, np.ndarray]
    Per-neighborhood templates used for alignment.
neighborhoods_ : SphereNeighborhoods or Dict
    Computed neighborhoods (searchlight or piecewise).
n_voxels_ : int
    Total number of voxels in the mask.
mask_ : Nifti1Image
    Brain mask used for fitting.

Examples:
>>> import numpy as np
>>> from nltools.algorithms.alignment import LocalAlignment
>>> # Create synthetic multi-subject data (voxels, samples)
>>> data = [np.random.randn(1000, 100) for _ in range(5)]
>>> la = LocalAlignment(scheme='searchlight', method='procrustes', radius_mm=10.0)
>>> la.fit(data, mask)
>>> aligned = la.transform(data)

Notes:
Based on Bazeille et al. 2021 "An empirical evaluation of functional
alignment using inter-subject decoding". Center-only aggregation is
used to preserve local orthogonality of transforms.

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Fit local alignment on multi-subject data.
[`fit_transform`](#algorithms-fit-transform) | Fit alignment and transform data in one step.
[`transform`](#algorithms-transform) | Apply local transforms to data.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`aggregation`](#algorithms-aggregation) | <code>[str](#str)</code> | 
`backend_` | <code>[Backend](#nltools.algorithms.backends.Backend) \| None</code> | 
`mask_` | <code>[Any](#typing.Any) \| None</code> | 
`max_memory_gb` | <code>[float](#float)</code> | 
`method` | <code>[str](#str)</code> | 
`n_features` | <code>[int](#int) \| None</code> | 
`n_iter` | <code>[int](#int)</code> | 
`n_jobs` | <code>[int](#int)</code> | 
`n_neighborhoods_batch` | <code>[int](#int) \| None</code> | 
`n_voxels_` | <code>[int](#int) \| None</code> | 
`neighborhoods_` | <code>[SphereNeighborhoods](#nltools.data.braindata.neighborhoods.SphereNeighborhoods) \| [dict](#dict)[[int](#int), [ndarray](#numpy.ndarray)] \| None</code> | 
`parallel` | <code>[str](#str) \| None</code> | 
`parcellation` | <code>[Any](#typing.Any) \| None</code> | 
`radius_mm` | <code>[float](#float)</code> | 
`scheme` | <code>[str](#str)</code> | 
`template_` | <code>[dict](#dict)[[int](#int), [ndarray](#numpy.ndarray)] \| None</code> | 
`transforms_` | <code>[dict](#dict)[[int](#int), [list](#list)[[ndarray](#numpy.ndarray)]] \| None</code> | 



######### Attributes####

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
mask_: Any | None = field(default=None, repr=False)
```

########## `max_memory_gb`

```python
max_memory_gb: float = 4.0
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

########## `parcellation`

```python
parcellation: Any | None = None
```

########## `radius_mm`

```python
radius_mm: float = 10.0
```

########## `scheme`

```python
scheme: str = 'searchlight'
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

data : List[np.ndarray]
    List of subject data arrays, each shape (n_voxels, n_samples).
    Subjects can have different numbers of samples - the underlying
    alignment methods (SRM, HyperAlignment) handle this via zero-padding.
mask : Nifti1Image
    Brain mask defining the voxel space.

self : LocalAlignment
    Fitted alignment model.

########## `fit_transform`

```python
fit_transform(data: list[np.ndarray], mask: nib.Nifti1Image) -> list[np.ndarray]
```

Fit alignment and transform data in one step.

data : List[np.ndarray]
    List of subject data arrays, each shape (n_voxels, n_samples).
mask : Nifti1Image
    Brain mask defining the voxel space.

List[np.ndarray]
    Aligned data for each subject.

########## `transform`

```python
transform(data: list[np.ndarray]) -> list[np.ndarray]
```

Apply local transforms to data.

For searchlight scheme with center-only aggregation: each voxel uses
the transform from the neighborhood where it was the center.

For piecewise scheme: all voxels in each parcel use the same transform.

data : List[np.ndarray]
    List of subject data arrays, each shape (n_voxels, n_samples).

List[np.ndarray]
    Aligned data for each subject, shape (n_voxels, n_samples).

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
DetSRM(n_iter: int = 10, features: int = 50, rand_seed: int = 0) -> None
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

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Compute the Deterministic Shared Response Model.
[`transform`](#algorithms-transform) | Use the model to transform data to the Shared Response subspace.
[`transform_subject`](#algorithms-transform-subject) | Transform a new subject using the existing model.



######### Attributes####

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
fit(X: list[np.ndarray], y: Any | None = None, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0) -> DetSRM
```

Compute the Deterministic Shared Response Model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples]</code> | Each element in the list contains the fMRI data of one subject. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": GPU acceleration (not yet implemented, falls back to CPU) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Maximum GPU memory budget in GB (default: 4.0). Only used when parallel="gpu". Defaults to 4.0. | <code>4.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`self` | <code>[DetSRM](#nltools.algorithms.alignment.srm.DetSRM)</code> | Fitted model

########## `transform`

```python
transform(X: list[np.ndarray], y: Any | None = None, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray]
```

Use the model to transform data to the Shared Response subspace.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples_i]</code> | Each element in the list contains the fMRI data of one subject. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": GPU acceleration (not yet implemented, falls back to CPU) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`s` | <code>list of 2D arrays, element i has shape=[features_i, samples_i]</code> | Shared responses from input data (X)

########## `transform_subject`

```python
transform_subject(X: np.ndarray) -> np.ndarray
```

Transform a new subject using the existing model.
The subject is assumed to have recieved equivalent stimulation

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>2D array, shape=[voxels, timepoints]</code> | The fMRI data of the new subject. | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`w` | <code>2D array, shape=[voxels, features]</code> | Orthogonal mapping `W_{new}` for new subject

######## `SRM`

```python
SRM(n_iter: int = 10, features: int = 50, rand_seed: int = 0) -> None
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

**Methods:**

Name | Description
---- | -----------
[`fit`](#algorithms-fit) | Compute the probabilistic Shared Response Model.
[`transform`](#algorithms-transform) | Use the model to transform matrix to Shared Response space.
[`transform_subject`](#algorithms-transform-subject) | Transform a new subject using the existing model.



######### Attributes####

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
fit(X: list[np.ndarray], y: Any | None = None, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, pad_samples: bool = True) -> SRM
```

Compute the probabilistic Shared Response Model.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples]</code> | Each element in the list contains the fMRI data of one subject. Subjects can have different numbers of samples if pad_samples=True. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": GPU acceleration (not yet implemented, falls back to CPU) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Maximum GPU memory budget in GB (default: 4.0). Only used when parallel="gpu". Defaults to 4.0. | <code>4.0</code>
`pad_samples` | <code>[bool](#bool)</code> | If True (default), automatically zero-pad subjects with fewer samples to match the longest subject. This allows fitting SRM on data with unequal numbers of time points across subjects. | <code>True</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`self` | <code>[SRM](#nltools.algorithms.alignment.srm.SRM)</code> | Fitted model

########## `transform`

```python
transform(X: list[np.ndarray], y: Any | None = None, parallel: str | None = 'cpu', n_jobs: int = -1) -> list[np.ndarray | None]
```

Use the model to transform matrix to Shared Response space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>list of 2D arrays, element i has shape=[voxels_i, samples_i]</code> | Each element in the list contains the fMRI data of one subject. Note that number of voxels and samples can vary across subjects. | *required*
`y` | <code>[Any](#typing.Any) \| None</code> | not used (as it is unsupervised learning) | <code>None</code>
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU parallelization via joblib (default, multi-subject processing) - "gpu": GPU acceleration (not yet implemented, falls back to CPU) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = auto-detect based on memory). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`s` | <code>list of 2D arrays, element i has shape=[features_i, samples_i]</code> | Shared responses from input data (X)

########## `transform_subject`

```python
transform_subject(X: np.ndarray) -> np.ndarray
```

Transform a new subject using the existing model.
The subject is assumed to have recieved equivalent stimulation

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`X` | <code>2D array, shape=[voxels, timepoints]</code> | The fMRI data of the new subject. | *required*

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
[`auto_select_backend`](#algorithms-auto-select-backend) | Automatically select backend based on problem size.
[`check_gpu_available`](#algorithms-check-gpu-available) | Check if GPU acceleration is available.
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

**Returns:**

Type | Description
---- | -----------
 | Backend array (numpy ndarray or torch Tensor).

######## `asarray_like`

```python
asarray_like(x, ref)
```

Convert *x* to an array matching *ref*'s dtype (and device for torch).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`x` |  | Input data. | *required*
`ref` |  | Reference array whose dtype/device to match. | *required*

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

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`*inputs` |  | Arrays, lists of arrays, or None. | <code>()</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`list` |  | Converted arrays in the same order as inputs.

######## `concatenate`

```python
concatenate(arrays, axis = 0)
```

Concatenate arrays along an axis.

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

**Returns:**

Type | Description
---- | -----------
 | str or None: e.g. "float32", "float64", or None if input was None.

######## `expand_dims`

```python
expand_dims(array, axis)
```

Insert a new axis.

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

**Returns:**

Name | Type | Description
---- | ---- | -----------
`array` |  | Result of A @ B

######## `ones_like`

```python
ones_like(array, shape = None, dtype = None, device = None)
```

Create ones array, optionally with a different shape.

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

**Returns:**

Name | Type | Description
---- | ---- | -----------
`tuple` |  | (U, s, Vt) where: - U (array): Left singular vectors - s (array): Singular values - Vt (array): Right singular vectors (transposed)

######## `to_cpu`

```python
to_cpu(array)
```

Transfer array to CPU. No-op for numpy.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Input array or tensor. | *required*

**Returns:**

Type | Description
---- | -----------
 | Array on CPU.

######## `to_device`

```python
to_device(arr: np.ndarray)
```

Transfer array to backend device.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`arr` | <code>[ndarray](#numpy.ndarray)</code> | Input numpy array | *required*

**Returns:**

Name | Type | Description
---- | ---- | -----------
`array` |  | Array on device (numpy array or torch tensor)

######## `to_gpu`

```python
to_gpu(array, device = None)
```

Transfer array to GPU. No-op for numpy.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Input array or tensor. | *required*
`device` |  | Target device (defaults to backend's device). | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | Array on GPU device.

######## `to_numpy`

```python
to_numpy(arr)
```

Convert array back to NumPy.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`arr` | <code>[ndarray](#numpy.ndarray) or [Tensor](#torch.Tensor)</code> | Array to convert | *required*

**Returns:**

Type | Description
---- | -----------
 | np.ndarray: NumPy array

######## `zeros_like`

```python
zeros_like(array, shape = None, dtype = None, device = None)
```

Create zeros array, optionally with a different shape.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` |  | Reference array for dtype inference. | *required*
`shape` |  | Output shape. If None, uses array.shape. | <code>None</code>
`dtype` |  | Output dtype. If None, uses array.dtype. | <code>None</code>
`device` |  | Target device (torch only). If None, uses array's device. | <code>None</code>



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

tr:

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
dhrf : array of shape(length / t_r * oversampling), dtype=float
      dhrf sampling on the oversampled time grid

###### `glover_hrf`

```python
glover_hrf(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Implement the Glover :term:`HRF` model.

Parameters
----------
t_r : `float`
    :term:`Repetition time<TR>`, in seconds (sampling period).

tr:

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
hrf : array of shape(length / t_r * oversampling, dtype=float)
     :term:`HRF` sampling on the oversampled time grid.

###### `glover_time_derivative`

```python
glover_time_derivative(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Implement the Glover time derivative :term:`HRF` (dhrf) model.

Parameters
----------
t_r : `float`
    :term:`Repetition time<TR>`, in seconds (sampling period).

tr:

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
dhrf : array of shape(length / t_r), dtype=float
      dhrf sampling on the provided grid

###### `spm_dispersion_derivative`

```python
spm_dispersion_derivative(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Implement the :term:`SPM` dispersion derivative :term:`HRF` model.

Parameters
----------
t_r : `float`
    :term:`Repetition time<TR>`, in seconds (sampling period).

tr:

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
dhrf : array of shape(length / tr * oversampling), dtype=float
      dhrf sampling on the oversampled time grid

###### `spm_hrf`

```python
spm_hrf(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Implement the :term:`SPM` :term:`HRF` model.

Parameters
----------
t_r : `float`
    :term:`Repetition time<TR>`, in seconds (sampling period).

tr:

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
hrf : array of shape(length / t_r * oversampling, dtype=float)
     :term:`HRF` sampling on the oversampled time grid

###### `spm_time_derivative`

```python
spm_time_derivative(t_r, oversampling = 50, time_length = 32.0, onset = 0.0)
```

Implement the :term:`SPM` time derivative :term:`HRF` (dhrf) model.

Parameters
----------
t_r : `float`
    :term:`Repetition time<TR>`, in seconds (sampling period).

tr:

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
dhrf : array of shape(length / t_r, dtype=float)
      dhrf sampling on the provided grid

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
- Drop-in replacement for nltools.stats functions

</details>

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
>>> result = one_sample_permutation_test(data, n_permute=10000, backend='torch')
>>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")
```

<details class="performance" open markdown="1">
<summary>Performance</summary>

- CPU (NumPy): Good for small problems (< 5K permutations)
- GPU (PyTorch): Excellent for large problems (> 5K permutations)
- CPU Parallel (joblib): Efficient fallback when GPU unavailable
- Auto-selection: Use backend='auto' for best performance

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

**Modules:**

Name | Description
---- | -----------
[`bootstrap`](#algorithms-bootstrap) | Bootstrap inference utilities with CPU/GPU support.
[`correlation`](#algorithms-correlation) | Correlation permutation test implementations.
[`isc`](#algorithms-isc) | Intersubject Correlation (ISC) with GPU-Accelerated Permutation Testing.
[`matrix`](#algorithms-matrix) | Matrix permutation test implementations (Mantel test).
[`one_sample`](#algorithms-one-sample) | One-sample permutation test implementations.
[`timeseries`](#algorithms-timeseries) | Time-series permutation test implementations.
[`two_sample`](#algorithms-two-sample) | Two-sample permutation test implementations.
[`utils`](#algorithms-utils) | Utility functions for permutation testing.
[`validation`](#algorithms-validation) | Shared validation utilities for algorithms module.

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

**Methods:**

Name | Description
---- | -----------
[`get_results`](#algorithms-get-results) | Compute final bootstrap statistics.
`update` | Update statistics with a new bootstrap sample.

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
get_results() -> dict[str, np.ndarray]
```

Compute final bootstrap statistics.

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | Dictionary containing:
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'mean': Bootstrap mean
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'std': Bootstrap standard deviation
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'Z': Z-scores (mean/std)
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'p': Two-tailed p-values
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'ci_lower': Lower confidence bound
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'ci_upper': Upper confidence bound
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'samples': All samples (only if save_samples=True)

Examples:
**Basic usage:**
>>> stats = OnlineBootstrapStats(shape=(100,), save_samples=False)
>>> for i in range(1000):
...     sample = np.random.randn(100)
...     stats.update(sample)
>>> results = stats.get_results()
>>> print(results.keys())
dict_keys(['mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper'])

**Usage:**
>>> from nltools.algorithms.inference.bootstrap import OnlineBootstrapStats
>>> from nltools.data import BrainData
>>>
>>> # Initialize with shape matching your data
>>> stats = OnlineBootstrapStats(
...     shape=(bootstrap_samples.shape[1],),  # Number of voxels/features
...     save_samples=False,  # Set True if you need 'samples' key
...     percentiles=(2.5, 97.5)  # For confidence intervals
... )
>>>
>>> # Update with each bootstrap sample
>>> for sample in bootstrap_samples:  # Iterate over samples
...     stats.update(sample.data)  # Pass 1D array of voxel values
>>>
>>> # Get results (equivalent to summarize_bootstrap output)
>>> result = stats.get_results()
>>> # Returns: {'mean': array, 'std': array, 'Z': array, 'p': array,
>>> #           'ci_lower': array, 'ci_upper': array}
>>>
>>> # Convert to BrainData if needed (reproduce old API format)
>>> mean_brain = bootstrap_samples[0].copy()
>>> mean_brain.data = result['mean']
>>> z_brain = bootstrap_samples[0].copy()
>>> z_brain.data = result['Z']
>>> p_brain = bootstrap_samples[0].copy()
>>> p_brain.data = result['p']
>>>
>>> # Result equivalent to old summarize_bootstrap():
>>> equivalent_result = {
...     'mean': mean_brain,
...     'Z': z_brain,
...     'p': p_brain
... }
>>> # Optionally include samples if save_samples=True:
>>> if 'samples' in result:
...     equivalent_result['samples'] = result['samples']

######## `update`

```python
update(sample: np.ndarray) -> None
```

Update statistics with a new bootstrap sample.

Uses Welford's algorithm for numerical stability.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sample` | <code>[ndarray](#numpy.ndarray)</code> | New bootstrap sample with shape matching self.shape. | *required*



##### Methods

(algorithms-circle-shift)=
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

(algorithms-correlation-permutation-test)=
###### `correlation_permutation_test`

```python
correlation_permutation_test(data1: np.ndarray, data2: np.ndarray, n_permute: int = 5000, metric: str = 'pearson', tail: int | str = 2, return_null: bool = False, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None) -> dict
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
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type (default: 2) - 'two' or 2: Two-tailed test (r != 0) - 'upper' or 1: One-tailed upper (r > 0, positive correlation) - 'lower' or -1: One-tailed lower (r < 0, negative correlation) For MCP correction (FDR), use 'upper' or 'lower' for consistent direction. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`parallel` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when parallel='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Maximum GPU memory to use in GB (default: 4.0) Controls automatic batching to prevent OOM errors. Only used with parallel='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>4.0</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'correlation' (float or np.ndarray): Observed correlation(s) - 'p' (float or np.ndarray): P-value(s) - 'null_dist' (np.ndarray): Null distribution (if return_null=True) - 'parallel' (str): Parallelization method used

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
>>> result = correlation_permutation_test(data1, data2, n_permute=5000, parallel='gpu')
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (parallel='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
    - Pearson correlation: Fully vectorized across all features (5-20× speedup for multi-feature)
    - Spearman/Kendall: Only supported with parallel='cpu' or parallel=None (GPU not yet implemented)
- Single-threaded (parallel=None): Use for small problems or debugging
- For multi-feature data, each feature pair tested independently
- Kendall is O(n^2) complexity, slower than Pearson/Spearman for large samples

</details>

(algorithms-distance-correlation)=
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

(algorithms-double-center)=
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

(algorithms-isc-group-permutation-test)=
###### `isc_group_permutation_test`

```python
isc_group_permutation_test(group1: np.ndarray, group2: np.ndarray, n_permute: int = 5000, metric: Literal['median', 'mean'] = 'median', method: Literal['permute', 'bootstrap'] = 'permute', summary_statistic: Literal['leave-one-out', 'pairwise'] = 'pairwise', ci_percentile: float = 95, tail: Literal[1, 2] = 2, parallel: Literal['cpu', 'gpu'] | None = 'cpu', n_jobs: int = -1, random_state: int | None = None, return_null: bool = False, progress_bar: bool = True, exclude_self_corr: bool = True, sim_metric: str = 'correlation') -> dict[str, Any]
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
`metric` | <code>[Literal](#typing.Literal)['median', 'mean']</code> | Summary statistic for aggregating ISC values: - 'median': Direct median (robust to outliers) - 'mean': Fisher z-transformed mean (unbiased averaging) Defaults to 'median'. | <code>'median'</code>
`method` | <code>[Literal](#typing.Literal)['permute', 'bootstrap']</code> | Resampling method for p-value computation: - 'permute': Subject-wise permutation (combines groups, permutes labels) - 'bootstrap': Subject-wise bootstrap (resamples within each group) Defaults to 'permute'. | <code>'permute'</code>
`summary_statistic` | <code>[Literal](#typing.Literal)['leave-one-out', 'pairwise']</code> | ISC computation method: - 'pairwise': Average all pairwise correlations - 'leave-one-out': Correlate each subject with mean of others Defaults to 'pairwise'. | <code>'pairwise'</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95 for 95% CI). Defaults to 95. | <code>95</code>
`tail` | <code>[Literal](#typing.Literal)[1, 2]</code> | One-tailed (1) or two-tailed (2) p-value. Defaults to 2. | <code>2</code>
`parallel` | <code>[Literal](#typing.Literal)['cpu', 'gpu'] \| None</code> | Parallelization method: - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (10-30× speedup for voxel-wise LOO) - None: Single-threaded NumPy (for debugging/small problems) Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when parallel='cpu'. Defaults to -1. | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, return null distribution in result dict. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during bootstrap/permutation. Defaults to True. | <code>True</code>
`exclude_self_corr` | <code>[bool](#bool)</code> | Mask self-correlations in bootstrap (pairwise only). Defaults to True. | <code>True</code>
`sim_metric` | <code>[str](#str)</code> | Similarity metric for pairwise ISC computation. See sklearn.metrics.pairwise_distances for valid options. Only applies when summary_statistic='pairwise'. Defaults to 'correlation'. | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with the following keys:
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'isc_group_difference': Observed ISC difference (float or array per voxel)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'p': P-value (Phipson-Smyth corrected)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'ci': Confidence interval tuple (lower, upper)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'parallel': Parallelization method used
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'null_dist': (optional) Bootstrap/permutation distribution

Examples:
>>> # Single-feature ISC group comparison
>>> group1 = np.random.randn(100, 10)  # 10 subjects
>>> group2 = np.random.randn(100, 10)
>>> result = isc_group_permutation_test(group1, group2, n_permute=1000)
>>> print(f"ISC difference: {result['isc_group_difference']:.3f}, p: {result['p']:.3f}")

>>> # Voxel-wise ISC group comparison with GPU acceleration
>>> group1_voxels = np.random.randn(100, 10, 5000)  # 5K voxels
>>> group2_voxels = np.random.randn(100, 10, 5000)
>>> result = isc_group_permutation_test(
...     group1_voxels,
...     group2_voxels,
...     summary_statistic='leave-one-out',
...     parallel='gpu',  # GPU for LOO computation
...     n_permute=5000
... )
>>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")

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
###### `isc_permutation_test`

```python
isc_permutation_test(data: np.ndarray, n_permute: int = 5000, metric: Literal['median', 'mean'] = 'median', summary_statistic: Literal['leave-one-out', 'pairwise'] = 'pairwise', method: Literal['bootstrap', 'circle_shift', 'phase_randomize'] = 'bootstrap', ci_percentile: float = 95, tail: Literal[1, 2] = 2, return_null: bool = False, progress_bar: bool = True, exclude_self_corr: bool = True, sim_metric: str = 'correlation', parallel: Literal['cpu', 'gpu'] | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None) -> dict[str, Any]
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
`metric` | <code>[Literal](#typing.Literal)['median', 'mean']</code> | Summary statistic to aggregate ISC values. - 'median': Direct median (robust to outliers) - 'mean': Fisher z-transformed mean (unbiased averaging) Defaults to 'median'. | <code>'median'</code>
`summary_statistic` | <code>[Literal](#typing.Literal)['leave-one-out', 'pairwise']</code> | ISC computation method. Options: - 'leave-one-out': Correlate each subject with mean of others. O(n_subjects), unbiased, recommended by Chen et al. 2016. - 'pairwise': Average all pairwise correlations. O(n_subjects²), captures full correlation structure. Note: These methods are statistically different and monotonically but non-linearly related (see Chen et al. 2016, Figure 3). Defaults to 'pairwise'. | <code>'pairwise'</code>
`method` | <code>[Literal](#typing.Literal)['bootstrap', 'circle_shift', 'phase_randomize']</code> | Resampling method for p-value computation: - 'bootstrap': Subject-wise bootstrap (default, Chen et al. 2016) - 'circle_shift': Circular time-series shift (preserves autocorrelation) - 'phase_randomize': FFT phase randomization (preserves power spectrum) Defaults to 'bootstrap'. | <code>'bootstrap'</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95 for 95% CI). Defaults to 95. | <code>95</code>
`tail` | <code>[Literal](#typing.Literal)[1, 2]</code> | One-tailed (1) or two-tailed (2) p-value. Defaults to 2. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return bootstrap/permutation distribution in result dict. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during bootstrap/permutation. Defaults to True. | <code>True</code>
`exclude_self_corr` | <code>[bool](#bool)</code> | If True, mask self-correlations (perfect correlations from duplicate subjects in bootstrap samples) as NaN. If False, include them in the summary statistic. Only applies when method='bootstrap' and summary_statistic='pairwise'. Defaults to True. | <code>True</code>
`sim_metric` | <code>[str](#str)</code> | Similarity metric for pairwise ISC computation. See sklearn.metrics.pairwise_distances for valid options. Only applies when summary_statistic='pairwise'. For 'correlation', uses optimized np.corrcoef. Other metrics use pairwise_distances. Defaults to 'correlation'. | <code>'correlation'</code>
`parallel` | <code>[Literal](#typing.Literal)['cpu', 'gpu'] \| None</code> | Parallelization method: - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (10-30× speedup for voxel-wise LOO) - None: Single-threaded NumPy (for debugging/small problems) Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when parallel='cpu'. Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel='gpu'). Defaults to 4. | <code>4.0</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with the following keys:
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'isc': Observed ISC value (float or array per voxel)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'p': P-value (Phipson-Smyth corrected)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'ci': Confidence interval tuple (lower, upper)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'parallel': Parallelization method used
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'null_dist': (optional) Bootstrap/permutation distribution

Examples:
>>> # Single-feature ISC
>>> data = np.random.randn(100, 10)  # 100 timepoints, 10 subjects
>>> result = isc_permutation_test(data, n_permute=1000)
>>> print(f"ISC: {result['isc']:.3f}, p: {result['p']:.3f}")

>>> # Voxel-wise ISC with GPU acceleration
>>> data_voxels = np.random.randn(100, 50, 5000)  # 5K voxels
>>> result = isc_permutation_test(
...     data_voxels,
...     summary_statistic='leave-one-out',
...     parallel='gpu',  # GPU for LOO computation
...     n_permute=5000
... )
>>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")

>>> # Compare LOO vs pairwise
>>> result_loo = isc_permutation_test(data, summary_statistic='leave-one-out')
>>> result_pair = isc_permutation_test(data, summary_statistic='pairwise')
>>> print(f"LOO: {result_loo['isc']:.3f}, Pairwise: {result_pair['isc']:.3f}")

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

(algorithms-matrix-permutation-test)=
###### `matrix_permutation_test`

```python
matrix_permutation_test(data1: np.ndarray, data2: np.ndarray, n_permute: int = 5000, metric: str = 'pearson', how: str = 'upper', include_diag: bool = False, tail: int | str = 2, parallel: str | None = 'cpu', n_jobs: int = -1, return_null: bool = False, random_state: int | None = None) -> dict
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
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type (default: 2) - 'two' or 2: Two-tailed test (r != 0) - 'upper' or 1: One-tailed upper (r > 0) - 'lower' or -1: One-tailed lower (r < 0) | <code>2</code>
`parallel` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel workers, -1 = all cores (default: -1) Only used when parallel='cpu' | <code>-1</code>
`return_null` | <code>[bool](#bool)</code> | Return null distribution (default: False) | <code>False</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'correlation' (float): Observed correlation coefficient - 'p' (float): P-value using Phipson-Smyth correction - 'parallel' (str): Parallelization method used ('cpu' or None) - 'null_dist' (np.ndarray): Null distribution (if return_null=True)

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
one_sample_permutation_test(data: np.ndarray, n_permute: int = 5000, tail: int | str = 2, return_null: bool = False, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None) -> dict
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
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type (default: 2) - 'two' or 2: Two-tailed test (mean != 0) - 'upper' or 1: One-tailed upper (mean > 0) - 'lower' or -1: One-tailed lower (mean < 0) For MCP correction (FDR), use 'upper' or 'lower' for consistent direction. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`parallel` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when parallel='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Maximum GPU memory to use in GB (default: 4.0) Controls automatic batching to prevent OOM errors. Only used with parallel='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>4.0</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'mean' (float or np.ndarray): Observed mean(s) - 'p' (float or np.ndarray): P-value(s) - 'null_dist' (np.ndarray): Null distribution (if return_null=True) - 'parallel' (str): Parallelization method used

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
>>> result = one_sample_permutation_test(data, n_permute=5000, parallel='gpu')
>>> result['mean'].shape
(10000,)
>>> result['p'].shape
(10000,)
```

```pycon
>>> # Single-threaded (for debugging)
>>> result = one_sample_permutation_test(data, n_permute=5000, parallel=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (parallel='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
- Single-threaded (parallel=None): Use for small problems or debugging
- For voxel-wise tests, each voxel tested independently
- Progress bars show completion for both CPU parallel and GPU batched modes

</details>

(algorithms-phase-randomize)=
###### `phase_randomize`

```python
phase_randomize(data: np.ndarray, backend: str | None = None, random_state: int | np.random.RandomState | None = None) -> np.ndarray
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
`backend` | <code>[str](#str) \| None</code> | Computation backend ('numpy' or 'torch'). - 'numpy': CPU implementation using NumPy FFT (default, float64 precision) - 'torch': GPU implementation using PyTorch FFT (float32 precision, faster) - None: Defaults to 'numpy' | <code>None</code>
`random_state` | <code>[int](#int) \| [RandomState](#numpy.random.RandomState) \| None</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Phase-randomized data with same shape as input

<details class="notes" open markdown="1">
<summary>Notes</summary>

- **CRITICAL**: Preserves power spectrum exactly (within numerical precision)
- GPU acceleration: Use `backend='torch'` for GPU-accelerated FFT (5-20× faster for large data)
- Precision: NumPy backend uses float64, PyTorch backend uses float32
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
>>> x_rand_gpu = phase_randomize(x_large, backend='torch', random_state=42)
```

(algorithms-timeseries-correlation-permutation-test)=
###### `timeseries_correlation_permutation_test`

```python
timeseries_correlation_permutation_test(data1: np.ndarray, data2: np.ndarray, method: Literal['circle_shift', 'phase_randomize'] = 'circle_shift', n_permute: int = 5000, metric: Literal['pearson', 'spearman', 'kendall'] = 'pearson', tail: int = 2, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, return_null: bool = False, random_state: int | np.random.RandomState | None = None) -> dict
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
`tail` | <code>[int](#int)</code> | Test type (1=one-tailed, 2=two-tailed) | <code>2</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs (-1 = all cores) Only used when parallel='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Maximum GPU memory to use in GB (default: 4.0) Controls automatic batching to prevent OOM errors. Only used with parallel='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>4.0</code>
`return_null` | <code>[bool](#bool)</code> | Whether to return null distribution | <code>False</code>
`random_state` | <code>[int](#int) \| [RandomState](#numpy.random.RandomState) \| None</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with keys: - 'correlation': Observed correlation coefficient - 'p': P-value - 'null_dist': (if return_null=True) Null distribution - 'parallel': Parallelization method used

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
...     x, y, method='phase_randomize', parallel='gpu', n_permute=5000
... )
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (parallel='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): 5-20× faster for large problems (n_samples > 1000)
- Single-threaded (parallel=None): Use for small problems or debugging
- For independent data, use regular correlation_permutation_test
- circle_shift is faster and suitable for most fMRI time series
- phase_randomize preserves power spectrum exactly (tests nonlinearity)
- Only data1 is randomized; data2 remains fixed to test correlation
- phase_randomize benefits most from GPU (FFT acceleration)

</details>

(algorithms-two-sample-permutation-test)=
###### `two_sample_permutation_test`

```python
two_sample_permutation_test(data1: np.ndarray, data2: np.ndarray, n_permute: int = 5000, tail: int | str = 2, return_null: bool = False, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None) -> dict
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
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type (default: 2) - 'two' or 2: Two-tailed test (mean1 != mean2) - 'upper' or 1: One-tailed upper (mean1 > mean2) - 'lower' or -1: One-tailed lower (mean1 < mean2) For MCP correction (FDR), use 'upper' or 'lower' for consistent direction. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`parallel` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when parallel='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Maximum GPU memory to use in GB (default: 4.0) Controls automatic batching to prevent OOM errors. Only used with parallel='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>4.0</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'mean_diff' (float or np.ndarray): Observed mean difference (data1 - data2) - 'p' (float or np.ndarray): P-value(s) - 'null_dist' (np.ndarray): Null distribution (if return_null=True) - 'parallel' (str): Parallelization method used

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
>>> result = two_sample_permutation_test(data1, data2, n_permute=5000, parallel='gpu')
>>> result['mean_diff'].shape
(10000,)
>>> result['p'].shape
(10000,)
```

```pycon
>>> # Single-threaded (for debugging)
>>> result = two_sample_permutation_test(data1, data2, n_permute=5000, parallel=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (parallel='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
- Single-threaded (parallel=None): Use for small problems or debugging
- For voxel-wise tests, each voxel tested independently
- Group sizes can be unequal

</details>

(algorithms-u-center)=
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

**Classes:**

Name | Description
---- | -----------
[`OnlineBootstrapStats`](#algorithms-onlinebootstrapstats) | Memory-efficient online statistics aggregator for bootstrap samples.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`FITTED_METHODS`](#algorithms-fitted-methods) |  | 
`SIMPLE_METHODS` |  | 



####### Attributes##

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

**Methods:**

Name | Description
---- | -----------
[`get_results`](#algorithms-get-results) | Compute final bootstrap statistics.
`update` | Update statistics with a new bootstrap sample.

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
get_results() -> dict[str, np.ndarray]
```

Compute final bootstrap statistics.

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | Dictionary containing:
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'mean': Bootstrap mean
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'std': Bootstrap standard deviation
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'Z': Z-scores (mean/std)
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'p': Two-tailed p-values
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'ci_lower': Lower confidence bound
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'ci_upper': Upper confidence bound
<code>[dict](#dict)[[str](#str), [ndarray](#numpy.ndarray)]</code> | - 'samples': All samples (only if save_samples=True)

Examples:
**Basic usage:**
>>> stats = OnlineBootstrapStats(shape=(100,), save_samples=False)
>>> for i in range(1000):
...     sample = np.random.randn(100)
...     stats.update(sample)
>>> results = stats.get_results()
>>> print(results.keys())
dict_keys(['mean', 'std', 'Z', 'p', 'ci_lower', 'ci_upper'])

**Usage:**
>>> from nltools.algorithms.inference.bootstrap import OnlineBootstrapStats
>>> from nltools.data import BrainData
>>>
>>> # Initialize with shape matching your data
>>> stats = OnlineBootstrapStats(
...     shape=(bootstrap_samples.shape[1],),  # Number of voxels/features
...     save_samples=False,  # Set True if you need 'samples' key
...     percentiles=(2.5, 97.5)  # For confidence intervals
... )
>>>
>>> # Update with each bootstrap sample
>>> for sample in bootstrap_samples:  # Iterate over samples
...     stats.update(sample.data)  # Pass 1D array of voxel values
>>>
>>> # Get results (equivalent to summarize_bootstrap output)
>>> result = stats.get_results()
>>> # Returns: {'mean': array, 'std': array, 'Z': array, 'p': array,
>>> #           'ci_lower': array, 'ci_upper': array}
>>>
>>> # Convert to BrainData if needed (reproduce old API format)
>>> mean_brain = bootstrap_samples[0].copy()
>>> mean_brain.data = result['mean']
>>> z_brain = bootstrap_samples[0].copy()
>>> z_brain.data = result['Z']
>>> p_brain = bootstrap_samples[0].copy()
>>> p_brain.data = result['p']
>>>
>>> # Result equivalent to old summarize_bootstrap():
>>> equivalent_result = {
...     'mean': mean_brain,
...     'Z': z_brain,
...     'p': p_brain
... }
>>> # Optionally include samples if save_samples=True:
>>> if 'samples' in result:
...     equivalent_result['samples'] = result['samples']

########## `update`

```python
update(sample: np.ndarray) -> None
```

Update statistics with a new bootstrap sample.

Uses Welford's algorithm for numerical stability.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`sample` | <code>[ndarray](#numpy.ndarray)</code> | New bootstrap sample with shape matching self.shape. | *required*



####### Functions

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
correlation_permutation_test(data1: np.ndarray, data2: np.ndarray, n_permute: int = 5000, metric: str = 'pearson', tail: int | str = 2, return_null: bool = False, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None) -> dict
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
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type (default: 2) - 'two' or 2: Two-tailed test (r != 0) - 'upper' or 1: One-tailed upper (r > 0, positive correlation) - 'lower' or -1: One-tailed lower (r < 0, negative correlation) For MCP correction (FDR), use 'upper' or 'lower' for consistent direction. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`parallel` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when parallel='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Maximum GPU memory to use in GB (default: 4.0) Controls automatic batching to prevent OOM errors. Only used with parallel='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>4.0</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'correlation' (float or np.ndarray): Observed correlation(s) - 'p' (float or np.ndarray): P-value(s) - 'null_dist' (np.ndarray): Null distribution (if return_null=True) - 'parallel' (str): Parallelization method used

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
>>> result = correlation_permutation_test(data1, data2, n_permute=5000, parallel='gpu')
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (parallel='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
    - Pearson correlation: Fully vectorized across all features (5-20× speedup for multi-feature)
    - Spearman/Kendall: Only supported with parallel='cpu' or parallel=None (GPU not yet implemented)
- Single-threaded (parallel=None): Use for small problems or debugging
- For multi-feature data, each feature pair tested independently
- Kendall is O(n^2) complexity, slower than Pearson/Spearman for large samples

</details>

(algorithms-isc)=
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
isc_group_permutation_test(group1: np.ndarray, group2: np.ndarray, n_permute: int = 5000, metric: Literal['median', 'mean'] = 'median', method: Literal['permute', 'bootstrap'] = 'permute', summary_statistic: Literal['leave-one-out', 'pairwise'] = 'pairwise', ci_percentile: float = 95, tail: Literal[1, 2] = 2, parallel: Literal['cpu', 'gpu'] | None = 'cpu', n_jobs: int = -1, random_state: int | None = None, return_null: bool = False, progress_bar: bool = True, exclude_self_corr: bool = True, sim_metric: str = 'correlation') -> dict[str, Any]
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
`metric` | <code>[Literal](#typing.Literal)['median', 'mean']</code> | Summary statistic for aggregating ISC values: - 'median': Direct median (robust to outliers) - 'mean': Fisher z-transformed mean (unbiased averaging) Defaults to 'median'. | <code>'median'</code>
`method` | <code>[Literal](#typing.Literal)['permute', 'bootstrap']</code> | Resampling method for p-value computation: - 'permute': Subject-wise permutation (combines groups, permutes labels) - 'bootstrap': Subject-wise bootstrap (resamples within each group) Defaults to 'permute'. | <code>'permute'</code>
`summary_statistic` | <code>[Literal](#typing.Literal)['leave-one-out', 'pairwise']</code> | ISC computation method: - 'pairwise': Average all pairwise correlations - 'leave-one-out': Correlate each subject with mean of others Defaults to 'pairwise'. | <code>'pairwise'</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95 for 95% CI). Defaults to 95. | <code>95</code>
`tail` | <code>[Literal](#typing.Literal)[1, 2]</code> | One-tailed (1) or two-tailed (2) p-value. Defaults to 2. | <code>2</code>
`parallel` | <code>[Literal](#typing.Literal)['cpu', 'gpu'] \| None</code> | Parallelization method: - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (10-30× speedup for voxel-wise LOO) - None: Single-threaded NumPy (for debugging/small problems) Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when parallel='cpu'. Defaults to -1. | <code>-1</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>
`return_null` | <code>[bool](#bool)</code> | If True, return null distribution in result dict. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during bootstrap/permutation. Defaults to True. | <code>True</code>
`exclude_self_corr` | <code>[bool](#bool)</code> | Mask self-correlations in bootstrap (pairwise only). Defaults to True. | <code>True</code>
`sim_metric` | <code>[str](#str)</code> | Similarity metric for pairwise ISC computation. See sklearn.metrics.pairwise_distances for valid options. Only applies when summary_statistic='pairwise'. Defaults to 'correlation'. | <code>'correlation'</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with the following keys:
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'isc_group_difference': Observed ISC difference (float or array per voxel)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'p': P-value (Phipson-Smyth corrected)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'ci': Confidence interval tuple (lower, upper)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'parallel': Parallelization method used
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'null_dist': (optional) Bootstrap/permutation distribution

Examples:
>>> # Single-feature ISC group comparison
>>> group1 = np.random.randn(100, 10)  # 10 subjects
>>> group2 = np.random.randn(100, 10)
>>> result = isc_group_permutation_test(group1, group2, n_permute=1000)
>>> print(f"ISC difference: {result['isc_group_difference']:.3f}, p: {result['p']:.3f}")

>>> # Voxel-wise ISC group comparison with GPU acceleration
>>> group1_voxels = np.random.randn(100, 10, 5000)  # 5K voxels
>>> group2_voxels = np.random.randn(100, 10, 5000)
>>> result = isc_group_permutation_test(
...     group1_voxels,
...     group2_voxels,
...     summary_statistic='leave-one-out',
...     parallel='gpu',  # GPU for LOO computation
...     n_permute=5000
... )
>>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")

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
isc_permutation_test(data: np.ndarray, n_permute: int = 5000, metric: Literal['median', 'mean'] = 'median', summary_statistic: Literal['leave-one-out', 'pairwise'] = 'pairwise', method: Literal['bootstrap', 'circle_shift', 'phase_randomize'] = 'bootstrap', ci_percentile: float = 95, tail: Literal[1, 2] = 2, return_null: bool = False, progress_bar: bool = True, exclude_self_corr: bool = True, sim_metric: str = 'correlation', parallel: Literal['cpu', 'gpu'] | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None) -> dict[str, Any]
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
`metric` | <code>[Literal](#typing.Literal)['median', 'mean']</code> | Summary statistic to aggregate ISC values. - 'median': Direct median (robust to outliers) - 'mean': Fisher z-transformed mean (unbiased averaging) Defaults to 'median'. | <code>'median'</code>
`summary_statistic` | <code>[Literal](#typing.Literal)['leave-one-out', 'pairwise']</code> | ISC computation method. Options: - 'leave-one-out': Correlate each subject with mean of others. O(n_subjects), unbiased, recommended by Chen et al. 2016. - 'pairwise': Average all pairwise correlations. O(n_subjects²), captures full correlation structure. Note: These methods are statistically different and monotonically but non-linearly related (see Chen et al. 2016, Figure 3). Defaults to 'pairwise'. | <code>'pairwise'</code>
`method` | <code>[Literal](#typing.Literal)['bootstrap', 'circle_shift', 'phase_randomize']</code> | Resampling method for p-value computation: - 'bootstrap': Subject-wise bootstrap (default, Chen et al. 2016) - 'circle_shift': Circular time-series shift (preserves autocorrelation) - 'phase_randomize': FFT phase randomization (preserves power spectrum) Defaults to 'bootstrap'. | <code>'bootstrap'</code>
`ci_percentile` | <code>[float](#float)</code> | Confidence interval percentile (e.g., 95 for 95% CI). Defaults to 95. | <code>95</code>
`tail` | <code>[Literal](#typing.Literal)[1, 2]</code> | One-tailed (1) or two-tailed (2) p-value. Defaults to 2. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return bootstrap/permutation distribution in result dict. Defaults to False. | <code>False</code>
`progress_bar` | <code>[bool](#bool)</code> | Show progress bar during bootstrap/permutation. Defaults to True. | <code>True</code>
`exclude_self_corr` | <code>[bool](#bool)</code> | If True, mask self-correlations (perfect correlations from duplicate subjects in bootstrap samples) as NaN. If False, include them in the summary statistic. Only applies when method='bootstrap' and summary_statistic='pairwise'. Defaults to True. | <code>True</code>
`sim_metric` | <code>[str](#str)</code> | Similarity metric for pairwise ISC computation. See sklearn.metrics.pairwise_distances for valid options. Only applies when summary_statistic='pairwise'. For 'correlation', uses optimized np.corrcoef. Other metrics use pairwise_distances. Defaults to 'correlation'. | <code>'correlation'</code>
`parallel` | <code>[Literal](#typing.Literal)['cpu', 'gpu'] \| None</code> | Parallelization method: - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (10-30× speedup for voxel-wise LOO) - None: Single-threaded NumPy (for debugging/small problems) Defaults to 'cpu'. | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when parallel='cpu'. Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel='gpu'). Defaults to 4. | <code>4.0</code>
`random_state` | <code>[int](#int) \| None</code> | Random seed for reproducibility. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with the following keys:
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'isc': Observed ISC value (float or array per voxel)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'p': P-value (Phipson-Smyth corrected)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'ci': Confidence interval tuple (lower, upper)
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'parallel': Parallelization method used
<code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | - 'null_dist': (optional) Bootstrap/permutation distribution

Examples:
>>> # Single-feature ISC
>>> data = np.random.randn(100, 10)  # 100 timepoints, 10 subjects
>>> result = isc_permutation_test(data, n_permute=1000)
>>> print(f"ISC: {result['isc']:.3f}, p: {result['p']:.3f}")

>>> # Voxel-wise ISC with GPU acceleration
>>> data_voxels = np.random.randn(100, 50, 5000)  # 5K voxels
>>> result = isc_permutation_test(
...     data_voxels,
...     summary_statistic='leave-one-out',
...     parallel='gpu',  # GPU for LOO computation
...     n_permute=5000
... )
>>> print(f"Significant voxels: {(result['p'] < 0.05).sum()}")

>>> # Compare LOO vs pairwise
>>> result_loo = isc_permutation_test(data, summary_statistic='leave-one-out')
>>> result_pair = isc_permutation_test(data, summary_statistic='pairwise')
>>> print(f"LOO: {result_loo['isc']:.3f}, Pairwise: {result_pair['isc']:.3f}")

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

**Methods:**

Name | Description
---- | -----------
[`distance_correlation`](#algorithms-distance-correlation) | Compute the distance correlation between 2 arrays to test for multivariate dependence (linear or non-linear).
[`double_center`](#algorithms-double-center) | Double center a 2d array.
[`matrix_permutation_test`](#algorithms-matrix-permutation-test) | Matrix permutation test (Mantel test) for correlating two square matrices.
[`u_center`](#algorithms-u-center) | U-center a 2d array. U-centering is a bias-corrected form of double-centering.

**Attributes:**

Name | Type | Description
---- | ---- | -----------
[`MAX_INT`](#algorithms-max-int) |  | 



####### Attributes##

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

######## `matrix_permutation_test`

```python
matrix_permutation_test(data1: np.ndarray, data2: np.ndarray, n_permute: int = 5000, metric: str = 'pearson', how: str = 'upper', include_diag: bool = False, tail: int | str = 2, parallel: str | None = 'cpu', n_jobs: int = -1, return_null: bool = False, random_state: int | None = None) -> dict
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
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type (default: 2) - 'two' or 2: Two-tailed test (r != 0) - 'upper' or 1: One-tailed upper (r > 0) - 'lower' or -1: One-tailed lower (r < 0) | <code>2</code>
`parallel` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel workers, -1 = all cores (default: -1) Only used when parallel='cpu' | <code>-1</code>
`return_null` | <code>[bool](#bool)</code> | Return null distribution (default: False) | <code>False</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'correlation' (float): Observed correlation coefficient - 'p' (float): P-value using Phipson-Smyth correction - 'parallel' (str): Parallelization method used ('cpu' or None) - 'null_dist' (np.ndarray): Null distribution (if return_null=True)

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
one_sample_permutation_test(data: np.ndarray, n_permute: int = 5000, tail: int | str = 2, return_null: bool = False, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None) -> dict
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
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type (default: 2) - 'two' or 2: Two-tailed test (mean != 0) - 'upper' or 1: One-tailed upper (mean > 0) - 'lower' or -1: One-tailed lower (mean < 0) For MCP correction (FDR), use 'upper' or 'lower' for consistent direction. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`parallel` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when parallel='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Maximum GPU memory to use in GB (default: 4.0) Controls automatic batching to prevent OOM errors. Only used with parallel='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>4.0</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'mean' (float or np.ndarray): Observed mean(s) - 'p' (float or np.ndarray): P-value(s) - 'null_dist' (np.ndarray): Null distribution (if return_null=True) - 'parallel' (str): Parallelization method used

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
>>> result = one_sample_permutation_test(data, n_permute=5000, parallel='gpu')
>>> result['mean'].shape
(10000,)
>>> result['p'].shape
(10000,)
```

```pycon
>>> # Single-threaded (for debugging)
>>> result = one_sample_permutation_test(data, n_permute=5000, parallel=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (parallel='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
- Single-threaded (parallel=None): Use for small problems or debugging
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
phase_randomize(data: np.ndarray, backend: str | None = None, random_state: int | np.random.RandomState | None = None) -> np.ndarray
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
`backend` | <code>[str](#str) \| None</code> | Computation backend ('numpy' or 'torch'). - 'numpy': CPU implementation using NumPy FFT (default, float64 precision) - 'torch': GPU implementation using PyTorch FFT (float32 precision, faster) - None: Defaults to 'numpy' | <code>None</code>
`random_state` | <code>[int](#int) \| [RandomState](#numpy.random.RandomState) \| None</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Phase-randomized data with same shape as input

<details class="notes" open markdown="1">
<summary>Notes</summary>

- **CRITICAL**: Preserves power spectrum exactly (within numerical precision)
- GPU acceleration: Use `backend='torch'` for GPU-accelerated FFT (5-20× faster for large data)
- Precision: NumPy backend uses float64, PyTorch backend uses float32
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
>>> x_rand_gpu = phase_randomize(x_large, backend='torch', random_state=42)
```

######## `timeseries_correlation_permutation_test`

```python
timeseries_correlation_permutation_test(data1: np.ndarray, data2: np.ndarray, method: Literal['circle_shift', 'phase_randomize'] = 'circle_shift', n_permute: int = 5000, metric: Literal['pearson', 'spearman', 'kendall'] = 'pearson', tail: int = 2, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, return_null: bool = False, random_state: int | np.random.RandomState | None = None) -> dict
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
`tail` | <code>[int](#int)</code> | Test type (1=one-tailed, 2=two-tailed) | <code>2</code>
`parallel` | <code>[str](#str) \| None</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of parallel jobs (-1 = all cores) Only used when parallel='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Maximum GPU memory to use in GB (default: 4.0) Controls automatic batching to prevent OOM errors. Only used with parallel='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>4.0</code>
`return_null` | <code>[bool](#bool)</code> | Whether to return null distribution | <code>False</code>
`random_state` | <code>[int](#int) \| [RandomState](#numpy.random.RandomState) \| None</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[dict](#dict)</code> | Dictionary with keys: - 'correlation': Observed correlation coefficient - 'p': P-value - 'null_dist': (if return_null=True) Null distribution - 'parallel': Parallelization method used

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
...     x, y, method='phase_randomize', parallel='gpu', n_permute=5000
... )
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (parallel='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): 5-20× faster for large problems (n_samples > 1000)
- Single-threaded (parallel=None): Use for small problems or debugging
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
two_sample_permutation_test(data1: np.ndarray, data2: np.ndarray, n_permute: int = 5000, tail: int | str = 2, return_null: bool = False, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None) -> dict
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
`tail` | <code>[int](#int) \| [str](#str)</code> | Test type (default: 2) - 'two' or 2: Two-tailed test (mean1 != mean2) - 'upper' or 1: One-tailed upper (mean1 > mean2) - 'lower' or -1: One-tailed lower (mean1 < mean2) For MCP correction (FDR), use 'upper' or 'lower' for consistent direction. | <code>2</code>
`return_null` | <code>[bool](#bool)</code> | If True, return full null distribution (default: False) | <code>False</code>
`parallel` | <code>[str](#str)</code> | Parallelization method (default: 'cpu') - None: Single-threaded NumPy (for debugging/small problems) - 'cpu': CPU parallelization via joblib (default, 4-8× speedup) - 'gpu': GPU acceleration via PyTorch (fastest for large problems) | <code>'cpu'</code>
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (default: -1 = all cores) Only used when parallel='cpu' | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | Maximum GPU memory to use in GB (default: 4.0) Controls automatic batching to prevent OOM errors. Only used with parallel='gpu'. Larger values allow more permutations per batch but risk OOM on smaller GPUs. | <code>4.0</code>
`random_state` | <code>[int](#int)</code> | Random seed for reproducibility | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)</code> | Dictionary with keys: - 'mean_diff' (float or np.ndarray): Observed mean difference (data1 - data2) - 'p' (float or np.ndarray): P-value(s) - 'null_dist' (np.ndarray): Null distribution (if return_null=True) - 'parallel' (str): Parallelization method used

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
>>> result = two_sample_permutation_test(data1, data2, n_permute=5000, parallel='gpu')
>>> result['mean_diff'].shape
(10000,)
>>> result['p'].shape
(10000,)
```

```pycon
>>> # Single-threaded (for debugging)
>>> result = two_sample_permutation_test(data1, data2, n_permute=5000, parallel=None)
```

<details class="notes" open markdown="1">
<summary>Notes</summary>

- Default (parallel='cpu'): CPU parallelization with joblib (4-8× speedup)
- GPU parallelization ('gpu'): Fastest for large problems with automatic batching
- Single-threaded (parallel=None): Use for small problems or debugging
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
    >>> from nltools.algorithms.validation import validate_parallel_parameter
    >>> validate_parallel_parameter("cpu")  # OK
    >>> validate_parallel_parameter("invalid")  # Raises ValueError

</details>

**Methods:**

Name | Description
---- | -----------
[`validate_alpha`](#algorithms-validate-alpha) | Validate regularization parameter alpha.
`validate_array_shape` | Validate array dimensionality.
`validate_array_shape_range` | Validate array dimensionality is within a range.
`validate_bootstrap_data` | Validate input data for bootstrapping.
`validate_bootstrap_method` | Validate bootstrap method name.
`validate_how_parameter` | Validate 'how' parameter for matrix operations.
`validate_isc_parameters` | Validate ISC parameter values.
`validate_metric_parameter` | Validate metric parameter.
`validate_n_samples` | Validate number of samples.
`validate_parallel_parameter` | Validate parallel parameter.
`validate_parallel_parameter_matrix` | Validate parallel parameter for matrix operations.
`validate_percentiles` | Validate percentile values for confidence intervals.
`validate_same_first_dimension` | Validate two arrays have same first dimension.
`validate_same_shape` | Validate two arrays have same shape.
`validate_shape_compatibility` | Validate that X and y have compatible shapes for regression.
`validate_square_matrix` | Validate matrix is square.
`validate_tail_parameter` | Validate and normalize tail parameter.



####### Functions##

(algorithms-validate-alpha)=
###### `validate_alpha`

```python
validate_alpha(alpha: float, name: str = 'alpha') -> None
```

Validate regularization parameter alpha.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`alpha` | <code>[float](#float)</code> | Regularization parameter | *required*
`name` | <code>[str](#str)</code> | Name of parameter for error message | <code>'alpha'</code>

######## `validate_array_shape`

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

######## `validate_how_parameter`

```python
validate_how_parameter(how: str) -> None
```

Validate 'how' parameter for matrix operations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`how` | <code>[str](#str)</code> | How parameter value | *required*

######## `validate_isc_parameters`

```python
validate_isc_parameters(metric: str, summary_statistic: str, method: str | None = None) -> None
```

Validate ISC parameter values.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`metric` | <code>[str](#str)</code> | Summary statistic metric | *required*
`summary_statistic` | <code>[str](#str)</code> | ISC computation method | *required*
`method` | <code>[str](#str) \| None</code> | Resampling method (optional) | <code>None</code>

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

######## `validate_n_samples`

```python
validate_n_samples(n_samples: int, min_samples: int = 2, name: str = 'n_samples') -> None
```

Validate number of samples.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`n_samples` | <code>[int](#int)</code> | Number of samples | *required*
`min_samples` | <code>[int](#int)</code> | Minimum required samples | <code>2</code>
`name` | <code>[str](#str)</code> | Name of parameter for error message | <code>'n_samples'</code>

######## `validate_parallel_parameter`

```python
validate_parallel_parameter(parallel: str | None) -> None
```

Validate parallel parameter.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`parallel` | <code>[str](#str) \| None</code> | Parallel parameter value | *required*

######## `validate_parallel_parameter_matrix`

```python
validate_parallel_parameter_matrix(parallel: str | None) -> None
```

Validate parallel parameter for matrix operations.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`parallel` | <code>[str](#str) \| None</code> | Parallel parameter value | *required*

######## `validate_percentiles`

```python
validate_percentiles(percentiles: tuple[float, float]) -> None
```

Validate percentile values for confidence intervals.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`percentiles` | <code>[tuple](#tuple)[[float](#float), [float](#float)]</code> | Percentile values (lower, upper) | *required*

######## `validate_same_first_dimension`

```python
validate_same_first_dimension(array1: np.ndarray, array2: np.ndarray, name1: str = 'array1', name2: str = 'array2') -> None
```

Validate two arrays have same first dimension.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array1` | <code>[ndarray](#numpy.ndarray)</code> | First array | *required*
`array2` | <code>[ndarray](#numpy.ndarray)</code> | Second array | *required*
`name1` | <code>[str](#str)</code> | Name of first array for error message | <code>'array1'</code>
`name2` | <code>[str](#str)</code> | Name of second array for error message | <code>'array2'</code>

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

Validate and normalize tail parameter.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`tail` | <code>[int](#int) \| [str](#str)</code> | Tail parameter value. Can be: - 'two' or 2: Two-tailed test (|obs| > |null|) - 'upper' or 1: One-tailed upper (obs > null, for testing positive effects) - 'lower' or -1: One-tailed lower (obs < null, for testing negative effects) | *required*

**Returns:**

Type | Description
---- | -----------
<code>[str](#str)</code> | Normalized tail string: 'two', 'upper', or 'lower'

<details class="notes" open markdown="1">
<summary>Notes</summary>

For multiple comparisons correction (FDR, Bonferroni), use 'upper' or 'lower'
to ensure consistent direction across all tests. The old tail=1 behavior
(auto-detecting direction per test based on sign) can lead to incorrect
MCP-adjusted p-values. See GH #315.

</details>

(algorithms-random)=
#### `random`

Shared random state utilities for algorithms module.

This module provides common random state handling to ensure consistent
random number generation across the algorithms module.

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
[`get_random_state`](#algorithms-get-random-state) | Get RandomState instance from seed.



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

This implementation matches the RNG pattern from nltools.stats.one_sample_permutation
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

(algorithms-get-random-state)=
###### `get_random_state`

```python
get_random_state(random_state: int | None = None)
```

Get RandomState instance from seed.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`random_state` | <code>[int](#int) \| None</code> | Random seed (int, RandomState, or None) | <code>None</code>

**Returns:**

Type | Description
---- | -----------
 | RandomState instance

<details class="note" open markdown="1">
<summary>Note</summary>

Uses sklearn.utils.check_random_state for consistency

</details>

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

**Modules:**

Name | Description
---- | -----------
[`core`](#algorithms-core) | Ridge regression algorithms using SVD decomposition.
[`solvers`](#algorithms-solvers) | Ridge regression solvers with cross-validation.
[`utils`](#algorithms-utils) | Utility functions for ridge regression.

**Methods:**

Name | Description
---- | -----------
[`cross_val_predict_ridge`](#algorithms-cross-val-predict-ridge) | Held-out ridge predictions per CV fold under a (per-target) alpha.
[`generate_dirichlet_samples`](#algorithms-generate-dirichlet-samples) | Generate samples from a Dirichlet distribution.
[`ridge_cv`](#algorithms-ridge-cv) | Ridge regression with cross-validation for hyperparameter selection.
[`ridge_svd`](#algorithms-ridge-svd) | Solve ridge regression using Singular Value Decomposition.
[`solve_banded_ridge_cv`](#algorithms-solve-banded-ridge-cv) | Solve banded ridge regression with cross-validation using random search.
[`solve_ridge_cv`](#algorithms-solve-ridge-cv) | Solve ridge regression with cross-validation.



##### Methods

(algorithms-cross-val-predict-ridge)=
###### `cross_val_predict_ridge`

```python
cross_val_predict_ridge(X: np.ndarray, Y: np.ndarray, *, alphas: float | np.ndarray, cv: int | BaseCrossValidator = 5, fit_intercept: bool = False, n_targets_batch: int | None = None, n_alphas_batch: int | None = None, Y_in_cpu: bool = True, score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0) -> dict[str, Any]
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
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when parallel="cpu". | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel="gpu"). | <code>4.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with keys: - 'predictions': (n_samples, n_targets) held-out per-target   predictions on the original Y scale (CPU numpy). - 'folds': (n_samples,) int fold index per row (CPU numpy). - 'scores': (n_splits, n_targets) per-fold R² (or   ``score_func``) at the supplied alpha (CPU numpy). - 'parallel': Backend used (for transparency).

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
ridge_cv(X: np.ndarray, y: np.ndarray, alphas: np.ndarray | None = None, cv: int | BaseCrossValidator = 5, fit_intercept: bool = False, parallel: str | None = 'cpu', max_gpu_memory_gb: float = 4.0, random_state: int | None = None) -> dict
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
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU-only using NumPy (default) - "gpu": GPU acceleration via PyTorch (falls back to CPU if GPU unavailable) Defaults to "cpu". | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel='gpu'). Defaults to 4.0. | <code>4.0</code>
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
- When parallel='gpu' is requested but GPU is unavailable, gracefully falls back to CPU

</details>

###### `ridge_svd`

```python
ridge_svd(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, parallel: str | None = None, max_gpu_memory_gb: float = 4.0, random_state: int | None = None) -> np.ndarray
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
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU-only using NumPy (default) - "gpu": GPU acceleration via PyTorch (falls back to CPU if GPU unavailable) Defaults to None. | <code>None</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel='gpu'). Defaults to 4.0. | <code>4.0</code>
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
solve_banded_ridge_cv(Xs: list[np.ndarray], Y: np.ndarray, n_iter: int | np.ndarray = 100, concentration: float | list[float] = [0.1, 1.0], alphas: float | np.ndarray | list[float] = [0.1, 1.0, 10.0], cv: int | BaseCrossValidator = 5, local_alpha: bool = True, n_targets_batch: int | None = None, n_targets_batch_refit: int | None = None, n_alphas_batch: int | None = None, Y_in_cpu: bool = True, score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, fit_intercept: bool = False, progress_bar: bool = False, conservative: bool = False, jitter_alphas: bool = False, return_weights: bool = True, diagonalize_method: str = 'svd', warn: bool = True, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None) -> dict[str, Any]
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
`n_iter` | <code>[int](#int) \| [ndarray](#numpy.ndarray)</code> | Number of feature-space weights combination to search, or array of shape (n_iter, n_spaces). If an array is given, the solver uses it as the list of weights to try, instead of sampling from a Dirichlet distribution. Defaults to 100. | <code>100</code>
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
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel="gpu"). Defaults to 4.0. | <code>4.0</code>
`random_state` | <code>[int](#int) \| None</code> | Random generator seed. Use an int for deterministic search. Defaults to None. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with keys: - 'deltas': Best log feature-space weights for each target,     shape (n_spaces, n_targets). deltas = log(gamma / alpha), where     gamma are the feature space weights. - 'cv_scores': Cross-validation scores per iteration, averaged over splits,     for the best alpha, shape (n_iter, n_targets). Always returned on CPU     (numpy array). - 'coefs': Ridge coefficients refit on entire dataset using best hyperparameters,     shape (n_features_total, n_targets), or None if return_weights=False.     Always returned on CPU (numpy array). - 'intercept': Intercept of shape (n_targets,), or None if     fit_intercept=False or return_weights=False. - 'parallel': Backend used (for transparency).

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
See ``nltools.algorithms.ridge.DESIGN.md`` for detailed algorithm explanation.

</details>

(algorithms-solve-ridge-cv)=
###### `solve_ridge_cv`

```python
solve_ridge_cv(X: np.ndarray, Y: np.ndarray, alphas: float | np.ndarray | list[float] = [0.1, 1.0, 10.0], cv: int | BaseCrossValidator = 5, local_alpha: bool = True, n_targets_batch: int | None = None, n_targets_batch_refit: int | None = None, n_alphas_batch: int | None = None, Y_in_cpu: bool = True, score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, fit_intercept: bool = False, progress_bar: bool = False, conservative: bool = False, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None) -> dict[str, Any]
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
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel="gpu"). Defaults to 4.0. | <code>4.0</code>
`random_state` | <code>[int](#int) \| None</code> | Random generator seed. Use an int for deterministic search. Defaults to None. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with keys: - 'best_alphas': Selected best alpha for each target (or same alpha repeated     if local_alpha=False), shape (n_targets,). - 'coefs': Ridge coefficients refit on entire dataset using best alphas,     shape (n_features, n_targets). Always returned on CPU (numpy array). - 'cv_scores': Cross-validation scores for best alphas, shape (n_splits, n_alphas, n_targets).     Always returned on CPU (numpy array). - 'intercept': Per-target intercept of shape (n_targets,). Only present     when ``fit_intercept=True``. - 'parallel': Backend used (for transparency).

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
See ``nltools.algorithms.ridge.DESIGN.md`` for detailed algorithm explanation.

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
- See `nltools.algorithms.ridge.DESIGN.md` for detailed algorithm explanation

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
ridge_cv(X: np.ndarray, y: np.ndarray, alphas: np.ndarray | None = None, cv: int | BaseCrossValidator = 5, fit_intercept: bool = False, parallel: str | None = 'cpu', max_gpu_memory_gb: float = 4.0, random_state: int | None = None) -> dict
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
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU-only using NumPy (default) - "gpu": GPU acceleration via PyTorch (falls back to CPU if GPU unavailable) Defaults to "cpu". | <code>'cpu'</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel='gpu'). Defaults to 4.0. | <code>4.0</code>
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
- When parallel='gpu' is requested but GPU is unavailable, gracefully falls back to CPU

</details>

######## `ridge_svd`

```python
ridge_svd(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, parallel: str | None = None, max_gpu_memory_gb: float = 4.0, random_state: int | None = None) -> np.ndarray
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
`parallel` | <code>[str](#str)</code> | Execution backend. - None: Single-threaded NumPy (debugging/small problems) - "cpu": CPU-only using NumPy (default) - "gpu": GPU acceleration via PyTorch (falls back to CPU if GPU unavailable) Defaults to None. | <code>None</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel='gpu'). Defaults to 4.0. | <code>4.0</code>
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
cross_val_predict_ridge(X: np.ndarray, Y: np.ndarray, *, alphas: float | np.ndarray, cv: int | BaseCrossValidator = 5, fit_intercept: bool = False, n_targets_batch: int | None = None, n_alphas_batch: int | None = None, Y_in_cpu: bool = True, score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0) -> dict[str, Any]
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
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when parallel="cpu". | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel="gpu"). | <code>4.0</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with keys: - 'predictions': (n_samples, n_targets) held-out per-target   predictions on the original Y scale (CPU numpy). - 'folds': (n_samples,) int fold index per row (CPU numpy). - 'scores': (n_splits, n_targets) per-fold R² (or   ``score_func``) at the supplied alpha (CPU numpy). - 'parallel': Backend used (for transparency).

######## `solve_banded_ridge_cv`

```python
solve_banded_ridge_cv(Xs: list[np.ndarray], Y: np.ndarray, n_iter: int | np.ndarray = 100, concentration: float | list[float] = [0.1, 1.0], alphas: float | np.ndarray | list[float] = [0.1, 1.0, 10.0], cv: int | BaseCrossValidator = 5, local_alpha: bool = True, n_targets_batch: int | None = None, n_targets_batch_refit: int | None = None, n_alphas_batch: int | None = None, Y_in_cpu: bool = True, score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, fit_intercept: bool = False, progress_bar: bool = False, conservative: bool = False, jitter_alphas: bool = False, return_weights: bool = True, diagonalize_method: str = 'svd', warn: bool = True, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None) -> dict[str, Any]
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
`n_iter` | <code>[int](#int) \| [ndarray](#numpy.ndarray)</code> | Number of feature-space weights combination to search, or array of shape (n_iter, n_spaces). If an array is given, the solver uses it as the list of weights to try, instead of sampling from a Dirichlet distribution. Defaults to 100. | <code>100</code>
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
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel="gpu"). Defaults to 4.0. | <code>4.0</code>
`random_state` | <code>[int](#int) \| None</code> | Random generator seed. Use an int for deterministic search. Defaults to None. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with keys: - 'deltas': Best log feature-space weights for each target,     shape (n_spaces, n_targets). deltas = log(gamma / alpha), where     gamma are the feature space weights. - 'cv_scores': Cross-validation scores per iteration, averaged over splits,     for the best alpha, shape (n_iter, n_targets). Always returned on CPU     (numpy array). - 'coefs': Ridge coefficients refit on entire dataset using best hyperparameters,     shape (n_features_total, n_targets), or None if return_weights=False.     Always returned on CPU (numpy array). - 'intercept': Intercept of shape (n_targets,), or None if     fit_intercept=False or return_weights=False. - 'parallel': Backend used (for transparency).

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
See ``nltools.algorithms.ridge.DESIGN.md`` for detailed algorithm explanation.

</details>

######## `solve_ridge_cv`

```python
solve_ridge_cv(X: np.ndarray, Y: np.ndarray, alphas: float | np.ndarray | list[float] = [0.1, 1.0, 10.0], cv: int | BaseCrossValidator = 5, local_alpha: bool = True, n_targets_batch: int | None = None, n_targets_batch_refit: int | None = None, n_alphas_batch: int | None = None, Y_in_cpu: bool = True, score_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None, fit_intercept: bool = False, progress_bar: bool = False, conservative: bool = False, parallel: str | None = 'cpu', n_jobs: int = -1, max_gpu_memory_gb: float = 4.0, random_state: int | None = None) -> dict[str, Any]
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
`n_jobs` | <code>[int](#int)</code> | Number of CPU cores for parallelization (-1 = all cores). Only used when parallel="cpu". Defaults to -1. | <code>-1</code>
`max_gpu_memory_gb` | <code>[float](#float)</code> | GPU memory budget in GB (only used if parallel="gpu"). Defaults to 4.0. | <code>4.0</code>
`random_state` | <code>[int](#int) \| None</code> | Random generator seed. Use an int for deterministic search. Defaults to None. | <code>None</code>

**Returns:**

Name | Type | Description
---- | ---- | -----------
`dict` | <code>[dict](#dict)[[str](#str), [Any](#typing.Any)]</code> | Dictionary with keys: - 'best_alphas': Selected best alpha for each target (or same alpha repeated     if local_alpha=False), shape (n_targets,). - 'coefs': Ridge coefficients refit on entire dataset using best alphas,     shape (n_features, n_targets). Always returned on CPU (numpy array). - 'cv_scores': Cross-validation scores for best alphas, shape (n_splits, n_alphas, n_targets).     Always returned on CPU (numpy array). - 'intercept': Per-target intercept of shape (n_targets,). Only present     when ``fit_intercept=True``. - 'parallel': Backend used (for transparency).

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
See ``nltools.algorithms.ridge.DESIGN.md`` for detailed algorithm explanation.

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

Shared shape manipulation utilities for algorithms module.

This module provides common shape manipulation functions to reduce code
duplication and ensure consistent behavior across the algorithms module.

<details class="key-functions" open markdown="1">
<summary>Key functions</summary>

- extract_triangle_elements: Extract upper/lower triangle from matrices
- permute_matrix_symmetric: Apply symmetric permutation (key for matrix tests)
- ensure_2d: Ensure arrays are 2D (adds dimension if needed)
- batch_or_skip: Elegant pattern for handling scalar vs per-target operations

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
[`batch_or_skip`](#algorithms-batch-or-skip) | Apply batch or skip if dimension is 1.
[`ensure_2d`](#algorithms-ensure-2d) | Ensure array is 2D, adding dimension if needed.
[`extract_triangle_elements`](#algorithms-extract-triangle-elements) | Extract triangle elements from square matrix.
[`permute_matrix_symmetric`](#algorithms-permute-matrix-symmetric) | Apply symmetric row+column permutation to square matrix.



##### Methods

(algorithms-batch-or-skip)=
###### `batch_or_skip`

```python
batch_or_skip(array: np.ndarray, batch: slice, axis: int) -> np.ndarray
```

Apply batch or skip if dimension is 1.

This elegant pattern from himalaya handles both scalar and per-target
operations without branching in the hot loop.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` | <code>[ndarray](#numpy.ndarray)</code> | Array to batch. | *required*
`batch` | <code>[slice](#slice)</code> | Batch slice to apply. | *required*
`axis` | <code>[int](#int)</code> | Axis to batch along (0 or 1). | *required*

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | Batched array, or original if dimension is 1.

**Examples:**

```pycon
>>> # Scalar alpha (shape: (1,))
>>> alphas = np.array([1.0])
>>> batch_or_skip(alphas, slice(0, 10), 0)  # Returns full array
array([1.0])
```

```pycon
>>> # Per-target alpha (shape: (10,))
>>> alphas = np.array([1.0] * 10)
>>> batch_or_skip(alphas, slice(0, 5), 0).shape  # Returns slice
(5,)
```

(algorithms-ensure-2d)=
###### `ensure_2d`

```python
ensure_2d(array: np.ndarray, name: str = 'array') -> np.ndarray
```

Ensure array is 2D, adding dimension if needed.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`array` | <code>[ndarray](#numpy.ndarray)</code> | Input array (1D or 2D) | *required*
`name` | <code>[str](#str)</code> | Name for error messages | <code>'array'</code>

**Returns:**

Type | Description
---- | -----------
<code>[ndarray](#numpy.ndarray)</code> | 2D array (shape [n, 1] if input was 1D)

**Examples:**

```pycon
>>> x = np.array([1, 2, 3])
>>> ensure_2d(x).shape
(3, 1)
>>> x2d = np.array([[1, 2], [3, 4]])
>>> ensure_2d(x2d).shape
(2, 2)
```

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

