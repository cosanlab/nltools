---
title: Simulation
label: page-tasks-simulation
---

Synthetic data with a known signal, for testing a pipeline end to end. `Simulator` builds [BrainData](../data/brain_data.md) with Gaussian-blob signal, several subjects, and noise. `SimulateGrid` builds 2D grids, which is enough to exercise thresholding and multiple-comparison correction without a mask.

**Classes:**

Name | Description
---- | -----------
[`Simulator`](#tasks-simulation-simulator) | Simulate fMRI data with realistic spatial and temporal characteristics.
[`SimulateGrid`](#tasks-simulation-simulategrid) | Simulate 2D grid data for testing statistical methods.

## Classes

(tasks-simulation-simulator)=
### `Simulator`

```python
Simulator(*, brain_mask = None, output_dir = None, random_state = None)
```

Simulate fMRI data with realistic spatial and temporal characteristics.

This class provides methods for generating synthetic fMRI data with
controlled signal patterns, including Gaussian blobs, multi-subject
datasets, and various noise structures. Useful for testing analysis
pipelines and power analyses.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`brain_mask` | <code>str \| Nifti1Image</code> | Path to a NIfTI brain mask file, a nibabel image, or None to use the default template mask. | <code>None</code>
`output_dir` | <code>str</code> | Directory for saving generated data. Defaults to the current working directory. | <code>None</code>
`random_state` | <code>int \| RandomState</code> | Seed or RandomState for reproducibility. | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`brain_mask` | <code>Nifti1Image</code> | The brain mask image used for simulation.
`output_dir` | <code>str</code> | Output directory path.
`random_state` | <code>RandomState</code> | Random state for reproducible simulations.
`data` | <code>[BrainData](#page-data-brain-data) \| Nifti1Image</code> | Most recently simulated data; set by the `create_*` methods.
`y` | <code>DataFrame \| ndarray</code> | Outcome values paired with `data`; set by the `create_*` methods.
`rep_id` | <code>DataFrame \| list</code> | Repetition/subject id per observation; set by the `create_*` methods.

**Methods:**

Name | Description
---- | -----------
[`create_cov_data`](#tasks-simulation-create-cov-data) | Create continuous simulated data with covariance within a single region.
[`create_data`](#tasks-simulation-create-data) | Create simulated data with discrete intensity levels.
[`create_ncov_data`](#tasks-simulation-create-ncov-data) | Create continuous simulated data with covariance across multiple regions.
[`gaussian`](#tasks-simulation-gaussian) | Create a 3D gaussian signal normalized to a given intensity.
[`n_spheres`](#tasks-simulation-n-spheres) | Generate a set of spheres in the brain mask space.
[`normal_noise`](#tasks-simulation-normal-noise) | Produce a normal noise distribution for all points in the brain mask.
[`sphere`](#tasks-simulation-sphere) | Create a sphere of given radius at some point p in the brain mask.
[`to_nifti`](#tasks-simulation-to-nifti) | Convert a numpy array to a NIfTI image with the brain mask's affine.



**Examples:**

```python
from nltools.data.simulator import Simulator

sim = Simulator(random_state=42)
# Create a dataset with signal in specific regions
data = sim.create_data(levels=[1, -1, 1, -1], sigma=1, reps=10)
```

#### Methods

(tasks-simulation-create-cov-data)=
##### `create_cov_data`

```python
create_cov_data(cor, cov, sigma, *, mask = None, reps = 1, n_sub = 1, output_dir = None)
```

Create continuous simulated data with covariance within a single region.

Results are stored on `self.data` (a 4-D `nibabel.Nifti1Image`), `self.y`, and
`self.rep_id`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cor` | <code>float</code> | Covariance between each voxel and the outcome `y`. | *required*
`cov` | <code>float</code> | Covariance between voxels. | *required*
`sigma` | <code>float</code> | Standard deviation of the added noise. | *required*
`mask` | <code>Nifti1Image</code> | Region where activations are placed. Defaults to a sphere of radius 10 at the mask center. | <code>None</code>
`reps` | <code>int</code> | Number of repetitions per subject. Default 1. | <code>1</code>
`n_sub` | <code>int</code> | Number of subjects to simulate. Default 1. | <code>1</code>
`output_dir` | <code>str</code> | Directory to write the image, `y.csv`, and `rep_id.csv` into. If None, nothing is written. | <code>None</code>

(tasks-simulation-create-data)=
##### `create_data`

```python
create_data(levels, sigma, *, radius = 5, center = None, reps = 1, output_dir = None)
```

Create simulated data with discrete intensity levels.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`levels` | <code>list</code> | Intensities or class labels, one per image in a repetition. | *required*
`sigma` | <code>float</code> | Standard deviation of the added noise. | *required*
`radius` | <code>int \| list[int]</code> | Sphere radius, or one radius per sphere. | <code>5</code>
`center` | <code>list</code> | Sphere center `[x, y, z]`, or one center per sphere `[[x1, y1, z1], ...]`. None places every sphere at the mask center. | <code>None</code>
`reps` | <code>int</code> | Number of repetitions (e.g. trials or subjects). Default 1. | <code>1</code>
`output_dir` | <code>str</code> | Directory to write `data.nii.gz`, `y.csv`, and `rep_id.csv` into. If None, nothing is written. | <code>None</code>

**Returns:**

Type | Description
---- | -----------
<code>[BrainData](#page-data-brain-data)</code> | The simulated images with `Y` set to the levels.

(tasks-simulation-create-ncov-data)=
##### `create_ncov_data`

```python
create_ncov_data(cor, cov, sigma, *, masks = None, reps = 1, n_sub = 1, output_dir = None)
```

Create continuous simulated data with covariance across multiple regions.

Results are stored on `self.data` (a 4-D `nibabel.Nifti1Image`), `self.y`, and
`self.rep_id`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cor` | <code>float \| list[float]</code> | Covariance between each region's voxels and the outcome `y`; one value per region. | *required*
`cov` | <code>float \| list[list[float]]</code> | Covariance between voxels; a scalar for a single region or a region-by-region matrix. | *required*
`sigma` | <code>float</code> | Standard deviation of the added noise. | *required*
`masks` | <code>Nifti1Image \| list[Nifti1Image]</code> | Region(s) where activations are placed. Defaults to a sphere of radius 10 at the mask center. | <code>None</code>
`reps` | <code>int</code> | Number of repetitions per subject. Default 1. | <code>1</code>
`n_sub` | <code>int</code> | Number of subjects to simulate. Default 1. | <code>1</code>
`output_dir` | <code>str</code> | Directory to write the image, `y.csv`, and `rep_id.csv` into. If None, nothing is written. | <code>None</code>

(tasks-simulation-gaussian)=
##### `gaussian`

```python
gaussian(mu, sigma, i_tot)
```

Create a 3D gaussian signal normalized to a given intensity.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mu` | <code>array - like</code> | Center of the gaussian in voxel coordinates `[x, y, z]`. | *required*
`sigma` | <code>array - like</code> | Standard deviation per axis `[sx, sy, sz]`. | *required*
`i_tot` | <code>float</code> | Total activation; the gaussian is rescaled so its sum within the brain mask equals this value. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | 3-D array the shape of the brain mask.

(tasks-simulation-n-spheres)=
##### `n_spheres`

```python
n_spheres(radius, center)
```

Generate a set of spheres in the brain mask space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`radius` | <code>int \| list[int]</code> | Sphere radius, or one radius per sphere. | *required*
`center` | <code>list</code> | Sphere center `[x, y, z]`, or one center per sphere `[[x1, y1, z1], ...]`. None places every sphere at the mask center. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | 3-D array the shape of the brain mask with the spheres summed.

(tasks-simulation-normal-noise)=
##### `normal_noise`

```python
normal_noise(mu, sigma)
```

Produce a normal noise distribution for all points in the brain mask.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mu` | <code>float</code> | Mean of the noise (usually 0). | *required*
`sigma` | <code>float</code> | Standard deviation of the noise. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | 3-D array the shape of the brain mask filled with noise inside     the mask.

(tasks-simulation-sphere)=
##### `sphere`

```python
sphere(r, p)
```

Create a sphere of given radius at some point p in the brain mask.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`r` | <code>int \| float</code> | Radius of the sphere in voxels. | *required*
`p` | <code>array - like</code> | Center of the sphere in voxel coordinates `[x, y, z]`. | *required*

**Returns:**

Type | Description
---- | -----------
<code>ndarray</code> | 3-D array the shape of the brain mask, 1 inside the sphere and     0 elsewhere.

(tasks-simulation-to-nifti)=
##### `to_nifti`

```python
to_nifti(m)
```

Convert a numpy array to a NIfTI image with the brain mask's affine.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`m` | <code>ndarray</code> | 3-D (or 4-D) array to convert. | *required*

**Returns:**

Type | Description
---- | -----------
<code>Nifti1Image</code> | The array as a float32 image.

(tasks-simulation-simulategrid)=
### `SimulateGrid`

```python
SimulateGrid(*, grid_width = 100, signal_width = 20, n_subjects = 20, sigma = 1, signal_amplitude = None, random_state = None)
```

Simulate 2D grid data for testing statistical methods.

Creates a 2D grid (e.g., 100x100 pixels) with optional embedded signal
regions and Gaussian noise. Useful for testing multiple comparison
correction methods, threshold selection, and visualization of
statistical maps.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`grid_width` | <code>int</code> | Width/height of the square grid. Default 100. | <code>100</code>
`signal_width` | <code>int</code> | Width of the embedded signal region. Default 20. | <code>20</code>
`n_subjects` | <code>int</code> | Number of simulated subjects. Default 20. | <code>20</code>
`sigma` | <code>float</code> | Standard deviation of the Gaussian noise. Default 1. | <code>1</code>
`signal_amplitude` | <code>float</code> | Amplitude of the embedded signal. If None, no signal is added. | <code>None</code>
`random_state` | <code>int \| RandomState</code> | Seed or RandomState for reproducibility. | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`data` | <code>ndarray</code> | Simulated data of shape `(grid_width, grid_width, n_subjects)`.
`signal_mask` | <code>ndarray \| None</code> | Binary grid marking the signal region, or None when no signal was added.
`t_values` | <code>ndarray \| None</code> | T-statistic map after `fit()`.
`p_values` | <code>ndarray \| None</code> | P-value map after `fit()`.
`thresholded` | <code>ndarray \| None</code> | Thresholded statistical map after `threshold_simulation()`.
`isfit` | <code>bool</code> | Whether `fit()` has been called.

**Methods:**

Name | Description
---- | -----------
[`add_signal`](#tasks-simulation-add-signal) | Add a square signal region, centered in the grid, to `self.data`.
[`create_mask`](#tasks-simulation-create-mask) | Create the binary `signal_mask` marking a centered square of the grid.
[`fit`](#tasks-simulation-fit) | Run a one-sample t-test on self.data.
[`plot_grid_simulation`](#tasks-simulation-plot-grid-simulation) | Plot the t-map, its thresholded version, and the false positive distribution.
[`run_multiple_simulations`](#tasks-simulation-run-multiple-simulations) | Run repeated simulations to estimate the false positive rate.
[`threshold_simulation`](#tasks-simulation-threshold-simulation) | Threshold the fitted simulation and store `thresholded` plus hit rates.



**Examples:**

```python
from nltools.data.simulator import SimulateGrid

sim = SimulateGrid(signal_amplitude=0.5, random_state=42)
sim.fit()
sim.plot_grid_simulation(threshold=0.05, threshold_type="q", correction="fdr")
```

#### Methods

(tasks-simulation-add-signal)=
##### `add_signal`

```python
add_signal(signal_width = 20, signal_amplitude = 1)
```

Add a square signal region, centered in the grid, to `self.data`.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`signal_width` | <code>int</code> | Width of the signal box in pixels. Default 20. | <code>20</code>
`signal_amplitude` | <code>float</code> | Intensity added inside the box. Default 1. | <code>1</code>

(tasks-simulation-create-mask)=
##### `create_mask`

```python
create_mask(signal_width)
```

Create the binary `signal_mask` marking a centered square of the grid.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`signal_width` | <code>int</code> | Width of the signal box in pixels. | *required*

(tasks-simulation-fit)=
##### `fit`

```python
fit()
```

Run a one-sample t-test on self.data.

(tasks-simulation-plot-grid-simulation)=
##### `plot_grid_simulation`

```python
plot_grid_simulation(threshold, threshold_type, n_simulations = 100, correction = None)
```

Plot the t-map, its thresholded version, and the false positive distribution.

Fits and thresholds the simulation first if needed, then calls
`run_multiple_simulations`. Adds a signal-recovery histogram when a signal is
present.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`threshold` | <code>float</code> | Threshold value to apply. | *required*
`threshold_type` | <code>str</code> | `'t'`, `'p'`, or `'q'` (see `threshold_simulation`). | *required*
`n_simulations` | <code>int</code> | Number of simulations to run. Default 100. | <code>100</code>
`correction` | <code>str</code> | Multiple-comparison correction; `'fdr'` or None. | <code>None</code>

(tasks-simulation-run-multiple-simulations)=
##### `run_multiple_simulations`

```python
run_multiple_simulations(threshold, threshold_type, n_simulations = 100, correction = None)
```

Run repeated simulations to estimate the false positive rate.

Stores per-simulation results on `multiple_thresholded`, `multiple_fp`, and
`fpr` (plus `multiple_tp` and `multiple_fdr` when a signal is present).

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`threshold` | <code>float</code> | Threshold value to apply to each simulation. | *required*
`threshold_type` | <code>str</code> | `'t'`, `'p'`, or `'q'` (see `threshold_simulation`). | *required*
`n_simulations` | <code>int</code> | Number of simulations to run. Default 100. | <code>100</code>
`correction` | <code>str</code> | Multiple-comparison correction; `'fdr'` or None. | <code>None</code>

(tasks-simulation-threshold-simulation)=
##### `threshold_simulation`

```python
threshold_simulation(threshold, threshold_type, correction = None)
```

Threshold the fitted simulation and store `thresholded` plus hit rates.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`threshold` | <code>float</code> | Threshold value to apply. | *required*
`threshold_type` | <code>str</code> | `'t'` (absolute t-value), `'p'` (p-value), or `'q'` (FDR-corrected q-value; requires `correction='fdr'`). | *required*
`correction` | <code>str</code> | Multiple-comparison correction; `'fdr'` or None. | <code>None</code>
