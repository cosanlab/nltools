(simulator-simulator)=
## `simulator`

Tools to simulate multivariate brain and grid data for testing analysis pipelines.

**Classes:**

Name | Description
---- | -----------
[`SimulateGrid`](#simulator-simulategrid) | Simulate 2D grid data for testing statistical methods.
[`Simulator`](#simulator-simulator) | Simulate fMRI data with realistic spatial and temporal characteristics.



### Classes

(simulator-simulategrid)=
#### `SimulateGrid`

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
`grid_width` |  | Width/height of the square grid (default: 100). | <code>100</code>
`signal_width` |  | Width of the embedded signal region (default: 20). | <code>20</code>
`n_subjects` |  | Number of simulated subjects (default: 20). | <code>20</code>
`sigma` |  | Standard deviation of the Gaussian noise (default: 1). | <code>1</code>
`signal_amplitude` |  | Amplitude of the embedded signal. If None, no signal is added. | <code>None</code>
`random_state` |  | Random seed or numpy RandomState for reproducibility. | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`data` |  | The simulated data array of shape (n_subjects, grid_width, grid_width).
`t_values` |  | T-statistic values after fitting.
`p_values` |  | P-values after fitting.
`thresholded` |  | Thresholded statistical map.
`isfit` |  | Whether fit() has been called.

**Methods:**

Name | Description
---- | -----------
[`add_signal`](#simulator-add-signal) | Add a rectangular signal to self.data.
[`create_mask`](#simulator-create-mask) | Create a mask for where the signal is located in grid.
[`fit`](#simulator-fit) | Run a one-sample t-test on self.data.
[`plot_grid_simulation`](#simulator-plot-grid-simulation) | Create a plot of the simulations.
[`run_multiple_simulations`](#simulator-run-multiple-simulations) | Run multiple simulations to calculate the overall false positive rate.
[`threshold_simulation`](#simulator-threshold-simulation) | Threshold the fitted simulation.



**Examples:**

```pycon
>>> from nltools.data.simulator import SimulateGrid
>>> sim = SimulateGrid(signal_amplitude=0.5, random_state=42)
>>> sim.fit()
>>> sim.plot()
```

##### Methods

(simulator-add-signal)=
###### `add_signal`

```python
add_signal(signal_width = 20, signal_amplitude = 1)
```

Add a rectangular signal to self.data.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`signal_width` | <code>[int](#int)</code> | width of signal box | <code>20</code>
`signal_amplitude` | <code>[int](#int)</code> | intensity of signal | <code>1</code>

(simulator-create-mask)=
###### `create_mask`

```python
create_mask(signal_width)
```

Create a mask for where the signal is located in grid.

(simulator-fit)=
###### `fit`

```python
fit()
```

Run a one-sample t-test on self.data.

(simulator-plot-grid-simulation)=
###### `plot_grid_simulation`

```python
plot_grid_simulation(threshold, threshold_type, n_simulations = 100, correction = None)
```

Create a plot of the simulations.

(simulator-run-multiple-simulations)=
###### `run_multiple_simulations`

```python
run_multiple_simulations(threshold, threshold_type, n_simulations = 100, correction = None)
```

Run multiple simulations to calculate the overall false positive rate.

(simulator-threshold-simulation)=
###### `threshold_simulation`

```python
threshold_simulation(threshold, threshold_type, correction = None)
```

Threshold the fitted simulation.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`threshold` | <code>[float](#float)</code> | threshold to apply to simulation | *required*
`threshold_type` | <code>[str](#str)</code> | type of threshold to use can be a specific t-value or p-value ['t', 'p', 'q'] | *required*

#### `Simulator`

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
`brain_mask` |  | Path to a NIfTI brain mask file, a nibabel image object, or None to use the default MNI template mask. | <code>None</code>
`output_dir` |  | Directory for saving generated data. Defaults to the current working directory. | <code>None</code>
`random_state` |  | Random seed or numpy RandomState for reproducibility. | <code>None</code>

**Attributes:**

Name | Type | Description
---- | ---- | -----------
`brain_mask` |  | The brain mask image used for simulation.
`output_dir` |  | Output directory path.
`random_state` |  | Random state for reproducible simulations.

**Methods:**

Name | Description
---- | -----------
[`create_cov_data`](#simulator-create-cov-data) | Create continuous simulated data with covariance within a single region.
[`create_data`](#simulator-create-data) | Create simulated data with discrete intensity levels.
[`create_ncov_data`](#simulator-create-ncov-data) | Create continuous simulated data with covariance across multiple regions.
[`gaussian`](#simulator-gaussian) | Create a 3D gaussian signal normalized to a given intensity.
[`n_spheres`](#simulator-n-spheres) | Generate a set of spheres in the brain mask space.
[`normal_noise`](#simulator-normal-noise) | Produce a normal noise distribution for all points in the brain mask.
[`sphere`](#simulator-sphere) | Create a sphere of given radius at some point p in the brain mask.
[`to_nifti`](#simulator-to-nifti) | Convert a numpy matrix to the nifti format and assign it the brain_mask's affine matrix.



**Examples:**

```pycon
>>> from nltools.data.simulator import Simulator
>>> sim = Simulator(random_state=42)
>>> # Create a dataset with signal in specific regions
>>> data = sim.create_data(levels=[1, -1, 1, -1], sigma=1, reps=10)
```

##### Methods

(simulator-create-cov-data)=
###### `create_cov_data`

```python
create_cov_data(cor, cov, sigma, *, mask = None, reps = 1, n_sub = 1, output_dir = None)
```

Create continuous simulated data with covariance within a single region.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cor` |  | amount of covariance between each voxel and Y variable | *required*
`cov` |  | amount of covariance between voxels | *required*
`sigma` |  | amount of noise to add | *required*
`mask` |  | region where activations are placed (a single mask image); defaults to a sphere if None | <code>None</code>
`reps` |  | number of data repetitions | <code>1</code>
`n_sub` |  | number of subjects to simulate | <code>1</code>
`output_dir` |  | string path of directory to output data.  If None, no data will be written | <code>None</code>

(simulator-create-data)=
###### `create_data`

```python
create_data(levels, sigma, *, radius = 5, center = None, reps = 1, output_dir = None)
```

Create simulated data with discrete intensity levels.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`levels` |  | vector of intensities or class labels | *required*
`sigma` |  | amount of noise to add | *required*
`radius` |  | vector of radius.  Will create multiple spheres if len(radius) > 1 | <code>5</code>
`center` |  | center(s) of sphere(s) of the form [px, py, pz] or [[px1, py1, pz1], ..., [pxn, pyn, pzn]] | <code>None</code>
`reps` |  | number of data repetitions useful for trials or subjects | <code>1</code>
`output_dir` |  | string path of directory to output data.  If None, no data will be written | <code>None</code>

(simulator-create-ncov-data)=
###### `create_ncov_data`

```python
create_ncov_data(cor, cov, sigma, *, masks = None, reps = 1, n_sub = 1, output_dir = None)
```

Create continuous simulated data with covariance across multiple regions.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cor` |  | amount of covariance between each voxel and Y variable (an int or a vector) | *required*
`cov` |  | amount of covariance between voxels (an int or a matrix) | *required*
`sigma` |  | amount of noise to add | *required*
`masks` |  | region(s) where we will have activations (list if more than one) | <code>None</code>
`reps` |  | number of data repetitions | <code>1</code>
`n_sub` |  | number of subjects to simulate | <code>1</code>
`output_dir` |  | string path of directory to output data.  If None, no data will be written | <code>None</code>

(simulator-gaussian)=
###### `gaussian`

```python
gaussian(mu, sigma, i_tot)
```

Create a 3D gaussian signal normalized to a given intensity.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mu` |  | average value of the gaussian signal (usually set to 0) | *required*
`sigma` |  | standard deviation | *required*
`i_tot` |  | sum total of activation (numerical integral over the gaussian returns this value) | *required*

(simulator-n-spheres)=
###### `n_spheres`

```python
n_spheres(radius, center)
```

Generate a set of spheres in the brain mask space.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`radius` |  | vector of radius.  Will create multiple spheres if len(radius) > 1 | *required*
`center` |  | a vector of sphere centers of the form [px, py, pz] or [[px1, py1, pz1], ..., [pxn, pyn, pzn]] | *required*

(simulator-normal-noise)=
###### `normal_noise`

```python
normal_noise(mu, sigma)
```

Produce a normal noise distribution for all points in the brain mask.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mu` |  | average value of the gaussian signal (usually set to 0) | *required*
`sigma` |  | standard deviation | *required*

(simulator-sphere)=
###### `sphere`

```python
sphere(r, p)
```

Create a sphere of given radius at some point p in the brain mask.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`r` |  | radius of the sphere | *required*
`p` |  | point (in coordinates of the brain mask) of the center of the sphere | *required*

(simulator-to-nifti)=
###### `to_nifti`

```python
to_nifti(m)
```

Convert a numpy matrix to the nifti format and assign it the brain_mask's affine matrix.

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`m` |  | the 3D numpy matrix we wish to convert to .nii | *required*



### Methods