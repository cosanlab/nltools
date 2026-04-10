## `simulator`

NeuroLearn Simulator Tools
==========================

Tools to simulate multivariate data.

**Classes:**

Name | Description
---- | -----------
[`SimulateGrid`](#SimulateGrid) | Simulate 2D grid data for testing statistical methods.
[`Simulator`](#Simulator) | Simulate fMRI data with realistic spatial and temporal characteristics.



### Classes

#### `SimulateGrid`

```python
SimulateGrid(grid_width = 100, signal_width = 20, n_subjects = 20, sigma = 1, signal_amplitude = None, random_state = None)
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
[`data`](#data) |  | The simulated data array of shape (n_subjects, grid_width, grid_width).
[`t_values`](#t_values) |  | T-statistic values after fitting.
[`p_values`](#p_values) |  | P-values after fitting.
[`thresholded`](#thresholded) |  | Thresholded statistical map.
[`isfit`](#isfit) |  | Whether fit() has been called.

**Examples:**

```pycon
>>> from nltools.data.simulator import SimulateGrid
>>> sim = SimulateGrid(signal_amplitude=0.5, random_state=42)
>>> sim.fit(n_permute=1000)
>>> sim.plot()
```

**Methods:**

Name | Description
---- | -----------
[`add_signal`](#add_signal) | Add rectangular signal to self.data
[`create_mask`](#create_mask) | Create a mask for where the signal is located in grid.
[`fit`](#fit) | Run ttest on self.data
[`plot_grid_simulation`](#plot_grid_simulation) | Create a plot of the simulations
[`run_multiple_simulations`](#run_multiple_simulations) | This method will run multiple simulations to calculate overall false positive rate
[`threshold_simulation`](#threshold_simulation) | Threshold simulation

##### Methods

###### `add_signal`

```python
add_signal(signal_width = 20, signal_amplitude = 1)
```

Add rectangular signal to self.data

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`signal_width` | <code>[int](#int)</code> | width of signal box | <code>20</code>
`signal_amplitude` | <code>[int](#int)</code> | intensity of signal | <code>1</code>

###### `create_mask`

```python
create_mask(signal_width)
```

Create a mask for where the signal is located in grid.

###### `fit`

```python
fit()
```

Run ttest on self.data

###### `plot_grid_simulation`

```python
plot_grid_simulation(threshold, threshold_type, n_simulations = 100, correction = None)
```

Create a plot of the simulations

###### `run_multiple_simulations`

```python
run_multiple_simulations(threshold, threshold_type, n_simulations = 100, correction = None)
```

This method will run multiple simulations to calculate overall false positive rate

###### `threshold_simulation`

```python
threshold_simulation(threshold, threshold_type, correction = None)
```

Threshold simulation

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`threshold` | <code>[float](#float)</code> | threshold to apply to simulation | *required*
`threshhold_type` | <code>[str](#str)</code> | type of threshold to use can be a specific t-value or p-value ['t', 'p', 'q'] | *required*

#### `Simulator`

```python
Simulator(brain_mask = None, output_dir = None, random_state = None)
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
[`brain_mask`](#brain_mask) |  | The brain mask image used for simulation.
[`nifti_masker`](#nifti_masker) |  | Fitted NiftiMasker for converting between 4D data and 2D arrays.
[`output_dir`](#output_dir) |  | Output directory path.
[`random_state`](#random_state) |  | Random state for reproducible simulations.

**Examples:**

```pycon
>>> from nltools.data.simulator import Simulator
>>> sim = Simulator(random_state=42)
>>> # Create a dataset with signal in specific regions
>>> data = sim.create_data(y=[1, -1, 1, -1], sigma=1, n_reps=10)
```

**Methods:**

Name | Description
---- | -----------
[`create_cov_data`](#create_cov_data) | create continuous simulated data with covariance
[`create_data`](#create_data) | create simulated data with integers
[`create_ncov_data`](#create_ncov_data) | create continuous simulated data with covariance
[`gaussian`](#gaussian) | create a 3D gaussian signal normalized to a given intensity
[`n_spheres`](#n_spheres) | generate a set of spheres in the brain mask space
[`normal_noise`](#normal_noise) | produce a normal noise distribution for all all points in the brain mask
[`sphere`](#sphere) | create a sphere of given radius at some point p in the brain mask
[`to_nifti`](#to_nifti) | convert a numpy matrix to the nifti format and assign to it the brain_mask's affine matrix

##### Methods

###### `create_cov_data`

```python
create_cov_data(cor, cov, sigma, mask = None, reps = 1, n_sub = 1, output_dir = None)
```

create continuous simulated data with covariance

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cor` |  | amount of covariance between each voxel and Y variable | *required*
`cov` |  | amount of covariance between voxels | *required*
`sigma` |  | amount of noise to add | *required*
`radius` |  | vector of radius.  Will create multiple spheres if len(radius) > 1 | *required*
`center` |  | center(s) of sphere(s) of the form [px, py, pz] or [[px1, py1, pz1], ..., [pxn, pyn, pzn]] | *required*
`reps` |  | number of data repetitions | <code>1</code>
`n_sub` |  | number of subjects to simulate | <code>1</code>
`output_dir` |  | string path of directory to output data.  If None, no data will be written | <code>None</code>
`**kwargs` |  | Additional keyword arguments to pass to the prediction algorithm | *required*

###### `create_data`

```python
create_data(levels, sigma, radius = 5, center = None, reps = 1, output_dir = None)
```

create simulated data with integers

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`levels` |  | vector of intensities or class labels | *required*
`sigma` |  | amount of noise to add | *required*
`radius` |  | vector of radius.  Will create multiple spheres if len(radius) > 1 | <code>5</code>
`center` |  | center(s) of sphere(s) of the form [px, py, pz] or [[px1, py1, pz1], ..., [pxn, pyn, pzn]] | <code>None</code>
`reps` |  | number of data repetitions useful for trials or subjects | <code>1</code>
`output_dir` |  | string path of directory to output data.  If None, no data will be written | <code>None</code>
`**kwargs` |  | Additional keyword arguments to pass to the prediction algorithm | *required*

###### `create_ncov_data`

```python
create_ncov_data(cor, cov, sigma, masks = None, reps = 1, n_sub = 1, output_dir = None)
```

create continuous simulated data with covariance

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`cor` |  | amount of covariance between each voxel and Y variable (an int or a vector) | *required*
`cov` |  | amount of covariance between voxels (an int or a matrix) | *required*
`sigma` |  | amount of noise to add | *required*
`mask` |  | region(s) where we will have activations (list if more than one) | *required*
`reps` |  | number of data repetitions | <code>1</code>
`n_sub` |  | number of subjects to simulate | <code>1</code>
`output_dir` |  | string path of directory to output data.  If None, no data will be written | <code>None</code>
`**kwargs` |  | Additional keyword arguments to pass to the prediction algorithm | *required*

###### `gaussian`

```python
gaussian(mu, sigma, i_tot)
```

create a 3D gaussian signal normalized to a given intensity

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mu` |  | average value of the gaussian signal (usually set to 0) | *required*
`sigma` |  | standard deviation | *required*
`i_tot` |  | sum total of activation (numerical integral over the gaussian returns this value) | *required*

###### `n_spheres`

```python
n_spheres(radius, center)
```

generate a set of spheres in the brain mask space

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`radius` |  | vector of radius.  Will create multiple spheres if len(radius) > 1 | *required*
`centers` |  | a vector of sphere centers of the form [px, py, pz] or [[px1, py1, pz1], ..., [pxn, pyn, pzn]] | *required*

###### `normal_noise`

```python
normal_noise(mu, sigma)
```

produce a normal noise distribution for all all points in the brain mask

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`mu` |  | average value of the gaussian signal (usually set to 0) | *required*
`sigma` |  | standard deviation | *required*

###### `sphere`

```python
sphere(r, p)
```

create a sphere of given radius at some point p in the brain mask

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`r` |  | radius of the sphere | *required*
`p` |  | point (in coordinates of the brain mask) of the center of the sphere | *required*

###### `to_nifti`

```python
to_nifti(m)
```

convert a numpy matrix to the nifti format and assign to it the brain_mask's affine matrix

**Parameters:**

Name | Type | Description | Default
---- | ---- | ----------- | -------
`m` |  | the 3D numpy matrix we wish to convert to .nii | *required*



### Methods