import matplotlib

matplotlib.use("Agg")

import pytest
import nibabel as nib
from nltools.data.simulator import Simulator, SimulateGrid
import numpy as np


def _small_mask():
    """A tiny all-ones brain mask for fast Simulator smoke tests."""
    return nib.Nifti1Image(np.ones((12, 12, 12), dtype=np.float32), affine=np.eye(4))


def test_simulator_accepts_nibabel_mask():
    """F086: a valid Nifti1Image brain_mask must not raise."""
    mask = _small_mask()
    sim = Simulator(brain_mask=mask)
    assert isinstance(sim.brain_mask, nib.nifti1.Nifti1Image)


def test_n_spheres_multi_radius_default_center():
    """F102: multiple radii with center=None must build one center per radius."""
    sim = Simulator(brain_mask=_small_mask())
    A = sim.n_spheres([3, 3], None)
    assert A.shape == (12, 12, 12)
    assert A.sum() > 0
    # A float scalar radius must also be accepted (not just Python int).
    B = sim.n_spheres(3.0, None)
    assert B.sum() > 0


def test_create_ncov_data_int_cov():
    """F087: an integer cov must be listified into a covariance matrix."""
    sim = Simulator(brain_mask=_small_mask(), random_state=0)
    sphere = sim.n_spheres(3, None)
    masks = nib.Nifti1Image(sphere.astype(np.float32), affine=sim.brain_mask.affine)
    sim.create_ncov_data(
        cor=1, cov=1, sigma=1, masks=masks, reps=5, n_sub=1, output_dir=None
    )
    assert sim.data.shape[-1] == 5


def test_plot_grid_simulation_already_fit():
    """F101: plotting an already-fit grid must still threshold (thresholded != None)."""
    sim = SimulateGrid(grid_width=10, n_subjects=10, random_state=0)
    sim.fit()  # sets isfit=True but leaves self.thresholded == None
    assert sim.thresholded is None
    sim.plot_grid_simulation(threshold=0.05, threshold_type="p", n_simulations=10)
    assert sim.thresholded is not None


def test_sphere_builds_binary_region():
    """F189: sphere() returns a 0/1 volume centered at p with the mask's shape."""
    sim = Simulator(brain_mask=_small_mask())
    center = [6, 6, 6]
    A = sim.sphere(3, center)
    assert A.shape == (12, 12, 12)
    # Values are strictly binary and the center voxel is inside the sphere.
    assert set(np.unique(A)) <= {0.0, 1.0}
    assert A[6, 6, 6] == 1
    assert A.sum() > 1


def test_gaussian_normalizes_to_total_intensity():
    """F189: gaussian() rescales the blob so its voxels sum to i_tot."""
    sim = Simulator(brain_mask=_small_mask())
    i_tot = 100.0
    g = sim.gaussian(
        mu=np.array([6, 6, 6]), sigma=np.array([2.0, 2.0, 2.0]), i_tot=i_tot
    )
    assert g.shape == (12, 12, 12)
    # The blob is normalized so the total activation equals i_tot (mask is
    # all-ones, so masking removes nothing).
    assert np.isclose(g.sum(), i_tot)
    assert np.all(g >= 0)


@pytest.mark.filterwarnings("ignore:covariance is not symmetric positive-semidefinite")
def test_create_cov_data_single_subject():
    """F189: create_cov_data() lays reps into the 4th dim with a matching y."""
    sim = Simulator(brain_mask=_small_mask(), random_state=0)
    sphere = sim.n_spheres(3, None)
    mask = nib.Nifti1Image(sphere.astype(np.float32), affine=sim.brain_mask.affine)
    reps = 8
    sim.create_cov_data(cor=0.8, cov=0.4, sigma=1, mask=mask, reps=reps, n_sub=1)
    assert sim.data.shape[-1] == reps
    assert len(sim.y) == reps
    assert sim.rep_id == [1] * reps


@pytest.mark.filterwarnings("ignore:covariance is not symmetric positive-semidefinite")
def test_create_cov_data_multi_subject_concats():
    """F189: n_sub>1 concatenates per-subject reps and grows y/rep_id in step."""
    sim = Simulator(brain_mask=_small_mask(), random_state=0)
    sphere = sim.n_spheres(3, None)
    mask = nib.Nifti1Image(sphere.astype(np.float32), affine=sim.brain_mask.affine)
    reps, n_sub = 4, 2
    sim.create_cov_data(cor=0.8, cov=0.4, sigma=1, mask=mask, reps=reps, n_sub=n_sub)
    assert sim.data.shape[-1] == reps * n_sub
    assert len(sim.y) == reps * n_sub
    assert len(sim.rep_id) == reps * n_sub
    assert set(sim.rep_id) == {1, 2}


@pytest.mark.slow
def test_simulator(tmpdir):
    sim = Simulator()
    sigma = 1
    y = [0, 1]
    n_reps = 3
    dat = sim.create_data(y, sigma, reps=n_reps, output_dir=None)
    assert len(dat) == n_reps * len(y)
    assert len(dat.Y) == n_reps * len(y)


@pytest.mark.slow
def test_simulategrid_fpr(tmpdir):
    grid_width = 10
    n_subjects = 25
    n_simulations = 100
    thresh = 0.05
    bonferroni_threshold = thresh / (grid_width**2)
    simulation = SimulateGrid(
        grid_width=grid_width, n_subjects=n_subjects, random_state=0
    )
    simulation.fit()
    simulation.threshold_simulation(threshold=bonferroni_threshold, threshold_type="p")
    simulation.run_multiple_simulations(
        threshold=bonferroni_threshold,
        threshold_type="p",
        n_simulations=n_simulations,
    )

    assert simulation.isfit
    assert simulation.grid_width == grid_width
    assert simulation.p_values.shape == (grid_width, grid_width)
    assert simulation.thresholded.shape == (grid_width, grid_width)
    assert simulation.fp_percent <= bonferroni_threshold
    assert len(simulation.multiple_fp) == n_simulations
    assert np.sum(simulation.multiple_fp > 0) / n_simulations <= (thresh + 0.03)


@pytest.mark.slow
def test_simulategrid_fdr(tmpdir):
    grid_width = 100
    n_subjects = 25
    n_simulations = 100
    thresh = 0.05
    signal_amplitude = 1
    signal_width = 10
    simulation = SimulateGrid(
        signal_amplitude=signal_amplitude,
        signal_width=signal_width,
        grid_width=grid_width,
        n_subjects=n_subjects,
        random_state=0,
    )
    simulation.fit()
    simulation.threshold_simulation(
        threshold=thresh, threshold_type="q", correction="fdr"
    )
    simulation.run_multiple_simulations(
        threshold=thresh,
        threshold_type="q",
        n_simulations=n_simulations,
    )

    assert len(simulation.multiple_fdr) == n_simulations
    assert np.mean(simulation.multiple_fdr) < thresh
    assert simulation.signal_width == signal_width
    assert simulation.correction == "fdr"
