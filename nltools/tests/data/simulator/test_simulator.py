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


def test_plot_grid_simulation_forwards_correction_to_run_multiple_simulations():
    """C3 (q31x trs9, row 16): plot_grid_simulation must forward `correction` to
    the `run_multiple_simulations` call it makes internally.

    Both of `plot_grid_simulation`'s internal calls validate `correction`, so an
    unsupported value raises whichever one is reached first; the test pins that
    the value is forwarded at all, not which call rejects it.
    """
    sim = SimulateGrid(grid_width=10, n_subjects=10, random_state=0)
    sim.fit()
    sim.threshold_simulation(threshold=0.05, threshold_type="p")
    assert sim.thresholded is not None

    with pytest.raises(ValueError, match="correction"):
        sim.plot_grid_simulation(
            threshold=0.05, threshold_type="p", n_simulations=5, correction="bogus"
        )


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


def _isotropic_mask(voxel_size, extent_mm=72.0):
    """An all-ones cubic brain mask with isotropic `voxel_size` mm voxels centered on the origin."""
    n = int(round(extent_mm / voxel_size))
    shape = (n, n, n)
    affine = np.diag([voxel_size, voxel_size, voxel_size, 1.0])
    affine[:3, 3] = -voxel_size * (np.array(shape) // 2)
    return nib.Nifti1Image(np.ones(shape, dtype=np.float32), affine)


@pytest.mark.parametrize("voxel_size", [3.0])
def test_n_spheres_radius_is_millimeters(voxel_size):
    """Simulator radii are millimeters, so a sphere's volume does not depend on the grid."""
    sim = Simulator(brain_mask=_isotropic_mask(voxel_size))
    radius = 9.0

    volume = sim.n_spheres(radius, [0.0, 0.0, 0.0]).sum() * voxel_size**3

    analytic = 4.0 / 3.0 * np.pi * radius**3
    assert abs(volume - analytic) / analytic < 0.15


def test_n_spheres_default_center_is_the_world_grid_center():
    """center=None places the sphere at the world coordinate of the grid center."""
    mask = _isotropic_mask(3.0)
    sim = Simulator(brain_mask=mask)

    sphere = sim.n_spheres(9.0, None)

    grid_center = nib.affines.apply_affine(mask.affine, np.array(mask.shape) // 2)
    world = nib.affines.apply_affine(mask.affine, np.argwhere(sphere > 0))
    assert np.allclose(world.mean(axis=0), grid_center, atol=3.0)


@pytest.mark.parametrize("voxel_size", [3.0])
@pytest.mark.parametrize("isotropic_scalar", [True])
def test_gaussian_mu_is_world_and_sigma_is_millimeters(voxel_size, isotropic_scalar):
    """`gaussian` takes a world-millimeter center and millimeter widths.

    `sigma` may be a bare scalar (isotropic) or one width per axis.
    """
    mask = _isotropic_mask(voxel_size)
    sim = Simulator(brain_mask=mask)
    mu = [6.0, -6.0, 0.0]
    sigma = 9.0

    blob = sim.gaussian(
        mu=mu,
        sigma=sigma if isotropic_scalar else np.array([sigma] * 3),
        i_tot=100.0,
    )

    ijk = np.argwhere(np.ones_like(blob, dtype=bool))
    world = nib.affines.apply_affine(mask.affine, ijk)
    weights = blob[tuple(ijk.T)]
    weights = weights / weights.sum()
    centroid = (world * weights[:, None]).sum(axis=0)
    spread = np.sqrt((((world - centroid) ** 2) * weights[:, None]).sum(axis=0))

    assert np.allclose(centroid, mu, atol=voxel_size)
    assert np.allclose(spread, sigma, rtol=0.15)


def test_create_data_radius_and_center_are_millimeters():
    """create_data forwards millimeter geometry to the same sphere builder as n_spheres."""
    mask = _isotropic_mask(3.0)
    sim = Simulator(brain_mask=mask, random_state=0)
    center = [12.0, -10.0, 8.0]

    data = sim.create_data([1, -1], 0.0, radius=9.0, center=center, reps=1)

    expected = int(sim.n_spheres(9.0, center).sum())
    assert int(np.sum(np.abs(data[0].to_nifti().get_fdata()) > 1e-6)) == expected


def test_create_data_keeps_the_simulators_brain_mask():
    """E-02: the simulated images stay on the mask the Simulator was built with.

    The result used to be resampled onto the template mask, so a custom grid was
    silently replaced by the 3mm MNI one.
    """
    mask = _isotropic_mask(3.0, extent_mm=36.0)
    sim = Simulator(brain_mask=mask, random_state=0)

    data = sim.create_data([1, -1], sigma=0.0, radius=6.0)

    assert data.mask.shape == mask.shape
    assert np.allclose(data.mask.affine, mask.affine)
    assert data.data.shape[1] == int(np.prod(mask.shape))


class _CovarianceSpy:
    """Stands in for a RandomState and records the covariance it is handed."""

    def __init__(self):
        self.cov = None

    def multivariate_normal(self, mean, cov, size=1):
        self.cov = np.asarray(cov)
        return np.zeros((size, len(mean)))

    def standard_normal(self, size=None):
        return np.zeros(size)


def _one_voxel_mask(sim, ijk):
    """A single-voxel region mask on the simulator's grid."""
    data = np.zeros(sim.brain_mask.shape, dtype=np.float32)
    data[ijk] = 1.0
    return nib.Nifti1Image(data, affine=sim.brain_mask.affine)


def test_create_ncov_data_covariance_blocks_follow_each_region_size():
    """E-01: the cross-region blocks used the wrong region's voxel count.

    With regions of one and two voxels the matrix came out asymmetric, so numpy
    factorized something other than the requested covariance.
    """
    sim = Simulator(brain_mask=_small_mask(), random_state=0)
    one_voxel = _one_voxel_mask(sim, (1, 1, 1))
    two_voxel = np.zeros(sim.brain_mask.shape, dtype=np.float32)
    two_voxel[5, 5, 5] = 1.0
    two_voxel[5, 5, 6] = 1.0
    masks = [one_voxel, nib.Nifti1Image(two_voxel, affine=sim.brain_mask.affine)]
    spy = _CovarianceSpy()
    sim.random_state = spy

    sim.create_ncov_data(
        cor=[0, 0], cov=[[0.2, 0.1], [0.1, 0.3]], sigma=0, masks=masks, reps=2
    )

    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.1, 0.1],
            [0.0, 0.1, 1.0, 0.3],
            [0.0, 0.1, 0.3, 1.0],
        ]
    )
    assert np.allclose(spy.cov, expected)


def test_create_ncov_data_draws_noise_for_every_repetition():
    """E-07: one voxel-shaped noise draw was broadcast across every repetition."""
    sim = Simulator(brain_mask=_small_mask(), random_state=0)
    sphere = sim.n_spheres(3, None)
    masks = nib.Nifti1Image(sphere.astype(np.float32), affine=sim.brain_mask.affine)

    sim.create_ncov_data(cor=1, cov=1, sigma=1, masks=masks, reps=3)

    outside_sphere = sim.data.get_fdata()[sphere == 0]
    assert not np.allclose(outside_sphere[:, 0], outside_sphere[:, 1])
    assert not np.allclose(outside_sphere[:, 1], outside_sphere[:, 2])


def test_signal_mask_is_exactly_signal_width_wide():
    """E-09: floor/ceil around the grid center made odd-parity masks a pixel wide.

    The oversized mask also inflated recovery rates past 1, because both rates
    divided by `signal_width**2` instead of the mask's own size.
    """
    sim = SimulateGrid(grid_width=6, signal_width=3, signal_amplitude=1, random_state=0)

    assert sim.signal_mask.sum() == 9
    assert sim._calc_true_positives(np.ones((6, 6))) == 1.0
    assert sim._calc_false_positives((sim.signal_mask == 0).astype(float)) == 1.0


def test_false_discovery_rate_counts_negative_discoveries():
    """E-10: only positive discoveries were counted, and an empty map gave NaN.

    Thresholding is two-sided, so a negative discovery is a discovery; a map with
    none at all has a false discovery rate of 0, not NaN.
    """
    sim = SimulateGrid(grid_width=4, signal_width=2, signal_amplitude=1, random_state=0)
    thresholded = np.zeros((4, 4))
    thresholded[0, 0] = -1.0  # outside the centered 2x2 signal box
    thresholded[1, 1] = -1.0  # inside it

    assert sim._calc_false_discovery_rate(thresholded) == 0.5
    assert sim._calc_false_discovery_rate(np.zeros((4, 4))) == 0.0


def test_add_signal_clears_the_previous_fit():
    """E-11: statistics computed before `add_signal` used to survive it.

    The object reported t-values for data it no longer held and refused to refit.
    """
    sim = SimulateGrid(grid_width=4, n_subjects=10, random_state=0)
    sim.fit()
    sim.threshold_simulation(threshold=0.05, threshold_type="p")

    sim.add_signal(signal_width=2, signal_amplitude=10)

    assert sim.isfit is False
    assert sim.thresholded is None
    sim.fit()
    assert sim.t_values[sim.signal_mask == 1].mean() > sim.t_values[
        sim.signal_mask == 0
    ].mean()


def test_plot_grid_simulation_rethresholds_at_the_requested_threshold():
    """E-12: a cached thresholded map was drawn under the new threshold's caption.

    The middle panel showed the old map while the panel title and the null
    distribution came from the arguments of this call.
    """
    sim = SimulateGrid(grid_width=2, n_subjects=10, random_state=0)
    sim.fit()
    sim.t_values = np.array([[2.0, 4.0], [0.0, 0.0]])
    sim.p_values = np.full((2, 2), 0.01)
    sim.threshold_simulation(threshold=1.0, threshold_type="t")

    sim.plot_grid_simulation(threshold=3.0, threshold_type="t", n_simulations=2)

    assert sim.thresholded.tolist() == [[0.0, 4.0], [0.0, 0.0]]
    assert sim.threshold == 3.0


def test_corrected_threshold_records_the_requested_q():
    """E-15: the recorded FDR cutoff came from the default q, not the requested one.

    Thresholding used `q=threshold`, so the stored cutoff could say nothing
    survived while three pixels did.
    """
    sim = SimulateGrid(grid_width=2, n_subjects=10, random_state=0)
    sim.fit()
    sim.t_values = np.array([[3.0, 2.5], [2.0, 1.0]])
    sim.p_values = np.array([[0.02, 0.04], [0.08, 0.9]])

    sim.threshold_simulation(threshold=0.2, threshold_type="q", correction="fdr")

    assert sim.corrected_threshold == pytest.approx(0.08)
    assert sim.thresholded.tolist() == [[3.0, 2.5], [2.0, 0.0]]
