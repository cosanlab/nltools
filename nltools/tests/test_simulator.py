from nltools.simulator import Simulator, SimulateGrid
import numpy as np

def test_simulator(tmpdir):
    sim = Simulator()
    r = 10
    sigma = 1
    y = [0, 1]
    n_reps = 3
    output_dir = str(tmpdir)
    shape = (91, 109, 91)
    dat = sim.create_data(y, sigma, reps=n_reps, output_dir=None)
    assert len(dat) == n_reps*len(y)
    assert len(dat.Y) == n_reps*len(y)

def test_simulategrid_fpr(tmpdir):
    grid_width = 10
    n_subjects = 25
    n_simulations = 100
    thresh = .05
    bonferroni_threshold = thresh/(grid_width**2)
    simulation = SimulateGrid(grid_width=grid_width, n_subjects=n_subjects )
    simulation.plot_grid_simulation(threshold=bonferroni_threshold, threshold_type='p', n_simulations=n_simulations)

    assert simulation.isfit
    assert simulation.grid_width == grid_width
    assert simulation.p_values.shape == (grid_width, grid_width)
    assert simulation.thresholded.shape == (grid_width, grid_width)
    assert simulation.fp_percent <= bonferroni_threshold
    assert len(simulation.multiple_fp) == n_simulations
    assert np.sum(simulation.multiple_fp > 0)/n_simulations <= (thresh + .03)

def test_simulategrid_fdr(tmpdir):
    grid_width = 100
    n_subjects = 25
    n_simulations = 100
    thresh = .05
    signal_amplitude = 1
    signal_width = 10
    simulation = SimulateGrid(signal_amplitude=signal_amplitude, signal_width=signal_width, grid_width=grid_width, n_subjects=n_subjects)
    simulation.plot_grid_simulation(threshold=thresh, threshold_type='q', n_simulations=n_simulations, correction='fdr')

    assert len(simulation.multiple_fdr) == n_simulations
    assert np.mean(simulation.multiple_fdr) < thresh
    assert simulation.signal_width == signal_width
    assert simulation.correction == 'fdr'
