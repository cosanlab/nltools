import os
import numpy as np
import nibabel as nb
# from nilearn._utils import testing

from nltools import analysis, simulator


def test_simulator(tmpdir):
    sim = simulator.Simulator()
    r = 10
    sigma = 1
    y = [0, 1]
    n_reps = 3
    output_dir = str(tmpdir)
    sim.create_data(y, sigma, reps=n_reps, output_dir=output_dir)

    shape = (91, 109, 91)
    sim_img = nb.load(os.path.join(output_dir, 'centered_sphere_0_0.nii.gz'))
    assert len(sim.data) == n_reps*len(y)
    assert sim_img.shape == shape
