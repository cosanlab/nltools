import os
import numpy as np
import nibabel as nb
import glob
from nltools.simulator import Simulator

def test_simulator(tmpdir):
    sim = Simulator()
    r = 10
    sigma = 1
    y = [0, 1]
    n_reps = 3
    output_dir = str(tmpdir)
    sim.create_data(y, sigma, reps=n_reps, output_dir=output_dir)
    flist = glob.glob(str(tmpdir.join('centered*nii.gz')))

    shape = (91, 109, 91)
    sim_img = nb.concat_images(flist)
    assert len(sim.data) == n_reps*len(y)
    assert sim_img.shape[0:3] == shape