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
    shape = (91, 109, 91)
    dat = sim.create_data(y, sigma, reps=n_reps, output_dir=None)
    assert len(dat) == n_reps*len(y)
    assert len(dat.Y) == n_reps*len(y)
