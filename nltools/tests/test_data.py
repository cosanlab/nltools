import os
import numpy as np
import nibabel as nb
import pandas as pd
import glob
from nltools.simulator import Simulator
from nltools.data import Brain_Data
from nltools.data import threshold

def test_data(tmpdir):
    sim = Simulator()
    r = 10
    sigma = 1
    y = [0, 1]
    n_reps = 3
    output_dir = str(tmpdir)
    sim.create_data(y, sigma, reps=n_reps, output_dir=output_dir)

    shape_3d = (91, 109, 91)
    shape_2d = (6, 238955)
    y=pd.read_csv(os.path.join(str(tmpdir.join('y.csv'))),header=None,index_col=None)
    y=np.array(y)[0]
    flist = glob.glob(str(tmpdir.join('centered*.nii.gz')))
    dat = Brain_Data(data=flist,Y=y)

    # Test shape
    assert dat.shape() == shape_2d

    # Test Mean
    assert dat.mean().shape()[0] == shape_2d[1]

    # Test Std
    assert dat.std().shape()[0] == shape_2d[1]

    # Test to_nifti
    d = dat.to_nifti
    assert d().shape[0:3] == shape_3d

    # # Test T-test
    out = dat.ttest()
    assert out['t'].shape()[0]==shape_2d[1]

    # Test Regress
    dat.X = pd.DataFrame({'Intercept':np.ones(len(dat.Y)),'X1':dat.Y},index=None)
    out = dat.regress()
    assert out['beta'].shape() == (2,shape_2d[1])

    # Test indexing
    assert out['t'][1].shape()[0] == shape_2d[1]

    # Test threshold
    i=1
    tt = threshold(out['t'][i], out['p'][i], threshold_dict={'fdr':.05})
    assert tt.shape()[0] == shape_2d[1]
