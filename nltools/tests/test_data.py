import os
import numpy as np
import nibabel as nb
import pandas as pd
import glob
from nltools.simulator import Simulator
from nltools.data import Brain_Data
from nltools.data import threshold
from nltools.mask import create_sphere
import matplotlib
matplotlib.use('TkAgg')

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
    y=pd.read_csv(os.path.join(str(tmpdir.join('y.csv'))),header=None,index_col=None).T
    flist = glob.glob(str(tmpdir.join('centered*.nii.gz')))
    
    # Test load list
    dat = Brain_Data(data=flist,Y=y)

    # Test load file
    assert Brain_Data(flist[0])

    # Test to_nifti
    d = dat.to_nifti()
    assert d.shape[0:3] == shape_3d

    # Test load nibabel
    assert Brain_Data(d)

    # Test shape
    assert dat.shape() == shape_2d

    # Test Mean
    assert dat.mean().shape()[0] == shape_2d[1]

    # Test Std
    assert dat.std().shape()[0] == shape_2d[1]

    # Test add
    new = dat + dat
    assert new.shape() == shape_2d

    # Test subtract
    new = dat - dat
    assert new.shape() == shape_2d

    # Test multiply
    new = dat * dat
    assert new.shape() == shape_2d

    # Test Iterator
    x = [x for x in dat]
    assert len(x)==len(dat)
    assert len(x[0].data.shape)==1

    # # Test T-test
    out = dat.ttest()
    assert out['t'].shape()[0]==shape_2d[1]

    # # # Test T-test - permutation method
    # out = dat.ttest(threshold_dict={'permutation':'tfce','n_permutations':50,'n_jobs':1})
    # assert out['t'].shape()[0]==shape_2d[1]

    # Test Regress
    dat.X = pd.DataFrame({'Intercept':np.ones(len(dat.Y)),'X1':np.array(dat.Y).flatten()},index=None)
    out = dat.regress()
    assert out['beta'].shape() == (2,shape_2d[1])

    # Test indexing
    assert out['t'][1].shape()[0] == shape_2d[1]

    # Test threshold
    i=1
    tt = threshold(out['t'][i], out['p'][i], .05)
    assert isinstance(tt,Brain_Data)

    # Test write
    dat.write(os.path.join(str(tmpdir.join('test_write.nii'))))
    assert Brain_Data(os.path.join(str(tmpdir.join('test_write.nii'))))

    # Test append
    assert dat.append(dat).shape()[0]==shape_2d[0]*2

    # Test distance
    distance = dat.distance(method='euclidean')
    assert distance.shape==(shape_2d[0],shape_2d[0])

    # Test predict
    stats = dat.predict(algorithm='svm', cv_dict={'type': 'kfolds','n_folds': 2, 'n':len(dat.Y)}, plot=False,**{'kernel':"linear"})

    # Support Vector Regression, with 5 fold cross-validation with Platt Scaling
    # This will output probabilities of each class
    stats = dat.predict(algorithm='svm', cv_dict=None, plot=False,**{'kernel':'linear','probability':True})

    assert isinstance(stats['weight_map'],Brain_Data)
    # Logistic classificiation, with 5 fold stratified cross-validation.  

    stats = dat.predict(algorithm='logistic', cv_dict={'type': 'kfolds','n_folds': 5, 'n':len(dat.Y)}, plot=False)
    assert isinstance(stats['weight_map'],Brain_Data)

    # Ridge classificiation, with 5 fold between-subject cross-validation, where data for each subject is held out together.
    stats = dat.predict(algorithm='ridgeClassifier', cv_dict=None,plot=False)
    assert isinstance(stats['weight_map'],Brain_Data)

    # Test Similarity
    r = dat.similarity(stats['weight_map'])
    assert len(r)==shape_2d[0]
    r2 = dat.similarity(stats['weight_map'].to_nifti())
    assert len(r2)==shape_2d[0]
    
    # Test apply_mask - might move part of this to test mask suite
    s1 = create_sphere([41, 64, 55], radius=10)
    assert isinstance(s1,nb.Nifti1Image)
    s2 = Brain_Data(s1)
    masked_dat = dat.apply_mask(s1)
    assert masked_dat.shape()[1]==np.sum(s2.data!=0)

    # Test extract_roi
    mask = create_sphere([41, 64, 55], radius=10)
    assert len(dat.extract_roi(mask))==shape_2d[0]

    # Test r_to_z
    z = dat.r_to_z()
    assert z.shape() == dat.shape()

    # Test copy
    d_copy = dat.copy()
    assert d_copy.shape() == dat.shape()

    # Test detrend
    detrend = dat.detrend()
    assert detrend.shape() == dat.shape()

    # # Test Plot
    # dat.plot()
    
    # Test Bootstrap

    # Test multivariate_similarity

