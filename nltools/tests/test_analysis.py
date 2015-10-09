import os
import numpy as np
import nibabel as nb
# from nilearn._utils import testing

from nltools import analysis, simulator

def test_simulator(tmpdir):
    sim = simulator.Simulator()
    r = 10
    sigma = 1
    y = [0,1]
    n_reps = 10
    sim.create_data(y, sigma, reps = n_reps, output_dir = tmpdir)
    assert len(sim.data) == n_reps*len(y)

def test_predict_svm(tmpdir):
    sim = simulator.Simulator()
    r = 10
    sigma = .2
    y = [0,1]
    n_reps = 10
    sim.create_data(y, sigma, reps = n_reps, output_dir = tmpdir)

    # shape = (40, 41, 42)
    # length = 17

    # img_dat, _ = testing.generate_fake_fmri(shape=shape, length=length)
    # Y = np.random.randint(2, size=length)

    algorithm = 'svm'
    output_dir = str(tmpdir)
    cv = {'type': 'kfolds', 'n_folds': 5,'subject_id':sim.rep_id}
    extra = {'kernel': 'linear'}
    weightmap_name = "%s_weightmap.nii.gz" % algorithm

    predict = analysis.Predict(sim.data, sim.y, algorithm=algorithm,
                               output_dir=output_dir,
                               cv_dict=cv,
                               **extra)

    predict.predict()

    weightmap_img = nb.load(os.path.join(output_dir, weightmap_name))

    assert predict.mcr_xval==.99
    assert weightmap_img.shape == sim.data[0].shape

def test_predict_svr(tmpdir):
    sim = simulator.Simulator()
    r = 10
    sigma = .1
    y = [1,2,3]
    n_reps = 10
    sim.create_data(y, sigma, reps = n_reps, output_dir = tmpdir)

    algorithm = 'svm'
    output_dir = str(tmpdir)
    cv = {'type': 'kfolds', 'n_folds': 5,'subject_id':sim.rep_id}
    extra = {'kernel': 'linear'}
    weightmap_name = "%s_weightmap.nii.gz" % algorithm

    predict = analysis.Predict(sim.data, sim.y, algorithm=algorithm,
                               output_dir=output_dir,
                               cv_dict=cv,
                               **extra)

    predict.predict()

    weightmap_img = nb.load(os.path.join(output_dir, weightmap_name))

    assert predict.r_xval >= .99
    assert weightmap_img.shape == sim.data[0].shape

