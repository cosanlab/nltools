import os
import numpy as np
import nibabel as nb
import pandas as pd
# from nilearn._utils import testing
from nltools import analysis, simulator
from nltools.data import Brain_Data


def test_predict_svm(tmpdir, sim):
    r = 10
    sigma = .2
    y = [0, 1]
    n_reps = 10
    output_dir = str(tmpdir)
    sim.create_data(y, sigma, reps=n_reps, output_dir=None)

    # shape = (40, 41, 42)
    # length = 17

    # img_dat, _ = testing.generate_fake_fmri(shape=shape, length=length)
    # Y = np.random.randint(2, size=length)

    algorithm = 'svm'
    cv = {'type': 'kfolds', 'n_folds': 5, 'subject_id': sim.rep_id}
    extra = {'kernel': 'linear'}
    weightmap_name = "%s_weightmap.nii.gz" % algorithm

    predict = analysis.Predict(sim.data, sim.y, algorithm=algorithm,
                               output_dir=output_dir,
                               cv_dict=cv,
                               **extra)

    # predict.predict() # Save_Plot isn't working for SVM analysis, planning on deprecating analysis.Predict at some point, so not a big deal
    predict.predict(save_plot=False)

    weightmap_img = nb.load(os.path.join(output_dir, weightmap_name))

    assert predict.mcr_xval >= .99
    assert weightmap_img.shape == sim.data[0].shape


def test_predict_svr(tmpdir, sim):
    r = 10
    sigma = .1
    y = [1, 2, 3]
    n_reps = 10
    output_dir = str(tmpdir)
    sim.create_data(y, sigma, reps=n_reps, output_dir=None)

    algorithm = 'svr'
    cv = {'type': 'kfolds', 'n_folds': 5, 'subject_id': sim.rep_id}
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


def test_roc(tmpdir, sim):
    r = 10
    sigma = .1
    y = [0, 1]
    n_reps = 10
    output_dir = str(tmpdir)
    sim.create_data(y, sigma, reps=n_reps, output_dir=None)
    dat = Brain_Data(data=sim.data,Y=pd.DataFrame(sim.y))

    algorithm = 'svm'
    output_dir = str(tmpdir)
    # cv = {'type': 'kfolds', 'n_folds': 5, 'subject_id': sim.rep_id}
    extra = {'kernel': 'linear'}

    output = dat.predict(algorithm='svm',**extra)
    # predict = analysis.Predict(sim.data, sim.y, algorithm=algorithm,
    #                            output_dir=output_dir,
    #                            cv_dict=cv,
    #                            **extra)

    # predict.predict() # Save_Plot isn't working for SVM analysis, planning on deprecating analysis.Predict at some point, so not a big deal
    # predict.predict(save_plot=False)

    # Single-Interval
    roc = analysis.Roc(input_values=output['yfit_all'], binary_outcome=output['Y'] == 1)
    # roc = analysis.Roc(
    #     input_values=predict.yfit_xval, binary_outcome=np.array(sim.y) == 1)
    roc.plot()
    roc.summary()
    assert roc.accuracy == 1
