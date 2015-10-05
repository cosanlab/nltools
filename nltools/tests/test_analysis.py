import os
import numpy as np
import nibabel as nb
from nilearn._utils import testing

from nltools import analysis


def test_predict(tmpdir):
    shape = (40, 41, 42)
    length = 17

    img_dat, _ = testing.generate_fake_fmri(shape=shape, length=length)
    Y = np.random.randint(2, size=length)

    algorithm = 'svm'
    output_dir = str(tmpdir)
    cv = {'type': 'kfolds', 'n_folds': 5}
    extra = {'kernel': 'linear'}
    weightmap_name = "%s_weightmap.nii.gz" % algorithm

    predict = analysis.Predict(img_dat, Y, algorithm=algorithm,
                               output_dir=output_dir,
                               cv_dict=cv,
                               **extra)

    predict.predict(save_output=False, save_plot=False)

    weightmap_img = nb.load(os.path.join(output_dir, weightmap_name))

    assert weightmap_img.shape == shape
