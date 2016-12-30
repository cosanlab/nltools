import os
import numpy as np
import nibabel as nb
import pandas as pd
import glob
from nltools.cross_validation import KFoldStratified
from sklearn.model_selection.tests.test_split import (check_valid_split,
                                                      check_cv_coverage,
                                                      test_kfold_indices)

def test_stratified_kfold_ratios():
    y = pd.DataFrame(np.random.randn(1000))*20+50
    n_folds = 5
    cv = KFoldStratified(n_splits=n_folds)
    for train, test in KFoldStratified(n_folds).split(np.zeros(len(y)), y):
        assert (y.iloc[train].mean()[0]>=47) & (y.iloc[train].mean()[0]<=53)
            
def test_kfoldstratified():
    y = pd.DataFrame(np.random.randn(50))*20+50
    n_folds = 5
    cv = KFoldStratified(n_splits=n_folds)
    check_cv_coverage(cv, X=np.zeros(len(y)),y=y, groups=None, expected_n_splits=n_folds)

    y = pd.DataFrame(np.random.randn(51))*20+50
    n_folds = 5
    cv = KFoldStratified(n_splits=n_folds)
    check_cv_coverage(cv, X=np.zeros(len(y)),y=y, groups=None, expected_n_splits=n_folds)
