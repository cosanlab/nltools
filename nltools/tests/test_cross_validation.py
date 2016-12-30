import os
import numpy as np
import nibabel as nb
import pandas as pd
import glob
from nltools.cross_validation import KFoldStratified

def check_valid_split(train, test, n_samples=None):
    # Use python sets to get more informative assertion failure messages
    train, test = set(train), set(test)

    # Train and test split should not overlap
    assert train.intersection(test)==set()

    if n_samples is not None:
        # Check that the union of train an test split cover all the indices
        assert train.union(test)==set(range(n_samples))


def check_cv_coverage(cv, X, y, groups, expected_n_splits=None):
    n_samples = X.shape[0]
    # Check that a all the samples appear at least once in a test fold
    if expected_n_splits is not None:
        assert cv.get_n_splits(X, y, groups)==expected_n_splits
    else:
        expected_n_splits = cv.get_n_splits(X, y, groups)

    collected_test_samples = set()
    iterations = 0
    for train, test in cv.split(X, y, groups):
        check_valid_split(train, test, n_samples=n_samples)
        iterations += 1
        collected_test_samples.update(test)

    # Check that the accumulated test samples cover the whole dataset
    assert iterations==expected_n_splits
    if n_samples is not None:
        assert collected_test_samples==set(range(n_samples))

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
