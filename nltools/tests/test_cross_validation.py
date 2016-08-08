import os
import numpy as np
import nibabel as nb
import pandas as pd
import glob
from nltools.cross_validation import KFoldSubject, KFoldStratified, LeaveOneSubjectOut

def check_valid_split(train, test, n_samples=None):
    # Use python sets to get more informative assertion failure messages
    train, test = set(train), set(test)

    # Train and test split should not overlap
    assert train.intersection(test) == set()

    if n_samples is not None:
        # Check that the union of train an test split cover all the indices
        assert train.union(test) == set(range(n_samples))


def check_cv_coverage(cv, expected_n_iter=None, n_samples=None):
    # Check that a all the samples appear at least once in a test fold
    if expected_n_iter is not None:
        assert len(cv) == expected_n_iter
    else:
        expected_n_iter = len(cv)

    collected_test_samples = set()
    iterations = 0
    for train, test in cv:
        check_valid_split(train, test, n_samples=n_samples)
        iterations += 1
        collected_test_samples.update(test)

    # Check that the accumulated test samples cover the whole dataset
    assert iterations == expected_n_iter
    if n_samples is not None:
        assert collected_test_samples == set(range(n_samples))


def test_kfoldsubject_indices():
    
    # n=17 - 2 measurements per condition
    holdout = np.concatenate([np.arange(1,18),np.arange(1,18)])
    n_folds = 5
    cv = KFoldSubject(len(holdout), holdout, n_folds=n_folds)

    check_cv_coverage(cv, expected_n_iter=n_folds, n_samples=len(holdout))
    
