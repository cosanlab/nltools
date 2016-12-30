from __future__ import division

'''
Cross-Validation Data Classes
=============================

Scikit-learn compatible classes for performing various 
types of cross-validation

'''

__all__ = ['KFoldStratified',
            'set_cv']
__author__ = ["Luke Chang"]
__license__ = "MIT"

from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import check_array
import numpy as np
import random
import pandas as pd

class KFoldStratified(_BaseKFold):
    """K-Folds cross validation iterator which stratifies continuous data (unlike scikit-learn equivalent).

    Provides train/test indices to split data in train test sets. Split
    dataset into k consecutive folds while ensuring that same subject is held
    out within each fold 
    Each fold is then used a validation set once while the k - 1 remaining
    folds form the training set.
    Extension of KFold from scikit-learn cross_validation model
    
    Args:
        n_splits: int, default=3
            Number of folds. Must be at least 2.
        shuffle: boolean, optional
            Whether to shuffle the data before splitting into batches.
        random_state: None, int or RandomState
            Pseudo-random number generator state used for random
            sampling. If None, use default numpy RNG for shuffling
    
    """

    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        super(KFoldStratified, self).__init__(n_splits, shuffle, random_state)

    def _make_test_folds(self, X, y=None, groups=None):
        y = pd.DataFrame(y)
        y_sort = y.sort_values(0)
        test_folds = np.nan*np.ones(len(y_sort))
        for k in range(self.n_splits):
            test_idx = y_sort.index[np.arange(k,len(y_sort),self.n_splits)]
            test_folds[y_sort.iloc[test_idx].index] = k
        return test_folds

    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.
        
        Args:
            X : array-like, shape (n_samples, n_features)
                Training data, where n_samples is the number of samples
                and n_features is the number of features.
                Note that providing ``y`` is sufficient to generate the splits and
                hence ``np.zeros(n_samples)`` may be used as a placeholder for
                ``X`` instead of actual training data.
            y : array-like, shape (n_samples,)
                The target variable for supervised learning problems.
                Stratification is done based on the y labels.
            groups : (object) Always ignored, exists for compatibility.

        Returns:
            train : (ndarray) The training set indices for that split.
            test : (ndarray) The testing set indices for that split.
        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return super(KFoldStratified, self).split(X, y, groups)

def set_cv(Y=None, cv_dict=None):
    """ Helper function to create a sci-kit learn compatible cv object using common parameters for prediction analyses.

    Args:
        Y:  (pd.DataFrame) Pandas Dataframe of Y labels
        cv_dict: (dict) Type of cross_validation to use. A dictionary of
            {'type': 'kfolds', 'n_folds': n},
            {'type': 'kfolds', 'n_folds': n, 'stratified': Y},
            {'type': 'kfolds', 'n_folds': n, 'subject_id': holdout}, or
            {'type': 'loso', 'subject_id': holdout}
    Returns:
        cv: a scikit-learn model-selection generator

     """

    if type(cv_dict) is dict:
        if cv_dict['type'] == 'kfolds':
            if 'subject_id' in cv_dict: # Hold out subjects within each fold
                from sklearn.model_selection import GroupKFold
                gkf = GroupKFold(n_splits=cv_dict['n_folds'])
                cv = gkf.split(X=np.zeros(len(Y)), y=Y, groups=cv_dict['subject_id'])
            elif 'stratified' in cv_dict: # Stratified K-Folds Continuous
                from  nltools.cross_validation import KFoldStratified
                kfs = KFoldStratified(n_splits=cv_dict['n_folds'])
                cv = kfs.split(X=np.zeros(len(Y)), y=Y)
            else: # Normal K-Folds
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=cv_dict['n_folds'])
                cv = kf.split(X=np.zeros(len(Y)), y=Y)
        elif cv_dict['type'] == 'loso': # Leave One Subject Out
            from sklearn.model_selection import LeaveOneGroupOut
            loso = LeaveOneGroupOut()
            cv = loso.split(X=np.zeros(len(Y)), y=Y, groups=cv_dict['subject_id'])
        else:
            raise ValueError("""Make sure you specify a dictionary of
            {'type': 'kfolds', 'n_folds': n},
            {'type': 'kfolds', 'n_folds': n, 'stratified': Y},
            {'type': 'kfolds', 'n_folds': n, 'subject_id': holdout}, or
            {'type': 'loso', 'subject_id': holdout},
            where n = number of folds, and subject = vector of subject ids that corresponds to self.Y""")
    else:
        raise ValueError("Make sure 'cv_dict' is a dictionary.")
    return cv


