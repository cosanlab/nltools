
from __future__ import division

'''
    Cross-Validation Data Classes
    =============================
    Scikit-learn compatible classes for performing various 
    types of cross-validation
'''

__all__ = ['KFoldSubject','KFoldStratified','LeaveOneSubjectOut','set_cv']
__author__ = ["Luke Chang"]
__license__ = "MIT"

from sklearn.cross_validation import _BaseKFold, check_random_state
import numpy as np
import random
import pandas as pd

class KFoldSubject(_BaseKFold):
    """K-Folds cross validation iterator which holds out same subjects.

    Provides train/test indices to split data in train test sets. Split
    dataset into k consecutive folds while ensuring that same subject is held
    out within each fold 
    Each fold is then used a validation set once while the k - 1 remaining
    fold form the training set.
    Extension of KFold from scikit-learn cross_validation model
    
    Args:
        n: int
            Total number of elements.
        labels: vector of length Y indicating subject IDs
        n_folds: int, default=3
            Number of folds. Must be at least 2.
        shuffle: boolean, optional
            Whether to shuffle the data before splitting into batches.
        random_state: None, int or RandomState
            Pseudo-random number generator state used for random
            sampling. If None, use default numpy RNG for shuffling
    
    """

    def __init__(self, n, labels, n_folds=3, shuffle=False, random_state=None):
        super(KFoldSubject, self).__init__(n, n_folds, shuffle, random_state)
        self.idxs = np.arange(n)
        self.labels = np.array(labels, copy=True)
        self.n_subs = len(np.unique(self.labels))
        if shuffle:
            rng = check_random_state(self.random_state)
            rng.shuffle(self.idxs)

    def _iter_test_indices(self):
        subs = np.unique(self.labels)
        n_sub_fold = self.n_subs/self.n_folds
        random.shuffle(subs) # shuffle subjects    
        divide_subs = lambda x,y: [ x[i:i+y] for i in np.arange(0,len(x),y)]
        sub_divs = divide_subs(subs, n_sub_fold)
        for d in sub_divs:
            idx = np.in1d(self.labels,d)
            yield self.idxs[np.where(idx)[0]]

    def __repr__(self):
        return '%s.%s(n=%i, n_subs=%i, n_folds=%i, shuffle=%s, random_state=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
            self.n_subs,
            self.n_folds,
            self.shuffle,
            self.random_state,
        )

    def __len__(self):
        return self.n_folds

class KFoldStratified(_BaseKFold):
    """K-Folds cross validation iterator which stratifies continuous data (unlike scikit-learn equivalent).

    Provides train/test indices to split data in train test sets. Split
    dataset into k consecutive folds while ensuring that same subject is held
    out within each fold 
    Each fold is then used a validation set once while the k - 1 remaining
    folds form the training set.
    Extension of KFold from scikit-learn cross_validation model
    
    Args:
        y : array-like, [n_samples]
            Samples to split in K folds.
        n_folds: int, default=5
            Number of folds. Must be at least 2.
        shuffle: boolean, optional
            Whether to shuffle the data before splitting into batches.
        random_state: None, int or RandomState
            Pseudo-random number generator state used for random
            sampling. If None, use default numpy RNG for shuffling
    
    """

    def __init__(self, y, n_folds=5, shuffle=False, random_state=None):
        super(KFoldStratified, self).__init__(len(y), n_folds, shuffle, random_state)
        self.y = y
        self.idxs = np.arange(len(y))
        self.sort_indx = self.y.argsort()
        if shuffle:
            rng = check_random_state(self.random_state)
            rng.shuffle(self.idxs)

    def _iter_test_indices(self):
        for k in range(0, self.n_folds):
            # yield self.idxs[self.sort_indx[range(k,len(self.y),self.n_folds)]]
            yield self.sort_indx[range(k,len(self.y),self.n_folds)]

    def __repr__(self):
        return '%s.%s(n_folds=%i, shuffle=%s, random_state=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.y,
            self.n_folds,
            self.shuffle,
            self.random_state,
        )

    def __len__(self):
        return self.n_folds

class LeaveOneSubjectOut(_BaseKFold):
    """LOSO cross validation iterator which holds out same subjects.

    Provides train/test indices to split data in train test sets. Split
    dataset into n_subject consecutive folds
    Each fold is then used a validation set once while the n_subject - 1 remaining
    folds form the training set.
    Extension of KFold from scikit-learn cross_validation model

    Args:
        labels: vector of length Y indicating subject IDs
        shuffle: boolean, optional
            Whether to shuffle the data before splitting into batches.
        random_state: None, int or RandomState
            Pseudo-random number generator state used for random
            sampling. If None, use default numpy RNG for shuffling

    """

    def __init__(self, n, labels, shuffle=False, random_state=None):
        super(LeaveOneSubjectOut, self).__init__(n, len(np.unique(labels)), shuffle, random_state)
        self.idxs = np.arange(len(labels))
        self.labels = np.array(labels, copy=True)
        self.n_subs = len(np.unique(self.labels))
        if shuffle:
            rng = check_random_state(self.random_state)
            rng.shuffle(self.idxs)

    def _iter_test_indices(self):
        for d in np.unique(self.labels):
            idx = np.in1d(self.labels,d)
            yield self.idxs[np.where(idx)[0]]

    def __repr__(self):
        return '%s.%s(n=%i, n_subs=%i, shuffle=%s, random_state=%s)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.n,
            self.n_subs,
            self.shuffle,
            self.random_state,
        )

    def __len__(self):
        return self.n_subs

def set_cv(cv_dict):
    """ Helper function to create a sci-kit learn compatible cv object using common parameters for prediction analyses.

    Args:
        cv_dict: Type of cross_validation to use. A dictionary of
            {'type': 'kfolds', 'n_folds': n},
            {'type': 'kfolds', 'n_folds': n, 'stratified': Y},
            {'type': 'kfolds', 'n_folds': n, 'subject_id': holdout}, or
            {'type': 'loso', 'subject_id': holdout}
    Returns:
        cv: a scikit-learn cross-validation instance

     """

    if type(cv_dict) is dict:
        if cv_dict['type'] == 'kfolds':
            if 'subject_id' in cv_dict:
                # Hold out subjects within each fold
                from  nltools.cross_validation import KFoldSubject
                cv = KFoldSubject(len(cv_dict['subject_id']), cv_dict['subject_id'], n_folds=cv_dict['n_folds'])
            elif 'stratified' in cv_dict:
                # Stratified K-Folds
                from  nltools.cross_validation import KFoldStratified
                if isinstance(cv_dict['stratified'],pd.DataFrame):
                    # need to pass numpy array not pandas
                    cv_dict['stratified'] = np.array(cv_dict['stratified']).flatten()
                cv = KFoldStratified(cv_dict['stratified'], n_folds=cv_dict['n_folds'])
            else:
                # Normal K-Folds
                from sklearn.cross_validation import KFold
                cv = KFold(n=cv_dict['n'], n_folds=cv_dict['n_folds'])
        elif cv_dict['type'] == 'loso':
            # Leave One Subject Out
            from nltools.cross_validation import LeaveOneSubjectOut
            cv = LeaveOneSubjectOut(len(cv_dict['subject_id']), labels=cv_dict['subject_id'])
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


