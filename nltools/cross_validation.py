
__all__ = ['KFoldSubject']

from sklearn.cross_validation import _BaseKFold
import numpy as np
import random

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

    def __init__(self, n, labels, n_folds=3, indices=None, shuffle=False, random_state=None):
        super(KFoldSubject, self).__init__(n, n_folds, indices, shuffle, random_state)
        self.idxs = np.arange(n)
        self.labels = np.array(labels, copy=True)
        self.n_subs = len(np.unique(self.labels))
        if shuffle:
            rng = check_random_state(self.random_state)
            rng.shuffle(self.idxs)

    def _iter_test_indices(self):
        n = self.n
        n_folds = self.n_folds
        n_subs_fold = self.n/self.n_subs
        subs = np.unique(self.labels)
        random.shuffle(subs) # shuffle subjects       
        divide_subs = lambda x,y: [ x[i:i+y] for i in range(0,len(x),y)]
        sub_divs = divide_subs(subs, self.n_folds+1) # seems to be adding one fold for some reason
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
