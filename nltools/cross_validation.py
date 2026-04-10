"""
Cross-Validation Data Classes
=============================

Scikit-learn compatible classes for performing various
types of cross-validation

"""

__all__ = ["KFoldStratified"]

from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import check_array
import numpy as np


class KFoldStratified(_BaseKFold):
    """K-Folds cross validation iterator which stratifies continuous data
    (unlike scikit-learn equivalent).

    Provides train/test indices to split data in train test sets. Split
    dataset into k consecutive folds while ensuring that same subject is
    held out within each fold.  Each fold is then used a validation set
    once while the k - 1 remaining folds form the training set.
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
        super(KFoldStratified, self).__init__(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def _make_test_folds(self, X, y=None, groups=None):
        y_arr = np.asarray(y).ravel()
        order = np.argsort(y_arr, kind="stable")
        test_folds = np.full(len(y_arr), np.nan)
        for k in range(self.n_splits):
            test_folds[order[np.arange(k, len(y_arr), self.n_splits)]] = k
        return test_folds

    def _iter_test_masks(self, X=None, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Args:
            X : array-like, shape (n_samples, n_features)
                Training data, where n_samples is the number of samples
                and n_features is the number of features.
                Note that providing ``y`` is sufficient to generate the splits
                and hence ``np.zeros(n_samples)`` may be used as a placeholder
                for ``X`` instead of actual training data.
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
