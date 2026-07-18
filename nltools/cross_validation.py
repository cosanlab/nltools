"""Scikit-learn-compatible cross-validation data classes."""

__all__ = ["KFoldStratified"]

from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
import numpy as np


class KFoldStratified(_BaseKFold):
    """Stratify continuous targets across K-fold cross-validation.

    Unlike the scikit-learn equivalent, this iterator stratifies continuous data.

    Provides train/test indices to split data in train test sets. Samples are
    ordered by their continuous target ``y`` and dealt round-robin into k folds
    so each fold spans the full range of ``y``. Each fold is then used as a
    validation set once while the k - 1 remaining folds form the training set.

    Args:
        n_splits: Number of folds. Must be at least 2. Defaults to 3.
        shuffle: Whether to shuffle the data before splitting into batches.
        random_state: Pseudo-random number generator state used for random
            sampling. If None, use the default numpy RNG for shuffling.

    """

    def __init__(self, n_splits=3, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _make_test_folds(self, X, y=None, groups=None):
        y_arr = np.asarray(y).ravel()
        n = len(y_arr)
        if self.shuffle:
            # Sort by y (stratification) but break ties randomly so that
            # shuffle/random_state actually vary the fold assignment. lexsort
            # uses the last key as the primary sort, so y_arr stays primary and
            # the random tiebreak only reorders samples that share a y value.
            rng = check_random_state(self.random_state)
            tiebreak = rng.permutation(n)
            order = np.lexsort((tiebreak, y_arr))
        else:
            order = np.argsort(y_arr, kind="stable")
        test_folds = np.full(n, np.nan)
        for k in range(self.n_splits):
            test_folds[order[np.arange(k, n, self.n_splits)]] = k
        return test_folds

    def _iter_test_masks(self, X=None, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Args:
            X: Training data of shape `(n_samples, n_features)`, where
                `n_samples` is the number of samples and `n_features` is the
                number of features. Note that providing `y` is sufficient to
                generate the splits, hence `np.zeros(n_samples)` may be used as
                a placeholder for `X` instead of actual training data.
            y: The target variable of shape `(n_samples,)` for supervised
                learning problems. Stratification is done based on the y labels.
            groups: Always ignored, exists for compatibility.

        Returns:
            train: The training set indices for that split (ndarray).
            test: The testing set indices for that split (ndarray).

        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return super().split(X, y, groups)
