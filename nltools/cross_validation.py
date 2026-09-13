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
    ordered by their continuous target `y` and dealt round-robin into k folds
    so each fold spans the full range of `y`. Each fold is then used as a
    validation set once while the k - 1 remaining folds form the training set.

    Args:
        n_splits (int): Number of folds. Must be at least 2. Defaults to 3.
        shuffle (bool): Whether to break ties in `y` randomly before dealing samples
            into folds. Default False.
        random_state (int | np.random.RandomState, optional): Seed or RandomState for
            the tie-break shuffle. If None, use the default numpy RNG.
    """

    def __init__(self, n_splits=3, *, shuffle=False, random_state=None):
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
            X (array-like): Training data of shape `(n_samples, n_features)`. Only
                `y` is needed to generate the splits, so `np.zeros(n_samples)` works
                as a placeholder.
            y (array-like): Continuous target of shape `(n_samples,)`; stratification
                is based on its ordering.
            groups (array-like, optional): Always ignored; exists for sklearn
                compatibility.

        Yields:
            tuple[np.ndarray, np.ndarray]: `(train, test)` — the training set indices
                and the testing set indices for that split.
        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return super().split(X, y, groups)
