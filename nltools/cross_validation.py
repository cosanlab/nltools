"""Scikit-learn-compatible cross-validation data classes."""

__all__ = ["KFoldStratified", "resolve_cv"]

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
            X: Training data of shape `(n_samples, n_features)`, where
                `n_samples` is the number of samples and `n_features` is the
                number of features. Note that providing `y` is sufficient to
                generate the splits, hence `np.zeros(n_samples)` may be used as
                a placeholder for `X` instead of actual training data.
            y: The target variable of shape `(n_samples,)` for supervised
                learning problems. Stratification is done based on the y labels.
            groups: Always ignored, exists for compatibility.

        Yields:
            tuple[np.ndarray, np.ndarray]: `(train, test)` — the training set indices
                and the testing set indices for that split.

        """
        y = check_array(y, ensure_2d=False, dtype=None)
        return super().split(X, y, groups)


def resolve_cv(
    cv,
    *,
    groups=None,
    classifier: bool = False,
    shuffle: bool = False,
    random_state: int | None = None,
):
    """Resolve a cv spec (int, sklearn-style name, or splitter) into an sklearn splitter.

    The single cv-resolution rule shared by `BrainData.predict`,
    `BrainCollection.predict`, and `BrainCollection.predict_group`. String
    names follow sklearn's splitter classes; an int spec must honor
    ``groups`` when one is supplied — plain ``KFold`` silently ignores its
    ``groups`` argument, which previously produced folds byte-identical to
    passing no groups at all.

    Args:
        cv: ``'loo'`` (`LeaveOneOut`), ``'logo'`` (`LeaveOneGroupOut` — pass
            the grouping variable via ``groups``), an int fold count, or an
            sklearn splitter (returned unchanged).
        groups: Group labels, or None. Only consulted for int specs.
        classifier: Whether the downstream model is a classifier — an int
            spec becomes the stratified variant (`StratifiedKFold`, or
            `StratifiedGroupKFold` with groups) for classifiers.
        shuffle: Whether an int spec's KFold variant shuffles samples before
            splitting. Ignored for the group variants (fold membership is
            set by ``groups``).
        random_state: Seed for ``shuffle``.

    Returns:
        BaseCrossValidator: An sklearn splitter instance.

    Raises:
        ValueError: On an unknown string spec, including the pre-v0.6.0
            names ``'loso'`` / ``'loro'`` (use ``'logo'`` with ``groups=``).
    """
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        LeaveOneGroupOut,
        LeaveOneOut,
        StratifiedGroupKFold,
        StratifiedKFold,
    )

    if isinstance(cv, str):
        if cv == "loo":
            return LeaveOneOut()
        if cv == "logo":
            return LeaveOneGroupOut()
        if cv in ("loso", "loro"):
            raise ValueError(
                f"cv={cv!r} was removed in v0.6.0 — both names were "
                f"LeaveOneGroupOut with an implied grouping. Use cv='logo' "
                f"and say the grouping explicitly via groups= "
                f"(predict_group defaults groups to one per subject; "
                f"pass groups='run' for leave-one-run-out)."
            )
        raise ValueError(
            f"unknown cv spec {cv!r}: expected 'loo', 'logo', an int fold "
            f"count, or an sklearn splitter."
        )
    if isinstance(cv, int):
        if groups is not None:
            return (
                StratifiedGroupKFold(n_splits=cv)
                if classifier
                else GroupKFold(n_splits=cv)
            )
        # sklearn refuses random_state without shuffle — drop it when unused.
        rs = random_state if shuffle else None
        return (
            StratifiedKFold(n_splits=cv, shuffle=shuffle, random_state=rs)
            if classifier
            else KFold(n_splits=cv, shuffle=shuffle, random_state=rs)
        )
    return cv
