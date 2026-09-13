import numpy as np
import pandas as pd
from nltools.cross_validation import KFoldStratified


def check_valid_split(train, test, n_samples=None):
    """Helper: Check that train/test split is valid."""
    train, test = set(train), set(test)

    # Train and test split should not overlap
    assert train.intersection(test) == set()

    if n_samples is not None:
        # Check that the union of train and test split cover all the indices
        assert train.union(test) == set(range(n_samples))


def check_cv_coverage(cv, X, y, groups, expected_n_splits=None):
    """Helper: Check that CV splitter covers all samples."""
    n_samples = X.shape[0]
    # Check that all samples appear at least once in a test fold
    if expected_n_splits is not None:
        assert cv.get_n_splits(X, y, groups) == expected_n_splits
    else:
        expected_n_splits = cv.get_n_splits(X, y, groups)

    collected_test_samples = set()
    iterations = 0
    for train, test in cv.split(X, y, groups):
        check_valid_split(train, test, n_samples=n_samples)
        iterations += 1
        collected_test_samples.update(test)

    # Check that the accumulated test samples cover the whole dataset
    assert iterations == expected_n_splits
    if n_samples is not None:
        assert collected_test_samples == set(range(n_samples))


class TestKFoldStratifiedBasic:
    """Basic functionality tests for KFoldStratified."""

    def test_stratified_kfold_ratios(self):
        """Test that stratification keeps similar means across folds."""
        y = pd.DataFrame(np.random.randn(1000)) * 20 + 50
        n_folds = 5
        cv = KFoldStratified(n_splits=n_folds)
        for train, test in cv.split(np.zeros(len(y)), y):
            # Mean should be similar across folds (within reasonable range)
            # Original mean is ~50, std is ~20, so mean should be in [47, 53]
            train_mean = y.iloc[train].mean()[0]
            test_mean = y.iloc[test].mean()[0]
            assert (train_mean >= 47) & (train_mean <= 53)
            assert (test_mean >= 47) & (test_mean <= 53)

    def test_kfoldstratified_coverage_even(self):
        """Test CV coverage with even number of samples."""
        y = pd.DataFrame(np.random.randn(50)) * 20 + 50
        n_folds = 5
        cv = KFoldStratified(n_splits=n_folds)
        check_cv_coverage(
            cv, X=np.zeros(len(y)), y=y, groups=None, expected_n_splits=n_folds
        )

    def test_kfoldstratified_coverage_odd(self):
        """Test CV coverage with odd number of samples."""
        y = pd.DataFrame(np.random.randn(51)) * 20 + 50
        n_folds = 5
        cv = KFoldStratified(n_splits=n_folds)
        check_cv_coverage(
            cv, X=np.zeros(len(y)), y=y, groups=None, expected_n_splits=n_folds
        )

    def test_small_dataset(self):
        """Test with very small dataset."""
        y = pd.DataFrame(np.random.randn(10))
        cv = KFoldStratified(n_splits=2)
        check_cv_coverage(cv, X=np.zeros(len(y)), y=y, groups=None, expected_n_splits=2)


class TestKFoldStratifiedInputValidation:
    """Test input validation and edge cases."""

    def test_y_as_array(self):
        """Test that y can be numpy array."""
        y = np.random.randn(100)
        X = np.zeros((100, 10))
        cv = KFoldStratified(n_splits=5)
        check_cv_coverage(cv, X, y, groups=None, expected_n_splits=5)

    def test_continuous_targets(self):
        """Test that continuous targets work (unlike sklearn StratifiedKFold)."""
        # sklearn's StratifiedKFold would raise ValueError here
        y = np.random.randn(100) * 10 + 50  # Continuous values
        X = np.zeros((100, 10))
        cv = KFoldStratified(n_splits=5)

        # Should work without errors
        splits = list(cv.split(X, y))
        assert len(splits) == 5


class TestKFoldStratifiedShuffle:
    """Regression tests for F150: shuffle/random_state must affect folds."""

    @staticmethod
    def _fold_assignment(cv, y):
        """Return the per-sample test-fold labels produced by cv."""
        n = len(y)
        folds = np.full(n, -1)
        for k, (_, test) in enumerate(cv.split(np.zeros(n), y)):
            folds[test] = k
        return folds

    def test_shuffle_true_reproducible(self):
        """Same seed reproduces the same folds."""
        y = np.repeat(np.arange(10), 5).astype(float)
        cv_a = KFoldStratified(n_splits=5, shuffle=True, random_state=7)
        cv_b = KFoldStratified(n_splits=5, shuffle=True, random_state=7)
        np.testing.assert_array_equal(
            self._fold_assignment(cv_a, y), self._fold_assignment(cv_b, y)
        )

    def test_shuffle_still_stratifies(self):
        """Shuffling ties preserves balanced (stratified) fold means."""
        np.random.seed(0)
        y = np.arange(100).astype(float)
        cv = KFoldStratified(n_splits=5, shuffle=True, random_state=3)
        fold_means = [y[test].mean() for _, test in cv.split(np.zeros(len(y)), y)]
        assert np.std(fold_means) < 5.0
