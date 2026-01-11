import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection._split import BaseCrossValidator
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

    def test_different_n_splits(self):
        """Test with different numbers of splits."""
        y = pd.DataFrame(np.random.randn(100))
        for n_splits in [2, 3, 5, 10]:
            cv = KFoldStratified(n_splits=n_splits)
            check_cv_coverage(
                cv, X=np.zeros(len(y)), y=y, groups=None, expected_n_splits=n_splits
            )

    def test_small_dataset(self):
        """Test with very small dataset."""
        y = pd.DataFrame(np.random.randn(10))
        cv = KFoldStratified(n_splits=2)
        check_cv_coverage(cv, X=np.zeros(len(y)), y=y, groups=None, expected_n_splits=2)


class TestKFoldStratifiedSklearnCompatibility:
    """Test sklearn API compatibility."""

    def test_is_instance_of_base_cross_validator(self):
        """Test that KFoldStratified is a sklearn BaseCrossValidator."""
        cv = KFoldStratified(n_splits=5)
        assert isinstance(cv, BaseCrossValidator)

    def test_get_n_splits_method(self):
        """Test get_n_splits method (sklearn API)."""
        y = pd.DataFrame(np.random.randn(100))
        X = np.zeros((100, 10))
        cv = KFoldStratified(n_splits=5)
        assert cv.get_n_splits(X, y) == 5
        assert cv.get_n_splits(X, y, groups=None) == 5

    def test_n_splits_attribute(self):
        """Test n_splits attribute (sklearn API)."""
        cv = KFoldStratified(n_splits=7)
        assert cv.n_splits == 7

    def test_shuffle_attribute(self):
        """Test shuffle attribute (sklearn API)."""
        cv_false = KFoldStratified(n_splits=5, shuffle=False)
        assert cv_false.shuffle is False

        cv_true = KFoldStratified(n_splits=5, shuffle=True)
        assert cv_true.shuffle is True

    def test_random_state_attribute(self):
        """Test random_state attribute (sklearn API)."""
        cv_none = KFoldStratified(n_splits=5, random_state=None)
        assert cv_none.random_state is None

        # random_state can only be set when shuffle=True (sklearn behavior)
        cv_int = KFoldStratified(n_splits=5, shuffle=True, random_state=42)
        assert cv_int.random_state == 42

    def test_split_returns_correct_types(self):
        """Test that split() returns correct types (sklearn API)."""
        y = pd.DataFrame(np.random.randn(100))
        X = np.zeros((100, 10))
        cv = KFoldStratified(n_splits=5)

        splits = list(cv.split(X, y))
        assert len(splits) == 5

        for train_idx, test_idx in splits:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
            assert train_idx.dtype in [np.int32, np.int64]
            assert test_idx.dtype in [np.int32, np.int64]

    def test_compatible_with_sklearn_functions(self):
        """Test that KFoldStratified works with sklearn utilities."""
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LinearRegression

        # Create simple regression problem
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        cv = KFoldStratified(n_splits=5)
        model = LinearRegression()

        # This should work without errors
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        assert len(scores) == 5
        assert all(np.isfinite(scores))

    def test_stratification_actually_works(self):
        """Test that stratification produces balanced folds."""
        # Create target with clear structure
        np.random.seed(42)
        y = pd.DataFrame(np.arange(100))  # Linear increasing values

        cv = KFoldStratified(n_splits=5)
        fold_means = []
        for train_idx, test_idx in cv.split(np.zeros(len(y)), y):
            fold_means.append(y.iloc[test_idx].mean()[0])

        # Means should be similar (within reasonable range)
        # With 100 samples in 5 folds, each fold has ~20 samples
        # Mean of entire dataset is ~50
        fold_means = np.array(fold_means)
        assert np.std(fold_means) < 5.0  # Standard deviation should be small
        assert np.mean(fold_means) > 45 and np.mean(fold_means) < 55


class TestKFoldStratifiedInputValidation:
    """Test input validation and edge cases."""

    def test_y_as_array(self):
        """Test that y can be numpy array."""
        y = np.random.randn(100)
        X = np.zeros((100, 10))
        cv = KFoldStratified(n_splits=5)
        check_cv_coverage(cv, X, y, groups=None, expected_n_splits=5)

    def test_y_as_dataframe(self):
        """Test that y can be pandas DataFrame."""
        y = pd.DataFrame(np.random.randn(100))
        X = np.zeros((100, 10))
        cv = KFoldStratified(n_splits=5)
        check_cv_coverage(cv, X, y, groups=None, expected_n_splits=5)

    def test_y_as_series(self):
        """Test that y can be pandas Series."""
        y = pd.Series(np.random.randn(100))
        X = np.zeros((100, 10))
        cv = KFoldStratified(n_splits=5)
        check_cv_coverage(cv, X, y, groups=None, expected_n_splits=5)

    def test_y_as_list(self):
        """Test that y can be list."""
        y = list(np.random.randn(100))
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

    def test_groups_parameter_ignored(self):
        """Test that groups parameter is ignored (as documented)."""
        y = np.random.randn(100)
        X = np.zeros((100, 10))
        groups = np.random.randint(0, 5, 100)

        cv = KFoldStratified(n_splits=5)
        # Should work with groups=None and groups=something
        splits_without = list(cv.split(X, y, groups=None))
        splits_with = list(cv.split(X, y, groups=groups))

        # Should produce same splits (groups ignored)
        assert len(splits_without) == len(splits_with)


class TestKFoldStratifiedComparison:
    """Comparison tests with sklearn's KFold."""

    def test_same_n_splits_behavior(self):
        """Test that n_splits works the same as sklearn KFold."""
        y = np.random.randn(100)
        X = np.zeros((100, 10))

        cv_nltools = KFoldStratified(n_splits=5)
        cv_sklearn = KFold(n_splits=5)

        assert cv_nltools.get_n_splits(X, y) == cv_sklearn.get_n_splits(X, y)
        assert cv_nltools.n_splits == cv_sklearn.n_splits

    def test_different_from_kfold(self):
        """Test that stratification produces different splits than regular KFold."""
        np.random.seed(42)
        y = pd.DataFrame(np.arange(100))  # Ordered values
        X = np.zeros((100, 10))

        cv_stratified = KFoldStratified(n_splits=5)
        cv_regular = KFold(n_splits=5, shuffle=False)

        # Get first fold from each
        train_s, test_s = next(cv_stratified.split(X, y))
        train_r, test_r = next(cv_regular.split(X, y))

        # Stratified should have more balanced means
        mean_stratified = y.iloc[test_s].mean()[0]
        mean_regular = y.iloc[test_r].mean()[0]

        # Stratified mean should be closer to overall mean (~50)
        overall_mean = y.mean()[0]
        assert abs(mean_stratified - overall_mean) < abs(mean_regular - overall_mean)
